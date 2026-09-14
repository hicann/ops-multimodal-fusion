#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# indice_conv (Ascend C sparse-convolution per-pair matmul) precision/functional test suite.
#
# Op under test:
#   torch.ops.ops_multimodal_fusion.indice_conv(gathered, filters, k_idx, s_idx, P) -> Tensor
#     partial[s_idx[p], :] = gathered[p, :] @ filters[k_idx[p], :, :]
#
# Golden design (NOT an NPU pure-torch reference): every numeric expectation is computed on the
# host from first principles.
#   * fp32 golden = exact float64 per-pair matmul (CPU).  The kernel accumulates in exact fp32
#     (SIMT fmaf / float4 FMA), so its scale-relative max-abs deviation from the fp64 truth is
#     ~1e-7, well below the 1e-6 gate and far below the ops-precision fp32 MERE threshold 2^-13.
#   * fp16 golden = a faithful fp16 accumulation simulation on CPU (round-to-fp16 after every
#     multiply-add, mirroring the kernel's generic scalar loop order); the fp16 kernel matches it
#     bit-exactly.
#   * full-convolution golden = exact float64 sparse-conv reference built over real 3-D voxel
#     geometries (subm / stride-2 conv / deconv / generic C_out) that share the kernel's semantics
#     (output[o] += features[i] @ filters[k], host-scattered by out index).
#
# Run (from OUTSIDE the source tree so an installed wheel is not shadowed):
#     cd /tmp && pytest /workspace/ops-multimodal-fusion/tests/indice_conv/ -v
# The module also falls back to <repo>/build/lib/libops_multimodal_fusion.so when the wheel is not
# installed (e.g. after a local `cmake --build` only).

import itertools
import math
import os
import random

import pytest
import torch
import torch_npu  # noqa: F401

# ── library loading: installed wheel first, then <repo>/build/lib ──────────
def _ensure_loaded():
    if hasattr(torch.ops.ops_multimodal_fusion, "indice_conv"):
        return True
    try:
        import ops_multimodal_fusion  # noqa: F401
    except Exception:
        pass
    if hasattr(torch.ops.ops_multimodal_fusion, "indice_conv"):
        return True
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    for cand in (
        os.path.join(repo, "build", "lib", "libops_multimodal_fusion.so"),
        os.path.join(repo, "ops_multimodal_fusion", "libops_multimodal_fusion.so"),
    ):
        if os.path.exists(cand):
            try:
                torch.ops.load_library(cand)
            except Exception:
                pass
            break
    return hasattr(torch.ops.ops_multimodal_fusion, "indice_conv")


if not _ensure_loaded():
    pytest.skip(
        "ops_multimodal_fusion.indice_conv not registered for current build; module skipped",
        allow_module_level=True,
    )

requires_npu = pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")

OMF = torch.ops.ops_multimodal_fusion

# ── metrics (ops-precision style) ──────────────────────────────────────────
def _scale_err(a_cpu, gold_f64):
    """max-abs deviation scaled by the overall |gold| magnitude (run.py PASS gate < 1e-6)."""
    return (a_cpu.double() - gold_f64).abs().max().item() / (gold_f64.abs().amax().item() + 1e-12)


def _mere(a_cpu, gold_f64, eps=1e-7):
    """MERE = mean |a-g|/(|g|+eps); fp32 ops-precision threshold 2^-13 ~= 1.22e-4."""
    rel = (a_cpu.double() - gold_f64).abs() / (gold_f64.abs() + eps)
    return rel.mean().item()


def _assert_fp32_close(a_cpu, gold_f64, cid, scale_th=1e-6, mere_th=2.0 ** -13):
    assert a_cpu.shape == gold_f64.shape, f"{cid}: shape {tuple(a_cpu.shape)} != {tuple(gold_f64.shape)}"
    se = _scale_err(a_cpu, gold_f64)
    me = _mere(a_cpu, gold_f64)
    assert se < scale_th, f"{cid}: scale-relative max-abs err {se:.3e} >= {scale_th:.1e}"
    assert me < mere_th, f"{cid}: MERE {me:.3e} >= {mere_th:.1e}"
    return se, me


# ── 3-D sparse-convolution geometry (CPU, self-consistent with kernel semantics) ──
def _ksize_offsets(ksize):
    return list(itertools.product(*[range(int(k)) for k in ksize]))


def _center_offset(ksize):
    return tuple((int(k) - 1) // 2 for k in ksize)


def _out_shape(spatial, ksize, stride, padding, transpose):
    nd = len(spatial)
    if transpose:
        return [int((spatial[i] - 1) * stride[i] - 2 * padding[i] + (ksize[i] - 1) + 1)
                for i in range(nd)]
    return [int((spatial[i] + 2 * padding[i] - (ksize[i] - 1) - 1) // stride[i] + 1)
            for i in range(nd)]


def _sample_coords(spatial, n_active, seed):
    rng = random.Random(seed)
    cells = list(itertools.product(*[range(int(s)) for s in spatial]))
    return rng.sample(cells, min(n_active, len(cells)))


def _gen_sparse_case(cfg):
    """Build a realistic sparse-conv (in_idx, out_idx, k_idx) triple set + M (+ subm center).

    Semantics shared with the kernel and mmcv_sparse_conv:
      conv   : out = (in + pad - koff) / stride  (must divide, in range)
      deconv : out = in*stride - pad + koff       (transpose=True)
      subm   : out = in + (koff - center); output voxel set == active input voxel set
    Subm excludes the center offset (the model folds the center into a dense mm).
    """
    spatial = cfg["spatial"]
    ksize = cfg["ksize"]
    stride = cfg["stride"]
    padding = cfg["padding"]
    subm = cfg.get("subm", False)
    transpose = cfg.get("transpose", False)
    seed = cfg.get("seed", 0)
    nd = len(spatial)
    offsets = _ksize_offsets(ksize)
    K = len(offsets)
    center = _center_offset(ksize)
    center_idx = offsets.index(center)
    active = _sample_coords(spatial, cfg["n_active"], seed)
    coord_set = set(active)
    coord_to_id = {c: i for i, c in enumerate(active)}
    out_shape = list(spatial) if subm else _out_shape(spatial, ksize, stride, padding, transpose)

    if subm:
        triples = []  # (in_id, out_id, k_idx)
        for v, c in enumerate(active):
            for kk, off in enumerate(offsets):
                if kk == center_idx:
                    continue  # center handled as a dense mm on the host
                out_c = tuple(c[d] + (off[d] - center[d]) for d in range(nd))
                if out_c in coord_set:
                    triples.append((v, coord_to_id[out_c], kk))
        M = len(active)
        if not triples:
            return [], [], [], M, center_idx
        in_ids = [t[0] for t in triples]
        out_ids = [t[1] for t in triples]
        k_ids = [t[2] for t in triples]
        return in_ids, out_ids, k_ids, M, center_idx

    out_coords = set()
    for c in active:
        for off in offsets:
            if transpose:
                out_c = tuple(c[d] * stride[d] - padding[d] + off[d] for d in range(nd))
            else:
                num = tuple(c[d] + padding[d] - off[d] for d in range(nd))
                if any(num[d] % stride[d] != 0 for d in range(nd)):
                    continue
                out_c = tuple(num[d] // stride[d] for d in range(nd))
            if all(0 <= out_c[d] < out_shape[d] for d in range(nd)):
                out_coords.add(out_c)
    M = len(out_coords)
    out_id = {c: i for i, c in enumerate(sorted(out_coords))}
    in_ids, out_ids, k_ids = [], [], []
    for v, c in enumerate(active):
        for kk, off in enumerate(offsets):
            if transpose:
                out_c = tuple(c[d] * stride[d] - padding[d] + off[d] for d in range(nd))
            else:
                num = tuple(c[d] + padding[d] - off[d] for d in range(nd))
                if any(num[d] % stride[d] != 0 for d in range(nd)):
                    continue
                out_c = tuple(num[d] // stride[d] for d in range(nd))
            if all(0 <= out_c[d] < out_shape[d] for d in range(nd)):
                in_ids.append(v)
                out_ids.append(out_id[out_c])
                k_ids.append(kk)
    return in_ids, out_ids, k_ids, M, center_idx


def _exact_full_cpu(features_cpu, filters_k_cpu, in_ids, out_ids, k_ids, M, subm_center):
    """Exact fp64 full-conv reference: out[o] = sum_p features[in_p] @ filters[k_p]."""
    C_in = features_cpu.shape[1]
    C_out = filters_k_cpu.shape[2]
    K = filters_k_cpu.shape[0]
    f64 = features_cpu.double()
    fk = filters_k_cpu.double()
    out = torch.zeros(M, C_out, dtype=torch.float64)
    if subm_center is not None and M:
        out += f64 @ fk[subm_center]
    if not in_ids:
        return out
    in_idx = torch.tensor(in_ids, dtype=torch.long)
    out_idx = torch.tensor(out_ids, dtype=torch.long)
    k_idx = torch.tensor(k_ids, dtype=torch.long)
    for k in range(K):
        m = k_idx == k
        if not bool(m.any()):
            continue
        contrib = f64.index_select(0, in_idx[m]) @ fk[k]
        out.index_add_(0, out_idx[m], contrib)
    return out


# ── fast path under test (mirrors mmcv_sparse_conv.indice_conv_fast, NPU op) ──
def _fast_full_cpu(features_cpu, filters_k_cpu, in_ids, out_ids, k_ids, M, subm_center):
    """Host-gather + Ascend C per-pair matmul + CPU fp32 scatter (center mm in fp32 CPU)."""
    dtype = features_cpu.dtype
    C_out = filters_k_cpu.shape[2]
    out = torch.zeros(M, C_out, dtype=dtype)
    if subm_center is not None and M:
        out = features_cpu @ filters_k_cpu[subm_center]
    if not in_ids:
        return out
    in_idx = torch.tensor(in_ids, dtype=torch.long)
    out_idx = torch.tensor(out_ids, dtype=torch.long)
    gathered_cpu = features_cpu[in_idx]                      # (P, C_in)
    k_idx = torch.tensor(k_ids, dtype=torch.int32)
    s_idx = torch.arange(gathered_cpu.shape[0], dtype=torch.int32)
    P = gathered_cpu.shape[0]
    partial = OMF.indice_conv(gathered_cpu.npu(), filters_k_cpu.npu(),
                              k_idx.npu(), s_idx.npu(), P).cpu()
    out.index_add_(0, out_idx, partial)
    return out


def _partial_golden_fp64(gathered_cpu, filters_k_cpu, k_ids, s_ids):
    """fp64 golden of the per-pair matmul op: partial[s_idx[p]] = gathered[p] @ filters[k_idx[p]]."""
    P = gathered_cpu.shape[0]
    C_out = filters_k_cpu.shape[2]
    K = filters_k_cpu.shape[0]
    gd = gathered_cpu.double()
    fd = filters_k_cpu.double()
    out = torch.zeros(P, C_out, dtype=torch.float64)
    if P == 0:
        return out
    kk = torch.tensor(k_ids, dtype=torch.long)
    si = torch.tensor(s_ids, dtype=torch.long)
    for k in range(K):
        m = kk == k
        if not bool(m.any()):
            continue
        contrib = gd[m] @ fd[k]
        out.index_add_(0, si[m], contrib)
    return out


def _case_tensors(features_cpu, filters_k_cpu, in_ids, k_ids, perm_sidx=False, seed=0):
    in_idx = torch.tensor(in_ids, dtype=torch.long)
    gathered_cpu = features_cpu[in_idx]
    k_idx = torch.tensor(k_ids, dtype=torch.int32)
    if perm_sidx:
        g = torch.Generator().manual_seed(seed)
        s_idx = torch.randperm(gathered_cpu.shape[0], generator=g).to(torch.int32)
    else:
        s_idx = torch.arange(gathered_cpu.shape[0], dtype=torch.int32)
    P = gathered_cpu.shape[0]
    partial = OMF.indice_conv(gathered_cpu.npu(), filters_k_cpu.npu(), k_idx.npu(),
                              s_idx.npu(), P).cpu()
    return partial, gathered_cpu, filters_k_cpu, k_idx, s_idx, P


# ── deterministic seeds / case table ───────────────────────────────────────
def _seed(tag):
    return sum(ord(ch) for ch in tag) + len(tag) * 131


# Representative layer shapes from the real Part-A2 SparseUNet (mmcv_sparse_conv routing):
# subm c16/c32/c64 (K=27 center excluded), stride-2 conv, deconv(inverse), generic/non-16 C_out.
FORM_CFG = [
    dict(tag="subm-c16",    spatial=(10, 10, 10), ksize=(3, 3, 3), stride=(1, 1, 1),
         padding=(1, 1, 1), subm=True,  C_in=16, C_out=16, n_active=380),
    dict(tag="subm-c32",    spatial=(10, 10, 10), ksize=(3, 3, 3), stride=(1, 1, 1),
         padding=(1, 1, 1), subm=True,  C_in=32, C_out=32, n_active=380),
    dict(tag="subm-c64",    spatial=(9, 9, 9),  ksize=(3, 3, 3), stride=(1, 1, 1),
         padding=(1, 1, 1), subm=True,  C_in=64, C_out=64, n_active=300),
    dict(tag="conv-stride2", spatial=(16, 16, 8), ksize=(3, 3, 3), stride=(2, 2, 2),
         padding=(1, 1, 1), subm=False, C_in=16, C_out=16, n_active=520),
    dict(tag="deconv",       spatial=(8, 8, 8), ksize=(3, 3, 3), stride=(2, 2, 2),
         padding=(1, 1, 1), subm=False, transpose=True, C_in=16, C_out=16, n_active=260),
    dict(tag="generic-cout", spatial=(10, 10, 8), ksize=(3, 1, 1), stride=(1, 1, 1),
         padding=(1, 0, 0), subm=False, C_in=32, C_out=9, n_active=420),
    dict(tag="wide-c128",    spatial=(8, 8, 8), ksize=(3, 3, 3), stride=(1, 1, 1),
         padding=(1, 1, 1), subm=False, C_in=64, C_out=128, n_active=220),
]


def _form_case(cfg, dtype=torch.float32, perm_sidx=False):
    in_ids, out_ids, k_ids, M, center = _gen_sparse_case(cfg)
    N = cfg["n_active"]
    if cfg.get("subm"):
        N = cfg["n_active"]
    C_in, C_out = cfg["C_in"], cfg["C_out"]
    K = math.prod(cfg["ksize"])
    g = torch.Generator().manual_seed(_seed(cfg["tag"]))
    features_cpu = torch.randn(N, C_in, generator=g).to(dtype)
    filters_k = (torch.randn(K, C_in, C_out, generator=g) * 0.5).to(dtype)
    return dict(tag=cfg["tag"], features_cpu=features_cpu, filters_k=filters_k,
                in_ids=in_ids, out_ids=out_ids, k_ids=k_ids, M=M,
                subm_center=center if cfg.get("subm") else None, perm_sidx=perm_sidx)


# ═══════════════════════════════════════════════════════════════════════════
# L0: interface + minimal direct op correctness
# ═══════════════════════════════════════════════════════════════════════════
def test_indice_conv_l0_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "indice_conv"), \
        "indice_conv not registered in torch.ops.ops_multimodal_fusion"


@requires_npu
def test_indice_conv_l0_small_fp32():
    cfg = dict(spatial=(8, 8, 8), ksize=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
               subm=True, C_in=16, C_out=16, n_active=120, tag="l0")
    c = _form_case(cfg)
    partial, gathered_cpu, filters_k, k_idx, s_idx, P = _case_tensors(
        c["features_cpu"], c["filters_k"], c["in_ids"], c["k_ids"])
    assert partial.shape == (P, c["filters_k"].shape[2]), f"shape {tuple(partial.shape)}"
    assert partial.dtype == torch.float32, f"dtype {partial.dtype}"
    gold = _partial_golden_fp64(gathered_cpu, filters_k, c["k_ids"], s_idx.tolist())
    _assert_fp32_close(partial, gold, "L0")


# ═══════════════════════════════════════════════════════════════════════════
# L1: op-level correctness across real sparse-conv forms
# ═══════════════════════════════════════════════════════════════════════════
@requires_npu
@pytest.mark.parametrize("cfg", FORM_CFG, ids=[c["tag"] for c in FORM_CFG])
def test_indice_conv_l1_form_partial(cfg):
    c = _form_case(cfg)
    partial, gathered_cpu, filters_k, k_idx, s_idx, P = _case_tensors(
        c["features_cpu"], c["filters_k"], c["in_ids"], c["k_ids"])
    gold = _partial_golden_fp64(gathered_cpu, filters_k, c["k_ids"], s_idx.tolist())
    se, me = _assert_fp32_close(partial, gold, c["tag"])
    print(f"{c['tag']}: P={P} M={c['M']} C_in={c['features_cpu'].shape[1]} "
          f"C_out={c['filters_k'].shape[2]} scale_err={se:.3e} MERE={me:.3e}")


@requires_npu
@pytest.mark.parametrize("cfg", FORM_CFG, ids=[c["tag"] for c in FORM_CFG])
def test_indice_conv_l1_full_flow(cfg):
    c = _form_case(cfg)
    # subm full flow includes the fp32 center mm (on the host path it is a dense mm); the exact
    # reference computes the center in fp64, so gate the end-to-end comparison at 1e-5 scale.
    scale_th = 1e-5 if c["subm_center"] is not None else 1e-6
    fast = _fast_full_cpu(c["features_cpu"], c["filters_k"], c["in_ids"], c["out_ids"],
                          c["k_ids"], c["M"], c["subm_center"])
    ref = _exact_full_cpu(c["features_cpu"], c["filters_k"], c["in_ids"], c["out_ids"],
                          c["k_ids"], c["M"], c["subm_center"])
    assert fast.shape == ref.shape, f"{c['tag']}: shape {tuple(fast.shape)} != {tuple(ref.shape)}"
    se, me = _assert_fp32_close(fast, ref, c["tag"], scale_th=scale_th)
    print(f"{c['tag']} full-flow: M={c['M']} scale_err={se:.3e} MERE={me:.3e}")


@requires_npu
@pytest.mark.parametrize("cfg", [FORM_CFG[0], FORM_CFG[5]], ids=["subm-c16-perm", "generic-perm"])
def test_indice_conv_l1_scatter_sidx(cfg):
    """s_idx permutation (bijective scatter) exercises the indexed store path."""
    c = _form_case(cfg, perm_sidx=True)
    partial, gathered_cpu, filters_k, k_idx, s_idx, P = _case_tensors(
        c["features_cpu"], c["filters_k"], c["in_ids"], c["k_ids"], perm_sidx=True)
    gold = _partial_golden_fp64(gathered_cpu, filters_k, c["k_ids"], s_idx.tolist())
    _assert_fp32_close(partial, gold, c["tag"] + "-perm")


def _pos_form_case(cfg):
    """Well-conditioned U(0.5, 1.5) features/filters so the reference stays away from zero."""
    in_ids, out_ids, k_ids, M, center = _gen_sparse_case(cfg)
    C_in, C_out = cfg["C_in"], cfg["C_out"]
    K = math.prod(cfg["ksize"])
    g = torch.Generator().manual_seed(_seed(cfg["tag"]) ^ 0x9E3779B9)
    features_cpu = torch.rand(cfg["n_active"], C_in, generator=g) + 0.5
    filters_k = torch.rand(K, C_in, C_out, generator=g) + 0.5
    return dict(tag=cfg["tag"], features_cpu=features_cpu, filters_k=filters_k,
                in_ids=in_ids, out_ids=out_ids, k_ids=k_ids, M=M,
                subm_center=center if cfg.get("subm") else None)


@requires_npu
@pytest.mark.parametrize("cfg", FORM_CFG, ids=[c["tag"] for c in FORM_CFG])
def test_indice_conv_l1_wellconditioned_standard(cfg):
    """Well-conditioned data: both MERE (<2^-13) and MARE (<10*2^-13) meet the fp32 ops-precision
    standard -- i.e. the kernel matches the exact fp64 per-pair matmul to fp32 exactness."""
    c = _pos_form_case(cfg)
    partial, gathered_cpu, filters_k, k_idx, s_idx, P = _case_tensors(
        c["features_cpu"], c["filters_k"], c["in_ids"], c["k_ids"])
    gold = _partial_golden_fp64(gathered_cpu, filters_k, c["k_ids"], s_idx.tolist())
    rel = (partial.double() - gold).abs() / (gold.abs() + 1e-7)
    mere = rel.mean().item()
    mare = rel.max().item()
    se = _scale_err(partial, gold)
    assert mere < 2.0 ** -13, f"{c['tag']}: MERE {mere:.3e} >= 2^-13"
    assert mare < 10 * 2.0 ** -13, f"{c['tag']}: MARE {mare:.3e} >= 10*2^-13"
    assert se < 1e-6, f"{c['tag']}: scale err {se:.3e} >= 1e-6"
    print(f"{c['tag']} well-cond: scale={se:.2e} MERE={mere:.2e} MARE={mare:.2e}")


# ═══════════════════════════════════════════════════════════════════════════
# Boundary / edge cases
# ═══════════════════════════════════════════════════════════════════════════
@requires_npu
def test_indice_conv_empty_P():
    """P=0: gathered/k_idx/s_idx all empty -> (0, C_out) tensor."""
    filters_k = torch.randn(8, 4, 16, dtype=torch.float32).npu()
    gathered = torch.zeros(0, 4, dtype=torch.float32).npu()
    k_idx = torch.zeros(0, dtype=torch.int32).npu()
    s_idx = torch.zeros(0, dtype=torch.int32).npu()
    out = OMF.indice_conv(gathered, filters_k, k_idx, s_idx, 0)
    assert tuple(out.shape) == (0, 16), f"empty-P shape {tuple(out.shape)}"
    assert out.dtype == torch.float32 and out.device.type == "npu"
    assert out.cpu().numel() == 0


@requires_npu
def test_indice_conv_full_flow_M0_empty():
    """M=0 (no output voxels, no pairs) -> the whole fast flow returns an empty output."""
    cfg = dict(spatial=(6, 6, 6), ksize=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
               subm=True, C_in=16, C_out=16, n_active=0, tag="m0")
    c = _form_case(cfg)
    assert c["M"] == 0 and len(c["in_ids"]) == 0
    fast = _fast_full_cpu(c["features_cpu"], c["filters_k"], c["in_ids"], c["out_ids"],
                          c["k_ids"], c["M"], c["subm_center"])
    ref = _exact_full_cpu(c["features_cpu"], c["filters_k"], c["in_ids"], c["out_ids"],
                          c["k_ids"], c["M"], c["subm_center"])
    assert tuple(fast.shape) == (0, 16) and tuple(ref.shape) == (0, 16)


@requires_npu
def test_indice_conv_fp16_accumulate():
    """fp16 path accumulates in fp16 (unchanged kernel design): match a faithful fp16 CPU sim.

    The only permitted deviation is one fp16 ULP per element (compiler may contract the
    float multiply-add into a single-rounding fma).
    """
    g = torch.Generator().manual_seed(7)
    P, C_in, C_out, K = 96, 16, 16, 8
    gc = (torch.rand(P, C_in, generator=g, dtype=torch.float64) * 2 - 1)
    fc = (torch.rand(K, C_in, C_out, generator=g, dtype=torch.float64) * 2 - 1)
    kk = torch.randint(0, K, (P,), generator=g)
    si = torch.arange(P)
    out = OMF.indice_conv(gc.half().npu(), fc.half().npu(), kk.int().npu(), si.int().npu(), P).cpu()
    assert out.dtype == torch.float16
    sim = _fp16_sim_cpu(gc.half().float(), fc.half().float(), kk, si, C_out)
    diff = (out.float() - sim.float()).abs()
    ulp = torch.clamp(torch.pow(2.0, torch.floor(torch.log2(sim.float().abs() + 1e-30)) - 10),
                      min=2.0 ** -24)
    assert bool((diff <= ulp).all()), \
        f"fp16 kernel deviates from fp16-accumulate sim by >1 fp16 ULP; maxdiff=" \
        f"{diff.max().item()}"


def _fp16_sim_cpu(g16, f16, k_idx, s_idx, C_out):
    """fp16 per-pair matmul simulated on CPU: acc rounded to fp16 after every multiply-add."""
    P = g16.shape[0]
    C_in = g16.shape[1]
    out = torch.zeros(P, C_out, dtype=torch.float16)
    for p in range(P):
        k = int(k_idx[p])
        row = int(s_idx[p])
        for c in range(C_out):
            acc = torch.tensor(0.0, dtype=torch.float32)
            for i in range(C_in):
                acc = (acc + g16[p, i] * f16[k, i, c]).half().float()
            out[row, c] = acc.half()
    return out


# ── Rejections (host-side TORCH_CHECK, device must stay healthy afterwards) ──
def _health_check():
    g = torch.Generator().manual_seed(123)
    gd = (torch.rand(8, 4, generator=g) * 2 - 1).npu()
    fl = (torch.rand(4, 4, 16, generator=g) * 2 - 1).npu()
    ki = torch.randint(0, 4, (8,), generator=g).int().npu()
    si = torch.arange(8).int().npu()
    out = OMF.indice_conv(gd, fl, ki, si, 8).cpu()
    gold = torch.zeros(8, 16, dtype=torch.float64)
    for k in range(4):
        m = ki.cpu().long() == k
        if bool(m.any()):
            gold[m] = gd.cpu().double()[m] @ fl.cpu().double()[k]
    assert _scale_err(out, gold) < 1e-6, "device health check failed after a rejected call"


def _mk_valid(P=8, C_in=4, C_out=16, K=4, dev="npu"):
    gd = torch.randn(P, C_in).to(dev)
    fl = torch.randn(K, C_in, C_out).to(dev)
    ki = torch.randint(0, K, (P,)).to(device=dev, dtype=torch.int32)
    si = torch.arange(P, device=dev, dtype=torch.int32)
    return gd, fl, ki, si


@requires_npu
@pytest.mark.parametrize("cid,builder", [
    ("rej-noncontig-gathered", lambda: torch.randn(8, 8).npu()[:, ::2]),
    ("rej-noncontig-filters", lambda: torch.randn(4, 8, 16).npu()[:, ::2, :]),
    ("rej-noncontig-kidx", lambda: torch.randint(0, 4, (16,), dtype=torch.int32).npu()[::2]),
    ("rej-noncontig-sidx", lambda: torch.randint(0, 8, (16,), dtype=torch.int32).npu()[::2]),
], ids=["rej-noncontig-gathered", "rej-noncontig-filters",
        "rej-noncontig-kidx", "rej-noncontig-sidx"])
def test_indice_conv_reject_noncontig(cid, builder):
    gd, fl, ki, si = _mk_valid()
    bad = builder()
    with pytest.raises(RuntimeError):
        if cid == "rej-noncontig-gathered":
            OMF.indice_conv(bad, fl, ki, si, 8)
        elif cid == "rej-noncontig-filters":
            OMF.indice_conv(gd, bad, ki, si, 8)
        elif cid == "rej-noncontig-kidx":
            OMF.indice_conv(gd, fl, bad, si, 8)
        else:
            OMF.indice_conv(gd, fl, ki, bad, 8)
    _health_check()


@requires_npu
@pytest.mark.parametrize("cid,fn", [
    ("rej-dtype-gathered-int", lambda g, f, k, s: OMF.indice_conv(
        g.to(torch.int32), f, k, s, g.shape[0])),
    ("rej-dtype-gathered-bool", lambda g, f, k, s: OMF.indice_conv(
        g.gt(0), f, k, s, g.shape[0])),
    ("rej-dtype-filters-mismatch", lambda g, f, k, s: OMF.indice_conv(
        g, f.to(torch.float16), k, s, g.shape[0])),
    ("rej-dtype-kidx", lambda g, f, k, s: OMF.indice_conv(
        g, f, k.to(torch.float32), s, g.shape[0])),
    ("rej-dtype-sidx", lambda g, f, k, s: OMF.indice_conv(
        g, f, k, s.to(torch.float32), g.shape[0])),
    ("rej-shape-gathered-1d", lambda g, f, k, s: OMF.indice_conv(
        g[0], f, k, s, g.shape[0])),
    ("rej-shape-cin-mismatch", lambda g, f, k, s: OMF.indice_conv(
        torch.randn(g.shape[0], 7).npu(), f, k, s, g.shape[0])),
    ("rej-p-mismatch", lambda g, f, k, s: OMF.indice_conv(
        g, f, k, s, g.shape[0] + 1)),
    ("rej-p-negative", lambda g, f, k, s: OMF.indice_conv(
        g, f, k, s, -1)),
], ids=["rej-dtype-gathered-int", "rej-dtype-gathered-bool", "rej-dtype-filters-mismatch",
        "rej-dtype-kidx", "rej-dtype-sidx", "rej-shape-gathered-1d",
        "rej-shape-cin-mismatch", "rej-p-mismatch", "rej-p-negative"])
def test_indice_conv_reject_checks(cid, fn):
    gd, fl, ki, si = _mk_valid()
    with pytest.raises(RuntimeError):
        fn(gd, fl, ki, si)
    _health_check()


# ── Range rejections (R1: k_idx in [0, K), s_idx in [0, P)) ────────────────
@requires_npu
@pytest.mark.parametrize("cid,mutate", [
    ("rej-kidx-oob",      lambda k, f: k.__setitem__(0, f.shape[0])),
    ("rej-kidx-negative", lambda k, f: k.__setitem__(1, -1)),
    ("rej-sidx-oob",      lambda _, s: s.__setitem__(0, s.shape[0])),
    ("rej-sidx-negative", lambda _, s: s.__setitem__(2, -1)),
], ids=["rej-kidx-oob", "rej-kidx-negative", "rej-sidx-oob", "rej-sidx-negative"])
def test_indice_conv_reject_range(cid, mutate):
    """OOB k_idx / s_idx must be rejected host-side (not silently GM OOB read/write)."""
    gd, fl, ki, si = _mk_valid()
    with pytest.raises(RuntimeError):
        if "kidx" in cid:
            mutate(ki, fl)
        else:
            mutate(ki, si)
        OMF.indice_conv(gd, fl, ki, si, 8)
    _health_check()


# ── Meta dispatch ──────────────────────────────────────────────────────────
def test_indice_conv_meta_shape_inference():
    for (P, C_in, C_out) in [(64, 16, 32), (0, 4, 16), (8, 32, 128)]:
        g = torch.empty(P, C_in, device="meta", dtype=torch.float32)
        f = torch.empty(8, C_in, C_out, device="meta", dtype=torch.float32)
        k = torch.empty(P, device="meta", dtype=torch.int32)
        s = torch.empty(P, device="meta", dtype=torch.int32)
        out = OMF.indice_conv(g, f, k, s, P)
        assert tuple(out.shape) == (P, C_out) and out.dtype == torch.float32
        assert out.device.type == "meta"
    # bad shapes must also be caught on the Meta path
    with pytest.raises(RuntimeError):
        OMF.indice_conv(torch.empty(64, device="meta"),
                        torch.empty(8, 16, 32, device="meta"),
                        torch.empty(64, device="meta", dtype=torch.int32),
                        torch.empty(64, device="meta", dtype=torch.int32), 64)
