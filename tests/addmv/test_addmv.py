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
# addmv operator precision / functional test suite.
#
# Implements the L0/L1/L2 case matrix of `.cannbot/2.1-测试方案设计.md` (v14, 233 cases:
# 229 executed + 4 environment skips) against the truth source `.cannbot/1.1-需求分析.md` (v8.6).
# 另含 3.3 白盒补全 7 条（`test_addmv_wb_*`），共 240 条。
#
# Self-contained single file (repo convention: no conftest / pytest.ini / CSV / external data table).
# MUST be run from OUTSIDE the source tree so the installed wheel is not shadowed:
#     cd /tmp && pytest /workspace/asc-ops/ops-multimodal-fusion-tpc/tests/addmv/ -v
#
# Golden design (test plan §2.2 / §2.7):
#   * float golden = the §2.7.2 per-scalar split formula evaluated in float64, NOT rounded back to the
#     output dtype.  It implements the operator contract directly (BLAS gemv short circuit on the raw
#     Scalar per P9, betaQ/alphaQ quantized to fp32 per P10) and therefore serves as the single
#     reference for every category -- ⓪ short circuit / quantization, ① numerical, ② special value,
#     ③ overflow.  It never consumes `torch.addmv` output (6.2 条款6).
#   * int32 golden = int64 arithmetic with toward-zero truncated beta/alpha, two's-complement wrapped
#     to int32 (P13), compared bit-exactly.
#   * precision judged by MERE/MARE on the finite elements (thresholds fp32 2^-13 / fp16 2^-10 /
#     bf16 2^-7, MARE < 10x); NaN/+Inf/-Inf judged by per-element mask equality (equal_nan, stricter).

import logging
import math
import zlib

import pytest
import torch
import torch_npu  # noqa: F401

import ops_multimodal_fusion  # noqa: F401

# ── module-level NPU_ARCH guard (test plan §6.1) ───────────────────────────
if not hasattr(torch.ops.ops_multimodal_fusion, "addmv"):
    pytest.skip(
        "ops_multimodal_fusion.addmv not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )

requires_npu = pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")

NAN = float("nan")
INF = float("inf")
NINF = float("-inf")

# MERE/MARE thresholds by dtype (需求 2.7 / 测试方案 §2.5). Never widened.
_THRESH = {
    torch.float32: 2.0 ** -13,   # 1.22e-4
    torch.float16: 2.0 ** -10,   # 9.77e-4
    torch.bfloat16: 2.0 ** -7,   # 7.81e-3
}
_INT32_MIN = -2147483648
_INT32_MAX = 2147483647


# ═══════════════════════════════════════════════════════════════════════════
# Interface registration (L0-01)
# ═══════════════════════════════════════════════════════════════════════════
def test_addmv_l0_interface_exist():
    """L0-01: the addmv operator is registered in torch.ops.ops_multimodal_fusion."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "addmv"), \
        "The 'addmv' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ═══════════════════════════════════════════════════════════════════════════
# Input builders
# ═══════════════════════════════════════════════════════════════════════════
def _seed(cid):
    """Deterministic per-case seed frozen at design time (no runtime resampling, 测试方案 2.8)."""
    return zlib.crc32(cid.encode()) & 0x7FFFFFFF


def _gen_base(M, K, dtype, family, seed, matval, vecval, selfval):
    """Generate contiguous self (M,) / mat (M,K) / vec (K,) in the target dtype."""
    g = torch.Generator().manual_seed(seed)
    fdt = torch.float64  # generate high precision, cast once at the end

    if family == "D1":                       # 良态正值：mat,vec ~ U(0.5,1.5)/sqrt(K), self ~ U(0.5,1.5)
        s = 1.0 / math.sqrt(K) if K > 0 else 1.0
        mat = (torch.rand(M, K, generator=g, dtype=fdt) + 0.5) * s
        vec = (torch.rand(K, generator=g, dtype=fdt) + 0.5) * s
        self_ = torch.rand(M, generator=g, dtype=fdt) + 0.5
    elif family == "D1i":                     # int32 良态
        mat = torch.randint(-3, 4, (M, K), generator=g).to(fdt)
        vec = torch.randint(-3, 4, (K,), generator=g).to(fdt)
        self_ = torch.randint(-100, 101, (M,), generator=g).to(fdt)
    elif family == "D2":                      # 一般正负分布 U(-1,1)
        mat = torch.rand(M, K, generator=g, dtype=fdt) * 2 - 1
        vec = torch.rand(K, generator=g, dtype=fdt) * 2 - 1
        self_ = torch.rand(M, generator=g, dtype=fdt) * 2 - 1
    elif family == "D4":                      # 离群点：D2 基础上 k 个位置 x1000
        mat = torch.rand(M, K, generator=g, dtype=fdt) * 2 - 1
        vec = torch.rand(K, generator=g, dtype=fdt) * 2 - 1
        self_ = torch.rand(M, generator=g, dtype=fdt) + 0.5
        n = M * K
        k = max(1, n // 1000)
        flat = mat.view(-1)
        idx = torch.randperm(n, generator=g)[:k]
        flat[idx] *= 1000.0
    elif family == "D5":                      # 小值域 U(-1e-3,1e-3)
        mat = (torch.rand(M, K, generator=g, dtype=fdt) * 2 - 1) * 1e-3
        vec = (torch.rand(K, generator=g, dtype=fdt) * 2 - 1) * 1e-3
        self_ = (torch.rand(M, generator=g, dtype=fdt) * 2 - 1) * 1e-3
    elif family == "D6":                      # 大值域 N(100,25)
        mat = torch.randn(M, K, generator=g, dtype=fdt) * 25 + 100
        vec = torch.randn(K, generator=g, dtype=fdt) * 25 + 100
        self_ = torch.randn(M, generator=g, dtype=fdt) * 25 + 100
    elif family == "ones":                    # 特殊值母体：mat=1, vec=1, self~U(0.5,1.5)
        mat = torch.ones(M, K, dtype=fdt)
        vec = torch.ones(K, dtype=fdt)
        self_ = torch.rand(M, generator=g, dtype=fdt) + 0.5
    elif family == "zeros":                   # 全零
        mat = torch.zeros(M, K, dtype=fdt)
        vec = torch.zeros(K, dtype=fdt)
        self_ = torch.zeros(M, dtype=fdt)
    elif family == "const":                   # 定值：mat=matval, vec=vecval, self=selfval
        mat = torch.full((M, K), float(matval), dtype=fdt)
        vec = torch.full((K,), float(vecval), dtype=fdt)
        self_ = torch.full((M,), float(selfval), dtype=fdt)
    else:
        raise ValueError(f"unknown family {family}")

    return self_.to(dtype), mat.to(dtype), vec.to(dtype)


def _apply_poison(self_, mat, vec, poison):
    tbl = {"self": self_, "mat": mat, "vec": vec}
    vmap = {"nan": NAN, "inf": INF, "ninf": NINF}
    for tgt, idx, val in poison:
        t = tbl[tgt]
        v = vmap.get(val, val)
        if idx == "all":
            t.fill_(v)
        else:
            t[idx] = v


def _build(spec):
    """Return (self_cpu, mat_cpu, vec_cpu) shaped for the natural self form."""
    M, K, dtype = spec["M"], spec["K"], spec["dtype"]
    self_, mat, vec = _gen_base(M, K, dtype, spec["family"], spec["seed"],
                                spec.get("matval"), spec.get("vecval"), spec.get("selfval"))
    _apply_poison(self_, mat, vec, spec.get("poison", []))

    form = spec.get("form", "normal")
    if form in ("scalar1", "expand"):
        self_ = self_[:1].clone()             # (1,)
    elif form == "zerodim":
        self_ = self_[0].clone()              # 0-dim
    return self_, mat, vec


def _self_to_npu(self_cpu, form, M, cid):
    """Move self to NPU; broadcast/expand views MUST be built NPU-side (测试方案 6.2 条款1 / 6.2.1)."""
    if form == "normal":
        return self_cpu.npu()
    if form == "scalar1":
        t = self_cpu.npu()                    # (1,), numel()==1 -> broadcast predicate
        assert t.shape == (1,), f"{cid}: scalar1 self must be shape (1,), got {tuple(t.shape)}"
        return t
    if form == "zerodim":
        t = self_cpu.npu()                    # 0-dim, numel()==1 -> broadcast predicate
        assert t.dim() == 0, f"{cid}: zerodim self must be 0-D, got {t.dim()}-D"
        return t
    if form == "expand":
        # [构造顺序强制] 必须先 .npu() 再 .expand()：
        #   cpu_t.npu().expand(M)  -> stride=(0,)  正确，kernel 走 stride-0 直读路径（需求 2.4 规则 1）
        #   cpu_t.expand(M).npu()  -> stride=(1,)  错误且不报错！视图被 H2D 物化成连续张量，覆盖点静默失效
        t = self_cpu.npu().expand(M)
        assert t.stride() == (0,), f"{cid}: expand view lost stride-0 ({t.stride()}); coverage point dead"
        return t
    raise ValueError(form)


# ═══════════════════════════════════════════════════════════════════════════
# Golden (test plan §2.2 / §2.4 / §2.7.2)
# ═══════════════════════════════════════════════════════════════════════════
def _self_double(self_cpu, M):
    sd = self_cpu.double().reshape(-1)
    if sd.numel() == 1:                       # broadcast form: single value applies to all M rows
        return sd.expand(M)
    return sd


def _golden_float(self_cpu, mat_cpu, vec_cpu, beta, alpha, M, K):
    """§2.7.2 split formula in float64: out = T_self + T_mat (never rounded to output dtype)."""
    betaQ = float(torch.tensor(beta, dtype=torch.float64).to(torch.float32))    # P10: fp32 承载
    alphaQ = float(torch.tensor(alpha, dtype=torch.float64).to(torch.float32))
    beta_is_zero = (float(beta) == 0.0)       # P9: judge zero on the raw Scalar
    alpha_is_zero = (float(alpha) == 0.0)

    if beta_is_zero:                          # self not referenced, its NaN/Inf does not propagate
        t_self = torch.zeros(M, dtype=torch.float64)
    else:
        t_self = betaQ * _self_double(self_cpu, M)
    if alpha_is_zero or K == 0:               # mat/vec not referenced (or empty K -> product 0)
        t_mat = torch.zeros(M, dtype=torch.float64)
    else:
        t_mat = alphaQ * (mat_cpu.double() @ vec_cpu.double())
    return t_self + t_mat


def _golden_int_raw(self_cpu, mat_cpu, vec_cpu, beta, alpha, M, K):
    """int64 arithmetic with toward-zero truncated scalars (pre-wrap)."""
    beta_i = int(math.trunc(float(beta)))
    alpha_i = int(math.trunc(float(alpha)))
    sd = self_cpu.long().reshape(-1)
    if sd.numel() == 1:
        sd = sd.expand(M)
    t_self = beta_i * sd
    if K == 0:
        t_mat = torch.zeros(M, dtype=torch.int64)
    else:
        t_mat = alpha_i * (mat_cpu.long() @ vec_cpu.long())
    return t_self + t_mat


def _wrap_int32(raw):
    """Two's-complement wrap to int32 (P13)."""
    return ((raw + 2 ** 31) % 2 ** 32) - 2 ** 31


# ═══════════════════════════════════════════════════════════════════════════
# Checkers (test plan §2.5)
# ═══════════════════════════════════════════════════════════════════════════
def _assert_contract(out, M, dtype, cid):
    assert out.shape == (M,), f"{cid}: out.shape {tuple(out.shape)} != ({M},)"
    assert out.dtype == dtype, f"{cid}: out.dtype {out.dtype} != {dtype}"
    assert out.is_contiguous(), f"{cid}: out must be contiguous"
    assert "npu" in str(out.device), f"{cid}: out.device {out.device} is not npu"


def _check_float(actual_cpu, golden_f64, dtype, cid, subnormal_guard=False):
    af = actual_cpu.double()
    exp_dt = golden_f64.to(dtype).double()    # reference rounded to output dtype (IEEE), for mask + finiteness

    # 断言 2: NaN / +Inf / -Inf masks match element-wise (equal_nan semantics, stricter)
    assert torch.equal(torch.isnan(af), torch.isnan(exp_dt)), \
        f"{cid}: NaN mask mismatch\n actual={actual_cpu.tolist()}\n golden={golden_f64.to(dtype).tolist()}"
    assert torch.equal(af == INF, exp_dt == INF), \
        f"{cid}: +Inf mask mismatch\n actual={actual_cpu.tolist()}\n golden={golden_f64.to(dtype).tolist()}"
    assert torch.equal(af == NINF, exp_dt == NINF), \
        f"{cid}: -Inf mask mismatch\n actual={actual_cpu.tolist()}\n golden={golden_f64.to(dtype).tolist()}"

    # 断言 3: MERE/MARE only on elements whose expected value is finite (数值精度判据，容差/期望不变)
    finite = torch.isfinite(exp_dt)
    if finite.any():
        a = af[finite]
        g = golden_f64[finite]
        rel = (a - g).abs() / (g.abs() + 1e-7)
        mere = rel.mean().item()
        mare = rel.max().item()
        th = _THRESH[dtype]
        # allclose / max-abs-diff only for diagnostics (not a pass criterion, 测试方案 §2.5)
        max_abs = (a - g).abs().max().item()
        assert mere < th and mare < 10 * th, \
            f"{cid}: precision FAIL dtype={dtype} MERE={mere:.3e} (<{th:.3e}) " \
            f"MARE={mare:.3e} (<{10 * th:.3e}) max_abs_diff={max_abs:.3e}"

    # 断言 3b (白盒 3.3 加强，专探 fp16/bf16 输出 Cast 的次正规 flush；MERE 之外，非放宽):
    #   断言 3 的分母 |g|+1e-7 中 eps=1e-7 会淹没 |g|~1e-39 级次正规的误差——即便 NPU 把次正规
    #   输出 flush 成 0，MERE ≈ |g|/1e-7 仍远小于阈值而误判 pass（L1-188/190 盲区）。故对「期望值
    #   本身是次正规」的元素，额外断言 逐位相等 + 非零：这是行为断言（是否保留次正规），与上方数值
    #   判据分工互补，容差与期望值均未改动。依据：ops-precision-standard MERE 的 eps 对次正规无鉴别力。
    if subnormal_guard:
        exp_sub = golden_f64.to(dtype)
        sub = (exp_dt != 0) & torch.isfinite(exp_dt)
        assert sub.any(), f"{cid}: subnormal_guard set but expected has no finite non-zero element"
        assert (af[sub] != 0).all(), \
            f"{cid}: subnormal expected value flushed to 0 (FTZ/flush regression; MERE eps-masked it)\n" \
            f" actual={actual_cpu.tolist()}"
        assert torch.equal(actual_cpu[sub], exp_sub[sub]), \
            f"{cid}: subnormal not bit-exact (abs err != 0)\n actual={actual_cpu.tolist()}\n" \
            f" expected={exp_sub.tolist()}"


def _check_int(actual_cpu, golden_raw, cid, check_range=True):
    if check_range and golden_raw.numel() > 0:
        assert golden_raw.min().item() >= _INT32_MIN and golden_raw.max().item() <= _INT32_MAX, \
            f"{cid}: golden {golden_raw.tolist()} escaped int32 range (data-gen bug)"
    expected = _wrap_int32(golden_raw).to(torch.int32)
    assert torch.equal(actual_cpu.to(torch.int32), expected), \
        f"{cid}: int32 bit-exact FAIL\n actual={actual_cpu.tolist()}\n expected={expected.tolist()}"


# ═══════════════════════════════════════════════════════════════════════════
# Unified case runner
# ═══════════════════════════════════════════════════════════════════════════
def _run_case(spec):
    cid = spec["id"]
    M, K, dtype = spec["M"], spec["K"], spec["dtype"]
    beta, alpha = spec["beta"], spec["alpha"]
    form = spec.get("form", "normal")

    self_cpu, mat_cpu, vec_cpu = _build(spec)

    if dtype == torch.int32:
        golden_raw = _golden_int_raw(self_cpu, mat_cpu, vec_cpu, beta, alpha, M, K)
    else:
        golden = _golden_float(self_cpu, mat_cpu, vec_cpu, beta, alpha, M, K)
        if spec.get("goodcond"):              # 良态前置断言 (仅 D2/D4/D5)：只校验前提、不放宽阈值
            fin = torch.isfinite(golden)
            gv = golden[fin].abs()
            if gv.numel() > 0:
                assert gv.min().item() >= 1e-3 * gv.mean().item(), \
                    f"{cid}: ill-conditioned sample (min|g|<1e-3*mean|g|); design-time seed needs re-pick"

    self_npu = _self_to_npu(self_cpu, form, M, cid)
    mat_npu = mat_cpu.npu()
    vec_npu = vec_cpu.npu()
    out = torch.ops.ops_multimodal_fusion.addmv(self_npu, mat_npu, vec_npu, beta, alpha)
    _assert_contract(out, M, dtype, cid)
    actual = out.cpu()

    if dtype == torch.int32:
        _check_int(actual, golden_raw, cid)
    else:
        _check_float(actual, golden, dtype, cid, subnormal_guard=spec.get("subnormal_guard", False))

    if spec.get("large"):                     # 测试方案 6.2 条款5：大 shape 用完即释放
        del out, actual, self_npu, mat_npu, vec_npu
        torch.npu.empty_cache()


def _device_health_check():
    """After a rejected call, confirm the device context is not poisoned (测试方案 4.3 / C1)."""
    g = torch.Generator().manual_seed(12345)
    s = torch.rand(8, generator=g, dtype=torch.float64) + 0.5
    m = torch.rand(8, 16, generator=g, dtype=torch.float64) + 0.5
    v = torch.rand(16, generator=g, dtype=torch.float64) + 0.5
    out = torch.ops.ops_multimodal_fusion.addmv(
        s.float().npu(), m.float().npu(), v.float().npu(), 1, 1).cpu().double()
    golden = s + m @ v
    rel = ((out - golden).abs() / (golden.abs() + 1e-7)).max().item()
    assert rel < _THRESH[torch.float32], \
        f"device health check FAILED: addmv wrong after exception (rel={rel:.3e}); C1 not honored?"


# fp/int helpers for concise spec construction --------------------------------
def _c(cid, M, K, dtype, beta=1, alpha=1, family="D1", form="normal",
       poison=None, large=False, goodcond=False, matval=None, vecval=None, selfval=None,
       subnormal_guard=False):
    return dict(id=cid, M=M, K=K, dtype=dtype, beta=beta, alpha=alpha, family=family, form=form,
                poison=poison or [], seed=_seed(cid), large=large, goodcond=goodcond,
                matval=matval, vecval=vecval, selfval=selfval, subnormal_guard=subnormal_guard)


F32, F16, BF16, I32 = torch.float32, torch.float16, torch.bfloat16, torch.int32
FLOATS = [F32, F16, BF16]
ALL_DTYPES = [F32, F16, BF16, I32]


# ═══════════════════════════════════════════════════════════════════════════
# L0 门槛用例 (17 条：L0-01 接口 + L0-02~17)
# ═══════════════════════════════════════════════════════════════════════════
L0_CASES = [
    _c("L0-02", 64, 128, F32, 1, 1),
    _c("L0-03", 64, 128, F16, 1, 1),
    _c("L0-04", 64, 128, BF16, 1, 1),
    _c("L0-05", 64, 128, I32, 1, 1, family="D1i"),
    _c("L0-06", 64, 128, F32, 2.5, -1.5),
    _c("L0-07", 64, 128, F32, 0, 1, family="D1", poison=[("self", "all", "nan")]),  # beta=0 忽略 self(全 NaN)
    _c("L0-08", 64, 128, F32, 2, 0),                                                # alpha=0 -> out=2*self
    _c("L0-09", 100, 1023, F32, 1, 1),
    _c("L0-10", 1, 1024, F32, 1, 1, form="scalar1"),                                # M=1 走广播分支
    _c("L0-11", 1024, 1, F32, 1, 1),
    _c("L0-12", 56, 512, F32, 1, 1),
    _c("L0-13", 0, 64, F32, 1, 1),                                                  # 空张量 M=0
    _c("L0-14", 64, 0, F32, 2, 1),                                                  # K=0 -> out=2*self
    _c("L0-15", 64, 128, F32, 1, 1, form="scalar1"),                               # 广播 (1,)
    _c("L0-16", 64, 128, F32, 1, 1, form="zerodim"),                               # 广播 0-dim
    _c("L0-17", 64, 128, F32, 1, 1, form="expand"),                                # 广播 expand view
]


@requires_npu
@pytest.mark.parametrize("spec", L0_CASES, ids=[c["id"] for c in L0_CASES])
def test_addmv_l0_basic(spec):
    _run_case(spec)


# ═══════════════════════════════════════════════════════════════════════════
# L1 功能用例 (194 条：190 正常执行 + 4 环境 skip)
# ═══════════════════════════════════════════════════════════════════════════
def _g1():
    shapes = [(2, 8), (8, 64), (64, 128), (100, 1000), (256, 127), (333, 1023)]
    out = []
    n = 1
    for (M, K) in shapes:                     # shape-major, dtype-minor
        for dt in ALL_DTYPES:
            cid = f"L1-{n:03d}"
            if dt == I32:
                out.append(_c(cid, M, K, dt, 2, 3, family="D1i"))
            else:
                out.append(_c(cid, M, K, dt, 1.5, 0.75, family="D1"))
            n += 1
    return out


def _g2():
    out = []
    n = 25
    for K in [7, 31, 33, 127, 129, 1023, 4097]:  # K-major
        for dt in [F32, BF16]:
            out.append(_c(f"L1-{n:03d}", 100, K, dt, 1, 1, family="D1"))
            n += 1
    return out


def _g3():
    out = []
    out += [_c(f"L1-{39 + i:03d}", 1, 1024, dt, 1, 1,
               family="D1i" if dt == I32 else "D1", form="scalar1")
            for i, dt in enumerate(ALL_DTYPES)]                     # 039-042
    out += [_c(f"L1-{43 + i:03d}", 1024, 1, dt, 1, 1,
               family="D1i" if dt == I32 else "D1")
            for i, dt in enumerate(ALL_DTYPES)]                     # 043-046
    out += [_c(f"L1-{47 + i:03d}", 1, 1, dt, 1, 1,
               family="D1i" if dt == I32 else "D1", form="scalar1")
            for i, dt in enumerate(ALL_DTYPES)]                     # 047-050
    out += [_c("L1-051", 1, 65536, F32, 1, 1, form="scalar1")]
    out += [_c("L1-052", 65536, 1, F32, 1, 1, large=True)]
    return out


def _g4():
    return [
        _c("L1-053", 65536, 1024, F32, 1, 1, large=True),
        _c("L1-054", 4096, 16384, F32, 1, 1, large=True),
        _c("L1-055", 1024, 65536, F32, 1, 1, large=True),
        _c("L1-056", 8192, 8192, F32, 1, 1, large=True),
        _c("L1-057", 4096, 16384, F16, 1, 1, large=True),
        _c("L1-058", 4096, 16384, BF16, 1, 1, large=True),
        _c("L1-059", 4096, 16384, I32, 1, 1, family="D1i", large=True),
        _c("L1-060", 65536, 1024, F16, 1, 1, large=True),
    ]


def _g5():
    out = []
    ms = [8, 32, 55, 56, 57, 111, 112, 113, 168, 224]
    for i, M in enumerate(ms):
        out.append(_c(f"L1-{61 + i:03d}", M, 512, F32, 1, 1))     # 061-070
    for i, M in enumerate([55, 56, 57]):
        out.append(_c(f"L1-{71 + i:03d}", M, 512, F16, 1, 1))     # 071-073
    return out


def _g6():
    return [
        _c("L1-074", 0, 64, F32, 1, 1),
        _c("L1-075", 64, 0, F32, 2, 1),
        _c("L1-076", 0, 0, F32, 1, 1),
        _c("L1-077", 0, 64, F16, 1, 1),
        _c("L1-078", 64, 0, F16, 2, 1),
        _c("L1-079", 0, 64, BF16, 1, 1),
        _c("L1-080", 64, 0, BF16, 2, 1),
        _c("L1-081", 0, 64, I32, 1, 1, family="D1i"),
        _c("L1-082", 64, 0, I32, 2, 1, family="D1i"),
        _c("L1-083", 64, 0, F32, 2, 1, form="zerodim"),           # K=0 + 广播 self 叠加
    ]


def _g7():
    out = []
    forms = ["zerodim", "scalar1", "expand"]
    n = 84
    for fm in forms:                          # form-major x 4 dtype @ (64,128)
        for dt in ALL_DTYPES:
            out.append(_c(f"L1-{n:03d}", 64, 128, dt, 1, 1,
                          family="D1i" if dt == I32 else "D1", form=fm))
            n += 1
    for i, fm in enumerate(forms):            # 3 form x f32 @ (4096,16384) 大 shape
        out.append(_c(f"L1-{96 + i:03d}", 4096, 16384, F32, 1, 1, form=fm, large=True))
    for i, fm in enumerate(forms):            # 3 form x f32 @ (64,128) beta=0
        out.append(_c(f"L1-{99 + i:03d}", 64, 128, F32, 0, 1, form=fm))
    return out


def _g8():
    out = []
    f32_pairs = [(1, 1), (0, 1), (2, 0), (0, 0), (2.5, -1.5), (-1, 2), (1e-3, 1e3), (1e3, 1e-3)]
    for i, (b, a) in enumerate(f32_pairs):
        out.append(_c(f"L1-{102 + i:03d}", 64, 128, F32, b, a))              # 102-109
    for i, (b, a) in enumerate([(0, 1), (2, 0), (0, 0)]):
        out.append(_c(f"L1-{110 + i:03d}", 64, 128, F16, b, a))             # 110-112
    for i, (b, a) in enumerate([(0, 1), (2, 0), (0, 0)]):
        out.append(_c(f"L1-{113 + i:03d}", 64, 128, BF16, b, a))            # 113-115
    for i, (b, a) in enumerate([(0, 1), (2, 0), (0, 0), (2, 3), (-1, 2)]):
        out.append(_c(f"L1-{116 + i:03d}", 64, 128, I32, b, a, family="D1i"))  # 116-120
    trunc = [(0.9, 0.9), (2.5, 1), (1, 1.5), (-2.5, -1.5)]
    for i, (b, a) in enumerate(trunc):
        out.append(_c(f"L1-{121 + i:03d}", 64, 128, I32, b, a, family="D1i"))  # 121-124 D1t
    return out


def _g9():
    out = []
    # SV1 125-127: self 含 NaN, beta=1 -> 传播
    for i, dt in enumerate(FLOATS):
        out.append(_c(f"L1-{125 + i:03d}", 64, 128, dt, 1, 1, family="D1", poison=[("self", 0, "nan")]))
    # SV2 128-130: self 含 NaN, beta=0 -> 不传播
    for i, dt in enumerate(FLOATS):
        out.append(_c(f"L1-{128 + i:03d}", 64, 128, dt, 0, 1, family="D1", poison=[("self", 0, "nan")]))
    # SV3 131-132: self 含 +Inf/-Inf, beta=1 (f32)
    out.append(_c("L1-131", 64, 128, F32, 1, 1, family="D1", poison=[("self", 0, "inf")]))
    out.append(_c("L1-132", 64, 128, F32, 1, 1, family="D1", poison=[("self", 0, "ninf")]))
    # SV4 133: self 含 +Inf, beta=0 -> 不传播
    out.append(_c("L1-133", 64, 128, F32, 0, 1, family="D1", poison=[("self", 0, "inf")]))
    # SV5 134: mat 第 0 行含 NaN, alpha=1 -> 仅 out[0] NaN
    out.append(_c("L1-134", 64, 128, F32, 1, 1, family="ones", poison=[("mat", (0, 0), "nan")]))
    # SV6 135: vec 含 NaN, alpha=1 -> out 全 NaN
    out.append(_c("L1-135", 64, 128, F32, 1, 1, family="ones", poison=[("vec", 0, "nan")]))
    # SV7 136: mat +Inf 且 vec 对应位=0 -> out[0] NaN (Inf x 0)
    out.append(_c("L1-136", 64, 128, F32, 1, 1, family="ones",
                  poison=[("mat", (0, 0), "inf"), ("vec", 0, 0.0)]))
    # SV8 137-144: alpha==0 统一短路 (beta=2, 双零用 beta=0)
    out.append(_c("L1-137", 64, 128, F32, 2, 0, family="ones", poison=[("mat", (0, 0), "nan")]))
    out.append(_c("L1-138", 32, 128, F16, 2, 0, family="ones", poison=[("mat", (0, 0), "nan")]))
    out.append(_c("L1-139", 64, 128, F16, 2, 0, family="ones", poison=[("mat", (0, 0), "nan")]))
    out.append(_c("L1-140", 32, 128, BF16, 2, 0, family="ones", poison=[("mat", (0, 0), "nan")]))
    out.append(_c("L1-141", 33, 128, BF16, 2, 0, family="ones", poison=[("mat", (0, 0), "nan")]))
    out.append(_c("L1-142", 64, 128, F16, 2, 0, family="ones", poison=[("vec", 0, "inf")]))
    out.append(_c("L1-143", 64, 128, F32, 0, 0, family="ones",
                  poison=[("self", "all", "nan"), ("mat", (0, 0), "nan")]))
    out.append(_c("L1-144", 64, 128, F16, 0, 0, family="ones",
                  poison=[("self", "all", "nan"), ("mat", (0, 0), "nan")]))
    # SV9 145-146: 全零输入
    out.append(_c("L1-145", 64, 128, F32, 1, 1, family="zeros"))
    out.append(_c("L1-146", 64, 128, I32, 1, 1, family="zeros"))
    # SV10a 147: fp16 最终真值超范围 -> +Inf (类别 ③)
    out.append(_c("L1-147", 4, 8, F16, 1, 1, family="const", matval=100.0, vecval=100.0, selfval=0.0))
    # SV10b 148: fp16 部分和溢出但最终有限 -> 20000 (类别 ①, 判别性)
    out.append(_c("L1-148", 4, 3, F16, 1, 1, family="const", matval=40000.0, vecval=1.0, selfval=0.0,
                  poison=[("mat", (slice(None), 2), -60000.0)]))
    # SV11 149: bf16 大值域
    out.append(_c("L1-149", 64, 128, BF16, 1, 1, family="D6"))
    # SV12 150-151: 小值域 (良态断言)
    out.append(_c("L1-150", 64, 128, F32, 1, 1, family="D5", goodcond=True))
    out.append(_c("L1-151", 64, 128, F16, 1, 1, family="D5", goodcond=True))
    # SV13 152-153: 离群点分布 (良态断言)
    out.append(_c("L1-152", 64, 128, F32, 1, 1, family="D4", goodcond=True))
    out.append(_c("L1-153", 64, 128, F16, 1, 1, family="D4", goodcond=True))
    # SV14 154: fp16 次正规输入 (风险用例 R7) —— 输出 1.526e-5 亦为 fp16 次正规，加强断言探 flush
    out.append(_c("L1-154", 64, 128, F16, 0, 1, family="const", matval=1e-7, vecval=1.0, selfval=0.0,
                  subnormal_guard=True))
    # SV15 155: int32 正/负/零混合
    out.append(_c("L1-155", 64, 128, I32, 2, 3, family="D1i"))
    # SV16 156-158: 一般正负分布 (良态断言) @ (256,1024)
    out.append(_c("L1-156", 256, 1024, F32, 1, 1, family="D2", goodcond=True))
    out.append(_c("L1-157", 256, 1024, F16, 1, 1, family="D2", goodcond=True))
    out.append(_c("L1-158", 256, 1024, BF16, 1, 1, family="D2", goodcond=True))
    # SV17-A 159-166: 判零口径探针 (P9 + W2 分项公式)
    out.append(_c("L1-159", 64, 128, F16, 2, 1e-8, family="ones", poison=[("mat", (0, 0), "nan")]))
    out.append(_c("L1-160", 64, 128, F16, 2, 0.0, family="ones", poison=[("mat", (0, 0), "nan")]))
    out.append(_c("L1-161", 64, 128, BF16, 2, 1e-42, family="ones", poison=[("mat", (0, 0), "nan")]))
    out.append(_c("L1-162", 64, 128, BF16, 2, 0.0, family="ones", poison=[("mat", (0, 0), "nan")]))
    out.append(_c("L1-163", 64, 128, F16, 1e-8, 1, family="ones", poison=[("self", "all", "nan")]))
    out.append(_c("L1-164", 64, 128, F16, 0.0, 1, family="ones", poison=[("self", "all", "nan")]))
    # 混合格 (最强单点探针)：beta 精确 0 + alpha 量化为 0; mat 除 [0,0]=NaN 外全 0
    out.append(_c("L1-165", 64, 128, F16, 0.0, 1e-8, family="zeros",
                  poison=[("self", "all", "nan"), ("mat", (0, 0), "nan"), ("vec", "all", 1.0)]))
    out.append(_c("L1-166", 64, 128, F16, 0.0, 0.0, family="zeros",
                  poison=[("self", "all", "nan"), ("mat", (0, 0), "nan"), ("vec", "all", 1.0)]))
    # SV17-B 167-172,186-187: 差异 D2 (W1, scalar=1e-46)
    out.append(_c("L1-167", 64, 128, F32, 2, 1e-46, family="ones", poison=[("mat", (0, 0), "nan")]))
    out.append(_c("L1-168", 64, 128, F32, 2, 0.0, family="ones", poison=[("mat", (0, 0), "nan")]))
    out.append(_c("L1-169", 64, 128, F32, 1e-46, 1, family="ones", poison=[("self", "all", "nan")]))
    out.append(_c("L1-170", 64, 128, F32, 0.0, 1, family="ones", poison=[("self", "all", "nan")]))
    out.append(_c("L1-171", 64, 128, BF16, 2, 1e-46, family="ones", poison=[("mat", (0, 0), "nan")]))
    out.append(_c("L1-172", 64, 128, BF16, 2, 0.0, family="ones", poison=[("mat", (0, 0), "nan")]))
    out.append(_c("L1-186", 64, 128, BF16, 1e-46, 1, family="ones", poison=[("self", "all", "nan")]))
    out.append(_c("L1-187", 64, 128, BF16, 0.0, 1, family="ones", poison=[("self", "all", "nan")]))
    # SV17-C 173-175: 差异 D3 (W2, 全有限)
    out.append(_c("L1-173", 64, 128, F16, 1, 1e-8, family="const", matval=234.375, vecval=1.0, selfval=0.0))
    out.append(_c("L1-174", 32, 128, BF16, 1, 1e-42, family="const", matval=234.0, vecval=1.0, selfval=0.0))
    out.append(_c("L1-175", 64, 128, F16, 1e-8, 0.0, family="const", matval=1.0, vecval=1.0, selfval=30000.0))
    # SV18 176-185: 低精度特殊值补充
    out.append(_c("L1-176", 64, 128, F16, 1, 1, family="D1", poison=[("self", 0, "inf")]))
    out.append(_c("L1-177", 64, 128, BF16, 1, 1, family="D1", poison=[("self", 0, "inf")]))
    out.append(_c("L1-178", 64, 128, F16, 0, 1, family="D1", poison=[("self", 0, "inf")]))
    out.append(_c("L1-179", 64, 128, BF16, 0, 1, family="D1", poison=[("self", 0, "inf")]))
    out.append(_c("L1-180", 64, 128, F16, 1, 1, family="ones", poison=[("mat", (0, 0), "nan")]))
    out.append(_c("L1-181", 64, 128, BF16, 1, 1, family="ones", poison=[("mat", (0, 0), "nan")]))
    out.append(_c("L1-182", 64, 128, F16, 1, 1, family="ones", poison=[("vec", 0, "nan")]))
    out.append(_c("L1-183", 64, 128, BF16, 1, 1, family="ones", poison=[("vec", 0, "nan")]))
    out.append(_c("L1-184", 64, 128, F16, 1, 1, family="ones",
                  poison=[("mat", (0, 0), "inf"), ("vec", 0, 0.0)]))
    out.append(_c("L1-185", 64, 128, BF16, 1, 1, family="ones",
                  poison=[("mat", (0, 0), "inf"), ("vec", 0, 0.0)]))
    # SV19 188-190: 次正规 Cast (P4 已定稿, 走类别 ①) —— 加强断言探 bf16/fp16 Cast flush (白盒 3.3 任务 A)
    out.append(_c("L1-188", 64, 100, BF16, 0, 1, family="const", matval=2.0 ** -133, vecval=1.0, selfval=0.0,
                  subnormal_guard=True))
    out.append(_c("L1-189", 64, 100, F16, 0, 1, family="const", matval=2.0 ** -24, vecval=1.0, selfval=0.0,
                  subnormal_guard=True))
    out.append(_c("L1-190", 32, 100, BF16, 0, 1, family="const", matval=2.0 ** -133, vecval=1.0, selfval=0.0,
                  subnormal_guard=True))
    return out


L1_CASES = _g1() + _g2() + _g3() + _g4() + _g5() + _g6() + _g7() + _g8() + _g9()


@requires_npu
@pytest.mark.parametrize("spec", L1_CASES, ids=[c["id"] for c in L1_CASES])
def test_addmv_l1_functional(spec):
    _run_case(spec)


# G10 191-194: 非连续输入 (环境 skip — torch_npu 缺 D2D strided copy, error 561103; 需求 3.3)
_SKIP_REASON = (
    "torch_npu in the current CANN release lacks D2D strided copy (error 561103): non-contiguous "
    "mat/vec and non-stride-0 non-contiguous self cannot be moved to / made contiguous on NPU. The "
    "kernel handles them via host .contiguous(); re-enable once torch_npu ships strided D2D "
    "(需求 3.3 / 测试方案 6.3, 用户知情确认)."
)


@pytest.mark.skip(reason=_SKIP_REASON)
@pytest.mark.parametrize("cid", ["L1-191", "L1-192", "L1-193", "L1-194"])
def test_addmv_l1_non_contiguous(cid):
    K, M = 128, 64
    if cid == "L1-191":                       # mat 转置视图
        mat = torch.randn(K, M).npu().t()
        vec = torch.randn(K).npu()
        self_ = torch.randn(M).npu()
    elif cid == "L1-192":                     # mat 跨步视图
        mat = torch.randn(M, 2 * K).npu()[:, ::2]
        vec = torch.randn(K).npu()
        self_ = torch.randn(M).npu()
    elif cid == "L1-193":                     # vec 跨步视图
        mat = torch.randn(M, K).npu()
        vec = torch.randn(2 * K).npu()[::2]
        self_ = torch.randn(M).npu()
    else:                                     # self 非 stride-0 跨步视图 (M>=2, 不命中规则 1)
        mat = torch.randn(M, K).npu()
        vec = torch.randn(K).npu()
        self_ = torch.randn(2 * M).npu()[::2]
    out = torch.ops.ops_multimodal_fusion.addmv(self_, mat, vec).cpu()
    assert out.shape == (M,)


# ═══════════════════════════════════════════════════════════════════════════
# L2 异常用例 (22 条)
# ═══════════════════════════════════════════════════════════════════════════
def _mk(shape, dtype, dev="npu"):
    if dtype == I32:
        t = torch.randint(-5, 5, shape, dtype=I32)
    else:
        t = torch.randn(*shape).to(dtype) if shape else torch.randn(()).to(dtype)
    return t.npu() if dev == "npu" else t


# L2-01..16 + L2-19 + L2-21 : 拦截定值，期望 RuntimeError，后置设备健康校验
def _rej_l2_01():   # mat 1-D
    return dict(self=_mk((64,), F32), mat=_mk((128,), F32), vec=_mk((128,), F32))


def _rej_l2_02():   # mat 3-D
    return dict(self=_mk((64,), F32), mat=_mk((4, 64, 128), F32), vec=_mk((128,), F32))


def _rej_l2_03():   # vec 2-D
    return dict(self=_mk((64,), F32), mat=_mk((64, 128), F32), vec=_mk((128, 1), F32))


def _rej_l2_04():   # vec 0-dim
    return dict(self=_mk((64,), F32), mat=_mk((64, 128), F32), vec=_mk((), F32))


def _rej_l2_05():   # vec.size(0) != mat.size(1)
    return dict(self=_mk((64,), F32), mat=_mk((64, 128), F32), vec=_mk((127,), F32))


def _rej_l2_06():   # self numel>1 且 != M
    return dict(self=_mk((65,), F32), mat=_mk((64, 128), F32), vec=_mk((128,), F32))


def _rej_l2_07():   # self 2-D
    return dict(self=_mk((64, 1), F32), mat=_mk((64, 128), F32), vec=_mk((128,), F32))


def _rej_l2_08():   # dtype 不一致 self=f32 mat=f16 vec=f16
    return dict(self=_mk((64,), F32), mat=_mk((64, 128), F16), vec=_mk((128,), F16))


def _rej_l2_09():   # dtype 不一致 mat=f32 vec=i32 self=f32
    return dict(self=_mk((64,), F32), mat=_mk((64, 128), F32), vec=_mk((128,), I32))


def _rej_dtype(bad):
    def f():
        return dict(self=torch.randn(64).to(bad).npu(), mat=torch.randn(64, 128).to(bad).npu(),
                    vec=torch.randn(128).to(bad).npu())
    return f


def _rej_l2_15():   # 三输入均 CPU
    return dict(self=_mk((64,), F32, "cpu"), mat=_mk((64, 128), F32, "cpu"), vec=_mk((128,), F32, "cpu"))


def _rej_l2_16():   # 混合 device: self CPU, mat/vec NPU
    return dict(self=_mk((64,), F32, "cpu"), mat=_mk((64, 128), F32), vec=_mk((128,), F32))


def _rej_l2_19():   # int32 标量越界 beta=2^31
    return dict(self=_mk((4,), I32), mat=_mk((4, 3), I32), vec=_mk((3,), I32), beta=2147483648.0)


def _rej_l2_21():   # 截断后在范围内但原值越界 beta=2147483647.5
    return dict(self=_mk((4,), I32), mat=_mk((4, 3), I32), vec=_mk((3,), I32), beta=2147483647.5)


L2_REJECT = [
    ("L2-01", _rej_l2_01), ("L2-02", _rej_l2_02), ("L2-03", _rej_l2_03), ("L2-04", _rej_l2_04),
    ("L2-05", _rej_l2_05), ("L2-06", _rej_l2_06), ("L2-07", _rej_l2_07), ("L2-08", _rej_l2_08),
    ("L2-09", _rej_l2_09),
    ("L2-10", _rej_dtype(torch.float64)), ("L2-11", _rej_dtype(torch.int64)),
    ("L2-12", _rej_dtype(torch.int8)), ("L2-13", _rej_dtype(torch.bool)),
    ("L2-14", _rej_dtype(torch.complex64)),
    ("L2-15", _rej_l2_15), ("L2-16", _rej_l2_16),
    ("L2-19", _rej_l2_19), ("L2-21", _rej_l2_21),
]


@requires_npu
@pytest.mark.parametrize("cid,factory", L2_REJECT, ids=[c for c, _ in L2_REJECT])
def test_addmv_l2_reject(cid, factory):
    kw = factory()
    beta = kw.pop("beta", 1)
    alpha = kw.pop("alpha", 1)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.addmv(kw["self"], kw["mat"], kw["vec"], beta, alpha)
    # 后置设备健康校验：host 侧 TORCH_CHECK 拦截不得污染设备上下文 (C1)
    _device_health_check()


# L2-17/18: 超上界不拦截 (信息性)，只断言未被拦截 + 输出契约，不断言数值正确性
@requires_npu
@pytest.mark.parametrize("cid,M,K", [("L2-17", 65537, 1), ("L2-18", 8192, 16384)],
                         ids=["L2-17", "L2-18"])
def test_addmv_l2_overbound(cid, M, K):
    g = torch.Generator().manual_seed(_seed(cid))
    s = 1.0 / math.sqrt(K)
    self_ = (torch.rand(M, generator=g) + 0.5)
    mat = (torch.rand(M, K, generator=g) + 0.5) * s
    vec = (torch.rand(K, generator=g) + 0.5) * s
    out = torch.ops.ops_multimodal_fusion.addmv(self_.npu(), mat.npu(), vec.npu(), 1, 1)
    assert out.shape == (M,), f"{cid}: over-bound shape {tuple(out.shape)} != ({M},)"
    assert out.dtype == F32 and "npu" in str(out.device)
    res = out.cpu()
    logging.info("%s over-bound (M=%d,K=%d) informational: mean=%.4g std=%.4g finite=%s",
                 cid, M, K, res.mean().item(), res.std().item(), bool(torch.isfinite(res).all()))
    del out, res
    torch.npu.empty_cache()


# L2-20: INT32_MIN 边界接受 (正常计算); L2-22: int32 算术回绕 P13
@requires_npu
def test_addmv_l2_20_int32_min_accepted():
    g = torch.Generator().manual_seed(_seed("L2-20"))
    self_ = torch.randint(-100, 101, (4,), generator=g, dtype=I32)
    mat = torch.randint(-3, 4, (4, 3), generator=g, dtype=I32)
    vec = torch.randint(-3, 4, (3,), generator=g, dtype=I32)
    beta = -2147483648.0  # INT32_MIN，谓词闭区间接受
    out = torch.ops.ops_multimodal_fusion.addmv(self_.npu(), mat.npu(), vec.npu(), beta, 0).cpu()
    raw = _golden_int_raw(self_, mat, vec, beta, 0, 4, 3)
    _check_int(out, raw, "L2-20", check_range=False)  # beta*self 可能回绕，按二补数比对


@requires_npu
def test_addmv_l2_22_int32_wraparound():
    self_ = torch.full((4,), 7, dtype=I32)
    mat = torch.full((4, 4), 3, dtype=I32)          # mat@vec = 4*3 = 12 per row
    vec = torch.full((4,), 1, dtype=I32)
    beta, alpha = 1, _INT32_MAX
    out = torch.ops.ops_multimodal_fusion.addmv(self_.npu(), mat.npu(), vec.npu(), beta, alpha).cpu()
    raw = _golden_int_raw(self_, mat, vec, beta, alpha, 4, 4)     # 7 + 2147483647*12 = 25769803771
    assert raw[0].item() == 25769803771, "L2-22 data-gen check: raw math value"
    _check_int(out, raw, "L2-22", check_range=False)             # 回绕 -> -5
    assert _wrap_int32(raw)[0].item() == -5, "L2-22 expected wrap == -5"


# ═══════════════════════════════════════════════════════════════════════════
# 白盒补全 (3.3)：反解源码分支、补黑盒未保证覆盖的执行路径
#   见 .cannbot/3.3-白盒覆盖说明.md 的分支枚举与「分支→用例」映射。
# ═══════════════════════════════════════════════════════════════════════════

# WB-K 组：K 切分多块 + 非对齐尾 k-chunk。K=65535：一行 mat tile 需 65535*perElemBytes ≥ 655 KiB
# 远超单核 UB，故 `k*perElemBytes > tileBudget` 恒成立 -> numKChunks ≥ 2；且 65535 % 64 = 63 ≠ 0,
# 保证最后一个 k-chunk 的 kLen 非 VL(64) 对齐 -> 尾 chunk 走 UpdateMask ZEROING 掩码路径。
# 同时 kt<k -> mt=1 (不变式 I1)，且 b16 走「每 chunk 转 vec + 跨 chunk 累加」路径。
# M=256 (256*65535 = 16.78M < 2^26=67.11M，在验证范围内)。
WB_KCHUNK = [
    _c("WB-K01", 256, 65535, F32, 1.5, 0.75, family="D1", large=True),
    _c("WB-K02", 256, 65535, F16, 1.5, 0.75, family="D1", large=True),
    _c("WB-K03", 256, 65535, BF16, 1.5, 0.75, family="D1", large=True),
    _c("WB-K04", 256, 65535, I32, 2, 3, family="D1i", large=True),
]


@requires_npu
@pytest.mark.parametrize("spec", WB_KCHUNK, ids=[c["id"] for c in WB_KCHUNK])
def test_addmv_wb_kchunk(spec):
    _run_case(spec)


# WB-M：Meta 分派路径 (addmv_check checkDevice=false + Meta 形状推断)。eager 用例走 PrivateUse1，
# 不触发 Meta 键；torch.compile / autograd / meta 设备才走 Meta。此处用 meta 张量直接触发。
# WB-INT：int32 alpha 越界拒绝。addmv_check 有两条顺序的 int32 范围 TORCH_CHECK (beta 先、alpha 后)；
# L2-19/L2-21 只触发 beta 那条，alpha 拒绝分支 (源码 L139) 黑盒未覆盖。beta 传合法值使 beta 检查通过、
# 再让 alpha 越界，方能触达并验证 alpha 分支。
@requires_npu
@pytest.mark.parametrize("alpha", [2147483648.0, 2147483647.5], ids=["WB-INT-alpha-2^31", "WB-INT-alpha-.5"])
def test_addmv_wb_int32_alpha_overflow(alpha):
    self_ = _mk((4,), I32)
    mat = _mk((4, 3), I32)
    vec = _mk((3,), I32)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.addmv(self_, mat, vec, 1, alpha)   # beta 合法, alpha 越界
    _device_health_check()


@requires_npu
def test_addmv_wb_meta_shape_inference():
    for (M, K, dt) in [(64, 128, F32), (100, 1023, F16), (0, 64, BF16), (65536, 1, I32)]:
        s = torch.empty(M, device="meta", dtype=dt)
        m = torch.empty(M, K, device="meta", dtype=dt)
        v = torch.empty(K, device="meta", dtype=dt)
        out = torch.ops.ops_multimodal_fusion.addmv(s, m, v, 1, 1)
        assert out.shape == (M,) and out.dtype == dt and out.device.type == "meta", \
            f"meta infer wrong: {tuple(out.shape)} {out.dtype} {out.device}"
    # Meta 路径同样跑 K1~K5 校验 (checkDevice=false)：坏 shape 必须抛错，且不触及 device 校验
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.addmv(
            torch.empty(64, device="meta"), torch.empty(128, device="meta"),  # mat 1-D
            torch.empty(128, device="meta"))
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.addmv(
            torch.empty(64, device="meta"), torch.empty(64, 128, device="meta"),
            torch.empty(127, device="meta"))                                    # vec/mat 尺寸不匹配
