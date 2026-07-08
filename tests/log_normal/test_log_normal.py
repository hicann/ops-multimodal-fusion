#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Tianjin University Ltd
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tests for ops_multimodal_fusion.log_normal.

Fills a tensor (shape/dtype/device taken from x; x data unused) with i.i.d.
LogNormal(mean, std) samples via Box-Muller + exp on an AscendC::PhiloxRandom
uniform pair:
    R = sqrt(-2 ln(max(u1, FLT_MIN))); Z = R*cos(2*pi*u2)
    X = exp(Z*std + mean)

The NPU Philox counter->element map differs from PyTorch's curand, so
bit-exact comparison is impossible. LogNormal is heavy-tailed in X but its
log is exactly normal, so acceptance is done in log space + structural:
  - mean(ln X)   ~= mean        (ln X ~ N(mean, std^2), light-tailed)
  - std(ln X)    ~= std
  - median(X)    ~= exp(mean)   (log-normal median)
  - every sample > 0            (log-normal support is (0, inf))
  - same seed + shape => identical output; different seed => different
  - shape/dtype preserved; std<=0 => host reject

Stats are computed on the finite, strictly-positive subset in float64.

Test case counts:
  - test_log_normal_small            : 10 cases
  - test_log_normal_large            : 9 cases
  - test_log_normal_interface_exist  : 1
  - test_log_normal_invalid_std      : 2 (std=0.0, -1.0)
  - Total                            : 22 cases
"""

import math
from collections import namedtuple

import pytest
import torch
import torch_npu  # noqa: F401  (registers NPU dispatch key)

import ops_multimodal_fusion  # noqa: F401  (registers torch.ops.ops_multimodal_fusion.log_normal)

if not hasattr(torch.ops.ops_multimodal_fusion, "log_normal"):
    pytest.skip(
        "ops_multimodal_fusion.log_normal not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_log_normal_interface_exist():
    """The 'ops_multimodal_fusion.log_normal' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "log_normal"),\
        "The 'log_normal' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _run_raw(shape, mean, std, seed, dtype):
    """Return the sampled tensor on CPU in its native dtype."""
    x = torch.empty(*shape, dtype=dtype).npu()
    out = torch.ops.ops_multimodal_fusion.log_normal(x, float(mean), float(std), int(seed))
    assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"
    assert out.dtype == dtype, f"dtype mismatch: {out.dtype} vs {dtype}"
    return out.cpu()


def _pos_finite(shape, mean, std, seed, dtype):
    """Sampled tensor as float64, restricted to the finite & >0 subset."""
    raw = _run_raw(shape, mean, std, seed, dtype)
    v = raw.to(torch.float64).flatten()
    keep = torch.isfinite(v) & (v > 0.0)
    n = v.numel()
    assert keep.sum().item() >= 0.98 * n,\
        f"too many non-finite/non-positive samples: {n - int(keep.sum())}/{n}"
    return v[keep], n


def _assert_logmoments(shape, mean, std, seed, dtype):
    """Log-space moments of the samples match the lognormal parameters."""
    pos, n = _pos_finite(shape, mean, std, seed, dtype)
    lg = torch.log(pos)
    lmean = lg.mean().item()
    lstd = lg.std().item()
    med = pos.median().item()
    med_t = math.exp(mean)

    tight = n >= (1 << 20)
    if dtype == torch.float32:
        m_tol = (0.02 if tight else 0.05) * (std + 1.0)
        s_rel = 0.04 if tight else 0.08
        d_rel = 0.04 if tight else 0.08
    else:  # fp16: quantization + RNG implementation differences
        m_tol = 0.12 * (std + 1.0)
        s_rel = 0.15
        d_rel = 0.15

    assert abs(lmean - mean) < m_tol,\
        f"mean(lnX) {lmean:.5f} off mean {mean} (abs tol {m_tol:.5f})"
    assert abs(lstd - std) / std < s_rel,\
        f"std(lnX) {lstd:.5f} off std {std} (rel tol {s_rel})"
    assert abs(med - med_t) / med_t < d_rel,\
        f"median(X) {med:.5f} off exp(mean) {med_t:.5f} (rel tol {d_rel})"


def _assert_positive(shape, mean, std, seed, dtype):
    """fp32: every sample finite and strictly > 0; fp16: >=98% so."""
    raw = _run_raw(shape, mean, std, seed, dtype)
    v = raw.to(torch.float64)
    good = torch.isfinite(v) & (v > 0.0)
    if dtype == torch.float32:
        assert good.all(), "fp32 log-normal output must be finite and > 0"
    else:
        assert good.float().mean().item() >= 0.98,\
            "fp16 log-normal output must be mostly finite and > 0"


def _assert_shape_finite(shape, mean, std, seed, dtype):
    """Shape/dtype preserved; finite & >0 dominated."""
    raw = _run_raw(shape, mean, std, seed, dtype)
    v = raw.to(torch.float64)
    good = torch.isfinite(v) & (v > 0.0)
    if dtype == torch.float32:
        assert good.all(), "fp32 log-normal output must be finite and > 0"
    else:
        assert good.float().mean().item() >= 0.98, "fp16 mostly finite and > 0"


def _assert_determinism(shape, mean, std, seed, dtype):
    """Same seed and shape produce byte-identical output."""
    a = _run_raw(shape, mean, std, seed, dtype)
    b = _run_raw(shape, mean, std, seed, dtype)
    assert torch.equal(a, b), "same seed must give identical samples"


def _assert_divergence(shape, mean, std, seed, dtype):
    """A different seed produces substantially different output."""
    a = _run_raw(shape, mean, std, seed, dtype)
    b = _run_raw(shape, mean, std, seed + 1, dtype)
    assert not torch.equal(a, b), "different seeds must give different samples"
    diff = (a.to(torch.float64) != b.to(torch.float64)).float().mean().item()
    assert diff > 0.5, f"streams too correlated across seeds (diff frac {diff:.3f})"


_DISPATCH = {
    "logm": _assert_logmoments,
    "pos": _assert_positive,
    "fin": _assert_shape_finite,
    "det": _assert_determinism,
    "div": _assert_divergence,
}


# One test case's full parameter set, bundled so the case-driven runner takes a
# single argument instead of a long positional list.
Case = namedtuple("Case", "shape mean std seed dtype kind label")


def _exec(case):
    _DISPATCH[case.kind](case.shape, case.mean, case.std, case.seed, case.dtype)


# ---------------------------------------------------------------------------
# Case matrices. Octuple: (shape, mean, std, seed, dtype, kind, label).
# Labels avoid the substrings "small"/"large" (pytest -k collision).
# Params kept moderate so fp32 exp() stays finite.
# ---------------------------------------------------------------------------

# Fast set: minimal coverage.
#   - dtype fp32/fp16 ~1:1
#   - log-space moment check on a moderate-N shape per dtype
#   - vec_width boundaries (fp32=64, fp16=128): -1 / exact / +1
#   - positivity + 3D shape passthrough
#   - determinism for fp32 and fp16, plus divergence for fp32
CASES_SMALL = [
    ((1 << 16,), 0.0, 1.0, 11, torch.float32, "logm", "logm_m0_f32"),
    ((1 << 16,), 0.5, 0.5, 12, torch.float16, "logm", "logm_m05_f16"),
    ((63,), 0.0, 1.0, 13, torch.float32, "fin", "vecm1_f32"),
    ((64,), 0.0, 1.0, 14, torch.float32, "fin", "vec_f32"),
    ((128,), 0.0, 0.5, 15, torch.float16, "fin", "vec_f16"),
    ((129,), 0.0, 0.5, 16, torch.float16, "fin", "vecp1_f16"),
    ((4, 16, 32), 0.5, 1.0, 17, torch.float32, "pos", "pos3d_f32"),
    ((1 << 16,), 0.0, 1.0, 18, torch.float32, "det", "det_f32"),
    ((1 << 16,), 0.0, 0.5, 19, torch.float16, "det", "det_f16"),
    ((1 << 16,), 0.0, 1.0, 20, torch.float32, "div", "div_f32"),
]

# Full set: large-N log moments, >uint16 count boundary, positivity,
# multi-dim, fp16 large-N, determinism across the count split.
CASES_LARGE = [
    ((1 << 20,), 0.0, 1.0, 31, torch.float32, "logm", "logmbig_m0_f32"),
    ((1 << 20,), 0.5, 1.0, 32, torch.float32, "logm", "logmbig_m05_f32"),
    ((1 << 20,), 0.0, 0.5, 33, torch.float16, "logm", "logmbig_f16"),
    ((70000,), 0.0, 1.0, 34, torch.float32, "logm", "count_gt_u16_f32"),
    ((1 << 20,), 0.5, 1.0, 35, torch.float32, "pos", "pos_f32"),
    ((8, 128), 0.0, 1.0, 36, torch.float32, "fin", "shape2d_f32"),
    ((4096,), 0.0, 0.5, 37, torch.float16, "fin", "midn_f16"),
    ((70000,), 0.0, 1.0, 38, torch.float32, "det", "detbig_f32"),
    ((1 << 20,), 0.0, 1.0, 39, torch.float32, "div", "divbig_f32"),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "case", [Case(*c) for c in CASES_SMALL],
    ids=[c[-1] for c in CASES_SMALL])


def test_log_normal_small(case):
    _exec(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "case", [Case(*c) for c in CASES_LARGE],
    ids=[c[-1] for c in CASES_LARGE])


def test_log_normal_large(case):
    _exec(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("bad_std", [0.0, -1.0])
def test_log_normal_invalid_std(bad_std):
    """std <= 0 is rejected on host (matches PyTorch log_normal_impl_)."""
    x = torch.empty(16, dtype=torch.float32).npu()
    with pytest.raises(RuntimeError, match=r"std > 0.0"):
        torch.ops.ops_multimodal_fusion.log_normal(x, 0.0, bad_std, 0)
