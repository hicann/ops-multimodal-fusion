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

"""Tests for ops_multimodal_fusion.gamma.

The input concentration enters as alpha and each output is a Gamma sample at
unit rate, drawn by Marsaglia-Tsang acceptance and rejection on an AscendC
PhiloxRandom uniform stream with a Box-Muller normal and a boost branch when
the concentration is below one.

The NPU Philox counter to element mapping differs from PyTorch's curand, so
bit-exact comparison is impossible. The Gamma distribution has finite
light-tailed moments equal to the concentration, so acceptance is moment
based and structural:
  - the sample mean approximates the concentration
  - the sample variance approximates the concentration
  - every sample is positive, clamped above the smallest normal, matching
    PyTorch
  - the boost branch below one and the main branch at or above one are both
    covered
  - a zero concentration produces a near-zero clamped sample
  - the same seed, shape and input give an identical output while a
    different seed diverges
  - shape and dtype are preserved and negative or not-a-number inputs are
    rejected on the host

Marsaglia-Tsang is exact for all positive concentrations and the SIMD form
caps rejection steps, leaving negligible truncation.

Test case counts:
  - test_gamma_small            : 11 cases
  - test_gamma_large            :  9 cases
  - test_gamma_interface_exist  :  1 case
  - test_gamma_alpha_zero       :  2 cases
  - test_gamma_invalid          :  2 cases
  - Total                       : 25 cases
"""

import torch
import torch_npu  # noqa: F401  (registers NPU dispatch key)
import pytest
import ops_multimodal_fusion  # noqa: F401  (registers torch.ops.ops_multimodal_fusion.gamma)

if not hasattr(torch.ops.ops_multimodal_fusion, "gamma"):
    pytest.skip(
        "ops_multimodal_fusion.gamma not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_gamma_interface_exist():
    """The 'ops_multimodal_fusion.gamma' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "gamma"),\
        "The 'gamma' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _run(shape, a, seed, dtype):
    """Sample with a constant concentration tensor; return output on CPU."""
    alpha = torch.full(shape, float(a), dtype=dtype).npu()
    out = torch.ops.ops_multimodal_fusion.gamma(alpha, int(seed))
    assert out.shape == alpha.shape, f"shape mismatch: {out.shape} vs {alpha.shape}"
    assert out.dtype == dtype, f"dtype mismatch: {out.dtype} vs {dtype}"
    return out.cpu()


def _assert_moments(shape, a, seed, dtype):
    """The sample mean and variance both approximate the concentration."""
    raw = _run(shape, a, seed, dtype)
    s = raw.to(torch.float64).flatten()
    fin = s[torch.isfinite(s) & (s > 0)]
    n = s.numel()
    assert fin.numel() >= 0.98 * n, f"too many non-finite/non-positive: {n - fin.numel()}/{n}"
    mean = fin.mean().item()
    var = fin.var(unbiased=True).item()

    tight = n >= (1 << 20)
    if dtype == torch.float32:
        m_rel = 0.03 if tight else 0.06
        v_rel = 0.06 if tight else 0.18
    else:  # fp16
        m_rel, v_rel = 0.10, 0.30

    assert abs(mean - a) / a < m_rel,\
        f"mean {mean:.5f} off alpha {a} (rel tol {m_rel})"
    assert abs(var - a) / a < v_rel,\
        f"var {var:.5f} off alpha {a} (rel tol {v_rel})"


def _assert_positive(shape, a, seed, dtype):
    """In single precision every sample is finite and positive; in half
    precision at least most are.
    """
    raw = _run(shape, a, seed, dtype)
    v = raw.to(torch.float64)
    good = torch.isfinite(v) & (v > 0.0)
    if dtype == torch.float32:
        assert good.all(), "fp32 gamma output must be finite and > 0"
    else:
        assert good.float().mean().item() >= 0.98, "fp16 gamma output mostly finite and > 0"


def _assert_shape_finite(shape, a, seed, dtype):
    _assert_positive(shape, a, seed, dtype)


def _assert_determinism(shape, a, seed, dtype):
    """The same seed, shape and input give a byte-identical output."""
    x = _run(shape, a, seed, dtype)
    y = _run(shape, a, seed, dtype)
    assert torch.equal(x, y), "same seed must give identical samples"


def _assert_divergence(shape, a, seed, dtype):
    """A different seed gives a substantially different output."""
    x = _run(shape, a, seed, dtype)
    y = _run(shape, a, seed + 1, dtype)
    assert not torch.equal(x, y), "different seeds must give different samples"
    diff = (x.to(torch.float64) != y.to(torch.float64)).float().mean().item()
    assert diff > 0.5, f"streams too correlated across seeds (diff frac {diff:.3f})"


_DISPATCH = {
    "mom": _assert_moments,
    "pos": _assert_positive,
    "fin": _assert_shape_finite,
    "det": _assert_determinism,
    "div": _assert_divergence,
}


def _exec(case):
    shape, a, seed, dtype, kind, label = case
    _DISPATCH[kind](shape, a, seed, dtype)


# ---------------------------------------------------------------------------
# Case matrices. Each entry carries shape, concentration, seed, dtype, kind
# and label. Labels avoid the substrings "small" and "large" to dodge a
# pytest filter collision. Element counts stay modest in the fast set since
# each element runs the rejection loop. A concentration below one exercises
# the boost branch and one or more the main branch.
# ---------------------------------------------------------------------------

CASES_SMALL = [
    ((4096,), 2.0, 11, torch.float32, "mom", "mom_a2_f32"),
    ((4096,), 0.5, 12, torch.float32, "mom", "mom_boost_f32"),
    ((4096,), 2.0, 13, torch.float16, "mom", "mom_a2_f16"),
    ((4096,), 0.7, 14, torch.float16, "mom", "mom_boost_f16"),
    ((63,), 2.0, 15, torch.float32, "pos", "vecm1_f32"),
    ((64,), 2.0, 16, torch.float32, "pos", "vec_f32"),
    ((128,), 2.0, 17, torch.float16, "pos", "vec_f16"),
    ((129,), 2.0, 18, torch.float16, "pos", "vecp1_f16"),
    ((4, 16, 32), 1.5, 19, torch.float32, "pos", "pos3d_f32"),
    ((4096,), 2.0, 20, torch.float32, "det", "det_f32"),
    ((4096,), 2.0, 21, torch.float32, "div", "div_f32"),
]

CASES_LARGE = [
    ((1 << 20,), 2.0, 31, torch.float32, "mom", "mombig_a2_f32"),
    ((1 << 20,), 0.5, 32, torch.float32, "mom", "mombig_boost_f32"),
    ((1 << 20,), 5.0, 33, torch.float32, "mom", "mombig_a5_f32"),
    ((1 << 20,), 2.0, 34, torch.float16, "mom", "mombig_f16"),
    ((70000,), 2.0, 35, torch.float32, "mom", "count_gt_u16_f32"),
    ((1 << 20,), 2.0, 36, torch.float32, "pos", "pos_f32"),
    ((8, 128), 1.5, 37, torch.float32, "fin", "shape2d_f32"),
    ((70000,), 2.0, 38, torch.float32, "det", "detbig_f32"),
    ((1 << 20,), 2.0, 39, torch.float32, "div", "divbig_f32"),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "case", CASES_SMALL,
    ids=[c[-1] for c in CASES_SMALL])


def test_gamma_small(case):
    _exec(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "case", CASES_LARGE,
    ids=[c[-1] for c in CASES_LARGE])


def test_gamma_large(case):
    _exec(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_gamma_alpha_zero(dtype):
    """A zero concentration yields a near-zero sample clamped above the
    smallest normal.
    """
    raw = _run((4096,), 0.0, 7, dtype).to(torch.float64)
    assert (raw.abs() < 1e-30).all(), "alpha=0 must produce ~0 (<= FLT_MIN)"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("bad", ["neg", "nan"])
def test_gamma_invalid(bad):
    """A negative or not-a-number concentration is rejected on the host."""
    a = torch.full((16,), 2.0, dtype=torch.float32)
    a[3] = -1.0 if bad == "neg" else float("nan")
    with pytest.raises(RuntimeError, match="non-negative"):
        torch.ops.ops_multimodal_fusion.gamma(a.npu(), 0)
