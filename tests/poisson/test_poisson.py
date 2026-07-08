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
"""Tests for the poisson operator under ops_multimodal_fusion.

Unlike the other distribution ops, the input tensor itself carries the
per-element rate lambda and its data is read; each output element is a Poisson
draw at the matching rate via Knuth's product algorithm on an AscendC
PhiloxRandom uniform stream.

The NPU Philox counter-to-element map differs from PyTorch curand, so
bit-exact comparison is impossible. Poisson has finite, light-tailed moments,
with mean and variance both equal to the rate, so acceptance is moment-based
plus structural:
  - sample mean approximates the rate
  - sample variance approximates the rate
  - every sample is a non-negative integer
  - a rate of zero yields an output of exactly zero
  - the same seed, shape and input give identical output, while a different
    seed differs
  - shape and dtype are preserved; a negative or NaN rate is rejected on host

Knuth is exact for any non-negative rate; the SIMD form caps the inner
iteration count, a negligible truncation for the modest rates tested here.

Test case counts:
  - test_poisson_small            : 10 cases
  - test_poisson_large            :  9 cases
  - test_poisson_interface_exist  :  1
  - test_poisson_zero_rate        :  2 dtype cases for fp32 and fp16
  - test_poisson_invalid_rate     :  2 cases for negative and NaN rates
  - Total                         : 24 cases
"""

from collections import namedtuple

import pytest
import torch
import torch_npu  # noqa: F401  (registers NPU dispatch key)
import ops_multimodal_fusion  # noqa: F401  (registers torch.ops.ops_multimodal_fusion.poisson)

if not hasattr(torch.ops.ops_multimodal_fusion, "poisson"):
    pytest.skip(
        "ops_multimodal_fusion.poisson not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_poisson_interface_exist():
    """The 'ops_multimodal_fusion.poisson' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "poisson"),\
        "The 'poisson' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _run_rate(shape, lam, seed, dtype):
    """Sample with a constant-rate lambda tensor; return output on CPU."""
    x = torch.full(shape, float(lam), dtype=dtype).npu()
    out = torch.ops.ops_multimodal_fusion.poisson(x, int(seed))
    assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"
    assert out.dtype == dtype, f"dtype mismatch: {out.dtype} vs {dtype}"
    return out.cpu()


def _assert_moments(shape, lam, seed, dtype):
    """Sample mean approximates lambda and sample variance approximates lambda, since both equal lambda."""
    raw = _run_rate(shape, lam, seed, dtype)
    s = raw.to(torch.float64).flatten()
    assert torch.isfinite(s).all(), "poisson output must be finite"
    n = s.numel()
    mean = s.mean().item()
    var = s.var(unbiased=True).item()

    tight = n >= (1 << 20)
    if dtype == torch.float32:
        m_rel = 0.03 if tight else 0.06
        v_rel = 0.06 if tight else 0.15
    else:  # fp16
        m_rel = 0.10
        v_rel = 0.25

    assert abs(mean - lam) / lam < m_rel,\
        f"mean {mean:.5f} off lambda {lam} (rel tol {m_rel})"
    assert abs(var - lam) / lam < v_rel,\
        f"var {var:.5f} off lambda {lam} (rel tol {v_rel})"


def _assert_integral(shape, lam, seed, dtype):
    """Every sample is a finite, non-negative integer."""
    raw = _run_rate(shape, lam, seed, dtype)
    v = raw.to(torch.float64)
    assert torch.isfinite(v).all(), "poisson output must be finite"
    assert (v >= 0).all(), "poisson support is {0,1,2,...} (>= 0)"
    is_int = (v == v.round())
    if dtype == torch.float32:
        assert is_int.all(), "fp32 poisson output must be integer-valued"
    else:
        assert is_int.float().mean().item() >= 0.99,\
            "fp16 poisson output must be ~all integer-valued"


def _assert_shape_finite(shape, lam, seed, dtype):
    """Shape/dtype preserved; finite, non-negative integers."""
    _assert_integral(shape, lam, seed, dtype)


def _assert_determinism(shape, lam, seed, dtype):
    """Same seed, shape and input give byte-identical output."""
    a = _run_rate(shape, lam, seed, dtype)
    b = _run_rate(shape, lam, seed, dtype)
    assert torch.equal(a, b), "same seed must give identical samples"


def _assert_divergence(shape, lam, seed, dtype):
    """A different seed gives substantially different output."""
    a = _run_rate(shape, lam, seed, dtype)
    b = _run_rate(shape, lam, seed + 1, dtype)
    assert not torch.equal(a, b), "different seeds must give different samples"
    diff = (a.to(torch.float64) != b.to(torch.float64)).float().mean().item()
    assert diff > 0.3, f"streams too correlated across seeds (diff frac {diff:.3f})"


_DISPATCH = {
    "mom": _assert_moments,
    "int": _assert_integral,
    "fin": _assert_shape_finite,
    "det": _assert_determinism,
    "div": _assert_divergence,
}


# The case bundle keeps the parametrize row intact while limiting the executor
# to a single argument, which keeps the function parameter count within the
# codecheck limit.
Case = namedtuple("Case", ["shape", "lam", "seed", "dtype", "kind", "label"])


def _exec(case):
    _DISPATCH[case.kind](case.shape, case.lam, case.seed, case.dtype)


# ---------------------------------------------------------------------------
# Case matrices. Each row carries shape, rate, seed, dtype, kind and label.
# Labels avoid the substrings small and large to dodge pytest filtering
# collisions. The element count stays modest in the fast set, since Knuth runs
# its capped inner iteration count per tile.
# ---------------------------------------------------------------------------

CASES_SMALL = [
    Case((8192,), 2.0, 11, torch.float32, "mom", "mom_l2_f32"),
    Case((8192,), 5.0, 12, torch.float16, "mom", "mom_l5_f16"),
    Case((63,), 3.0, 13, torch.float32, "int", "vecm1_f32"),
    Case((64,), 3.0, 14, torch.float32, "int", "vec_f32"),
    Case((128,), 3.0, 15, torch.float16, "int", "vec_f16"),
    Case((129,), 3.0, 16, torch.float16, "int", "vecp1_f16"),
    Case((4, 16, 32), 4.0, 17, torch.float32, "int", "int3d_f32"),
    Case((8192,), 2.0, 18, torch.float32, "det", "det_f32"),
    Case((8192,), 5.0, 19, torch.float16, "det", "det_f16"),
    Case((8192,), 2.0, 20, torch.float32, "div", "div_f32"),
]

CASES_LARGE = [
    Case((1 << 20,), 2.0, 31, torch.float32, "mom", "mombig_l2_f32"),
    Case((1 << 20,), 10.0, 32, torch.float32, "mom", "mombig_l10_f32"),
    Case((1 << 20,), 5.0, 33, torch.float16, "mom", "mombig_f16"),
    Case((70000,), 3.0, 34, torch.float32, "mom", "count_gt_u16_f32"),
    Case((1 << 20,), 5.0, 35, torch.float32, "int", "intbig_f32"),
    Case((8, 128), 4.0, 36, torch.float32, "fin", "shape2d_f32"),
    Case((4096,), 5.0, 37, torch.float16, "fin", "midn_f16"),
    Case((70000,), 3.0, 38, torch.float32, "det", "detbig_f32"),
    Case((1 << 20,), 2.0, 39, torch.float32, "div", "divbig_f32"),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_SMALL, ids=[c.label for c in CASES_SMALL])
def test_poisson_small(case):
    _exec(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_LARGE, ids=[c.label for c in CASES_LARGE])
def test_poisson_large(case):
    _exec(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_poisson_zero_rate(dtype):
    """A lambda of zero makes every element exactly zero, matching PyTorch."""
    x = torch.zeros(4096, dtype=dtype).npu()
    out = torch.ops.ops_multimodal_fusion.poisson(x, 7).cpu().to(torch.float64)
    assert torch.equal(out, torch.zeros_like(out)),\
        "lambda=0 must produce an all-zero tensor"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("bad", ["neg", "nan"])
def test_poisson_invalid_rate(bad):
    """Negative or NaN rate is rejected on host (matches PyTorch assert)."""
    x = torch.full((16,), 1.0, dtype=torch.float32)
    x[3] = -1.0 if bad == "neg" else float("nan")
    x = x.npu()
    with pytest.raises(RuntimeError, match="non-negative"):
        torch.ops.ops_multimodal_fusion.poisson(x, 0)
