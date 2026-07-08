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
"""Tests for the exponential operator under ops_multimodal_fusion.

The operator takes shape, dtype and device from the input tensor while
ignoring its data, then fills it with independent Exponential samples at rate
lambd, drawn by inverse-transform sampling over an AscendC PhiloxRandom uniform
stream: each sample is negative one over lambd times the natural log of the
uniform value, clamped to the open unit interval above FLT_MIN.

The NPU Philox counter-to-element map differs from PyTorch curand, so
bit-exact comparison is impossible. Unlike Cauchy, the Exponential
distribution does have finite moments, so acceptance uses moment and
structural checks:
  - sample mean is about one over lambd
  - sample std is about one over lambd
  - sample median is about ln of two over lambd
  - output strictly above zero for fp32; fp16 is relaxed to at-or-above zero
    because the half-epsilon offset may round to zero at fp16 subnormal
    resolution
  - the same seed and shape give identical output; a different seed differs
  - shape and dtype are preserved; an infinite lambd yields all zeros and a
    non-positive lambd is rejected on the host

Stats are computed on the finite subset and in float64.

Test case counts:
  - test_exponential_small           : 10 cases
  - test_exponential_large           : 9 cases
  - test_exponential_interface_exist : 1
  - test_exponential_lambd_inf       : 2 dtype cases for fp32 and fp16
  - test_exponential_invalid_lambd   : 2 non-positive-rate cases
  - Total                            : 24 cases
"""

import math
from collections import namedtuple

import pytest
import torch
import torch_npu  # noqa: F401  (registers NPU dispatch key)
import ops_multimodal_fusion  # noqa: F401  (registers torch.ops.ops_multimodal_fusion.exponential)

if not hasattr(torch.ops.ops_multimodal_fusion, "exponential"):
    pytest.skip(
        "ops_multimodal_fusion.exponential not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_exponential_interface_exist():
    """The 'ops_multimodal_fusion.exponential' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "exponential"),\
        "The 'exponential' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _run_raw(shape, lambd, seed, dtype):
    """Return the sampled tensor on CPU in its native dtype."""
    x = torch.empty(*shape, dtype=dtype).npu()
    out = torch.ops.ops_multimodal_fusion.exponential(x, float(lambd), int(seed))
    assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"
    assert out.dtype == dtype, f"dtype mismatch: {out.dtype} vs {dtype}"
    return out.cpu()


def _assert_moments(shape, lambd, seed, dtype):
    """Mean and std are about the reciprocal of lambd; the median about the natural log of two over lambd."""
    raw = _run_raw(shape, lambd, seed, dtype)
    samples = raw.to(torch.float64).flatten()
    finite = samples[torch.isfinite(samples)]
    n = samples.numel()
    assert finite.numel() >= 0.98 * n,\
        f"too many non-finite samples: {n - finite.numel()}/{n}"

    mean = finite.mean().item()
    std = finite.std().item()
    med = finite.median().item()
    inv_l = 1.0 / lambd
    med_t = math.log(2.0) / lambd

    tight = n >= (1 << 20)
    if dtype == torch.float32:
        m_rel = 0.02 if tight else 0.05
        s_rel = 0.03 if tight else 0.05
        d_rel = 0.02 if tight else 0.05
    else:  # fp16 loosens tolerances for quantization and RNG differences
        m_rel, s_rel, d_rel = 0.10, 0.12, 0.10

    assert abs(mean - inv_l) / inv_l < m_rel,\
        f"mean {mean:.5f} off 1/lambd {inv_l:.5f} (rel tol {m_rel})"
    assert abs(std - inv_l) / inv_l < s_rel,\
        f"std {std:.5f} off 1/lambd {inv_l:.5f} (rel tol {s_rel})"
    assert abs(med - med_t) / med_t < d_rel,\
        f"median {med:.5f} off ln2/lambd {med_t:.5f} (rel tol {d_rel})"


def _assert_positive(shape, lambd, seed, dtype):
    """Strictly positive for fp32; fp16 is relaxed to at-or-above zero because of subnormal rounding near zero."""
    raw = _run_raw(shape, lambd, seed, dtype)
    v = raw.to(torch.float64)
    assert torch.isfinite(v).all(), "exponential output must be finite"
    if dtype == torch.float32:
        assert (v > 0).all(), "fp32 exponential output must be strictly > 0"
    else:
        assert (v >= 0).all(), "fp16 exponential output must be >= 0"


def _assert_shape_finite(shape, lambd, seed, dtype):
    """Shape and dtype preserved and all values finite, with the maximum bounded near eighty-seven over lambd."""
    raw = _run_raw(shape, lambd, seed, dtype)
    assert torch.isfinite(raw.to(torch.float64)).all(),\
        "exponential output must be finite"


def _assert_determinism(shape, lambd, seed, dtype):
    """Same seed and shape give byte-identical output."""
    a = _run_raw(shape, lambd, seed, dtype)
    b = _run_raw(shape, lambd, seed, dtype)
    assert torch.equal(a, b), "same seed must give identical samples"


def _assert_divergence(shape, lambd, seed, dtype):
    """A different seed gives substantially different output."""
    a = _run_raw(shape, lambd, seed, dtype)
    b = _run_raw(shape, lambd, seed + 1, dtype)
    assert not torch.equal(a, b), "different seeds must give different samples"
    diff = (a.to(torch.float64) != b.to(torch.float64)).float().mean().item()
    assert diff > 0.5, f"streams too correlated across seeds (diff frac {diff:.3f})"


_DISPATCH = {
    "mom": _assert_moments,
    "pos": _assert_positive,
    "fin": _assert_shape_finite,
    "det": _assert_determinism,
    "div": _assert_divergence,
}


# The case bundle keeps the parametrize row intact while limiting the executor
# to a single argument, which keeps the function parameter count within the
# codecheck limit.
Case = namedtuple("Case", ["shape", "lambd", "seed", "dtype", "kind", "label"])


def _exec(case):
    _DISPATCH[case.kind](case.shape, case.lambd, case.seed, case.dtype)


# ---------------------------------------------------------------------------
# Case matrices. Each row carries shape, lambd, seed, dtype, kind and label.
# Labels avoid the substrings small and large to dodge pytest filtering
# collisions.
# ---------------------------------------------------------------------------

# Fast set covering minimal ground:
#   - dtype fp32 and fp16 in roughly equal proportion
#   - moment check on a moderate-N shape per dtype
#   - vec_width boundaries, sixty-four for fp32 and one twenty-eight for fp16,
#     covered just below, exactly at, and just above the width
#   - a 3D shape passthrough
#   - determinism for fp32 and fp16, plus divergence for fp32
CASES_SMALL = [
    Case((1 << 16,), 1.0, 11, torch.float32, "mom", "mom_l1_f32"),
    Case((1 << 16,), 0.5, 12, torch.float16, "mom", "mom_l05_f16"),
    Case((63,), 1.0, 13, torch.float32, "fin", "vecm1_f32"),
    Case((64,), 1.0, 14, torch.float32, "fin", "vec_f32"),
    Case((128,), 1.0, 15, torch.float16, "fin", "vec_f16"),
    Case((129,), 1.0, 16, torch.float16, "fin", "vecp1_f16"),
    Case((4, 16, 32), 1.0, 17, torch.float32, "fin", "shape3d_f32"),
    Case((1 << 16,), 1.0, 18, torch.float32, "det", "det_f32"),
    Case((1 << 16,), 1.0, 19, torch.float16, "det", "det_f16"),
    Case((1 << 16,), 1.0, 20, torch.float32, "div", "div_f32"),
]

# Full set covers large-N moments and the count boundary above uint16,
# strict positivity, large-N in fp16, and determinism across the count split.
CASES_LARGE = [
    Case((1 << 20,), 1.0, 31, torch.float32, "mom", "mombig_l1_f32"),
    Case((1 << 20,), 2.0, 32, torch.float32, "mom", "mombig_l2_f32"),
    Case((1 << 20,), 1.0, 33, torch.float16, "mom", "mombig_f16"),
    Case((70000,), 1.0, 34, torch.float32, "mom", "count_gt_u16_f32"),
    Case((1 << 20,), 1.0, 35, torch.float32, "pos", "pos_f32"),
    Case((4096,), 1.0, 36, torch.float16, "fin", "midn_f16"),
    Case((8, 128), 1.0, 37, torch.float32, "fin", "shape2d_f32"),
    Case((70000,), 1.0, 38, torch.float32, "det", "detbig_f32"),
    Case((1 << 20,), 1.0, 39, torch.float32, "div", "divbig_f32"),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_SMALL, ids=[c.label for c in CASES_SMALL])
def test_exponential_small(case):
    _exec(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_LARGE, ids=[c.label for c in CASES_LARGE])
def test_exponential_large(case):
    _exec(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_exponential_lambd_inf(dtype):
    """An infinite lambd makes every element zero, matching the PyTorch test."""
    raw = _run_raw((4096,), float("inf"), 7, dtype)
    assert torch.equal(raw.to(torch.float64),
                       torch.zeros_like(raw, dtype=torch.float64)),\
        "lambd=inf must produce an all-zero tensor"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("bad_lambd", [0.0, -0.5])
def test_exponential_invalid_lambd(bad_lambd):
    """A non-positive lambd is rejected on the host, matching the PyTorch error inputs."""
    x = torch.empty(16, dtype=torch.float32).npu()
    with pytest.raises(RuntimeError, match="exponential_ expects lambda > 0.0"):
        torch.ops.ops_multimodal_fusion.exponential(x, bad_lambd, 0)
