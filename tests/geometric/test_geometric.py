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
"""Tests for the geometric operator under ops_multimodal_fusion.

The operator takes shape, dtype and device from the input tensor while
ignoring its data, then fills it with independent Geometric samples on the
positive integers. The unit-probability is the success probability and the
probability of value k falls off geometrically in k. Sampling is by
inverse transform over an AscendC PhiloxRandom uniform stream: each sample is
the ceiling of the natural log of the uniform value, floored at FLT_MIN,
divided by the natural log of one minus the success probability.

The NPU Philox counter-to-element map differs from PyTorch curand, so
bit-exact comparison is impossible. The Geometric distribution has finite
moments with a light tail, so acceptance uses moment and structural checks:
  - sample mean is about the reciprocal of the success probability
  - sample std is about the square root of one minus the success probability,
    divided by the success probability
  - the empirical fraction of unit samples is about the success probability,
    using a bounded estimator
  - every sample is an integer of at least one, since the NPU uniform lies in
    the half-open unit interval and the support starts at one; the PyTorch
    curand half-open variant can in principle round down to zero with
    negligible probability
  - the same seed and shape give identical output, while a different seed
    gives a different result
  - shape and dtype are preserved; a success probability at or below zero, or
    at or above one, is rejected on the host

Stats are computed on the finite subset and in float64.

Test case counts:
  - test_geometric_small            : 10 cases
  - test_geometric_large            : 9 cases
  - test_geometric_interface_exist  : 1
  - test_geometric_invalid_p        : 4 invalid success-probability cases
  - Total                           : 24 cases
"""

import math
from collections import namedtuple

import pytest
import torch
import torch_npu  # noqa: F401  (registers NPU dispatch key)
import ops_multimodal_fusion  # noqa: F401  (registers torch.ops.ops_multimodal_fusion.geometric)

if not hasattr(torch.ops.ops_multimodal_fusion, "geometric"):
    pytest.skip(
        "ops_multimodal_fusion.geometric not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


# Each case bundles shape, success probability, seed, dtype, kind and label
# into a namedtuple so test signatures stay within the parameter-count limit.
Case = namedtuple("Case", ["shape", "p", "seed", "dtype", "kind", "label"])


def test_geometric_interface_exist():
    """The 'ops_multimodal_fusion.geometric' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "geometric"), \
        "The 'geometric' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _run_raw(shape, p, seed, dtype):
    """Return the sampled tensor on CPU in its native dtype."""
    x = torch.empty(*shape, dtype=dtype).npu()
    out = torch.ops.ops_multimodal_fusion.geometric(x, float(p), int(seed))
    assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"
    assert out.dtype == dtype, f"dtype mismatch: {out.dtype} vs {dtype}"
    return out.cpu()


def _assert_moments(shape, p, seed, dtype):
    """Mean is about the reciprocal of the success probability; std about its spread on the finite subset."""
    raw = _run_raw(shape, p, seed, dtype)
    samples = raw.to(torch.float64).flatten()
    finite = samples[torch.isfinite(samples)]
    n = samples.numel()
    assert finite.numel() >= 0.98 * n, \
        f"too many non-finite samples: {n - finite.numel()}/{n}"

    mean = finite.mean().item()
    std = finite.std().item()
    p1 = (finite == 1.0).double().mean().item()
    inv_p = 1.0 / p
    std_t = math.sqrt(1.0 - p) / p

    tight = n >= (1 << 20)
    if dtype == torch.float32:
        m_rel = 0.02 if tight else 0.05
        s_rel = 0.04 if tight else 0.08
        p1_tol = 0.01 if tight else 0.03
    else:  # fp16 loosens tolerances for quantization and RNG differences
        m_rel, s_rel, p1_tol = 0.10, 0.15, 0.05

    assert abs(mean - inv_p) / inv_p < m_rel, \
        f"mean {mean:.5f} off 1/p {inv_p:.5f} (rel tol {m_rel})"
    assert abs(std - std_t) / std_t < s_rel, \
        f"std {std:.5f} off sqrt(1-p)/p {std_t:.5f} (rel tol {s_rel})"
    assert abs(p1 - p) < p1_tol, \
        f"P(X=1) {p1:.5f} off p {p} (abs tol {p1_tol})"


def _assert_integral(shape, p, seed, dtype):
    """Every sample is finite, an integer, and at least one, since the support starts at one."""
    raw = _run_raw(shape, p, seed, dtype)
    v = raw.to(torch.float64)
    assert torch.isfinite(v).all(), "geometric output must be finite"
    assert (v >= 1.0).all(), "geometric support is {1,2,3,...} (min >= 1)"
    is_int = (v == v.round())
    if dtype == torch.float32:
        assert is_int.all(), "fp32 geometric output must be integer-valued"
    else:
        # In half precision exact integers reach only into the low thousands,
        # so a rare large tail value may quantize off an exact integer.
        assert is_int.float().mean().item() >= 0.99, \
            "fp16 geometric output must be ~all integer-valued"


def _assert_shape_finite(shape, p, seed, dtype):
    """Shape/dtype preserved and all values finite."""
    raw = _run_raw(shape, p, seed, dtype)
    v = raw.to(torch.float64)
    assert torch.isfinite(v).all(), "geometric output must be finite"
    assert (v >= 1.0).all(), "geometric support is {1,2,3,...} (min >= 1)"


def _assert_determinism(shape, p, seed, dtype):
    """Same seed and shape give byte-identical output."""
    a = _run_raw(shape, p, seed, dtype)
    b = _run_raw(shape, p, seed, dtype)
    assert torch.equal(a, b), "same seed must give identical samples"


def _assert_divergence(shape, p, seed, dtype):
    """A different seed gives substantially different output."""
    a = _run_raw(shape, p, seed, dtype)
    b = _run_raw(shape, p, seed + 1, dtype)
    assert not torch.equal(a, b), "different seeds must give different samples"
    diff = (a.to(torch.float64) != b.to(torch.float64)).float().mean().item()
    assert diff > 0.5, f"streams too correlated across seeds (diff frac {diff:.3f})"


_DISPATCH = {
    "mom": _assert_moments,
    "int": _assert_integral,
    "fin": _assert_shape_finite,
    "det": _assert_determinism,
    "div": _assert_divergence,
}


def _exec(case):
    _DISPATCH[case.kind](case.shape, case.p, case.seed, case.dtype)


# ---------------------------------------------------------------------------
# Case matrices. Each row carries shape, success probability, seed, dtype,
# kind and label. Labels avoid the substrings small and large to dodge
# pytest filtering collisions.
# ---------------------------------------------------------------------------

# Fast set covering minimal ground:
#   - fp32 and fp16 in roughly equal proportion
#   - moment check on a moderate-N shape per dtype
#   - vec_width boundaries, sixty-four for fp32 and one twenty-eight for fp16,
#     covered just below, exactly at, and just above the width
#   - integrality and a 3D shape passthrough
#   - determinism for fp32 and fp16, plus divergence for fp32
CASES_SMALL = [
    Case((1 << 16,), 0.3, 11, torch.float32, "mom", "mom_p3_f32"),
    Case((1 << 16,), 0.5, 12, torch.float16, "mom", "mom_p5_f16"),
    Case((63,), 0.5, 13, torch.float32, "fin", "vecm1_f32"),
    Case((64,), 0.5, 14, torch.float32, "fin", "vec_f32"),
    Case((128,), 0.5, 15, torch.float16, "fin", "vec_f16"),
    Case((129,), 0.5, 16, torch.float16, "fin", "vecp1_f16"),
    Case((4, 16, 32), 0.3, 17, torch.float32, "int", "int3d_f32"),
    Case((1 << 16,), 0.5, 18, torch.float32, "det", "det_f32"),
    Case((1 << 16,), 0.5, 19, torch.float16, "det", "det_f16"),
    Case((1 << 16,), 0.5, 20, torch.float32, "div", "div_f32"),
]

# Full set covering large-N moments, the count boundary above uint16,
# integrality at a low success probability where values run larger, multi-dim
# shapes, large-N in fp16, and determinism across the count split.
CASES_LARGE = [
    Case((1 << 20,), 0.3, 31, torch.float32, "mom", "mombig_p3_f32"),
    Case((1 << 20,), 0.1, 32, torch.float32, "mom", "mombig_p1_f32"),
    Case((1 << 20,), 0.5, 33, torch.float16, "mom", "mombig_f16"),
    Case((70000,), 0.3, 34, torch.float32, "mom", "count_gt_u16_f32"),
    Case((1 << 20,), 0.1, 35, torch.float32, "int", "intbig_p1_f32"),
    Case((8, 128), 0.5, 36, torch.float32, "fin", "shape2d_f32"),
    Case((4096,), 0.5, 37, torch.float16, "fin", "midn_f16"),
    Case((70000,), 0.3, 38, torch.float32, "det", "detbig_f32"),
    Case((1 << 20,), 0.3, 39, torch.float32, "div", "divbig_f32"),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_SMALL, ids=[c.label for c in CASES_SMALL])
def test_geometric_small(case):
    _exec(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_LARGE, ids=[c.label for c in CASES_LARGE])
def test_geometric_large(case):
    _exec(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("bad_p", [0.0, 1.0, -0.5, 1.5])
def test_geometric_invalid_p(bad_p):
    """A success probability outside the open unit interval is rejected on the host, matching PyTorch."""
    x = torch.empty(16, dtype=torch.float32).npu()
    with pytest.raises(RuntimeError, match=r"p to be in \(0, 1\)"):
        torch.ops.ops_multimodal_fusion.geometric(x, bad_p, 0)
