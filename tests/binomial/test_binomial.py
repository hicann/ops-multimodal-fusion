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

"""Tests for ops_multimodal_fusion.binomial.

Both inputs carry per-element parameters that are read together: a count
of trials and a probability of success per element. Each output sample is
drawn from a Binomial distribution using the sum of Bernoulli identity over
the trials, evaluated against an AscendC PhiloxRandom uniform stream.

The NPU Philox counter to element map differs from the PyTorch curand map,
so a bit exact comparison is impossible. Binomial has finite, light tailed
moments, so acceptance is based on moments plus structure:
  - sample mean approaches the count times the probability
  - sample var approaches the count times the probability times its complement
  - every sample is an integer between zero and the count
  - probability zero gives all zeros; probability one gives all counts;
    a zero count gives all zeros
  - same seed, shape and inputs give an identical output, a different seed
    diverges
  - shape and dtype are preserved; an invalid count or prob is rejected on host

The sum of Bernoulli form is exact for a non-negative integer count and a
probability within the unit interval. The SIMD form caps trials at the inner
maximum, enforced on host without truncation.

Test case counts:
  - test_binomial_small            : 10 cases
  - test_binomial_large            :  9 cases
  - test_binomial_interface_exist  :  1
  - test_binomial_pq_edges         :  3 edges for prob and count
  - test_binomial_invalid          :  6
  - Total                          : 29 cases
"""

import torch
import torch_npu  # noqa: F401  (registers NPU dispatch key)
import pytest
import ops_multimodal_fusion  # noqa: F401  (registers torch.ops.ops_multimodal_fusion.binomial)

if not hasattr(torch.ops.ops_multimodal_fusion, "binomial"):
    pytest.skip(
        "ops_multimodal_fusion.binomial not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_binomial_interface_exist():
    """The 'ops_multimodal_fusion.binomial' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "binomial"),\
        "The 'binomial' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _run(shape, n, p, seed, dtype):
    """Sample with constant n p parameter tensors; return output on CPU."""
    count = torch.full(shape, float(n), dtype=dtype).npu()
    prob = torch.full(shape, float(p), dtype=dtype).npu()
    out = torch.ops.ops_multimodal_fusion.binomial(count, prob, int(seed))
    assert out.shape == count.shape, f"shape mismatch: {out.shape} vs {count.shape}"
    assert out.dtype == dtype, f"dtype mismatch: {out.dtype} vs {dtype}"
    return out.cpu()


def _assert_moments(shape, n, p, seed, dtype):
    """Check sample mean against n times p and sample var against n p variance."""
    raw = _run(shape, n, p, seed, dtype)
    s = raw.to(torch.float64).flatten()
    assert torch.isfinite(s).all(), "binomial output must be finite"
    cnt = s.numel()
    mean = s.mean().item()
    var = s.var(unbiased=True).item()
    mt = n * p
    vt = n * p * (1.0 - p)

    tight = cnt >= (1 << 20)
    if dtype == torch.float32:
        m_rel = 0.03 if tight else 0.06
        v_rel = 0.06 if tight else 0.15
    else:  # fp16
        m_rel, v_rel = 0.10, 0.25

    assert abs(mean - mt) / mt < m_rel,\
        f"mean {mean:.5f} off n*p {mt} (rel tol {m_rel})"
    assert abs(var - vt) / vt < v_rel,\
        f"var {var:.5f} off n*p*(1-p) {vt} (rel tol {v_rel})"


def _assert_integral(shape, n, p, seed, dtype):
    """Every sample is a finite integer in 0 n ."""
    raw = _run(shape, n, p, seed, dtype)
    v = raw.to(torch.float64)
    assert torch.isfinite(v).all(), "binomial output must be finite"
    assert (v >= 0).all() and (v <= n).all(),\
        f"binomial support is {{0,...,{n}}}"
    is_int = (v == v.round())
    if dtype == torch.float32:
        assert is_int.all(), "fp32 binomial output must be integer-valued"
    else:
        assert is_int.float().mean().item() >= 0.99,\
            "fp16 binomial output must be ~all integer-valued"


def _assert_shape_finite(shape, n, p, seed, dtype):
    _assert_integral(shape, n, p, seed, dtype)


def _assert_determinism(shape, n, p, seed, dtype):
    """Same seed, shape and inputs give a byte identical output."""
    a = _run(shape, n, p, seed, dtype)
    b = _run(shape, n, p, seed, dtype)
    assert torch.equal(a, b), "same seed must give identical samples"


def _assert_divergence(shape, n, p, seed, dtype):
    """Different seed to substantially different output."""
    a = _run(shape, n, p, seed, dtype)
    b = _run(shape, n, p, seed + 1, dtype)
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


def _exec(case):
    shape, n, p, seed, dtype, kind, label = case
    _DISPATCH[kind](shape, n, p, seed, dtype)


# ---------------------------------------------------------------------------
# Case matrices. Each tuple holds shape, n, p, seed, dtype, kind, label.
# Labels avoid the substrings "small"/"large" to dodge pytest -k collision.
# N kept modest in the fast set since each trial is an inner Bernoulli step.
# ---------------------------------------------------------------------------

CASES_SMALL = [
    ((8192,), 20, 0.3, 11, torch.float32, "mom", "mom_a_f32"),
    ((8192,), 50, 0.5, 12, torch.float16, "mom", "mom_b_f16"),
    ((63,), 10, 0.5, 13, torch.float32, "int", "vecm1_f32"),
    ((64,), 10, 0.5, 14, torch.float32, "int", "vec_f32"),
    ((128,), 10, 0.5, 15, torch.float16, "int", "vec_f16"),
    ((129,), 10, 0.5, 16, torch.float16, "int", "vecp1_f16"),
    ((4, 16, 32), 8, 0.4, 17, torch.float32, "int", "int3d_f32"),
    ((8192,), 20, 0.3, 18, torch.float32, "det", "det_f32"),
    ((8192,), 50, 0.5, 19, torch.float16, "det", "det_f16"),
    ((8192,), 20, 0.3, 20, torch.float32, "div", "div_f32"),
]

CASES_LARGE = [
    ((1 << 20,), 20, 0.3, 31, torch.float32, "mom", "mombig_a_f32"),
    ((1 << 20,), 50, 0.5, 32, torch.float32, "mom", "mombig_b_f32"),
    ((1 << 20,), 30, 0.5, 33, torch.float16, "mom", "mombig_f16"),
    ((70000,), 20, 0.3, 34, torch.float32, "mom", "count_gt_u16_f32"),
    ((1 << 20,), 40, 0.6, 35, torch.float32, "int", "intbig_f32"),
    ((8, 128), 16, 0.5, 36, torch.float32, "fin", "shape2d_f32"),
    ((4096,), 24, 0.5, 37, torch.float16, "fin", "midn_f16"),
    ((70000,), 20, 0.3, 38, torch.float32, "det", "detbig_f32"),
    ((1 << 20,), 20, 0.3, 39, torch.float32, "div", "divbig_f32"),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "case", CASES_SMALL,
    ids=[c[-1] for c in CASES_SMALL])


def test_binomial_small(case):
    _exec(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "case", CASES_LARGE,
    ids=[c[-1] for c in CASES_LARGE])


def test_binomial_large(case):
    _exec(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("edge", ["p0", "p1", "n0"])
def test_binomial_pq_edges(edge):
    """p 0 to all 0; p 1 to all n; n 0 to all 0 matches PyTorch edges ."""
    n = 0 if edge == "n0" else 16
    p = {"p0": 0.0, "p1": 1.0, "n0": 0.5}[edge]
    raw = _run((4096,), n, p, 7, torch.float32).to(torch.float64)
    expected = float(n) if edge == "p1" else 0.0
    assert torch.equal(raw, torch.full_like(raw, expected)),\
        f"edge {edge}: expected all {expected}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "bad", ["prob_neg", "prob_gt1", "prob_nan", "count_neg", "count_frac", "shape_mismatch"])


def test_binomial_invalid(bad):
    """Invalid count or prob is rejected on host."""
    n = torch.full((16,), 10.0, dtype=torch.float32)
    p = torch.full((16,), 0.5, dtype=torch.float32)
    pat = "prob to be in"
    if bad == "prob_neg":
        p[2] = -0.1
    elif bad == "prob_gt1":
        p[2] = 1.5
    elif bad == "prob_nan":
        p[2] = float("nan")
    elif bad == "count_neg":
        n[2] = -1.0
        pat = "non-negative integer"
    elif bad == "count_frac":
        n[2] = 3.5
        pat = "non-negative integer"
    else:  # shape_mismatch
        p = torch.full((8,), 0.5, dtype=torch.float32)
        pat = "same shape"
    with pytest.raises(RuntimeError, match=pat):
        torch.ops.ops_multimodal_fusion.binomial(n.npu(), p.npu(), 0)
