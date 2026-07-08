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

"""Tests for ops_multimodal_fusion.dirichlet.

The last dimension is the simplex axis. The concentration vector enters as
alpha and the operator returns one Dirichlet sample row whose entries are
positive and sum to one. Sampling draws each coordinate from a Gamma
distribution via Marsaglia-Tsang on an AscendC PhiloxRandom stream, with a
Box-Muller normal and a boost branch when the concentration is below one,
then normalises by the row sum and clamps to the supported range.

The NPU Philox counter to element mapping differs from PyTorch's curand, so
bit-exact comparison is impossible. Acceptance is structural and statistical
in float64:
  - simplex check: every row sums to approximately one and every element
    lies in the open unit interval
  - moments of the symmetric Dirichlet match the closed-form mean and
    variance across both the boost branch and the main branch
  - when the simplex axis has a single coordinate every element is near one
  - the same seed, shape and input give an identical result while a
    different seed diverges
  - shape and dtype are preserved and negative, not-a-number or scalar
    inputs are rejected on the host

Marsaglia-Tsang is exact for all positive concentrations and the SIMD form
caps rejection steps, leaving negligible truncation.

Test case counts:
  - test_dirichlet_small            : 12 cases
  - test_dirichlet_large            :  9 cases
  - test_dirichlet_interface_exist  :  1 case
  - test_dirichlet_k1               :  2 cases
  - test_dirichlet_invalid          :  3 cases
  - Total                           : 27 cases
"""

import torch
import torch_npu  # noqa: F401  (registers NPU dispatch key)
import pytest
import ops_multimodal_fusion  # noqa: F401  (registers torch.ops.ops_multimodal_fusion.dirichlet)

if not hasattr(torch.ops.ops_multimodal_fusion, "dirichlet"):
    pytest.skip(
        "ops_multimodal_fusion.dirichlet not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_dirichlet_interface_exist():
    """The 'ops_multimodal_fusion.dirichlet' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "dirichlet"),\
        "The 'dirichlet' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _run(shape, a, seed, dtype):
    """Sample with a constant (symmetric) concentration; return output on CPU."""
    alpha = torch.full(shape, float(a), dtype=dtype).npu()
    out = torch.ops.ops_multimodal_fusion.dirichlet(alpha, int(seed))
    assert out.shape == alpha.shape, f"shape mismatch: {out.shape} vs {alpha.shape}"
    assert out.dtype == dtype, f"dtype mismatch: {out.dtype} vs {dtype}"
    return out.cpu()


def _assert_simplex(shape, a, seed, dtype):
    """Every row sums to approximately one and every element is within the
    unit interval.
    """
    raw = _run(shape, a, seed, dtype)
    v = raw.to(torch.float64)
    assert torch.isfinite(v).all(), "dirichlet output must be finite"
    assert (v > 0.0).all(), "dirichlet support is the unit interval clamped above the minimum"
    assert (v <= 1.0 + 1e-6).all(), "dirichlet elements must not exceed one"
    k = shape[-1]
    rows = v.reshape(-1, k)
    row_sum = rows.sum(dim=-1)
    # Normalising by the row sum makes each row sum to one in exact
    # arithmetic; the two-sided clamp and the half-precision downcast are the
    # only perturbations.
    atol = 2e-3 if dtype == torch.float32 else 4e-2
    assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=atol),\
        f"row sums off 1: max dev {(row_sum - 1).abs().max().item():.3e} (atol {atol})"


def _assert_moments(shape, a, seed, dtype):
    """Check the symmetric Dirichlet mean and variance against closed form.

    The simplex row sum and bounds are exact per row so they gate
    correctness at any row count. The mean estimate converges slowly and the
    variance estimate needs many samples, so the variance is only asserted in
    the large set. Small sizes use few rows, giving a loose mean and no
    variance assertion.
    """
    _assert_simplex(shape, a, seed, dtype)
    raw = _run(shape, a, seed, dtype)
    k = shape[-1]
    s = raw.to(torch.float64).flatten()
    n = s.numel()
    mean = s.mean().item()
    exp_mean = 1.0 / k

    small_n = n < (1 << 12)
    if dtype == torch.float32:
        m_rel = 0.20 if small_n else (0.03 if n >= (1 << 20) else 0.06)
        v_rel = 0.15 if n >= (1 << 20) else 0.30
    else:  # fp16
        m_rel = 0.30 if small_n else 0.10
        v_rel = 0.45

    assert abs(mean - exp_mean) / exp_mean < m_rel,\
        f"mean {mean:.6f} off 1/K {exp_mean:.6f} (rel tol {m_rel})"
    if not small_n:
        var = s.var(unbiased=True).item()
        exp_var = (k - 1.0) / (k * k * (k * a + 1.0))
        if exp_var > 0:
            assert abs(var - exp_var) / exp_var < v_rel,\
                f"var {var:.3e} off {exp_var:.3e} (rel tol {v_rel})"


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
    "smp": _assert_simplex,
    "mom": _assert_moments,
    "det": _assert_determinism,
    "div": _assert_divergence,
}


def _exec(case):
    shape, a, seed, dtype, kind, label = case
    _DISPATCH[kind](shape, a, seed, dtype)


# ---------------------------------------------------------------------------
# Case matrices. Each entry carries shape, concentration, seed, dtype, kind
# and label, with the last shape axis acting as the simplex axis. Labels
# avoid the substrings "small" and "large" to dodge a pytest filter
# collision. Row counts stay modest in the fast set because each row tile
# runs the rejection loop; broad moment convergence lives in the large set.
# A concentration below one exercises the boost branch and one or more the
# main branch. The simplex axis spans the reduce vector-width boundary.
# ---------------------------------------------------------------------------

# Row counts stay tiny because per-row processing runs the rejection loop on
# each row tile. The simplex row sum and bounds are exact per row so a
# handful of rows fully gates correctness; tight moment and variance
# convergence over many rows lives in the large set.
CASES_SMALL = [
    ((16, 16), 2.0, 11, torch.float32, "mom", "mom_main_f32"),
    ((16, 16), 0.5, 12, torch.float32, "mom", "mom_boost_f32"),
    ((16, 16), 2.0, 13, torch.float16, "mom", "mom_main_f16"),
    ((16, 16), 0.7, 14, torch.float16, "mom", "mom_boost_f16"),
    ((8, 63), 2.0, 15, torch.float32, "smp", "kvecm1_f32"),
    ((8, 64), 2.0, 16, torch.float32, "smp", "kvec_f32"),
    ((8, 65), 2.0, 17, torch.float32, "smp", "kvecp1_f32"),
    ((4, 128), 2.0, 18, torch.float16, "smp", "kvec_f16"),
    ((4, 129), 2.0, 19, torch.float16, "smp", "kvecp1_f16"),
    ((2, 4, 16), 1.5, 20, torch.float32, "smp", "smp3d_f32"),
    ((8, 16), 2.0, 21, torch.float32, "det", "det_f32"),
    ((8, 16), 2.0, 22, torch.float32, "div", "div_f32"),
]

CASES_LARGE = [
    ((1 << 14, 8), 2.0, 31, torch.float32, "mom", "mombig_main_f32"),
    ((1 << 14, 8), 0.5, 32, torch.float32, "mom", "mombig_boost_f32"),
    ((1 << 14, 8), 5.0, 33, torch.float32, "mom", "mombig_a5_f32"),
    ((1 << 13, 8), 2.0, 34, torch.float16, "mom", "mombig_f16"),
    ((64, 200), 2.0, 35, torch.float32, "smp", "kgtvec_chunked_f32"),
    ((9000, 8), 2.0, 36, torch.float32, "mom", "rows_gt_u16_f32"),
    ((1 << 13, 16), 1.5, 37, torch.float32, "smp", "smpbig_f32"),
    ((9000, 8), 2.0, 38, torch.float32, "det", "detbig_f32"),
    ((1 << 13, 8), 2.0, 39, torch.float32, "div", "divbig_f32"),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "case", CASES_SMALL,
    ids=[c[-1] for c in CASES_SMALL])


def test_dirichlet_small(case):
    _exec(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "case", CASES_LARGE,
    ids=[c[-1] for c in CASES_LARGE])


def test_dirichlet_large(case):
    _exec(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_dirichlet_k1(dtype):
    """A single simplex coordinate clamps every element near one."""
    raw = _run((16, 1), 2.0, 7, dtype).to(torch.float64)
    assert (raw > 0.0).all() and (raw <= 1.0 + 1e-6).all(),\
        "K=1 output must be in (0, 1]"
    atol = 1e-3 if dtype == torch.float32 else 2e-3
    assert torch.allclose(raw, torch.ones_like(raw), atol=atol),\
        f"K=1 must produce ~1 (max_value); max dev {(raw - 1).abs().max().item():.3e}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("bad", ["neg", "nan", "scalar"])
def test_dirichlet_invalid(bad):
    """Negative, not-a-number or scalar inputs are rejected on the host."""
    if bad == "scalar":
        a = torch.tensor(2.0, dtype=torch.float32)  # scalar, no simplex axis
        with pytest.raises(RuntimeError, match="at least 1 dimension"):
            torch.ops.ops_multimodal_fusion.dirichlet(a.npu(), 0)
    else:
        a = torch.full((4, 8), 2.0, dtype=torch.float32)
        a[1, 3] = -1.0 if bad == "neg" else float("nan")
        with pytest.raises(RuntimeError, match="non-negative"):
            torch.ops.ops_multimodal_fusion.dirichlet(a.npu(), 0)
