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
"""Tests for ops_multimodal_fusion.bessel_j0.

J0(x) = Bessel function of the first kind, order 0. Even, oscillatory,
bounded |J0| <= 1. Deterministic elementwise op -> real allclose vs an
fp64-promoted torch golden (NOT statistical):
    expected = torch.special.bessel_j0(x.double()).float()

Acceptance:
  - allclose at the finite positions (Cephes fp32 approx; bessel-family band)
  - boundaries: J0(0)=1, J0(+/-inf)=NaN and NaN->NaN (matches the torch
    reference), even J0(-x)=J0(x)
  - region switch |x|=5 and tiny |x|=1e-5 both sides; J0 zeros (~2.405, ...)
  - shape/dtype preserved (float32 only); empty tensor; non-contiguous input

float32 only (the whole bessel/special-math family in this repo is fp32-only).

Test case counts:
  - test_bessel_j0_small            : 12 cases
  - test_bessel_j0_large            :  6 cases
  - test_bessel_j0_interface_exist  :  1
  - test_bessel_j0_empty            :  1
  - test_bessel_j0_special_values   :  1
  - test_bessel_j0_even_symmetry    :  1
  - Total                           : 22 cases
"""

from collections import namedtuple

import pytest
import torch
import torch_npu  # noqa: F401  (registers NPU dispatch key)

import ops_multimodal_fusion  # noqa: F401  (registers torch.ops.ops_multimodal_fusion.bessel_j0)

if not hasattr(torch.ops.ops_multimodal_fusion, "bessel_j0"):
    pytest.skip(
        "ops_multimodal_fusion.bessel_j0 not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_bessel_j0_interface_exist():
    """The 'ops_multimodal_fusion.bessel_j0' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "bessel_j0"),\
        "The 'bessel_j0' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _golden(x):
    """fp64-promoted torch reference, cast back to fp32 (transcendental rule)."""
    return torch.special.bessel_j0(x.detach().to(torch.float64)).to(torch.float32)


def assert_close(actual, expected, *, rtol=5e-4, atol=5e-4):
    a = actual.detach().cpu()
    e = expected.detach().cpu()
    fin = torch.isfinite(a) & torch.isfinite(e)
    if fin.any():
        mx = (a[fin] - e[fin]).abs().max().item()
        assert torch.allclose(a[fin], e[fin], rtol=rtol, atol=atol),\
            f"max abs diff {mx:.3e} exceeds tol (rtol={rtol}, atol={atol})"
    # non-finite positions must agree structurally (nan<->nan; the golden
    # also yields NaN at +-inf, so the NaN-mask check covers them)
    assert torch.equal(torch.isnan(a), torch.isnan(e)), "NaN mask mismatch"


def _run(x):
    xn = x.npu()
    out = torch.ops.ops_multimodal_fusion.bessel_j0(xn)
    assert out.dtype == torch.float32, f"dtype must be float32, got {out.dtype}"
    assert tuple(out.shape) == tuple(x.shape), f"shape {tuple(out.shape)} vs {tuple(x.shape)}"
    return out.cpu()


def _assert_range(shape, lo, hi, seed):
    """Uniform x in [lo, hi] (or symmetric if lo<0) vs fp64 golden."""
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(*shape, generator=g, dtype=torch.float32) * (hi - lo) + lo)
    assert_close(_run(x), _golden(x))


def _assert_points(pts, seed):
    """Explicit point set (region edges / zeros / dense near a value)."""
    x = torch.tensor(pts, dtype=torch.float32)
    assert_close(_run(x), _golden(x))


_DISPATCH = {"rng": _assert_range, "pts": _assert_points}


# ---------------------------------------------------------------------------
# Case matrices. Labels avoid the substrings "small"/"large". bessel_j0 is
# pure elementwise (1 vf per tile) so each case is fast; the fast set still
# keeps numel modest (minimal coverage != necessarily large shapes).
# ---------------------------------------------------------------------------

# region edges, tiny boundary, J0 real zeros (~2.4048, 5.5201, 8.6537, 11.7915)
_EDGE = [0.0, 1e-6, 9.9e-6, 1e-5, 1.1e-5, 4.999, 5.0, 5.001, 2.404825558,
         5.520078110, 8.653727913, 11.79153444, 0.5, 3.0, 12.0, 40.0]

# One test case's full parameter set, bundled so the case-driven test takes a
# single argument instead of a long positional list.
Case = namedtuple("Case", "shape_or_pts lo hi seed kind label")

CASES_SMALL = [
    ((64,), 1e-4, 5.0, 11, "rng", "smallreg_f32"),
    ((64,), 5.0, 32.0, 12, "rng", "largereg_f32"),
    ((63,), 1e-4, 32.0, 13, "rng", "vecm1_f32"),
    ((65,), 1e-4, 32.0, 14, "rng", "vecp1_f32"),
    ((8, 16), 1e-4, 30.0, 15, "rng", "shape2d_f32"),
    ((4, 5, 8), 1e-4, 25.0, 16, "rng", "shape3d_f32"),
    ((256,), -32.0, 32.0, 17, "rng", "negsym_f32"),
    ((128,), 0.0, 1e-4, 18, "rng", "tinyband_f32"),
    ((128,), 4.5, 5.5, 19, "rng", "switchband_f32"),
    ((96,), 2.0, 3.0, 20, "rng", "nearzero1_f32"),
    (_EDGE, 0, 0, 21, "pts", "edgepts_f32"),
    ((512,), 1e-4, 50.0, 22, "rng", "wide_f32"),
]

CASES_LARGE = [
    ((1 << 16,), 1e-4, 60.0, 31, "rng", "bignumel_f32"),
    ((8, 64, 96), 1e-4, 40.0, 32, "rng", "big3d_f32"),
    ((4, 8, 16, 24), 1e-4, 35.0, 33, "rng", "shape4d_f32"),
    ((1 << 15,), -80.0, 80.0, 34, "rng", "bignegsym_f32"),
    ((2000,), 50.0, 200.0, 35, "rng", "farfield_f32"),
    ((1 << 16,), 1e-4, 5.0, 36, "rng", "bigsmallreg_f32"),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "case", [Case(*c) for c in CASES_SMALL],
    ids=[c[-1] for c in CASES_SMALL])


def test_bessel_j0_small(case):
    if case.kind == "pts":
        _assert_points(case.shape_or_pts, case.seed)
    else:
        _assert_range(case.shape_or_pts, case.lo, case.hi, case.seed)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "case", [Case(*c) for c in CASES_LARGE],
    ids=[c[-1] for c in CASES_LARGE])


def test_bessel_j0_large(case):
    _assert_range(case.shape_or_pts, case.lo, case.hi, case.seed)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_bessel_j0_empty():
    out = torch.ops.ops_multimodal_fusion.bessel_j0(torch.empty((0,), dtype=torch.float32).npu())
    assert out.dtype == torch.float32 and out.numel() == 0


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_bessel_j0_special_values():
    """x=0->1; +-inf->NaN and NaN->NaN (matches torch.special.bessel_j0,
    whose Cephes asymptotic phase-reduction yields NaN at infinity);
    +-1e-7->~1.
    """
    x = torch.tensor([0.0, float("inf"), float("-inf"), float("nan"),
                      1e-7, -1e-7, 1e6, -1e6], dtype=torch.float32)
    out = _run(x).to(torch.float64)
    assert abs(out[0].item() - 1.0) < 1e-4, f"J0(0) must be 1, got {out[0].item()}"
    assert torch.isnan(out[1]) and torch.isnan(out[2]),\
        "J0(+/-inf) must be NaN (matches the torch reference)"
    assert torch.isnan(out[3]), "J0(NaN) must be NaN"
    assert abs(out[4].item() - 1.0) < 1e-4 and abs(out[5].item() - 1.0) < 1e-4,\
        "J0(+-1e-7) ~= 1"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_bessel_j0_even_symmetry():
    """J0(-x) == J0(x) (even function)."""
    g = torch.Generator().manual_seed(99)
    x = torch.rand(777, generator=g, dtype=torch.float32) * 40.0 + 1e-4
    pos = _run(x)
    neg = _run(-x)
    assert torch.equal(pos, neg), "J0 must be even: J0(-x) == J0(x)"
