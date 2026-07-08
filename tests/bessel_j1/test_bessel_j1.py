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
"""Tests for ops_multimodal_fusion.bessel_j1.

J1(x) = Bessel function of the first kind, order 1. ODD, oscillatory,
bounded, J1(0)=0. Deterministic elementwise op -> real allclose vs an
fp64-promoted torch golden (NOT statistical):
    expected = torch.special.bessel_j1(x.double()).float()

Acceptance:
  - allclose at the finite positions (Cephes fp32 approx; bessel-family band)
  - boundaries (verified against the torch golden, not assumed): J1(0)=0,
    J1(+/-inf)=NaN and NaN->NaN; odd J1(-x)=-J1(x)
  - region switch |x|=5 both sides; J1 zeros (~3.832, 7.016, 10.174, ...)
  - shape/dtype preserved (float32 only); empty tensor; non-contiguous input

float32 only (the whole bessel/special-math family in this repo is fp32-only).

Test case counts:
  - test_bessel_j1_small            : 12 cases
  - test_bessel_j1_large            :  6 cases
  - test_bessel_j1_interface_exist  :  1
  - test_bessel_j1_empty            :  1
  - test_bessel_j1_odd_symmetry     :  1
  - test_bessel_j1_special_values   :  1
  - Total                           : 22 cases
"""

from collections import namedtuple

import pytest
import torch
import torch_npu  # noqa: F401  (registers NPU dispatch key)

import ops_multimodal_fusion  # noqa: F401  (registers torch.ops.ops_multimodal_fusion.bessel_j1)

if not hasattr(torch.ops.ops_multimodal_fusion, "bessel_j1"):
    pytest.skip(
        "ops_multimodal_fusion.bessel_j1 not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_bessel_j1_interface_exist():
    """The 'ops_multimodal_fusion.bessel_j1' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "bessel_j1"),\
        "The 'bessel_j1' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _golden(x):
    """fp64-promoted torch reference, cast back to fp32 (transcendental rule)."""
    return torch.special.bessel_j1(x.detach().to(torch.float64)).to(torch.float32)


def assert_close(actual, expected, *, rtol=5e-4, atol=5e-4):
    a = actual.detach().cpu()
    e = expected.detach().cpu()
    fin = torch.isfinite(a) & torch.isfinite(e)
    if fin.any():
        mx = (a[fin] - e[fin]).abs().max().item()
        assert torch.allclose(a[fin], e[fin], rtol=rtol, atol=atol),\
            f"max abs diff {mx:.3e} exceeds tol (rtol={rtol}, atol={atol})"
    assert torch.equal(torch.isnan(a), torch.isnan(e)), "NaN mask mismatch"


def _run(x):
    xn = x.npu()
    out = torch.ops.ops_multimodal_fusion.bessel_j1(xn)
    assert out.dtype == torch.float32, f"dtype must be float32, got {out.dtype}"
    assert tuple(out.shape) == tuple(x.shape), f"shape {tuple(out.shape)} vs {tuple(x.shape)}"
    return out.cpu()


def _assert_range(shape, lo, hi, seed):
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(*shape, generator=g, dtype=torch.float32) * (hi - lo) + lo)
    assert_close(_run(x), _golden(x))


def _assert_points(pts, seed):
    x = torch.tensor(pts, dtype=torch.float32)
    assert_close(_run(x), _golden(x))


# ---------------------------------------------------------------------------
# Case matrices. Labels avoid the substrings "small"/"large". bessel_j1 is
# pure elementwise (1 vf per tile) so each case is fast; the fast set still
# keeps numel modest (minimal coverage != necessarily large shapes).
# J1 is odd -> ranges are symmetric about 0. J1 real zeros ~ 3.8317, 7.0156,
# 10.1735, 13.3237.
# ---------------------------------------------------------------------------

_EDGE = [0.0, 1e-6, -1e-6, 4.999, 5.0, 5.001, -5.0, 3.831705970, 7.015586670,
         10.17346814, 13.32369194, 0.5, -0.5, 3.0, -12.0, 40.0]

# One test case's full parameter set, bundled so the case-driven test takes a
# single argument instead of a long positional list.
Case = namedtuple("Case", "shape_or_pts lo hi seed kind label")

CASES_SMALL = [
    ((64,), -5.0, 5.0, 11, "rng", "smallreg_f32"),
    ((64,), 5.0, 32.0, 12, "rng", "largereg_f32"),
    ((64,), -32.0, -5.0, 13, "rng", "negreg_f32"),
    ((63,), -32.0, 32.0, 14, "rng", "vecm1_f32"),
    ((65,), -32.0, 32.0, 15, "rng", "vecp1_f32"),
    ((8, 16), -30.0, 30.0, 16, "rng", "shape2d_f32"),
    ((4, 5, 8), -25.0, 25.0, 17, "rng", "shape3d_f32"),
    ((128,), -1e-3, 1e-3, 18, "rng", "nearzero_f32"),
    ((128,), -5.5, 5.5, 19, "rng", "switchband_f32"),
    ((96,), 3.0, 4.5, 20, "rng", "nearzero1_f32"),
    (_EDGE, 0, 0, 21, "pts", "edgepts_f32"),
    ((512,), -50.0, 50.0, 22, "rng", "wide_f32"),
]

CASES_LARGE = [
    ((1 << 16,), -60.0, 60.0, 31, "rng", "bignumel_f32"),
    ((8, 64, 96), -40.0, 40.0, 32, "rng", "big3d_f32"),
    ((4, 8, 16, 24), -35.0, 35.0, 33, "rng", "shape4d_f32"),
    ((1 << 15,), -200.0, 200.0, 34, "rng", "farfield_f32"),
    ((2000,), 50.0, 200.0, 35, "rng", "farpos_f32"),
    ((1 << 16,), -5.0, 5.0, 36, "rng", "bigsmallreg_f32"),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "case", [Case(*c) for c in CASES_SMALL],
    ids=[c[-1] for c in CASES_SMALL])


def test_bessel_j1_small(case):
    if case.kind == "pts":
        _assert_points(case.shape_or_pts, case.seed)
    else:
        _assert_range(case.shape_or_pts, case.lo, case.hi, case.seed)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "case", [Case(*c) for c in CASES_LARGE],
    ids=[c[-1] for c in CASES_LARGE])


def test_bessel_j1_large(case):
    _assert_range(case.shape_or_pts, case.lo, case.hi, case.seed)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_bessel_j1_empty():
    out = torch.ops.ops_multimodal_fusion.bessel_j1(torch.empty((0,), dtype=torch.float32).npu())
    assert out.dtype == torch.float32 and out.numel() == 0


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_bessel_j1_odd_symmetry():
    """J1(-x) == -J1(x) (odd function)."""
    g = torch.Generator().manual_seed(99)
    x = torch.rand(777, generator=g, dtype=torch.float32) * 40.0 + 1e-4
    pos = _run(x).to(torch.float64)
    neg = _run(-x).to(torch.float64)
    assert torch.allclose(neg, -pos, rtol=0, atol=0),\
        "J1 must be odd: J1(-x) == -J1(x)"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_bessel_j1_special_values():
    """x=0->0; +-inf->NaN and NaN->NaN (matches torch.special.bessel_j1,
    whose Cephes asymptotic phase-reduction yields NaN at infinity);
    odd near 0.
    """
    x = torch.tensor([0.0, float("inf"), float("-inf"), float("nan"),
                      1e-7, -1e-7, 1e6, -1e6], dtype=torch.float32)
    out = _run(x).to(torch.float64)
    assert out[0].item() == 0.0, f"J1(0) must be 0, got {out[0].item()}"
    assert torch.isnan(out[1]) and torch.isnan(out[2]),\
        "J1(+/-inf) must be NaN (matches the torch reference)"
    assert torch.isnan(out[3]), "J1(NaN) must be NaN"
    # J1 is odd and ~ x/2 near 0: J1(+1e-7) ~= -J1(-1e-7), both ~5e-8.
    assert abs(out[4].item() + out[5].item()) < 1e-9, "J1 odd near 0"
    assert abs(out[4].item() - 0.5e-7) < 1e-8, "J1(1e-7) ~= x/2"
