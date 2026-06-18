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
"""Tests for ops_multimodal_fusion.bessel_y0.

Y0(x) = Bessel function of the second kind, order 0. Defined for x > 0
(log singularity at 0: Y0(0+)->-inf), oscillatory, bounded for large x.
Deterministic elementwise op -> real allclose vs an fp64-promoted torch
golden (NOT statistical):
    expected = torch.special.bessel_y0(x.double()).float()

Acceptance:
  - allclose at the finite positions (Cephes fp32 approx; bessel-family band)
  - domain/boundary verified against the torch golden (not assumed):
    Y0(0)=-inf, Y0(x<0)=NaN, Y0(+/-inf)=NaN, NaN->NaN
  - region switch x=5 both sides; Y0 zeros (~0.894, 3.958, 7.086, ...)
  - shape/dtype preserved (float32 only); empty tensor; non-contiguous input

float32 only (the whole bessel/special-math family in this repo is fp32-only).

Test case counts:
  - test_bessel_y0_small            : 12 cases
  - test_bessel_y0_large            :  6 cases
  - test_bessel_y0_interface_exist  :  1
  - test_bessel_y0_empty            :  1
  - test_bessel_y0_domain           :  1
  - test_bessel_y0_special_values   :  1
  - Total                           : 22 cases
"""

from collections import namedtuple

import pytest
import torch
import torch_npu  # noqa: F401  (registers NPU dispatch key)

import ops_multimodal_fusion  # noqa: F401  (registers torch.ops.ops_multimodal_fusion.bessel_y0)

if not hasattr(torch.ops.ops_multimodal_fusion, "bessel_y0"):
    pytest.skip(
        "ops_multimodal_fusion.bessel_y0 not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_bessel_y0_interface_exist():
    """The 'ops_multimodal_fusion.bessel_y0' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "bessel_y0"), \
        "The 'bessel_y0' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _golden(x):
    """fp64-promoted torch reference, cast back to fp32 (transcendental rule)."""
    return torch.special.bessel_y0(x.detach().to(torch.float64)).to(torch.float32)


def assert_close(actual, expected, *, rtol=5e-4, atol=5e-4):
    a = actual.detach().cpu()
    e = expected.detach().cpu()
    fin = torch.isfinite(a) & torch.isfinite(e)
    if fin.any():
        mx = (a[fin] - e[fin]).abs().max().item()
        assert torch.allclose(a[fin], e[fin], rtol=rtol, atol=atol), \
            f"max abs diff {mx:.3e} exceeds tol (rtol={rtol}, atol={atol})"
    assert torch.equal(torch.isnan(a), torch.isnan(e)), "NaN mask mismatch"
    # -inf positions (Y0 singularity at 0) must agree in sign+inf-ness
    assert torch.equal(torch.isinf(a) & (a < 0), torch.isinf(e) & (e < 0)), \
        "-inf mask mismatch"


def _run(x):
    xn = x.npu()
    out = torch.ops.ops_multimodal_fusion.bessel_y0(xn)
    assert out.dtype == torch.float32, f"dtype must be float32, got {out.dtype}"
    assert tuple(out.shape) == tuple(x.shape), f"shape {tuple(out.shape)} vs {tuple(x.shape)}"
    return out.cpu()


def _assert_range(shape, lo, hi, seed):
    """x in (lo, hi]; for Y0 the fast/large ranges use lo > 0 (domain)."""
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(*shape, generator=g, dtype=torch.float32) * (hi - lo) + lo)
    assert_close(_run(x), _golden(x))


def _assert_points(pts, seed):
    x = torch.tensor(pts, dtype=torch.float32)
    assert_close(_run(x), _golden(x))


# ---------------------------------------------------------------------------
# Case matrices. Labels avoid the substrings "small"/"large". bessel_y0 is
# pure elementwise (J0 helper + Y0, 2 vf per tile) so each case is fast; the
# fast set still keeps numel modest. Y0 is defined for x>0 -> ranges are
# strictly positive (lo > 0). Y0 real zeros ~ 0.8936, 3.9577, 7.0861, 10.222.
# ---------------------------------------------------------------------------

_EDGE = [1e-6, 1e-4, 1e-2, 0.1, 0.5, 4.999, 5.0, 5.001, 0.893576966,
         3.957678419, 7.086051060, 10.22234504, 1.0, 3.0, 12.0, 40.0]

# One test case's full parameter set, bundled so the case-driven test takes a
# single argument instead of a long positional list.
Case = namedtuple("Case", "shape_or_pts lo hi seed kind label")

CASES_SMALL = [
    ((64,), 1e-3, 5.0, 11, "rng", "smallreg_f32"),
    ((64,), 5.0, 32.0, 12, "rng", "largereg_f32"),
    ((64,), 1e-4, 1e-1, 13, "rng", "nearsing_f32"),
    ((63,), 1e-3, 32.0, 14, "rng", "vecm1_f32"),
    ((65,), 1e-3, 32.0, 15, "rng", "vecp1_f32"),
    ((8, 16), 1e-3, 30.0, 16, "rng", "shape2d_f32"),
    ((4, 5, 8), 1e-3, 25.0, 17, "rng", "shape3d_f32"),
    ((128,), 4.5, 5.5, 18, "rng", "switchband_f32"),
    ((96,), 0.5, 1.5, 19, "rng", "nearzero1_f32"),
    ((128,), 1e-5, 1e-2, 20, "rng", "tinypos_f32"),
    (_EDGE, 0, 0, 21, "pts", "edgepts_f32"),
    ((512,), 1e-3, 50.0, 22, "rng", "wide_f32"),
]

CASES_LARGE = [
    ((1 << 16,), 1e-3, 60.0, 31, "rng", "bignumel_f32"),
    ((8, 64, 96), 1e-3, 40.0, 32, "rng", "big3d_f32"),
    ((4, 8, 16, 24), 1e-3, 35.0, 33, "rng", "shape4d_f32"),
    ((1 << 15,), 50.0, 200.0, 34, "rng", "farfield_f32"),
    ((2000,), 1e-4, 1.0, 35, "rng", "nearsingbig_f32"),
    ((1 << 16,), 1e-3, 5.0, 36, "rng", "bigsmallreg_f32"),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "case", [Case(*c) for c in CASES_SMALL],
    ids=[c[-1] for c in CASES_SMALL])
def test_bessel_y0_small(case):
    if case.kind == "pts":
        _assert_points(case.shape_or_pts, case.seed)
    else:
        _assert_range(case.shape_or_pts, case.lo, case.hi, case.seed)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "case", [Case(*c) for c in CASES_LARGE],
    ids=[c[-1] for c in CASES_LARGE])
def test_bessel_y0_large(case):
    _assert_range(case.shape_or_pts, case.lo, case.hi, case.seed)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_bessel_y0_empty():
    out = torch.ops.ops_multimodal_fusion.bessel_y0(torch.empty((0,), dtype=torch.float32).npu())
    assert out.dtype == torch.float32 and out.numel() == 0


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_bessel_y0_domain():
    """x=0 -> -inf ; x<0 -> NaN (verified against the torch golden)."""
    x = torch.tensor([0.0, -1e-3, -1.0, -5.0, -40.0], dtype=torch.float32)
    out = _run(x).to(torch.float64)
    g = torch.special.bessel_y0(x.to(torch.float64))
    assert torch.isinf(out[0]) and out[0].item() < 0, "Y0(0) must be -inf"
    for i in range(1, 5):
        assert torch.isnan(out[i]), f"Y0({x[i].item()}) must be NaN (x<0)"
        assert torch.isnan(g[i]), "golden sanity: Y0(x<0) is NaN"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_bessel_y0_special_values():
    """0->-inf; x<0->NaN; +-inf->NaN; NaN->NaN (matches torch.special.bessel_y0)."""
    x = torch.tensor([0.0, float("inf"), float("-inf"), float("nan"),
                      1e-7, 1e6, 2.0, -2.0], dtype=torch.float32)
    out = _run(x).to(torch.float64)
    assert torch.isinf(out[0]) and out[0].item() < 0, "Y0(0) must be -inf"
    assert torch.isnan(out[1]) and torch.isnan(out[2]), \
        "Y0(+/-inf) must be NaN (matches the torch reference)"
    assert torch.isnan(out[3]), "Y0(NaN) must be NaN"
    assert torch.isnan(out[7]), "Y0(-2) must be NaN (x<0 domain)"
    # finite positive samples match the golden
    g = torch.special.bessel_y0(x.to(torch.float64))
    for i in (4, 5, 6):
        assert abs(out[i].item() - g[i].item()) < 5e-4 + 5e-4 * abs(g[i].item()), \
            f"Y0({x[i].item()}) {out[i].item()} vs golden {g[i].item()}"
