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
"""Tests for ops_multimodal_fusion.cauchy.

Fills a tensor (shape/dtype/device taken from x; x data unused) with i.i.d.
Cauchy(median, sigma) samples via inverse-transform sampling on an
AscendC::PhiloxRandom uniform stream.

Cauchy has no mean/variance (heavy tails), and the NPU Philox
counter->element map differs from PyTorch's curand, so bit-exact comparison
is impossible. Acceptance is distributional + structural:
  - sample median  ~= median          (Cauchy median = location param)
  - sample IQR / 2 ~= sigma           (Q1=median-sigma, Q3=median+sigma)
  - same seed + shape => identical output (deterministic kernel)
  - different seed => different output (independent stream)
  - shape/dtype preserved; finite (fp32 always; fp16 allows rare tail
    saturation to inf, consistent with PyTorch CUDA float-eps clipping)
  - median=+inf => all inf;  sigma<=0 => host reject

Stats are computed on the finite subset and in float64.

Test case counts:
  - test_cauchy_small            : 10 cases
  - test_cauchy_large            : 8 cases
  - test_cauchy_interface_exist  : 1
  - test_cauchy_median_inf       : 2 (fp32, fp16)
  - test_cauchy_invalid_sigma    : 2 (sigma=0.0, sigma=-1.0)
  - Total                        : 23 cases
"""

from collections import namedtuple

import pytest
import torch
import torch_npu  # noqa: F401  (registers NPU dispatch key)

import ops_multimodal_fusion  # noqa: F401  (registers torch.ops.ops_multimodal_fusion.cauchy)

if not hasattr(torch.ops.ops_multimodal_fusion, "cauchy"):
    pytest.skip(
        "ops_multimodal_fusion.cauchy not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_cauchy_interface_exist():
    """The 'ops_multimodal_fusion.cauchy' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "cauchy"),\
        "The 'cauchy' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _run_raw(shape, median, sigma, seed, dtype):
    """Return the sampled tensor on CPU in its native dtype."""
    x = torch.empty(*shape, dtype=dtype).npu()
    out = torch.ops.ops_multimodal_fusion.cauchy(x, float(median), float(sigma), int(seed))
    assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"
    assert out.dtype == dtype, f"dtype mismatch: {out.dtype} vs {dtype}"
    return out.cpu()


def _assert_distribution(shape, median, sigma, seed, dtype):
    """median ~= median param and IQR/2 ~= sigma on the finite subset."""
    raw = _run_raw(shape, median, sigma, seed, dtype)
    samples = raw.to(torch.float64).flatten()
    finite = samples[torch.isfinite(samples)]
    n = samples.numel()
    assert finite.numel() >= 0.98 * n,\
        f"too many non-finite samples: {n - finite.numel()}/{n}"

    med = finite.median().item()
    qs = torch.quantile(finite, torch.tensor([0.25, 0.75], dtype=torch.float64))
    iqr_half = (qs[1] - qs[0]).item() / 2.0

    tight = n >= (1 << 20)
    if dtype == torch.float32:
        med_tol = (0.02 if tight else 0.05) * sigma
        iqr_rel = 0.04 if tight else 0.08
    else:  # fp16: looser (quantization + RNG implementation differences)
        med_tol = 0.10 * sigma
        iqr_rel = 0.15

    assert abs(med - median) < med_tol,\
        f"median {med:.5f} off target {median} (tol {med_tol:.5f})"
    assert abs(iqr_half - sigma) / sigma < iqr_rel,\
        f"IQR/2 {iqr_half:.5f} off sigma {sigma} (rel tol {iqr_rel})"


def _assert_shape_finite(shape, median, sigma, seed, dtype):
    """Shape/dtype preserved; fp32 fully finite, fp16 finite-dominated."""
    raw = _run_raw(shape, median, sigma, seed, dtype)
    fin = torch.isfinite(raw.to(torch.float64))
    if dtype == torch.float32:
        assert fin.all(), "fp32 output must be finite (float-eps clip bounds tan)"
    else:
        assert fin.float().mean().item() >= 0.98, "fp16 output mostly finite"


def _assert_determinism(shape, median, sigma, seed, dtype):
    """Same seed and shape produce byte-identical output."""
    a = _run_raw(shape, median, sigma, seed, dtype)
    b = _run_raw(shape, median, sigma, seed, dtype)
    assert torch.equal(a, b), "same seed must give identical samples"


def _assert_divergence(shape, median, sigma, seed, dtype):
    """A different seed produces substantially different output."""
    a = _run_raw(shape, median, sigma, seed, dtype)
    b = _run_raw(shape, median, sigma, seed + 1, dtype)
    assert not torch.equal(a, b), "different seeds must give different samples"
    diff = (a.to(torch.float64) != b.to(torch.float64)).float().mean().item()
    assert diff > 0.5, f"streams too correlated across seeds (diff frac {diff:.3f})"


_DISPATCH = {
    "dist": _assert_distribution,
    "fin": _assert_shape_finite,
    "det": _assert_determinism,
    "div": _assert_divergence,
}


# One test case's full parameter set, bundled so the case-driven runner takes a
# single argument instead of a long positional list.
Case = namedtuple("Case", "shape median sigma seed dtype kind label")


def _exec(case):
    _DISPATCH[case.kind](case.shape, case.median, case.sigma, case.seed, case.dtype)


# ---------------------------------------------------------------------------
# Case matrices. Septuple: (shape, median, sigma, seed, dtype, kind, label).
# Labels avoid the substrings "small"/"large" (pytest -k collision).
# ---------------------------------------------------------------------------

# Fast set: minimal coverage.
#   - dtype fp32/fp16 ~1:1
#   - distribution check on a moderate-N shape per dtype
#   - vec_width boundaries (fp32=64, fp16=128): -1 / exact / +1
#   - 3D shape passthrough
#   - determinism for fp32 and fp16, plus divergence for fp32
CASES_SMALL = [
    ((1 << 16,), 0.0, 1.0, 11, torch.float32, "dist", "dist_std_f32"),
    ((1 << 16,), 2.5, 0.5, 12, torch.float16, "dist", "dist_shift_f16"),
    ((63,), 0.0, 1.0, 13, torch.float32, "fin", "vecm1_f32"),
    ((64,), 0.0, 1.0, 14, torch.float32, "fin", "vec_f32"),
    ((128,), 0.0, 1.0, 15, torch.float16, "fin", "vec_f16"),
    ((129,), 0.0, 1.0, 16, torch.float16, "fin", "vecp1_f16"),
    ((4, 16, 32), 1.0, 1.0, 17, torch.float32, "fin", "shape3d_f32"),
    ((1 << 16,), 0.0, 1.0, 18, torch.float32, "det", "det_f32"),
    ((1 << 16,), 0.0, 1.0, 19, torch.float16, "det", "det_f16"),
    ((1 << 16,), 0.0, 1.0, 20, torch.float32, "div", "div_f32"),
]

# Full set: large-N distribution, above the uint16 count boundary,
# multi-dim distribution, fp16 large-N, determinism across the count split.
CASES_LARGE = [
    ((1 << 20,), 0.0, 1.0, 31, torch.float32, "dist", "distbig_std_f32"),
    ((1 << 20,), -3.0, 2.0, 32, torch.float32, "dist", "distbig_neg_f32"),
    ((1 << 20,), 1.0, 1.0, 33, torch.float16, "dist", "distbig_f16"),
    ((70000,), 0.0, 1.0, 34, torch.float32, "dist", "count_gt_u16_f32"),
    ((8, 128), 0.5, 1.5, 35, torch.float32, "fin", "shape2d_f32"),
    ((4096,), 0.0, 1.0, 36, torch.float16, "fin", "midn_f16"),
    ((70000,), 0.0, 1.0, 37, torch.float32, "det", "detbig_f32"),
    ((1 << 20,), 0.0, 1.0, 38, torch.float32, "div", "divbig_f32"),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "case", [Case(*c) for c in CASES_SMALL],
    ids=[c[-1] for c in CASES_SMALL])


def test_cauchy_small(case):
    _exec(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "case", [Case(*c) for c in CASES_LARGE],
    ids=[c[-1] for c in CASES_LARGE])


def test_cauchy_large(case):
    _exec(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cauchy_median_inf(dtype):
    """median=+inf => every element is inf (matches PyTorch test_cauchy)."""
    raw = _run_raw((4096,), float("inf"), 0.5, 7, dtype)
    assert torch.isinf(raw.to(torch.float64)).all(),\
        "median=inf must produce an all-inf tensor"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("bad_sigma", [0.0, -1.0])
def test_cauchy_invalid_sigma(bad_sigma):
    """sigma <= 0 is rejected on host (matches PyTorch error_inputs_cauchy)."""
    x = torch.empty(16, dtype=torch.float32).npu()
    with pytest.raises(RuntimeError, match="sigma > 0.0"):
        torch.ops.ops_multimodal_fusion.cauchy(x, 0.0, bad_sigma, 0)
