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

"""Tests for ops_multimodal_fusion.multinomial.

The input is a tensor of per-row non-negative weights, either one or two
dimensional; the output is integer category indices preserving the leading
shape. Sampling follows the with-replacement inverse cumulative path: a row
sum and a scalar prefix accumulation, then for each draw a Philox uniform
value selects the matching category, with trailing zero-probability
categories backed off exactly as the reference does.

The NPU Philox stream differs from the reference generator, so a bit-exact
comparison cannot be done; acceptance instead relies on structural plus
statistical checks at double precision:
  - every index lies within the valid category range, is integer typed, and
    preserves shape
  - a category with weight zero is never selected
  - empirical category frequency approximates the normalized weights
  - the same seed and input give identical draws; a different seed differs
  - unsupported dtype, non-positive sample count, wrong dimensionality,
    negative or non-finite weights, an all-zero row, an oversized category
    count, or without-replacement multi-draw are all rejected on the host

This release supports the with-replacement path and the single-draw case;
without-replacement multi-draw and oversized category counts are deferred and
rejected on the host.

Test case counts:
  - test_multinomial_small                       : 12 cases
  - test_multinomial_large                       :  8 cases
  - test_multinomial_interface_exist             :  1
  - test_multinomial_zero_prob                   :  2
  - test_multinomial_invalid                     :  7
  - test_multinomial_deferred_without_replacement:  1
  - Total                                        : 31 cases
"""

import torch
import torch_npu  # noqa: F401  (registers NPU dispatch key)
import pytest
import ops_multimodal_fusion  # noqa: F401  (registers torch.ops.ops_multimodal_fusion.multinomial)

if not hasattr(torch.ops.ops_multimodal_fusion, "multinomial"):
    pytest.skip(
        "ops_multimodal_fusion.multinomial not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_multinomial_interface_exist():
    """The 'ops_multimodal_fusion.multinomial' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "multinomial"),\
        "The 'multinomial' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _run(probs, n, replacement, seed, dtype):
    """Run the operator.

    probs is a flat list or a list of lists; returns the integer output moved
    back to CPU alongside the CPU input.
    """
    x = torch.tensor(probs, dtype=dtype).npu()
    out = torch.ops.ops_multimodal_fusion.multinomial(x, int(n), bool(replacement), int(seed))
    assert out.dtype == torch.int64, f"dtype must be int64, got {out.dtype}"
    return out.cpu(), x.cpu()


def _expected_p(x):
    """Row-normalized probabilities in double precision, one row per distribution."""
    xf = x.to(torch.float64)
    if xf.dim() == 1:
        xf = xf.reshape(1, -1)
    return xf / xf.sum(dim=1, keepdim=True)


def _freqs(out, num_categories):
    """Per-row empirical category frequency, one row per distribution."""
    o = out.reshape(1, -1) if out.dim() == 1 else out
    nd, n = o.shape
    f = torch.zeros((nd, num_categories), dtype=torch.float64)
    for r in range(nd):
        f[r] = torch.bincount(o[r], minlength=num_categories).to(torch.float64) / n
    return f


def _assert_structural(probs, n, replacement, seed, dtype):
    """Check structural invariants of the drawn indices.

    Indices stay within range, are integer typed, preserve shape, and never
    land on a zero-weight category.
    """
    out, x = _run(probs, n, replacement, seed, dtype)
    num_categories = x.shape[-1]
    exp_shape = (n,) if x.dim() == 1 else (x.shape[0], n)
    assert tuple(out.shape) == exp_shape, f"shape {tuple(out.shape)} vs {exp_shape}"
    assert (out >= 0).all() and (out < num_categories).all(), "indices must be valid categories"
    p = _expected_p(x)
    o = out.reshape(1, -1) if out.dim() == 1 else out
    for r in range(o.shape[0]):
        zero_cats = (p[r] == 0).nonzero().flatten().tolist()
        for zc in zero_cats:
            assert (o[r] != zc).all(), f"zero-weight category {zc} was selected"


def _assert_freq(probs, n, replacement, seed, dtype):
    """Empirical frequency approximates the normalized weights, statistically."""
    _assert_structural(probs, n, replacement, seed, dtype)
    out, x = _run(probs, n, replacement, seed, dtype)
    num_categories = x.shape[-1]
    p = _expected_p(x)
    f = _freqs(out, num_categories)
    # Sampling tolerance uses several binomial standard deviations plus a dtype
    # floor. Tight convergence with many draws lives in the large set.
    import math
    for r in range(p.shape[0]):
        for c in range(num_categories):
            pc = p[r, c].item()
            std = math.sqrt(max(pc * (1.0 - pc), 1e-6) / n)
            tol = 3.5 * std + (0.02 if dtype == torch.float32 else 0.04)
            assert abs(f[r, c].item() - pc) < tol, (
                f"freq[{r},{c}]={f[r,c]:.4f} vs p={pc:.4f} (tol {tol:.4f}, n={n})")


def _assert_determinism(probs, n, replacement, seed, dtype):
    a, _ = _run(probs, n, replacement, seed, dtype)
    b, _ = _run(probs, n, replacement, seed, dtype)
    assert torch.equal(a, b), "same seed must give identical samples"


def _assert_divergence(probs, n, replacement, seed, dtype):
    a, _ = _run(probs, n, replacement, seed, dtype)
    b, _ = _run(probs, n, replacement, seed + 1, dtype)
    assert not torch.equal(a, b), "different seeds must differ"
    diff = (a != b).float().mean().item()
    assert diff > 0.4, f"streams too correlated across seeds (diff {diff:.3f})"


_DISPATCH = {
    "frq": _assert_freq,
    "str": _assert_structural,
    "det": _assert_determinism,
    "div": _assert_divergence,
}


def _exec(case):
    probs, n, replacement, seed, dtype, kind, label = case
    _DISPATCH[kind](probs, n, replacement, seed, dtype)


# ---------------------------------------------------------------------------
# Case matrices. Each row carries weights, sample count, replacement flag,
# seed, dtype, check kind, and a label. Labels avoid the words used by the
# small and large filters to dodge a name collision. Kernel cost grows with the
# number of distributions times the number of draws; the fast set keeps that
# product tiny, and tight frequency convergence lives in the large set.
# ---------------------------------------------------------------------------

_U4 = [1.0, 1.0, 1.0, 1.0]                       # uniform, four categories
_SK4 = [0.1, 0.2, 0.3, 0.4]                      # skewed, four categories
_Z5 = [0.5, 0.0, 0.3, 0.0, 0.2]                  # zero-weight categories, five total
_Z4 = [0.5, 0.0, 0.3, 0.2]                       # zero-weight category for a two-row case
_U8 = [1.0] * 8                                  # uniform, eight categories

CASES_SMALL = [
    (_U4, 256, True, 11, torch.float32, "frq", "uni4_f32"),
    (_SK4, 256, True, 12, torch.float32, "frq", "skew4_f32"),
    (_U4, 256, True, 13, torch.float16, "frq", "uni4_f16"),
    (_Z5, 200, True, 14, torch.float32, "str", "zero5_f32"),
    ([_U4, _SK4], 128, True, 15, torch.float32, "frq", "row2_f32"),
    ([_SK4, _U4], 128, True, 16, torch.float16, "str", "row2_f16"),
    (_U8, 64, True, 17, torch.float32, "str", "uni8_1samp_ish_f32"),
    (list(range(1, 65)), 4, True, 18, torch.float32, "str", "kvec64_f32"),
    ([1.0] * 63, 4, True, 19, torch.float32, "str", "kvec63_f32"),
    (_SK4, 1, False, 20, torch.float32, "str", "n1_norepl_f32"),
    (_U4, 192, True, 21, torch.float32, "det", "det_f32"),
    (_U4, 192, True, 22, torch.float32, "div", "div_f32"),
]

CASES_LARGE = [
    (_U8, 20000, True, 31, torch.float32, "frq", "unibig8_f32"),
    (_SK4, 20000, True, 32, torch.float32, "frq", "skewbig4_f32"),
    (_U8, 12000, True, 33, torch.float16, "frq", "unibig8_f16"),
    ([_SK4, _U4, _Z4], 8000, True, 34, torch.float32, "frq", "rowsbig_f32"),
    (list(range(1, 33)), 16000, True, 35, torch.float32, "frq", "kvec32big_f32"),
    (_U4, 40000, True, 36, torch.float32, "frq", "uni4big_f32"),
    (_U8, 33000, True, 37, torch.float32, "det", "detbig_tile_f32"),
    (_SK4, 20000, True, 38, torch.float32, "div", "divbig_f32"),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "case", CASES_SMALL,
    ids=[c[-1] for c in CASES_SMALL])


def test_multinomial_small(case):
    _exec(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "case", CASES_LARGE,
    ids=[c[-1] for c in CASES_LARGE])


def test_multinomial_large(case):
    _exec(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_multinomial_zero_prob(dtype):
    """A category with weight 0 is never selected."""
    out, x = _run([0.0, 0.6, 0.0, 0.4, 0.0], 1000, True, 5, dtype)
    o = out.flatten()
    for zc in (0, 2, 4):
        assert (o != zc).all(), f"zero-weight category {zc} was selected"
    assert (o >= 0).all() and (o < 5).all()


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("bad", [
    "dtype", "nsamples", "dim0", "dim3", "negative", "naninf", "rowsum0",
])


def test_multinomial_invalid(bad):
    """Host TORCH_CHECK rejects per PyTorch/v1-scope rules."""
    if bad == "dtype":
        x = torch.tensor([1, 2, 3], dtype=torch.int32).npu()
        with pytest.raises(RuntimeError, match="float32 and float16"):
            torch.ops.ops_multimodal_fusion.multinomial(x, 2, True, 0)
    elif bad == "nsamples":
        x = torch.tensor([0.5, 0.5], dtype=torch.float32).npu()
        with pytest.raises(RuntimeError, match="num_samples"):
            torch.ops.ops_multimodal_fusion.multinomial(x, 0, True, 0)
    elif bad == "dim0":
        x = torch.tensor(1.0, dtype=torch.float32).npu()
        with pytest.raises(RuntimeError, match="1 or 2 dimensional"):
            torch.ops.ops_multimodal_fusion.multinomial(x, 1, True, 0)
    elif bad == "dim3":
        x = torch.ones((2, 2, 2), dtype=torch.float32).npu()
        with pytest.raises(RuntimeError, match="1 or 2 dimensional"):
            torch.ops.ops_multimodal_fusion.multinomial(x, 1, True, 0)
    elif bad == "negative":
        x = torch.tensor([0.5, -0.1, 0.6], dtype=torch.float32).npu()
        with pytest.raises(RuntimeError, match="non-negative"):
            torch.ops.ops_multimodal_fusion.multinomial(x, 2, True, 0)
    elif bad == "naninf":
        x = torch.tensor([0.5, float("inf"), 0.5], dtype=torch.float32).npu()
        with pytest.raises(RuntimeError, match="inf or nan"):
            torch.ops.ops_multimodal_fusion.multinomial(x, 2, True, 0)
    elif bad == "rowsum0":
        x = torch.zeros(5, dtype=torch.float32).npu()
        with pytest.raises(RuntimeError, match="row sum|non-negative|invalid distribution"):
            torch.ops.ops_multimodal_fusion.multinomial(x, 2, True, 0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_multinomial_deferred_without_replacement():
    """replacement False && num_samples>1 is a documented v1 reject."""
    x = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32).npu()
    with pytest.raises(RuntimeError, match="without replacement|deferred"):
        torch.ops.ops_multimodal_fusion.multinomial(x, 3, False, 0)
