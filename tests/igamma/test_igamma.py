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


import torch
import torch_npu  # noqa: F401
import pytest
import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "igamma"):
    pytest.skip(
        "ops_multimodal_fusion.igamma not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_igamma_interface_exist():
    """The 'ops_multimodal_fusion.igamma' operator must be registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "igamma"),\
        "The 'igamma' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


SHAPES = [
    (1,),
    (7,),
    (1024,),
    (10000,),
    (32, 32),
    (100, 100),
    (10, 100),
    (256, 512),
    (16, 32, 64),
    (1, 3, 32, 32),
    (4, 3, 64, 64),
    # Larger shapes that exceed UB capacity in one core, forcing multi-tile.
    (50000,),
    (256, 256),
]

DTYPES = [torch.float32]


def _sample_a(shape, seed):
    """Sample positive shape parameters in a moderate fp32-friendly range."""
    g = torch.Generator().manual_seed(seed)
    return torch.empty(*shape, dtype=torch.float32).uniform_(0.5, 8.0, generator=g)


def _sample_x(shape, seed):
    """Sample x values that exercise both lower-series and continued-fraction branches."""
    g = torch.Generator().manual_seed(seed)
    return torch.empty(*shape, dtype=torch.float32).uniform_(0.05, 12.0, generator=g)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_igamma_operator(shape, dtype):
    """Compare NPU igamma against torch.igamma across both numeric branches."""
    seed = abs(hash((shape, dtype))) % 10_000_000
    a = _sample_a(shape, seed=seed)
    x = _sample_x(shape, seed=seed + 1)

    c_npu = torch.ops.ops_multimodal_fusion.igamma(a.npu(), x.npu())
    c = c_npu.cpu()

    # Reference in float64 to keep the comparison meaningful at fp32 tol.
    expected = torch.igamma(a.to(torch.float64), x.to(torch.float64)).to(dtype)

    assert c.dtype == dtype, f"dtype mismatch: {c.dtype} vs {dtype}"
    assert c.shape == a.shape, f"shape mismatch: {c.shape} vs {a.shape}"
    assert torch.allclose(c, expected, rtol=1e-3, atol=5e-4), (
        f"igamma mismatch (dtype={dtype}, shape={shape}): "
        f"max abs diff = {(c - expected).abs().max().item()}, "
        f"max rel diff = "
        f"{((c - expected) / expected.abs().clamp_min(1e-30)).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_igamma_named_values():
    """Closed-form / tabulated values for spot-checks."""
    # P 1 x 1 - exp -x lower regularized CDF of Exp 1
    # P 0.5 x erf sqrt x
    # P 2 1 approx 0.2642411
    a = torch.tensor([1.0, 1.0, 1.0, 2.0, 0.5, 3.0, 4.0], dtype=torch.float32)
    x = torch.tensor([0.5, 1.0, 2.0, 1.0, 1.0, 2.0, 4.0], dtype=torch.float32)
    expected = torch.igamma(a.to(torch.float64), x.to(torch.float64)).to(torch.float32)

    c = torch.ops.ops_multimodal_fusion.igamma(a.npu(), x.npu()).cpu()
    assert torch.allclose(c, expected, rtol=1e-3, atol=5e-4), (
        f"named-values mismatch; got {c.tolist()} expected {expected.tolist()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_igamma_zero_x_is_zero():
    """P a 0 0 for all valid a > 0."""
    a = torch.tensor([0.5, 1.0, 2.0, 3.5, 7.0], dtype=torch.float32)
    x = torch.zeros_like(a)

    c = torch.ops.ops_multimodal_fusion.igamma(a.npu(), x.npu()).cpu()
    # log 0 -inf ⇒ exp -inf 0 ⇒ result 0 sum 0
    assert torch.allclose(c, torch.zeros_like(c), atol=1e-6), (
        f"P(a, 0) should be 0; got {c.tolist()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_igamma_continued_fraction_region():
    """x > a + 1 must use the complement continued-fraction branch."""
    a = torch.tensor([0.5, 1.0, 2.0, 4.0, 8.0], dtype=torch.float32)
    x = torch.tensor([8.0, 10.0, 12.0, 16.0, 24.0], dtype=torch.float32)
    expected = torch.igamma(a.to(torch.float64), x.to(torch.float64)).to(torch.float32)

    c = torch.ops.ops_multimodal_fusion.igamma(a.npu(), x.npu()).cpu()
    assert torch.allclose(c, expected, rtol=1e-3, atol=5e-4), (
        f"continued-fraction region mismatch; got {c.tolist()} expected {expected.tolist()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_igamma_domain_edges():
    """Domain endpoints and invalid inputs should match torch.igamma behavior."""
    a = torch.tensor([1.0, 2.0, 0.0, -1.0, 2.0], dtype=torch.float32)
    x = torch.tensor([float("inf"), 0.0, 1.0, 1.0, -1.0], dtype=torch.float32)
    expected = torch.igamma(a.to(torch.float64), x.to(torch.float64)).to(torch.float32)

    c = torch.ops.ops_multimodal_fusion.igamma(a.npu(), x.npu()).cpu()
    finite = torch.isfinite(expected)
    assert torch.allclose(c[finite], expected[finite], rtol=1e-3, atol=5e-4)
    assert torch.equal(torch.isnan(c), torch.isnan(expected)), (
        f"NaN mask mismatch; got {c.tolist()} expected {expected.tolist()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_igamma_monotonic_in_x():
    """For fixed a P a x is monotonically non-decreasing in x."""
    a = torch.full((4096,), 2.0, dtype=torch.float32)
    x = torch.linspace(0.05, 6.0, 4096, dtype=torch.float32)

    c = torch.ops.ops_multimodal_fusion.igamma(a.npu(), x.npu()).cpu()
    diffs = c[1:] - c[:-1]
    # Allow a small fp32 slack: in regions where successive c values are
    # very close, rounding can flip sign at the last bit.
    assert (diffs >= -1e-5).all(), (
        f"monotonicity violated; min diff = {diffs.min().item()}"
    )
    # And the global trend should be increasing toward 1.
    assert c[-1] > c[0]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_igamma_bounded_in_unit_interval():
    """P a x is in 0 1 for any a > 0 x > 0."""
    a = _sample_a((4096,), seed=33)
    x = _sample_x((4096,), seed=34)
    c = torch.ops.ops_multimodal_fusion.igamma(a.npu(), x.npu()).cpu()
    # Allow tiny fp32 overshoot at the boundaries.
    assert (c >= -1e-5).all(), f"c < 0; min = {c.min().item()}"
    assert (c <= 1.0 + 1e-4).all(), f"c > 1; max = {c.max().item()}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_igamma_empty_tensor():
    """Empty tensor passes through with matching shape/dtype (no kernel launch)."""
    a = torch.empty((0,), dtype=torch.float32).npu()
    x = torch.empty((0,), dtype=torch.float32).npu()
    c = torch.ops.ops_multimodal_fusion.igamma(a, x).cpu()
    assert c.shape == (0,)
    assert c.dtype == torch.float32


@pytest.mark.skip(
    reason=(
        "torch_npu in the current CANN release lacks D2D strided copy support "
        "(aclnnInplaceCopy fails with error 561103 for any non-contiguous NPU "
        "tensor, regardless of dtype). The kernel itself only operates on "
        "contiguous buffers; callers holding a transposed/strided tensor must "
        "materialize it themselves before invoking the op. "
        "Re-enable once torch_npu ships strided D2D."
    )
)


def test_igamma_non_contiguous_input():
    """Non-contiguous tensors are handled (kernel makes a contiguous copy)."""
    base_a = _sample_a((32, 32), seed=19)
    base_x = _sample_x((32, 32), seed=20)
    a = base_a.t()
    x = base_x.t()
    assert not a.is_contiguous() and not x.is_contiguous()

    c = torch.ops.ops_multimodal_fusion.igamma(a.npu(), x.npu()).cpu()
    expected = torch.igamma(
        a.contiguous().to(torch.float64),
        x.contiguous().to(torch.float64),
    ).to(torch.float32)

    assert torch.allclose(c, expected, rtol=1e-3, atol=5e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_igamma_rejects_unsupported_dtype():
    """fp64 / fp16 are not in the supported-dtype list per current scope."""
    for unsupported in [torch.float64, torch.float16]:
        a = torch.tensor([1.0, 2.0], dtype=unsupported).npu()
        x = torch.tensor([1.0, 1.0], dtype=unsupported).npu()
        with pytest.raises(RuntimeError, match="igamma"):
            torch.ops.ops_multimodal_fusion.igamma(a, x)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_igamma_rejects_shape_mismatch():
    """Mismatched input shapes must raise."""
    a = torch.full((4,), 1.0, dtype=torch.float32).npu()
    x = torch.full((5,), 1.0, dtype=torch.float32).npu()
    with pytest.raises(RuntimeError, match="igamma"):
        torch.ops.ops_multimodal_fusion.igamma(a, x)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_igamma_rejects_dtype_mismatch():
    """Mismatched input dtypes must raise."""
    a = torch.full((4,), 1.0, dtype=torch.float32).npu()
    x = torch.full((4,), 1.0, dtype=torch.float64).npu()
    with pytest.raises(RuntimeError, match="igamma"):
        torch.ops.ops_multimodal_fusion.igamma(a, x)
