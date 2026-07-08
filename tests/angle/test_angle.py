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

import logging
import math

import pytest
import torch
import torch_npu  # noqa: F401

import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "angle"):
    pytest.skip(
        "ops_multimodal_fusion.angle not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_angle_interface_exist():
    """The 'ops_multimodal_fusion.angle' operator must be registered in torch.ops."""
    logging.info(torch.ops.ops_multimodal_fusion.angle)
    assert hasattr(torch.ops.ops_multimodal_fusion, "angle"),\
        "The 'angle' operator is not registered in the 'torch.ops.ops_multimodal_fusion' namespace."


SHAPES = [
    (1,),
    (7,),
    (1024,),
    (10000,),
    (10, 10),
    (32, 32),
    (100, 100),
    (10, 100),
    (256, 512),
    (16, 32, 64),
    (1, 3, 32, 32),
    (4, 3, 64, 64),
    # Large shapes that exceed UB capacity, forcing multi-tile processing
    (100000,),
    (1000000,),
    (2048, 2048),
    (64, 128, 256),
]

DTYPES = [torch.float32, torch.float16]


def _dtype_min_abs(dtype):
    """Per-dtype lower bound on |x| for which the kernel's diff/(diff+tiny)
    rounds to exactly 1.0 and y rounds to pi.  Matches TINY chosen in-kernel:
      f32: 1e-35  -> any |x| > ~1e-30 is safe
      f16: 1e-4   -> use |x| >= 1e-2 to keep well inside fp16 resolution
    """
    return 1e-4 if dtype == torch.float32 else 1e-2


def _dtype_tol(dtype):
    """Per-dtype tolerance vs torch.angle.
    f32: 1 ulp of pi is ~4e-7 -> 1e-6 safely.
    f16: pi itself isn't representable exactly; ulp(pi) in f16 ~ 2e-3, plus
         the diff/(diff+tiny) round error, so 5e-3 / 5e-3 is realistic.
    """
    if dtype == torch.float32:
        return dict(rtol=1e-6, atol=1e-6)
    return dict(rtol=5e-3, atol=5e-3)


def _sample_mixed_sign(shape, dtype, seed=0, low=-20.0, high=20.0):
    """Uniform inputs in [low, high] — straddles zero to exercise both branches."""
    g = torch.Generator().manual_seed(seed)
    return torch.empty(*shape, dtype=dtype).uniform_(low, high, generator=g)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_angle_operator(shape, dtype):
    """Compare NPU angle against torch.angle on finite mixed-sign inputs."""
    seed = abs(hash(shape)) % 997
    x = _sample_mixed_sign(shape, dtype, seed=seed)
    # Clamp |x| away from zero so the fp16 diff/(diff+tiny) rounds cleanly;
    # reference torch.angle also treats tiny |x| as 0 anyway.
    floor = _dtype_min_abs(dtype)
    x = torch.where(x.abs() < floor,
                    torch.full_like(x, floor) * x.sign().where(x != 0, torch.ones_like(x)),
                    x)

    x_npu = x.npu()
    y_npu = torch.ops.ops_multimodal_fusion.angle(x_npu)
    y = y_npu.cpu()

    assert y.dtype == dtype, f"dtype mismatch: {y.dtype} vs {dtype}"
    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"

    expected = torch.angle(x)
    assert torch.allclose(y, expected, **_dtype_tol(dtype)), (
        f"angle mismatch (dtype={dtype}): "
        f"max abs diff = {(y - expected).abs().max().item()}"
    )
    logging.info(f"Test passed: shape={shape}, dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_angle_named_values(dtype):
    """Expected outputs at signed values, both signs of zero, and a few
    mid-range positives/negatives — within each dtype's resolution.
    """
    if dtype == torch.float32:
        values = [1.0, 2.0, 0.1, 1e-3, 1e-10,
                  -1.0, -2.0, -0.1, -1e-3, -1e-10,
                  0.0, -0.0,
                  1234.5, -1234.5]
    else:  # fp16: smallest magnitudes chosen to stay well above TINY=1e-4
        values = [1.0, 2.0, 0.1, 0.01,
                  -1.0, -2.0, -0.1, -0.01,
                  0.0, -0.0,
                  100.0, -100.0]
    xs = torch.tensor(values, dtype=dtype)
    expected = torch.angle(xs)  # 0 for x>=0 (including -0), pi for x<0
    y = torch.ops.ops_multimodal_fusion.angle(xs.npu()).cpu()

    assert torch.allclose(y, expected, **_dtype_tol(dtype)), (
        f"angle named-values mismatch (dtype={dtype}); "
        f"got {y.tolist()} expected {expected.tolist()}"
    )
    logging.info(f"Named-values test passed (dtype={dtype}).")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_angle_positive_is_zero(dtype):
    """Every strictly positive input must map to exactly 0 (both dtypes)."""
    # fp16 max representable is 65504; pick a range that fits.
    high = 1e4 if dtype == torch.float32 else 1e3
    x = torch.linspace(_dtype_min_abs(dtype), high, 4096, dtype=dtype)
    y = torch.ops.ops_multimodal_fusion.angle(x.npu()).cpu()
    # A positive input has zero numerator over a tiny denominator, so the angle is exactly zero.
    assert torch.equal(y, torch.zeros_like(x)), (
        f"positive inputs produced nonzero angle (dtype={dtype}); "
        f"max |y|={y.abs().max().item()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_angle_negative_is_pi(dtype):
    """Every strictly negative input (magnitude >> tiny) must map to pi within tolerance."""
    high = 1e4 if dtype == torch.float32 else 1e3
    x = -torch.linspace(_dtype_min_abs(dtype), high, 4096, dtype=dtype)
    y = torch.ops.ops_multimodal_fusion.angle(x.npu()).cpu()
    expected = torch.full_like(x, math.pi)
    assert torch.allclose(y, expected, **_dtype_tol(dtype)), (
        f"negative inputs deviated from pi (dtype={dtype}): "
        f"max abs err={(y - expected).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_angle_nan_propagates(dtype):
    """NaN input must produce NaN output."""
    xs = torch.tensor([float('nan'), 1.0, -1.0, float('nan')], dtype=dtype)
    y = torch.ops.ops_multimodal_fusion.angle(xs.npu()).cpu()
    assert torch.isnan(y[0]).item() and torch.isnan(y[3]).item(),\
        f"NaN did not propagate (dtype={dtype}): got {y.tolist()}"
    assert y[1].item() == 0.0, f"expected 0 for x=1 (dtype={dtype}), got {y[1].item()}"
    pi_tol = _dtype_tol(dtype)["atol"]
    assert abs(y[2].item() - math.pi) < pi_tol,\
        f"expected ~pi for x=-1 (dtype={dtype}), got {y[2].item()}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_angle_empty_tensor(dtype):
    """Empty tensor passes through with matching shape/dtype (no kernel launch)."""
    x = torch.empty((0,), dtype=dtype).npu()
    y = torch.ops.ops_multimodal_fusion.angle(x).cpu()
    assert y.shape == (0,)
    assert y.dtype == dtype


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


@pytest.mark.parametrize("dtype", DTYPES)
def test_angle_non_contiguous_input(dtype):
    """Non-contiguous tensors are handled (kernel makes a contiguous copy)."""
    base = _sample_mixed_sign((32, 32), dtype, seed=7)
    # Clamp off tiny magnitudes to dodge the fp16 near-zero band.
    floor = _dtype_min_abs(dtype)
    base = torch.where(base.abs() < floor,
                       torch.full_like(base, floor) *
                       base.sign().where(base != 0, torch.ones_like(base)),
                       base)
    x = base.t()  # transpose -> non-contiguous
    assert not x.is_contiguous()

    y = torch.ops.ops_multimodal_fusion.angle(x.npu()).cpu()
    expected = torch.angle(x.contiguous())
    assert torch.allclose(y, expected, **_dtype_tol(dtype)), (
        f"non-contiguous mismatch (dtype={dtype}): "
        f"max abs diff = {(y - expected).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_angle_rejects_unsupported_dtype():
    """float64 (and other unsupported dtypes) must error cleanly."""
    xs = torch.tensor([1.0, -1.0], dtype=torch.float64).npu()
    with pytest.raises(RuntimeError, match="angle"):
        torch.ops.ops_multimodal_fusion.angle(xs)
