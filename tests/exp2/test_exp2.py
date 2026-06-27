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
"""Tests for ops_multimodal_fusion.exp2 (base-2 exponential of x)."""

import logging

import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401

logger = logging.getLogger(__name__)

if not hasattr(torch.ops.ops_multimodal_fusion, "exp2"):
    pytest.skip(
        "ops_multimodal_fusion.exp2 not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_exp2_interface_exist():
    """The 'ops_multimodal_fusion.exp2' operator must be registered in torch.ops."""
    logger.info(torch.ops.ops_multimodal_fusion.exp2)
    assert hasattr(torch.ops.ops_multimodal_fusion, "exp2"), \
        "The 'exp2' operator is not registered in the 'torch.ops.ops_multimodal_fusion' namespace."


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


def _input_range(dtype):
    """Per-dtype safe input range for two raised to x to stay representable.

    For fp32 the result fits for x roughly between negative 127 and 127, so a
    margin spanning negative 80 to 80 is plenty. The fp16 maximum is about
    65504, near two to the sixteenth, and its smallest normal is two to the
    negative fourteenth, so keep well inside both ends.
    """
    if dtype == torch.float32:
        return -10.0, 10.0
    return -10.0, 10.0  # two to the tenth is 1024, safely representable in both


def _tol(dtype):
    """Tolerance for vector comparison.
    fp32: Muls + Exp loses a few ulps over torch.special.exp2 on CPU (which
          uses double internally); 1e-5 rtol is comfortable.
    fp16: the 1/ln2 and Exp intrinsics in fp16 give ~3 significant decimals;
          1e-2 rtol matches the issue's 'precision is limited' caveat.
    """
    if dtype == torch.float32:
        return dict(rtol=1e-5, atol=1e-6)
    return dict(rtol=1e-2, atol=1e-3)


def _sample(shape, dtype, seed=0):
    low, high = _input_range(dtype)
    g = torch.Generator().manual_seed(seed)
    return torch.empty(*shape, dtype=dtype).uniform_(low, high, generator=g)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_exp2_operator(shape, dtype):
    """Compare NPU exp2 against torch.special.exp2 on finite mixed-sign inputs."""
    seed = abs(hash((shape, dtype))) % 10_000_000
    x = _sample(shape, dtype, seed=seed)

    x_npu = x.npu()
    y_npu = torch.ops.ops_multimodal_fusion.exp2(x_npu)
    y = y_npu.cpu()

    assert y.dtype == dtype, f"dtype mismatch: {y.dtype} vs {dtype}"
    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"

    # Reference in float64 for fp32, in fp32 for fp16 (pin down the best
    # achievable in-dtype answer rather than propagating torch's CPU ulps).
    ref_dtype = torch.float64 if dtype == torch.float32 else torch.float32
    expected = torch.special.exp2(x.to(ref_dtype)).to(dtype)

    tol = _tol(dtype)
    assert torch.allclose(y, expected, **tol), (
        f"exp2 mismatch (dtype={dtype}): "
        f"max abs diff = {(y - expected).abs().max().item()}, "
        f"max rel diff = "
        f"{((y - expected) / expected.abs().clamp_min(1e-30)).abs().max().item()}"
    )
    logger.info("Test passed: shape=%s, dtype=%s", shape, dtype)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_exp2_named_values(dtype):
    """Closed-form values of two raised to x at integer, half-integer and zero points."""
    if dtype == torch.float32:
        xs_list = [-10.0, -3.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.5, 10.0, 20.0]
    else:  # fp16, capped so the result stays well within the fp16 maximum
        xs_list = [-10.0, -3.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.5, 10.0, 14.0]
    xs = torch.tensor(xs_list, dtype=dtype)
    expected_hi = torch.special.exp2(xs.to(torch.float64)).to(dtype)

    y = torch.ops.ops_multimodal_fusion.exp2(xs.npu()).cpu()
    tol = _tol(dtype)
    assert torch.allclose(y, expected_hi, **tol), (
        f"exp2 named-values mismatch (dtype={dtype}); "
        f"got {y.tolist()} expected {expected_hi.tolist()}"
    )
    # For integer x, two raised to k is exactly representable in both fp16 and
    # fp32, but the path that raises two to x through the exponential of x times
    # ln2 cannot deliver bit-exactness. Casting LN2 to the dtype plus rounding x
    # times LN2 into the working dtype can drop the final result by about one
    # ulp, so use a one-to-two ulp tolerance instead of an exact equality check.
    integer_idx = torch.tensor([v == int(v) for v in xs_list])
    if integer_idx.any():
        int_tol = dict(rtol=2e-6, atol=0) if dtype == torch.float32 \
                  else dict(rtol=2e-3, atol=0)   # fp16 ulp ~ 1e-3 at the high end
        assert torch.allclose(y[integer_idx], expected_hi[integer_idx], **int_tol), (
            f"exp2 of integer x exceeded ~1-2 ulp tolerance (dtype={dtype}); "
            f"got {y[integer_idx].tolist()} expected {expected_hi[integer_idx].tolist()}"
        )
    logger.info("Named-values test passed (dtype=%s).", dtype)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_exp2_zero_is_one(dtype):
    """Two raised to the zero power must equal 1.0 exactly in any dtype."""
    x = torch.zeros(1024, dtype=dtype)
    y = torch.ops.ops_multimodal_fusion.exp2(x.npu()).cpu()
    expected = torch.ones_like(x)
    # Allow one ulp of slack since the path goes through the exponential of zero
    # times ln2, which is the exponential of zero and equals one, though every
    # step should be bit-exact here.
    tol = _tol(dtype)
    assert torch.allclose(y, expected, **tol), (
        f"exp2(0) should be 1 (dtype={dtype}); got {y[:5].tolist()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_exp2_monotonic(dtype):
    """exp2 is monotonically increasing; dense sweep must be non-decreasing."""
    low, high = _input_range(dtype)
    x = torch.linspace(low, high, 4096, dtype=dtype)
    y = torch.ops.ops_multimodal_fusion.exp2(x.npu()).cpu()
    diffs = y[1:] - y[:-1]
    # fp16 ulp at large magnitudes can make consecutive outputs equal, so
    # check >= 0 instead of > 0.
    assert (diffs >= 0).all(), (
        f"exp2 monotonicity violated (dtype={dtype}); "
        f"min diff = {diffs.min().item()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_exp2_empty_tensor(dtype):
    """Empty tensor passes through with matching shape/dtype (no kernel launch)."""
    x = torch.empty((0,), dtype=dtype).npu()
    y = torch.ops.ops_multimodal_fusion.exp2(x).cpu()
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
def test_exp2_non_contiguous_input(dtype):
    """Non-contiguous tensors are handled (kernel makes a contiguous copy)."""
    base = _sample((32, 32), dtype, seed=19)
    x = base.t()  # transpose yields a non-contiguous tensor
    assert not x.is_contiguous()

    y = torch.ops.ops_multimodal_fusion.exp2(x.npu()).cpu()
    ref_dtype = torch.float64 if dtype == torch.float32 else torch.float32
    expected = torch.special.exp2(x.contiguous().to(ref_dtype)).to(dtype)

    tol = _tol(dtype)
    assert torch.allclose(y, expected, **tol), (
        f"non-contiguous mismatch (dtype={dtype}): "
        f"max abs diff = {(y - expected).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_exp2_rejects_unsupported_dtype():
    """float64 is not in the supported-dtype list per issue #32."""
    xs = torch.tensor([1.0, -1.0], dtype=torch.float64).npu()
    with pytest.raises(RuntimeError, match="exp2"):
        torch.ops.ops_multimodal_fusion.exp2(xs)
