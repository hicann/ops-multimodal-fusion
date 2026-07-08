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

if not hasattr(torch.ops.ops_multimodal_fusion, "log_add_exp2"):
    pytest.skip(
        "ops_multimodal_fusion.log_add_exp2 not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_log_add_exp2_interface_exist():
    """The 'ops_multimodal_fusion.log_add_exp2' operator must be registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "log_add_exp2"),\
        "The 'log_add_exp2' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


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
    # Large shapes that exceed UB capacity, forcing multi-tile processing.
    (100000,),
    (1000000,),
    (2048, 2048),
    (64, 128, 256),
]

DTYPES = [torch.float32, torch.float16]


def _input_range(dtype):
    """Per-dtype safe input range.

    The stable form keeps interior exponents at or below zero regardless of
    input magnitude, so the output stays within the dtype for any representable
    inputs. Use a moderate range for sane reference precision.
    """
    if dtype == torch.float32:
        return -30.0, 30.0
    return -10.0, 10.0


def _tol(dtype):
    """Tolerance for vector comparison.

    For float32 the longer reduction chain accumulates a few ulps versus the
    reference, which uses double precision internally on CPU. For float16 the
    exponential and logarithm intrinsics give roughly three significant decimals.
    """
    if dtype == torch.float32:
        return dict(rtol=1e-5, atol=1e-5)
    return dict(rtol=2e-2, atol=5e-3)


def _sample(shape, dtype, seed=0):
    low, high = _input_range(dtype)
    g = torch.Generator().manual_seed(seed)
    return torch.empty(*shape, dtype=dtype).uniform_(low, high, generator=g)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_log_add_exp2_operator(shape, dtype):
    """Compare NPU log_add_exp2 against torch.logaddexp2 on finite inputs."""
    seed = abs(hash((shape, dtype))) % 10_000_000
    a = _sample(shape, dtype, seed=seed)
    b = _sample(shape, dtype, seed=seed + 1)

    a_npu = a.npu()
    b_npu = b.npu()
    c_npu = torch.ops.ops_multimodal_fusion.log_add_exp2(a_npu, b_npu)
    c = c_npu.cpu()

    assert c.dtype == dtype, f"dtype mismatch: {c.dtype} vs {dtype}"
    assert c.shape == a.shape, f"shape mismatch: {c.shape} vs {a.shape}"

    # Reference in float64 for fp32, fp32 for fp16.
    ref_dtype = torch.float64 if dtype == torch.float32 else torch.float32
    expected = torch.logaddexp2(a.to(ref_dtype), b.to(ref_dtype)).to(dtype)

    tol = _tol(dtype)
    assert torch.allclose(c, expected, **tol), (
        f"log_add_exp2 mismatch (dtype={dtype}, shape={shape}): "
        f"max abs diff = {(c - expected).abs().max().item()}, "
        f"max rel diff = "
        f"{((c - expected) / expected.abs().clamp_min(1e-30)).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_log_add_exp2_named_values(dtype):
    """Closed-form reference values for a few hand-picked input pairs."""
    pairs = [
        (0.0, 0.0, 1.0),                        # equal small inputs
        (1.0, 1.0, 2.0),                        # equal inputs raise by one
        (-3.0, -3.0, -2.0),                     # equal negative inputs
        (2.0, 4.0, 4.321928094887363),          # dominated by the larger input
        (-1.0, 1.0, 1.321928094887363),         # mixed-sign inputs
    ]
    a_list = torch.tensor([p[0] for p in pairs], dtype=dtype)
    b_list = torch.tensor([p[1] for p in pairs], dtype=dtype)
    expected = torch.tensor([p[2] for p in pairs], dtype=dtype)

    c = torch.ops.ops_multimodal_fusion.log_add_exp2(a_list.npu(), b_list.npu()).cpu()

    tol = _tol(dtype)
    assert torch.allclose(c, expected, **tol), (
        f"named-values mismatch (dtype={dtype}); got {c.tolist()} "
        f"expected {expected.tolist()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_log_add_exp2_symmetric(dtype):
    """The operator is symmetric in its two inputs and must match bit-exactly."""
    a = _sample((2048,), dtype, seed=42)
    b = _sample((2048,), dtype, seed=43)

    c_ab = torch.ops.ops_multimodal_fusion.log_add_exp2(a.npu(), b.npu()).cpu()
    c_ba = torch.ops.ops_multimodal_fusion.log_add_exp2(b.npu(), a.npu()).cpu()

    assert torch.equal(c_ab, c_ba), (
        f"symmetry violated (dtype={dtype}); "
        f"max diff = {(c_ab - c_ba).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_log_add_exp2_large_gap(dtype):
    """When one input greatly exceeds the other the result approaches the larger
    input, and the stable form must not overflow.
    """
    # For float32 a naive exponential of the larger input would overflow, but the
    # stable form stays in range. For float16 the chosen inputs stay representable.
    if dtype == torch.float32:
        a = torch.full((1024,), 60.0, dtype=dtype)
        b = torch.full((1024,), -60.0, dtype=dtype)
    else:
        a = torch.full((1024,), 14.0, dtype=dtype)
        b = torch.full((1024,), -14.0, dtype=dtype)

    c = torch.ops.ops_multimodal_fusion.log_add_exp2(a.npu(), b.npu()).cpu()

    # The smaller term contributes essentially nothing, so the result should
    # equal the larger input within tolerance.
    tol = _tol(dtype)
    assert torch.allclose(c, a, **tol), (
        f"large-gap stability violated (dtype={dtype}); "
        f"max diff vs a = {(c - a).abs().max().item()}"
    )
    assert torch.isfinite(c).all(),\
        f"non-finite output for large-gap inputs (dtype={dtype})"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_log_add_exp2_infinities(dtype):
    """Infinite inputs must match the reference instead of producing spurious NaNs."""
    a = torch.tensor(
        [float("inf"), float("inf"), 1.0, -float("inf"), -float("inf")],
        dtype=dtype,
    )
    b = torch.tensor(
        [1.0, float("inf"), -float("inf"), -float("inf"), 2.0],
        dtype=dtype,
    )
    expected = torch.logaddexp2(a.to(torch.float32), b.to(torch.float32)).to(dtype)

    c = torch.ops.ops_multimodal_fusion.log_add_exp2(a.npu(), b.npu()).cpu()
    assert torch.equal(c, expected), (
        f"infinity handling mismatch (dtype={dtype}); got {c.tolist()} "
        f"expected {expected.tolist()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_log_add_exp2_empty_tensor(dtype):
    """Empty tensor passes through with matching shape/dtype (no kernel launch)."""
    a = torch.empty((0,), dtype=dtype).npu()
    b = torch.empty((0,), dtype=dtype).npu()
    c = torch.ops.ops_multimodal_fusion.log_add_exp2(a, b).cpu()
    assert c.shape == (0,)
    assert c.dtype == dtype


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
def test_log_add_exp2_non_contiguous_input(dtype):
    """Non-contiguous tensors are handled (kernel makes a contiguous copy)."""
    base_a = _sample((32, 32), dtype, seed=19)
    base_b = _sample((32, 32), dtype, seed=20)
    a = base_a.t()
    b = base_b.t()
    assert not a.is_contiguous() and not b.is_contiguous()

    c = torch.ops.ops_multimodal_fusion.log_add_exp2(a.npu(), b.npu()).cpu()
    ref_dtype = torch.float64 if dtype == torch.float32 else torch.float32
    expected = torch.logaddexp2(
        a.contiguous().to(ref_dtype), b.contiguous().to(ref_dtype)
    ).to(dtype)

    tol = _tol(dtype)
    assert torch.allclose(c, expected, **tol), (
        f"non-contiguous mismatch (dtype={dtype}): "
        f"max abs diff = {(c - expected).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_log_add_exp2_rejects_unsupported_dtype():
    """float64 is not in the supported-dtype list per issue #91."""
    a = torch.tensor([1.0, -1.0], dtype=torch.float64).npu()
    b = torch.tensor([1.0, 1.0], dtype=torch.float64).npu()
    with pytest.raises(RuntimeError, match="log_add_exp2"):
        torch.ops.ops_multimodal_fusion.log_add_exp2(a, b)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_log_add_exp2_rejects_shape_mismatch():
    """Mismatched input shapes must raise."""
    a = torch.zeros((4,), dtype=torch.float32).npu()
    b = torch.zeros((5,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError, match="log_add_exp2"):
        torch.ops.ops_multimodal_fusion.log_add_exp2(a, b)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_log_add_exp2_rejects_dtype_mismatch():
    """Mismatched input dtypes must raise."""
    a = torch.zeros((4,), dtype=torch.float32).npu()
    b = torch.zeros((4,), dtype=torch.float16).npu()
    with pytest.raises(RuntimeError, match="log_add_exp2"):
        torch.ops.ops_multimodal_fusion.log_add_exp2(a, b)
