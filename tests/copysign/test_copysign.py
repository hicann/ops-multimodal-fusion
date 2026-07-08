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

if not hasattr(torch.ops.ops_multimodal_fusion, "copysign"):
    pytest.skip(
        "ops_multimodal_fusion.copysign not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_copysign_interface_exist():
    """The 'ops_multimodal_fusion.copysign' operator must be registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "copysign"),\
        "The 'copysign' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


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


def _sample(shape, dtype, seed=0, low=-100.0, high=100.0):
    g = torch.Generator().manual_seed(seed)
    return torch.empty(*shape, dtype=dtype).uniform_(low, high, generator=g)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_copysign_operator(shape, dtype):
    """copysign is bit-level; result must match torch.copysign exactly."""
    seed = abs(hash((shape, dtype))) % 10_000_000
    a = _sample(shape, dtype, seed=seed)
    b = _sample(shape, dtype, seed=seed + 1)

    a_npu = a.npu()
    b_npu = b.npu()
    c_npu = torch.ops.ops_multimodal_fusion.copysign(a_npu, b_npu)
    c = c_npu.cpu()

    expected = torch.copysign(a, b)

    assert c.dtype == dtype, f"dtype mismatch: {c.dtype} vs {dtype}"
    assert c.shape == a.shape, f"shape mismatch: {c.shape} vs {a.shape}"
    assert torch.equal(c, expected), (
        f"copysign mismatch (dtype={dtype}, shape={shape}): "
        f"max abs diff = {(c.float() - expected.float()).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_copysign_named_values(dtype):
    """Sign-of-b decides the output sign; magnitude always |a|."""
    pairs = [
        # a b expected
        (2.0, 3.0, 2.0),
        (-2.0, 3.0, 2.0),
        (2.0, -3.0, -2.0),
        (-2.0, -3.0, -2.0),
        (0.0, 1.0, 0.0),
        (5.0, 0.0, 5.0),
    ]
    a = torch.tensor([p[0] for p in pairs], dtype=dtype)
    b = torch.tensor([p[1] for p in pairs], dtype=dtype)
    expected = torch.tensor([p[2] for p in pairs], dtype=dtype)

    c = torch.ops.ops_multimodal_fusion.copysign(a.npu(), b.npu()).cpu()
    assert torch.equal(c, expected), (
        f"named-values mismatch (dtype={dtype}); got {c.tolist()} "
        f"expected {expected.tolist()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_copysign_signed_zero(dtype):
    """copysign respects ±0 on the sign source: copysign 2 -0.0 -2."""
    a = torch.tensor([2.0, 2.0, -3.0, -3.0], dtype=dtype)
    # +0.0 and -0.0 - the bit-level kernel must pick up the sign bit.
    b = torch.tensor([0.0, -0.0, 0.0, -0.0], dtype=dtype)
    expected = torch.copysign(a, b)
    # torch.copysign on this specific input: 2 -2 3 -3

    c = torch.ops.ops_multimodal_fusion.copysign(a.npu(), b.npu()).cpu()
    assert torch.equal(c, expected), (
        f"signed-zero mismatch (dtype={dtype}); got {c.tolist()} "
        f"expected {expected.tolist()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_copysign_infinities(dtype):
    """±inf magnitudes and ±inf sign sources must be bit-exact."""
    a = torch.tensor(
        [float("inf"), float("inf"), -float("inf"), -float("inf"), 5.0, 5.0],
        dtype=dtype,
    )
    b = torch.tensor(
        [1.0, -1.0, 1.0, -1.0, float("inf"), -float("inf")],
        dtype=dtype,
    )
    expected = torch.copysign(a, b)
    c = torch.ops.ops_multimodal_fusion.copysign(a.npu(), b.npu()).cpu()
    assert torch.equal(c, expected), (
        f"infinity mismatch (dtype={dtype}); got {c.tolist()} "
        f"expected {expected.tolist()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_copysign_idempotent_on_self(dtype):
    """copysign a a a for all a bit pattern preserved ."""
    a = _sample((4096,), dtype, seed=7, low=-50.0, high=50.0)
    c = torch.ops.ops_multimodal_fusion.copysign(a.npu(), a.npu()).cpu()
    assert torch.equal(c, a), (
        f"copysign(a,a) != a (dtype={dtype}); "
        f"max diff = {(c - a).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_copysign_magnitude_preserved(dtype):
    """|copysign a b | |a| for any b bit-exact."""
    a = _sample((4096,), dtype, seed=11)
    b = _sample((4096,), dtype, seed=12)
    c = torch.ops.ops_multimodal_fusion.copysign(a.npu(), b.npu()).cpu()
    assert torch.equal(c.abs(), a.abs()), (
        f"|copysign(a,b)| != |a| (dtype={dtype}); "
        f"max diff = {(c.abs() - a.abs()).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_copysign_empty_tensor(dtype):
    """Empty tensor passes through with matching shape/dtype (no kernel launch)."""
    a = torch.empty((0,), dtype=dtype).npu()
    b = torch.empty((0,), dtype=dtype).npu()
    c = torch.ops.ops_multimodal_fusion.copysign(a, b).cpu()
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
def test_copysign_non_contiguous_input(dtype):
    """Non-contiguous tensors are handled (kernel makes a contiguous copy)."""
    base_a = _sample((32, 32), dtype, seed=19)
    base_b = _sample((32, 32), dtype, seed=20)
    a = base_a.t()
    b = base_b.t()
    assert not a.is_contiguous() and not b.is_contiguous()

    c = torch.ops.ops_multimodal_fusion.copysign(a.npu(), b.npu()).cpu()
    expected = torch.copysign(a.contiguous(), b.contiguous())
    assert torch.equal(c, expected), (
        f"non-contiguous mismatch (dtype={dtype}): "
        f"max abs diff = {(c - expected).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_copysign_rejects_unsupported_dtype():
    """float64 is not in the supported-dtype list per issue #94."""
    a = torch.tensor([1.0, -1.0], dtype=torch.float64).npu()
    b = torch.tensor([1.0, 1.0], dtype=torch.float64).npu()
    with pytest.raises(RuntimeError, match="copysign"):
        torch.ops.ops_multimodal_fusion.copysign(a, b)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_copysign_rejects_shape_mismatch():
    """Mismatched input shapes must raise."""
    a = torch.zeros((4,), dtype=torch.float32).npu()
    b = torch.zeros((5,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError, match="copysign"):
        torch.ops.ops_multimodal_fusion.copysign(a, b)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_copysign_rejects_dtype_mismatch():
    """Mismatched input dtypes must raise."""
    a = torch.zeros((4,), dtype=torch.float32).npu()
    b = torch.zeros((4,), dtype=torch.float16).npu()
    with pytest.raises(RuntimeError, match="copysign"):
        torch.ops.ops_multimodal_fusion.copysign(a, b)
