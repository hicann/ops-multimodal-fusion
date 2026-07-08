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

if not hasattr(torch.ops.ops_multimodal_fusion, "lcm"):
    pytest.skip(
        "ops_multimodal_fusion.lcm not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_lcm_interface_exist():
    """The 'ops_multimodal_fusion.lcm' operator must be registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "lcm"),\
        "The 'lcm' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


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
    # Sizes large enough to span multiple tiles per core. Kept smaller than
    # the float-arith ops because each lcm element runs a scalar GCD loop
    # O log max |a| |b| work - keeps the test suite reasonable.
    (50000,),
    (256, 256),
    (64, 128, 32),
]

DTYPES = [torch.int32]


def _sample(shape, dtype, seed=0, low=-200, high=200):
    """Sample modest integers in [low, high) so the scalar GCD loop stays fast."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(low, high, shape, dtype=dtype, generator=g)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_lcm_operator(shape, dtype):
    """lcm is exact integer arithmetic; result must match torch.lcm exactly."""
    seed = abs(hash((shape, dtype))) % 10_000_000
    a = _sample(shape, dtype, seed=seed)
    b = _sample(shape, dtype, seed=seed + 1)

    a_npu = a.npu()
    b_npu = b.npu()
    c_npu = torch.ops.ops_multimodal_fusion.lcm(a_npu, b_npu)
    c = c_npu.cpu()

    expected = torch.lcm(a, b)

    assert c.dtype == dtype, f"dtype mismatch: {c.dtype} vs {dtype}"
    assert c.shape == a.shape, f"shape mismatch: {c.shape} vs {a.shape}"
    assert torch.equal(c, expected), (
        f"lcm mismatch (dtype={dtype}, shape={shape}): "
        f"first 5 mismatches at "
        f"{(c != expected).nonzero(as_tuple=False)[:5].tolist()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_lcm_named_values():
    """A handful of textbook cases verifiable by inspection."""
    pairs = [
        # a b expected_lcm
        (4, 6, 12),
        (12, 18, 36),
        (7, 13, 91),     # coprime
        (5, 5, 5),
        (1, 100, 100),
        (-4, 6, 12),     # sign insensitive
        (4, -6, 12),
        (-4, -6, 12),
        (0, 5, 0),      # zero absorbs
        (5, 0, 0),
        (0, 0, 0),
        (15, 25, 75),
    ]
    a = torch.tensor([p[0] for p in pairs], dtype=torch.int32)
    b = torch.tensor([p[1] for p in pairs], dtype=torch.int32)
    expected = torch.tensor([p[2] for p in pairs], dtype=torch.int32)

    c = torch.ops.ops_multimodal_fusion.lcm(a.npu(), b.npu()).cpu()
    assert torch.equal(c, expected), (
        f"named-values mismatch; got {c.tolist()} expected {expected.tolist()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_lcm_zeros():
    """lcm 0 x lcm x 0 0 for every x."""
    x = _sample((1024,), torch.int32, seed=33)
    zeros = torch.zeros_like(x)

    c1 = torch.ops.ops_multimodal_fusion.lcm(zeros.npu(), x.npu()).cpu()
    c2 = torch.ops.ops_multimodal_fusion.lcm(x.npu(), zeros.npu()).cpu()

    assert torch.equal(c1, zeros)
    assert torch.equal(c2, zeros)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_lcm_idempotent_on_self():
    """lcm a a |a| note: lcm result is non-negative ."""
    a = _sample((4096,), torch.int32, seed=7)
    c = torch.ops.ops_multimodal_fusion.lcm(a.npu(), a.npu()).cpu()
    expected = a.abs()
    assert torch.equal(c, expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_lcm_with_one():
    """lcm 1 x |x|."""
    x = _sample((4096,), torch.int32, seed=42)
    ones = torch.ones_like(x)
    c = torch.ops.ops_multimodal_fusion.lcm(ones.npu(), x.npu()).cpu()
    assert torch.equal(c, x.abs())


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_lcm_symmetric():
    """lcm a b lcm b a ."""
    a = _sample((2048,), torch.int32, seed=51)
    b = _sample((2048,), torch.int32, seed=52)

    c_ab = torch.ops.ops_multimodal_fusion.lcm(a.npu(), b.npu()).cpu()
    c_ba = torch.ops.ops_multimodal_fusion.lcm(b.npu(), a.npu()).cpu()
    assert torch.equal(c_ab, c_ba)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_lcm_negative_inputs_yield_nonnegative():
    """torch.lcm always returns a non-negative integer."""
    a = _sample((4096,), torch.int32, seed=61, low=-500, high=0)
    b = _sample((4096,), torch.int32, seed=62, low=-500, high=0)
    c = torch.ops.ops_multimodal_fusion.lcm(a.npu(), b.npu()).cpu()
    assert (c >= 0).all(), "lcm must be non-negative"
    expected = torch.lcm(a, b)
    assert torch.equal(c, expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_lcm_int32_boundaries_and_overflow():
    """INT32_MIN and overflowing products must match torch.lcm without UB."""
    min_i32 = -2147483648
    max_i32 = 2147483647
    a = torch.tensor(
        [min_i32, min_i32, min_i32, max_i32, 46341, 50000, -50000, 0],
        dtype=torch.int32,
    )
    b = torch.tensor(
        [1, -1, min_i32, 2, 46343, 50000, 50021, min_i32],
        dtype=torch.int32,
    )
    expected = torch.lcm(a, b)

    c = torch.ops.ops_multimodal_fusion.lcm(a.npu(), b.npu()).cpu()
    assert torch.equal(c, expected), (
        f"boundary/overflow mismatch; got {c.tolist()} expected {expected.tolist()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_lcm_empty_tensor():
    """Empty tensor passes through with matching shape/dtype (no kernel launch)."""
    a = torch.empty((0,), dtype=torch.int32).npu()
    b = torch.empty((0,), dtype=torch.int32).npu()
    c = torch.ops.ops_multimodal_fusion.lcm(a, b).cpu()
    assert c.shape == (0,)
    assert c.dtype == torch.int32


@pytest.mark.skip(
    reason=(
        "torch_npu in the current CANN release lacks D2D strided copy support "
        "(aclnnInplaceCopy fails with error 561103 for any non-contiguous NPU "
        "tensor, regardless of dtype). The lcm kernel itself only operates on "
        "contiguous int32 buffers; callers that hold a transposed/strided int32 "
        "tensor must materialize it themselves before invoking lcm. "
        "Re-enable this test once the underlying torch_npu strided D2D path "
        "is implemented for int32."
    )
)


def test_lcm_non_contiguous_input():
    """Non-contiguous tensors are handled (kernel makes a contiguous copy)."""
    base_a = _sample((32, 32), torch.int32, seed=19)
    base_b = _sample((32, 32), torch.int32, seed=20)

    a = base_a.npu().t()
    b = base_b.npu().t()
    assert not a.is_contiguous() and not b.is_contiguous()

    c = torch.ops.ops_multimodal_fusion.lcm(a, b).cpu()
    expected = torch.lcm(base_a.t().contiguous(), base_b.t().contiguous())
    assert torch.equal(c, expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_lcm_rejects_unsupported_dtype():
    """Float dtypes and int64/int8 must be rejected per current scope."""
    for unsupported in [torch.float32, torch.float16, torch.int64, torch.int16]:
        a = torch.zeros((4,), dtype=unsupported).npu()
        b = torch.ones((4,), dtype=unsupported).npu()
        with pytest.raises(RuntimeError, match="lcm"):
            torch.ops.ops_multimodal_fusion.lcm(a, b)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_lcm_rejects_shape_mismatch():
    """Mismatched input shapes must raise."""
    a = torch.zeros((4,), dtype=torch.int32).npu()
    b = torch.zeros((5,), dtype=torch.int32).npu()
    with pytest.raises(RuntimeError, match="lcm"):
        torch.ops.ops_multimodal_fusion.lcm(a, b)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_lcm_rejects_dtype_mismatch():
    """Mismatched input dtypes must raise."""
    a = torch.zeros((4,), dtype=torch.int32).npu()
    b = torch.zeros((4,), dtype=torch.int64).npu()
    with pytest.raises(RuntimeError, match="lcm"):
        torch.ops.ops_multimodal_fusion.lcm(a, b)
