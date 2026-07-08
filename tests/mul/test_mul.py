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

if not hasattr(torch.ops.ops_multimodal_fusion, "mul"):
    pytest.skip(
        "ops_multimodal_fusion.mul not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_mul_interface_exist():
    """Test that the 'ops_multimodal_fusion.mul' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "mul"),\
        "The 'mul' operator is not registered in the 'torch.ops.ops_multimodal_fusion' namespace."


SHAPES = [
    (1,),
    (3,),
    (10,),
    (100,),
    (1024,),
    (10000,),
    (10, 10),
    (32, 32),
    (100, 100),
    (10, 100),
    (100, 10),
    (256, 512),
    (5, 10, 15),
    (16, 32, 64),
    (32, 64, 128),
    (1, 3, 32, 32),
    (4, 3, 64, 64),
    (8, 3, 128, 128),
    (1000, 1000),
    # Large shapes that exceed UB capacity, forcing multi-tile processing
    (100000,),
    (1000000,),
    (2048, 2048),
    (4096, 1024),
    (64, 128, 256),
    (16, 16, 128, 128),
]

DTYPES = [
    torch.float32,
    torch.float16,
    torch.int32,
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_mul_operator(shape, dtype):
    """
    Test the mul operator with various shapes and data types.

    Parameters:
        shape: Tensor shape
        dtype: Data type
    """
    if dtype in [torch.int32]:
        a = torch.randint(-100, 100, shape, dtype=dtype)
        b = torch.randint(-100, 100, shape, dtype=dtype)
    else:
        a = torch.randn(*shape, dtype=dtype)
        b = torch.randn(*shape, dtype=dtype)

    expected = a * b
    a_npu = a.npu()
    b_npu = b.npu()
    result_npu = torch.ops.ops_multimodal_fusion.mul(a_npu, b_npu)
    result = result_npu.cpu()

    if dtype in [torch.int32]:
        assert torch.equal(result, expected),\
            f"Mul failed for shape {shape}, dtype {dtype}. " \
            f"Expected {expected}, but got {result}"
    else:
        assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4),\
            f"Mul failed for shape {shape}, dtype {dtype}. " \
            f"Max diff: {torch.max(torch.abs(result - expected)):.6f}"
