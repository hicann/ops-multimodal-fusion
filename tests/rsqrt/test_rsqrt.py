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

if not hasattr(torch.ops.ops_multimodal_fusion, "rsqrt"):
    pytest.skip(
        "ops_multimodal_fusion.rsqrt not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_rsqrt_interface_exist():
    """Test that the 'ops_multimodal_fusion.rsqrt' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "rsqrt"),\
        "The 'rsqrt' operator is not registered in the 'torch.ops.ops_multimodal_fusion' namespace."


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
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rsqrt_operator(shape, dtype):
    """
    Test the rsqrt operator with various shapes and data types.

    Parameters:
        shape: Tensor shape
        dtype: Data type
    """
    # Use positive values only since rsqrt x 1/sqrt x requires x > 0
    a = torch.rand(*shape, dtype=dtype).clamp(min=0.01)

    expected = torch.rsqrt(a)
    a_npu = a.npu()
    result_npu = torch.ops.ops_multimodal_fusion.rsqrt(a_npu)
    result = result_npu.cpu()

    # AscendC Rsqrt is a hardware approximation with ~1% relative error
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2),\
        f"Rsqrt failed for shape {shape}, dtype {dtype}. " \
        f"Max diff: {torch.max(torch.abs(result - expected)):.6f}"
