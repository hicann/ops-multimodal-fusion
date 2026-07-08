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
import torch.nn.functional as F
import pytest
import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "gelu"):
    pytest.skip(
        "ops_multimodal_fusion.gelu not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_gelu_interface_exist():
    """
    Test that the 'ascend_ops.gelu' operator is present in torch.ops.
    """
    assert hasattr(torch.ops.ops_multimodal_fusion, "gelu"),\
        "The 'gelu' operator is not registered in the 'torch.ops.ops_multimodal_fusion.' namespace."


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
    # Non-aligned element counts (not multiple of 8 for fp32)
    (7,),            # single tile, non-aligned
    (33,),           # just over 32
    (127,),          # one less than power-of-2
    (1023,),         # just under 1024
    (17, 19),        # non-aligned 2D
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
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gelu_operator(shape, dtype):
    """
    Test the functionality of the gelu operator.

    GeLU uses tanh approximation:
    GeLU x x / 1 + exp x / 0.044715 + x^3 -1.595769121 0.044715
    """
    a = torch.randn(*shape, dtype=dtype)

    expected = F.gelu(a, approximate='tanh')

    a_npu = a.npu()
    result_npu = torch.ops.ops_multimodal_fusion.gelu(a_npu)
    result = result_npu.cpu()

    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4),\
        f"Gelu failed for shape {shape}, dtype {dtype}. " \
        f"Max diff: {torch.max(torch.abs(result - expected)):.6f}"
