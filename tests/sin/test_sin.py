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


import math
import torch
import torch_npu  # noqa: F401
import pytest
import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "sin"):
    pytest.skip(
        "ops_multimodal_fusion.sin not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_sin_interface_exist():
    """Test that the 'ops_multimodal_fusion.sin' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "sin"),\
        "The 'sin' operator is not registered in the 'torch.ops.ops_multimodal_fusion' namespace."


# shape dtype generator-tag - explicit tensors exercise special values
# 0 k pi k pi/2 ; randn tensors exercise general shapes/dtypes.
RANDN_SHAPES = [
    (1,), (3,), (10,), (100,), (1024,), (10000,),
    (10, 10), (32, 32), (100, 100), (10, 100), (100, 10), (256, 512),
    (5, 10, 15), (16, 32, 64), (32, 64, 128),
    (1, 3, 32, 32), (4, 3, 64, 64), (8, 3, 128, 128), (1000, 1000),
    (100000,), (1000000,), (2048, 2048), (4096, 1024),
    (64, 128, 256), (16, 16, 128, 128),
]
RANDN_DTYPES = [torch.float32, torch.float16]

# Special-value cases: tag builder
SPECIAL_CASES = [
    ("zeros", lambda: torch.zeros(10, dtype=torch.float32)),
    ("k_pi", lambda: torch.tensor(
        [-3.0 * math.pi, -2.0 * math.pi, -math.pi, 0.0, math.pi, 2.0 * math.pi, 3.0 * math.pi],
        dtype=torch.float32)),
    ("k_half_pi", lambda: torch.tensor(
        [-1.5 * math.pi, -0.5 * math.pi, 0.5 * math.pi, 1.5 * math.pi],
        dtype=torch.float32)),
]

CASES = (
    [("randn", shape, dtype) for shape in RANDN_SHAPES for dtype in RANDN_DTYPES]
    + [("special", tag, builder) for tag, builder in SPECIAL_CASES]
)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("kind,arg1,arg2", CASES)
def test_sin_operator(kind, arg1, arg2):
    """Test the sin operator across shapes, dtypes, and special values."""
    if kind == "randn":
        shape, dtype = arg1, arg2
        a = torch.randn(*shape, dtype=dtype)
        label = f"shape={shape}, dtype={dtype}"
    else:
        a = arg2()  # builder
        label = f"special={arg1}"

    expected = torch.sin(a)
    result = torch.ops.ops_multimodal_fusion.sin(a.npu()).cpu()

    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2),\
        f"Sin failed for {label}. Max diff: {torch.max(torch.abs(result - expected)):.6f}"
