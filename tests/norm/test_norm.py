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

if not hasattr(torch.ops.ops_multimodal_fusion, "norm"):
    pytest.skip(
        "ops_multimodal_fusion.norm not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_norm_interface_exist():
    """Test that the 'ops_multimodal_fusion.norm' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "norm"),\
        "The 'norm' operator is not registered in the 'torch.ops.ops_multimodal_fusion' namespace."


SHAPE_DIM_CASES = [
    # 1D
    ((10,), 0),
    ((100,), 0),
    ((1024,), 0),
    ((10000,), 0),
    # 2D
    ((10, 10), 0),
    ((10, 10), 1),
    ((32, 64), 0),
    ((32, 64), 1),
    ((100, 100), 0),
    ((100, 100), 1),
    ((256, 512), 0),
    ((256, 512), 1),
    # 3D
    ((5, 10, 15), 0),
    ((5, 10, 15), 1),
    ((5, 10, 15), 2),
    ((16, 32, 64), 0),
    ((16, 32, 64), 1),
    ((16, 32, 64), 2),
    # 4D
    ((4, 3, 64, 64), 0),
    ((4, 3, 64, 64), 1),
    ((4, 3, 64, 64), 2),
    ((4, 3, 64, 64), 3),
    # Non-aligned innerSize (not multiple of 8 for fp32)
    ((8, 7), 0),         # innerSize 7
    ((16, 13), 0),       # innerSize 13
    ((32, 17), 0),       # innerSize 17
    ((10, 5, 9), 1),     # dim 1 to innerSize 9
    ((4, 3, 7, 11), 2),  # dim 2 to innerSize 11
    # Large shapes that exceed UB capacity, forcing multi-tile processing
    ((100000,), 0),
    ((1024, 1024), 0),
    ((1024, 1024), 1),
    ((256, 4096), 0),
    ((256, 4096), 1),
    ((2048, 2048), 0),
    ((2048, 2048), 1),
    ((64, 128, 256), 0),
    ((64, 128, 256), 1),
    ((64, 128, 256), 2),
    ((8, 16, 256, 256), 0),
    ((8, 16, 256, 256), 2),
    ((8, 16, 256, 256), 3),
]

# p values: 1.0 L1 special branch 2.0 L2 special branch 3.0 general branch
# 0.5 (fractional general branch), float('inf') (special max branch)
P_VALUES = [0.5, 1.0, 2.0, 3.0, float('inf')]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape,dim", SHAPE_DIM_CASES)
@pytest.mark.parametrize("p", P_VALUES)
@pytest.mark.parametrize("keepdim", [False, True])
def test_norm_float32(shape, dim, p, keepdim):
    """Test norm operator for float32 against PyTorch CPU reference."""
    a = torch.randn(*shape, dtype=torch.float32)

    expected = torch.norm(a, p=p, dim=dim, keepdim=keepdim)
    a_npu = a.npu()
    result_npu = torch.ops.ops_multimodal_fusion.norm(a_npu, p, dim, keepdim)
    result = result_npu.cpu()

    assert result.shape == expected.shape,\
        f"Shape mismatch: got {result.shape}, expected {expected.shape}"
    rtol = 1e-3
    atol = 1e-3
    assert torch.allclose(result, expected, rtol=rtol, atol=atol),\
        f"Norm failed for shape {shape}, dim {dim}, p {p}, keepdim {keepdim}. " \
        f"Max diff: {torch.max(torch.abs(result - expected)):.6f}"
