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

if not hasattr(torch.ops.ops_multimodal_fusion, "mean"):
    pytest.skip(
        "ops_multimodal_fusion.mean not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_mean_interface_exist():
    """Test that the 'ops_multimodal_fusion.mean' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "mean"),\
        "The 'mean' operator is not registered in the 'torch.ops.ops_multimodal_fusion' namespace."


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
    # Non-aligned innerSize (innerSize not multiple of 8 for fp32 / 16 for fp16)
    ((8, 7), 0),         # innerSize 7 non-aligned
    ((16, 13), 0),       # innerSize 13
    ((32, 17), 0),       # innerSize 17
    ((10, 5, 9), 0),     # innerSize 5 9 45 to outerSize 10 dimSize 1? No: dim 0 to outerSize 1 dimSize 10 innerSize 45
    ((10, 5, 9), 1),     # dim 1 to outerSize 10 dimSize 5 innerSize 9
    ((4, 3, 7, 11), 2),  # dim 2 to outerSize 12 dimSize 7 innerSize 11
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

FLOAT_DTYPES = [
    torch.float32,
    torch.float16,
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape,dim", SHAPE_DIM_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("keepdim", [False, True])
def test_mean_float(shape, dim, dtype, keepdim):
    """Test mean operator for float types against PyTorch CPU reference."""
    a = torch.randn(*shape, dtype=dtype)

    expected = torch.mean(a, dim=dim, keepdim=keepdim)
    a_npu = a.npu()
    result_npu = torch.ops.ops_multimodal_fusion.mean(a_npu, dim, keepdim)
    result = result_npu.cpu()

    assert result.shape == expected.shape,\
        f"Shape mismatch: got {result.shape}, expected {expected.shape}"
    rtol = 1e-1 if dtype == torch.float16 else 1e-2
    atol = 1e-1 if dtype == torch.float16 else 1e-2
    assert torch.allclose(result, expected, rtol=rtol, atol=atol),\
        f"Mean failed for shape {shape}, dim {dim}, dtype {dtype}, keepdim {keepdim}. " \
        f"Max diff: {torch.max(torch.abs(result.float() - expected.float())):.6f}"
