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

if not hasattr(torch.ops.ops_multimodal_fusion, "any"):
    pytest.skip(
        "ops_multimodal_fusion.any not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_any_interface_exist():
    """Test that the 'ops_multimodal_fusion.any' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "any"),\
        "The 'any' operator is not registered in the 'torch.ops.ops_multimodal_fusion' namespace."


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

DTYPES = [torch.float32, torch.float16, torch.int32, torch.bool]


def _make_input(shape, dtype):
    if dtype == torch.bool:
        return torch.rand(*shape) < 0.5
    if dtype == torch.int32:
        a = torch.randint(-100, 100, shape, dtype=torch.int32)
        mask = (torch.rand(*shape) < 0.5).int()
        return a * mask
    a = torch.randn(*shape, dtype=dtype)
    mask = (torch.rand(*shape) < 0.5).to(dtype)
    return a * mask


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape,dim", SHAPE_DIM_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("keepdim", [False, True])
def test_any_operator(shape, dim, dtype, keepdim):
    """Test any operator against PyTorch CPU reference across dtypes and shapes."""
    a = _make_input(shape, dtype)
    expected = torch.any(a, dim=dim, keepdim=keepdim)
    result = torch.ops.ops_multimodal_fusion.any(a.npu(), dim, keepdim).cpu()

    assert result.dtype == torch.bool, f"Expected bool output, got {result.dtype}"
    assert result.shape == expected.shape,\
        f"Shape mismatch: got {result.shape}, expected {expected.shape}"
    assert torch.equal(result, expected),\
        f"Any failed for shape {shape}, dim {dim}, dtype={dtype}, keepdim={keepdim}. " \
        f"Mismatches: {(result != expected).sum().item()}"
