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

if not hasattr(torch.ops.ops_multimodal_fusion, "hypot"):
    pytest.skip(
        "ops_multimodal_fusion.hypot not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_hypot_interface_exist():
    """Test that the 'ops_multimodal_fusion.hypot' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "hypot"),\
        "The 'hypot' operator is not registered."


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

DTYPES = [torch.float32, torch.float16]


def _tolerances(dtype):
    if dtype == torch.float16:
        # sqrt of a² + b² for b close to a can lose relative precision;
        # fp16 has only ~10 bits of mantissa so allow ~5e-3.
        return dict(rtol=5e-3, atol=5e-3)
    return dict(rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_hypot_operator(shape, dtype):
    """Test hypot across shapes and floating dtypes against torch.hypot."""
    # Keep magnitudes moderate so x² + y² stays well away from fp16/fp32 overflow.
    lim = 4.0 if dtype == torch.float16 else 100.0
    a = torch.empty(*shape, dtype=dtype).uniform_(-lim, lim)
    b = torch.empty(*shape, dtype=dtype).uniform_(-lim, lim)

    expected = torch.hypot(a, b)
    result = torch.ops.ops_multimodal_fusion.hypot(a.npu(), b.npu()).cpu()

    tol = _tolerances(dtype)
    assert torch.allclose(result, expected, **tol),\
        f"Hypot failed for shape {shape}, dtype {dtype}. " \
        f"Max diff: {torch.max(torch.abs(result - expected)):.6f}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_hypot_zero(dtype):
    """hypot 0 0 0; hypot x 0 |x|; hypot 0 y |y|."""
    zeros = torch.zeros(64, dtype=dtype)
    x = torch.empty(64, dtype=dtype).uniform_(-3.0, 3.0)

    # hypot 0 0
    r_zz = torch.ops.ops_multimodal_fusion.hypot(zeros.npu(), zeros.npu()).cpu()
    assert torch.equal(r_zz, zeros), f"hypot(0,0) expected zeros, got {r_zz}"

    # hypot x 0 |x|
    r_xz = torch.ops.ops_multimodal_fusion.hypot(x.npu(), zeros.npu()).cpu()
    tol = _tolerances(dtype)
    assert torch.allclose(r_xz, x.abs(), **tol),\
        f"hypot(x,0) expected |x|, max diff {torch.max(torch.abs(r_xz - x.abs())):.6f}"

    # hypot 0 x |x|
    r_zx = torch.ops.ops_multimodal_fusion.hypot(zeros.npu(), x.npu()).cpu()
    assert torch.allclose(r_zx, x.abs(), **tol),\
        f"hypot(0,x) expected |x|, max diff {torch.max(torch.abs(r_zx - x.abs())):.6f}"
