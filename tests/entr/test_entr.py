#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging

import pytest
import torch
import torch_npu

import ops_multimodal_fusion

if not hasattr(torch.ops.ops_multimodal_fusion, "entr"):
    pytest.skip(
        "ops_multimodal_fusion.entr not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_entr_interface_exist():
    """Test that the 'ops_multimodal_fusion.entr' operator is registered in torch.ops."""
    logging.info(torch.ops.ops_multimodal_fusion.entr)
    assert hasattr(torch.ops.ops_multimodal_fusion, "entr"), \
        "The 'entr' operator is not registered in the 'torch.ops.ops_multimodal_fusion' namespace."


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
        return dict(rtol=5e-3, atol=5e-3)
    return dict(rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_entr_positive(shape, dtype):
    """Positive-only inputs — main -x*ln(x) path."""
    # Keep inputs in a range where fp16 ln is well-conditioned.
    a = torch.rand(*shape, dtype=dtype) * 0.9 + 0.05  # [0.05, 0.95]

    expected = torch.special.entr(a)
    result = torch.ops.ops_multimodal_fusion.entr(a.npu()).cpu()

    tol = _tolerances(dtype)
    assert torch.allclose(result, expected, **tol), \
        f"Entr (positive) failed for shape {shape}, dtype {dtype}. " \
        f"Max diff: {torch.max(torch.abs(result - expected)):.6f}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_entr_zero(dtype):
    """entr(0) must equal 0 exactly."""
    a = torch.zeros(128, dtype=dtype)
    expected = torch.special.entr(a)   # all zeros
    result = torch.ops.ops_multimodal_fusion.entr(a.npu()).cpu()

    assert torch.equal(result, expected), \
        f"Entr(0) expected all zeros but got {result}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_entr_negative(dtype):
    """entr(x) must equal -inf for x < 0."""
    a = -torch.rand(128, dtype=dtype) - 0.01  # all strictly negative
    expected = torch.special.entr(a)          # all -inf
    result = torch.ops.ops_multimodal_fusion.entr(a.npu()).cpu()

    assert torch.all(torch.isneginf(result)), \
        f"Entr(negative) expected -inf everywhere, got {result}"
    # Ensure every element matches the reference (all -inf).
    assert torch.equal(result, expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", [(1024,), (32, 32), (4, 3, 64, 64)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_entr_mixed(shape, dtype):
    """Mixed-sign inputs exercise all three branches in one tensor.

    The Relu-based sign mask has a narrow transition window around zero
    (O(1/SHARPNESS) = O(1e-6) for fp32, O(1e-4) for fp16); ensure non-zero
    inputs stay well outside it so the x<0 branch evaluates to -inf exactly.
    """
    gap = 1e-2 if dtype == torch.float16 else 1e-5
    a = torch.empty(*shape, dtype=dtype).uniform_(-1.0, 1.0)
    # Push any tiny-magnitude values away from 0 without flipping sign.
    too_small = a.abs() < gap
    a = torch.where(too_small, torch.full_like(a, gap) * a.sign(), a)
    flat = a.view(-1)
    flat[:: max(1, flat.numel() // 32)] = 0.0  # scatter a few exact zeros

    expected = torch.special.entr(a)
    result = torch.ops.ops_multimodal_fusion.entr(a.npu()).cpu()

    # Check -inf positions agree.
    assert torch.equal(torch.isneginf(result), torch.isneginf(expected)), \
        f"-inf positions diverge for shape {shape}, dtype {dtype}"

    # Compare finite positions within tolerance.
    finite_mask = torch.isfinite(expected)
    tol = _tolerances(dtype)
    assert torch.allclose(result[finite_mask], expected[finite_mask], **tol), \
        f"Entr (mixed) finite-region mismatch for shape {shape}, dtype {dtype}. " \
        f"Max diff: {torch.max(torch.abs(result[finite_mask] - expected[finite_mask])):.6f}"
