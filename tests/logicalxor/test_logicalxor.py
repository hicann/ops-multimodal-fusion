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

if not hasattr(torch.ops.ops_multimodal_fusion, "logicalxor"):
    pytest.skip(
        "ops_multimodal_fusion.logicalxor not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_logicalxor_interface_exist():
    """Test that the 'ops_multimodal_fusion.logicalxor' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "logicalxor"),\
        "The 'logicalxor' operator is not registered."


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
    (100000,),
    (1000000,),
    (2048, 2048),
    (4096, 1024),
    (64, 128, 256),
    (16, 16, 128, 128),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
def test_logicalxor(shape):
    """Compare NPU logicalxor against torch.logical_xor on random bool inputs."""
    a = torch.randint(0, 2, shape, dtype=torch.bool)
    b = torch.randint(0, 2, shape, dtype=torch.bool)

    expected = torch.logical_xor(a, b)
    result = torch.ops.ops_multimodal_fusion.logicalxor(a.npu(), b.npu()).cpu()

    assert result.dtype == torch.bool, f"Expected bool output, got {result.dtype}"
    assert torch.equal(result, expected),\
        f"LogicalXor mismatch for shape {shape}. " \
        f"Mismatches: {(result != expected).sum().item()} / {expected.numel()}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_logicalxor_truth_table():
    """Exhaustively verify the 2×2 truth table with a tiny explicit tensor."""
    a = torch.tensor([False, False, True, True], dtype=torch.bool)
    b = torch.tensor([False, True, False, True], dtype=torch.bool)
    expected = torch.tensor([False, True, True, False], dtype=torch.bool)

    result = torch.ops.ops_multimodal_fusion.logicalxor(a.npu(), b.npu()).cpu()
    assert torch.equal(result, expected),\
        f"Truth table mismatch: expected {expected}, got {result}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("value", [False, True])
def test_logicalxor_all_same(value):
    """a XOR a == False for any a (and XOR with all-false equals a)."""
    shape = (128,)
    a = torch.full(shape, value, dtype=torch.bool)

    # XOR with self -> all False
    r_self = torch.ops.ops_multimodal_fusion.logicalxor(a.npu(), a.npu()).cpu()
    assert torch.equal(r_self, torch.zeros_like(a)), f"a XOR a failed for value={value}"

    # XOR with all-False -> a unchanged
    zeros = torch.zeros_like(a)
    r_zero = torch.ops.ops_multimodal_fusion.logicalxor(a.npu(), zeros.npu()).cpu()
    assert torch.equal(r_zero, a), f"a XOR 0 failed for value={value}"
