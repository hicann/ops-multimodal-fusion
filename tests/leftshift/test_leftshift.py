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

if not hasattr(torch.ops.ops_multimodal_fusion, "leftshift"):
    pytest.skip(
        "ops_multimodal_fusion.leftshift not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_leftshift_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "leftshift")


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
    (256, 512),
    (5, 10, 15),
    (16, 32, 64),
    (32, 64, 128),
    (4, 3, 64, 64),
    (1000, 1000),
    (100000,),
    (1000000,),
    (2048, 2048),
    (64, 128, 256),
]

# Cover typical positions across the 32-bit range (including the sign-bit shift).
SHIFTS = [0, 1, 4, 7, 16, 24, 30, 31]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("shift", SHIFTS)
def test_leftshift(shape, shift):
    """Compare NPU leftshift against torch.bitwise_left_shift on int32 inputs."""
    # Use a full-range int32 distribution so the wrap-around / sign-flip paths
    # are exercised (small shifts of large values overflow; large shifts of
    # any value but 0 wrap). Keep the distribution wide but not at INT_MAX to
    # avoid CPU-side UB quirks from the reference.
    lo, hi = -(1 << 20), (1 << 20)
    a = torch.randint(lo, hi, shape, dtype=torch.int32)

    expected = torch.bitwise_left_shift(a, shift)
    result = torch.ops.ops_multimodal_fusion.leftshift(a.npu(), shift).cpu()

    assert result.dtype == torch.int32
    assert torch.equal(result, expected),\
        f"LeftShift failed for shape {shape}, shift {shift}. " \
        f"Mismatches: {(result != expected).sum().item()} / {expected.numel()}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_leftshift_zero_shift():
    """shift 0 must be a no-op."""
    a = torch.randint(-1_000_000, 1_000_000, (2048,), dtype=torch.int32)
    result = torch.ops.ops_multimodal_fusion.leftshift(a.npu(), 0).cpu()
    assert torch.equal(result, a)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_leftshift_sign_bit():
    """Shifting 1 by 31 produces the int32 sign-bit pattern (INT_MIN)."""
    a = torch.ones((128,), dtype=torch.int32)
    result = torch.ops.ops_multimodal_fusion.leftshift(a.npu(), 31).cpu()
    expected = torch.full((128,), -(2 ** 31), dtype=torch.int32)
    assert torch.equal(result, expected), f"Got {result[:4]}, expected {expected[:4]}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("bad_shift", [-1, 32, 100])
def test_leftshift_out_of_range_rejected(bad_shift):
    """Shift amounts outside 0 31 must be rejected by the dispatch layer."""
    a = torch.zeros((32,), dtype=torch.int32)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.leftshift(a.npu(), bad_shift)
