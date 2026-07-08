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
import torch.nn.functional as F
import torch_npu  # noqa: F401
import pytest
import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "avg_pool2d"):
    pytest.skip(
        "ops_multimodal_fusion.avg_pool2d not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_avg_pool2d_interface_exist():
    """Test that the 'ops_multimodal_fusion.avg_pool2d' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "avg_pool2d"),\
        "The 'avg_pool2d' operator is not registered in the 'torch.ops.ops_multimodal_fusion' namespace."


# input_shape kernel_size stride padding ceil_mode count_include_pad divisor_override
POOL_CONFIGS = [
    # Basic non-overlapping pooling
    ((1, 1, 4, 4), (2, 2), (2, 2), (0, 0), False, True, None),
    ((1, 3, 8, 8), (2, 2), (2, 2), (0, 0), False, True, None),
    ((2, 16, 32, 32), (2, 2), (2, 2), (0, 0), False, True, None),
    # Overlapping pooling where stride is smaller than the kernel
    ((1, 1, 4, 4), (2, 2), (1, 1), (0, 0), False, True, None),
    ((1, 3, 8, 8), (3, 3), (1, 1), (0, 0), False, True, None),
    ((1, 3, 8, 8), (3, 3), (2, 2), (0, 0), False, True, None),
    # With padding
    ((1, 1, 4, 4), (3, 3), (1, 1), (1, 1), False, True, None),
    ((1, 3, 8, 8), (3, 3), (1, 1), (1, 1), False, True, None),
    ((2, 8, 16, 16), (3, 3), (2, 2), (1, 1), False, True, None),
    # count_include_pad False with padding
    ((1, 1, 4, 4), (3, 3), (1, 1), (1, 1), False, False, None),
    ((1, 3, 8, 8), (3, 3), (2, 2), (1, 1), False, False, None),
    # ceil_mode True
    ((1, 1, 5, 5), (3, 3), (2, 2), (0, 0), True, True, None),
    ((1, 3, 7, 7), (3, 3), (2, 2), (1, 1), True, True, None),
    # divisor_override
    ((1, 1, 4, 4), (2, 2), (2, 2), (0, 0), False, True, 2),
    ((1, 3, 8, 8), (3, 3), (1, 1), (0, 0), False, True, 5),
    # Non-square kernel
    ((1, 1, 8, 8), (2, 3), (1, 1), (0, 0), False, True, None),
    ((1, 1, 8, 8), (3, 2), (2, 1), (0, 0), False, True, None),
    # Stride defaults to kernel_size (pass empty stride)
    ((1, 3, 8, 8), (2, 2), (), (0, 0), False, True, None),
    ((1, 3, 9, 9), (3, 3), (), (0, 0), False, True, None),
    # Larger shapes
    ((4, 3, 64, 64), (3, 3), (2, 2), (1, 1), False, True, None),
    ((2, 32, 28, 28), (2, 2), (2, 2), (0, 0), False, True, None),
    # Single-element output
    ((1, 1, 3, 3), (3, 3), (1, 1), (0, 0), False, True, None),
    # 1x1 kernel (identity-like)
    ((1, 3, 4, 4), (1, 1), (1, 1), (0, 0), False, True, None),
    # ceil_mode True with count_include_pad False, combined attribute branch
    ((1, 3, 7, 7), (3, 3), (2, 2), (1, 1), True, False, None),
    ((2, 8, 16, 16), (3, 3), (2, 2), (1, 1), True, False, None),
    # divisor_override with padding (combined attribute branch)
    ((1, 3, 8, 8), (3, 3), (1, 1), (1, 1), False, True, 4),
    # width stride above one path through the reduce kernel, various configs
    ((2, 4, 16, 16), (2, 2), (3, 3), (0, 0), False, True, None),
    ((1, 1, 8, 8), (2, 2), (2, 3), (0, 0), False, True, None),
    # Non-square stride
    ((1, 3, 16, 16), (3, 3), (2, 3), (1, 1), False, True, None),
    ((1, 3, 16, 16), (3, 3), (3, 2), (1, 1), False, False, None),
    # Large shapes that exceed UB capacity, forcing multi-tile processing
    ((4, 64, 56, 56), (3, 3), (2, 2), (1, 1), False, True, None),
    ((8, 3, 128, 128), (2, 2), (2, 2), (0, 0), False, True, None),
    # Large with width stride above one, reduce path plus multi tile
    ((4, 16, 64, 64), (3, 3), (2, 2), (1, 1), False, True, None),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", POOL_CONFIGS)
def test_avg_pool2d_operator(case):
    """Test avg_pool2d operator against PyTorch CPU reference."""
    (shape, kernel_size, stride, padding, ceil_mode,
     count_include_pad, divisor_override) = case
    a = torch.randn(*shape, dtype=torch.float32)

    # CPU reference
    expected = F.avg_pool2d(
        a,
        kernel_size=kernel_size,
        stride=stride if stride else None,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    # NPU computation
    a_npu = a.npu()
    result_npu = torch.ops.ops_multimodal_fusion.avg_pool2d(
        a_npu,
        list(kernel_size),
        list(stride),
        list(padding),
        ceil_mode,
        count_include_pad,
        divisor_override,
    )
    result = result_npu.cpu()

    assert result.shape == expected.shape,\
        f"Shape mismatch: got {result.shape}, expected {expected.shape}"
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3),\
        f"AvgPool2d failed for shape {shape}, kernel_size {kernel_size}, " \
        f"stride {stride}, padding {padding}. " \
        f"Max diff: {torch.max(torch.abs(result - expected)):.6f}"
