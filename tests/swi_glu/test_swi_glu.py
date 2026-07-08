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

import pytest
import torch
import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "swi_glu"):
    pytest.skip(
        "ops_multimodal_fusion.swi_glu not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_swi_glu_interface():
    assert hasattr(torch.ops.ops_multimodal_fusion, "swi_glu")


# ---------------------------------------------------------------------------
# Shape & dtype coverage
# ---------------------------------------------------------------------------
#
# swi_glu halves the last dimension: N M to N M/2 .
# Input last dim must be even.
#
# 32-byte alignment (for half-dim after split):
# float32: half-dim multiple of 8 8 4 32 to aligned
# float16: half-dim multiple of 16 16 2 32 to aligned
# bfloat16: half-dim multiple of 16 16 2 32 to aligned
# ---------------------------------------------------------------------------

SHAPES = [
    # --- Basic shapes aligned fits UB ---
    (1, 32),         # single row, small
    (64, 256),       # typical
    (128, 512),      # medium
    (256, 128),      # more rows than cols
    # --- Non-aligned last dim after halving half-dim not aligned ---
    (32, 34),        # half 17 fp32: 17 4 68 pad to 96
    (32, 66),        # half 33 just over 32 boundary
    (64, 100),       # half 50 arbitrary non-aligned
    (16, 254),       # half 127 one less than power-of-2
    (16, 258),       # half 129 one more than power-of-2
    # --- Non-aligned row counts ---
    (1, 256),        # single row
    (3, 128),        # odd row count
    (7, 256),        # prime row count
    (17, 512),       # prime, many rows
    (63, 128),       # just under power-of-2
    (65, 128),       # just over power-of-2
    # --- UB-exceeding shapes (forces multi-tile processing) ---
    (1024, 2048),
    (4096, 1024),
    (256, 8192),
    (2, 16384),      # large last dim
    # --- Multi-tile with non-aligned last dim ---
    (4, 8190),       # multi-tile + non-aligned half-dim
    (2, 10002),      # multi-tile + arbitrary non-aligned
    # --- 3D shapes (N-D support) ---
    (32, 64, 128),
    (64, 256, 512),
    (1, 1, 128),     # single token decode
    (3, 7, 66),      # non-aligned in all dims
]

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_swi_glu_accuracy(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device="npu")
    result = torch.ops.ops_multimodal_fusion.swi_glu(x)

    # Reference: silu left_half right_half
    half_dim = shape[-1] // 2
    left = x[..., :half_dim].float()
    right = x[..., half_dim:].float()
    ref = torch.nn.functional.silu(left) * right

    expected_shape = list(shape)
    expected_shape[-1] = half_dim
    assert list(result.shape) == expected_shape
    assert result.dtype == dtype

    rtol = 1e-2 if dtype != torch.float32 else 1e-4
    atol = 1e-2 if dtype != torch.float32 else 1e-4
    assert torch.allclose(result.cpu().float(), ref.cpu(), rtol=rtol, atol=atol),\
        f"swi_glu failed for shape {shape}, dtype {dtype}. " \
        f"Max diff: {torch.max(torch.abs(result.cpu().float() - ref.cpu())):.6f}"
