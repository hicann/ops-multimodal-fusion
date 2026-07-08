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
import torch.nn.functional as F
import pytest
import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "layer_norm"):
    pytest.skip(
        "ops_multimodal_fusion.layer_norm not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_layer_norm_interface_exist():
    """
    Test that the 'ascend_ops.layer_norm' operator is present in torch.ops.
    """
    assert hasattr(torch.ops.ops_multimodal_fusion, "layer_norm"),\
        "The 'layer_norm' operator is not registered in the 'torch.ops.ops_multimodal_fusion.' namespace."


# ---------------------------------------------------------------------------
# Shape & dtype coverage
# ---------------------------------------------------------------------------
#
# 32-byte alignment rule:
# float32: hiddenSize multiple of 8 8 4 32 to aligned
# float16: hiddenSize multiple of 16 16 2 32 to aligned
#
# Execution paths (layer_norm.asc):
# fp32 whole-row: hiddenSize fits in UB with 7 float buffers isTiled_ false
# fp32 tiled: hiddenSize exceeds whole-row UB capacity isTiled_ true
#   fp16 tiled:     always tiled, cast to float32 internally
#
# UB capacity thresholds approx UB 192KB :
#   fp32 whole-row: 7*bufSize + 32 <= UB -> hiddenSize <= ~6900
#   fp32 tiled:     5 float buffers -> tileHidden ~ (192K-32) / 20 ~ 9827
#   fp16 tiled:     2 half + 3 float buffers -> tileHidden ~ (192K-32) / 16 ~ 12284
# ---------------------------------------------------------------------------

# Format: rows hiddenSize
SHAPES = [
    # --- Basic shapes aligned hiddenSize fits UB whole-row for fp32 ---
    (2, 4),          # very small
    (1, 256),        # single row
    (16, 128),       # small
    (128, 768),      # BERT-base hidden
    (64, 1024),      # typical
    (4096, 128),     # many rows, small hidden
    # --- Aligned hiddenSize (hiddenSize*elemSize is multiple of 32) ---
    (32, 8),         # minimum aligned for fp32 8 4 32
    (32, 16),        # minimum aligned for both fp32 & fp16
    (16, 64),        # few rows
    (8, 512),        # medium hidden
    (4, 2048),       # larger hidden
    (2, 4096),       # near whole-row UB boundary for fp32
    # --- Non-aligned hiddenSize (requires DataCopyPad zero-fill) ---
    (32, 9),         # fp32: 9 4 36 pad to 64; fp16: 9 2 18 pad to 32
    (32, 13),        # fp32: 13 4 52 pad to 64
    (64, 17),        # fp32: 17 4 68 pad to 96; fp16: 17 2 34 pad to 64
    (64, 100),       # fp32: 100 4 400 pad to 416
    (16, 127),       # one less than power-of-2
    (16, 129),       # one more than power-of-2
    (32, 255),       # close to 256 boundary
    (8, 1000),       # arbitrary non-aligned
    (4, 3333),       # arbitrary large non-aligned
    # --- Non-aligned row counts (core distribution edge cases) ---
    (1, 128),        # single row
    (3, 256),        # odd row count
    (7, 128),        # prime row count
    (17, 256),       # prime, many rows
    (63, 128),       # just under power-of-2
    (65, 128),       # just over power-of-2
    # --- UB-exceeding: hidden > whole-row UB capacity ---
    # fp32: forces tiled path isTiled_ true
    # fp16: always tiled, these also exceed tileHidden forcing multi-tile
    (8, 8192),       # exceeds fp32 whole-row capacity (~6900)
    (4, 12288),      # exceeds fp32 tileHidden (~9827), large model hidden
    (2, 16384),      # power-of-2, exceeds both fp32 & fp16 tileHidden
    (1, 20000),      # large non-aligned, ~2 tiles for fp16
    (2, 25000),      # ~2-3 tiles for fp16, ~3 tiles for fp32
    (1, 32768),      # large power-of-2, 3+ tiles
    # --- Multi-tile with non-aligned hiddenSize (padding + tiling combined) ---
    (2, 13001),      # multi-tile + 1 element past alignment
    (1, 16383),      # multi-tile + just under power-of-2
    (2, 9999),       # near fp32 tileHidden boundary, non-aligned
    (1, 10007),      # prime hiddenSize, multi-tile for fp32
]

DTYPES = [
    torch.float32,
    torch.float16,
]

# eps attribute coverage: different epsilon values affect numerical stability
EPS_VALUES = [1e-6, 1e-5]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("rows, hidden_size", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eps", EPS_VALUES)
def test_layer_norm_operator(rows, hidden_size, dtype, eps):
    """
    Test the functionality of the layer_norm operator.

    LayerNorm x gamma x - mean / sqrt var + eps + beta
    """

    x = torch.randn(rows, hidden_size, dtype=dtype)
    gamma = torch.randn(hidden_size, dtype=dtype)
    beta = torch.randn(hidden_size, dtype=dtype)

    expected = F.layer_norm(x, [hidden_size], gamma, beta, eps=eps)

    x_npu = x.npu()
    gamma_npu = gamma.npu()
    beta_npu = beta.npu()
    result_npu = torch.ops.ops_multimodal_fusion.layer_norm(x_npu, gamma_npu, beta_npu, eps)
    result = result_npu.cpu()

    rtol = 5e-2 if dtype == torch.float16 else 1e-3
    atol = 1e-2 if dtype == torch.float16 else 1e-4

    assert torch.allclose(result, expected, rtol=rtol, atol=atol),\
        f"LayerNorm failed for shape ({rows}, {hidden_size}), dtype {dtype}. " \
        f"Max diff: {torch.max(torch.abs(result - expected)):.6f}"
