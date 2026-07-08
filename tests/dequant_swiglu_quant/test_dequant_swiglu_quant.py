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

if not hasattr(torch.ops.ops_multimodal_fusion, "dequant_swiglu_quant"):
    pytest.skip(
        "ops_multimodal_fusion.dequant_swiglu_quant not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_dequant_swiglu_quant_interface():
    assert hasattr(torch.ops.ops_multimodal_fusion, "dequant_swiglu_quant")


def _ref_float_int8(x_input):
    """Golden: float -> swiglu -> dynamic quant -> int8."""
    rows, cols = x_input.shape
    half = cols // 2
    xf = x_input.float()
    gate = xf[:, :half]
    act = xf[:, half:]
    silu_act = act / (1.0 + torch.exp(-act))
    swiglu = silu_act * gate

    abs_max = swiglu.abs().amax(dim=-1, keepdim=True)
    scale = abs_max / 127.0
    inv_scale = torch.where(abs_max > 0, 127.0 / abs_max, torch.zeros_like(abs_max))
    y = torch.clamp(torch.round(swiglu * inv_scale), -128, 127).to(torch.int8)
    return y, scale.squeeze(-1)


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------
#
# dequant_swiglu_quant halves the last dimension (SwiGLU gate/act split).
# Input cols must be even. Three variants cover all dispatch branches:
#
# mode "float": bf16/fp16 input no scales to hasAScale false
# mode "int32_ws": int32 input weight_scale only
# mode "int32_ws_as": int32 input weight_scale + act_scale to hasAScale true
#
# Accuracy checks (float path only) run on a smaller sub-set.
# ---------------------------------------------------------------------------

SHAPES = [
    # --- Basic shapes aligned fits UB ---
    (8, 256),
    (16, 128),
    (64, 512),
    (128, 256),
    # --- Non-aligned cols after halving half-cols not aligned to 32 bytes ---
    (8, 34),         # half 17 bf16: 17 2 34 needs pad
    (16, 66),        # half 33 just over 32 boundary
    (32, 100),       # half 50 arbitrary non-aligned
    (8, 254),        # half 127 one less than power-of-2
    (8, 258),        # half 129 one more than power-of-2
    # --- Non-aligned row counts ---
    (1, 256),        # single row
    (3, 128),        # odd row count
    (7, 256),        # prime row count
    (17, 128),       # prime, many rows
    # --- UB-exceeding shapes (forces multi-tile processing) ---
    (1024, 2048),
    (256, 8192),
    (2, 16384),      # large cols
    # --- Multi-tile with non-aligned cols ---
    (4, 8190),
    (2, 10002),
]

ACCURACY_SHAPES = [
    (16, 128),
    (64, 512),
    (128, 256),
    (8, 34),
    (8, 254),
    (1, 256),
]

# mode dtype - one parametrize axis covers both dispatch branch and dtype.
MODES = [
    ("float", torch.bfloat16),
    ("float", torch.float16),
    ("int32_ws", torch.int32),
    ("int32_ws_as", torch.int32),
]


def _run(mode, dtype, shape):
    rows, in_cols = shape
    out_cols = in_cols // 2
    if mode == "float":
        x = torch.randn(shape, dtype=dtype, device="npu")
        y, scale = torch.ops.ops_multimodal_fusion.dequant_swiglu_quant(x, None, None)
    else:
        x = torch.randint(-100, 100, (rows, in_cols), dtype=torch.int32, device="npu")
        weight_scale = torch.rand(in_cols, dtype=torch.float32, device="npu") * 0.1 + 0.01
        act_scale = (torch.full((rows,), 0.5, dtype=torch.float32, device="npu")
                     if mode == "int32_ws_as" else None)
        y, scale = torch.ops.ops_multimodal_fusion.dequant_swiglu_quant(x, weight_scale, act_scale)

    assert y.shape == (rows, out_cols)
    assert y.dtype == torch.int8
    assert scale.shape == (rows,)
    assert scale.dtype == torch.float32
    return x, y, scale


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
@pytest.mark.parametrize("mode,dtype", MODES)
@pytest.mark.parametrize("shape", SHAPES)
def test_dequant_swiglu_quant_operator(mode, dtype, shape):
    """Shape/dtype coverage for all three dispatch branches."""
    _run(mode, dtype, shape)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
@pytest.mark.parametrize("shape", ACCURACY_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dequant_swiglu_quant_float_accuracy(shape, dtype):
    """Verify float path output matches CPU reference (quantization: ±1 rounding)."""
    x, y, scale = _run("float", dtype, shape)
    ref_y, ref_scale = _ref_float_int8(x.cpu())

    assert torch.allclose(scale.cpu(), ref_scale, rtol=1e-2, atol=1e-2),\
        f"Scale mismatch for shape {shape}, dtype {dtype}. " \
        f"Max diff: {torch.max(torch.abs(scale.cpu() - ref_scale)):.6f}"
    diff = (y.cpu().int() - ref_y.int()).abs()
    assert diff.max() <= 1,\
        f"Quantized value mismatch for shape {shape}, dtype {dtype}. Max diff: {diff.max()}"
