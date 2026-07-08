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

if not hasattr(torch.ops.ops_multimodal_fusion, "dynamic_quant"):
    pytest.skip(
        "ops_multimodal_fusion.dynamic_quant not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_dynamic_quant_interface():
    assert hasattr(torch.ops.ops_multimodal_fusion, "dynamic_quant")


def _ref_symmetric_int8(x_float, scale_max=127.0):
    """Golden: per-row symmetric quantization."""
    abs_max = x_float.abs().amax(dim=-1, keepdim=True)
    scale = abs_max / scale_max
    y = torch.clamp(torch.round(x_float / scale), -127, 127).to(torch.int8)
    return y, scale.squeeze(-1)


# ---------------------------------------------------------------------------
# Shape & dtype & attr coverage
# ---------------------------------------------------------------------------
#
# 32-byte alignment rule (for cols dimension):
# float16: cols multiple of 16 16 2 32 to aligned
# bfloat16: cols multiple of 16 16 2 32 to aligned
#
# Execution paths (dynamic_quant.asc):
#   FULL_LOAD:   entire row fits in UB, single pass per row
#   LARGE_SHAPE: row exceeds UB, tiled multi-pass per row
#
# Quantization modes:
# symmetric True isSymFlag 1 : no offset tensor
# symmetric False isSymFlag 0 : creates offset tensor
# ---------------------------------------------------------------------------

SHAPES = [
    # --- Basic shapes aligned cols fits UB ---
    (32, 128),
    (64, 256),
    (128, 512),
    # --- Aligned cols ---
    (32, 16),        # minimum aligned for fp16/bf16
    (16, 64),
    (8, 1024),
    (4, 2048),
    # --- Non-aligned cols (requires padding) ---
    (32, 17),        # fp16: 17 2 34 pad to 64
    (32, 33),        # just over 32 boundary
    (64, 100),       # arbitrary non-aligned
    (16, 127),       # one less than power-of-2
    (16, 129),       # one more than power-of-2
    (8, 255),        # close to 256 boundary
    # --- Non-aligned row counts ---
    (1, 256),        # single row
    (3, 128),        # odd row count
    (7, 256),        # prime row count
    (63, 128),       # just under power-of-2
    (65, 128),       # just over power-of-2
    # --- UB-exceeding shapes (forces LARGE_SHAPE / multi-tile path) ---
    (1024, 2048),
    (4096, 1024),
    (256, 8192),
    (2, 16384),      # power-of-2, large cols
    # --- Multi-tile with non-aligned cols ---
    (2, 8191),       # multi-tile + just under power-of-2
    (4, 10007),      # prime cols, multi-tile
]

DTYPES = [torch.float16, torch.bfloat16]

SYMMETRIC = [True, False]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("symmetric", SYMMETRIC)
def test_dynamic_quant(shape, dtype, symmetric):
    x = torch.randn(shape, dtype=dtype, device="npu")
    y, scale = torch.ops.ops_multimodal_fusion.dynamic_quant(x, symmetric)

    assert y.shape == x.shape
    assert y.dtype == torch.int8
    assert scale.shape == (shape[0],)
    assert scale.dtype == torch.float32

    if symmetric:
        ref_y, ref_scale = _ref_symmetric_int8(x.cpu().float())
        assert torch.allclose(scale.cpu(), ref_scale, rtol=1e-3, atol=1e-3),\
            f"Scale mismatch for shape {shape}, dtype {dtype}. " \
            f"Max diff: {torch.max(torch.abs(scale.cpu() - ref_scale)):.6f}"
        # Allow +-1 quantization rounding difference
        diff = (y.cpu().int() - ref_y.int()).abs()
        assert diff.max() <= 1,\
            f"Quantized y mismatch for shape {shape}, dtype {dtype}. " \
            f"Max diff: {diff.max()}"
