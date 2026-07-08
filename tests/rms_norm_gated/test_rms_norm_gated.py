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

if not hasattr(torch.ops.ops_multimodal_fusion, "rms_norm_gated"):
    pytest.skip(
        "ops_multimodal_fusion.rms_norm_gated not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_rms_norm_gated_interface_exist():
    """Test that the 'ops_multimodal_fusion.rms_norm_gated' operator is registered."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "rms_norm_gated"),\
        "The 'rms_norm_gated' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


def rms_norm_gated_reference_cpu(hidden_states, gate, gamma, epsilon=1e-6):
    """CPU reference matching the golden implementation."""
    input_dtype = hidden_states.dtype
    dv = hidden_states.shape[-1]
    x = hidden_states.to(torch.float32)
    normed = F.rms_norm(x, (dv,), weight=gamma.to(torch.float32), eps=epsilon)
    gate_activated = F.silu(gate.to(torch.float32))
    output = torch.mul(normed, gate_activated)
    return output.to(input_dtype)


def rms_norm_gated_reference_npu(hidden_states, gate, gamma, epsilon=1e-6):
    """CPU reference matching the golden implementation."""
    x_npu = hidden_states.npu()
    gamma_npu = gamma.npu()
    gate_npu = gate.npu()
    normed = torch_npu.npu_rms_norm(x_npu, gamma_npu, epsilon=epsilon)[0]
    gate_activated = F.silu(gate_npu)
    output = torch.mul(normed, gate_activated)
    return output.cpu()

# ---------------------------------------------------------------------------
# Shape coverage
# ---------------------------------------------------------------------------
#
# Organized by the kernel branches each category exercises. The operator
# normalizes along the last dim (Dv); leading dims are flattened, so
# rank-agnostic cases can share one test body.
#
# 32-byte alignment rule:
# fp32 aligned: Dv multiple of 8 8 4 32
# fp16/bf16: Dv multiple of 16 16 2 32
#
# tileHidden threshold approx UB 192KB :
#   fp16/bf16: ~12284 elements   fp32: ~9827 elements
# ---------------------------------------------------------------------------

SHAPES_2D = [
    # --- Basic aligned Dv single tile ---
    (1, 128), (64, 64), (256, 128), (128, 256), (1024, 128), (4096, 128),
    # --- Aligned Dv, Dv*elemSize multiple of 32 bytes ---
    (32, 8), (32, 16), (64, 32), (16, 64), (128, 512),
    (32, 1024), (8, 2048), (4, 4096), (2, 8192),
    # --- Non-aligned Dv (DataCopyPad zero-fill path) ---
    (32, 9), (32, 10), (64, 17), (32, 31), (32, 33),
    (128, 65), (64, 100), (16, 127), (16, 129), (32, 255),
    (16, 257), (8, 1000), (4, 3333),
    # --- Non-aligned row counts (core distribution edges) ---
    (1, 128), (3, 128), (7, 64), (17, 128), (63, 256), (65, 256),
    # --- Large row counts (stress core distribution + batch K tail) ---
    (8192, 128), (16384, 64), (32768, 32), (65537, 16), (4096, 256),
    # --- Large rows + non-aligned Dv ---
    (8192, 65), (16384, 33),
    # --- UB-exceeding Dv (multi-tile processing) ---
    (2, 13000), (2, 16384), (1, 20000), (4, 10000), (2, 25000), (1, 32768),
    # --- Multi-tile + non-aligned Dv ---
    (2, 13001), (1, 16383), (2, 20001), (1, 9999), (2, 10007),
    # --- Single-row UB-exceeding (multi-tile tail) ---
    (1, 16000),
    # --- Dv exactly at common tile boundaries (no tail tile) ---
    (4, 8), (4, 16), (4, 24), (4, 32), (4, 64), (4, 128),
    (4, 256), (4, 512), (4, 1024), (4, 2048), (4, 4096),
]

SHAPES_3D = [
    # --- Small leading dims (original coverage) ---
    (1, 128, 128), (2, 64, 128), (4, 256, 128),
    (1, 1, 128), (3, 7, 65), (1, 1, 13000),
    # --- Medium leading dims ---
    (16, 64, 128), (32, 32, 128), (64, 16, 128), (128, 8, 128),
    (256, 4, 128), (512, 4, 64),
    (4, 128, 128), (4, 256, 128), (8, 512, 128), (2, 1024, 64),
    (32, 64, 128), (64, 32, 64), (128, 128, 32),
    # --- Large d0 (batch/sequence reaches thousands) ---
    (1024, 8, 128), (2048, 4, 128), (4096, 2, 128),
    (1024, 32, 64), (2048, 16, 64), (8192, 4, 64),
    # --- Large d1 (middle dim reaches thousands) ---
    (4, 1024, 128), (8, 2048, 64), (16, 4096, 32),
    (2, 2048, 128), (4, 4096, 64),
    # --- Both leading dims large ---
    (512, 256, 64), (1024, 128, 32), (256, 512, 32),
    (128, 1024, 32), (512, 512, 16),
    # --- Non-aligned Dv with larger leading dims ---
    (16, 16, 65), (32, 8, 257), (8, 32, 1000),
    (256, 32, 65), (512, 16, 33),
    # --- Multi-tile Dv with larger leading dims ---
    (8, 16, 13000), (4, 32, 10007), (16, 64, 16384), (8, 128, 10000),
]

SHAPES_4D = [
    # --- Original small case ---
    (2, 3, 4, 128),
    # --- Medium dims at each position ---
    (4, 8, 16, 128), (8, 4, 16, 128), (16, 8, 4, 128),
    (2, 16, 32, 128), (32, 4, 8, 128), (8, 16, 32, 64),
    (2, 4, 8, 1024), (4, 4, 4, 2048),
    # --- Large d0 ---
    (256, 4, 8, 128), (512, 2, 4, 128), (1024, 2, 2, 64),
    # --- Large d1 ---
    (4, 256, 8, 128), (2, 512, 4, 128), (4, 1024, 2, 64),
    # --- Large d2 ---
    (4, 8, 256, 128), (2, 4, 512, 128), (4, 2, 1024, 64),
    # --- Multiple leading dims large ---
    (64, 64, 16, 128), (128, 32, 8, 128), (32, 128, 16, 64),
    (256, 16, 16, 64), (64, 64, 64, 32),
    # --- Non-aligned Dv ---
    (4, 4, 4, 65), (2, 3, 5, 257), (64, 16, 8, 65), (32, 32, 4, 33),
]

# Attach dtype per shape-group so each kernel branch runs under every supported
# dtype once, without explicit per-dtype functions. bf16 is limited to shapes
# where numerical checks in bf16 are meaningful (single-tile plus the 3D/4D
# branches), matching prior golden-reference coverage.
DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# Gamma modes: ones (typical init) and random positive (trained weights).
GAMMA_MODES = ["ones", "non_ones"]


def _make_gamma(dv, dtype, mode):
    if mode == "ones":
        return torch.ones(dv, dtype=dtype)
    return (torch.randn(dv, dtype=dtype).abs() + 0.1)


def _tol(dtype):
    if dtype == torch.float16:
        return 1e-1, 1e-1
    return 1e-2, 1e-2


SHAPES_ALL = (
    [("2d", s) for s in SHAPES_2D]
    + [("3d", s) for s in SHAPES_3D]
    + [("4d", s) for s in SHAPES_4D]
)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("rank_tag,shape", SHAPES_ALL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("gamma_mode", GAMMA_MODES)
def test_rms_norm_gated_operator(rank_tag, shape, dtype, gamma_mode):
    """Unified coverage: all shape branches x dtypes x gamma variants.

    Validates shape, dtype, no NaN/Inf, and numerical match vs CPU fp32 reference.
    """
    dv = shape[-1]
    hidden_states = torch.randn(*shape, dtype=dtype)
    gate = torch.randn(*shape, dtype=dtype)
    gamma = _make_gamma(dv, dtype, gamma_mode)
    expected = rms_norm_gated_reference_cpu(hidden_states, gate, gamma, 1e-6)

    result = torch.ops.ops_multimodal_fusion.rms_norm_gated(
        hidden_states.npu(), gate.npu(), gamma.npu(), 1e-6).cpu()

    assert result.shape == expected.shape,\
        f"shape mismatch: got {result.shape}, expected {expected.shape}"
    assert result.dtype == dtype, f"dtype mismatch: got {result.dtype}"
    assert not torch.isnan(result).any(), "NaN in output"
    assert not torch.isinf(result).any(), "Inf in output"

    rtol, atol = _tol(dtype)
    max_diff = (result.float() - expected.float()).abs().max().item()
    assert torch.allclose(result.float(), expected.float(), rtol=rtol, atol=atol),\
        f"rms_norm_gated failed [rank={rank_tag}, shape={shape}, dtype={dtype}, " \
        f"gamma={gamma_mode}]. max_diff={max_diff:.6f}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES_3D + SHAPES_4D)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_gated_nd_vs_2d_consistency(shape, dtype):
    """Multi-dim inputs must produce byte-identical results to their 2D flattening."""
    dv = shape[-1]
    hidden_states = torch.randn(*shape, dtype=dtype)
    gate = torch.randn(*shape, dtype=dtype)
    gamma = torch.ones(dv, dtype=dtype)

    gm_npu = gamma.npu()
    result_nd = torch.ops.ops_multimodal_fusion.rms_norm_gated(
        hidden_states.npu(), gate.npu(), gm_npu, 1e-6).cpu()

    h_2d = hidden_states.reshape(-1, dv).npu()
    g_2d = gate.reshape(-1, dv).npu()
    result_2d = torch.ops.ops_multimodal_fusion.rms_norm_gated(h_2d, g_2d, gm_npu, 1e-6).cpu()
    result_2d = result_2d.reshape(*shape)

    diff = (result_nd.float() - result_2d.float()).abs().max().item()
    assert diff == 0.0, f"ND vs 2D mismatch for shape={shape}, dtype={dtype}: {diff}"
