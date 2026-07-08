#!/usr/bin/env python3
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
import torch_npu

import ops_multimodal_fusion


if not hasattr(torch.ops.ops_multimodal_fusion, "bessel_y1"):
    pytest.skip(
        "ops_multimodal_fusion.bessel_y1 not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def reference(x: torch.Tensor) -> torch.Tensor:
    return torch.special.bessel_y1(x)


def run_op(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.ops_multimodal_fusion.bessel_y1(x.npu()).cpu()


def assert_close(actual: torch.Tensor, expected: torch.Tensor, *, rtol=4e-4, atol=4e-4):
    actual = actual.float()
    expected = expected.float()
    finite_mask = torch.isfinite(actual) & torch.isfinite(expected)
    if finite_mask.any():
        max_diff = (actual[finite_mask] - expected[finite_mask]).abs().max().item()
        assert torch.allclose(actual[finite_mask], expected[finite_mask], rtol=rtol, atol=atol), max_diff
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isposinf(actual), torch.isposinf(expected))
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))


SHAPES_2D = [
    # --- Basic (aligned Dv, single tile) ---
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


def test_bessel_y1_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "bessel_y1")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_bessel_y1_empty():
    x = torch.empty((0,), dtype=torch.float32)
    out = run_op(x)
    assert out.shape == x.shape
    assert out.dtype == torch.float32
    assert out.numel() == 0


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "shape",
    [(1,), (7,), (17,), (128,), (2, 17), (4, 33), (3, 5, 7), (2, 3, 4, 5)],
)


def test_bessel_y1_regular_positive_random(shape):
    x = torch.rand(*shape, dtype=torch.float32) * 32.0 + 1e-4
    expected = reference(x)
    out = run_op(x)
    assert out.shape == x.shape
    assert out.dtype == torch.float32
    assert_close(out, expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES_2D)
def test_bessel_y1_regbase_2d_shapes(shape):
    x = torch.rand(*shape, dtype=torch.float32) * 32.0 + 1e-6
    expected = reference(x)
    out = run_op(x)
    assert out.shape == x.shape
    assert_close(out, expected, rtol=5e-4, atol=5e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES_3D)
def test_bessel_y1_regbase_3d_shapes(shape):
    x = torch.rand(*shape, dtype=torch.float32) * 32.0 + 1e-6
    expected = reference(x)
    out = run_op(x)
    assert out.shape == x.shape
    assert_close(out, expected, rtol=5e-4, atol=5e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES_4D)
def test_bessel_y1_regbase_4d_shapes(shape):
    x = torch.rand(*shape, dtype=torch.float32) * 32.0 + 1e-6
    expected = reference(x)
    out = run_op(x)
    assert out.shape == x.shape
    assert_close(out, expected, rtol=5e-4, atol=5e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_bessel_y1_non_contiguous_input():
    base = torch.rand(9, 17, dtype=torch.float32) * 24.0 + 1e-4
    x = base[:, ::2]
    assert not x.is_contiguous()
    out = run_op(x)
    expected = reference(x.contiguous())
    assert out.shape == x.shape
    assert_close(out, expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_bessel_y1_boundary_samples():
    x = torch.tensor(
        [
            1e-12,
            1e-8,
            1e-6,
            1e-4,
            1e-3,
            0.1,
            0.5,
            1.0,
            2.0,
            4.999,
            5.0,
            5.001,
            8.0,
            16.0,
            32.0,
        ],
        dtype=torch.float32,
    )
    expected = reference(x)
    out = run_op(x)
    assert_close(out, expected, rtol=5e-4, atol=5e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_bessel_y1_special_values():
    x = torch.tensor(
        [-1.0, -0.0, 0.0, 0.5, 1.0, 5.0, 5.5, float("inf"), float("nan")],
        dtype=torch.float32,
    )
    expected = reference(x)
    out = run_op(x)
    assert_close(out, expected, rtol=5e-4, atol=5e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_bessel_y1_small_argument_singularity():
    x = torch.tensor([1e-12, 1e-9, 1e-6, 1e-3], dtype=torch.float32)
    expected = reference(x)
    out = run_op(x)
    assert_close(out, expected, rtol=8e-4, atol=8e-4)
