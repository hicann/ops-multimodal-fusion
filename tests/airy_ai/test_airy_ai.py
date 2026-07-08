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
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401
if not hasattr(torch.ops.ops_multimodal_fusion, "airy_ai"):
    pytest.skip(
        "ops_multimodal_fusion.airy_ai not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def reference(x: torch.Tensor) -> torch.Tensor:
    return torch.special.airy_ai(x)


def run_op(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.ops_multimodal_fusion.airy_ai(x.npu()).cpu()


def assert_close(actual: torch.Tensor, expected: torch.Tensor, *, rtol=5e-5, atol=5e-5):
    max_diff = (actual.float() - expected.float()).abs().max().item() if actual.numel() else 0.0
    assert torch.allclose(actual.float(), expected.float(), rtol=rtol, atol=atol, equal_nan=True), max_diff


SHAPES_2D = [
    (1, 128), (64, 64), (256, 128), (128, 256), (1024, 128), (4096, 128),
    (32, 8), (32, 16), (64, 32), (16, 64), (128, 512),
    (32, 1024), (8, 2048), (4, 4096), (2, 8192),
    (32, 9), (32, 10), (64, 17), (32, 31), (32, 33),
    (128, 65), (64, 100), (16, 127), (16, 129), (32, 255),
    (16, 257), (8, 1000), (4, 3333),
    (1, 128), (3, 128), (7, 64), (17, 128), (63, 256), (65, 256),
    (8192, 128), (16384, 64), (32768, 32), (65537, 16), (4096, 256),
    (8192, 65), (16384, 33),
    (2, 13000), (2, 16384), (1, 20000), (4, 10000), (2, 25000), (1, 32768),
    (2, 13001), (1, 16383), (2, 20001), (1, 9999), (2, 10007),
    (1, 16000),
    (4, 8), (4, 16), (4, 24), (4, 32), (4, 64), (4, 128),
    (4, 256), (4, 512), (4, 1024), (4, 2048), (4, 4096),
]

SHAPES_3D = [
    (1, 128, 128), (2, 64, 128), (4, 256, 128),
    (1, 1, 128), (3, 7, 65), (1, 1, 13000),
    (16, 64, 128), (32, 32, 128), (64, 16, 128), (128, 8, 128),
    (256, 4, 128), (512, 4, 64),
    (4, 128, 128), (4, 256, 128), (8, 512, 128), (2, 1024, 64),
    (32, 64, 128), (64, 32, 64), (128, 128, 32),
    (1024, 8, 128), (2048, 4, 128), (4096, 2, 128),
    (1024, 32, 64), (2048, 16, 64), (8192, 4, 64),
    (4, 1024, 128), (8, 2048, 64), (16, 4096, 32),
    (2, 2048, 128), (4, 4096, 64),
    (512, 256, 64), (1024, 128, 32), (256, 512, 32),
    (128, 1024, 32), (512, 512, 16),
    (16, 16, 65), (32, 8, 257), (8, 32, 1000),
    (256, 32, 65), (512, 16, 33),
    (8, 16, 13000), (4, 32, 10007), (16, 64, 16384), (8, 128, 10000),
]

SHAPES_4D = [
    (2, 3, 4, 128),
    (4, 8, 16, 128), (8, 4, 16, 128), (16, 8, 4, 128),
    (2, 16, 32, 128), (32, 4, 8, 128), (8, 16, 32, 64),
    (2, 4, 8, 1024), (4, 4, 4, 2048),
    (256, 4, 8, 128), (512, 2, 4, 128), (1024, 2, 2, 64),
    (4, 256, 8, 128), (2, 512, 4, 128), (4, 1024, 2, 64),
    (4, 8, 256, 128), (2, 4, 512, 128), (4, 2, 1024, 64),
    (64, 64, 16, 128), (128, 32, 8, 128), (32, 128, 16, 64),
    (256, 16, 16, 64), (64, 64, 64, 32),
    (4, 4, 4, 65), (2, 3, 5, 257), (64, 16, 8, 65), (32, 32, 4, 33),
]


def test_airy_ai_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "airy_ai")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_airy_ai_empty():
    x = torch.empty((0,), dtype=torch.float32)
    out = run_op(x)
    assert out.shape == x.shape
    assert out.dtype == torch.float32
    assert out.numel() == 0


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", [(1,), (7,), (17,), (128,), (2, 17), (4, 33), (3, 5, 7), (2, 3, 4, 5)])
def test_airy_ai_regular_random(shape):
    x = (torch.rand(*shape, dtype=torch.float32) - 0.5) * 64.0
    expected = reference(x)
    out = run_op(x)
    assert out.shape == x.shape
    assert out.dtype == torch.float32
    assert_close(out, expected, rtol=8e-5, atol=8e-5)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES_2D)
def test_airy_ai_regbase_2d_shapes(shape):
    x = (torch.rand(*shape, dtype=torch.float32) - 0.5) * 64.0
    expected = reference(x)
    out = run_op(x)
    assert out.shape == x.shape
    assert_close(out, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES_3D)
def test_airy_ai_regbase_3d_shapes(shape):
    x = (torch.rand(*shape, dtype=torch.float32) - 0.5) * 64.0
    expected = reference(x)
    out = run_op(x)
    assert out.shape == x.shape
    assert_close(out, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES_4D)
def test_airy_ai_regbase_4d_shapes(shape):
    x = (torch.rand(*shape, dtype=torch.float32) - 0.5) * 64.0
    expected = reference(x)
    out = run_op(x)
    assert out.shape == x.shape
    assert_close(out, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_airy_ai_boundary_samples():
    x = torch.tensor(
        [
            -128.0,
            -64.0,
            -16.0,
            -8.0,
            -2.1001,
            -2.09,
            -2.0799,
            -1.0,
            -1e-4,
            -0.0,
            0.0,
            1e-4,
            1.0,
            2.0799,
            2.09,
            2.0901,
            8.3203,
            8.3203353,
            8.3204,
            16.0,
            32.0,
            64.0,
            103.8919,
            103.892,
            103.8921,
        ],
        dtype=torch.float32,
    )
    expected = reference(x)
    out = run_op(x)
    assert_close(out, expected, rtol=1.5e-4, atol=1.5e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_airy_ai_dense_piecewise_ranges():
    x = torch.cat(
        [
            torch.linspace(-128.0, -2.2, steps=1024, dtype=torch.float32),
            torch.linspace(-2.2, 2.2, steps=2048, dtype=torch.float32),
            torch.linspace(2.2, 32.0, steps=1024, dtype=torch.float32),
        ]
    )
    expected = reference(x)
    out = run_op(x)
    assert_close(out, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_airy_ai_non_contiguous_input():
    base = (torch.rand(9, 17, dtype=torch.float32) - 0.5) * 40.0
    x = base[:, ::2]
    assert not x.is_contiguous()
    out = run_op(x)
    expected = reference(x.contiguous())
    assert out.shape == x.shape
    assert_close(out, expected, rtol=8e-5, atol=8e-5)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_airy_ai_special_values():
    x = torch.tensor(
        [-0.0, 0.0, -1.0, 1.0, 16.0, 104.0, float("inf"), float("-inf"), float("nan")],
        dtype=torch.float32,
    )
    out = run_op(x)
    expected = reference(x)

    assert_close(out[:6], expected[:6], rtol=1.5e-4, atol=1.5e-4)
    assert torch.isnan(out[6])
    assert torch.isnan(out[7])
    assert torch.isnan(out[8])


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_airy_ai_finite_outputs_on_finite_inputs():
    x = torch.linspace(-64.0, 64.0, steps=4096, dtype=torch.float32)
    out = run_op(x)
    assert torch.isfinite(out).all()
