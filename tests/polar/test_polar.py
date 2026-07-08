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

import math

import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "polar"):
    pytest.skip(
        "ops_multimodal_fusion.polar not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float16, torch.float32]
SHAPES = [
    (1,),
    (7,),
    (8,),
    (9,),
    (1024,),
    (3, 5),
    (2, 3, 7),
    (10000,),
]


def _make_inputs(shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    radius = torch.empty(*shape, dtype=torch.float32).uniform_(-4.0, 4.0, generator=g)
    angle = torch.empty(*shape, dtype=torch.float32).uniform_(-math.pi, math.pi, generator=g)
    return radius.to(dtype), angle.to(dtype)


def _assert_close(actual, expected, msg):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    if expected.real.dtype == torch.float16:
        rtol, atol = 2e-2, 2e-2
    else:
        rtol, atol = 5e-3, 5e-3
    assert torch.allclose(actual.cpu(), expected.cpu(), rtol=rtol, atol=atol), (
        f"{msg}: max abs diff = {(actual.cpu() - expected.cpu()).abs().max().item()}"
    )


def _expected_polar(radius, angle, dtype):
    calc_radius = radius.float() if dtype == torch.float16 else radius
    calc_angle = angle.float() if dtype == torch.float16 else angle
    return torch.complex(
        (calc_radius * torch.cos(calc_angle)).to(dtype),
        (calc_radius * torch.sin(calc_angle)).to(dtype),
    )


def test_polar_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "polar")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_polar_operator(shape, dtype):
    seed = abs(hash((shape, dtype))) % 10_000_000
    radius, angle = _make_inputs(shape, dtype, seed)

    y = torch.ops.ops_multimodal_fusion.polar(radius.npu(), angle.npu()).cpu()
    expected = _expected_polar(radius, angle, dtype)

    _assert_close(y, expected, f"polar mismatch for shape={shape}, dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "radius_shape,angle_shape",
    [
        ((17, 1), (1, 9)),
        ((1, 17), (3, 17)),
        ((2, 1, 17), (1, 3, 1)),
        ((2, 3, 17), (17,)),
        ((2, 1, 3, 17), (4, 1, 1)),
    ],
)


def test_polar_broadcast_special_shapes(radius_shape, angle_shape, dtype):
    radius, _ = _make_inputs(radius_shape, dtype, 101)
    _, angle = _make_inputs(angle_shape, dtype, 202)

    y = torch.ops.ops_multimodal_fusion.polar(radius.npu(), angle.npu()).cpu()
    expected = _expected_polar(radius, angle, dtype)

    _assert_close(
        y, expected,
        f"polar broadcast mismatch for radius={radius_shape}, angle={angle_shape}, dtype={dtype}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_polar_named_angles():
    radius = torch.tensor([0.0, 1.0, 2.0, -3.0, 4.0, 5.0], dtype=torch.float32)
    angle = torch.tensor(
        [0.0, math.pi / 6.0, math.pi / 2.0, -math.pi / 2.0, math.pi, -math.pi],
        dtype=torch.float32,
    )

    y = torch.ops.ops_multimodal_fusion.polar(radius.npu(), angle.npu()).cpu()
    expected = torch.complex(radius * torch.cos(angle), radius * torch.sin(angle))

    _assert_close(y, expected, "polar named-angle mismatch")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_polar_empty_tensor():
    radius = torch.empty((0, 3), dtype=torch.float32).npu()
    angle = torch.empty((1, 3), dtype=torch.float32).npu()
    y = torch.ops.ops_multimodal_fusion.polar(radius, angle).cpu()
    assert y.shape == (0, 3)
    assert y.dtype == torch.complex64


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_polar_rejects_unsupported_dtype():
    for unsupported in [torch.bfloat16, torch.float64, torch.int32]:
        radius = torch.ones(4, dtype=unsupported).npu()
        angle = torch.ones(4, dtype=unsupported).npu()
        with pytest.raises(RuntimeError, match="polar"):
            torch.ops.ops_multimodal_fusion.polar(radius, angle)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_polar_rejects_dtype_mismatch():
    radius = torch.ones(4, dtype=torch.float32).npu()
    angle = torch.ones(4, dtype=torch.float16).npu()
    with pytest.raises(RuntimeError, match="polar"):
        torch.ops.ops_multimodal_fusion.polar(radius, angle)
