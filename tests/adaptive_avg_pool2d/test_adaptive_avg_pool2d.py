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
import torch.nn.functional as F
import torch_npu  # noqa: F401

import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "adaptive_avg_pool2d"):
    pytest.skip(
        "ops_multimodal_fusion.adaptive_avg_pool2d not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float16, torch.float32]


def _make_input(shape, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(*shape, dtype=torch.float32).uniform_(-2.0, 2.0, generator=g)
    if x.numel() > 7:
        flat = x.flatten()
        flat[1::7] = 0.0
        flat[3::11] = -1.5
        flat[5::13] = 1.25
    return x.to(dtype)


def _reference(x, output_size):
    ref_input = x.float() if x.dtype == torch.float16 else x
    return F.adaptive_avg_pool2d(ref_input, output_size).to(x.dtype)


def _assert_close(actual, expected, label):
    actual = actual.cpu()
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    rtol, atol = (8e-3, 8e-3) if expected.dtype == torch.float16 else (1e-5, 1e-5)
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"{label}: max_abs={(actual - expected).abs().max().item() if actual.numel() else 0.0} "
        f"actual={actual.flatten()[:8]} expected={expected.flatten()[:8]}"
    )


def test_adaptive_avg_pool2d_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "adaptive_avg_pool2d")


CASES_4D = [
    ((1, 1, 1, 1), (1, 1)),
    ((1, 1, 1, 16), (1, 16)),
    ((1, 1, 16, 1), (16, 1)),
    ((1, 1, 4, 4), (2, 2)),
    ((1, 1, 4, 4), (4, 4)),
    ((1, 3, 8, 8), (1, 1)),
    ((1, 2, 8, 1), (4, 1)),
    ((1, 2, 1, 8), (1, 4)),
    ((2, 3, 7, 5), (3, 2)),
    ((1, 1, 7, 5), (4, 3)),
    ((1, 2, 5, 7), (2, 4)),
    ((1, 2, 9, 7), (4, 3)),
    ((2, 4, 6, 10), (6, 5)),
    ((1, 1, 5, 9), (2, 4)),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,output_size", CASES_4D)
def test_adaptive_avg_pool2d_4d(dtype, shape, output_size):
    x = _make_input(shape, dtype, seed=sum(shape) + output_size[0] * 17 + output_size[1])
    actual = torch.ops.ops_multimodal_fusion.adaptive_avg_pool2d(x.npu(), list(output_size))
    expected = _reference(x, output_size)
    _assert_close(actual, expected, f"4d dtype={dtype} shape={shape} out={output_size}")


CASES_3D = [
    ((1, 1, 1), (1, 1)),
    ((1, 4, 4), (2, 2)),
    ((1, 4, 4), (4, 4)),
    ((2, 1, 8), (1, 4)),
    ((2, 8, 1), (4, 1)),
    ((3, 8, 6), (4, 3)),
    ((2, 7, 5), (1, 1)),
    ((2, 7, 5), (4, 2)),
    ((4, 5, 9), (5, 3)),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,output_size", CASES_3D)
def test_adaptive_avg_pool2d_3d(dtype, shape, output_size):
    x = _make_input(shape, dtype, seed=sum(shape) + output_size[0] * 31 + output_size[1])
    actual = torch.ops.ops_multimodal_fusion.adaptive_avg_pool2d(x.npu(), list(output_size))
    expected = _reference(x, output_size)
    _assert_close(actual, expected, f"3d dtype={dtype} shape={shape} out={output_size}")



@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_adaptive_avg_pool2d_empty_batch(dtype):
    x = torch.empty((0, 3, 4, 5), dtype=dtype)
    actual = torch.ops.ops_multimodal_fusion.adaptive_avg_pool2d(x.npu(), [2, 1]).cpu()
    assert actual.dtype == dtype
    assert actual.shape == (0, 3, 2, 1)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_adaptive_avg_pool2d_known_values(dtype):
    x = torch.arange(1, 1 + 1 * 1 * 4 * 6, dtype=torch.float32).reshape(1, 1, 4, 6)
    x = x.to(dtype)
    actual = torch.ops.ops_multimodal_fusion.adaptive_avg_pool2d(x.npu(), [2, 3])
    expected = _reference(x, (2, 3))
    _assert_close(actual, expected, f"known values dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_adaptive_avg_pool2d_extreme_values(dtype):
    x = torch.tensor(
        [[[[1000.0, -1000.0, 0.5, -0.5], [3.0, -7.0, 11.0, -13.0],
           [17.0, -19.0, 23.0, -29.0], [31.0, -37.0, 41.0, -43.0]]]],
        dtype=dtype,
    )
    actual = torch.ops.ops_multimodal_fusion.adaptive_avg_pool2d(x.npu(), [2, 2])
    expected = _reference(x, (2, 2))
    _assert_close(actual, expected, f"extreme values dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_adaptive_avg_pool2d_invalid_ndim():
    x = torch.randn(1, 2, 3, 4, 5)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.adaptive_avg_pool2d(x.npu(), [2, 2])


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_adaptive_avg_pool2d_invalid_dtype():
    x = torch.randint(0, 5, (1, 1, 4, 4), dtype=torch.int32)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.adaptive_avg_pool2d(x.npu(), [2, 2])


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("output_size", [[0, 2], [2, 0], [-1, 2]])
def test_adaptive_avg_pool2d_invalid_output_size(output_size):
    x = torch.randn(1, 1, 4, 4)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.adaptive_avg_pool2d(x.npu(), output_size)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_adaptive_avg_pool2d_rejects_upsampling():
    x = torch.randn(1, 1, 4, 4)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.adaptive_avg_pool2d(x.npu(), [5, 4])
