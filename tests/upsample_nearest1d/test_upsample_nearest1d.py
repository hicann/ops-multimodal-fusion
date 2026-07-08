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


if not hasattr(torch.ops.ops_multimodal_fusion, "upsample_nearest1d"):
    pytest.skip(
        "ops_multimodal_fusion.upsample_nearest1d not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float16, torch.float32]


def _make_tensor(shape, dtype, seed, low=-1.0, high=1.0):
    gen = torch.Generator().manual_seed(seed)
    return torch.empty(*shape, dtype=torch.float32).uniform_(low, high, generator=gen).to(dtype)


def _reference(input_, output_size):
    return F.interpolate(input_.float(), size=output_size, mode="nearest").to(input_.dtype)


def _reference_scale(input_, scale):
    return F.interpolate(input_.float(), scale_factor=scale, mode="nearest").to(input_.dtype)


def _custom(input_, output_size, scale=None):
    if scale is None:
        out = torch.ops.ops_multimodal_fusion.upsample_nearest1d(input_.npu(), output_size)
    else:
        out = torch.ops.ops_multimodal_fusion.upsample_nearest1d(input_.npu(), output_size, scale)
    return out.cpu()


def _assert_close(actual, expected, label):
    assert actual.shape == expected.shape, (
        f"{label}: shape mismatch {actual.shape} vs {expected.shape}"
    )
    assert actual.dtype == expected.dtype, (
        f"{label}: dtype mismatch {actual.dtype} vs {expected.dtype}"
    )
    assert torch.equal(actual, expected), (
        f"{label}: actual={actual} expected={expected}"
    )


def _case(shape, output_size, dtype, seed):
    input_ = _make_tensor(shape, dtype, seed)
    actual = _custom(input_, output_size)
    expected = _reference(input_, output_size)
    _assert_close(actual, expected, f"shape={shape} output_size={output_size} dtype={dtype}")


def _scale_case(shape, scale, dtype, seed):
    input_ = _make_tensor(shape, dtype, seed)
    expected = _reference_scale(input_, scale)
    actual = _custom(input_, expected.size(-1), scale=scale)
    _assert_close(actual, expected, f"shape={shape} scale={scale} dtype={dtype}")


def test_upsample_nearest1d_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "upsample_nearest1d")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape,output_size",
    [
        ((1, 1, 3), 6),
        ((2, 3, 5), 8),
        ((1, 2, 5), 5),
        ((2, 1, 1), 4),
        ((1, 1, 4), 1),
    ],
)


def test_upsample_nearest1d_matches_torch(dtype, shape, output_size):
    seed = 100 + shape[0] * 17 + shape[1] * 11 + shape[2] * 7 + output_size
    _case(shape, output_size, dtype, seed)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_upsample_nearest1d_known_floor_mapping(dtype):
    input_ = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]], dtype=dtype)
    actual = _custom(input_, 8)
    expected = torch.tensor([[[0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0]]], dtype=dtype)
    _assert_close(actual, expected, f"known-floor-mapping dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_upsample_nearest1d_multi_channel_known_values(dtype):
    input_ = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
            [[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]],
        ],
        dtype=dtype,
    )
    actual = _custom(input_, 5)
    expected = _reference(input_, 5)
    _assert_close(actual, expected, f"multi-channel-known dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape,scale",
    [
        ((1, 1, 4), 1.7),
        ((2, 2, 6), 1.7),
        ((1, 3, 3), 2.3),
        ((2, 1, 7), 0.6),
    ],
)


def test_upsample_nearest1d_scale_factor_matches_torch(dtype, shape, scale):
    seed = 500 + shape[0] * 17 + shape[1] * 11 + shape[2] * 7 + int(scale * 10)
    _scale_case(shape, scale, dtype, seed)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_upsample_nearest1d_scale_factor_uses_torch_default_not_recompute(dtype):
    input_ = torch.arange(4, dtype=dtype).reshape(1, 1, 4)
    actual = _custom(input_, 6, scale=1.7)
    expected = F.interpolate(input_.float(), scale_factor=1.7, mode="nearest").to(dtype)
    recomputed = F.interpolate(
        input_.float(),
        scale_factor=1.7,
        mode="nearest",
        recompute_scale_factor=True,
    ).to(dtype)
    assert not torch.equal(expected, recomputed)
    _assert_close(actual, expected, f"scale-factor-default dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "values,scale,expected_values",
    [
        ([0.0, 1.0, 2.0, 3.0], 1.7, [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]),
        ([0.0, 1.0, 2.0], 2.3, [0.0, 0.0, 0.0, 1.0, 1.0, 2.0]),
    ],
)


def test_upsample_nearest1d_scale_factor_known_values(dtype, values, scale, expected_values):
    input_ = torch.tensor(values, dtype=dtype).reshape(1, 1, -1)
    output_size = int(input_.size(-1) * scale)
    actual = _custom(input_, output_size, scale=scale)
    expected = torch.tensor(expected_values, dtype=dtype).reshape(1, 1, -1)
    _assert_close(actual, expected, f"scale-factor-known scale={scale} dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape,scale",
    [
        ((1, 1, 5), 1.0),
        ((2, 2, 4), 2.0),
        ((1, 3, 1), 3.0),
    ],
)


def test_upsample_nearest1d_scale_factor_boundaries(dtype, shape, scale):
    seed = 900 + shape[0] * 17 + shape[1] * 11 + shape[2] * 7 + int(scale * 10)
    _scale_case(shape, scale, dtype, seed)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_upsample_nearest1d_rejects_scale_output_size_mismatch():
    input_ = torch.randn(1, 1, 4, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_, 7, scale=1.7)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_upsample_nearest1d_rejects_rank2():
    input_ = torch.randn(2, 3, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_, 6)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_upsample_nearest1d_rejects_rank4():
    input_ = torch.randn(1, 1, 2, 3, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_, 6)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("output_size", [0, -1])
def test_upsample_nearest1d_rejects_invalid_output_size(output_size):
    input_ = torch.randn(1, 1, 3, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_, output_size)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", [(0, 1, 3), (1, 0, 3), (1, 1, 0)])
def test_upsample_nearest1d_rejects_empty_dimensions(shape):
    input_ = torch.empty(shape, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_, 4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("scale", [0.0, -2.0, float("nan"), float("inf")])
def test_upsample_nearest1d_rejects_invalid_scale(scale):
    input_ = torch.randn(1, 1, 3, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_, 6, scale=scale)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_upsample_nearest1d_rejects_unsupported_dtype():
    input_ = torch.randint(0, 5, (1, 1, 3), dtype=torch.int32)
    with pytest.raises(RuntimeError):
        _custom(input_, 6)
