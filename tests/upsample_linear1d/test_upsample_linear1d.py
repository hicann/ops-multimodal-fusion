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


if not hasattr(torch.ops.ops_multimodal_fusion, "upsample_linear1d"):
    pytest.skip(
        "ops_multimodal_fusion.upsample_linear1d not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float16, torch.float32]


def _make_tensor(shape, dtype, seed, low=-1.0, high=1.0):
    gen = torch.Generator().manual_seed(seed)
    return torch.empty(*shape, dtype=torch.float32).uniform_(low, high, generator=gen).to(dtype)


def _reference(input_, output_size, align_corners):
    return F.interpolate(
        input_.float(),
        size=output_size,
        mode="linear",
        align_corners=align_corners,
    ).to(input_.dtype)


def _reference_scale(input_, scale, align_corners):
    return F.interpolate(
        input_.float(),
        scale_factor=scale,
        mode="linear",
        align_corners=align_corners,
    ).to(input_.dtype)


def _custom(input_, output_size, align_corners=False, scale=None):
    if scale is None:
        out = torch.ops.ops_multimodal_fusion.upsample_linear1d(
            input_.npu(), output_size, align_corners
        )
    else:
        out = torch.ops.ops_multimodal_fusion.upsample_linear1d(
            input_.npu(), output_size, align_corners, scale
        )
    return out.cpu()


def _assert_close(actual, expected, label):
    assert actual.shape == expected.shape, (
        f"{label}: shape mismatch {actual.shape} vs {expected.shape}"
    )
    assert actual.dtype == expected.dtype, (
        f"{label}: dtype mismatch {actual.dtype} vs {expected.dtype}"
    )
    rtol, atol = (2e-2, 2e-2) if expected.dtype == torch.float16 else (2e-4, 2e-4)
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"{label}: max_abs={(actual.float() - expected.float()).abs().max().item()} "
        f"actual={actual} expected={expected}"
    )


def _case(shape, output_size, dtype, seed, align_corners):
    input_ = _make_tensor(shape, dtype, seed)
    actual = _custom(input_, output_size, align_corners=align_corners)
    expected = _reference(input_, output_size, align_corners)
    _assert_close(
        actual,
        expected,
        f"shape={shape} output_size={output_size} dtype={dtype} align={align_corners}",
    )


def _scale_case(shape, scale, dtype, seed, align_corners):
    input_ = _make_tensor(shape, dtype, seed)
    expected = _reference_scale(input_, scale, align_corners)
    actual = _custom(input_, expected.size(-1), align_corners=align_corners, scale=scale)
    _assert_close(
        actual,
        expected,
        f"shape={shape} scale={scale} dtype={dtype} align={align_corners}",
    )


def test_upsample_linear1d_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "upsample_linear1d")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize(
    "shape,output_size",
    [
        ((1, 1, 3), 6),
        ((2, 3, 5), 8),
        ((1, 2, 7), 3),
        ((2, 1, 8), 5),
        ((1, 2, 5), 5),
        ((2, 1, 1), 4),
        ((1, 1, 4), 1),
    ],
)


def test_upsample_linear1d_size_matches_torch(dtype, align_corners, shape, output_size):
    seed = 100 + shape[0] * 17 + shape[1] * 11 + shape[2] * 7 + output_size + int(align_corners)
    _case(shape, output_size, dtype, seed, align_corners)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize(
    "shape,scale",
    [
        ((1, 1, 4), 1.7),
        ((2, 2, 6), 1.7),
        ((1, 3, 3), 2.3),
        ((2, 1, 7), 0.6),
    ],
)


def test_upsample_linear1d_scale_factor_matches_torch(dtype, align_corners, shape, scale):
    seed = 500 + shape[0] * 17 + shape[1] * 11 + shape[2] * 7 + int(scale * 10) + int(align_corners)
    _scale_case(shape, scale, dtype, seed, align_corners)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "align_corners,expected_values",
    [
        (False, [0.0, 0.5, 1.1666667, 1.8333333, 2.5, 3.0]),
        (True, [0.0, 0.6, 1.2, 1.8, 2.4, 3.0]),
    ],
)


def test_upsample_linear1d_known_size_values(dtype, align_corners, expected_values):
    input_ = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]], dtype=dtype)
    actual = _custom(input_, 6, align_corners=align_corners)
    expected = torch.tensor(expected_values, dtype=dtype).reshape(1, 1, -1)
    _assert_close(actual, expected, f"known-size dtype={dtype} align={align_corners}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "align_corners,expected_values",
    [
        (False, [0.0, 0.1, 0.5, 0.9, 1.0]),
        (True, [0.0, 0.25, 0.5, 0.75, 1.0]),
    ],
)


def test_upsample_linear1d_two_point_known_size_values(dtype, align_corners, expected_values):
    input_ = torch.tensor([[[0.0, 1.0]]], dtype=dtype)
    actual = _custom(input_, 5, align_corners=align_corners)
    expected = torch.tensor(expected_values, dtype=dtype).reshape(1, 1, -1)
    _assert_close(actual, expected, f"two-point-known-size dtype={dtype} align={align_corners}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_upsample_linear1d_scale_factor_uses_torch_default_not_recompute(dtype):
    input_ = torch.arange(4, dtype=dtype).reshape(1, 1, 4)
    actual = _custom(input_, 6, align_corners=False, scale=1.7)
    expected = F.interpolate(
        input_.float(),
        scale_factor=1.7,
        mode="linear",
        align_corners=False,
    ).to(dtype)
    recomputed = F.interpolate(
        input_.float(),
        scale_factor=1.7,
        mode="linear",
        align_corners=False,
        recompute_scale_factor=True,
    ).to(dtype)
    assert not torch.allclose(expected.float(), recomputed.float(), rtol=1e-4, atol=1e-4)
    _assert_close(actual, expected, f"scale-factor-default dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "values,scale,expected_values",
    [
        ([0.0, 1.0, 2.0, 3.0], 1.7, [0.0, 0.38235295, 0.9705882, 1.5588236, 2.1470587, 2.735294]),
        ([0.0, 1.0, 2.0], 2.3, [0.0, 0.15217394, 0.5869565, 1.0217391, 1.4565217, 1.8913043]),
    ],
)


def test_upsample_linear1d_scale_factor_known_values(dtype, values, scale, expected_values):
    input_ = torch.tensor(values, dtype=dtype).reshape(1, 1, -1)
    output_size = int(input_.size(-1) * scale)
    actual = _custom(input_, output_size, align_corners=False, scale=scale)
    expected = torch.tensor(expected_values, dtype=dtype).reshape(1, 1, -1)
    _assert_close(actual, expected, f"scale-factor-known scale={scale} dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize(
    "shape,scale",
    [
        ((1, 1, 5), 1.0),
        ((2, 2, 4), 2.0),
        ((1, 3, 1), 3.0),
    ],
)


def test_upsample_linear1d_scale_factor_boundaries(dtype, align_corners, shape, scale):
    seed = 900 + shape[0] * 17 + shape[1] * 11 + shape[2] * 7 + int(scale * 10) + int(align_corners)
    _scale_case(shape, scale, dtype, seed, align_corners)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_upsample_linear1d_rejects_scale_output_size_mismatch():
    input_ = torch.randn(1, 1, 4, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_, 7, align_corners=False, scale=1.7)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_upsample_linear1d_rejects_rank2():
    input_ = torch.randn(2, 3, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_, 6)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_upsample_linear1d_rejects_rank4():
    input_ = torch.randn(1, 1, 2, 3, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_, 6)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("output_size", [0, -1])
def test_upsample_linear1d_rejects_invalid_output_size(output_size):
    input_ = torch.randn(1, 1, 3, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_, output_size)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", [(0, 1, 3), (1, 0, 3), (1, 1, 0)])
def test_upsample_linear1d_rejects_empty_dimensions(shape):
    input_ = torch.empty(shape, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_, 4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("scale", [0.0, -2.0, float("nan"), float("inf")])
def test_upsample_linear1d_rejects_invalid_scale(scale):
    input_ = torch.randn(1, 1, 3, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_, 6, scale=scale)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_upsample_linear1d_rejects_unsupported_dtype():
    input_ = torch.randint(0, 5, (1, 1, 3), dtype=torch.int32)
    with pytest.raises(RuntimeError):
        _custom(input_, 6)
