#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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


if not hasattr(torch.ops.ops_multimodal_fusion, "upsample_trilinear3d"):
    pytest.skip(
        "ops_multimodal_fusion.upsample_trilinear3d not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float16, torch.float32]


def _make_tensor(shape, dtype, seed, low=-1.0, high=1.0):
    gen = torch.Generator().manual_seed(seed)
    return torch.empty(*shape, dtype=torch.float32).uniform_(low, high, generator=gen).to(dtype)


def _reference(input_, output_size, align_corners):
    return F.interpolate(
        input_.float(),
        size=tuple(output_size),
        mode="trilinear",
        align_corners=align_corners,
    ).to(input_.dtype)


def _reference_scale(input_, scales, align_corners):
    return F.interpolate(
        input_.float(),
        scale_factor=tuple(scales),
        mode="trilinear",
        align_corners=align_corners,
    ).to(input_.dtype)


def _custom(input_, output_size, align_corners=False, scales=None):
    if scales is None:
        out = torch.ops.ops_multimodal_fusion.upsample_trilinear3d(
            input_.npu(), list(output_size), align_corners
        )
    else:
        out = torch.ops.ops_multimodal_fusion.upsample_trilinear3d(
            input_.npu(), list(output_size), align_corners, scales[0], scales[1], scales[2]
        )
    return out.cpu()


def _assert_close(actual, expected, label):
    assert actual.shape == expected.shape, (
        f"{label}: shape mismatch {actual.shape} vs {expected.shape}"
    )
    assert actual.dtype == expected.dtype, (
        f"{label}: dtype mismatch {actual.dtype} vs {expected.dtype}"
    )
    rtol, atol = (3e-2, 3e-2) if expected.dtype == torch.float16 else (5e-4, 5e-4)
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"{label}: max_abs={(actual.float() - expected.float()).abs().max().item()} "
        f"actual={actual} expected={expected}"
    )


def _size_case(shape, output_size, dtype, seed, align_corners):
    input_ = _make_tensor(shape, dtype, seed)
    actual = _custom(input_, output_size, align_corners=align_corners)
    expected = _reference(input_, output_size, align_corners)
    _assert_close(
        actual,
        expected,
        f"shape={shape} output_size={output_size} dtype={dtype} align={align_corners}",
    )


def _scale_case(shape, scales, dtype, seed, align_corners):
    input_ = _make_tensor(shape, dtype, seed)
    expected = _reference_scale(input_, scales, align_corners)
    actual = _custom(input_, expected.shape[-3:], align_corners=align_corners, scales=scales)
    _assert_close(
        actual,
        expected,
        f"shape={shape} scales={scales} dtype={dtype} align={align_corners}",
    )


def test_upsample_trilinear3d_interface_exists():
    assert hasattr(torch.ops.ops_multimodal_fusion, "upsample_trilinear3d")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize(
    "shape,output_size",
    [
        ((1, 1, 2, 2, 2), (3, 3, 3)),
        ((2, 2, 3, 4, 5), (4, 5, 7)),
        ((1, 2, 4, 5, 6), (2, 3, 4)),
        ((1, 1, 3, 4, 5), (1, 1, 1)),
        ((1, 2, 1, 3, 4), (3, 5, 6)),
        ((2, 1, 3, 1, 2), (4, 3, 5)),
    ],
)
def test_upsample_trilinear3d_size_matches_torch(dtype, align_corners, shape, output_size):
    seed = 100 + sum(shape) * 7 + sum(output_size) * 11 + int(align_corners)
    _size_case(shape, output_size, dtype, seed, align_corners)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize(
    "shape,scales",
    [
        ((1, 1, 3, 4, 5), (1.7, 1.6, 1.4)),
        ((2, 1, 2, 3, 4), (2.0, 1.0, 1.5)),
        ((1, 2, 5, 4, 3), (0.6, 0.75, 1.8)),
    ],
)
def test_upsample_trilinear3d_scale_factor_matches_torch(dtype, align_corners, shape, scales):
    seed = 500 + sum(shape) * 13 + int(sum(scales) * 100) + int(align_corners)
    _scale_case(shape, scales, dtype, seed, align_corners)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_upsample_trilinear3d_known_eight_neighbor_values(dtype):
    input_ = torch.arange(8, dtype=dtype).reshape(1, 1, 2, 2, 2)
    actual = _custom(input_, (3, 3, 3), align_corners=False)
    expected = torch.tensor(
        [
            [
                [
                    [0.0, 0.5, 1.0],
                    [1.0, 1.5, 2.0],
                    [2.0, 2.5, 3.0],
                ],
                [
                    [2.0, 2.5, 3.0],
                    [3.0, 3.5, 4.0],
                    [4.0, 4.5, 5.0],
                ],
                [
                    [4.0, 4.5, 5.0],
                    [5.0, 5.5, 6.0],
                    [6.0, 6.5, 7.0],
                ],
            ]
        ],
        dtype=dtype,
    ).reshape(1, 1, 3, 3, 3)
    _assert_close(actual, expected, f"known-eight-neighbor dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "align_corners,expected_values",
    [
        (False, [0.0, 0.1, 0.5, 0.9, 1.0]),
        (True, [0.0, 0.25, 0.5, 0.75, 1.0]),
    ],
)
def test_upsample_trilinear3d_singleton_dims_known_width_values(
    dtype, align_corners, expected_values
):
    input_ = torch.tensor([[[[[0.0, 1.0]]]]], dtype=dtype)
    actual = _custom(input_, (1, 1, 5), align_corners=align_corners)
    expected = torch.tensor(expected_values, dtype=dtype).reshape(1, 1, 1, 1, 5)
    _assert_close(actual, expected, f"singleton-known-width dtype={dtype} align={align_corners}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_upsample_trilinear3d_scale_factor_uses_torch_default_not_recompute(dtype):
    input_ = torch.arange(3 * 4 * 5, dtype=dtype).reshape(1, 1, 3, 4, 5)
    scales = (1.7, 1.6, 1.4)
    actual = _custom(input_, (5, 6, 7), align_corners=False, scales=scales)
    expected = F.interpolate(
        input_.float(),
        scale_factor=scales,
        mode="trilinear",
        align_corners=False,
    ).to(dtype)
    recomputed = F.interpolate(
        input_.float(),
        scale_factor=scales,
        mode="trilinear",
        align_corners=False,
        recompute_scale_factor=True,
    ).to(dtype)
    assert not torch.allclose(expected.float(), recomputed.float(), rtol=1e-4, atol=1e-4)
    _assert_close(actual, expected, f"scale-factor-default dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_upsample_trilinear3d_rejects_scale_output_size_mismatch():
    input_ = torch.randn(1, 1, 3, 4, 5, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_, (6, 6, 7), align_corners=False, scales=(1.7, 1.6, 1.4))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_upsample_trilinear3d_rejects_mixed_scale_path():
    input_ = torch.randn(1, 1, 3, 4, 5, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_, (5, 6, 7), align_corners=False, scales=(-1.0, 1.6, 1.4))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", [(1, 2, 3, 4), (1, 1, 2, 3, 4, 5)])
def test_upsample_trilinear3d_rejects_invalid_rank(shape):
    input_ = torch.randn(shape, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_, (2, 3, 4))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("output_size", [(0, 2, 2), (2, -1, 2), (2, 2, 0), (2, 2)])
def test_upsample_trilinear3d_rejects_invalid_output_size(output_size):
    input_ = torch.randn(1, 1, 2, 2, 2, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_, output_size)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "shape",
    [
        (0, 1, 2, 2, 2),
        (1, 0, 2, 2, 2),
        (1, 1, 0, 2, 2),
        (1, 1, 2, 0, 2),
        (1, 1, 2, 2, 0),
    ],
)
def test_upsample_trilinear3d_rejects_empty_dimensions(shape):
    input_ = torch.empty(shape, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_, (2, 2, 2))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "scales",
    [
        (0.0, 1.0, 1.0),
        (1.0, -2.0, 1.0),
        (1.0, 1.0, float("nan")),
        (float("inf"), 1.0, 1.0),
    ],
)
def test_upsample_trilinear3d_rejects_invalid_scale(scales):
    input_ = torch.randn(1, 1, 3, 4, 5, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_, (3, 4, 5), scales=scales)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_upsample_trilinear3d_rejects_unsupported_dtype():
    input_ = torch.randint(0, 5, (1, 1, 2, 2, 2), dtype=torch.int32)
    with pytest.raises(RuntimeError):
        _custom(input_, (3, 3, 3))
