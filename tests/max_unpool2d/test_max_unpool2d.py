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


from dataclasses import dataclass
from typing import Optional, Tuple, Union

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "max_unpool2d"):
    pytest.skip(
        "ops_multimodal_fusion.max_unpool2d not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float16, torch.float32]


@dataclass(frozen=True)
class MaxUnpool2dParams:
    kernel_size: Union[int, Tuple[int, int]]
    stride: Optional[Union[int, Tuple[int, int]]] = None
    padding: Union[int, Tuple[int, int]] = 0
    output_size: Optional[Union[int, Tuple[int, int]]] = None


def _make_input(shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(*shape, dtype=torch.float32).uniform_(-3.0, 3.0, generator=g)
    if x.numel() > 8:
        flat = x.flatten()
        flat[1::5] += 10.0
        flat[3::7] -= 5.0
    return x.to(dtype)


def _pair(v):
    if isinstance(v, int):
        return (v, v)
    return tuple(v)


def _normalize_stride(kernel_size, stride):
    return _pair(kernel_size) if stride is None else _pair(stride)


def _normalize_output_size(pooled_shape, kernel_size, stride, padding, output_size):
    if output_size is not None:
        return _pair(output_size)
    return _default_output_size(
        pooled_shape, kernel_size, _normalize_stride(kernel_size, stride), padding
    )


def _default_output_size(pooled_shape, kernel_size, stride, padding):
    kernel_h, kernel_w = _pair(kernel_size)
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    in_h, in_w = pooled_shape[-2:]
    return (
        (in_h - 1) * stride_h - 2 * pad_h + kernel_h,
        (in_w - 1) * stride_w - 2 * pad_w + kernel_w,
    )


def _custom(pooled, indices, params):
    stride_pair = _normalize_stride(params.kernel_size, params.stride)
    output_pair = _normalize_output_size(
        pooled.shape, params.kernel_size, stride_pair, params.padding, params.output_size
    )
    return torch.ops.ops_multimodal_fusion.max_unpool2d(
        pooled.npu(),
        indices.npu(),
        list(_pair(params.kernel_size)),
        list(stride_pair),
        list(_pair(params.padding)),
        list(output_pair),
    )


def _reference(pooled, indices, params):
    ref_input = pooled.float() if pooled.dtype == torch.float16 else pooled
    out = F.max_unpool2d(
        ref_input,
        indices,
        kernel_size=params.kernel_size,
        stride=params.stride,
        padding=params.padding,
        output_size=params.output_size,
    )
    return out.to(pooled.dtype)


def _pooled_case(shape, dtype, params, seed):
    x = _make_input(shape, dtype, seed)
    ref_x = x.float() if dtype == torch.float16 else x
    pooled, indices = F.max_pool2d(
        ref_x,
        kernel_size=params.kernel_size,
        stride=params.stride,
        padding=params.padding,
        return_indices=True,
    )
    return pooled.to(dtype), indices


def _assert_close(actual, expected, label):
    actual = actual.cpu()
    assert actual.dtype == expected.dtype, (
        f"{label}: dtype mismatch {actual.dtype} vs {expected.dtype}"
    )
    assert actual.shape == expected.shape, (
        f"{label}: shape mismatch {actual.shape} vs {expected.shape}"
    )
    rtol, atol = (8e-3, 8e-3) if expected.dtype == torch.float16 else (1e-5, 1e-5)
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"{label}: max_abs={(actual - expected).abs().max().item() if actual.numel() else 0.0} "
        f"actual={actual.flatten()[:12]} expected={expected.flatten()[:12]}"
    )


def test_max_unpool2d_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "max_unpool2d")


CASES_4D = [
    ((1, 1, 4, 4), (2, 2), (2, 2), (0, 0)),
    ((1, 2, 5, 7), (2, 3), (2, 2), (0, 1)),
    ((2, 1, 6, 5), (3, 2), (2, 1), (1, 0)),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,kernel_size,stride,padding", CASES_4D)
def test_max_unpool2d_4d(dtype, shape, kernel_size, stride, padding):
    params = MaxUnpool2dParams(kernel_size, stride, padding, shape[-2:])
    pooled, indices = _pooled_case(
        shape, dtype, params, seed=sum(shape) + sum(kernel_size) * 17
    )
    actual = _custom(pooled, indices, params)
    expected = _reference(pooled, indices, params)
    _assert_close(actual, expected, f"4d dtype={dtype} shape={shape}")


CASES_3D = [
    ((1, 4, 4), (2, 2), (2, 2), (0, 0)),
    ((2, 5, 6), (2, 3), (2, 2), (0, 1)),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,kernel_size,stride,padding", CASES_3D)
def test_max_unpool2d_3d(dtype, shape, kernel_size, stride, padding):
    params = MaxUnpool2dParams(kernel_size, stride, padding, shape[-2:])
    pooled, indices = _pooled_case(
        shape, dtype, params, seed=sum(shape) + sum(stride) * 19
    )
    actual = _custom(pooled, indices, params)
    expected = _reference(pooled, indices, params)
    _assert_close(actual, expected, f"3d dtype={dtype} shape={shape}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_max_unpool2d_stride_none_semantics(dtype):
    kernel_size = (2, 2)
    padding = (0, 0)
    pool_params = MaxUnpool2dParams(kernel_size, kernel_size, padding)
    pooled, indices = _pooled_case((1, 1, 4, 4), dtype, pool_params, seed=123)
    params = MaxUnpool2dParams(
        kernel_size,
        None,
        padding,
        _default_output_size(pooled.shape, kernel_size, kernel_size, padding),
    )

    actual = _custom(pooled, indices, params)
    expected = F.max_unpool2d(
        pooled.float() if dtype == torch.float16 else pooled,
        indices,
        kernel_size=kernel_size,
        stride=None,
        padding=padding,
        output_size=params.output_size,
    ).to(dtype)
    _assert_close(actual, expected, f"stride-none dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_max_unpool2d_output_size_none_default(dtype):
    params = MaxUnpool2dParams((2, 2), (2, 2), (0, 0), None)
    pooled, indices = _pooled_case((1, 1, 4, 4), dtype, params, seed=234)

    actual = _custom(pooled, indices, params)
    expected = _reference(pooled, indices, params)
    _assert_close(actual, expected, f"output-size-none dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_max_unpool2d_scalar_params(dtype):
    pool_params = MaxUnpool2dParams(2, 2, 0)
    pooled, indices = _pooled_case((1, 1, 4, 4), dtype, pool_params, seed=345)
    params = MaxUnpool2dParams(2, None, 0, None)

    actual = _custom(pooled, indices, params)
    expected = _reference(pooled, indices, params)
    _assert_close(actual, expected, f"scalar-params dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_max_unpool2d_known_values(dtype):
    pooled = torch.tensor([[[[3.0, 4.0], [5.0, 6.0]]]], dtype=dtype)
    indices = torch.tensor([[[[0, 3], [12, 15]]]], dtype=torch.int64)
    params = MaxUnpool2dParams((2, 2), (2, 2), (0, 0), (4, 4))

    actual = _custom(pooled, indices, params)
    expected = _reference(pooled, indices, params)
    _assert_close(actual, expected, f"known-values dtype={dtype}")
    actual_cpu = actual.cpu()
    assert torch.count_nonzero(actual_cpu).item() == 4


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("output_size", [(3, 3), (5, 5)])
def test_max_unpool2d_explicit_output_size_legal_boundaries(dtype, output_size):
    pooled = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=dtype)
    indices = torch.tensor([[[[0, 2], [6, 8]]]], dtype=torch.int64)

    params = MaxUnpool2dParams((2, 2), (2, 2), (0, 0), output_size)
    actual = _custom(pooled, indices, params)
    expected = _reference(pooled, indices, params)
    _assert_close(actual, expected, f"legal-boundary dtype={dtype} out={output_size}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_max_unpool2d_explicit_output_size_larger_than_default(dtype):
    kernel_size = (2, 2)
    stride = (2, 2)
    padding = (0, 0)
    x = _make_input((1, 1, 5, 5), dtype, seed=456)
    pooled, indices = F.max_pool2d(
        x.float() if dtype == torch.float16 else x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        return_indices=True,
    )
    pooled = pooled.to(dtype)
    params = MaxUnpool2dParams(kernel_size, stride, padding, (5, 5))
    actual = _custom(pooled, indices, params)
    expected = _reference(pooled, indices, params)
    _assert_close(actual, expected, f"explicit-larger dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_max_unpool2d_duplicate_indices_last_write(dtype):
    pooled = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=dtype)
    indices = torch.tensor([[[[0, 0], [3, 3]]]], dtype=torch.int64)

    params = MaxUnpool2dParams((2, 2), (2, 2), (0, 0), (4, 4))
    actual = _custom(pooled, indices, params).cpu()
    expected = torch.zeros((1, 1, 4, 4), dtype=dtype)
    expected[0, 0, 0, 0] = 2.0
    expected[0, 0, 0, 3] = 4.0
    _assert_close(actual, expected, f"duplicate-last-write dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_max_unpool2d_empty_batch(dtype):
    pooled = torch.empty((0, 2, 2, 2), dtype=dtype)
    indices = torch.empty((0, 2, 2, 2), dtype=torch.int64)
    params = MaxUnpool2dParams((2, 2), (2, 2), (0, 0), (4, 4))
    actual = _custom(pooled, indices, params).cpu()
    assert actual.dtype == dtype
    assert actual.shape == (0, 2, 4, 4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_max_unpool2d_empty_channel(dtype):
    pooled = torch.empty((1, 0, 2, 2), dtype=dtype)
    indices = torch.empty((1, 0, 2, 2), dtype=torch.int64)
    params = MaxUnpool2dParams((2, 2), (2, 2), (0, 0), (4, 4))
    actual = _custom(pooled, indices, params).cpu()
    assert actual.dtype == dtype
    assert actual.shape == (1, 0, 4, 4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_max_unpool2d_invalid_dtype():
    pooled = torch.ones((1, 1, 2, 2), dtype=torch.int32)
    indices = torch.zeros((1, 1, 2, 2), dtype=torch.int64)
    with pytest.raises(RuntimeError):
        _custom(pooled, indices, MaxUnpool2dParams((2, 2), (2, 2), (0, 0), (4, 4)))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_max_unpool2d_invalid_rank():
    pooled = torch.ones((1, 2), dtype=torch.float32)
    indices = torch.zeros((1, 2), dtype=torch.int64)
    with pytest.raises(RuntimeError):
        _custom(pooled, indices, MaxUnpool2dParams((2, 2), (2, 2), (0, 0), (4, 4)))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_max_unpool2d_invalid_indices_dtype():
    pooled = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    indices = torch.zeros((1, 1, 2, 2), dtype=torch.int32)
    with pytest.raises(RuntimeError):
        _custom(pooled, indices, MaxUnpool2dParams((2, 2), (2, 2), (0, 0), (4, 4)))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_max_unpool2d_invalid_indices_shape():
    pooled = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    indices = torch.zeros((1, 1, 2, 3), dtype=torch.int64)
    with pytest.raises(RuntimeError):
        _custom(pooled, indices, MaxUnpool2dParams((2, 2), (2, 2), (0, 0), (4, 4)))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_max_unpool2d_empty_spatial_dim():
    pooled = torch.empty((1, 1, 0, 2), dtype=torch.float32)
    indices = torch.empty((1, 1, 0, 2), dtype=torch.int64)
    with pytest.raises(RuntimeError):
        _custom(pooled, indices, MaxUnpool2dParams((2, 2), (2, 2), (0, 0), (4, 4)))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "kernel_size,stride,padding,output_size",
    [
        ((0, 2), (2, 2), (0, 0), (4, 4)),
        ((2, 2), (0, 2), (0, 0), (4, 4)),
        ((2, 2), (2, 2), (-1, 0), (4, 4)),
        ((2,), (2, 2), (0, 0), (4, 4)),
        ((2, 2), (2, 2), (0, 0), (0, 4)),
        ((2, 2), (2, 2), (0, 0), (4,)),
    ],
)


def test_max_unpool2d_invalid_params(kernel_size, stride, padding, output_size):
    pooled = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    indices = torch.zeros((1, 1, 2, 2), dtype=torch.int64)
    with pytest.raises(RuntimeError):
        _custom(pooled, indices, MaxUnpool2dParams(kernel_size, stride, padding, output_size))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("output_size", [(2, 4), (6, 4), (4, 2), (4, 6), (7, 4)])
def test_max_unpool2d_invalid_output_size_range(output_size):
    pooled = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    indices = torch.zeros((1, 1, 2, 2), dtype=torch.int64)
    with pytest.raises(RuntimeError):
        _custom(pooled, indices, MaxUnpool2dParams((2, 2), (2, 2), (0, 0), output_size))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_max_unpool2d_indices_out_of_range():
    pooled = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    indices = torch.tensor([[[[0, 1], [2, 16]]]], dtype=torch.int64)
    with pytest.raises(RuntimeError):
        _custom(pooled, indices, MaxUnpool2dParams((2, 2), (2, 2), (0, 0), (4, 4)))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_max_unpool2d_negative_indices_out_of_range():
    pooled = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    indices = torch.tensor([[[[0, 1], [2, -1]]]], dtype=torch.int64)
    with pytest.raises(RuntimeError):
        _custom(pooled, indices, MaxUnpool2dParams((2, 2), (2, 2), (0, 0), (4, 4)))
