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


if not hasattr(torch.ops.ops_multimodal_fusion, "max_unpool3d"):
    pytest.skip(
        "ops_multimodal_fusion.max_unpool3d not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float16, torch.float32]


@dataclass(frozen=True)
class MaxUnpool3dParams:
    kernel_size: Union[int, Tuple[int, ...]]
    stride: Optional[Union[int, Tuple[int, ...]]] = None
    padding: Optional[Union[int, Tuple[int, ...]]] = None
    output_size: Optional[Union[int, Tuple[int, ...]]] = None


@dataclass(frozen=True)
class MaxUnpool3dCase:
    shape: Tuple[int, ...]
    kernel_size: Union[int, Tuple[int, ...]]
    stride: Optional[Union[int, Tuple[int, ...]]] = None
    padding: Optional[Union[int, Tuple[int, ...]]] = None
    output_size: Optional[Union[int, Tuple[int, ...]]] = None


def _make_input(shape, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(*shape, dtype=torch.float32).uniform_(-3.0, 3.0, generator=g)
    if x.numel() > 12:
        flat = x.flatten()
        flat[1::7] = -2.0
        flat[3::11] = 0.0
        flat[5::13] = 2.5
    return x.to(dtype)


def _pool(x, kernel_size, stride=None, padding=0):
    ref_x = x.float() if x.dtype == torch.float16 else x
    values, indices = F.max_pool3d(
        ref_x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        return_indices=True,
    )
    return values.to(x.dtype), indices


def _custom(input_, indices, params):
    stride_arg = [] if params.stride is None else list(params.stride)
    padding_arg = [] if params.padding is None else list(params.padding)
    output_arg = [] if params.output_size is None else list(params.output_size)
    return torch.ops.ops_multimodal_fusion.max_unpool3d(
        input_.npu(),
        indices.npu(),
        list(params.kernel_size),
        stride_arg,
        padding_arg,
        output_arg,
    ).cpu()


def _reference(input_, indices, params):
    padding = 0 if params.padding is None else params.padding
    return F.max_unpool3d(
        input_,
        indices,
        kernel_size=params.kernel_size,
        stride=params.stride,
        padding=padding,
        output_size=params.output_size,
    )


def _assert_equal(actual, expected, label):
    assert actual.dtype == expected.dtype, (
        f"{label}: dtype mismatch {actual.dtype} vs {expected.dtype}"
    )
    assert actual.shape == expected.shape, (
        f"{label}: shape mismatch {actual.shape} vs {expected.shape}"
    )
    assert torch.equal(actual, expected), (
        f"{label}: values mismatch actual={actual.flatten()[:12]} "
        f"expected={expected.flatten()[:12]}"
    )


def test_max_unpool3d_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "max_unpool3d")


CASES_5D = [
    MaxUnpool3dCase((1, 1, 4, 4, 4), (2, 2, 2), (2, 2, 2), (0, 0, 0), None),
    MaxUnpool3dCase((1, 2, 5, 4, 6), (2, 2, 3), (1, 2, 2), (0, 0, 0), (1, 2, 5, 4, 6)),
    MaxUnpool3dCase((2, 2, 4, 5, 4), (2, 3, 2), (2, 1, 2), (0, 1, 0), None),
    MaxUnpool3dCase((1, 1, 3, 3, 3), (1, 2, 2), (1, 1, 1), (0, 0, 0), None),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES_5D)
def test_max_unpool3d_5d(dtype, case):
    x = _make_input(case.shape, dtype, seed=sum(case.shape) + sum(case.kernel_size) * 17)
    pooled, indices = _pool(x, case.kernel_size, stride=case.stride, padding=case.padding)
    params = MaxUnpool3dParams(
        case.kernel_size, case.stride, case.padding, case.output_size
    )
    actual = _custom(pooled, indices, params)
    expected = _reference(pooled, indices, params)
    _assert_equal(actual, expected, f"5d dtype={dtype} shape={case.shape}")


CASES_4D = [
    MaxUnpool3dCase((1, 4, 4, 4), (2, 2, 2), (2, 2, 2), (0, 0, 0), None),
    MaxUnpool3dCase((2, 5, 4, 6), (2, 2, 3), (1, 2, 2), (0, 0, 0), (5, 4, 6)),
    MaxUnpool3dCase((1, 3, 4, 3), (1, 2, 1), (1, 1, 1), (0, 0, 0), None),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES_4D)
def test_max_unpool3d_4d(dtype, case):
    x = _make_input(case.shape, dtype, seed=sum(case.shape) + sum(case.stride) * 19)
    pooled, indices = _pool(x, case.kernel_size, stride=case.stride, padding=case.padding)
    params = MaxUnpool3dParams(
        case.kernel_size, case.stride, case.padding, case.output_size
    )
    actual = _custom(pooled, indices, params)
    expected = _reference(pooled, indices, params)
    _assert_equal(actual, expected, f"4d dtype={dtype} shape={case.shape}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_max_unpool3d_stride_default(dtype):
    x = _make_input((1, 1, 4, 4, 4), dtype, seed=101)
    pooled, indices = _pool(x, (2, 2, 2), stride=None, padding=0)
    params = MaxUnpool3dParams((2, 2, 2), None, None)
    actual = _custom(pooled, indices, params)
    expected = _reference(pooled, indices, params)
    _assert_equal(actual, expected, f"stride-default dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_max_unpool3d_spatial_output_size(dtype):
    x = _make_input((1, 1, 4, 4, 4), dtype, seed=202)
    pooled, indices = _pool(x, (2, 2, 2), stride=(2, 2, 2), padding=0)
    params = MaxUnpool3dParams((2, 2, 2), (2, 2, 2), (0, 0, 0), (4, 4, 4))
    actual = _custom(pooled, indices, params)
    expected = _reference(pooled, indices, params)
    _assert_equal(actual, expected, f"spatial-output dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_max_unpool3d_full_output_size(dtype):
    x = _make_input((1, 2, 4, 4, 4), dtype, seed=303)
    pooled, indices = _pool(x, (2, 2, 2), stride=(2, 2, 2), padding=0)
    params = MaxUnpool3dParams((2, 2, 2), (2, 2, 2), (0, 0, 0), (1, 2, 4, 4, 4))
    actual = _custom(pooled, indices, params)
    expected = _reference(pooled, indices, params)
    _assert_equal(actual, expected, f"full-output dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_max_unpool3d_single_value_params(dtype):
    x = _make_input((1, 1, 4, 4, 4), dtype, seed=309)
    pooled, indices = _pool(x, 2, stride=2, padding=0)
    params = MaxUnpool3dParams((2,), (2,), (0,))
    actual = _custom(pooled, indices, params)
    expected = _reference(pooled, indices, MaxUnpool3dParams(2, 2, 0))
    _assert_equal(actual, expected, f"single-value-params dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_max_unpool3d_4d_accepts_five_element_output_size(dtype):
    x = _make_input((2, 4, 4, 4), dtype, seed=311)
    pooled, indices = _pool(x, (2, 2, 2), stride=(2, 2, 2), padding=0)
    params = MaxUnpool3dParams((2, 2, 2), (2, 2, 2), (0, 0, 0), (7, 9, 4, 4, 4))
    actual = _custom(pooled, indices, params)
    expected = _reference(pooled, indices, params)
    _assert_equal(actual, expected, f"4d-five-output-size dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_max_unpool3d_ambiguous_without_output_size(dtype):
    x = _make_input((1, 2, 5, 4, 6), dtype, seed=313)
    kernel_size = (2, 2, 3)
    stride = (1, 2, 2)
    padding = (0, 0, 0)
    pooled, indices = _pool(x, kernel_size, stride=stride, padding=padding)
    params = MaxUnpool3dParams(kernel_size, stride, padding)
    with pytest.raises(RuntimeError):
        _reference(pooled, indices, params)
    with pytest.raises(RuntimeError):
        _custom(pooled, indices, params)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_max_unpool3d_special_shape_single_depth(dtype):
    x = _make_input((1, 1, 1, 4, 4), dtype, seed=404)
    pooled, indices = _pool(x, (1, 2, 2), stride=(1, 2, 2), padding=0)
    params = MaxUnpool3dParams((1, 2, 2), (1, 2, 2), (0, 0, 0))
    actual = _custom(pooled, indices, params)
    expected = _reference(pooled, indices, params)
    _assert_equal(actual, expected, f"single-depth dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_max_unpool3d_empty_batch(dtype):
    x = torch.empty((0, 2, 2, 2, 2), dtype=dtype)
    indices = torch.empty((0, 2, 2, 2, 2), dtype=torch.int64)
    actual = _custom(x, indices, MaxUnpool3dParams((2, 2, 2), (2, 2, 2), (0, 0, 0)))
    assert actual.dtype == dtype
    assert actual.shape == (0, 2, 4, 4, 4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_max_unpool3d_invalid_indices_dtype():
    x = torch.randn(1, 1, 2, 2, 2)
    indices = torch.zeros_like(x, dtype=torch.int32)
    with pytest.raises(RuntimeError):
        _custom(x, indices, MaxUnpool3dParams((2, 2, 2), (2, 2, 2), (0, 0, 0)))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_max_unpool3d_invalid_indices_shape():
    x = torch.randn(1, 1, 2, 2, 2)
    indices = torch.zeros(1, 1, 2, 2, 1, dtype=torch.int64)
    with pytest.raises(RuntimeError):
        _custom(x, indices, MaxUnpool3dParams((2, 2, 2), (2, 2, 2), (0, 0, 0)))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("kernel_size", [(0, 2, 2), (2, -1, 2)])
def test_max_unpool3d_invalid_kernel(kernel_size):
    x = torch.randn(1, 1, 2, 2, 2)
    indices = torch.zeros_like(x, dtype=torch.int64)
    with pytest.raises(RuntimeError):
        _custom(x, indices, MaxUnpool3dParams(kernel_size, (2, 2, 2), (0, 0, 0)))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "stride,padding",
    [
        ((0, 2, 2), (0, 0, 0)),
        ((2, -1, 2), (0, 0, 0)),
        ((2, 2), (0, 0, 0)),
        ((2, 2, 2), (-1, 0, 0)),
        ((2, 2, 2), (0, 0)),
    ],
)


def test_max_unpool3d_invalid_stride_padding(stride, padding):
    x = torch.randn(1, 1, 2, 2, 2)
    indices = torch.zeros_like(x, dtype=torch.int64)
    with pytest.raises(RuntimeError):
        _custom(x, indices, MaxUnpool3dParams((2, 2, 2), stride, padding))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_max_unpool3d_invalid_output_size():
    x = torch.randn(1, 1, 2, 2, 2)
    indices = torch.zeros_like(x, dtype=torch.int64)
    with pytest.raises(RuntimeError):
        _custom(
            x,
            indices,
            MaxUnpool3dParams((2, 2, 2), (2, 2, 2), (0, 0, 0), (16, 4, 4)),
        )
    with pytest.raises(RuntimeError):
        _custom(
            x,
            indices,
            MaxUnpool3dParams((2, 2, 2), (2, 2, 2), (0, 0, 0), (1, 4, 4, 4)),
        )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_max_unpool3d_invalid_index_out_of_range():
    x = torch.randn(1, 1, 1, 1, 1)
    indices = torch.tensor([[[[[8]]]]], dtype=torch.int64)
    with pytest.raises(RuntimeError):
        _custom(x, indices, MaxUnpool3dParams((2, 2, 2), (2, 2, 2), (0, 0, 0)))
