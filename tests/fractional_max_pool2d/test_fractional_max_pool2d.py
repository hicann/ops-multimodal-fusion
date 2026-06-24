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


if not hasattr(torch.ops.ops_multimodal_fusion, "fractional_max_pool2d"):
    pytest.skip(
        "ops_multimodal_fusion.fractional_max_pool2d not registered for current NPU_ARCH; "
        "skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float16, torch.float32]


def _make_input(shape, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(*shape, dtype=torch.float32).uniform_(-2.0, 2.0, generator=g)
    if x.numel() > 8:
        flat = x.flatten()
        flat[1::7] = 0.0
        flat[3::11] = -1.5
        flat[5::13] = 1.25
    return x.to(dtype)


def _make_random_samples(n, c, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.empty((n, c, 2), dtype=torch.float32).uniform_(0.0, 1.0, generator=g)


def _reference(x, kernel_size, output_size, random_samples):
    """Reference using PyTorch's fractional_max_pool2d. Inputs may be 3-D or 4-D."""
    is4d = x.dim() == 4
    x4 = x if is4d else x.unsqueeze(0)
    rs = random_samples
    if rs.dim() == 2:
        rs = rs.unsqueeze(0)
    x_ref = x4.float() if x4.dtype == torch.float16 else x4
    val, idx = F.fractional_max_pool2d(
        x_ref,
        kernel_size=kernel_size,
        output_size=output_size,
        return_indices=True,
        _random_samples=rs,
    )
    val = val.to(x4.dtype)
    if not is4d:
        val = val.squeeze(0)
        idx = idx.squeeze(0)
    return val, idx


def _assert_close(actual_val, actual_idx, expected_val, expected_idx, label):
    actual_val = actual_val.cpu()
    actual_idx = actual_idx.cpu()
    assert actual_val.dtype == expected_val.dtype, (
        f"{label}: dtype mismatch values {actual_val.dtype} vs {expected_val.dtype}"
    )
    assert actual_idx.dtype == torch.int64, (
        f"{label}: indices dtype {actual_idx.dtype} != int64"
    )
    assert actual_val.shape == expected_val.shape, (
        f"{label}: value shape {actual_val.shape} vs {expected_val.shape}"
    )
    assert actual_idx.shape == expected_idx.shape, (
        f"{label}: index shape {actual_idx.shape} vs {expected_idx.shape}"
    )

    rtol, atol = (8e-3, 8e-3) if expected_val.dtype == torch.float16 else (1e-5, 1e-5)
    assert torch.allclose(actual_val, expected_val, rtol=rtol, atol=atol), (
        f"{label}: values diverge max_abs="
        f"{(actual_val - expected_val).abs().max().item() if actual_val.numel() else 0.0} "
        f"actual={actual_val.flatten()[:8]} expected={expected_val.flatten()[:8]}"
    )
    assert torch.equal(actual_idx, expected_idx), (
        f"{label}: indices diverge actual={actual_idx.flatten()[:8]} "
        f"expected={expected_idx.flatten()[:8]}"
    )


def _run_case(x, kernel_size, output_size, random_samples, label):
    actual_val, actual_idx = torch.ops.ops_multimodal_fusion.fractional_max_pool2d(
        x.npu(),
        list(kernel_size),
        list(output_size),
        random_samples.npu(),
    )
    expected_val, expected_idx = _reference(x, kernel_size, output_size, random_samples)
    _assert_close(
        actual_val,
        actual_idx,
        expected_val,
        expected_idx,
        label,
    )


def test_fractional_max_pool2d_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "fractional_max_pool2d")


CASES_4D = [
    ((1, 1, 5, 5), (2, 2), (2, 2)),
    ((1, 2, 5, 7), (2, 2), (3, 4)),
    ((1, 2, 7, 5), (3, 2), (3, 3)),
    ((2, 3, 6, 8), (2, 3), (4, 4)),
    ((2, 3, 8, 6), (3, 2), (5, 4)),
    ((1, 4, 9, 9), (3, 3), (3, 3)),
    ((1, 1, 4, 4), (2, 2), (1, 1)),
    ((2, 2, 6, 6), (2, 2), (3, 3)),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,kernel_size,output_size", CASES_4D)
def test_fractional_max_pool2d_4d(dtype, shape, kernel_size, output_size):
    n, c = shape[0], shape[1]
    seed = sum(shape) + kernel_size[0] * 7 + kernel_size[1] * 11 + output_size[0] * 13
    x = _make_input(shape, dtype, seed=seed)
    random_samples = _make_random_samples(n, c, seed=seed + 17)

    _run_case(
        x,
        kernel_size,
        output_size,
        random_samples,
        f"4d dtype={dtype} shape={shape} k={kernel_size} out={output_size}",
    )


CASES_3D = [
    ((1, 5, 5), (2, 2), (2, 2)),
    ((2, 6, 7), (2, 3), (3, 3)),
    ((3, 8, 8), (3, 3), (3, 3)),
    ((2, 5, 5), (2, 2), (1, 1)),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,kernel_size,output_size", CASES_3D)
def test_fractional_max_pool2d_3d(dtype, shape, kernel_size, output_size):
    c = shape[0]
    seed = sum(shape) + kernel_size[0] * 31 + output_size[0] * 41
    x = _make_input(shape, dtype, seed=seed)
    random_samples = _make_random_samples(1, c, seed=seed + 23)

    _run_case(
        x,
        kernel_size,
        output_size,
        random_samples,
        f"3d dtype={dtype} shape={shape} k={kernel_size} out={output_size}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fractional_max_pool2d_known_values(dtype):
    x = torch.arange(1, 1 + 4 * 5, dtype=torch.float32).reshape(1, 1, 4, 5).to(dtype)
    random_samples = torch.tensor([[[0.25, 0.5]]], dtype=torch.float32)

    actual_val, actual_idx = torch.ops.ops_multimodal_fusion.fractional_max_pool2d(
        x.npu(), [2, 2], [2, 2], random_samples.npu()
    )
    expected_val, expected_idx = _reference(x, (2, 2), (2, 2), random_samples)
    _assert_close(
        actual_val,
        actual_idx,
        expected_val,
        expected_idx,
        f"known values dtype={dtype}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fractional_max_pool2d_tie_break_first(dtype):
    # All values equal. The strict-greater tie-break should pick the first cell
    # within each window. Window starts come from get_intervals; each window is
    # 2x2 inside a 4x4 input mapped to a 2x2 output, so each window has 4 ties
    # and the first cell of each window must be picked.
    x = torch.full((1, 1, 4, 4), 7.0, dtype=dtype)
    random_samples = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
    actual_val, actual_idx = torch.ops.ops_multimodal_fusion.fractional_max_pool2d(
        x.npu(), [2, 2], [2, 2], random_samples.npu()
    )
    expected_val, expected_idx = _reference(x, (2, 2), (2, 2), random_samples)
    _assert_close(
        actual_val, actual_idx, expected_val, expected_idx,
        f"tie-break dtype={dtype}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fractional_max_pool2d_random_sample_order(dtype):
    # PyTorch uses random_samples[..., 0] for W and random_samples[..., 1] for H.
    # This shape/sample pair produces different windows if the two entries are swapped.
    x = _make_input((1, 1, 5, 7), dtype, seed=1357)
    random_samples = torch.tensor([[[0.80, 0.15]]], dtype=torch.float32)
    actual_val, actual_idx = torch.ops.ops_multimodal_fusion.fractional_max_pool2d(
        x.npu(), [2, 2], [3, 4], random_samples.npu()
    )
    expected_val, expected_idx = _reference(x, (2, 2), (3, 4), random_samples)
    _assert_close(
        actual_val,
        actual_idx,
        expected_val,
        expected_idx,
        f"sample-order dtype={dtype}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fractional_max_pool2d_3d_random_samples_2d(dtype):
    x = _make_input((3, 5, 7), dtype, seed=2468)
    random_samples = torch.tensor(
        [[0.10, 0.90], [0.35, 0.20], [0.80, 0.45]], dtype=torch.float32
    )
    actual_val, actual_idx = torch.ops.ops_multimodal_fusion.fractional_max_pool2d(
        x.npu(), [2, 2], [3, 4], random_samples.npu()
    )
    expected_val, expected_idx = _reference(x, (2, 2), (3, 4), random_samples)
    _assert_close(
        actual_val,
        actual_idx,
        expected_val,
        expected_idx,
        f"3d-random-2d dtype={dtype}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fractional_max_pool2d_kernel_size_one_identity(dtype):
    x = _make_input((1, 2, 4, 5), dtype, seed=9753)
    random_samples = torch.tensor(
        [[[0.0, 0.0], [0.99, 0.99]]], dtype=torch.float32
    )
    actual_val, actual_idx = torch.ops.ops_multimodal_fusion.fractional_max_pool2d(
        x.npu(), [1, 1], [4, 5], random_samples.npu()
    )
    expected_val, expected_idx = _reference(x, (1, 1), (4, 5), random_samples)
    _assert_close(
        actual_val,
        actual_idx,
        expected_val,
        expected_idx,
        f"kernel-one dtype={dtype}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fractional_max_pool2d_sample_boundaries(dtype):
    x = _make_input((1, 2, 6, 6), dtype, seed=8642)
    random_samples = torch.tensor(
        [[[0.0, 0.0], [0.999, 0.999]]], dtype=torch.float32
    )
    actual_val, actual_idx = torch.ops.ops_multimodal_fusion.fractional_max_pool2d(
        x.npu(), [2, 2], [3, 3], random_samples.npu()
    )
    expected_val, expected_idx = _reference(x, (2, 2), (3, 3), random_samples)
    _assert_close(
        actual_val,
        actual_idx,
        expected_val,
        expected_idx,
        f"sample-boundary dtype={dtype}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fractional_max_pool2d_empty_batch(dtype):
    x = torch.empty((0, 3, 4, 5), dtype=dtype)
    random_samples = torch.empty((0, 3, 2), dtype=torch.float32)
    actual_val, actual_idx = torch.ops.ops_multimodal_fusion.fractional_max_pool2d(
        x.npu(), [2, 2], [2, 2], random_samples.npu()
    )
    actual_val = actual_val.cpu()
    actual_idx = actual_idx.cpu()
    assert actual_val.dtype == dtype
    assert actual_idx.dtype == torch.int64
    assert actual_val.shape == (0, 3, 2, 2)
    assert actual_idx.shape == (0, 3, 2, 2)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fractional_max_pool2d_invalid_dtype():
    x = torch.randint(0, 5, (1, 1, 4, 4), dtype=torch.int32)
    rs = torch.zeros((1, 1, 2), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.fractional_max_pool2d(x.npu(), [2, 2], [2, 2], rs.npu())


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fractional_max_pool2d_invalid_ndim():
    x = torch.randn(1, 2, 3, 4, 5)
    rs = torch.zeros((2, 3, 2), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.fractional_max_pool2d(x.npu(), [2, 2], [2, 2], rs.npu())


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("kernel_size", [[0, 2], [2, 0], [-1, 2]])
def test_fractional_max_pool2d_invalid_kernel_size(kernel_size):
    x = torch.randn(1, 1, 4, 4)
    rs = torch.zeros((1, 1, 2), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.fractional_max_pool2d(x.npu(), kernel_size, [2, 2], rs.npu())


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("output_size", [[0, 2], [2, 0], [-1, 2]])
def test_fractional_max_pool2d_invalid_output_size(output_size):
    x = torch.randn(1, 1, 4, 4)
    rs = torch.zeros((1, 1, 2), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.fractional_max_pool2d(x.npu(), [2, 2], output_size, rs.npu())


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fractional_max_pool2d_kernel_too_large():
    x = torch.randn(1, 1, 4, 4)
    rs = torch.zeros((1, 1, 2), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.fractional_max_pool2d(x.npu(), [5, 5], [1, 1], rs.npu())


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fractional_max_pool2d_output_too_large():
    x = torch.randn(1, 1, 4, 4)
    rs = torch.zeros((1, 1, 2), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.fractional_max_pool2d(x.npu(), [2, 2], [4, 2], rs.npu())


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fractional_max_pool2d_invalid_random_samples_dtype():
    x = torch.randn(1, 1, 4, 4)
    rs = torch.zeros((1, 1, 2), dtype=torch.float64)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.fractional_max_pool2d(x.npu(), [2, 2], [2, 2], rs.npu())


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fractional_max_pool2d_invalid_random_samples_shape():
    x = torch.randn(2, 3, 6, 6)
    rs = torch.zeros((1, 3, 2), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.fractional_max_pool2d(x.npu(), [2, 2], [3, 3], rs.npu())
