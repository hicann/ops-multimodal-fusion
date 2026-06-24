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


if not hasattr(torch.ops.ops_multimodal_fusion, "fractional_max_pool3d"):
    pytest.skip(
        "ops_multimodal_fusion.fractional_max_pool3d not registered for current NPU_ARCH; "
        "skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float16, torch.float32]


def _make_input(shape, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(*shape, dtype=torch.float32).uniform_(-2.0, 2.0, generator=g)
    if x.numel() > 12:
        flat = x.flatten()
        flat[1::7] = 0.0
        flat[3::11] = -1.5
        flat[5::13] = 1.25
    return x.to(dtype)


def _make_random_samples(n, c, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.empty((n, c, 3), dtype=torch.float32).uniform_(0.0, 1.0, generator=g)


def _reference(x, kernel_size, output_size, random_samples):
    is5d = x.dim() == 5
    x5 = x if is5d else x.unsqueeze(0)
    rs = random_samples
    if rs.dim() == 2:
        rs = rs.unsqueeze(0)
    x_ref = x5.float() if x5.dtype == torch.float16 else x5
    val, idx = F.fractional_max_pool3d(
        x_ref,
        kernel_size=kernel_size,
        output_size=output_size,
        return_indices=True,
        _random_samples=rs,
    )
    val = val.to(x5.dtype)
    if not is5d:
        val = val.squeeze(0)
        idx = idx.squeeze(0)
    return val, idx


def _custom(x, kernel_size, output_size, random_samples):
    return torch.ops.ops_multimodal_fusion.fractional_max_pool3d(
        x.npu(), list(kernel_size), list(output_size), random_samples.npu()
    )


def _verify_outputs(actual_val, actual_idx, expected_val, expected_idx, label):
    val_cpu = actual_val.cpu()
    idx_cpu = actual_idx.cpu()
    rtol, atol = (8e-3, 8e-3) if expected_val.dtype == torch.float16 else (1e-5, 1e-5)
    torch.testing.assert_close(val_cpu, expected_val, rtol=rtol, atol=atol, msg=label)
    if idx_cpu.dtype != torch.int64:
        raise AssertionError(f"{label}: indices dtype {idx_cpu.dtype} != int64")
    if idx_cpu.shape != expected_idx.shape:
        raise AssertionError(
            f"{label}: index shape {idx_cpu.shape} vs {expected_idx.shape}"
        )
    if not torch.equal(idx_cpu, expected_idx):
        raise AssertionError(
            f"{label}: indices diverge actual={idx_cpu.flatten()[:8]} "
            f"expected={expected_idx.flatten()[:8]}"
        )


def _run_case(x, kernel_size, output_size, random_samples, label):
    actual_val, actual_idx = _custom(x, kernel_size, output_size, random_samples)
    expected_val, expected_idx = _reference(x, kernel_size, output_size, random_samples)
    _verify_outputs(actual_val, actual_idx, expected_val, expected_idx, label)


def test_fractional_max_pool3d_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "fractional_max_pool3d")


CASES_5D = [
    ((1, 1, 4, 4, 4), (2, 2, 2), (2, 2, 2)),
    ((1, 2, 5, 4, 5), (2, 2, 3), (3, 2, 2)),
    ((2, 1, 4, 5, 4), (1, 3, 2), (3, 2, 2)),
    ((1, 1, 5, 5, 5), (3, 2, 2), (2, 3, 3)),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,kernel_size,output_size", CASES_5D)
def test_fractional_max_pool3d_5d(dtype, shape, kernel_size, output_size):
    n, c = shape[0], shape[1]
    seed = sum(shape) + kernel_size[0] * 7 + kernel_size[2] * 11 + output_size[0] * 13
    x = _make_input(shape, dtype, seed=seed)
    random_samples = _make_random_samples(n, c, seed=seed + 17)
    _run_case(
        x,
        kernel_size,
        output_size,
        random_samples,
        f"5d dtype={dtype} shape={shape} k={kernel_size} out={output_size}",
    )


CASES_4D = [
    ((1, 4, 4, 4), (2, 2, 2), (2, 2, 2)),
    ((2, 5, 4, 5), (2, 2, 3), (3, 2, 2)),
    ((2, 4, 5, 4), (1, 3, 2), (3, 2, 2)),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,kernel_size,output_size", CASES_4D)
def test_fractional_max_pool3d_4d(dtype, shape, kernel_size, output_size):
    c = shape[0]
    seed = sum(shape) + kernel_size[1] * 31 + output_size[2] * 41
    x = _make_input(shape, dtype, seed=seed)
    random_samples = _make_random_samples(1, c, seed=seed + 23)
    _run_case(
        x,
        kernel_size,
        output_size,
        random_samples,
        f"4d dtype={dtype} shape={shape} k={kernel_size} out={output_size}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fractional_max_pool3d_known_values(dtype):
    x = torch.arange(1, 1 + 4 * 4 * 5, dtype=torch.float32).reshape(1, 1, 4, 4, 5)
    x = x.to(dtype)
    random_samples = torch.tensor([[[0.25, 0.50, 0.75]]], dtype=torch.float32)

    actual_val, actual_idx = _custom(x, (2, 2, 2), (2, 2, 2), random_samples)
    expected_val, expected_idx = _reference(x, (2, 2, 2), (2, 2, 2), random_samples)
    _verify_outputs(
        actual_val,
        actual_idx,
        expected_val,
        expected_idx,
        f"known-values dtype={dtype}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fractional_max_pool3d_tie_break_first(dtype):
    x = torch.full((1, 1, 4, 4, 4), 7.0, dtype=dtype)
    random_samples = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)

    actual_val, actual_idx = _custom(x, (2, 2, 2), (2, 2, 2), random_samples)
    expected_val, expected_idx = _reference(x, (2, 2, 2), (2, 2, 2), random_samples)
    _verify_outputs(
        actual_val,
        actual_idx,
        expected_val,
        expected_idx,
        f"tie-break dtype={dtype}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fractional_max_pool3d_random_samples_2d_for_4d(dtype):
    x = _make_input((3, 5, 4, 5), dtype, seed=2468)
    random_samples = torch.tensor(
        [[0.10, 0.90, 0.30], [0.35, 0.20, 0.80], [0.80, 0.45, 0.15]],
        dtype=torch.float32,
    )
    actual_val, actual_idx = _custom(x, (2, 2, 2), (3, 2, 3), random_samples)
    expected_val, expected_idx = _reference(x, (2, 2, 2), (3, 2, 3), random_samples)
    _verify_outputs(
        actual_val,
        actual_idx,
        expected_val,
        expected_idx,
        f"4d-random-2d dtype={dtype}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fractional_max_pool3d_multi_batch_channel_independent(dtype):
    x = _make_input((2, 2, 5, 5, 5), dtype, seed=1122)
    random_samples = torch.tensor(
        [
            [[0.05, 0.95, 0.25], [0.40, 0.10, 0.80]],
            [[0.85, 0.20, 0.55], [0.15, 0.70, 0.35]],
        ],
        dtype=torch.float32,
    )

    actual_val, actual_idx = _custom(x, (2, 2, 2), (3, 3, 3), random_samples)
    expected_val, expected_idx = _reference(x, (2, 2, 2), (3, 3, 3), random_samples)
    _verify_outputs(
        actual_val,
        actual_idx,
        expected_val,
        expected_idx,
        f"multi-batch-channel dtype={dtype}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fractional_max_pool3d_random_sample_boundary_order(dtype):
    x = torch.arange(5 * 5 * 5, dtype=torch.float32).reshape(1, 1, 5, 5, 5)
    x = (x * 0.125 - 3.0).to(dtype)
    random_samples = torch.tensor([[[0.999, 0.0, 0.501]]], dtype=torch.float32)

    actual_val, actual_idx = _custom(x, (2, 2, 2), (3, 3, 3), random_samples)
    expected_val, expected_idx = _reference(x, (2, 2, 2), (3, 3, 3), random_samples)
    _verify_outputs(
        actual_val,
        actual_idx,
        expected_val,
        expected_idx,
        f"random-boundary-order dtype={dtype}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fractional_max_pool3d_non_contiguous_input(dtype):
    base = _make_input((1, 1, 6, 5, 5), dtype, seed=3344)
    x = base[:, :, ::2, :, :]
    assert not x.is_contiguous()
    random_samples = torch.tensor([[[0.25, 0.75, 0.50]]], dtype=torch.float32)

    actual_val, actual_idx = _custom(x, (1, 2, 2), (2, 3, 3), random_samples)
    expected_val, expected_idx = _reference(x, (1, 2, 2), (2, 3, 3), random_samples)
    _verify_outputs(
        actual_val,
        actual_idx,
        expected_val,
        expected_idx,
        f"non-contiguous-input dtype={dtype}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fractional_max_pool3d_non_contiguous_random_samples(dtype):
    x = _make_input((1, 2, 5, 5, 5), dtype, seed=4455)
    random_base = torch.tensor(
        [[[0.10, 9.0, 0.90, 9.0, 0.30, 9.0],
          [0.80, 9.0, 0.20, 9.0, 0.60, 9.0]]],
        dtype=torch.float32,
    )
    random_samples = random_base[:, :, ::2]
    assert not random_samples.is_contiguous()

    actual_val, actual_idx = _custom(x, (2, 2, 2), (3, 3, 3), random_samples)
    expected_val, expected_idx = _reference(x, (2, 2, 2), (3, 3, 3), random_samples)
    _verify_outputs(
        actual_val,
        actual_idx,
        expected_val,
        expected_idx,
        f"non-contiguous-random dtype={dtype}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fractional_max_pool3d_kernel_size_one_sparse_identity(dtype):
    x = _make_input((1, 2, 3, 4, 5), dtype, seed=9753)
    random_samples = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.99, 0.99, 0.99]]], dtype=torch.float32
    )
    actual_val, actual_idx = _custom(x, (1, 1, 1), (2, 3, 4), random_samples)
    expected_val, expected_idx = _reference(x, (1, 1, 1), (2, 3, 4), random_samples)
    _verify_outputs(
        actual_val,
        actual_idx,
        expected_val,
        expected_idx,
        f"kernel-one-sparse dtype={dtype}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fractional_max_pool3d_output_dim_one_boundary(dtype):
    x = _make_input((1, 2, 4, 5, 5), dtype, seed=8642)
    random_samples = torch.tensor(
        [[[0.0, 0.999, 0.25], [0.999, 0.0, 0.75]]], dtype=torch.float32
    )
    actual_val, actual_idx = _custom(x, (2, 2, 2), (1, 3, 3), random_samples)
    expected_val, expected_idx = _reference(x, (2, 2, 2), (1, 3, 3), random_samples)
    _verify_outputs(
        actual_val,
        actual_idx,
        expected_val,
        expected_idx,
        f"out-dim-one dtype={dtype}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fractional_max_pool3d_empty_batch(dtype):
    x = torch.empty((0, 3, 4, 5, 5), dtype=dtype)
    random_samples = torch.empty((0, 3, 3), dtype=torch.float32)
    actual_val, actual_idx = _custom(x, (2, 2, 2), (2, 2, 2), random_samples)
    actual_val = actual_val.cpu()
    actual_idx = actual_idx.cpu()
    assert actual_val.dtype == dtype
    assert actual_idx.dtype == torch.int64
    assert actual_val.shape == (0, 3, 2, 2, 2)
    assert actual_idx.shape == (0, 3, 2, 2, 2)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fractional_max_pool3d_empty_channel(dtype):
    x = torch.empty((1, 0, 4, 5, 5), dtype=dtype)
    random_samples = torch.empty((1, 0, 3), dtype=torch.float32)
    actual_val, actual_idx = _custom(x, (2, 2, 2), (2, 2, 2), random_samples)
    actual_val = actual_val.cpu()
    actual_idx = actual_idx.cpu()
    assert actual_val.dtype == dtype
    assert actual_idx.dtype == torch.int64
    assert actual_val.shape == (1, 0, 2, 2, 2)
    assert actual_idx.shape == (1, 0, 2, 2, 2)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fractional_max_pool3d_empty_spatial_dim():
    x = torch.empty((1, 1, 0, 4, 4), dtype=torch.float32)
    rs = torch.zeros((1, 1, 3), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(x, (1, 2, 2), (1, 1, 1), rs)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fractional_max_pool3d_invalid_dtype():
    x = torch.randint(0, 5, (1, 1, 4, 4, 4), dtype=torch.int32)
    rs = torch.zeros((1, 1, 3), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(x, (2, 2, 2), (2, 2, 2), rs)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fractional_max_pool3d_invalid_ndim():
    x = torch.randn(1, 2, 3)
    rs = torch.zeros((1, 2, 3), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(x, (2, 2, 2), (1, 1, 1), rs)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("kernel_size", [[0, 2, 2], [2, 0, 2], [2, 2, -1], [2, 2]])
def test_fractional_max_pool3d_invalid_kernel_size(kernel_size):
    x = torch.randn(1, 1, 4, 4, 4)
    rs = torch.zeros((1, 1, 3), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(x, kernel_size, (2, 2, 2), rs)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("output_size", [[0, 2, 2], [2, 0, 2], [2, 2, -1], [2, 2]])
def test_fractional_max_pool3d_invalid_output_size(output_size):
    x = torch.randn(1, 1, 4, 4, 4)
    rs = torch.zeros((1, 1, 3), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(x, (2, 2, 2), output_size, rs)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fractional_max_pool3d_kernel_too_large():
    x = torch.randn(1, 1, 4, 4, 4)
    rs = torch.zeros((1, 1, 3), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(x, (5, 2, 2), (1, 1, 1), rs)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fractional_max_pool3d_kernel_width_exceeds_reg_lanes():
    x = torch.randn(1, 1, 3, 3, 66)
    rs = torch.zeros((1, 1, 3), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(x, (1, 1, 65), (1, 1, 1), rs)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fractional_max_pool3d_output_too_large():
    x = torch.randn(1, 1, 4, 4, 4)
    rs = torch.zeros((1, 1, 3), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(x, (2, 2, 2), (3, 2, 2), rs)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fractional_max_pool3d_output_equal_input_rejected():
    x = torch.randn(1, 1, 3, 4, 5)
    rs = torch.zeros((1, 1, 3), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(x, (1, 1, 1), (3, 4, 5), rs)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("in_d,k_d,out_d", [(7, 2, 5), (6, 2, 4), (10, 2, 8)])
def test_fractional_max_pool3d_output_max_boundary_matches_torch(in_d, k_d, out_d):
    x = _make_input((1, 1, in_d, 4, 4), torch.float32, seed=7000 + in_d)
    rs = torch.tensor([[[0.25, 0.50, 0.75]]], dtype=torch.float32)

    actual_val, actual_idx = _custom(x, (k_d, 2, 2), (out_d, 2, 2), rs)
    expected_val, expected_idx = _reference(x, (k_d, 2, 2), (out_d, 2, 2), rs)
    _verify_outputs(
        actual_val,
        actual_idx,
        expected_val,
        expected_idx,
        f"output-max-boundary in={in_d} k={k_d} out={out_d}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fractional_max_pool3d_output_above_max_boundary_rejected():
    x = torch.randn(1, 1, 7, 4, 4)
    rs = torch.zeros((1, 1, 3), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(x, (2, 2, 2), (6, 2, 2), rs)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fractional_max_pool3d_invalid_random_samples_dtype():
    x = torch.randn(1, 1, 4, 4, 4)
    rs = torch.zeros((1, 1, 3), dtype=torch.float64)
    with pytest.raises(RuntimeError):
        _custom(x, (2, 2, 2), (2, 2, 2), rs)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fractional_max_pool3d_invalid_random_samples_shape():
    x = torch.randn(2, 3, 5, 5, 5)
    rs = torch.zeros((1, 3, 3), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(x, (2, 2, 2), (3, 3, 3), rs)
