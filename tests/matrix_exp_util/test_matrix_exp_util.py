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


if not hasattr(torch.ops.ops_multimodal_fusion, "matrix_exp_util"):
    pytest.skip(
        "ops_multimodal_fusion.matrix_exp_util not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float32, torch.float16]


def _reference(input_cpu, coefficients_cpu):
    dtype = input_cpu.dtype
    work_dtype = torch.float32
    spatial_shape = input_cpu.shape[1:]
    out = torch.zeros(
        (coefficients_cpu.shape[0], *spatial_shape),
        dtype=work_dtype,
    )
    for j in range(input_cpu.shape[0]):
        view_shape = (coefficients_cpu.shape[0],) + (1,) * len(spatial_shape)
        coef = coefficients_cpu[:, j].to(work_dtype).reshape(view_shape)
        out = out + coef * input_cpu[j].to(work_dtype)
    return out.to(dtype)


def _custom(input_cpu, coefficients_cpu):
    return torch.ops.ops_multimodal_fusion.matrix_exp_util(
        input_cpu.npu(), coefficients_cpu.npu()
    ).cpu()


def _assert_close(actual, expected, label):
    assert actual.shape == expected.shape, (
        f"{label}: shape mismatch {actual.shape} vs {expected.shape}"
    )
    assert actual.dtype == expected.dtype, (
        f"{label}: dtype mismatch {actual.dtype} vs {expected.dtype}"
    )
    if expected.dtype == torch.float16:
        assert torch.allclose(actual, expected, rtol=2e-3, atol=2e-3), (
            f"{label}: fp16 values mismatch actual={actual} expected={expected}"
        )
    else:
        assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5), (
            f"{label}: fp32 values mismatch actual={actual} expected={expected}"
        )


def test_matrix_exp_util_interface_exists():
    assert hasattr(torch.ops.ops_multimodal_fusion, "matrix_exp_util")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_matrix_exp_util_known_vector(dtype):
    input_cpu = torch.tensor([1.0, 2.0, -3.0], dtype=dtype)
    coefficients_cpu = torch.tensor(
        [[1.0, 0.0, 1.0], [0.5, -1.0, 2.0]],
        dtype=dtype,
    )
    expected = _reference(input_cpu, coefficients_cpu)
    actual = _custom(input_cpu, coefficients_cpu)
    _assert_close(actual, expected, f"known-vector dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_matrix_exp_util_t_one_matrix(dtype):
    input_cpu = torch.arange(1, 10, dtype=torch.float32).reshape(1, 3, 3).to(dtype)
    coefficients_cpu = torch.tensor([[2.0], [-1.0], [0.25]], dtype=dtype)
    expected = _reference(input_cpu, coefficients_cpu)
    actual = _custom(input_cpu, coefficients_cpu)
    _assert_close(actual, expected, f"t-one-matrix dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "dtype,t,n,spatial_shape",
    [
        (torch.float32, 4, 1, (5,)),
        (torch.float32, 4, 4, (2, 3)),
        (torch.float32, 5, 2, (4, 4)),
        (torch.float32, 3, 6, (2, 2, 2)),
        (torch.float16, 4, 1, (5,)),
        (torch.float16, 4, 4, (2, 3)),
        (torch.float16, 5, 2, (4, 4)),
        (torch.float16, 3, 6, (2, 2, 2)),
    ],
)


def test_matrix_exp_util_n_t_shape_variants(dtype, t, n, spatial_shape):
    numel = t
    for dim in spatial_shape:
        numel *= dim
    input_cpu = (
        torch.linspace(-1.5, 2.0, steps=numel, dtype=torch.float32)
        .reshape(t, *spatial_shape)
        .to(dtype)
    )
    coefficients_cpu = (
        torch.linspace(-0.75, 1.25, steps=n * t, dtype=torch.float32)
        .reshape(n, t)
        .to(dtype)
    )
    expected = _reference(input_cpu, coefficients_cpu)
    actual = _custom(input_cpu, coefficients_cpu)
    _assert_close(actual, expected, f"variants dtype={dtype} t={t} n={n}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_matrix_exp_util_zero_and_one_coefficients(dtype):
    input_cpu = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4).to(dtype)
    zero_coef = torch.zeros((2, 3), dtype=dtype)
    one_coef = torch.ones((2, 3), dtype=dtype)

    zero_actual = _custom(input_cpu, zero_coef)
    zero_expected = _reference(input_cpu, zero_coef)
    _assert_close(zero_actual, zero_expected, f"zero-coef dtype={dtype}")

    one_actual = _custom(input_cpu, one_coef)
    one_expected = _reference(input_cpu, one_coef)
    _assert_close(one_actual, one_expected, f"one-coef dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_matrix_exp_util_unit_row_selects_input_term(dtype):
    input_cpu = torch.arange(48, dtype=torch.float32).reshape(4, 3, 4).to(dtype)
    coefficients_cpu = torch.eye(4, dtype=dtype)
    actual = _custom(input_cpu, coefficients_cpu)
    expected = _reference(input_cpu, coefficients_cpu)
    _assert_close(actual, expected, f"unit-row dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_matrix_exp_util_larger_matrix_tile_fp32():
    input_cpu = torch.linspace(-0.2, 0.3, steps=5 * 32 * 32).reshape(5, 32, 32)
    coefficients_cpu = torch.tensor(
        [
            [1.0, -0.5, 0.25, -0.125, 0.0625],
            [0.0, 1.0, 1.0, 1.0, 1.0],
            [-1.0, 0.5, -0.25, 0.125, -0.0625],
        ],
        dtype=torch.float32,
    )
    expected = _reference(input_cpu, coefficients_cpu)
    actual = _custom(input_cpu, coefficients_cpu)
    _assert_close(actual, expected, "larger-matrix-fp32")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("t,n", [(8, 3), (10, 5), (14, 4)])
def test_matrix_exp_util_large_t_pade_like_terms(dtype, t, n):
    input_cpu = (
        torch.linspace(-0.25, 0.35, steps=t * 7, dtype=torch.float32)
        .reshape(t, 7)
        .to(dtype)
    )
    coefficients_cpu = (
        torch.linspace(-0.5, 0.75, steps=n * t, dtype=torch.float32)
        .reshape(n, t)
        .to(dtype)
    )
    expected = _reference(input_cpu, coefficients_cpu)
    actual = _custom(input_cpu, coefficients_cpu)
    _assert_close(actual, expected, f"large-t dtype={dtype} t={t} n={n}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_matrix_exp_util_matrix_64x64_fp32():
    input_cpu = torch.linspace(-0.125, 0.125, steps=6 * 64 * 64).reshape(6, 64, 64)
    coefficients_cpu = torch.tensor(
        [
            [1.0, -0.5, 0.25, -0.125, 0.0625, -0.03125],
            [0.0, 0.25, -0.5, 0.75, -1.0, 1.25],
        ],
        dtype=torch.float32,
    )
    expected = _reference(input_cpu, coefficients_cpu)
    actual = _custom(input_cpu, coefficients_cpu)
    _assert_close(actual, expected, "matrix-64x64-fp32")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_matrix_exp_util_matrix_256x256_forces_tiling_fp32():
    input_cpu = torch.linspace(-0.05, 0.05, steps=4 * 256 * 256).reshape(4, 256, 256)
    coefficients_cpu = torch.tensor(
        [[1.0, -0.75, 0.5, -0.25], [-0.5, 0.25, 0.125, 1.0]],
        dtype=torch.float32,
    )
    expected = _reference(input_cpu, coefficients_cpu)
    actual = _custom(input_cpu, coefficients_cpu)
    _assert_close(actual, expected, "matrix-256x256-tiling-fp32")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_matrix_exp_util_pade13_matrix_terms_fp32():
    input_cpu = torch.linspace(-0.1, 0.12, steps=14 * 32 * 32).reshape(14, 32, 32)
    coefficients_cpu = (
        torch.linspace(-0.75, 0.75, steps=3 * 14, dtype=torch.float32)
        .reshape(3, 14)
    )
    expected = _reference(input_cpu, coefficients_cpu)
    actual = _custom(input_cpu, coefficients_cpu)
    _assert_close(actual, expected, "pade13-matrix-fp32")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_matrix_exp_util_batched_matrix_16x16(dtype):
    input_cpu = (
        torch.linspace(-0.3, 0.4, steps=5 * 3 * 16 * 16, dtype=torch.float32)
        .reshape(5, 3, 16, 16)
        .to(dtype)
    )
    coefficients_cpu = (
        torch.tensor(
            [
                [1.0, 0.0, -0.5, 0.25, -0.125],
                [-0.25, 0.5, -0.75, 1.0, -1.25],
                [0.125, 0.25, 0.5, 1.0, 2.0],
            ],
            dtype=torch.float32,
        )
        .to(dtype)
    )
    expected = _reference(input_cpu, coefficients_cpu)
    actual = _custom(input_cpu, coefficients_cpu)
    _assert_close(actual, expected, f"batched-16x16 dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_matrix_exp_util_inner_size_one_high_rank(dtype):
    input_cpu = (
        torch.tensor([1.25, -2.0, 0.5, 3.0], dtype=torch.float32)
        .reshape(4, 1, 1, 1)
        .to(dtype)
    )
    coefficients_cpu = torch.tensor(
        [[2.0, -1.0, 0.25, -0.5], [-0.75, 0.5, 1.5, -2.0]],
        dtype=dtype,
    )
    expected = _reference(input_cpu, coefficients_cpu)
    actual = _custom(input_cpu, coefficients_cpu)
    _assert_close(actual, expected, f"inner-size-one dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_matrix_exp_util_mixed_sign_precision_stress(dtype):
    input_cpu = (
        torch.tensor(
            [
                -3.5, -1.25, 0.0, 2.75,
                1.5, -2.25, 3.0, -0.5,
                0.75, -1.75, 2.5, -3.0,
            ],
            dtype=torch.float32,
        )
        .reshape(3, 2, 2)
        .to(dtype)
    )
    coefficients_cpu = torch.tensor(
        [[8.0, -4.0, 2.0], [-7.5, 3.25, -1.5], [0.125, -0.25, 0.5]],
        dtype=dtype,
    )
    expected = _reference(input_cpu, coefficients_cpu)
    actual = _custom(input_cpu, coefficients_cpu)
    _assert_close(actual, expected, f"precision-stress dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_matrix_exp_util_empty_spatial_dimension(dtype):
    input_cpu = torch.empty((3, 0, 4), dtype=dtype)
    coefficients_cpu = torch.ones((2, 3), dtype=dtype)
    actual = _custom(input_cpu, coefficients_cpu)
    expected = _reference(input_cpu, coefficients_cpu)
    _assert_close(actual, expected, f"empty-spatial dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_matrix_exp_util_rejects_input_rank0():
    input_cpu = torch.tensor(1.0, dtype=torch.float32)
    coefficients_cpu = torch.ones((1, 1), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_cpu, coefficients_cpu)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("coefficients_cpu", [
    torch.ones(3, dtype=torch.float32),
    torch.ones((1, 2, 3), dtype=torch.float32),
])


def test_matrix_exp_util_rejects_bad_coefficients_rank(coefficients_cpu):
    input_cpu = torch.ones((3, 2), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_cpu, coefficients_cpu)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_matrix_exp_util_rejects_mismatched_t():
    input_cpu = torch.ones((3, 2), dtype=torch.float32)
    coefficients_cpu = torch.ones((2, 4), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_cpu, coefficients_cpu)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_matrix_exp_util_rejects_dtype_mismatch():
    input_cpu = torch.ones((3, 2), dtype=torch.float32)
    coefficients_cpu = torch.ones((2, 3), dtype=torch.float16)
    with pytest.raises(RuntimeError):
        _custom(input_cpu, coefficients_cpu)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_matrix_exp_util_rejects_unsupported_dtype():
    input_cpu = torch.ones((3, 2), dtype=torch.int32)
    coefficients_cpu = torch.ones((2, 3), dtype=torch.int32)
    with pytest.raises(RuntimeError):
        _custom(input_cpu, coefficients_cpu)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_matrix_exp_util_rejects_zero_t():
    input_cpu = torch.empty((0, 2), dtype=torch.float32)
    coefficients_cpu = torch.empty((2, 0), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_cpu, coefficients_cpu)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_matrix_exp_util_rejects_zero_n():
    input_cpu = torch.ones((3, 2), dtype=torch.float32)
    coefficients_cpu = torch.empty((0, 3), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(input_cpu, coefficients_cpu)
