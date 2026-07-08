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


if not hasattr(torch.ops.ops_multimodal_fusion, "unpack_pivots"):
    pytest.skip(
        "ops_multimodal_fusion.unpack_pivots not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.int32, torch.int64]


def _reference(pivots, perm_size):
    pivot_i64 = pivots.to(torch.int64)
    k = pivot_i64.shape[-1]
    batch_shape = pivot_i64.shape[:-1]
    if k == 0:
        identity = torch.arange(perm_size, dtype=torch.int64)
        return identity.expand(*batch_shape, perm_size).clone()

    flat = pivot_i64.reshape(-1, k)
    rows = []
    for row in flat:
        perm = list(range(perm_size))
        for i, pivot in enumerate(row.tolist()):
            j = pivot - 1
            perm[i], perm[j] = perm[j], perm[i]
        rows.append(perm)
    return torch.tensor(rows, dtype=torch.int64).reshape(*batch_shape, perm_size)


def _custom(pivots, perm_size):
    return torch.ops.ops_multimodal_fusion.unpack_pivots(pivots.npu(), perm_size).cpu()


def _assert_equal(actual, expected, label):
    assert actual.dtype == torch.int64, f"{label}: dtype mismatch {actual.dtype}"
    assert actual.shape == expected.shape, (
        f"{label}: shape mismatch {actual.shape} vs {expected.shape}"
    )
    assert torch.equal(actual, expected), (
        f"{label}: values mismatch actual={actual} expected={expected}"
    )


def test_unpack_pivots_interface_exists():
    assert hasattr(torch.ops.ops_multimodal_fusion, "unpack_pivots")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "pivots,perm_size,expected",
    [
        ([1, 2, 3], 3, [0, 1, 2]),
        ([2, 2, 3], 3, [1, 0, 2]),
        ([2, 3], 4, [1, 2, 0, 3]),
        ([4, 4, 4], 4, [3, 0, 1, 2]),
        ([3, 1, 3], 5, [1, 2, 0, 3, 4]),
    ],
)


def test_unpack_pivots_known_values(dtype, pivots, perm_size, expected):
    piv = torch.tensor(pivots, dtype=dtype)
    actual = _custom(piv, perm_size)
    ref = torch.tensor(expected, dtype=torch.int64)
    _assert_equal(actual, ref, f"known dtype={dtype} pivots={pivots}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_unpack_pivots_batched_2d(dtype):
    pivots = torch.tensor(
        [
            [1, 2, 3],
            [2, 2, 3],
            [3, 3, 3],
        ],
        dtype=dtype,
    )
    actual = _custom(pivots, 3)
    expected = _reference(pivots, 3)
    _assert_equal(actual, expected, f"batched-2d dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_unpack_pivots_batched_3d(dtype):
    pivots = torch.tensor(
        [
            [[1, 2], [2, 2]],
            [[3, 1], [2, 3]],
        ],
        dtype=dtype,
    )
    actual = _custom(pivots, 4)
    expected = _reference(pivots, 4)
    _assert_equal(actual, expected, f"batched-3d dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_unpack_pivots_k_less_than_perm_size(dtype):
    pivots = torch.tensor([[2, 3], [1, 4]], dtype=dtype)
    actual = _custom(pivots, 5)
    expected = _reference(pivots, 5)
    _assert_equal(actual, expected, f"k-less-than-m dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_unpack_pivots_zero_length_pivots(dtype):
    pivots = torch.empty((0,), dtype=dtype)
    actual = _custom(pivots, 5)
    expected = _reference(pivots, 5)
    _assert_equal(actual, expected, f"zero-length-1d dtype={dtype}")

    batched = torch.empty((3, 0), dtype=dtype)
    actual_batched = _custom(batched, 4)
    expected_batched = _reference(batched, 4)
    _assert_equal(actual_batched, expected_batched, f"zero-length-2d dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_unpack_pivots_empty_batch(dtype):
    pivots = torch.empty((0, 3), dtype=dtype)
    actual = _custom(pivots, 5)
    expected = _reference(pivots, 5)
    _assert_equal(actual, expected, f"empty-batch dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_unpack_pivots_batched_4d(dtype):
    pivots = torch.tensor(
        [
            [
                [[1, 2, 3], [2, 2, 3]],
                [[3, 3, 3], [4, 4, 4]],
            ],
            [
                [[2, 3, 4], [3, 1, 3]],
                [[1, 4, 4], [2, 2, 4]],
            ],
        ],
        dtype=dtype,
    )
    actual = _custom(pivots, 4)
    expected = _reference(pivots, 4)
    _assert_equal(actual, expected, f"batched-4d dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_unpack_pivots_large_batch_uses_multiple_blocks(dtype):
    rows = [[1, 2, 3], [2, 2, 3], [3, 3, 3], [4, 4, 4]]
    pivots = torch.tensor(rows * 20, dtype=dtype)
    actual = _custom(pivots, 4)
    expected = _reference(pivots, 4)
    _assert_equal(actual, expected, f"large-batch dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_unpack_pivots_repeated_row_swaps(dtype):
    pivots = torch.tensor([[4, 4, 4, 4], [2, 3, 4, 4]], dtype=dtype)
    actual = _custom(pivots, 4)
    expected = _reference(pivots, 4)
    _assert_equal(actual, expected, f"repeated-swaps dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_unpack_pivots_permutation_matrix_equivalence(dtype):
    pivots = torch.tensor([2, 3, 3], dtype=dtype)
    perm = _custom(pivots, 4)
    matrix = torch.eye(4, dtype=torch.int64)[perm]
    expected_perm = _reference(pivots, 4)
    expected_matrix = torch.eye(4, dtype=torch.int64)[expected_perm]
    _assert_equal(perm, expected_perm, f"perm-matrix-vector dtype={dtype}")
    assert torch.equal(matrix, expected_matrix)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_unpack_pivots_rejects_rank0():
    pivots = torch.tensor(1, dtype=torch.int32)
    with pytest.raises(RuntimeError):
        _custom(pivots, 1)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("perm_size", [0, -1])
def test_unpack_pivots_rejects_invalid_perm_size(perm_size):
    pivots = torch.tensor([1], dtype=torch.int32)
    with pytest.raises(RuntimeError):
        _custom(pivots, perm_size)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_unpack_pivots_rejects_k_greater_than_perm_size():
    pivots = torch.tensor([1, 2, 3], dtype=torch.int32)
    with pytest.raises(RuntimeError):
        _custom(pivots, 2)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("pivots", [[0, 2], [1, 5]])
def test_unpack_pivots_rejects_pivot_out_of_range(pivots):
    piv = torch.tensor(pivots, dtype=torch.int32)
    with pytest.raises(RuntimeError):
        _custom(piv, 4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("pivots", [[0, 2], [1, 5]])
def test_unpack_pivots_rejects_int64_pivot_out_of_range(pivots):
    piv = torch.tensor(pivots, dtype=torch.int64)
    with pytest.raises(RuntimeError):
        _custom(piv, 4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_unpack_pivots_rejects_unsupported_dtype():
    pivots = torch.tensor([1, 2, 3], dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(pivots, 3)
