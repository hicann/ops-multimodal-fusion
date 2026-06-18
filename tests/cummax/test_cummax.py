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
import torch_npu  # noqa: F401

import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "cummax"):
    pytest.skip(
        "ops_multimodal_fusion.cummax not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float16, torch.float32]
CASES = [
    ((1,), 0),
    ((1, 1), 0),
    ((1, 1), 1),
    ((1, 1, 1), -1),
    ((7,), 0),
    ((8,), -1),
    ((9,), 0),
    ((3, 5), 0),
    ((3, 5), 1),
    ((3, 17), 1),
    ((2, 3, 7), 0),
    ((2, 3, 7), 1),
    ((2, 3, 7), -1),
    ((2, 3, 7), -2),
    ((2, 3, 5), -1),
    ((2, 1, 7), 1),
    ((1, 3, 1), 1),
    ((4, 1, 8), 1),
    ((2, 5, 1), 1),
    ((1, 257, 3), 1),
    ((2, 257, 3), 1),
    ((4, 1025), -1),
    ((1025, 3), 0),
]


def _make_input(shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(*shape, dtype=torch.float32).uniform_(-3.0, 3.0, generator=g)
    if len(shape) and shape[-1] >= 4:
        flat = x.flatten()
        flat[1::13] = 1.25
        flat[3::17] = 1.25
        flat[5::19] = -2.0
    return x.to(dtype)


def _assert_result(actual, expected, msg, equal_nan=False):
    actual_v, actual_i = actual
    expected_v, expected_i = expected
    assert actual_v.dtype == expected_v.dtype
    assert actual_i.dtype == torch.int64
    assert actual_v.shape == expected_v.shape
    assert actual_i.shape == expected_i.shape
    rtol, atol = (0, 0) if expected_v.dtype == torch.float16 else (1e-6, 1e-6)
    assert torch.allclose(
        actual_v.cpu(), expected_v.cpu(), rtol=rtol, atol=atol, equal_nan=equal_nan
    ), msg
    assert torch.equal(actual_i.cpu(), expected_i.cpu()), msg


def test_cummax_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "cummax")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape,dim", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_cummax_operator(shape, dim, dtype):
    seed = abs(hash((shape, dim, dtype))) % 10_000_000
    x = _make_input(shape, dtype, seed)

    actual = torch.ops.ops_multimodal_fusion.cummax(x.npu(), int(dim))
    expected = torch.cummax(x, dim=dim)

    _assert_result(
        actual, expected,
        f"cummax mismatch shape={shape}, dim={dim}, dtype={dtype}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_cummax_tie_matches_torch_latest_index(dtype):
    # PyTorch cummax updates the index on equal maxima, so the returned index is
    # the latest position where the running max appears.
    x = torch.tensor([[1.0, 3.0, 3.0, 2.0], [2.0, 2.0, 4.0, 4.0]], dtype=dtype)

    actual = torch.ops.ops_multimodal_fusion.cummax(x.npu(), 1)
    expected = torch.cummax(x, dim=1)

    _assert_result(actual, expected,
                   f"cummax tie mismatch dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_cummax_non_contiguous_input(dtype):
    x = _make_input((5, 7), dtype, 123).t()
    assert not x.is_contiguous()

    actual = torch.ops.ops_multimodal_fusion.cummax(x.npu(), 1)
    expected = torch.cummax(x, dim=1)

    _assert_result(actual, expected,
                   f"cummax non-contiguous mismatch dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_cummax_nan_propagates(dtype):
    x = torch.tensor([[1.0, 2.0, float("nan"), 3.0]], dtype=dtype)

    actual = torch.ops.ops_multimodal_fusion.cummax(x.npu(), 1)
    expected = torch.cummax(x, dim=1)

    _assert_result(actual, expected,
                   f"cummax nan propagation mismatch dtype={dtype}", equal_nan=True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "values",
    [
        [1.0, float("inf"), 2.0, float("-inf"), 3.0],
        [float("-inf"), -2.0, -1.0, float("-inf")],
    ],
)
def test_cummax_inf_values(dtype, values):
    x = torch.tensor([values], dtype=dtype)

    actual = torch.ops.ops_multimodal_fusion.cummax(x.npu(), 1)
    expected = torch.cummax(x, dim=1)

    _assert_result(actual, expected,
                   f"cummax inf mismatch dtype={dtype}, values={values}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_cummax_empty_tensor():
    x = torch.empty((0, 3), dtype=torch.float32).npu()
    values, indices = torch.ops.ops_multimodal_fusion.cummax(x, 0)
    assert values.cpu().shape == (0, 3)
    assert indices.cpu().shape == (0, 3)
    assert indices.cpu().dtype == torch.int64


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_cummax_empty_scan_dim():
    x = torch.empty((2, 0, 3), dtype=torch.float32).npu()
    values, indices = torch.ops.ops_multimodal_fusion.cummax(x, 1)
    assert values.cpu().shape == (2, 0, 3)
    assert indices.cpu().shape == (2, 0, 3)
    assert indices.cpu().dtype == torch.int64


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_cummax_empty_non_scan_dim():
    x = torch.empty((0, 3, 4), dtype=torch.float32).npu()
    values, indices = torch.ops.ops_multimodal_fusion.cummax(x, 1)
    assert values.cpu().shape == (0, 3, 4)
    assert indices.cpu().shape == (0, 3, 4)
    assert indices.cpu().dtype == torch.int64


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_cummax_scalar_tensor():
    x = torch.tensor(2.5, dtype=torch.float32).npu()
    values, indices = torch.ops.ops_multimodal_fusion.cummax(x, 0)
    assert values.cpu().shape == torch.Size([])
    assert indices.cpu().shape == torch.Size([])
    assert torch.equal(values.cpu(), torch.tensor(2.5, dtype=torch.float32))
    assert torch.equal(indices.cpu(), torch.tensor(0, dtype=torch.int64))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_cummax_rejects_unsupported_dtype():
    for unsupported in [torch.bfloat16, torch.float64, torch.int32]:
        x = torch.ones(4, dtype=unsupported).npu()
        with pytest.raises(RuntimeError, match="cummax"):
            torch.ops.ops_multimodal_fusion.cummax(x, 0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_cummax_rejects_invalid_dim():
    x = torch.ones((2, 3), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError, match="cummax"):
        torch.ops.ops_multimodal_fusion.cummax(x, 2)
