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

import math

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "pdist"):
    pytest.skip(
        "ops_multimodal_fusion.pdist not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _make_strided(shape, dtype=torch.float32, offset=0.0):
    base = torch.linspace(
        -0.45 + offset,
        0.55 + offset,
        steps=math.prod(shape) * 2,
        dtype=dtype,
    ).reshape(shape[0], shape[1] * 2)
    return base[:, ::2]


def _reference(x_cpu, p, dtype):
    x_work = x_cpu.to(torch.float32)
    expected = F.pdist(x_work, p=float(p))
    return expected.to(dtype)


def _custom(x_cpu, p=None):
    if p is None:
        return torch.ops.ops_multimodal_fusion.pdist(x_cpu.npu()).cpu()
    return torch.ops.ops_multimodal_fusion.pdist(x_cpu.npu(), float(p)).cpu()


def _assert_close(actual, expected, dtype, label):
    assert actual.shape == expected.shape, (
        f"{label}: shape mismatch {actual.shape} vs {expected.shape}"
    )
    assert actual.dtype == expected.dtype, (
        f"{label}: dtype mismatch {actual.dtype} vs {expected.dtype}"
    )
    assert torch.isfinite(actual.float()).all(), f"{label}: actual contains NaN/Inf"
    if dtype == torch.float32:
        assert torch.allclose(actual, expected, rtol=3e-4, atol=3e-4), (
            f"{label}: fp32 mismatch actual={actual} expected={expected}"
        )
    elif dtype == torch.float16:
        assert torch.allclose(actual.float(), expected.float(), rtol=6e-2, atol=6e-2), (
            f"{label}: fp16 mismatch actual={actual} expected={expected}"
        )
    else:
        assert torch.allclose(actual.float(), expected.float(), rtol=8e-2, atol=8e-2), (
            f"{label}: bf16 mismatch actual={actual} expected={expected}"
        )


def _run_case(x_cpu, p, dtype, label):
    x_typed = x_cpu.to(dtype)
    expected = _reference(x_typed, p, dtype)
    actual = _custom(x_typed, p)
    _assert_close(actual, expected, dtype, label)


def test_pdist_interface_exists():
    assert hasattr(torch.ops.ops_multimodal_fusion, "pdist")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("p", [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, float("inf")])
def test_pdist_p_branches_fp32(p):
    x = torch.tensor(
        [
            [0.0, 0.5, -0.25],
            [1.0, -0.5, 0.75],
            [0.0, 0.5, -0.25],
            [-0.75, 0.25, 0.5],
        ],
        dtype=torch.float32,
    )
    _run_case(x, p, torch.float32, f"p-branches p={p}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_pdist_default_p_is_two():
    x = torch.tensor([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]], dtype=torch.float32)
    expected = _reference(x, 2.0, torch.float32)
    actual = _custom(x)
    _assert_close(actual, expected, torch.float32, "default-p")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_pdist_upper_triangular_order_known_values():
    x = torch.tensor([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]], dtype=torch.float32)
    actual = _custom(x, 2.0)
    expected = torch.tensor([5.0, 10.0, 5.0], dtype=torch.float32)
    _assert_close(actual, expected, torch.float32, "upper-triangular-order")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape,p",
    [
        ((2, 4), 2.0),
        ((5, 3), 1.0),
        ((4, 5), 0.5),
        ((3, 16), 3.0),
        ((3, 1), float("inf")),
    ],
)


def test_pdist_shape_dtype_variants(dtype, shape, p):
    x = _make_strided(shape, dtype=torch.float32, offset=0.1)
    _run_case(x, p, dtype, f"shape-dtype dtype={dtype} shape={shape} p={p}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_pdist_general_p_identical_rows_zero_distance(dtype):
    x = torch.tensor(
        [
            [0.25, -0.5, 0.75, 0.0],
            [0.25, -0.5, 0.75, 0.0],
            [-0.25, 0.5, -0.75, 1.0],
            [0.25, -0.5, 0.75, 0.0],
        ],
        dtype=torch.float32,
    )
    x_typed = x.to(dtype)
    expected = _reference(x_typed, 0.5, dtype)
    actual = _custom(x_typed, 0.5)
    _assert_close(actual, expected, dtype, f"general-p-identical-rows dtype={dtype}")
    assert torch.allclose(
        actual.float()[torch.tensor([0, 2, 4])],
        torch.zeros(3),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_pdist_p_zero_counts_nonzero_differences(dtype):
    x = torch.tensor(
        [
            [0.0, -0.0, 1.0, -2.0, 3.0],
            [-0.0, 0.0, 1.0, -2.0, 4.0],
            [0.0, -0.0, 1.0, -2.0, 3.0],
            [1.0, -0.0, 0.0, -2.5, 3.0],
        ],
        dtype=torch.float32,
    )
    x_typed = x.to(dtype)
    expected = _reference(x_typed, 0.0, dtype)
    actual = _custom(x_typed, 0.0)
    _assert_close(actual, expected, dtype, f"p-zero-count dtype={dtype}")
    known_counts = torch.tensor([1.0, 0.0, 3.0, 1.0, 4.0, 3.0])
    assert torch.allclose(actual.float(), known_counts, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_pdist_multi_tile_feature_dim():
    x = torch.stack(
        (
            torch.linspace(-0.2, 0.4, steps=2304),
            torch.linspace(0.3, -0.5, steps=2304),
        )
    ).to(torch.float32)
    _run_case(x, 2.0, torch.float32, "multi-tile-feature-dim")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", [(0, 3), (1, 3), (3, 0)])
def test_pdist_empty_and_zero_feature_shapes(shape):
    x = torch.empty(shape, dtype=torch.float32)
    actual = _custom(x, 2.0)
    expected = _reference(x, 2.0, torch.float32)
    _assert_close(actual, expected, torch.float32, f"empty shape={shape}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_pdist_non_contiguous_input():
    x = _make_strided((4, 6), dtype=torch.float32, offset=-0.1)
    assert not x.is_contiguous()
    _run_case(x, 1.5, torch.float32, "non-contiguous")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_pdist_non_contiguous_bfloat16_input():
    x = _make_strided((3, 6), dtype=torch.bfloat16, offset=0.2)
    assert not x.is_contiguous()
    expected = _reference(x, 2.0, torch.bfloat16)
    actual = _custom(x, 2.0)
    _assert_close(actual, expected, torch.bfloat16, "non-contiguous-bf16")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_pdist_invalid_args():
    x = torch.randn(3, 4, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.pdist(torch.randn(4).npu(), 2.0)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.pdist(torch.randn(2, 3, 4).npu(), 2.0)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.pdist(x.npu(), -1.0)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.pdist(x.npu(), float("nan"))
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.pdist(x.to(torch.int32).npu(), 2.0)
