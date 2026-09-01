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

"""Test suite for foreach_trunc; golden reference: torch.trunc per tensor."""

import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "foreach_trunc"):
    pytest.skip(
        "ops_multimodal_fusion.foreach_trunc not registered for current NPU_ARCH; "
        "skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float32, torch.float16]


def _reference(tensors):
    """Golden: torch.trunc on each tensor on CPU."""
    return [torch.trunc(t.cpu()) for t in tensors]


def _custom(tensors):
    out = torch.ops.ops_multimodal_fusion.foreach_trunc([t.npu() for t in tensors])
    return [t.cpu() for t in out]


def _assert_close(actual_list, expected_list, dtype, label, check_signed_zero=False):
    assert len(actual_list) == len(expected_list), (
        f"{label}: length mismatch {len(actual_list)} vs {len(expected_list)}"
    )
    for i, (actual, expected) in enumerate(zip(actual_list, expected_list)):
        assert actual.shape == expected.shape, (
            f"{label}[{i}]: shape mismatch {actual.shape} vs {expected.shape}"
        )
        assert actual.dtype == expected.dtype, (
            f"{label}[{i}]: dtype mismatch {actual.dtype} vs {expected.dtype}"
        )
        actual_f32 = actual.float()
        expected_f32 = expected.float()
        if dtype == torch.float32 or actual.dtype == torch.float32:
            rtol, atol = 0.0, 0.0
        else:
            rtol, atol = 1e-3, 1e-3
        assert torch.allclose(
            actual_f32, expected_f32, rtol=rtol, atol=atol, equal_nan=True
        ), f"{label}[{i}]: actual={actual} expected={expected}"
        if check_signed_zero:
            zero_mask = expected_f32 == 0
            assert torch.equal(
                torch.signbit(actual_f32)[zero_mask],
                torch.signbit(expected_f32)[zero_mask],
            ), f"{label}[{i}]: signed zero mismatch actual={actual} expected={expected}"


def _run_case(tensors, dtype, label, check_signed_zero=False):
    typed = [t.to(dtype) for t in tensors]
    actual = _custom(typed)
    expected = _reference(typed)
    _assert_close(actual, expected, dtype, label, check_signed_zero)


def _make_tensors(shapes, seed=2026):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(*shape, generator=gen, dtype=torch.float32) * 10.0
            for shape in shapes]


def test_foreach_trunc_interface_exists():
    assert hasattr(torch.ops.ops_multimodal_fusion, "foreach_trunc")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_trunc_single_tensor(dtype):
    _run_case([torch.tensor([-2.9, -1.1, -0.5, 0.5, 1.1, 2.9])],
              dtype, "single", check_signed_zero=True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_trunc_two_tensors(dtype):
    tensors = [
        torch.tensor([-3.7, -2.0, -0.1, 0.0], dtype=torch.float32),
        torch.tensor([[0.2, 1.2, 2.9], [-1.9, -0.9, -0.0]], dtype=torch.float32),
    ]
    _run_case(tensors, dtype, "two-tensors", check_signed_zero=True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_trunc_eight_tensors(dtype):
    tensors = _make_tensors(
        [(1,), (3,), (2, 2), (2, 3, 4), (7,), (4, 5), (1, 1, 9), (16,)],
        seed=3030,
    )
    typed = [t.to(dtype) for t in tensors]
    actual = _custom(typed)
    expected = _reference(typed)
    assert len(actual) == 8
    _assert_close(actual, expected, dtype, "eight-tensors", check_signed_zero=True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_trunc_known_values(dtype):
    x = torch.tensor([
        -3.9, -3.1, -3.0, -2.5, -1.9, -1.1,
        -0.9, -0.5, -0.1, -1.0e-3, -0.0, 0.0,
        0.1, 0.5, 0.9, 1.1, 1.9, 2.5, 3.0, 3.9,
    ], dtype=torch.float32)
    _run_case([x], dtype, "known-values", check_signed_zero=True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_trunc_special_values_fp32():
    x = torch.tensor([float("nan"), float("inf"), float("-inf"), -0.0, 0.0],
                     dtype=torch.float32)
    actual = _custom([x])
    expected = _reference([x])
    _assert_close(actual, expected, torch.float32, "special-fp32", check_signed_zero=True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_trunc_special_values_fp16():
    x = torch.tensor([float("nan"), float("inf"), float("-inf"), -0.0, 0.0],
                     dtype=torch.float16)
    actual = _custom([x])
    expected = _reference([x])
    _assert_close(actual, expected, torch.float16, "special-fp16", check_signed_zero=True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_trunc_negative_fraction_to_negative_zero(dtype):
    x = torch.tensor([-0.999, -0.5, -0.1, -1.0e-6, -0.0, 0.0, 0.1],
                     dtype=torch.float32).to(dtype)
    actual = _custom([x])[0]
    expected = torch.trunc(x)
    _assert_close([actual], [expected], dtype, "negative-fraction-zero",
                  check_signed_zero=True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_trunc_large_tensor(dtype):
    x = torch.linspace(-100.0, 100.0, steps=65536, dtype=torch.float32)
    _run_case([x], dtype, "large-tensor", check_signed_zero=True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_trunc_non_contiguous(dtype):
    base = torch.randn(10, 20, dtype=torch.float32)
    x = base[:, ::2]
    assert not x.is_contiguous()
    _run_case([x], dtype, "non-contiguous", check_signed_zero=True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_trunc_empty_tensor(dtype):
    x = torch.empty(0, dtype=torch.float32).to(dtype)
    actual = _custom([x])
    expected = _reference([x])
    assert len(actual) == 1
    assert actual[0].numel() == 0
    assert actual[0].shape == expected[0].shape
    assert actual[0].dtype == expected[0].dtype


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_trunc_mixed_dtypes():
    tensors = [
        torch.tensor([-2.9, -0.5, 0.5, 2.9], dtype=torch.float32),
        torch.tensor([-3.7, -0.1, 0.1, 3.7], dtype=torch.float16),
    ]
    actual = _custom(tensors)
    expected = _reference(tensors)
    _assert_close(actual, expected, None, "mixed-dtypes", check_signed_zero=True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_trunc_large_magnitude_fp32():
    x = torch.tensor([
        2.0**23, 2.0**24, 2.0**31, 1.0e10, 1.0e20, 3.4e38,
        -(2.0**23), -(2.0**24), -(2.0**31), -1.0e10, -1.0e20, -3.4e38,
    ], dtype=torch.float32)
    actual = torch.ops.ops_multimodal_fusion.foreach_trunc([x.npu()])[0].cpu()
    expected = torch.trunc(x)
    assert torch.equal(actual, expected), f"large-magnitude mismatch: {actual} vs {expected}"
    assert torch.equal(actual, x), f"large-magnitude should be identity: {actual} vs {x}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("numel", [32767, 32768, 32769, 65535])
def test_foreach_trunc_tile_boundaries(numel):
    gen = torch.Generator().manual_seed(4000 + numel)
    x = torch.randn(numel, generator=gen, dtype=torch.float32) * 20.0
    _run_case([x], torch.float32, f"tile-boundary-{numel}", check_signed_zero=True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_trunc_output_not_alias():
    x = torch.randn(4, dtype=torch.float32).npu()
    out = torch.ops.ops_multimodal_fusion.foreach_trunc([x])
    assert out[0].data_ptr() != x.data_ptr(), "output aliases input storage"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_trunc_invalid_args():
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_trunc([])

    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_trunc([torch.randn(3, 4)])

    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_trunc([
            torch.randn(3, 4).npu(),
            torch.randn(5,),
        ])

    for dtype in [torch.bfloat16, torch.float64, torch.int32, torch.bool]:
        with pytest.raises(RuntimeError):
            if dtype is torch.bool:
                tensor = torch.tensor([True, False]).npu()
            elif dtype is torch.int32:
                tensor = torch.tensor([1, 2, 3], dtype=dtype).npu()
            else:
                tensor = torch.randn(3, 4, dtype=dtype).npu()
            torch.ops.ops_multimodal_fusion.foreach_trunc([tensor])


def test_foreach_trunc_empty_list_rejected_cpu_meta():
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_trunc([])
