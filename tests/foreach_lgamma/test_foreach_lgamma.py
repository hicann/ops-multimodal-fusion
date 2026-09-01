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

"""Test suite for foreach_lgamma; golden reference: torch.lgamma per tensor."""

import math

import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "foreach_lgamma"):
    pytest.skip(
        "ops_multimodal_fusion.foreach_lgamma not registered for current NPU_ARCH; "
        "skipping module",
        allow_module_level=True,
    )


DTYPE = torch.float32


def _reference(tensors):
    """Golden: torch.lgamma on each tensor on CPU."""
    return [torch.lgamma(t.cpu()) for t in tensors]


def _custom(tensors):
    npu_tensors = [t.npu() for t in tensors]
    out = torch.ops.ops_multimodal_fusion.foreach_lgamma(npu_tensors)
    return [t.cpu() for t in out]


def _assert_close(actual_list, expected_list, label, rtol=2e-3, atol=2e-3):
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
        assert torch.allclose(
            actual, expected, rtol=rtol, atol=atol, equal_nan=True
        ), f"{label}[{i}]: actual={actual} expected={expected}"


def _run_case(tensors, label, rtol=2e-3, atol=2e-3):
    typed = [t.to(DTYPE) for t in tensors]
    actual = _custom(typed)
    expected = _reference(typed)
    _assert_close(actual, expected, label, rtol=rtol, atol=atol)


def _make_positive_tensors(shapes, seed=2026):
    gen = torch.Generator().manual_seed(seed)
    return [torch.rand(*shape, generator=gen, dtype=DTYPE) * 8.0 + 0.05
            for shape in shapes]


def test_foreach_lgamma_interface_exists():
    assert hasattr(torch.ops.ops_multimodal_fusion, "foreach_lgamma")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_lgamma_single_tensor_positive():
    _run_case([torch.tensor([0.1, 0.5, 1.0, 1.5, 2.0, 3.5, 10.0],
                            dtype=DTYPE)], "single-positive")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_lgamma_two_tensors_positive():
    tensors = [
        torch.tensor([0.25, 0.75, 1.25, 2.25], dtype=DTYPE),
        torch.tensor([[3.0, 4.0, 5.0], [6.5, 8.0, 12.0]], dtype=DTYPE),
    ]
    _run_case(tensors, "two-tensors-positive")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_lgamma_eight_tensors_varied_shapes():
    tensors = _make_positive_tensors(
        [(1,), (3,), (2, 2), (2, 3, 4), (7,), (4, 5), (1, 1, 9), (16,)]
    )
    actual = _custom(tensors)
    expected = _reference(tensors)
    assert len(actual) == 8
    _assert_close(actual, expected, "eight-tensors")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_lgamma_known_values():
    x = torch.tensor([
        0.5, 1.0, 2.0, 3.0, 4.0, 10.0,
        -0.5, -1.5, -2.5, -3.25, -4.75,
    ], dtype=DTYPE)
    _run_case([x], "known-values", rtol=3e-3, atol=3e-3)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_lgamma_negative_non_integer_reflection():
    x = torch.tensor([
        -0.1, -0.25, -0.5, -0.75, -1.25, -1.5,
        -2.25, -2.5, -3.75, -7.5,
    ], dtype=DTYPE)
    _run_case([x], "negative-non-integer", rtol=5e-3, atol=5e-3)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_lgamma_poles_and_infinities():
    x = torch.tensor([
        -4.0, -3.0, -2.0, -1.0, -0.0, 0.0,
        float("inf"), float("-inf"),
    ], dtype=DTYPE)
    actual = _custom([x])[0]
    expected = torch.lgamma(x)
    assert torch.equal(torch.isinf(actual), torch.isinf(expected))
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))
    assert torch.isposinf(actual).all()


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_lgamma_nan_propagates():
    x = torch.tensor([float("nan"), 0.5, -0.5, float("nan")], dtype=DTYPE)
    actual = _custom([x])[0]
    expected = torch.lgamma(x)
    assert torch.isnan(actual[[0, 3]]).all()
    _assert_close([actual[1:3]], [expected[1:3]], "nan-finite-slices")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_lgamma_large_tensor():
    x = torch.linspace(0.05, 50.0, steps=65536, dtype=DTYPE)
    _run_case([x], "large-tensor", rtol=3e-3, atol=3e-3)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_lgamma_non_contiguous_input():
    base = torch.rand(10, 20, dtype=DTYPE) * 5.0 + 0.1
    x = base[:, ::2]
    assert not x.is_contiguous()
    _run_case([x], "non-contiguous")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_lgamma_empty_tensor():
    x = torch.empty(0, dtype=DTYPE)
    actual = _custom([x])
    expected = _reference([x])
    assert len(actual) == 1
    assert actual[0].numel() == 0
    assert actual[0].shape == expected[0].shape
    assert actual[0].dtype == expected[0].dtype


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("numel", [7, 32767, 32768, 32769, 65535])
def test_foreach_lgamma_tile_boundaries(numel):
    x = torch.linspace(0.05, 30.0, steps=numel, dtype=DTYPE)
    _run_case([x], f"tile-boundary-{numel}", rtol=3e-3, atol=3e-3)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_lgamma_output_not_alias():
    x = torch.tensor([0.5, 1.5, 2.5], dtype=DTYPE).npu()
    out = torch.ops.ops_multimodal_fusion.foreach_lgamma([x])
    assert out[0].data_ptr() != x.data_ptr(), "output aliases input storage"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_lgamma_invalid_args():
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_lgamma([])

    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_lgamma([torch.randn(3, dtype=DTYPE)])

    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_lgamma([
            torch.randn(3, dtype=DTYPE).npu(),
            torch.randn(4, dtype=DTYPE),
        ])

    for dtype in [torch.float16, torch.bfloat16, torch.float64, torch.int32, torch.bool]:
        with pytest.raises(RuntimeError):
            if dtype is torch.bool:
                tensor = torch.tensor([True, False]).npu()
            elif dtype is torch.int32:
                tensor = torch.tensor([1, 2, 3], dtype=dtype).npu()
            else:
                tensor = torch.randn(3, dtype=dtype).npu()
            torch.ops.ops_multimodal_fusion.foreach_lgamma([tensor])


def test_foreach_lgamma_empty_list_rejected_cpu_meta():
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_lgamma([])


def test_reference_values_are_well_defined():
    x = torch.tensor([0.5, 1.0, 2.0, -0.5], dtype=DTYPE)
    expected = torch.lgamma(x)
    assert math.isclose(expected[0].item(), math.log(math.sqrt(math.pi)),
                        rel_tol=1e-6, abs_tol=1e-6)
    assert expected[1].item() == 0.0
    assert expected[2].item() == 0.0
    assert torch.isfinite(expected[3])
