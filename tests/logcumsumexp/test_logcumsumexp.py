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

from collections import namedtuple

import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "logcumsumexp"):
    pytest.skip(
        "ops_multimodal_fusion.logcumsumexp not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


# Relative and absolute tolerance pair carried together so assertion helpers
# stay within the five-parameter codecheck limit.
Tol = namedtuple("Tol", ["rtol", "atol"])

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
    ((2, 128, 3), 1),
    ((1, 513), -1),
    ((2, 3, 1031), 1),
    ((1031, 3), 1),
    ((2, 1, 3, 5), 1),
    ((1, 2, 257, 3), 2),
]


def _make_input(shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(*shape, dtype=torch.float32).uniform_(-6.0, 6.0, generator=g)
    if x.numel() > 8:
        flat = x.flatten()
        flat[1::13] = 0.0
        flat[3::17] = -4.0
        flat[5::19] = 5.0
    return x.to(dtype)


def _reference(x, dim):
    return torch.logcumsumexp(x.to(torch.float64), dim=dim).to(x.dtype)


def _assert_result(actual, expected, msg, equal_nan=False):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    rtol, atol = (5e-3, 5e-3) if expected.dtype == torch.float16 else (1e-5, 1e-5)
    assert torch.allclose(
        actual.cpu(), expected.cpu(), rtol=rtol, atol=atol, equal_nan=equal_nan
    ), msg


def _assert_result_with_tol(actual, expected, msg, tol, equal_nan=False):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.allclose(
        actual.cpu(), expected.cpu(), rtol=tol.rtol, atol=tol.atol, equal_nan=equal_nan
    ), msg


def test_logcumsumexp_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "logcumsumexp")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape,dim", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_logcumsumexp_operator(shape, dim, dtype):
    seed = abs(hash((shape, dim, dtype))) % 10_000_000
    x = _make_input(shape, dtype, seed)

    actual = torch.ops.ops_multimodal_fusion.logcumsumexp(x.npu(), int(dim))
    expected = _reference(x, dim)

    _assert_result(
        actual, expected,
        f"logcumsumexp mismatch shape={shape}, dim={dim}, dtype={dtype}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_logcumsumexp_non_contiguous_input(dtype):
    x = _make_input((5, 7), dtype, 123).t()
    assert not x.is_contiguous()

    actual = torch.ops.ops_multimodal_fusion.logcumsumexp(x.npu(), 1)
    expected = _reference(x, 1)

    _assert_result(actual, expected,
                   f"logcumsumexp non-contiguous mismatch dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_logcumsumexp_numeric_stability(dtype):
    x = torch.tensor(
        [[80.0, 80.0, 79.0, -80.0], [-80.0, -79.0, -78.0, -77.0]],
        dtype=dtype,
    )

    actual = torch.ops.ops_multimodal_fusion.logcumsumexp(x.npu(), 1)
    expected = _reference(x, 1)

    _assert_result(actual, expected,
                   f"logcumsumexp stability mismatch dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape,dim,pattern",
    [
        ((2, 128, 3), 1, "small"),
        ((1, 513), 1, "small"),
        ((2, 128, 3), 1, "decreasing"),
        ((1, 513), 1, "decreasing"),
    ],
)


def test_logcumsumexp_long_scan_precision(dtype, shape, dim, pattern):
    if pattern == "small":
        base = torch.linspace(-1.0e-3, 1.0e-3, steps=shape[dim], dtype=torch.float32)
    else:
        base = torch.linspace(20.0, -80.0, steps=shape[dim], dtype=torch.float32)
    view_shape = [1] * len(shape)
    view_shape[dim] = shape[dim]
    x = base.reshape(view_shape).expand(shape).clone().to(dtype)

    actual = torch.ops.ops_multimodal_fusion.logcumsumexp(x.npu(), dim)
    expected = _reference(x, dim)

    tol = Tol(1e-2, 1e-2) if dtype == torch.float16 else Tol(1e-5, 1e-5)
    _assert_result_with_tol(
        actual, expected,
        f"logcumsumexp long-scan precision mismatch dtype={dtype}, "
        f"shape={shape}, dim={dim}, pattern={pattern}",
        tol,
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_logcumsumexp_half_matches_float32_accumulation_reference():
    x = torch.linspace(-7.0, 7.0, steps=257, dtype=torch.float16).reshape(1, 257)

    actual = torch.ops.ops_multimodal_fusion.logcumsumexp(x.npu(), 1)
    expected = torch.logcumsumexp(x.float(), dim=1).half()

    _assert_result_with_tol(
        actual, expected,
        "logcumsumexp half should match fp32 accumulation rounded to half",
        Tol(5e-3, 5e-3),
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_logcumsumexp_nan_propagates(dtype):
    x = torch.tensor([[1.0, 2.0, float("nan"), 3.0]], dtype=dtype)

    actual = torch.ops.ops_multimodal_fusion.logcumsumexp(x.npu(), 1)
    expected = torch.logcumsumexp(x, dim=1)

    _assert_result(actual, expected,
                   f"logcumsumexp nan propagation mismatch dtype={dtype}",
                   equal_nan=True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "values",
    [
        [float("nan"), 1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0, float("nan")],
        [1.0, float("inf"), 2.0, float("nan"), 3.0],
        [float("-inf"), float("-inf"), float("-inf"), -4.0, -3.0, -2.0],
    ],
)


def test_logcumsumexp_special_value_ordering(dtype, values):
    x = torch.tensor([values], dtype=dtype)

    actual = torch.ops.ops_multimodal_fusion.logcumsumexp(x.npu(), 1)
    expected = torch.logcumsumexp(x, dim=1)

    _assert_result(actual, expected,
                   f"logcumsumexp special ordering mismatch dtype={dtype}, values={values}",
                   equal_nan=True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "values",
    [
        [float("-inf"), float("-inf"), -2.0, -1.0],
        [1.0, float("inf"), 2.0, float("-inf"), 3.0],
        [float("-inf"), 0.0, float("inf"), 1.0],
    ],
)


def test_logcumsumexp_inf_values(dtype, values):
    x = torch.tensor([values], dtype=dtype)

    actual = torch.ops.ops_multimodal_fusion.logcumsumexp(x.npu(), 1)
    expected = torch.logcumsumexp(x, dim=1)

    _assert_result(actual, expected,
                   f"logcumsumexp inf mismatch dtype={dtype}, values={values}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_logcumsumexp_empty_tensor():
    x = torch.empty((0, 3), dtype=torch.float32).npu()
    y = torch.ops.ops_multimodal_fusion.logcumsumexp(x, 0)
    assert y.cpu().shape == (0, 3)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_logcumsumexp_empty_scan_dim():
    x = torch.empty((2, 0, 3), dtype=torch.float32).npu()
    y = torch.ops.ops_multimodal_fusion.logcumsumexp(x, 1)
    assert y.cpu().shape == (2, 0, 3)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_logcumsumexp_empty_non_scan_dim():
    x = torch.empty((0, 3, 4), dtype=torch.float32).npu()
    y = torch.ops.ops_multimodal_fusion.logcumsumexp(x, 1)
    assert y.cpu().shape == (0, 3, 4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_logcumsumexp_scalar_tensor():
    x = torch.tensor(2.5, dtype=torch.float32).npu()
    y = torch.ops.ops_multimodal_fusion.logcumsumexp(x, 0)
    assert y.cpu().shape == torch.Size([])
    assert torch.equal(y.cpu(), torch.tensor(2.5, dtype=torch.float32))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_logcumsumexp_rejects_unsupported_dtype():
    for unsupported in [torch.bfloat16, torch.float64, torch.int32]:
        x = torch.ones(4, dtype=unsupported).npu()
        with pytest.raises(RuntimeError, match="logcumsumexp"):
            torch.ops.ops_multimodal_fusion.logcumsumexp(x, 0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_logcumsumexp_rejects_invalid_dim():
    x = torch.ones((2, 3), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError, match="logcumsumexp"):
        torch.ops.ops_multimodal_fusion.logcumsumexp(x, 2)
