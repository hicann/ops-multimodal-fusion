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


if not hasattr(torch.ops.ops_multimodal_fusion, "cumprod"):
    pytest.skip(
        "ops_multimodal_fusion.cumprod not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float16, torch.float32]
CASES = [
    ((1,), 0),
    ((7,), 0),
    ((8,), -1),
    ((9,), 0),
    ((3, 5), 0),
    ((3, 5), 1),
    ((2, 3, 7), 0),
    ((2, 3, 7), 1),
    ((2, 3, 7), -1),
    ((2, 3, 7), -2),
    ((2, 257, 3), 1),
    ((4, 1025), -1),
    ((1025, 3), 0),
]


def _make_input(shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(*shape, dtype=torch.float32).uniform_(0.75, 1.25, generator=g)
    if len(shape) and shape[-1] >= 4:
        flat = x.flatten()
        flat[1::17] = 0.0
        flat[3::29] = -0.5
    return x.to(dtype)


def _expected(x, dim, dtype):
    if dtype == torch.float16:
        return torch.cumprod(x.float(), dim=dim).to(dtype)
    return torch.cumprod(x, dim=dim)


def _assert_close(actual, expected, msg):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    if expected.dtype == torch.float16:
        rtol, atol = 2e-2, 2e-2
    else:
        rtol, atol = 1e-5, 1e-5
    assert torch.allclose(actual.cpu(), expected.cpu(), rtol=rtol, atol=atol), (
        f"{msg}: max abs diff = {(actual.cpu() - expected.cpu()).abs().max().item()}"
    )


def test_cumprod_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "cumprod")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape,dim", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_cumprod_operator(shape, dim, dtype):
    seed = abs(hash((shape, dim, dtype))) % 10_000_000
    x = _make_input(shape, dtype, seed)

    y = torch.ops.ops_multimodal_fusion.cumprod(x.npu(), int(dim)).cpu()
    expected = _expected(x, dim, dtype)

    _assert_close(y, expected, f"cumprod mismatch shape={shape}, dim={dim}, dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_cumprod_non_contiguous_input(dtype):
    x = _make_input((5, 7), dtype, 123).t()
    assert not x.is_contiguous()

    y = torch.ops.ops_multimodal_fusion.cumprod(x.npu(), 1).cpu()
    expected = _expected(x, 1, dtype)

    _assert_close(y, expected, f"cumprod non-contiguous mismatch dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_cumprod_empty_tensor():
    x = torch.empty((0, 3), dtype=torch.float32).npu()
    y = torch.ops.ops_multimodal_fusion.cumprod(x, 0).cpu()
    assert y.shape == (0, 3)
    assert y.dtype == torch.float32


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_cumprod_scalar_tensor():
    x = torch.tensor(2.5, dtype=torch.float32).npu()
    y = torch.ops.ops_multimodal_fusion.cumprod(x, 0).cpu()
    assert y.shape == torch.Size([])
    assert torch.equal(y, torch.tensor(2.5, dtype=torch.float32))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_cumprod_special_values(dtype):
    x = torch.tensor(
        [
            [1.0, 0.0, 3.0, 4.0],
            [1.0, float("inf"), 2.0, -1.0],
            [1.0, float("nan"), 2.0, 3.0],
        ],
        dtype=dtype,
    )

    y = torch.ops.ops_multimodal_fusion.cumprod(x.npu(), 1).cpu()
    expected = _expected(x, 1, dtype)

    assert torch.equal(torch.isnan(y), torch.isnan(expected))
    _assert_close(torch.nan_to_num(y), torch.nan_to_num(expected),
                  f"cumprod special-values mismatch dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_cumprod_float16_overflow():
    x = torch.full((32,), 2.0, dtype=torch.float16)

    y = torch.ops.ops_multimodal_fusion.cumprod(x.npu(), 0).cpu()
    expected = _expected(x, 0, torch.float16)

    assert torch.equal(torch.isinf(y), torch.isinf(expected))
    _assert_close(torch.nan_to_num(y), torch.nan_to_num(expected),
                  "cumprod float16 overflow mismatch")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_cumprod_rejects_unsupported_dtype():
    for unsupported in [torch.bfloat16, torch.float64, torch.int32]:
        x = torch.ones(4, dtype=unsupported).npu()
        with pytest.raises(RuntimeError, match="cumprod"):
            torch.ops.ops_multimodal_fusion.cumprod(x, 0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_cumprod_rejects_invalid_dim():
    x = torch.ones((2, 3), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError, match="cumprod"):
        torch.ops.ops_multimodal_fusion.cumprod(x, 2)
