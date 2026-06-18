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

import logging

import pytest
import torch
import torch_npu  # noqa: F401

import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "complex"):
    pytest.skip(
        "ops_multimodal_fusion.complex not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float16, torch.float32]
SHAPES = [
    (1,),
    (7,),
    (8,),
    (9,),
    (1024,),
    (3, 5),
    (2, 3, 7),
    (10000,),
]


def test_complex_interface_exist():
    logging.info(torch.ops.ops_multimodal_fusion.complex)
    assert hasattr(torch.ops.ops_multimodal_fusion, "complex")


def _make_inputs(shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    real = torch.randn(*shape, dtype=dtype, generator=g)
    imag = torch.randn(*shape, dtype=dtype, generator=g)
    return real, imag


def _assert_complex_equal(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual.real.cpu(), expected.real.cpu()), (
        f"complex real mismatch: actual={actual}, expected={expected}"
    )
    assert torch.equal(actual.imag.cpu(), expected.imag.cpu()), (
        f"complex mismatch: actual={actual}, expected={expected}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_complex_operator(shape, dtype):
    seed = abs(hash((shape, dtype))) % 10_000_000
    real, imag = _make_inputs(shape, dtype, seed)

    y = torch.ops.ops_multimodal_fusion.complex(real.npu(), imag.npu()).cpu()
    expected = torch.complex(real, imag)

    _assert_complex_equal(y, expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "real_shape,imag_shape",
    [
        ((17, 1), (1, 9)),
        ((1, 17), (3, 17)),
        ((2, 1, 17), (1, 3, 1)),
        ((2, 3, 17), (17,)),
        ((2, 1, 3, 17), (4, 1, 1)),
    ],
)
def test_complex_broadcast_special_shapes(real_shape, imag_shape, dtype):
    real, _ = _make_inputs(real_shape, dtype, 101)
    _, imag = _make_inputs(imag_shape, dtype, 202)

    y = torch.ops.ops_multimodal_fusion.complex(real.npu(), imag.npu()).cpu()
    expected = torch.complex(real, imag)

    _assert_complex_equal(y, expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_complex_empty_tensor():
    real = torch.empty((0, 3), dtype=torch.float32).npu()
    imag = torch.empty((1, 3), dtype=torch.float32).npu()
    y = torch.ops.ops_multimodal_fusion.complex(real, imag).cpu()
    assert y.shape == (0, 3)
    assert y.dtype == torch.complex64


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_complex_rejects_unsupported_dtype():
    for unsupported in [torch.bfloat16, torch.float64, torch.int32]:
        real = torch.ones(4, dtype=unsupported).npu()
        imag = torch.ones(4, dtype=unsupported).npu()
        with pytest.raises(RuntimeError, match="complex"):
            torch.ops.ops_multimodal_fusion.complex(real, imag)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_complex_rejects_dtype_mismatch():
    real = torch.ones(4, dtype=torch.float32).npu()
    imag = torch.ones(4, dtype=torch.float16).npu()
    with pytest.raises(RuntimeError, match="complex"):
        torch.ops.ops_multimodal_fusion.complex(real, imag)
