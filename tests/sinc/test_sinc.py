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

import logging

import pytest
import torch
import torch_npu  # noqa: F401

import ops_multimodal_fusion  # noqa: F401

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

if not hasattr(torch.ops.ops_multimodal_fusion, "sinc"):
    pytest.skip(
        "ops_multimodal_fusion.sinc not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_sinc_interface_exist():
    """Test that the 'ops_multimodal_fusion.sinc' operator is registered in torch.ops."""
    logging.info(torch.ops.ops_multimodal_fusion.sinc)
    assert hasattr(torch.ops.ops_multimodal_fusion, "sinc"), (
        "The 'sinc' operator is not registered in the "
        "'torch.ops.ops_multimodal_fusion' namespace."
    )


RANDN_SHAPES = [
    (1,),
    (3,),
    (10,),
    (100,),
    (1024,),
    (10000,),
    (10, 10),
    (32, 32),
    (100, 100),
    (10, 100),
    (100, 10),
    (256, 512),
    (5, 10, 15),
    (16, 32, 64),
    (32, 64, 128),
    (1, 3, 32, 32),
    (4, 3, 64, 64),
    (8, 3, 128, 128),
    (1000, 1000),
    (100000,),
    (1000000,),
    (2048, 2048),
    (4096, 1024),
    (64, 128, 256),
    (16, 16, 128, 128),
]
RANDN_DTYPES = [torch.float32, torch.float16]

SPECIAL_CASES = [
    ("zeros", torch.zeros(10, dtype=torch.float32)),
    (
        "integers",
        torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], dtype=torch.float32),
    ),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", RANDN_SHAPES)
@pytest.mark.parametrize("dtype", RANDN_DTYPES)
def test_sinc_operator(shape, dtype):
    """Compare NPU sinc against torch.sinc on random inputs."""
    x = torch.randn(*shape, dtype=dtype)
    expected = torch.sinc(x)
    result = torch.ops.ops_multimodal_fusion.sinc(x.npu()).cpu()
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2), (
        f"sinc failed for shape={shape}, dtype={dtype}. "
        f"Max diff: {torch.max(torch.abs(result - expected)):.6f}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("tag,expected_input", SPECIAL_CASES, ids=[c[0] for c in SPECIAL_CASES])
def test_sinc_special_values(tag, expected_input):
    expected = torch.sinc(expected_input)
    result = torch.ops.ops_multimodal_fusion.sinc(expected_input.npu()).cpu()
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2), (
        f"sinc failed for special={tag}. "
        f"Max diff: {torch.max(torch.abs(result - expected)):.6f}"
    )
