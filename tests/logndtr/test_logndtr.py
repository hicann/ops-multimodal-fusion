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
import torch_npu

import ops_multimodal_fusion

if not hasattr(torch.ops.ops_multimodal_fusion, "logndtr"):
    pytest.skip(
        "ops_multimodal_fusion.logndtr not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_logndtr_interface_exist():
    """Test that the 'ops_multimodal_fusion.logndtr' operator is registered in torch.ops."""
    logging.info(torch.ops.ops_multimodal_fusion.logndtr)
    assert hasattr(torch.ops.ops_multimodal_fusion, "logndtr"), \
        "The 'logndtr' operator is not registered in the 'torch.ops.ops_multimodal_fusion' namespace."


RANDN_SHAPES = [
    (1,), (3,), (10,), (100,), (1024,), (10000,),
    (10, 10), (32, 32), (100, 100), (10, 100), (100, 10), (256, 512),
    (5, 10, 15), (16, 32, 64), (32, 64, 128),
    (1, 3, 32, 32), (4, 3, 64, 64), (8, 3, 128, 128), (1000, 1000),
    (100000,), (1000000,), (2048, 2048), (4096, 1024),
]

# Special-value cases cover the boundary at x = -5 (asymptotic branch),
# the zero point, and the positive tail where log Φ(x) → 0.
SPECIAL_CASES = [
    ("zeros", lambda: torch.zeros(16, dtype=torch.float32)),
    ("near_boundary", lambda: torch.tensor(
        [-5.001, -5.0, -4.999, -5.5, -4.5, -6.0, -8.0, -10.0],
        dtype=torch.float32)),
    ("small_range", lambda: torch.linspace(-3.0, 3.0, 17, dtype=torch.float32)),
    ("large_negative", lambda: torch.tensor(
        [-5.0, -6.0, -8.0, -10.0, -12.0, -15.0, -20.0], dtype=torch.float32)),
    ("large_positive", lambda: torch.tensor(
        [1.0, 2.0, 3.0, 5.0, 7.0, 10.0], dtype=torch.float32)),
    ("mixed_sign", lambda: torch.tensor(
        [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0], dtype=torch.float32)),
]

CASES = (
    [("randn", shape, torch.float32) for shape in RANDN_SHAPES]
    + [("special", tag, builder) for tag, builder in SPECIAL_CASES]
)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("kind,arg1,arg2", CASES)
def test_logndtr_operator(kind, arg1, arg2):
    """Test logndtr against torch.special.log_ndtr across shapes and special values."""
    if kind == "randn":
        shape, dtype = arg1, arg2
        # Shift slightly toward negative so the asymptotic branch sees traffic too.
        a = torch.randn(*shape, dtype=dtype) * 3.0 - 1.0
        label = f"shape={shape}, dtype={dtype}"
    else:
        a = arg2()
        label = f"special={arg1}"

    expected = torch.special.log_ndtr(a.double()).to(torch.float32)
    result = torch.ops.ops_multimodal_fusion.logndtr(a.npu()).cpu()

    # Tolerance is generous because the kernel uses a float32 polynomial
    # approximation (A&S 7.1.26, max error ~1.5e-7 in erfc) and a truncated
    # asymptotic series. Log magnifies relative errors at extreme negatives,
    # so use absolute + relative tolerance.
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3), \
        f"LogNdtr failed for {label}. " \
        f"Max abs diff: {torch.max(torch.abs(result - expected)):.6f}, " \
        f"Max rel diff: {torch.max(torch.abs((result - expected) / (expected.abs() + 1e-10))):.6f}"
