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
import torch_npu

import ops_multimodal_fusion

if not hasattr(torch.ops.ops_multimodal_fusion, "erfcx"):
    pytest.skip(
        "ops_multimodal_fusion.erfcx not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_erfcx_interface_exist():
    """The 'ops_multimodal_fusion.erfcx' operator should be registered in torch.ops."""
    logging.info(torch.ops.ops_multimodal_fusion.erfcx)
    assert hasattr(torch.ops.ops_multimodal_fusion, "erfcx"),\
        "The 'erfcx' operator is not registered in the 'torch.ops.ops_multimodal_fusion' namespace."


RANDN_SHAPES = [
    (1,), (3,), (10,), (100,), (1024,), (10000,),
    (10, 10), (32, 32), (100, 100), (10, 100), (100, 10), (256, 512),
    (5, 10, 15), (16, 32, 64), (32, 64, 128),
    (1, 3, 32, 32), (4, 3, 64, 64), (8, 3, 128, 128), (1000, 1000),
    (100000,), (1000000,), (2048, 2048), (4096, 1024),
]

# Targeted ranges covering all three regions of the piecewise approximation
# plus the transition-to-saturation window for negative x.
SPECIAL_CASES = [
    # Region 1/2 with sign flip: exercises the 2·exp(x²) − erfcx(|x|) lift.
    ("neg_range", lambda: torch.linspace(-5.0, 0.0, 21, dtype=torch.float32)),
    # Region 1 (z ≤ 0.46875) and Region 2 (0.46875 < z ≤ 4).
    ("mid_range", lambda: torch.linspace(0.0, 4.0, 21, dtype=torch.float32)),
    # Region 3 (z > 4) — asymptotic branch.
    ("tail_range", lambda: torch.linspace(4.0, 15.0, 23, dtype=torch.float32)),
    # Region boundaries at 0.46875 and 4.0 (blend must not introduce a step).
    ("boundaries", lambda: torch.tensor(
        [0.0, 0.46874, 0.46875, 0.46876, 3.999, 4.0, 4.001],
        dtype=torch.float32)),
    # Approach to saturation: values with x² ≲ 88.7 must still be finite.
    ("near_saturation", lambda: torch.tensor(
        [-7.0, -8.0, -9.0, -9.3, -9.4],
        dtype=torch.float32)),
    ("zeros", lambda: torch.zeros(32, dtype=torch.float32)),
]

# x-values where fp32 erfcx overflows to +inf (|x| ≳ 9.4).
SATURATION_CASES = [
    ("saturation", lambda: torch.tensor(
        [-9.5, -10.0, -12.0, -15.0],
        dtype=torch.float32)),
]

CASES = (
    [("randn", shape, torch.float32) for shape in RANDN_SHAPES]
    + [("special", tag, builder) for tag, builder in SPECIAL_CASES]
)


def _golden(x_cpu: torch.Tensor) -> torch.Tensor:
    # Upcast to fp64 for the reference to avoid contaminating the comparison
    # with CPU-side fp32 approximation error in torch.special.erfcx.
    return torch.special.erfcx(x_cpu.double()).to(torch.float32)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("kind,arg1,arg2", CASES)
def test_erfcx_operator(kind, arg1, arg2):
    """Compare kernel output against a double-precision torch.special.erfcx."""
    if kind == "randn":
        shape, dtype = arg1, arg2
        # Samples use std three around zero, so roughly 99 percent land within
        # plus or minus nine and all three regions get exercised without saturation flooding.
        a = torch.randn(*shape, dtype=dtype) * 3.0
        label = f"shape={shape}, dtype={dtype}"
    else:
        a = arg2()
        label = f"special={arg1}"

    expected = _golden(a)
    result = torch.ops.ops_multimodal_fusion.erfcx(a.npu()).cpu()

    # Tolerance matches the repo convention for transcendental-family
    # unary ops in fp32. Cody 1969 in fp32 easily beats this; the headroom
    # absorbs rounding at region boundaries and near saturation.
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2), (
        f"Erfcx failed for {label}. "
        f"Max abs diff: {torch.max(torch.abs(result - expected)):.3e}, "
        f"Max rel diff: {torch.max(torch.abs((result - expected) / (expected.abs() + 1e-10))):.3e}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("tag,builder", SATURATION_CASES)
def test_erfcx_saturation(tag, builder):
    """For |x| ≳ 9.4 the result overflows fp32 → +inf. Verify saturation
    happens at exactly the same positions as the reference.
    """
    a = builder()
    expected = _golden(a)
    result = torch.ops.ops_multimodal_fusion.erfcx(a.npu()).cpu()

    exp_inf = torch.isposinf(expected)
    res_inf = torch.isposinf(result)
    assert torch.equal(exp_inf, res_inf), (
        f"Erfcx saturation mismatch for {tag}: "
        f"expected_inf={exp_inf.tolist()}, result_inf={res_inf.tolist()}"
    )
    finite = ~exp_inf & ~res_inf
    if finite.any():
        assert torch.allclose(result[finite], expected[finite], rtol=1e-2, atol=1e-2), (
            f"Erfcx finite-part mismatch for {tag}. "
            f"Max abs diff: {torch.max(torch.abs(result[finite] - expected[finite])):.3e}"
        )
