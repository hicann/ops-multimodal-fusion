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


import torch
import torch_npu  # noqa: F401
import pytest
import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "logit"):
    pytest.skip(
        "ops_multimodal_fusion.logit not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_logit_interface_exist():
    """Test that the 'ops_multimodal_fusion.logit' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "logit"),\
        "The 'logit' operator is not registered in the 'torch.ops.ops_multimodal_fusion' namespace."


SHAPES = [
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
    # Large shapes that exceed UB capacity, forcing multi-tile processing
    (100000,),
    (1000000,),
    (2048, 2048),
    (4096, 1024),
    (64, 128, 256),
    (16, 16, 128, 128),
]

DTYPES = [torch.float32, torch.float16]

# eps attribute coverage:
# None to useEps_ false no clamping branch
# 1e-6 to useEps_ true narrow clamping 1e-6 1-1e-6
# 1e-1 to useEps_ true wide clamping 0.1 0.9
EPS_VALUES = [None, 1e-6, 1e-1]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eps", EPS_VALUES)
def test_logit_operator(shape, dtype, eps):
    """Test the logit operator with and without eps clamping, across shapes/dtypes.

    Uses values in 0 1 clamped to a safe range; the eps 1e-6 configuration
    combined with small entries (e.g. 0.05) exercises the boundary-clamping path.
    """
    a = torch.rand(*shape, dtype=dtype).clamp(min=0.05, max=0.95)

    if eps is not None:
        expected = torch.logit(a, eps=eps)
        result = torch.ops.ops_multimodal_fusion.logit(a.npu(), eps=eps).cpu()
    else:
        expected = torch.logit(a)
        result = torch.ops.ops_multimodal_fusion.logit(a.npu()).cpu()

    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2),\
        f"Logit failed for shape {shape}, dtype {dtype}, eps {eps}. " \
        f"Max diff: {torch.max(torch.abs(result - expected)):.6f}"
