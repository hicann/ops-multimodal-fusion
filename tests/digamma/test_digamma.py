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
import math

import pytest
import torch
import torch_npu  # noqa: F401

import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "digamma"):
    pytest.skip(
        "ops_multimodal_fusion.digamma not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_digamma_interface_exist():
    """Test that the 'ops_multimodal_fusion.digamma' operator is registered in torch.ops."""
    logging.info(torch.ops.ops_multimodal_fusion.digamma)
    assert hasattr(torch.ops.ops_multimodal_fusion, "digamma"), \
        "The 'digamma' operator is not registered in the 'torch.ops.ops_multimodal_fusion' namespace."


SHAPES = [
    (1,),
    (7,),
    (1024,),
    (10000,),
    (10, 10),
    (32, 32),
    (100, 100),
    (10, 100),
    (256, 512),
    (16, 32, 64),
    (1, 3, 32, 32),
    (4, 3, 64, 64),
    # Large shapes that exceed UB capacity, forcing multi-tile processing
    (100000,),
    (1000000,),
    (2048, 2048),
    (64, 128, 256),
]

DTYPES = [torch.float32]


def _sample_positive(shape, dtype, low=0.1, high=20.0, seed=0):
    """Uniform positive inputs in [low, high]. Avoids x <= 0 where dav-3510 kernel
    (no reflection) doesn't reproduce the analytic continuation.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.empty(*shape, dtype=dtype).uniform_(low, high, generator=g)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_digamma_operator(shape, dtype):
    """Compare NPU digamma against torch.digamma on positive inputs."""
    x = _sample_positive(shape, dtype)

    x_npu = x.npu()
    y_npu = torch.ops.ops_multimodal_fusion.digamma(x_npu)
    y = y_npu.cpu()

    assert y.dtype == dtype, f"dtype mismatch: {y.dtype} vs {dtype}"
    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"

    expected = torch.digamma(x.to(torch.float32)).to(dtype)
    # Shifted asymptotic with N=8 + 8 reciprocals accumulates a few float32 ULPs
    # of error; 1e-4 rel / 1e-5 abs matches the tolerance used by other
    # transcendental ops in this repo.
    assert torch.allclose(y, expected, rtol=1e-4, atol=1e-5), (
        f"digamma mismatch: max abs diff = {(y - expected).abs().max().item()}, "
        f"max rel diff = {((y - expected) / expected.abs().clamp_min(1e-30)).abs().max().item()}"
    )
    logging.info(f"Test passed: shape={shape}, dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_digamma_named_values():
    """Closed-form digamma values at integer and half-integer reference points."""
    gamma = 0.5772156649015329
    values_and_expected = [
        (1.0, -gamma),
        (2.0, 1.0 - gamma),
        (3.0, 1.0 + 0.5 - gamma),
        (4.0, 1.0 + 0.5 + 1.0 / 3.0 - gamma),
        (0.5, -gamma - 2.0 * math.log(2.0)),
        (1.5, -gamma - 2.0 * math.log(2.0) + 2.0),
        (0.25, -gamma - 3.0 * math.log(2.0) - math.pi / 2.0),
        (10.0, torch.digamma(torch.tensor(10.0, dtype=torch.float64)).item()),
        (100.0, torch.digamma(torch.tensor(100.0, dtype=torch.float64)).item()),
    ]
    xs = torch.tensor([v for v, _ in values_and_expected], dtype=torch.float32)
    expected = torch.tensor([e for _, e in values_and_expected], dtype=torch.float32)

    y = torch.ops.ops_multimodal_fusion.digamma(xs.npu()).cpu()

    assert torch.allclose(y, expected, rtol=1e-4, atol=1e-5), (
        f"digamma named-value mismatch; got {y.tolist()} expected {expected.tolist()}"
    )
    logging.info("Named-values test passed.")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_digamma_wide_range():
    """Covers small (x ~ 1e-2), mid, and large (x ~ 1e4) positive inputs."""
    xs = torch.tensor([
        0.01, 0.05, 0.1, 0.25, 0.5,
        1.0, 1.5, 2.718281828, 3.14159265, 5.0,
        10.0, 50.0, 100.0, 500.0, 1000.0, 10000.0,
    ], dtype=torch.float32)

    y = torch.ops.ops_multimodal_fusion.digamma(xs.npu()).cpu()
    expected = torch.digamma(xs.to(torch.float64)).to(torch.float32)

    # Tiny x values (0.01, 0.05) have very large |psi(x)| (the 1/x term dominates),
    # so the absolute tolerance on the reconstruction needs to scale.
    max_abs = (y - expected).abs().max().item()
    max_rel = ((y - expected) / expected.abs().clamp_min(1e-6)).abs().max().item()
    assert max_rel < 1e-4 and max_abs < 1e-2, (
        f"digamma wide-range mismatch: max abs={max_abs}, max rel={max_rel}\n"
        f"got     = {y.tolist()}\n"
        f"expected= {expected.tolist()}"
    )
    logging.info(f"Wide-range test passed (max abs={max_abs:.3e}, max rel={max_rel:.3e}).")
