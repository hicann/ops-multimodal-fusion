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

if not hasattr(torch.ops.ops_multimodal_fusion, "polygamma"):
    pytest.skip(
        "ops_multimodal_fusion.polygamma not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_polygamma_interface_exist():
    """Test that the 'ops_multimodal_fusion.polygamma' operator is registered in torch.ops."""
    logging.info(torch.ops.ops_multimodal_fusion.polygamma)
    assert hasattr(torch.ops.ops_multimodal_fusion, "polygamma"), \
        "The 'polygamma' operator is not registered in the 'torch.ops.ops_multimodal_fusion' namespace."


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
ORDERS = [1, 2, 3, 4, 5, 6]


def _sample_positive(shape, dtype, low, high=20.0, seed=0):
    """Uniform positive inputs in [low, high].  x > 0 is required by the kernel
    (no reflection is applied for x <= 0).
    """
    g = torch.Generator().manual_seed(seed)
    return torch.empty(*shape, dtype=dtype).uniform_(low, high, generator=g)


def _tolerances_for(n):
    """Tolerances mirror digamma for n<=2; relax slightly for higher n because
    |psi^(n)(x)| scales roughly as n!/x^(n+1) and float32 loses resolution in
    large-magnitude regions near x ~ 0.5.
    """
    if n <= 2:
        return dict(rtol=1e-4, atol=1e-5)
    if n <= 4:
        return dict(rtol=3e-4, atol=1e-4)
    return dict(rtol=5e-4, atol=1e-4)


def _low_bound_for(n):
    # Constrain x away from 0 for higher n so the float32-representable
    # magnitude of psi^(n)(x) ~ n!/x^(n+1) stays well within ulp-significant range.
    if n == 1:
        return 0.1
    if n == 2:
        return 0.25
    if n <= 4:
        return 0.5
    return 0.75


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", ORDERS)
def test_polygamma_operator(shape, dtype, n):
    """Compare NPU polygamma(x, n) against torch.special.polygamma on positive inputs."""
    seed = n * 97 + (abs(hash(shape)) % 997)
    x = _sample_positive(shape, dtype, low=_low_bound_for(n), high=20.0, seed=seed)

    x_npu = x.npu()
    y_npu = torch.ops.ops_multimodal_fusion.polygamma(x_npu, n)
    y = y_npu.cpu()

    assert y.dtype == dtype, f"dtype mismatch: {y.dtype} vs {dtype}"
    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"

    # Reference in float64 → float32 to pin down the best achievable float32 answer.
    expected = torch.special.polygamma(n, x.to(torch.float64)).to(dtype)

    tol = _tolerances_for(n)
    assert torch.allclose(y, expected, **tol), (
        f"polygamma(n={n}) mismatch: "
        f"max abs diff = {(y - expected).abs().max().item()}, "
        f"max rel diff = "
        f"{((y - expected) / expected.abs().clamp_min(1e-30)).abs().max().item()}"
    )
    logging.info(f"Test passed: n={n}, shape={shape}, dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("n", ORDERS)
def test_polygamma_named_values(n):
    """Reference from torch.special.polygamma in float64 at several named points."""
    xs = torch.tensor([1.0, 2.0, 3.0, 0.5, 1.5, 5.0, 10.0, 100.0],
                      dtype=torch.float32)
    expected = torch.special.polygamma(n, xs.to(torch.float64)).to(torch.float32)

    y = torch.ops.ops_multimodal_fusion.polygamma(xs.npu(), n).cpu()

    tol = _tolerances_for(n)
    assert torch.allclose(y, expected, **tol), (
        f"polygamma(n={n}) named-value mismatch; got {y.tolist()} "
        f"expected {expected.tolist()}"
    )
    logging.info(f"Named-values test passed (n={n}).")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_polygamma_trigamma_identities():
    """Closed-form trigamma (n=1) values: psi'(1)=pi^2/6, psi'(1/2)=pi^2/2, psi'(2)=pi^2/6-1."""
    pi_sq_over_6 = math.pi * math.pi / 6.0
    xs = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32)
    expected = torch.tensor(
        [pi_sq_over_6, pi_sq_over_6 - 1.0, math.pi * math.pi / 2.0],
        dtype=torch.float32,
    )
    y = torch.ops.ops_multimodal_fusion.polygamma(xs.npu(), 1).cpu()
    assert torch.allclose(y, expected, rtol=1e-4, atol=1e-5), (
        f"trigamma identity mismatch: got {y.tolist()} expected {expected.tolist()}"
    )
    logging.info("Trigamma identity test passed.")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("n", ORDERS)
def test_polygamma_wide_range(n):
    """Covers small, mid, and large positive inputs per supported order."""
    low = _low_bound_for(n)
    xs = torch.tensor(
        [low, 0.5, 1.0, 1.5, 2.718281828, 3.14159265,
         5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 10000.0],
        dtype=torch.float32,
    )
    # Filter out any entry below the per-n floor (0.5 already >= low for n>=3).
    xs = xs[xs >= low]

    y = torch.ops.ops_multimodal_fusion.polygamma(xs.npu(), n).cpu()
    expected = torch.special.polygamma(n, xs.to(torch.float64)).to(torch.float32)

    max_abs = (y - expected).abs().max().item()
    max_rel = ((y - expected) / expected.abs().clamp_min(1e-6)).abs().max().item()

    tol = _tolerances_for(n)
    # The small-x entries have |psi^(n)| ~ n!/low^(n+1), so a relative bound is
    # the only meaningful check across the full range.
    assert max_rel < tol["rtol"], (
        f"polygamma(n={n}) wide-range mismatch: max abs={max_abs}, max rel={max_rel}\n"
        f"got     = {y.tolist()}\n"
        f"expected= {expected.tolist()}"
    )
    logging.info(f"Wide-range test passed (n={n}, max rel={max_rel:.3e}).")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_polygamma_rejects_unsupported_orders():
    """n=0 (digamma) and n>6 are out of scope on dav-3510 and must error cleanly."""
    xs = torch.tensor([1.0, 2.0], dtype=torch.float32).npu()
    with pytest.raises(RuntimeError, match="polygamma"):
        torch.ops.ops_multimodal_fusion.polygamma(xs, 0)
    with pytest.raises(RuntimeError, match="polygamma"):
        torch.ops.ops_multimodal_fusion.polygamma(xs, 7)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_polygamma_empty_tensor():
    """Empty tensor passes through with matching shape/dtype (no kernel launch)."""
    x = torch.empty((0,), dtype=torch.float32).npu()
    y = torch.ops.ops_multimodal_fusion.polygamma(x, 2).cpu()
    assert y.shape == (0,)
    assert y.dtype == torch.float32
