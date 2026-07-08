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

if not hasattr(torch.ops.ops_multimodal_fusion, "kaiserwindow"):
    pytest.skip(
        "ops_multimodal_fusion.kaiserwindow not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_kaiserwindow_interface_exist():
    """Test that the 'ops_multimodal_fusion.kaiserwindow' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "kaiserwindow"),\
        "The 'kaiserwindow' operator is not registered."


def _run_kaiser_npu(window_length, periodic, beta):
    """Thin wrapper: prepare arange input, invoke op, apply periodic slice."""
    effective_n = window_length + (1 if periodic else 0)
    x = torch.arange(effective_n, dtype=torch.float32).npu()
    y = torch.ops.ops_multimodal_fusion.kaiserwindow(x, float(beta), int(window_length), bool(periodic)).cpu()
    if periodic:
        y = y[:-1]
    return y


# Window lengths covering: small, tile-aligned, non-aligned, and large shapes.
WINDOW_LENGTHS = [
    4,
    16,
    64,
    257,           # non tile-aligned odd size
    1024,
    4096,
    10000,
    100001,
]

# Beta values spanning both Bessel approximation regions and common window
# shapes. A beta of zero gives the rectangular window of all ones. A beta of
# one stays inside the first approximation region for every argument. A beta
# of five crosses the split between the two regions. A beta of twelve is the
# torch default and lies well inside the second region. A beta of twenty is
# large and exercises the tail of the second region.
BETAS = [0.0, 1.0, 5.0, 12.0, 20.0]

PERIODIC = [False, True]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("window_length", WINDOW_LENGTHS)
@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("periodic", PERIODIC)
def test_kaiserwindow(window_length, beta, periodic):
    """Compare against the torch kaiser window reference.

    The series approximation peaks at a relative error on the order of a few
    parts in ten million, and single-precision rounding during the polynomial
    evaluation widens this slightly, which sets the tolerances.
    """
    expected = torch.kaiser_window(window_length, periodic=periodic, beta=beta)
    result = _run_kaiser_npu(window_length, periodic, beta)

    assert result.shape == expected.shape,\
        f"Shape mismatch: got {result.shape}, expected {expected.shape}"

    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4),\
        f"KaiserWindow failed for N={window_length}, beta={beta}, periodic={periodic}. " \
        f"Max diff: {torch.max(torch.abs(result - expected)):.6f}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("beta", [1.0, 12.0])
def test_kaiserwindow_peak_is_one(beta):
    """The peak at the center bin of a non-periodic odd-length window should
    be exactly one because the host and kernel share the same Bessel
    approximation.
    """
    n = 101  # an odd length places an exact center bin at the midpoint
    y = _run_kaiser_npu(n, periodic=False, beta=beta)
    peak = float(y[n // 2])
    assert abs(peak - 1.0) < 1e-5,\
        f"Expected peak ~1.0, got {peak} (diff {peak - 1.0:.3e})"
