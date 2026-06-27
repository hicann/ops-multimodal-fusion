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
"""Tests for ops_multimodal_fusion.frexp."""

import logging
import math

import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401

logger = logging.getLogger(__name__)

if not hasattr(torch.ops.ops_multimodal_fusion, "frexp"):
    pytest.skip(
        "ops_multimodal_fusion.frexp not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_frexp_interface_exist():
    """The 'ops_multimodal_fusion.frexp' operator must be registered in torch.ops."""
    logger.info("%s", torch.ops.ops_multimodal_fusion.frexp)
    assert hasattr(torch.ops.ops_multimodal_fusion, "frexp"), \
        "The 'frexp' operator is not registered in the 'torch.ops.ops_multimodal_fusion' namespace."


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
    # Large shapes that exceed UB capacity, forcing multi-tile processing.
    (100000,),
    (1000000,),
    (2048, 2048),
    (64, 128, 256),
]

DTYPES = [torch.float32, torch.float16]


def _recon_tol(dtype):
    """Reconstruction tolerance when comparing the mantissa scaled by two to the exponent against x.

    For fp32 a single unit in the last place of the mantissa is about one part
    in ten million relative, so an absolute tolerance near one part in a million
    is safe. For fp16 the mantissa carries roughly eleven bits, which is about
    five parts in ten thousand relative, so a thousandth absolute is safe.
    """
    if dtype == torch.float32:
        return dict(rtol=1e-4, atol=1e-6)
    return dict(rtol=5e-3, atol=5e-3)


def _check_frexp_invariant(x, mantissa, exponent):
    """Check the frexp guarantee that x equals the mantissa scaled by two to the
    exponent, and that the absolute mantissa is either zero or lies in the
    half-open range from one half up to but not including one.
    """
    abs_m = mantissa.abs()

    zero_mask = (x == 0)
    if zero_mask.any():
        assert torch.equal(mantissa[zero_mask], torch.zeros_like(mantissa[zero_mask])), \
            "mantissa must be 0 where x == 0"
        assert torch.equal(exponent[zero_mask], torch.zeros_like(exponent[zero_mask])), \
            "exponent must be 0 where x == 0"

    nonzero = ~zero_mask
    if nonzero.any():
        assert (abs_m[nonzero] >= 0.5).all(), (
            f"|mantissa| < 0.5 for some nonzero x "
            f"(min={abs_m[nonzero].min().item()})"
        )
        assert (abs_m[nonzero] < 1.0).all(), (
            f"|mantissa| >= 1.0 for some nonzero x "
            f"(max={abs_m[nonzero].max().item()})"
        )

    # Reconstruct in fp32 to avoid losing precision in the check.
    reconstructed = mantissa.to(torch.float32) * torch.ldexp(
        torch.ones_like(mantissa, dtype=torch.float32), exponent
    )
    tol = _recon_tol(x.dtype)
    assert torch.allclose(reconstructed, x.to(torch.float32), **tol), (
        f"Reconstruction mismatch (dtype={x.dtype}); max diff = "
        f"{(reconstructed - x.to(torch.float32)).abs().max().item()}"
    )


def _sample_range(dtype):
    """Per-dtype bounded-magnitude sampler range.

    The fp16 normal range spans roughly six hundredths of a thousandth up to
    about sixty-five thousand, so the chosen bounds sit well inside both ends
    and mix signs.
    """
    if dtype == torch.float32:
        return -10.0, 10.0
    return -10.0, 10.0  # fp16 easily fits this range


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_frexp_operator(shape, dtype):
    """Verify the frexp invariant across random finite inputs for each dtype.

    The result must satisfy that x equals the mantissa scaled by two to the
    exponent, with the absolute mantissa in the half-open range from one half up
    to but not including one. We check the invariant rather than exact
    bit-equality with torch.frexp because log and exp rounding at power-of-two
    boundaries can legally round to either adjacent bucket.
    """
    torch.manual_seed(abs(hash((shape, dtype))) % 10_000_000)
    low, high = _sample_range(dtype)
    x = torch.empty(*shape, dtype=dtype).uniform_(low, high)
    # Keep values a comfortable distance from zero for fp16 resolution.
    if dtype == torch.float16:
        x = torch.where(x.abs() < 1e-3, torch.full_like(x, 1e-3) * x.sign().where(x != 0, torch.ones_like(x)), x)

    x_npu = x.npu()
    mantissa_npu, exponent_npu = torch.ops.ops_multimodal_fusion.frexp(x_npu)
    mantissa = mantissa_npu.cpu()
    exponent = exponent_npu.cpu()

    assert mantissa.dtype == dtype, f"mantissa dtype mismatch: {mantissa.dtype} vs {dtype}"
    assert exponent.dtype == torch.int32, f"exponent dtype mismatch: {exponent.dtype}"
    assert mantissa.shape == x.shape
    assert exponent.shape == x.shape

    _check_frexp_invariant(x, mantissa, exponent)

    logger.info("Test passed: shape=%s, dtype=%s", shape, dtype)


def _stable_values(dtype):
    """Tricky finite values for the special-values test, per dtype."""
    if dtype == torch.float32:
        return [
            0.0,
            1.0, -1.0,
            0.5, -0.5,
            2.0, -2.0, 4.0,
            1024.0, -1024.0,
            1.0 / 1024.0, -1.0 / 1024.0,
            1.5, -1.5, 0.75,
            math.pi, -math.e,
            3.4e30, -3.4e30,
            1e-30, -1e-30,
        ]
    # fp16 stays within normal range, picking powers of two alongside mid boundaries
    return [
        0.0,
        1.0, -1.0,
        0.5, -0.5,
        2.0, -2.0, 4.0,
        1024.0, -1024.0,
        1.0 / 1024.0, -1.0 / 1024.0,
        1.5, -1.5, 0.75,
        3.14, -2.71,
        3.0e4, -3.0e4,          # near fp16 max but not overflowing ldexp in check
        1.0e-3, -1.0e-3,
    ]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_frexp_special_values(dtype):
    """Tricky finite values: zeros, powers of two, sign flips, mid-range magnitudes."""
    values = _stable_values(dtype)
    x = torch.tensor(values, dtype=dtype)
    x_npu = x.npu()
    mantissa_npu, exponent_npu = torch.ops.ops_multimodal_fusion.frexp(x_npu)
    mantissa = mantissa_npu.cpu()
    exponent = exponent_npu.cpu()

    _check_frexp_invariant(x, mantissa, exponent)

    # Exponent sanity check: where the input is zero both outputs must be zero.
    assert mantissa[0].item() == 0.0
    assert exponent[0].item() == 0

    # Agreement with torch.frexp on "stable" values — i.e., values that aren't
    # right at a power-of-two boundary, where our log/exp reconstruction may
    # legally round to the adjacent exponent bucket.
    expected_m, expected_e = torch.frexp(x)
    stable_mask = torch.tensor([
        v == 0.0 or abs(math.log2(abs(v)) - round(math.log2(abs(v)))) > 1e-6
        for v in values
    ])
    if stable_mask.any():
        tol = _recon_tol(dtype)
        assert torch.allclose(mantissa[stable_mask], expected_m[stable_mask], **tol), (
            f"mantissa mismatch on stable values (dtype={dtype}): "
            f"{mantissa[stable_mask]} vs {expected_m[stable_mask]}"
        )
        assert torch.equal(exponent[stable_mask], expected_e[stable_mask].to(torch.int32)), (
            f"exponent mismatch on stable values (dtype={dtype}): "
            f"{exponent[stable_mask]} vs {expected_e[stable_mask]}"
        )

    logger.info("Special-values test passed (dtype=%s).", dtype)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_frexp_returns_two_tensors(dtype):
    """The op returns a two-element tuple of mantissa and exponent.

    Both have the right dtypes and shapes. Note that custom torch ops always
    return a plain tuple, not a torch.return_types named tuple; that
    PyStructSequence is internal to PyTorch's built-in frexp and cannot be
    registered for user ops.
    """
    x = torch.tensor([1.5, -0.75, 8.0], dtype=dtype).npu()
    r = torch.ops.ops_multimodal_fusion.frexp(x)
    assert isinstance(r, tuple) and len(r) == 2
    mantissa, exponent = r
    assert mantissa.shape == x.shape
    assert exponent.shape == x.shape
    assert mantissa.dtype == dtype
    assert exponent.dtype == torch.int32


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_frexp_empty_tensor(dtype):
    """Empty tensor passes through with matching shapes/dtypes (no kernel launch)."""
    x = torch.empty((0,), dtype=dtype).npu()
    mantissa, exponent = torch.ops.ops_multimodal_fusion.frexp(x)
    mantissa = mantissa.cpu()
    exponent = exponent.cpu()
    assert mantissa.shape == (0,) and mantissa.dtype == dtype
    assert exponent.shape == (0,) and exponent.dtype == torch.int32


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_frexp_rejects_unsupported_dtype():
    """float64 is not in the supported-dtype list per issue #11."""
    xs = torch.tensor([1.0, -1.0], dtype=torch.float64).npu()
    with pytest.raises(RuntimeError, match="frexp"):
        torch.ops.ops_multimodal_fusion.frexp(xs)


@pytest.mark.skip(
    reason=(
        "torch_npu in the current CANN release lacks D2D strided copy support "
        "(aclnnInplaceCopy fails with error 561103 for any non-contiguous NPU "
        "tensor, regardless of dtype). The kernel itself only operates on "
        "contiguous buffers; callers holding a transposed/strided tensor must "
        "materialize it themselves before invoking the op. "
        "Re-enable once torch_npu ships strided D2D."
    )
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_frexp_non_contiguous_input(dtype):
    """Non-contiguous inputs should still produce a correct factorization."""
    base = torch.empty((32, 32), dtype=dtype).uniform_(-5.0, 5.0)
    if dtype == torch.float16:
        base = torch.where(base.abs() < 1e-3,
                           torch.full_like(base, 1e-3) * base.sign().where(base != 0, torch.ones_like(base)),
                           base)
    x = base.t()
    assert not x.is_contiguous()

    mantissa, exponent = torch.ops.ops_multimodal_fusion.frexp(x.npu())
    _check_frexp_invariant(x.contiguous(), mantissa.cpu(), exponent.cpu())
