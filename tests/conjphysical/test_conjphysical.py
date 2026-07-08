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
import math

import pytest
import torch
import torch_npu  # noqa: F401

import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "conjphysical"):
    pytest.skip(
        "ops_multimodal_fusion.conjphysical not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_conjphysical_interface_exist():
    """The 'ops_multimodal_fusion.conjphysical' operator must be registered in torch.ops."""
    logging.info(torch.ops.ops_multimodal_fusion.conjphysical)
    assert hasattr(torch.ops.ops_multimodal_fusion, "conjphysical"),\
        "The 'conjphysical' operator is not registered in the 'torch.ops.ops_multimodal_fusion' namespace."


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
    # Large shapes exceeding UB capacity, forcing multi-tile processing.
    (100000,),
    (1000000,),
    (2048, 2048),
    (64, 128, 256),
]

DTYPES = [torch.float32, torch.float16]


def _sample_mixed(shape, dtype, seed=0, low=-10.0, high=10.0):
    g = torch.Generator().manual_seed(seed)
    return torch.empty(*shape, dtype=dtype).uniform_(low, high, generator=g)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_conjphysical_operator(shape, dtype):
    """Real-input conjphysical must produce a tensor equal to the input."""
    seed = abs(hash(shape)) % 997
    x = _sample_mixed(shape, dtype, seed=seed)

    x_npu = x.npu()
    y_npu = torch.ops.ops_multimodal_fusion.conjphysical(x_npu)
    y = y_npu.cpu()

    assert y.dtype == dtype, f"dtype mismatch: {y.dtype} vs {dtype}"
    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"

    # For real input, torch.conj_physical(x) == x (per the PyTorch docs).
    # `x * 1.0` via AscendC::Muls is bit-exact for every normal/subnormal
    # value in both float32 and float16.
    expected = torch.conj_physical(x)
    assert torch.equal(y, expected), (
        f"conjphysical mismatch (dtype={dtype}): "
        f"max abs diff = {(y - expected).abs().max().item()}"
    )
    logging.info(f"Test passed: shape={shape}, dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_conjphysical_output_is_new_tensor():
    """Output must be a fresh tensor (not a view sharing storage with input)."""
    x = torch.tensor([1.0, -2.0, 3.5], dtype=torch.float32).npu()
    y = torch.ops.ops_multimodal_fusion.conjphysical(x)
    # Use storage identity rather than mutate-and-observe: torch_npu's
    # in-place ops on custom-op outputs can hit aclnnInplaceCopy (561103)
    # even on contiguous tensors, which is unrelated to aliasing.
    assert y.data_ptr() != x.data_ptr(),\
        "conjphysical returned a view aliased to the input; expected a copy"
    assert y.untyped_storage().data_ptr() != x.untyped_storage().data_ptr(),\
        "conjphysical output shares storage with the input"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_conjphysical_signed_zero_and_special_finite(dtype):
    """+0.0, -0.0, and assorted finite values including tiny subnormals for fp32."""
    if dtype == torch.float32:
        values = [0.0, -0.0,
                  1.0, -1.0,
                  3.14159265, -2.71828182,
                  1e-30, -1e-30,
                  3.4028234e38, -3.4028234e38]   # near FLT_MAX
    else:  # fp16 — avoid magnitudes that don't round-trip cleanly
        values = [0.0, -0.0,
                  1.0, -1.0,
                  3.14, -2.71,
                  1e-3, -1e-3,
                  65504.0, -65504.0]             # fp16 max
    xs = torch.tensor(values, dtype=dtype)
    expected = torch.conj_physical(xs)

    y = torch.ops.ops_multimodal_fusion.conjphysical(xs.npu()).cpu()
    assert torch.equal(y, expected), (
        f"conjphysical special-values mismatch (dtype={dtype}); "
        f"got {y.tolist()} expected {expected.tolist()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_conjphysical_nan_is_nan(dtype):
    """NaN input produces NaN output (bit payload may differ by hardware, but isnan holds)."""
    xs = torch.tensor(
        [float('nan'), 1.0, -1.0, float('nan'), 0.5],
        dtype=dtype,
    )
    y = torch.ops.ops_multimodal_fusion.conjphysical(xs.npu()).cpu()
    # NaN positions must remain NaN.
    assert torch.isnan(y[0]).item() and torch.isnan(y[3]).item(),\
        f"NaN was not preserved (dtype={dtype}): got {y.tolist()}"
    # Non-NaN positions must equal input exactly (x * 1.0 is bit-exact here).
    assert y[1].item() == 1.0 and y[2].item() == -1.0 and y[4].item() == 0.5,\
        f"non-NaN positions altered (dtype={dtype}): got {y.tolist()}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_conjphysical_inf_passthrough(dtype):
    """+inf and -inf pass through unchanged (inf * 1.0 == inf in IEEE-754)."""
    xs = torch.tensor([float('inf'), -float('inf'), 1.0, -1.0], dtype=dtype)
    expected = torch.conj_physical(xs)
    y = torch.ops.ops_multimodal_fusion.conjphysical(xs.npu()).cpu()
    assert torch.equal(y, expected), (
        f"inf passthrough mismatch (dtype={dtype}): "
        f"got {y.tolist()} expected {expected.tolist()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_conjphysical_empty_tensor(dtype):
    """Empty tensor passes through with matching shape/dtype (no kernel launch)."""
    x = torch.empty((0,), dtype=dtype).npu()
    y = torch.ops.ops_multimodal_fusion.conjphysical(x).cpu()
    assert y.shape == (0,)
    assert y.dtype == dtype


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
def test_conjphysical_non_contiguous_input(dtype):
    """Non-contiguous tensors are handled (kernel makes a contiguous copy)."""
    base = _sample_mixed((32, 32), dtype, seed=11)
    x = base.t()  # transpose -> non-contiguous
    assert not x.is_contiguous()

    y = torch.ops.ops_multimodal_fusion.conjphysical(x.npu()).cpu()
    expected = torch.conj_physical(x.contiguous())
    assert torch.equal(y, expected), (
        f"non-contiguous mismatch (dtype={dtype}): "
        f"max abs diff = {(y - expected).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_conjphysical_rejects_complex_dtype():
    """Complex dtypes are not supported on dav-3510 and must error cleanly."""
    # Build a complex tensor on CPU first, then try to move it — the op itself
    # is the one that will reject; we just need any complex input.
    x = torch.tensor([1.0 + 2.0j, -3.0 - 4.0j], dtype=torch.complex64)
    # Some NPU environments cannot allocate complex tensors at all. If that's
    # the case, .npu() itself may raise -- which is also a form of rejection,
    # so we accept either failure mode.
    with pytest.raises((RuntimeError, TypeError)):
        y = torch.ops.ops_multimodal_fusion.conjphysical(x.npu())
        _ = y.cpu()


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_conjphysical_rejects_float64():
    """float64 is not in the supported-dtype list per issue #31."""
    xs = torch.tensor([1.0, -1.0], dtype=torch.float64).npu()
    with pytest.raises(RuntimeError, match="conjphysical"):
        torch.ops.ops_multimodal_fusion.conjphysical(xs)
