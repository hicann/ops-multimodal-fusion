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

"""Tests for ops_multimodal_fusion.quantized_relu.

Computes an elementwise maximum of the input against the zero point on signed
byte, unsigned byte, and signed word integer tensors. This mirrors the quantized
relu in PyTorch, which operates on the underlying integer storage of the
quantized tensor types.

Output is byte-exact equal to the CPU integer-maximum reference, so all
assertions are exact.

Test case counts:
  - test_quantized_relu_small       : 7 cases, minimum coverage one per shape
  - test_quantized_relu_large       : 36 cases, multi-tile multi-core higher
                                       rank zero-point sweep empty edge
  - test_quantized_relu_interface_exist     : 1
  - test_quantized_relu_zp_out_of_range     : 3
  - test_quantized_relu_unsupported_dtype   : 3
  - Total                                   : 50 cases
"""

import torch
import torch_npu  # noqa: F401  (registers NPU dispatch key)
import pytest
import ops_multimodal_fusion  # noqa: F401  (registers torch.ops.ops_multimodal_fusion.quantized_relu)

if not hasattr(torch.ops.ops_multimodal_fusion, "quantized_relu"):
    pytest.skip(
        "ops_multimodal_fusion.quantized_relu not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_quantized_relu_interface_exist():
    """The 'ops_multimodal_fusion.quantized_relu' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "quantized_relu"),\
        "The 'quantized_relu' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Input generators.
#
# Three regimes exercise the three semantic branches of the maximum:
#   1. mixed     - random uniform, with some lanes below and some at or above
#                  the zero point.
#   2. all_below - every element below the zero point, so the output is the zero
#                  point everywhere.
#   3. all_above - every element at or above the zero point, so the output equals
#                  the input byte for byte.
# Each generator returns a contiguous CPU tensor of the requested dtype.
# ---------------------------------------------------------------------------

DTYPE_RANGES = {
    torch.int8: (-128, 128),    # upper bound is exclusive
    torch.uint8: (0, 256),
    torch.int32: (-(1 << 20), (1 << 20)),
}


def _seed(seed_key):
    torch.manual_seed(abs(hash(repr(seed_key))) % 10_000_000)


def _make_mixed(shape, dtype, zp, seed_key):
    """Random uniform across the dtype's full sample range."""
    _seed(seed_key)
    lo, hi = DTYPE_RANGES[dtype]
    return torch.randint(lo, hi, shape, dtype=dtype)


def _make_all_below(shape, dtype, zp, seed_key):
    """Every element strictly below the zero point.

    The output is the zero point everywhere.
    """
    _seed(seed_key)
    lo, _ = DTYPE_RANGES[dtype]
    assert zp > lo, f"all_below requires zp > dtype min; got zp={zp}, lo={lo}"
    return torch.randint(lo, zp, shape, dtype=dtype)


def _make_all_above(shape, dtype, zp, seed_key):
    """Every element at or above the zero point.

    The output equals the input byte for byte.
    """
    _seed(seed_key)
    _, hi = DTYPE_RANGES[dtype]
    assert zp < hi, f"all_above requires zp < dtype max; got zp={zp}, hi={hi}"
    return torch.randint(zp, hi, shape, dtype=dtype)


_GENERATORS = {
    "mixed": _make_mixed,
    "all_below": _make_all_below,
    "all_above": _make_all_above,
}


def _gen_input(shape, gen_name, dtype, zp, seed_key):
    return _GENERATORS[gen_name](shape, dtype, zp, seed_key)


def _expected(x_cpu, zero_point):
    """CPU reference: elementwise maximum against the zero point on the integer
    storage.

    Promote to the wide integer type before the maximum to avoid overflow in the
    narrow ranges, then cast back to the original dtype for byte-exact comparison.
    """
    zp_tensor = torch.tensor(zero_point, dtype=torch.int64)
    return torch.maximum(x_cpu.to(torch.int64), zp_tensor).to(x_cpu.dtype)


def _run(x_cpu, zero_point, label):
    expected = _expected(x_cpu, zero_point)
    result = torch.ops.ops_multimodal_fusion.quantized_relu(
        x_cpu.npu(), int(zero_point)
    ).cpu()
    assert result.dtype == x_cpu.dtype,\
        f"[{label}] dtype mismatch: got {result.dtype}, want {x_cpu.dtype}"
    assert result.shape == expected.shape,\
        f"[{label}] shape mismatch: got {tuple(result.shape)}, want {tuple(expected.shape)}"
    assert torch.equal(result, expected), (
        f"[{label}] value mismatch (dtype={x_cpu.dtype}, "
        f"shape={tuple(x_cpu.shape)}, zp={zero_point})\n"
        f"  diff count = {(result != expected).sum().item()} / {expected.numel()}"
    )


# ---------------------------------------------------------------------------
# Case matrix.
#
# Each row carries shape, dtype, zero point, generator name, and a label. The
# small set is minimum coverage, one dtype and generator per distinct shape
# boundary. The large set covers everything else: multi-tile, multi-core, higher
# rank, the zero-point sweep, and the empty edge.
# ---------------------------------------------------------------------------

# Vector register lane counts: the byte dtypes hold many more lanes per register
# than the word dtype, so the chosen shapes straddle each boundary.

CASES_SMALL = [
    ((1,), torch.int32, 0, "mixed", "scalar_int32"),
    ((63,), torch.int32, 0, "mixed", "int32_vecw_minus1"),
    ((257,), torch.int8, -10, "mixed", "int8_vecw_plus1"),
    ((128,), torch.uint8, 200, "all_below", "uint8_all_below"),
    ((128,), torch.int8, -100, "all_above", "int8_all_above"),
    ((8, 32), torch.uint8, 50, "mixed", "uint8_2d_mid_zp"),
    ((256,), torch.int8, 127, "mixed", "int8_zp_at_max"),
]

CASES_LARGE = [
    # Vector register lane neighbours not picked by the small set.
    ((65,), torch.int32, 0, "mixed", "L_int32_vecw_plus1"),
    ((64,), torch.int32, 100, "mixed", "L_int32_vecw_exact"),
    ((256,), torch.uint8, 0, "mixed", "L_uint8_vecw_exact"),
    ((255,), torch.uint8, 64, "mixed", "L_uint8_vecw_minus1"),
    ((512,), torch.int8, 0, "mixed", "L_int8_two_chunks"),
    # Empty tensor edge.
    ((0,), torch.int32, 0, "mixed", "L_empty_int32"),
    ((0,), torch.int8, 0, "mixed", "L_empty_int8"),
    ((4, 0, 8), torch.uint8, 10, "mixed", "L_empty_3d"),
    # Zero-point sweep over the signed byte type.
    ((1024,), torch.int8, -128, "mixed", "L_int8_zp_min"),
    ((1024,), torch.int8, -64, "mixed", "L_int8_zp_neg"),
    ((1024,), torch.int8, 0, "mixed", "L_int8_zp_zero"),
    ((1024,), torch.int8, 64, "mixed", "L_int8_zp_pos"),
    # Zero-point sweep over the unsigned byte type.
    ((1024,), torch.uint8, 0, "mixed", "L_uint8_zp_zero"),
    ((1024,), torch.uint8, 128, "mixed", "L_uint8_zp_mid"),
    ((1024,), torch.uint8, 255, "mixed", "L_uint8_zp_max"),
    # Zero-point sweep over the signed word type.
    ((1024,), torch.int32, -(1 << 30), "mixed", "L_int32_zp_neg_large"),
    ((1024,), torch.int32, -1, "mixed", "L_int32_zp_neg_one"),
    ((1024,), torch.int32, 0, "mixed", "L_int32_zp_zero"),
    ((1024,), torch.int32, 1 << 30, "mixed", "L_int32_zp_pos_large"),
    # Multi-tile single-core boundary, crossing several tiles within one block.
    ((4096,), torch.int8, 0, "mixed", "L_int8_4k"),
    ((4096,), torch.uint8, 100, "mixed", "L_uint8_4k"),
    ((4096,), torch.int32, 0, "mixed", "L_int32_4k"),
    # Multi-core boundary, large enough that work spreads across cores.
    ((131072,), torch.int8, -32, "mixed", "L_int8_128k"),
    ((131072,), torch.uint8, 128, "mixed", "L_uint8_128k"),
    ((131072,), torch.int32, 0, "mixed", "L_int32_128k"),
    # Higher-rank contiguous tensors, where the elementwise op is rank agnostic.
    ((32, 32), torch.int8, 0, "mixed", "L_int8_2d_square"),
    ((100, 100), torch.uint8, 127, "mixed", "L_uint8_2d_oblong"),
    ((4, 8, 16), torch.int32, 0, "mixed", "L_int32_3d"),
    ((2, 4, 8, 16), torch.int8, -16, "mixed", "L_int8_4d"),
    # The below-everything and above-everything regimes on larger shapes.
    ((4096,), torch.int32, 1 << 19, "all_below", "L_int32_all_below_4k"),
    ((4096,), torch.int8, -120, "all_above", "L_int8_all_above_4k"),
    ((1024,), torch.uint8, 254, "all_below", "L_uint8_all_below_high"),
    # Signed word type at the extreme zero points.
    ((512,), torch.int32, -(1 << 31), "mixed", "L_int32_zp_INT_MIN"),
    ((512,), torch.int32, (1 << 31) - 1, "all_below", "L_int32_zp_INT_MAX"),
    # Unsigned byte type near its minimum.
    ((1024,), torch.uint8, 1, "mixed", "L_uint8_zp_one"),
    # Signed byte type at its maximum.
    ((1024,), torch.int8, 127, "mixed", "L_int8_zp_max"),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape,dtype,zp,gen_name,label", CASES_SMALL)
def test_quantized_relu_small(shape, dtype, zp, gen_name, label):
    seed_key = ("small", tuple(shape), str(dtype), int(zp), gen_name)
    x_cpu = _gen_input(shape, gen_name, dtype, zp, seed_key)
    _run(x_cpu, zp, label)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape,dtype,zp,gen_name,label", CASES_LARGE)
def test_quantized_relu_large(shape, dtype, zp, gen_name, label):
    seed_key = ("large", tuple(shape), str(dtype), int(zp), gen_name)
    x_cpu = _gen_input(shape, gen_name, dtype, zp, seed_key)
    _run(x_cpu, zp, label)


# ---------------------------------------------------------------------------
# Negative-path tests: zero_point range + unsupported dtype.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype,bad_zp", [
    (torch.int8, 128),      # one past the signed byte maximum
    (torch.uint8, -1),      # one below the unsigned byte minimum
    (torch.int32, 1 << 33), # well past the signed word maximum
])


def test_quantized_relu_zp_out_of_range(dtype, bad_zp):
    """zero_point outside the input dtype's representable range must be rejected."""
    x = torch.zeros((32,), dtype=dtype).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.quantized_relu(x, int(bad_zp))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int64])
def test_quantized_relu_unsupported_dtype(dtype):
    """Float and int64 inputs are not in the supported dispatch set."""
    x = torch.zeros((32,), dtype=dtype).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.quantized_relu(x, 0)
