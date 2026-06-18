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
"""Tests for ops_multimodal_fusion.make_per_tensor_quantized.

Construct a per-tensor affine quantized tensor from integer storage:
y = x. Mirrors PyTorch's assign_quantized_tensor_cuda
(MakePerTensorQuantizedTensor.cu) which is `return scalar_t(value)` on
qint8 / quint8 / qint32 outputs; ops_multimodal_fusion does not expose quantized
tensor types, so the operator is registered on plain int8 / uint8 /
int32 tensors and collapses to byte-exact identity copy. scale and
zero_point are validated host-side (scale > 0; zero_point within the
input dtype range) and otherwise pass through.

Output is byte-exact equal to input by construction, so all positive
assertions use torch.equal (atol=rtol=0).

Test case counts:
  - test_make_per_tensor_quantized_small               : 7
  - test_make_per_tensor_quantized_large               : 32
  - test_make_per_tensor_quantized_valid_param_sweep   : 3
  - test_make_per_tensor_quantized_interface_exist     : 1
  - test_make_per_tensor_quantized_unsupported_dtype   : 3
  - test_make_per_tensor_quantized_invalid_scale       : 2
  - test_make_per_tensor_quantized_invalid_zero_point  : 4
  - Total                                              : 52 cases
"""

import pytest
import torch
import torch_npu  # noqa: F401  (registers NPU dispatch key)

import ops_multimodal_fusion  # noqa: F401  (registers torch.ops.ops_multimodal_fusion.*)

OP = "make_per_tensor_quantized"

if not hasattr(torch.ops.ops_multimodal_fusion, OP):
    pytest.skip(
        f"ops_multimodal_fusion.{OP} not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_make_per_tensor_quantized_interface_exist():
    """The 'ops_multimodal_fusion.make_per_tensor_quantized' operator is registered."""
    assert hasattr(torch.ops.ops_multimodal_fusion, OP), \
        f"The '{OP}' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Input generators (shared with int_repr — identity-copy regimes).
#
#   1. random    — uniform across the dtype's full sample range.
#   2. extremes  — every element forced to a boundary value (min / max / 0).
#   3. zeros     — all-zero tensor (smoke baseline including numel zero).
# Each generator returns a contiguous CPU tensor of the requested dtype.
# ---------------------------------------------------------------------------

DTYPE_RANGES = {
    torch.int8: (-128, 128),    # torch.randint hi is exclusive
    torch.uint8: (0, 256),
    torch.int32: (-(1 << 20), (1 << 20)),
}

DTYPE_EXTREMES = {
    torch.int8: [-128, 127, 0, -1, 1, 64, -64],
    torch.uint8: [0, 255, 1, 128, 254, 127, 200],
    torch.int32: [-(1 << 31), (1 << 31) - 1, 0, -1, 1, 1 << 30, -(1 << 30)],
}

# Representative valid (scale, zero_point) per dtype, baked into positive
# cases. scale must be > 0; zero_point must lie in the dtype's full range.
DTYPE_QUANT_PARAMS = {
    torch.int8: (0.5, -1),
    torch.uint8: (0.1, 128),
    torch.int32: (1.0, 0),
}


def _seed(seed_key):
    torch.manual_seed(abs(hash(repr(seed_key))) % 10_000_000)


def _make_random(shape, dtype, seed_key):
    _seed(seed_key)
    lo, hi = DTYPE_RANGES[dtype]
    return torch.randint(lo, hi, shape, dtype=dtype)


def _make_extremes(shape, dtype, seed_key):
    _seed(seed_key)
    numel = 1
    for d in shape:
        numel *= d
    if numel == 0:
        return torch.empty(shape, dtype=dtype)
    extremes = DTYPE_EXTREMES[dtype]
    lo, hi = DTYPE_RANGES[dtype]
    flat = torch.empty(numel, dtype=dtype)
    n_ext = min(len(extremes), numel)
    flat[:n_ext] = torch.tensor(extremes[:n_ext], dtype=dtype)
    if numel > n_ext:
        flat[n_ext:] = torch.randint(lo, hi, (numel - n_ext,), dtype=dtype)
    return flat.reshape(shape)


def _make_zeros(shape, dtype, seed_key):
    return torch.zeros(shape, dtype=dtype)


_GENERATORS = {
    "random": _make_random,
    "extremes": _make_extremes,
    "zeros": _make_zeros,
}


def _gen_input(shape, gen_name, dtype, seed_key):
    return _GENERATORS[gen_name](shape, dtype, seed_key)


def _expected(x_cpu):
    """CPU reference: identity copy (clone). scale / zero_point are not
    reflected in the output tensor for ops_multimodal_fusion (no qint dtype exposed).
    """
    return x_cpu.clone()


def _run(x_cpu, scale, zero_point, label):
    expected = _expected(x_cpu)
    result = torch.ops.ops_multimodal_fusion.make_per_tensor_quantized(
        x_cpu.npu(), scale, zero_point).cpu()
    assert result.dtype == x_cpu.dtype, \
        f"[{label}] dtype mismatch: got {result.dtype}, want {x_cpu.dtype}"
    assert result.shape == expected.shape, \
        f"[{label}] shape mismatch: got {tuple(result.shape)}, " \
        f"want {tuple(expected.shape)}"
    assert torch.equal(result, expected), (
        f"[{label}] value mismatch (dtype={x_cpu.dtype}, "
        f"shape={tuple(x_cpu.shape)}, scale={scale}, zp={zero_point})\n"
        f"  diff count = {(result != expected).sum().item()} / "
        f"{expected.numel()}"
    )


# ---------------------------------------------------------------------------
# Case matrix (§6.3 of the issue).
#
# Quadruple: (shape, dtype, gen_name, label).
# SMALL: minimum coverage = one (dtype, gen) per distinct shape boundary.
# LARGE: everything else such as multi-tile, multi-core, 3D and 4D, dtype extremes
#        sweep, and the numel zero edge.
# ---------------------------------------------------------------------------

# DMA 32B alignment boundary (no vector compute in this op; alignment is what
# matters, not VECTOR_REG_WIDTH lane count):
#   int8 and uint8: 32 lanes span 32B
#   int32: 8 lanes span 32B

CASES_SMALL = [
    # columns: shape, dtype, gen, label
    ((1,), torch.int32, "extremes", "scalar_int32"),
    ((33,), torch.int8, "random", "int8_align_plus1"),
    ((32,), torch.uint8, "random", "uint8_align_exact"),
    ((9,), torch.int32, "random", "int32_align_plus1"),
    ((8, 16), torch.int8, "extremes", "int8_2d_extremes"),
    ((4, 8), torch.uint8, "random", "uint8_2d"),
    ((128,), torch.int32, "extremes", "int32_extremes"),
]

CASES_LARGE = [
    # numel = 0 edge (host returns empty_like before kernel launch).
    ((0,), torch.int8, "zeros", "L_empty_int8"),
    ((0,), torch.int32, "zeros", "L_empty_int32"),
    ((4, 0, 8), torch.uint8, "zeros", "L_empty_3d_uint8"),
    # DMA 32B alignment neighbours not picked by small.
    ((31,), torch.int8, "random", "L_int8_align_minus1"),
    ((64,), torch.int32, "random", "L_int32_align_multi"),
    ((7,), torch.int32, "extremes", "L_int32_align_minus1"),
    ((16,), torch.uint8, "random", "L_uint8_align_multi"),
    ((256,), torch.int8, "random", "L_int8_align_8x"),
    # dtype extremes — boundary values must be byte-exact preserved.
    ((512,), torch.int8, "extremes", "L_int8_extremes_512"),
    ((512,), torch.uint8, "extremes", "L_uint8_extremes_512"),
    ((512,), torch.int32, "extremes", "L_int32_extremes_512"),
    # all-zero baselines on larger shapes.
    ((1024,), torch.int8, "zeros", "L_int8_zeros_1k"),
    ((1024,), torch.uint8, "zeros", "L_uint8_zeros_1k"),
    ((1024,), torch.int32, "zeros", "L_int32_zeros_1k"),
    # multi-tile single-core (cross several tiles within a single block).
    ((4096,), torch.int8, "random", "L_int8_4k"),
    ((4096,), torch.uint8, "random", "L_uint8_4k"),
    ((4096,), torch.int32, "random", "L_int32_4k"),
    # multi-core boundary — totalLength > MIN_ELEMS_PER_CORE * coreNum.
    ((131072,), torch.int8, "random", "L_int8_128k"),
    ((131072,), torch.uint8, "random", "L_uint8_128k"),
    ((131072,), torch.int32, "random", "L_int32_128k"),
    # 2D / 3D / 4D contiguous (view(-1) should be transparent).
    ((32, 32), torch.int8, "random", "L_int8_2d_square"),
    ((100, 100), torch.uint8, "random", "L_uint8_2d_oblong"),
    ((4, 8, 16), torch.int32, "random", "L_int32_3d"),
    ((2, 4, 8, 16), torch.int8, "extremes", "L_int8_4d_extremes"),
    # mixed random on medium 1D.
    ((1024,), torch.int8, "random", "L_int8_random_1k"),
    ((1024,), torch.uint8, "random", "L_uint8_random_1k"),
    ((1024,), torch.int32, "random", "L_int32_random_1k"),
    # Very small odd shapes (boundary stress).
    ((2,), torch.int8, "extremes", "L_int8_2_extremes"),
    ((3,), torch.uint8, "extremes", "L_uint8_3_extremes"),
    ((5,), torch.int32, "extremes", "L_int32_5_extremes"),
    # Long single-row 2D.
    ((1, 8192), torch.int32, "random", "L_int32_1x8k"),
    ((1, 65536), torch.int8, "random", "L_int8_1x64k"),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape,dtype,gen_name,label", CASES_SMALL)
def test_make_per_tensor_quantized_small(shape, dtype, gen_name, label):
    seed_key = ("small", tuple(shape), str(dtype), gen_name)
    x_cpu = _gen_input(shape, gen_name, dtype, seed_key)
    scale, zero_point = DTYPE_QUANT_PARAMS[dtype]
    _run(x_cpu, scale, zero_point, label)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape,dtype,gen_name,label", CASES_LARGE)
def test_make_per_tensor_quantized_large(shape, dtype, gen_name, label):
    seed_key = ("large", tuple(shape), str(dtype), gen_name)
    x_cpu = _gen_input(shape, gen_name, dtype, seed_key)
    scale, zero_point = DTYPE_QUANT_PARAMS[dtype]
    _run(x_cpu, scale, zero_point, label)


# ---------------------------------------------------------------------------
# Valid parameter sweep: vary scale and zero_point across the dtype's full
# legal range on a fixed small shape. Verifies host validation accepts the
# full legal range and kernel still produces byte-exact identity copy.
# ---------------------------------------------------------------------------

VALID_PARAM_SWEEP = [
    # columns: dtype, scale, zero_point, label
    (torch.int8, 1e-6, -128, "int8_min_scale_min_zp"),
    (torch.uint8, 12345.0, 255, "uint8_big_scale_max_zp"),
    (torch.int32, 0.25, (1 << 31) - 1, "int32_max_zp"),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype,scale,zero_point,label", VALID_PARAM_SWEEP)
def test_make_per_tensor_quantized_valid_param_sweep(
        dtype, scale, zero_point, label):
    seed_key = ("sweep", str(dtype), scale, zero_point)
    x_cpu = _gen_input((64,), "extremes", dtype, seed_key)
    _run(x_cpu, scale, zero_point, label)


# ---------------------------------------------------------------------------
# Negative-path tests: host TORCH_CHECK rejection.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int64])
def test_make_per_tensor_quantized_unsupported_dtype(dtype):
    """Float and int64 inputs are not in the supported dispatch set."""
    x = torch.zeros((32,), dtype=dtype).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.make_per_tensor_quantized(x, 1.0, 0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_make_per_tensor_quantized_invalid_scale(scale):
    """scale must be > 0."""
    x = torch.zeros((32,), dtype=torch.int8).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.make_per_tensor_quantized(x, scale, 0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype,zero_point", [
    (torch.int8, -129),
    (torch.int8, 128),
    (torch.uint8, -1),
    (torch.uint8, 256),
])
def test_make_per_tensor_quantized_invalid_zero_point(dtype, zero_point):
    """zero_point must lie within the input dtype's full value range."""
    x = torch.zeros((32,), dtype=dtype).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.make_per_tensor_quantized(x, 1.0, zero_point)
