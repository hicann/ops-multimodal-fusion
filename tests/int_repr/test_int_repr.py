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
"""Tests for ops_multimodal_fusion.int_repr.

Identity copy of the underlying integer storage of a (quantized) tensor:
y = x. Mirrors PyTorch's int_repr_quantized_cuda (IntReprQuant.cu) which
reads `value.val_` on qint8 / quint8 / qint32 inputs; ops_multimodal_fusion does not
expose quantized tensor types, so the operator is registered on plain
int8 / uint8 / int32 tensors and collapses to byte-exact identity copy.

Output is byte-exact equal to input by construction, so all assertions use
torch.equal (atol=rtol=0).

Test case counts:
  - test_int_repr_small             : 7  (minimum coverage; one per shape)
  - test_int_repr_large             : 32 (multi-tile / multi-core / 3D-4D /
                                          dtype-extremes sweep / numel=0 edge)
  - test_int_repr_interface_exist   : 1
  - test_int_repr_unsupported_dtype : 3
  - Total                           : 43 cases
"""

import pytest
import torch
import torch_npu  # noqa: F401  (registers NPU dispatch key)

import ops_multimodal_fusion  # noqa: F401  (registers torch.ops.ops_multimodal_fusion.int_repr)

if not hasattr(torch.ops.ops_multimodal_fusion, "int_repr"):
    pytest.skip(
        "ops_multimodal_fusion.int_repr not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_int_repr_interface_exist():
    """The 'ops_multimodal_fusion.int_repr' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "int_repr"),\
        "The 'int_repr' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Input generators.
#
# Three regimes exercise the byte-exact preservation requirement:
#   1. random    — uniform across the dtype's full sample range.
#   2. extremes  — every element forced to a boundary value (min / max / 0),
#                  catching any silent dtype demotion in the copy path.
#   3. zeros     — all-zero tensor (trivial baseline; smoke for size zero and
#                  alignment edge cases).
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


def _seed(seed_key):
    torch.manual_seed(abs(hash(repr(seed_key))) % 10_000_000)


def _make_random(shape, dtype, seed_key):
    """Uniform random across the dtype's full sample range."""
    _seed(seed_key)
    lo, hi = DTYPE_RANGES[dtype]
    return torch.randint(lo, hi, shape, dtype=dtype)


def _make_extremes(shape, dtype, seed_key):
    """Front-load dtype boundary values, fill the rest with random."""
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
    """All-zero tensor; smoke baseline including the numel=0 path."""
    return torch.zeros(shape, dtype=dtype)


_GENERATORS = {
    "random": _make_random,
    "extremes": _make_extremes,
    "zeros": _make_zeros,
}


def _gen_input(shape, gen_name, dtype, seed_key):
    return _GENERATORS[gen_name](shape, dtype, seed_key)


def _expected(x_cpu):
    """CPU reference: identity copy (clone)."""
    return x_cpu.clone()


def _run(x_cpu, label):
    expected = _expected(x_cpu)
    result = torch.ops.ops_multimodal_fusion.int_repr(x_cpu.npu()).cpu()
    assert result.dtype == x_cpu.dtype,\
        f"[{label}] dtype mismatch: got {result.dtype}, want {x_cpu.dtype}"
    assert result.shape == expected.shape,\
        f"[{label}] shape mismatch: got {tuple(result.shape)}, want {tuple(expected.shape)}"
    assert torch.equal(result, expected), (
        f"[{label}] value mismatch (dtype={x_cpu.dtype}, "
        f"shape={tuple(x_cpu.shape)})\n"
        f"  diff count = {(result != expected).sum().item()} / {expected.numel()}"
    )


# ---------------------------------------------------------------------------
# Case matrix (§6.3 of the issue).
#
# Quadruple: (shape, dtype, gen_name, label).
# SMALL: minimum coverage = one (dtype, gen) per distinct shape boundary.
# LARGE: everything else such as multi-tile, multi-core, 3D and 4D, dtype extremes sweep,
#        and the numel zero edge.
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
def test_int_repr_small(shape, dtype, gen_name, label):
    seed_key = ("small", tuple(shape), str(dtype), gen_name)
    x_cpu = _gen_input(shape, gen_name, dtype, seed_key)
    _run(x_cpu, label)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape,dtype,gen_name,label", CASES_LARGE)
def test_int_repr_large(shape, dtype, gen_name, label):
    seed_key = ("large", tuple(shape), str(dtype), gen_name)
    x_cpu = _gen_input(shape, gen_name, dtype, seed_key)
    _run(x_cpu, label)


# ---------------------------------------------------------------------------
# Negative-path tests: unsupported dtype rejected by host TORCH_CHECK.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int64])
def test_int_repr_unsupported_dtype(dtype):
    """Float and int64 inputs are not in the supported dispatch set."""
    x = torch.zeros((32,), dtype=dtype).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.int_repr(x)
