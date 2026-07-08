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

"""Tests for ops_multimodal_fusion.c2_boolean_mask.

Caffe2-style boolean mask selection, a gather along the first dimension.
The output gathers the rows of data whose mask entry is true, and it also
returns the positions of those true entries in order.
The mask is a one dim bool or uint8 tensor whose length matches the first
dimension of data, and the output length is the number of true entries in
the mask, which is data dependent.

Golden references use torch index_select over the nonzero positions of the
mask for the data, and the nonzero positions themselves cast to int64 for
the indices.

Test case counts:
  - test_c2_boolean_mask_small              : 7 cases, minimal coverage per
                                              shape, dtype and mask mode
  - test_c2_boolean_mask_large              : 11 cases (3 dim and 4 dim
                                              shapes, large N or K, density edges)
  - test_c2_boolean_mask_interface_exist    : 1
  - test_c2_boolean_mask_empty_data         : 1
  - test_c2_boolean_mask_all_false          : 1
  - test_c2_boolean_mask_all_true           : 1
  - test_c2_boolean_mask_invalid_ndim       : 1
  - test_c2_boolean_mask_size_mismatch      : 1
  - test_c2_boolean_mask_mask_dim           : 1
  - test_c2_boolean_mask_mask_dtype         : 1
  - test_c2_boolean_mask_bf16_rejected      : 1
  - test_c2_boolean_mask_int64_rejected     : 1
  - Total                                   : 28 cases
"""

import torch
import torch_npu  # noqa: F401
import pytest
import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "c2_boolean_mask"):
    pytest.skip(
        "ops_multimodal_fusion.c2_boolean_mask not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_c2_boolean_mask_interface_exist():
    """'ops_multimodal_fusion.c2_boolean_mask' is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "c2_boolean_mask"),\
        "The 'c2_boolean_mask' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Input generators. All deterministic via seed_key.
# ---------------------------------------------------------------------------


def _seed(seed_key):
    torch.manual_seed(abs(hash(repr(seed_key))) % 10_000_000)


def _gen_data(shape, dtype, sk):
    _seed(sk)
    if dtype == torch.int32:
        return torch.randint(-(2**20), 2**20, shape, dtype=dtype)
    return torch.randn(shape).to(dtype)


# Mask generators. Each returns a one dim tensor of length N.
# mask_dtype controls the output dtype, either bool or uint8.


def _mask_all_true(n, mask_dtype, sk):
    return torch.ones(n, dtype=mask_dtype)


def _mask_all_false(n, mask_dtype, sk):
    return torch.zeros(n, dtype=mask_dtype)


def _mask_single_true(n, mask_dtype, sk):
    m = torch.zeros(n, dtype=mask_dtype)
    if n > 0:
        m[n // 2] = 1
    return m


def _mask_random_50(n, mask_dtype, sk):
    _seed(("mask_random_50", sk))
    raw = torch.randint(0, 2, (n,), dtype=torch.int64)
    return raw.to(mask_dtype)


def _mask_sparse_10(n, mask_dtype, sk):
    # about 10 percent true: every 10th position true.
    _seed(("mask_sparse_10", sk))
    raw = (torch.rand(n) < 0.1).to(mask_dtype)
    return raw


def _mask_dense_90(n, mask_dtype, sk):
    # about 90 percent true
    _seed(("mask_dense_90", sk))
    raw = (torch.rand(n) < 0.9).to(mask_dtype)
    return raw


_MASK_GENERATORS = {
    "all_true": _mask_all_true,
    "all_false": _mask_all_false,
    "single_true": _mask_single_true,
    "random_50": _mask_random_50,
    "sparse_10": _mask_sparse_10,
    "dense_90": _mask_dense_90,
}


def _gen_input(shape, dtype, mask_mode, mask_dtype, seed_key):
    data = _gen_data(shape, dtype, seed_key)
    n = data.size(0)
    mask = _MASK_GENERATORS[mask_mode](n, mask_dtype, seed_key)
    return data, mask


# ---------------------------------------------------------------------------
# Reference: torch.index_select on dim 0 over nonzero(mask) -- semantically
# identical to Caffe2 BooleanMask forward.
# ---------------------------------------------------------------------------


def _reference(data: torch.Tensor, mask: torch.Tensor):
    mask_cpu = mask.cpu().to(torch.bool)
    indices = torch.nonzero(mask_cpu).squeeze(-1).to(torch.int64)
    masked_data = torch.index_select(data.cpu(), 0, indices)
    return masked_data, indices


def _run(data_cpu, mask_cpu, label_str):
    expected_data, expected_indices = _reference(data_cpu, mask_cpu)
    result_data_npu, result_indices_npu = torch.ops.ops_multimodal_fusion.c2_boolean_mask(
        data_cpu.npu(), mask_cpu.npu()
    )
    result_data = result_data_npu.cpu()
    result_indices = result_indices_npu.cpu()

    assert result_data.dtype == data_cpu.dtype,\
        f"[{label_str}] data dtype: got {result_data.dtype}, want {data_cpu.dtype}"
    expected_shape = (expected_indices.numel(),) + tuple(data_cpu.shape[1:])
    assert tuple(result_data.shape) == expected_shape,\
        f"[{label_str}] data shape: got {tuple(result_data.shape)}, want {expected_shape}"
    assert result_indices.dtype == torch.int64,\
        f"[{label_str}] indices dtype: got {result_indices.dtype}, want int64"
    assert tuple(result_indices.shape) == (expected_indices.numel(),),\
        f"[{label_str}] indices shape: got {tuple(result_indices.shape)}, " \
        f"want {(expected_indices.numel(),)}"

    # Both the gather and the nonzero path are bytewise exact for every dtype.
    assert torch.equal(result_data, expected_data), (
        f"[{label_str}] data bytewise mismatch (shape={tuple(data_cpu.shape)}, "
        f"dtype={data_cpu.dtype})"
    )
    assert torch.equal(result_indices, expected_indices), (
        f"[{label_str}] indices mismatch (shape={tuple(data_cpu.shape)}, "
        f"mask_dtype={mask_cpu.dtype})"
    )


# ---------------------------------------------------------------------------
# Case matrix.
# Labels avoid the substrings "small" or "large" so the pytest keyword
# filters do not accidentally collide.
# ---------------------------------------------------------------------------

# Small cases: minimal coverage per shape, dtype, vec width and mask mode.
# Each tuple holds label, shape, dtype, mask_mode and mask_dtype.
CASES_SMALL = [
    # vec width aligned fp32 with sixty four lanes, half random mask.
    ("fp32_n8_k64_random_bool", (8, 64), torch.float32, "random_50", torch.bool),
    # vec width aligned fp16 with one twenty eight lanes, all true boundary.
    ("fp16_n8_k128_alltrue_bool", (8, 128), torch.float16, "all_true", torch.bool),
    # int32 unaligned width just below the lane count, uint8 mask path.
    ("int32_n9_k63_random_uint8", (9, 63), torch.int32, "random_50", torch.uint8),
    # single column scalar gather with a single batch row.
    ("fp32_n1_k1_singletrue_bool", (1, 1), torch.float32, "single_true", torch.bool),
    # fp16 multi row per core, sparse mask, width crosses the lane boundary.
    ("fp16_n128_k129_sparse_bool", (128, 129), torch.float16, "sparse_10", torch.bool),
    # int32 partial core with fewer rows than cores, all false empty output edge.
    ("int32_n7_k65_allfalse_bool", (7, 65), torch.int32, "all_false", torch.bool),
    # fp32 large width within UB, dense mask with full multi core load.
    ("fp32_n32_k1024_dense_bool", (32, 1024), torch.float32, "dense_90", torch.bool),
]

# Large cases: three dim and four dim shapes, large N, mask edges.
CASES_LARGE = [
    # three dim shape that flattens the trailing dims.
    ("fp32_3d_n4_8x8_random_bool", (4, 8, 8), torch.float32, "random_50", torch.bool),
    ("fp16_3d_n8_16x16_dense_bool", (8, 16, 16), torch.float16, "dense_90", torch.bool),
    # four dim feature map at a typical ResNet FPN scale.
    ("fp32_4d_n16_3x7x7_random_bool", (16, 3, 7, 7), torch.float32, "random_50", torch.bool),
    ("fp16_4d_n8_4x14x14_sparse_bool", (8, 4, 14, 14), torch.float16, "sparse_10", torch.bool),
    ("int32_4d_n4_2x4x4_random_uint8", (4, 2, 4, 4), torch.int32, "random_50", torch.uint8),
    # Large batch, small width.
    ("fp32_n4096_k4_random_bool", (4096, 4), torch.float32, "random_50", torch.bool),
    # Width crosses the thirty two byte alignment boundary in a non trivial way.
    ("fp32_n16_k15_random_bool", (16, 15), torch.float32, "random_50", torch.bool),
    ("fp16_n16_k17_dense_bool", (16, 17), torch.float16, "dense_90", torch.bool),
    # All true with large width so the output passes through unchanged.
    ("fp32_n8_k512_alltrue_bool", (8, 512), torch.float32, "all_true", torch.bool),
    # Sparse on a larger batch with a low surviving row count.
    ("int32_n128_k32_sparse_bool", (128, 32), torch.int32, "sparse_10", torch.bool),
    # one dim data path along the first dimension only.
    ("fp32_n64_random_bool", (64,), torch.float32, "random_50", torch.bool),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("label_str,shape,dtype,mask_mode,mask_dtype", CASES_SMALL)
def test_c2_boolean_mask_small(label_str, shape, dtype, mask_mode, mask_dtype):
    seed_key = ("fast", label_str, tuple(shape), str(dtype), mask_mode, str(mask_dtype))
    data, mask = _gen_input(shape, dtype, mask_mode, mask_dtype, seed_key)
    _run(data, mask, label_str)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("label_str,shape,dtype,mask_mode,mask_dtype", CASES_LARGE)
def test_c2_boolean_mask_large(label_str, shape, dtype, mask_mode, mask_dtype):
    seed_key = ("heavy", label_str, tuple(shape), str(dtype), mask_mode, str(mask_dtype))
    data, mask = _gen_input(shape, dtype, mask_mode, mask_dtype, seed_key)
    _run(data, mask, label_str)


# ---------------------------------------------------------------------------
# Edge tests: empty and negative path validation.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_boolean_mask_empty_data():
    """Zero rows returns two empty tensors without launching the kernel."""
    data = torch.randn(0, 4).npu()
    mask = torch.zeros(0, dtype=torch.bool).npu()
    out_data, out_indices = torch.ops.ops_multimodal_fusion.c2_boolean_mask(data, mask)
    out_data = out_data.cpu()
    out_indices = out_indices.cpu()
    assert tuple(out_data.shape) == (0, 4)
    assert out_data.dtype == torch.float32
    assert tuple(out_indices.shape) == (0,)
    assert out_indices.dtype == torch.int64


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_boolean_mask_all_false():
    """Non-empty data with an all false mask yields a double empty output."""
    data = torch.randn(8, 16).npu()
    mask = torch.zeros(8, dtype=torch.bool).npu()
    out_data, out_indices = torch.ops.ops_multimodal_fusion.c2_boolean_mask(data, mask)
    out_data = out_data.cpu()
    out_indices = out_indices.cpu()
    assert tuple(out_data.shape) == (0, 16)
    assert out_data.dtype == torch.float32
    assert tuple(out_indices.shape) == (0,)
    assert out_indices.dtype == torch.int64


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_boolean_mask_all_true():
    """All true mask keeps every row, so indices count up and data is unchanged."""
    data_cpu = torch.randn(8, 16)
    mask = torch.ones(8, dtype=torch.bool)
    out_data, out_indices = torch.ops.ops_multimodal_fusion.c2_boolean_mask(
        data_cpu.npu(), mask.npu()
    )
    out_data = out_data.cpu()
    out_indices = out_indices.cpu()
    assert torch.equal(out_data, data_cpu)
    assert torch.equal(out_indices, torch.arange(8, dtype=torch.int64))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_boolean_mask_invalid_ndim():
    """data must have at least one dim."""
    data = torch.tensor(3.0).npu()
    mask = torch.tensor([True], dtype=torch.bool).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_boolean_mask(data, mask)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_boolean_mask_size_mismatch():
    """The mask length must equal the first dimension of data."""
    data = torch.randn(4, 8).npu()
    mask = torch.tensor([True, False, True], dtype=torch.bool).npu()  # wrong length
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_boolean_mask(data, mask)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_boolean_mask_mask_dim():
    """mask must be one dim."""
    data = torch.randn(4, 8).npu()
    mask = torch.tensor([[True, False], [True, True]], dtype=torch.bool).npu()  # two dim
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_boolean_mask(data, mask)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_boolean_mask_mask_dtype():
    """mask dtype must be bool or uint8, so int32, int64 and fp32 are rejected."""
    data = torch.randn(4, 8).npu()
    bad_mask_int32 = torch.tensor([1, 0, 1, 1], dtype=torch.int32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_boolean_mask(data, bad_mask_int32)
    bad_mask_fp32 = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_boolean_mask(data, bad_mask_fp32)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_boolean_mask_bf16_rejected():
    """bf16 data is deferred on this platform."""
    data = torch.randn(4, 8, dtype=torch.bfloat16).npu()
    mask = torch.tensor([True, False, True, False], dtype=torch.bool).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_boolean_mask(data, mask)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_boolean_mask_int64_rejected():
    """int64 and fp64 data are deferred on this platform."""
    data = torch.arange(32, dtype=torch.int64).reshape(4, 8).npu()
    mask = torch.tensor([True, False, True, False], dtype=torch.bool).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_boolean_mask(data, mask)
