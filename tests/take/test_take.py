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
"""Tests for ops_multimodal_fusion.take.

Functional schema: take consumes a self tensor and an index tensor and returns
a tensor.

Self is treated as flat; index addresses the flattened self, whose length is
K, the number of elements of self. Each result element is the flattened self at
the corresponding index value. The result shape matches the index shape; in the
first version the index is 1-D so the result is 1-D of length M. Negative
indices wrap: an index in the half-open range from minus K to K maps into the
range from zero to K.

First-version dispatch scope:
  - self is N-D with N at least one; K, the number of elements of self, lies
    between one and INT32_MAX
  - index is 1-D; M, the number of index elements, lies between zero and
    INT32_MAX, and the result is 1-D of length M
  - self has a dtype drawn from fp32, fp16 and int32, and the result inherits
    the self dtype
  - index is int64, narrowed on the host to int32 before the kernel; this is
    lossless since values are checked into the range from minus K to K, a
    subset of the int32 range

Tolerances:
  - all dtypes use strict equality via torch.equal, since gather is a lossless
    value copy

Test case counts:
  - test_take_interface_exist                  :   1
  - test_take_small                            :   9 cases  one dtype per shape
  - test_take_large                            :  27 cases  cartesian residual plus large-only shapes
  - test_take_index_2d_rejected                :   1
  - test_take_index_dtype_int32_rejected       :   1
  - test_take_index_out_of_bounds_rejected     :   1  index above the flat size
  - test_take_index_too_negative_rejected      :   1  index below the negative flat size
  - test_take_self_scalar_rejected             :   1  zero-dimensional self
  - test_take_dtype_unsupported_rejected       :   1  fp64 self
  - test_take_fp16_pool_oversize_rejected      :   1  fp16 pool above 65535
  - Total                                      :  44 cases
"""

import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "take"):
    pytest.skip(
        "ops_multimodal_fusion.take not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_take_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "take"),\
        "The 'take' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Pair generation. Returns the self tensor and the index tensor; self has the
# requested shape, index is 1-D int64 and contiguous.
# ---------------------------------------------------------------------------


def _gen_signed_index(num_idx, num_self):
    """Half the values are negative below zero and half are non-negative below num_self, then shuffled."""
    half = num_idx // 2
    pos = (torch.randint(0, num_self, (num_idx - half,), dtype=torch.int64)
           if num_idx - half > 0 else torch.empty(0, dtype=torch.int64))
    neg = (torch.randint(-num_self, 0, (half,), dtype=torch.int64)
           if half > 0 else torch.empty(0, dtype=torch.int64))
    return torch.cat([neg, pos])[torch.randperm(num_idx)]


def _gen_pair(self_shape, num_idx, mode, dtype, seed_key):
    torch.manual_seed(abs(hash(seed_key)) % 10_000_000)
    num_self = 1
    for s in self_shape:
        num_self *= s

    if num_idx == 0 or mode == "empty":
        index = torch.empty((0,), dtype=torch.int64)
        if dtype == torch.int32:
            self_raw = torch.randint(-1_000_000, 1_000_000, self_shape, dtype=torch.int32)
        else:
            self_raw = torch.randn(self_shape).to(dtype)
        return self_raw.contiguous(), index.contiguous()

    # ---- int64 index spanning the negative through positive pool range ----
    if mode == "spread":
        idx_vals = [i % num_self for i in range(num_idx)]
        index = torch.tensor(idx_vals, dtype=torch.int64)
    elif mode == "m1":
        index = torch.tensor([min(num_self - 1, num_self // 2)], dtype=torch.int64)
    elif mode == "neg_index":
        index = _gen_signed_index(num_idx, num_self)
    else:  # the random mode, which also covers a duplicate gather when there are more indices than pool entries
        index = torch.randint(0, num_self, (num_idx,), dtype=torch.int64)

    # ---- self pool values ----
    if dtype == torch.int32:
        self_raw = torch.randint(-1_000_000, 1_000_000, self_shape, dtype=torch.int32)
    else:
        self_raw = torch.randn(self_shape).to(dtype)
    return self_raw.contiguous(), index.contiguous()


# ---------------------------------------------------------------------------
# Case matrix. Each row lists self shape, index count, mode, allowed dtypes and label.
# SMALL_SELECT picks one dtype per shape; the rest fall into CASES_LARGE.
# Labels and modes intentionally avoid the substrings "large" and "small" so
# pytest -k 'not large' does not silently deselect a small case.
# ---------------------------------------------------------------------------

ALL_DT = [torch.float32, torch.float16, torch.int32]

SMALL_FAST_SHAPES = [
    # fields in order: self shape, index count, mode, allowed dtypes, label
    ((1,), 1, "spread", ALL_DT, "k1_pool"),
    ((8,), 0, "empty", ALL_DT, "m0_empty"),
    ((63,), 32, "random", ALL_DT, "vec_minus1"),
    ((64,), 64, "spread", ALL_DT, "vec_eq"),
    ((65,), 130, "random", ALL_DT, "m_gt_k_dup"),
    ((2, 8), 16, "neg_index", ALL_DT, "2d_neg"),
    ((8, 16), 64, "random", ALL_DT, "2d_random"),
    ((4, 4, 4), 1, "m1", ALL_DT, "3d_m1"),
    ((32, 32), 256, "neg_index", ALL_DT, "big_2d_neg"),
]

LARGE_ONLY_SHAPES = [
    ((4096,), 64, "random", ALL_DT, "huge_pool_4096"),
    ((16, 256), 512, "random", ALL_DT, "big_2d_k4096"),
    ((8, 8, 16), 2048, "neg_index", ALL_DT, "m_2x_k_neg"),
]

# One dtype per small shape. Distribution target: fp32 ~3, fp16 ~3, int32 ~3.
SMALL_SELECT = {
    "k1_pool": torch.float32,
    "m0_empty": torch.float16,
    "vec_minus1": torch.int32,
    "vec_eq": torch.float32,
    "m_gt_k_dup": torch.float16,
    "2d_neg": torch.float16,
    "2d_random": torch.int32,
    "3d_m1": torch.float32,
    "big_2d_neg": torch.float16,
}

CASES_SMALL = [
    (shape, num_idx, mode, SMALL_SELECT[label], label)
    for (shape, num_idx, mode, allowed_dt, label) in SMALL_FAST_SHAPES
]


def _build_cases_large():
    """Cartesian residual of SMALL_FAST_SHAPES plus all LARGE_ONLY_SHAPES."""
    cases = []
    for (shape, num_idx, mode, allowed_dt, label) in SMALL_FAST_SHAPES:
        for dt in allowed_dt:
            if dt != SMALL_SELECT[label]:
                cases.append((shape, num_idx, mode, dt, label))
    for (shape, num_idx, mode, allowed_dt, label) in LARGE_ONLY_SHAPES:
        for dt in allowed_dt:
            cases.append((shape, num_idx, mode, dt, label))
    return cases


CASES_LARGE = _build_cases_large()


def _run_case(self_shape, num_idx, mode, dtype, label):
    seed_key = ("take", label, self_shape, num_idx, mode, str(dtype))
    self_cpu, index_cpu = _gen_pair(self_shape, num_idx, mode, dtype, seed_key)

    expected = torch.take(self_cpu, index_cpu)

    result_npu = torch.ops.ops_multimodal_fusion.take(self_cpu.npu(), index_cpu.npu())
    result = result_npu.cpu()

    assert result.dtype == expected.dtype, (
        f"dtype mismatch (label={label}): got {result.dtype}, want {expected.dtype}")
    assert result.shape == expected.shape, (
        f"shape mismatch (label={label}): "
        f"got {tuple(result.shape)}, want {tuple(expected.shape)}")
    assert torch.equal(result, expected), (
        f"result mismatch (label={label}, num_self={self_cpu.numel()}, num_idx={num_idx}, "
        f"mode={mode}, dtype={dtype})\n"
        f"  index   : {index_cpu.tolist()[:16]}{'...' if num_idx > 16 else ''}\n"
        f"  self.view(-1)[:16]: {self_cpu.flatten().tolist()[:16]}\n"
        f"  expected[:16]: {expected.tolist()[:16]}\n"
        f"  got[:16]     : {result.tolist()[:16]}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("self_shape,num_idx,mode,dtype,label", CASES_SMALL)
def test_take_small(self_shape, num_idx, mode, dtype, label):
    _run_case(self_shape, num_idx, mode, dtype, label)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("self_shape,num_idx,mode,dtype,label", CASES_LARGE)
def test_take_large(self_shape, num_idx, mode, dtype, label):
    _run_case(self_shape, num_idx, mode, dtype, label)


# ---------------------------------------------------------------------------
# Negative-path tests: argument validation.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_take_index_2d_rejected():
    self_t = torch.arange(8, dtype=torch.float32).npu()
    index_t = torch.tensor([[0], [1]], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.take(self_t, index_t)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_take_index_dtype_int32_rejected():
    self_t = torch.arange(8, dtype=torch.float32).npu()
    index_t = torch.tensor([0, 1], dtype=torch.int32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.take(self_t, index_t)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_take_index_out_of_bounds_rejected():
    # with a channel count of four, an index value of five exceeds it so the host check rejects it.
    self_t = torch.arange(4, dtype=torch.float32).npu()
    index_t = torch.tensor([5], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.take(self_t, index_t)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_take_index_too_negative_rejected():
    # with a channel count of four the valid range is minus four to four, so index minus five is rejected.
    self_t = torch.arange(4, dtype=torch.float32).npu()
    index_t = torch.tensor([-5], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.take(self_t, index_t)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_take_self_scalar_rejected():
    # a zero-dimensional self should be rejected since at least one dimension is required.
    self_t = torch.tensor(3.0, dtype=torch.float32).npu()
    index_t = torch.tensor([0], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.take(self_t, index_t)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_take_dtype_unsupported_rejected():
    # fp64 self lies outside the supported set of fp32, fp16 and int32.
    self_t = torch.arange(8, dtype=torch.float64).npu()
    index_t = torch.tensor([0, 1], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.take(self_t, index_t)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_take_fp16_pool_oversize_rejected():
    # fp16 gather pool is capped at 65535 since dav-3510 b16 gather uses an unsigned 16-bit index.
    self_t = torch.zeros((65536,), dtype=torch.float16).npu()
    index_t = torch.tensor([0, 1], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.take(self_t, index_t)
