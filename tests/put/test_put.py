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
"""Tests for ops_multimodal_fusion.put.

Functional schema: put takes self, index and source tensors plus an accumulate
flag defaulting to false and returns a tensor.

Self is treated as flat; index addresses the flattened self, whose length is
K, the number of elements of self. When accumulate is false each source element
overwrites the flattened result at the matching index. When accumulate is true
each source element is added to the flattened result at the matching index.
Negative indices wrap: an index in the half-open range from minus K to K maps
into the range from zero to K.

First-version dispatch scope:
  - self is N-D with N at least one; K, the number of elements of self, lies
    between one and INT32_MAX
  - index and source are 1-D; M, the number of index elements, equals the
    number of source elements and lies between zero and INT32_MAX
  - self and source share a dtype drawn from fp32, fp16 and int32
  - the index uses int64
  - accumulate may be either true or false

Tolerances:
  - accumulate false: strict equality via torch.equal, since copy plus scatter
    is bit-exact for all dtypes
  - accumulate true: fp32 with rtol of 1e-3 and atol of 1e-3; fp16 with rtol of
    5e-3 and atol of 5e-3; int32 strict equality via torch.equal

Test case counts:
  - test_put_interface_exist                  :   1
  - test_put_small                            :   9 cases  one dtype and accumulate pairing per shape
  - test_put_large                            :  63 cases  cartesian residual plus large-only shapes
  - test_put_source_2d_rejected               :   1
  - test_put_index_2d_rejected                :   1
  - test_put_dtype_mismatch                   :   1
  - test_put_index_dtype_int32_rejected       :   1
  - test_put_index_out_of_bounds_rejected     :   1  index above the flat size
  - test_put_index_too_negative_rejected      :   1  index below the negative flat size
  - test_put_size_mismatch_rejected           :   1
  - test_put_self_scalar_rejected             :   1  zero-dimensional self
  - Total                                     :  81 cases
"""

import itertools
from collections import namedtuple

import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "put"):
    pytest.skip(
        "ops_multimodal_fusion.put not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


# A single test case carrying self shape, scatter count, index mode, dtype,
# accumulate flag and a human-readable label. Grouped into one record so the
# generator and runner stay within the per-function argument budget.
Case = namedtuple("Case", "self_shape num_idx mode dtype accumulate label")


def test_put_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "put"),\
        "The 'put' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Pair generation. Returns the self tensor, the index tensor and the source tensor; self has the
# requested shape, index/source are 1-D and contiguous.
# ---------------------------------------------------------------------------


def _randint_signed(count, num_self):
    """Half the values are negative below zero and half are non-negative below num_self, then shuffled."""
    half = count // 2
    pos = (torch.randint(0, num_self, (count - half,), dtype=torch.int64)
           if count - half > 0 else torch.empty(0, dtype=torch.int64))
    neg = (torch.randint(-num_self, 0, (half,), dtype=torch.int64)
           if half > 0 else torch.empty(0, dtype=torch.int64))
    index = torch.cat([neg, pos])
    return index[torch.randperm(count)]


def _gen_index(num_self, num_idx, mode):
    """Build the int64 index tensor for the requested mode."""
    if mode == "dup_index":
        pool = [0, 1, 0, 2, 1, 1, 3, 0, 2, 3, 1, 0, 2, 1, 0, 3]
        idx_vals = [pool[i % len(pool)] % num_self for i in range(num_idx)]
        return torch.tensor(idx_vals, dtype=torch.int64)
    if mode == "all_same_tgt":
        target = min(2, num_self - 1)
        return torch.full((num_idx,), target, dtype=torch.int64)
    if mode == "k1":
        return torch.zeros((num_idx,), dtype=torch.int64)
    if mode == "m1":
        return torch.tensor([min(num_self - 1, num_self // 2)], dtype=torch.int64)
    if mode == "spread":
        idx_vals = [i % num_self for i in range(num_idx)]
        return torch.tensor(idx_vals, dtype=torch.int64)
    if mode == "neg_index":
        return _randint_signed(num_idx, num_self)
    return torch.randint(0, num_self, (num_idx,), dtype=torch.int64)  # "random"


def _gen_values(self_shape, num_idx, dtype, accumulate):
    """Build the self and source value tensors for the requested dtype."""
    if dtype != torch.int32:
        return torch.randn(self_shape).to(dtype), torch.randn(num_idx).to(dtype)
    if accumulate:
        # Small magnitudes so sums over many duplicates stay in int32.
        self_raw = torch.randint(-5, 6, self_shape, dtype=torch.int32)
        source_raw = torch.randint(-3, 4, (num_idx,), dtype=torch.int32)
        return self_raw, source_raw
    self_raw = torch.randint(-1_000_000, 1_000_000, self_shape, dtype=torch.int32)
    source_raw = torch.randint(-1_000_000, 1_000_000, (num_idx,), dtype=torch.int32)
    return self_raw, source_raw


def _gen_pair(case, seed_key):
    self_shape = case.self_shape
    num_idx = case.num_idx
    mode = case.mode
    dtype = case.dtype

    torch.manual_seed(abs(hash(seed_key)) % 10_000_000)
    num_self = 1
    for s in self_shape:
        num_self *= s

    # Empty index short-circuit.
    if num_idx == 0 or mode == "empty":
        index = torch.empty((0,), dtype=torch.int64)
        if dtype == torch.int32:
            source = torch.empty((0,), dtype=torch.int32)
            self_raw = torch.randint(-100, 100, self_shape, dtype=torch.int32)
        else:
            source = torch.empty((0,), dtype=dtype)
            self_raw = torch.randn(self_shape).to(dtype)
        return self_raw.contiguous(), index.contiguous(), source.contiguous()

    index = _gen_index(num_self, num_idx, mode)
    self_raw, source_raw = _gen_values(self_shape, num_idx, dtype, case.accumulate)
    return self_raw.contiguous(), index.contiguous(), source_raw.contiguous()


# ---------------------------------------------------------------------------
# Case matrix. Each row lists self shape, index count, mode, allowed dtypes, allowed accumulate flags and label.
# SMALL_SELECT picks one dtype and accumulate pairing per shape; the rest fall into
# CASES_LARGE. Labels and modes intentionally avoid the substrings "large" and "small"
# so pytest -k 'not large' does not silently deselect a small case.
# ---------------------------------------------------------------------------

ALL_DT = [torch.float32, torch.float16, torch.int32]
ALL_ACC = [False, True]

SMALL_FAST_SHAPES = [
    # fields in order: self shape, index count, mode, allowed dtypes, allowed accumulate flags, label
    ((1,), 1, "spread", ALL_DT, ALL_ACC, "k1_minimal"),
    ((1,), 0, "empty", ALL_DT, ALL_ACC, "m0_noop"),
    ((63,), 32, "random", ALL_DT, ALL_ACC, "vec_minus1_accum"),
    ((64,), 64, "spread", ALL_DT, ALL_ACC, "vec_eq_full"),
    ((65,), 1, "m1", ALL_DT, ALL_ACC, "vec_plus1"),
    ((2, 8), 8, "random", ALL_DT, ALL_ACC, "2d_accum"),
    ((8, 16), 64, "dup_index", ALL_DT, ALL_ACC, "2d_dups_accum"),
    ((4, 4, 4), 32, "neg_index", ALL_DT, ALL_ACC, "3d_neg_index"),
    ((32, 32), 512, "spread", ALL_DT, ALL_ACC, "big_2d"),
]

LARGE_ONLY_SHAPES = [
    # K-stress + N-D + M close to K
    ((4096,), 64, "random", ALL_DT, ALL_ACC, "huge_1d_4096"),
    ((16, 256), 512, "random", ALL_DT, ALL_ACC, "big_2d_k4096"),
    ((8, 8, 16), 128, "neg_index", ALL_DT, ALL_ACC, "3d_k1024_neg"),
]

# One dtype and accumulate pairing per small shape. Distribution target:
#   dtype: fp32 ~3, fp16 ~3, int32 ~3
#   accumulate: False ~5, True ~4
SMALL_SELECT = {
    "k1_minimal": (torch.float32, False),
    "m0_noop": (torch.float16, False),
    "vec_minus1_accum": (torch.int32, True),
    "vec_eq_full": (torch.float32, False),
    "vec_plus1": (torch.float16, False),
    "2d_accum": (torch.float16, True),
    "2d_dups_accum": (torch.int32, True),
    "3d_neg_index": (torch.float32, False),
    "big_2d": (torch.float16, False),
}

CASES_SMALL = [
    Case(shape, num_idx, mode, *SMALL_SELECT[label], label)
    for (shape, num_idx, mode, allowed_dt, allowed_acc, label) in SMALL_FAST_SHAPES
]


def _build_cases_large():
    """Cartesian residual of SMALL_FAST_SHAPES plus all LARGE_ONLY_SHAPES."""
    cases = []
    for (shape, num_idx, mode, allowed_dt, allowed_acc, label) in SMALL_FAST_SHAPES:
        for dt, acc in itertools.product(allowed_dt, allowed_acc):
            if (dt, acc) != SMALL_SELECT[label]:
                cases.append(Case(shape, num_idx, mode, dt, acc, label))
    for (shape, num_idx, mode, allowed_dt, allowed_acc, label) in LARGE_ONLY_SHAPES:
        for dt, acc in itertools.product(allowed_dt, allowed_acc):
            cases.append(Case(shape, num_idx, mode, dt, acc, label))
    return cases


CASES_LARGE = _build_cases_large()


def _tol_accum(dtype):
    if dtype == torch.float32:
        return 1e-3, 1e-3
    if dtype == torch.float16:
        return 5e-3, 5e-3
    return 0.0, 0.0


def _run_case(case):
    label = case.label
    num_idx = case.num_idx
    mode = case.mode
    dtype = case.dtype
    accumulate = case.accumulate
    seed_key = ("put", label, case.self_shape, num_idx, mode, str(dtype), accumulate)
    self_cpu, index_cpu, source_cpu = _gen_pair(case, seed_key)

    expected = self_cpu.clone()
    expected.put_(index_cpu, source_cpu, accumulate=accumulate)

    result_npu = torch.ops.ops_multimodal_fusion.put(
        self_cpu.npu(), index_cpu.npu(), source_cpu.npu(), accumulate)
    result = result_npu.cpu()

    assert result.dtype == expected.dtype, (
        f"dtype mismatch (label={label}, accumulate={accumulate}): "
        f"got {result.dtype}, want {expected.dtype}")
    assert result.shape == expected.shape, (
        f"shape mismatch (label={label}, accumulate={accumulate}): "
        f"got {tuple(result.shape)}, want {tuple(expected.shape)}")

    if accumulate and dtype in (torch.float32, torch.float16):
        rtol, atol = _tol_accum(dtype)
        ok = torch.allclose(result, expected, rtol=rtol, atol=atol, equal_nan=True)
        assert ok, (
            f"fp accumulate mismatch (label={label}, num_self={self_cpu.numel()}, num_idx={num_idx}, "
            f"mode={mode}, dtype={dtype}, rtol={rtol}, atol={atol})\n"
            f"  expected.view(-1)[:16]: {expected.flatten().tolist()[:16]}\n"
            f"  got.view(-1)[:16]     : {result.flatten().tolist()[:16]}")
    else:
        assert torch.equal(result, expected), (
            f"strict mismatch (label={label}, num_self={self_cpu.numel()}, num_idx={num_idx}, "
            f"mode={mode}, dtype={dtype}, accumulate={accumulate})\n"
            f"  index   : {index_cpu.tolist()[:16]}{'...' if num_idx > 16 else ''}\n"
            f"  source  : {source_cpu.tolist()[:16]}{'...' if num_idx > 16 else ''}\n"
            f"  self.view(-1)[:16]: {self_cpu.flatten().tolist()[:16]}\n"
            f"  expected.view(-1)[:16]: {expected.flatten().tolist()[:16]}\n"
            f"  got.view(-1)[:16]     : {result.flatten().tolist()[:16]}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_SMALL, ids=lambda c: c.label)
def test_put_small(case):
    _run_case(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_LARGE, ids=lambda c: c.label)
def test_put_large(case):
    _run_case(case)


# ---------------------------------------------------------------------------
# Negative-path tests: argument validation.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_put_source_2d_rejected():
    self_t = torch.zeros((4,), dtype=torch.float32).npu()
    index_t = torch.tensor([0, 1], dtype=torch.int64).npu()
    source_t = torch.ones((2, 2), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.put(self_t, index_t, source_t, False)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_put_index_2d_rejected():
    self_t = torch.zeros((4,), dtype=torch.float32).npu()
    index_t = torch.tensor([[0], [1]], dtype=torch.int64).npu()
    source_t = torch.ones((2,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.put(self_t, index_t, source_t, False)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_put_dtype_mismatch():
    self_t = torch.zeros((4,), dtype=torch.float32).npu()
    index_t = torch.tensor([0, 1], dtype=torch.int64).npu()
    source_t = torch.ones((2,), dtype=torch.float16).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.put(self_t, index_t, source_t, False)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_put_index_dtype_int32_rejected():
    self_t = torch.zeros((4,), dtype=torch.float32).npu()
    index_t = torch.tensor([0, 1], dtype=torch.int32).npu()
    source_t = torch.ones((2,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.put(self_t, index_t, source_t, False)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_put_index_out_of_bounds_rejected():
    # index value 5 exceeds the flat size of 4 so the host check rejects it
    self_t = torch.zeros((4,), dtype=torch.float32).npu()
    index_t = torch.tensor([5], dtype=torch.int64).npu()
    source_t = torch.ones((1,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.put(self_t, index_t, source_t, False)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_put_index_too_negative_rejected():
    # with a flat size of 4 the valid range spans negative four up to four; an index of negative five is rejected
    self_t = torch.zeros((4,), dtype=torch.float32).npu()
    index_t = torch.tensor([-5], dtype=torch.int64).npu()
    source_t = torch.ones((1,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.put(self_t, index_t, source_t, False)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_put_size_mismatch_rejected():
    self_t = torch.zeros((8,), dtype=torch.float32).npu()
    index_t = torch.tensor([0, 1, 2], dtype=torch.int64).npu()
    source_t = torch.ones((2,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.put(self_t, index_t, source_t, False)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_put_self_scalar_rejected():
    # a zero-dimensional self should be rejected since at least one dimension is required
    self_t = torch.tensor(0.0, dtype=torch.float32).npu()
    index_t = torch.tensor([0], dtype=torch.int64).npu()
    source_t = torch.ones((1,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.put(self_t, index_t, source_t, False)
