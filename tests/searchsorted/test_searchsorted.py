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
"""Tests for ops_multimodal_fusion.searchsorted.

For each query value v, returns the smallest i such that sorted_sequence[i] >= v
(left-side insertion). Output is int64 with shape matching values. Reference is
torch.searchsorted on CPU; assertion is strict torch.equal (int64 indices).

First-version dispatch scope:
  - 1-D sorted_sequence only; n <= vec_width per dtype (fp32/int32=64, fp16=128)
  - dtype in {fp32, fp16, int32}; sorted_sequence and values must share dtype
  - right=False and out_int32=False only; right=True / out_int32=True deferred

Test case counts:
  - test_searchsorted_small        : 11 cases  (one (dtype) per shape, minimal fast-set coverage)
  - test_searchsorted_large        : 23 cases  (cartesian fallout + LARGE_ONLY_SHAPES)
  - test_searchsorted_interface_exist : 1
  - test_searchsorted_dtype_mismatch  : 1
  - test_searchsorted_sorted_2d_rejected : 1
  - test_searchsorted_right_true_rejected : 1
  - test_searchsorted_out_int32_rejected : 1
  - test_searchsorted_n_exceeds_vec_width : 1
  - Total                  : 40 cases
"""

import pytest
import torch
import torch_npu

import ops_multimodal_fusion

if not hasattr(torch.ops.ops_multimodal_fusion, "searchsorted"):
    pytest.skip(
        "ops_multimodal_fusion.searchsorted not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_searchsorted_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "searchsorted"), \
        "The 'searchsorted' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Pair generation: returns (sorted_sequence_cpu, values_cpu) with prescribed
# edge characteristics. All sorted sequences are 1-D and contiguous.
# ---------------------------------------------------------------------------

def _gen_sorted_sequence(n, mode, dtype):
    if mode == "mid_dup":
        # Explicit duplicates so left-side semantics is verifiable.
        base_dup = [1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15]
        if n <= len(base_dup):
            sorted_list = base_dup[:n]
        else:
            sorted_list = base_dup + list(range(17, 17 + n - len(base_dup)))
        return torch.tensor(sorted_list, dtype=dtype).contiguous()
    if dtype == torch.int32:
        # Unique integers in [0, 100), sorted.
        pool = torch.randperm(max(n * 2, 100))
        return pool[:n].sort()[0].to(torch.int32).contiguous()
    # fp32 / fp16: random normals sorted ascending.
    raw = torch.randn(n)
    return raw.sort()[0].to(dtype).contiguous()


def _gen_values(values_shape, mode, dtype, sorted_seq):
    if mode == "empty_values":
        return torch.empty(values_shape, dtype=dtype)
    if mode == "below_min":
        fill = float(sorted_seq[0]) - 100.0
        if dtype == torch.int32:
            return torch.full(values_shape, int(fill), dtype=torch.int32)
        return torch.full(values_shape, fill, dtype=dtype)
    if mode == "above_max":
        fill = float(sorted_seq[-1]) + 100.0
        if dtype == torch.int32:
            return torch.full(values_shape, int(fill), dtype=torch.int32)
        return torch.full(values_shape, fill, dtype=dtype)
    if mode == "mid_dup":
        # Query values land on duplicated elements of sorted to exercise left-side idx.
        query_pool = [1, 3, 5, 7, 9, 11, 13, 15, 3, 5, 9, 13, 1, 7, 11, 15]
        total = 1
        for d in values_shape:
            total *= d
        flat = torch.tensor(query_pool[:total] if total <= len(query_pool)
                            else query_pool * (total // len(query_pool) + 1),
                            dtype=torch.int32)[:total]
        return flat.reshape(values_shape).to(dtype)
    # "random"
    if dtype == torch.int32:
        lo = int(sorted_seq[0]) - 5
        hi = int(sorted_seq[-1]) + 5
        return torch.randint(lo, hi + 1, values_shape, dtype=torch.int32)
    return torch.randn(*values_shape).to(dtype) if len(values_shape) > 0 \
        else torch.randn(()).to(dtype)


def _gen_pair(n, values_shape, mode, dtype, seed_key):
    torch.manual_seed(abs(hash(seed_key)) % 10_000_000)
    sorted_seq = _gen_sorted_sequence(n, mode, dtype)
    values = _gen_values(values_shape, mode, dtype, sorted_seq)
    return sorted_seq, values


# ---------------------------------------------------------------------------
# Case matrix. Each row: (n, values_shape, gen_mode, allowed_dtypes, label).
# allowed_dtypes lists every dtype this (n, values_shape, gen_mode) is meaningful
# for; SMALL_SELECT picks one for the small set, the rest fall into the large set.
# ---------------------------------------------------------------------------

ALL_DT = [torch.float32, torch.float16, torch.int32]

SMALL_FAST_SHAPES = [
    # columns: n, values_shape, gen_mode, allowed_dtypes, label
    (64, (16,), "random", ALL_DT, "vec_eq"),
    (63, (16,), "random", ALL_DT, "vec_minus1"),
    (128, (16,), "random", [torch.float16], "vec_eq_fp16"),
    (8, (32,), "random", ALL_DT, "small_N"),
    (16, (8,), "below_min", ALL_DT, "below_min"),
    (16, (8,), "above_max", ALL_DT, "above_max"),
    (16, (8,), "mid_dup", [torch.int32], "mid_dup"),
    (16, (2, 4), "random", ALL_DT, "nd_values"),
    (16, (0,), "empty_values", ALL_DT, "empty_values"),
    (16, (1,), "random", ALL_DT, "single_query"),
    (1, (8,), "random", ALL_DT, "N1"),
]

LARGE_ONLY_SHAPES = [
    (64, (4096,), "random", ALL_DT, "large_M"),
    (64, (1024, 2), "random", [torch.float32, torch.float16], "nd_large"),
]

# One (dtype) per shape for small. Distribution: fp32×5 / fp16×3 / int32×3.
SMALL_SELECT = {
    "vec_eq": torch.float32,
    "vec_minus1": torch.float32,
    "vec_eq_fp16": torch.float16,
    "small_N": torch.int32,
    "below_min": torch.float16,
    "above_max": torch.float32,
    "mid_dup": torch.int32,
    "nd_values": torch.float32,
    "empty_values": torch.float16,
    "single_query": torch.int32,
    "N1": torch.float32,
}

CASES_SMALL = [
    (n, vs, mode, SMALL_SELECT[label], label)
    for (n, vs, mode, allowed, label) in SMALL_FAST_SHAPES
]


def _build_cases_large():
    cases = []
    for (n, vs, mode, allowed, label) in SMALL_FAST_SHAPES:
        for dt in allowed:
            if dt != SMALL_SELECT[label]:
                cases.append((n, vs, mode, dt, label))
    for (n, vs, mode, allowed, label) in LARGE_ONLY_SHAPES:
        for dt in allowed:
            cases.append((n, vs, mode, dt, label))
    return cases


CASES_LARGE = _build_cases_large()


def _run_case(n, values_shape, mode, dtype, label):
    seed_key = ("ss", label, n, tuple(values_shape), mode, str(dtype))
    sorted_cpu, values_cpu = _gen_pair(n, values_shape, mode, dtype, seed_key)
    expected = torch.searchsorted(sorted_cpu, values_cpu)
    result_npu = torch.ops.ops_multimodal_fusion.searchsorted(
        sorted_cpu.npu(), values_cpu.npu()
    )
    result = result_npu.cpu()
    assert result.dtype == torch.int64, \
        f"indices dtype mismatch: got {result.dtype}, want int64"
    assert result.shape == expected.shape, (
        f"shape mismatch: got {tuple(result.shape)}, want {tuple(expected.shape)} "
        f"(label={label}, dtype={dtype})"
    )
    assert torch.equal(result, expected), (
        f"indices mismatch (label={label}, dtype={dtype}, n={n}, "
        f"values_shape={tuple(values_shape)}, mode={mode})\n"
        f"  expected: {expected.flatten().tolist()[:32]}\n"
        f"  got     : {result.flatten().tolist()[:32]}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("n,values_shape,mode,dtype,label", CASES_SMALL)
def test_searchsorted_small(n, values_shape, mode, dtype, label):
    _run_case(n, values_shape, mode, dtype, label)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("n,values_shape,mode,dtype,label", CASES_LARGE)
def test_searchsorted_large(n, values_shape, mode, dtype, label):
    _run_case(n, values_shape, mode, dtype, label)


# Negative-path tests: argument validation.

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_searchsorted_dtype_mismatch():
    sorted_seq = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32).npu()
    values = torch.tensor([1.5, 2.5], dtype=torch.float16).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.searchsorted(sorted_seq, values)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_searchsorted_sorted_2d_rejected():
    sorted_seq = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32).npu()
    values = torch.tensor([1.5, 2.5], dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.searchsorted(sorted_seq, values)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_searchsorted_right_true_rejected():
    sorted_seq = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32).npu()
    values = torch.tensor([1.5, 2.5], dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.searchsorted(sorted_seq, values, False, True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_searchsorted_out_int32_rejected():
    sorted_seq = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32).npu()
    values = torch.tensor([1.5, 2.5], dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.searchsorted(sorted_seq, values, True, False)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_searchsorted_n_exceeds_vec_width():
    # fp32 vec_width = 64; n=65 must be rejected at host.
    sorted_seq = torch.arange(65, dtype=torch.float32).npu()
    values = torch.tensor([10.5, 30.5], dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.searchsorted(sorted_seq, values)
