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
"""Tests for ops_multimodal_fusion.index_reduce.

Functional schema:
  index_reduce(Tensor self, int dim, Tensor index, Tensor source, str reduce,
               bool include_self=True) -> Tensor

For each j in [0, m): result[index[j]] = reduce(result[index[j]], source[j])
reduce in {prod, amax, amin, mean}; reference is torch.Tensor.index_reduce
(out-of-place form) on CPU.

First-version dispatch scope:
  - 1-D self / source / index only, dim==0
  - k = self.size(0) in [1, INT32_MAX]; whole k stays resident in UB
  - m = source.size(0) in [1, vec_width(dtype)] (fp32/int32=64, fp16=128)
  - dtype(self)==dtype(source) in {fp32, fp16, int32}
  - index.dtype == int64
  - include_self True/False both supported

Tolerances (general-op tier):
  fp32 : rtol=1e-3, atol=1e-3
  fp16 : rtol=5e-3, atol=5e-3
  int32: strict torch.equal

Test case counts:
  - test_index_reduce_small                          :   9 cases  (one (dtype, reduce, include_self) per shape)
  - test_index_reduce_large                          : 231 cases  (cartesian residual + LARGE_ONLY_SHAPES)
  - test_index_reduce_interface_exist                :   1
  - test_index_reduce_self_2d_rejected               :   1
  - test_index_reduce_source_2d_rejected             :   1
  - test_index_reduce_index_2d_rejected              :   1
  - test_index_reduce_dim_nonzero_rejected           :   1
  - test_index_reduce_invalid_reduce_string_rejected :   1
  - test_index_reduce_m_exceeds_vec_width            :   1
  - test_index_reduce_dtype_mismatch                 :   1
  - test_index_reduce_index_dtype_int32_rejected     :   1
  - Total                                            : 249 cases
"""

from collections import namedtuple
from itertools import product

import pytest
import torch
import torch_npu

import ops_multimodal_fusion

if not hasattr(torch.ops.ops_multimodal_fusion, "index_reduce"):
    pytest.skip(
        "ops_multimodal_fusion.index_reduce not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_index_reduce_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "index_reduce"), \
        "The 'index_reduce' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# One test case's full parameter set, bundled so case-driven helpers take a
# single argument instead of a long positional list.
Case = namedtuple("Case", "k m mode dtype reduce include_self label")


def _case_id(case):
    dt = str(case.dtype).split(".")[-1]
    return f"{case.label}-{dt}-{case.reduce}-inc{int(case.include_self)}"


# ---------------------------------------------------------------------------
# Pair generation. Returns (self_cpu, index_cpu, source_cpu) with prescribed
# edge characteristics. All three tensors are 1-D and contiguous.
# ---------------------------------------------------------------------------

def _gen_pair(case):
    k, m, mode, dtype, reduce = case.k, case.m, case.mode, case.dtype, case.reduce
    seed_key = ("ir", case.label, k, m, mode, str(dtype), reduce, case.include_self)
    torch.manual_seed(abs(hash(seed_key)) % 10_000_000)

    # ---- index (always int64, values in [0, k)) ----
    if mode == "dup_index":
        # Indices have heavy duplicates landing on a few targets.
        pool = [0, 1, 0, 2, 1, 1, 3, 0, 2, 3, 1, 0, 2, 1, 0, 3]
        # Wrap modulo k to keep in range when k is small.
        idx_vals = [pool[i % len(pool)] % k for i in range(m)]
        index = torch.tensor(idx_vals, dtype=torch.int64)
    elif mode == "all_same_tgt":
        target = min(2, k - 1)
        index = torch.full((m,), target, dtype=torch.int64)
    elif mode == "k1":
        index = torch.zeros((m,), dtype=torch.int64)
    elif mode == "m1":
        index = torch.tensor([min(k - 1, k // 2)], dtype=torch.int64)
    elif mode == "spread":
        # Cover every position of k when m >= k.
        idx_vals = [i % k for i in range(m)]
        index = torch.tensor(idx_vals, dtype=torch.int64)
    else:  # "random"
        index = torch.randint(0, k, (m,), dtype=torch.int64)

    # ---- self / source values ----
    if dtype == torch.int32:
        # Keep magnitudes small so prod with up to vec_width entries stays in int32.
        self_raw = torch.randint(-5, 6, (k,), dtype=torch.int32)
        source_raw = torch.randint(-3, 4, (m,), dtype=torch.int32)
    else:
        # For prod, keep values close to 1 to avoid overflow/underflow over m up to vec_width.
        if reduce == "prod":
            self_raw = (torch.rand(k) * 0.4 + 0.8).to(dtype)   # in [0.8, 1.2]
            source_raw = (torch.rand(m) * 0.4 + 0.8).to(dtype)
            # Sprinkle a few negatives so sign mixing is exercised.
            sign_self = (torch.rand(k) < 0.25).to(self_raw.dtype) * -2 + 1
            sign_src = (torch.rand(m) < 0.25).to(source_raw.dtype) * -2 + 1
            self_raw = self_raw * sign_self
            source_raw = source_raw * sign_src
        else:
            self_raw = torch.randn(k).to(dtype)
            source_raw = torch.randn(m).to(dtype)
    return self_raw.contiguous(), index.contiguous(), source_raw.contiguous()


# ---------------------------------------------------------------------------
# Case matrix. Each row lists k, m, gen_mode, allowed_dtypes, allowed_reduces,
# allowed_include_self, label. SMALL_SELECT picks one dtype, reduce, inc_self
# per shape; the rest fall into CASES_LARGE.
# ---------------------------------------------------------------------------

ALL_DT = [torch.float32, torch.float16, torch.int32]
ALL_REDUCE = ["prod", "amax", "amin", "mean"]
ALL_INC = [True, False]

SMALL_FAST_SHAPES = [
    # columns: k, m, gen_mode, allowed_dtypes, allowed_reduces, allowed_inc, label
    (8, 4, "random", ALL_DT, ALL_REDUCE, ALL_INC, "small"),
    (16, 8, "random", ALL_DT, ALL_REDUCE, ALL_INC, "med"),
    (64, 64, "random", [torch.float32, torch.int32], ALL_REDUCE, ALL_INC, "vec_eq_64"),
    (128, 128, "random", [torch.float16], ALL_REDUCE, ALL_INC, "vec_eq_128"),
    (8, 6, "dup_index", ALL_DT, ALL_REDUCE, ALL_INC, "dup_index"),
    (8, 6, "all_same_tgt", ALL_DT, ALL_REDUCE, ALL_INC, "all_same_tgt"),
    (1, 4, "k1", ALL_DT, ALL_REDUCE, ALL_INC, "k1"),
    (8, 1, "m1", ALL_DT, ALL_REDUCE, ALL_INC, "m1"),
    (32, 32, "spread", ALL_DT, ALL_REDUCE, ALL_INC, "spread"),
]

LARGE_ONLY_SHAPES = [
    # k-stress shapes (only meaningful with m >= 1; m kept small to stay <= vec_width).
    (1024, 32, "random", [torch.float32, torch.float16, torch.int32], ALL_REDUCE, ALL_INC, "large_k_1024"),
    (4096, 32, "random", [torch.float32, torch.float16, torch.int32], ALL_REDUCE, ALL_INC, "large_k_4096"),
]

# One (dtype, reduce, include_self) per small shape. Distribution targets:
#   dtype: fp32 ~3, fp16 ~3, int32 ~3
#   reduce: prod ~2, amax ~2, amin ~2, mean ~3
#   include_self: True ~4, False ~5
# vec_eq_64 with fp32 and vec_eq_128 with fp16 both pick mean and inc_self False, so the
# mean-count book-keeping path is exercised at both vec_width boundaries in
# the small suite; the inc_self=True path is invariant to count fixup so it
# does not catch count book-keeping bugs.
SMALL_SELECT = {
    "small": (torch.float32, "prod", True),
    "med": (torch.float16, "amax", False),
    "vec_eq_64": (torch.float32, "mean", False),
    "vec_eq_128": (torch.float16, "mean", False),
    "dup_index": (torch.int32, "amin", False),
    "all_same_tgt": (torch.int32, "prod", True),
    "k1": (torch.float32, "mean", True),
    "m1": (torch.int32, "amax", False),
    "spread": (torch.float16, "amin", True),
}

CASES_SMALL = [
    Case(k, m, mode, *SMALL_SELECT[label], label)
    for (k, m, mode, allowed_dt, allowed_rd, allowed_inc, label) in SMALL_FAST_SHAPES
]


def _build_cases_large():
    cases = []
    for (k, m, mode, allowed_dt, allowed_rd, allowed_inc, label) in SMALL_FAST_SHAPES:
        for dt, rd, inc in product(allowed_dt, allowed_rd, allowed_inc):
            if (dt, rd, inc) != SMALL_SELECT[label]:
                cases.append(Case(k, m, mode, dt, rd, inc, label))
    for (k, m, mode, allowed_dt, allowed_rd, allowed_inc, label) in LARGE_ONLY_SHAPES:
        for dt, rd, inc in product(allowed_dt, allowed_rd, allowed_inc):
            cases.append(Case(k, m, mode, dt, rd, inc, label))
    return cases


CASES_LARGE = _build_cases_large()


def _tol(dtype):
    if dtype == torch.float32:
        return 1e-3, 1e-3
    if dtype == torch.float16:
        return 5e-3, 5e-3
    return 0.0, 0.0


def _run_case(case):
    k, m, mode = case.k, case.m, case.mode
    dtype, reduce, include_self, label = case.dtype, case.reduce, case.include_self, case.label
    self_cpu, index_cpu, source_cpu = _gen_pair(case)

    expected = self_cpu.clone().index_reduce(
        0, index_cpu, source_cpu, reduce, include_self=include_self)

    result_npu = torch.ops.ops_multimodal_fusion.index_reduce(
        self_cpu.npu(), 0, index_cpu.npu(), source_cpu.npu(),
        reduce, include_self)
    result = result_npu.cpu()

    assert result.dtype == expected.dtype, (
        f"dtype mismatch (label={label}, reduce={reduce}, inc_self={include_self}): "
        f"got {result.dtype}, want {expected.dtype}")
    assert result.shape == expected.shape, (
        f"shape mismatch (label={label}, reduce={reduce}, inc_self={include_self}): "
        f"got {tuple(result.shape)}, want {tuple(expected.shape)}")

    if dtype == torch.int32:
        assert torch.equal(result, expected), (
            f"int32 result mismatch (label={label}, reduce={reduce}, "
            f"inc_self={include_self}, k={k}, m={m}, mode={mode})\n"
            f"  index   : {index_cpu.tolist()}\n"
            f"  source  : {source_cpu.tolist()}\n"
            f"  self    : {self_cpu.tolist()}\n"
            f"  expected: {expected.tolist()}\n"
            f"  got     : {result.tolist()}")
    else:
        rtol, atol = _tol(dtype)
        ok = torch.allclose(result, expected, rtol=rtol, atol=atol, equal_nan=True)
        assert ok, (
            f"fp result mismatch (label={label}, reduce={reduce}, "
            f"inc_self={include_self}, k={k}, m={m}, mode={mode}, "
            f"dtype={dtype}, rtol={rtol}, atol={atol})\n"
            f"  expected[:16]: {expected.flatten().tolist()[:16]}\n"
            f"  got[:16]     : {result.flatten().tolist()[:16]}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_SMALL, ids=_case_id)
def test_index_reduce_small(case):
    _run_case(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_LARGE, ids=_case_id)
def test_index_reduce_large(case):
    _run_case(case)


# ---------------------------------------------------------------------------
# Negative-path tests: argument validation.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_index_reduce_self_2d_rejected():
    self_t = torch.zeros((4, 4), dtype=torch.float32).npu()
    index_t = torch.tensor([0, 1, 2], dtype=torch.int64).npu()
    source_t = torch.ones((3,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.index_reduce(self_t, 0, index_t, source_t, "prod", True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_index_reduce_source_2d_rejected():
    self_t = torch.zeros((4,), dtype=torch.float32).npu()
    index_t = torch.tensor([0, 1], dtype=torch.int64).npu()
    source_t = torch.ones((2, 2), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.index_reduce(self_t, 0, index_t, source_t, "prod", True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_index_reduce_index_2d_rejected():
    self_t = torch.zeros((4,), dtype=torch.float32).npu()
    index_t = torch.tensor([[0], [1]], dtype=torch.int64).npu()
    source_t = torch.ones((2,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.index_reduce(self_t, 0, index_t, source_t, "prod", True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_index_reduce_dim_nonzero_rejected():
    self_t = torch.zeros((4,), dtype=torch.float32).npu()
    index_t = torch.tensor([0, 1], dtype=torch.int64).npu()
    source_t = torch.ones((2,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.index_reduce(self_t, 1, index_t, source_t, "prod", True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_index_reduce_invalid_reduce_string_rejected():
    self_t = torch.zeros((4,), dtype=torch.float32).npu()
    index_t = torch.tensor([0, 1], dtype=torch.int64).npu()
    source_t = torch.ones((2,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.index_reduce(self_t, 0, index_t, source_t, "sum", True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_index_reduce_m_exceeds_vec_width():
    # fp32 vec_width = 64; m=65 must be rejected at host.
    self_t = torch.zeros((128,), dtype=torch.float32).npu()
    index_t = torch.zeros((65,), dtype=torch.int64).npu()
    source_t = torch.ones((65,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.index_reduce(self_t, 0, index_t, source_t, "prod", True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_index_reduce_dtype_mismatch():
    self_t = torch.zeros((4,), dtype=torch.float32).npu()
    index_t = torch.tensor([0, 1], dtype=torch.int64).npu()
    source_t = torch.ones((2,), dtype=torch.float16).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.index_reduce(self_t, 0, index_t, source_t, "prod", True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_index_reduce_index_dtype_int32_rejected():
    self_t = torch.zeros((4,), dtype=torch.float32).npu()
    index_t = torch.tensor([0, 1], dtype=torch.int32).npu()
    source_t = torch.ones((2,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.index_reduce(self_t, 0, index_t, source_t, "prod", True)
