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
"""Tests for ops_multimodal_fusion.index_copy.

Functional schema: index_copy takes self, an integer dim, an index tensor and a
source tensor, and returns a tensor.

For each position j in the source, the result at the index for j takes the
source value at j. Positions not named in index keep self's original value. The
reference clones self and applies the in-place index_copy on CPU.

First-version dispatch scope:
  - one-dimensional self, source and index only
  - dim restricted to zero or minus one, normalized to zero by the host
  - the self length along axis zero ranges from one to INT32_MAX and stays
    resident within UB
  - the source length equals the index length and ranges from zero to INT32_MAX
  - self and source share a dtype among fp32, fp16 and int32
  - the index dtype is int64

Tolerances:
  - All dtypes use a strict equality check, since copy and scatter are bit-exact
    with no floating-point arithmetic.

Test case counts:
  - test_index_copy_small                          :   9 cases, one dtype per shape
  - test_index_copy_large                          :  27 cases, cartesian residual plus large-only shapes
  - test_index_copy_interface_exist                :   1
  - test_index_copy_self_2d_rejected               :   1
  - test_index_copy_source_2d_rejected             :   1
  - test_index_copy_index_2d_rejected              :   1
  - test_index_copy_dim_nonzero_rejected           :   1
  - test_index_copy_dtype_mismatch                 :   1
  - test_index_copy_index_dtype_int32_rejected     :   1
  - test_index_copy_index_out_of_bounds_rejected   :   1
  - test_index_copy_size_mismatch_rejected         :   1
  - Total                                          :  45 cases
"""

from collections import namedtuple

import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "index_copy"):
    pytest.skip(
        "ops_multimodal_fusion.index_copy not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


# A single index_copy test case carrying the self size, the source and index
# size, the index-generation mode, the dim, the dtype and a human-readable label.
Case = namedtuple("Case", ["num_self", "num_idx", "mode", "dim", "dtype", "label"])


def test_index_copy_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "index_copy"), \
        "The 'index_copy' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Pair generation. Returns the self, index and source tensors, all 1-D and contiguous.
# ---------------------------------------------------------------------------

def _gen_index(num_self, num_idx, mode):
    """Build the int64 index tensor for the mode, with values from zero up to but below the self size."""
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
    return torch.randint(0, num_self, (num_idx,), dtype=torch.int64)  # "random"


def _gen_pair(num_self, num_idx, mode, dtype, seed_key):
    torch.manual_seed(abs(hash(seed_key)) % 10_000_000)

    # Empty index short-circuit: empty index/source, any self.
    if num_idx == 0 or mode == "empty":
        index = torch.empty((0,), dtype=torch.int64)
        if dtype == torch.int32:
            source = torch.empty((0,), dtype=torch.int32)
            self_raw = torch.randint(-100, 100, (num_self,), dtype=torch.int32)
        else:
            source = torch.empty((0,), dtype=dtype)
            self_raw = torch.randn(num_self).to(dtype)
        return self_raw.contiguous(), index.contiguous(), source.contiguous()

    index = _gen_index(num_self, num_idx, mode)

    # ---- self / source values ----
    if dtype == torch.int32:
        self_raw = torch.randint(-1_000_000, 1_000_000, (num_self,), dtype=torch.int32)
        source_raw = torch.randint(-1_000_000, 1_000_000, (num_idx,), dtype=torch.int32)
    else:
        self_raw = torch.randn(num_self).to(dtype)
        source_raw = torch.randn(num_idx).to(dtype)
    return self_raw.contiguous(), index.contiguous(), source_raw.contiguous()


# ---------------------------------------------------------------------------
# Case matrix. Each row lists in order the self size, the source and index size,
# the mode, the dim, the allowed dtypes and the label. The small-select mapping
# picks one dtype per shape; the rest fall into the large case list.
# ---------------------------------------------------------------------------

ALL_DT = [torch.float32, torch.float16, torch.int32]

SMALL_FAST_SHAPES = [
    # fields in order: num_self, num_idx, mode, dim, allowed_dtypes, label
    (1, 1, "spread", 0, ALL_DT, "k1_minimal"),
    (1, 0, "empty", 0, ALL_DT, "m0_noop"),
    (63, 32, "random", 0, ALL_DT, "vec_minus1"),
    (64, 64, "spread", 0, ALL_DT, "vec_eq_full"),
    (65, 1, "m1", -1, ALL_DT, "vec_plus1_neg_dim"),
    (128, 64, "random", 0, ALL_DT, "fp16_vec_eq"),
    (129, 200, "dup_index", 0, ALL_DT, "chunked_dups"),
    (256, 256, "all_same_tgt", 0, ALL_DT, "all_collide"),
    (1024, 1024, "spread", -1, ALL_DT, "k1024_neg_dim"),
]

LARGE_ONLY_SHAPES = [
    # stress the self size, stress a source far larger than self, and stress duplicates
    (4096, 64, "random", 0, [torch.float32, torch.int32], "huge_k_4096"),
    (8192, 32, "spread", 0, [torch.float16], "huge_k_8192"),
    (256, 512, "random", 0, ALL_DT, "m_gt_k"),
    (200, 1024, "dup_index", 0, ALL_DT, "m_5x_k_dups"),
]

# One dtype per small shape. Distribution target: fp32 ~3, fp16 ~3, int32 ~3.
SMALL_SELECT = {
    "k1_minimal": torch.float32,
    "m0_noop": torch.float16,
    "vec_minus1": torch.int32,
    "vec_eq_full": torch.float32,
    "vec_plus1_neg_dim": torch.float16,
    "fp16_vec_eq": torch.float16,
    "chunked_dups": torch.int32,
    "all_collide": torch.float32,
    "k1024_neg_dim": torch.float16,
}

CASES_SMALL = [
    Case(num_self, num_idx, mode, dim, SMALL_SELECT[label], label)
    for (num_self, num_idx, mode, dim, allowed_dt, label) in SMALL_FAST_SHAPES
]


def _build_cases_large():
    """Cartesian residual of SMALL_FAST_SHAPES plus all LARGE_ONLY_SHAPES."""
    cases = []
    for (num_self, num_idx, mode, dim, allowed_dt, label) in SMALL_FAST_SHAPES:
        for dt in allowed_dt:
            if dt != SMALL_SELECT[label]:
                cases.append(Case(num_self, num_idx, mode, dim, dt, label))
    for (num_self, num_idx, mode, dim, allowed_dt, label) in LARGE_ONLY_SHAPES:
        for dt in allowed_dt:
            cases.append(Case(num_self, num_idx, mode, dim, dt, label))
    return cases


CASES_LARGE = _build_cases_large()


def _run_case(case):
    seed_key = ("ic", case.label, case.num_self, case.num_idx, case.mode, case.dim, str(case.dtype))
    self_cpu, index_cpu, source_cpu = _gen_pair(
        case.num_self, case.num_idx, case.mode, case.dtype, seed_key)

    expected = self_cpu.clone().index_copy_(0, index_cpu, source_cpu)

    result_npu = torch.ops.ops_multimodal_fusion.index_copy(
        self_cpu.npu(), case.dim, index_cpu.npu(), source_cpu.npu())
    result = result_npu.cpu()

    assert result.dtype == expected.dtype, (
        f"dtype mismatch (label={case.label}): got {result.dtype}, want {expected.dtype}")
    assert result.shape == expected.shape, (
        f"shape mismatch (label={case.label}): "
        f"got {tuple(result.shape)}, want {tuple(expected.shape)}")
    assert torch.equal(result, expected), (
        f"result mismatch (label={case.label}, num_self={case.num_self}, num_idx={case.num_idx}, "
        f"mode={case.mode}, dim={case.dim}, dtype={case.dtype})\n"
        f"  index   : {index_cpu.tolist()[:16]}{'...' if case.num_idx > 16 else ''}\n"
        f"  source  : {source_cpu.tolist()[:16]}{'...' if case.num_idx > 16 else ''}\n"
        f"  self    : {self_cpu.tolist()[:16]}{'...' if case.num_self > 16 else ''}\n"
        f"  expected: {expected.tolist()[:16]}{'...' if case.num_self > 16 else ''}\n"
        f"  got     : {result.tolist()[:16]}{'...' if case.num_self > 16 else ''}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_SMALL, ids=[c.label for c in CASES_SMALL])
def test_index_copy_small(case):
    _run_case(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_LARGE, ids=[f"{c.label}_{c.dtype}" for c in CASES_LARGE])
def test_index_copy_large(case):
    _run_case(case)


# ---------------------------------------------------------------------------
# Negative-path tests: argument validation.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_index_copy_self_2d_rejected():
    self_t = torch.zeros((4, 4), dtype=torch.float32).npu()
    index_t = torch.tensor([0, 1, 2], dtype=torch.int64).npu()
    source_t = torch.ones((3,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.index_copy(self_t, 0, index_t, source_t)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_index_copy_source_2d_rejected():
    self_t = torch.zeros((4,), dtype=torch.float32).npu()
    index_t = torch.tensor([0, 1], dtype=torch.int64).npu()
    source_t = torch.ones((2, 2), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.index_copy(self_t, 0, index_t, source_t)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_index_copy_index_2d_rejected():
    self_t = torch.zeros((4,), dtype=torch.float32).npu()
    index_t = torch.tensor([[0], [1]], dtype=torch.int64).npu()
    source_t = torch.ones((2,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.index_copy(self_t, 0, index_t, source_t)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_index_copy_dim_nonzero_rejected():
    # For a one-dimensional tensor only dim zero or minus one is valid; dim one is out of range.
    self_t = torch.zeros((4,), dtype=torch.float32).npu()
    index_t = torch.tensor([0, 1], dtype=torch.int64).npu()
    source_t = torch.ones((2,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.index_copy(self_t, 1, index_t, source_t)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_index_copy_dtype_mismatch():
    self_t = torch.zeros((4,), dtype=torch.float32).npu()
    index_t = torch.tensor([0, 1], dtype=torch.int64).npu()
    source_t = torch.ones((2,), dtype=torch.float16).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.index_copy(self_t, 0, index_t, source_t)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_index_copy_index_dtype_int32_rejected():
    self_t = torch.zeros((4,), dtype=torch.float32).npu()
    index_t = torch.tensor([0, 1], dtype=torch.int32).npu()
    source_t = torch.ones((2,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.index_copy(self_t, 0, index_t, source_t)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_index_copy_index_out_of_bounds_rejected():
    # index value 5 exceeds the channel count of 4 so the host check rejects it
    self_t = torch.zeros((4,), dtype=torch.float32).npu()
    index_t = torch.tensor([5], dtype=torch.int64).npu()
    source_t = torch.ones((1,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.index_copy(self_t, 0, index_t, source_t)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_index_copy_size_mismatch_rejected():
    self_t = torch.zeros((8,), dtype=torch.float32).npu()
    index_t = torch.tensor([0, 1, 2], dtype=torch.int64).npu()
    source_t = torch.ones((2,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.index_copy(self_t, 0, index_t, source_t)
