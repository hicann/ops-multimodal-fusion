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

"""Tests for ops_multimodal_fusion.nested_binary_op.

Functional schema:
  nested_binary_op(Tensor values, Tensor offsets, Tensor dense, int op_mode) -> Tensor

For each batch index, the rows spanned by consecutive offsets are combined with
that batch's single dense row by elementwise add when op_mode is zero, otherwise
by elementwise multiply. Equivalent to the PyTorch CUDA op_dense_esuhm path for a
nested values tensor and a per-batch broadcast dense tensor.

First-version dispatch scope:
  - values: two dimensional, sum of lengths by dim
  - offsets: one dimensional, length batch plus one, int64, row units, starting
             at zero, non decreasing, ending at the row total
  - dense: three dimensional, batch by one by dim
  - dtype of values equals dtype of dense, one of fp32, fp16 or int32
  - op_mode is zero or one

Tolerances (simple elementwise per dtype):
  - fp32: rtol 1e-4, atol 1e-4
  - fp16: rtol 1e-3, atol 1e-3
  - int32: exact comparison via torch.equal, no fp rounding

Test case counts:
  - test_nested_binary_op_small                  :  14 cases
  - test_nested_binary_op_large                  :  36 cases
  - test_nested_binary_op_interface_exist        :   1
  - test_nested_binary_op_values_1d_rejected     :   1
  - test_nested_binary_op_offsets_2d_rejected    :   1
  - test_nested_binary_op_dense_2d_rejected      :   1
  - test_nested_binary_op_dense_size1_not_one    :   1
  - test_nested_binary_op_dim_mismatch           :   1
  - test_nested_binary_op_offsets_size_mismatch  :   1
  - test_nested_binary_op_offsets_int32_rejected :   1
  - test_nested_binary_op_offsets_nonzero_start  :   1
  - test_nested_binary_op_offsets_not_monotonic  :   1
  - test_nested_binary_op_offsets_wrong_total    :   1
  - test_nested_binary_op_op_mode_invalid        :   1
  - test_nested_binary_op_dtype_mismatch         :   1
  - test_nested_binary_op_bf16_rejected          :   1
  - Total                                        :  64 cases
"""

from collections import namedtuple
import random

import pytest
import torch
import torch_npu

import ops_multimodal_fusion

if not hasattr(torch.ops.ops_multimodal_fusion, "nested_binary_op"):
    pytest.skip(
        "ops_multimodal_fusion.nested_binary_op not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_nested_binary_op_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "nested_binary_op"),\
        "The 'nested_binary_op' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# Tolerance table; int32 uses exact comparison.
_TOL = {
    torch.float32: dict(rtol=1e-4, atol=1e-4),
    torch.float16: dict(rtol=1e-3, atol=1e-3),
    torch.int32: None,
}

# One case row carries everything a single run needs.
Case = namedtuple("Case", "op_mode b d l_pattern dtype label")


# Length-pattern semantics, given a batch count and a pattern string:
#   uniform_N   every length equals N
#   ragged_a-b  per-batch length drawn from a through b, seeded
#   with_zero   alternating zero and k, at least one of each, where k is the
#               larger of two and half the batch count plus one
#   all_zero    every length is zero, so the row total is zero


def _make_lengths(b, l_pattern, rng):
    if l_pattern.startswith("uniform_"):
        n = int(l_pattern.split("_", 1)[1])
        return [n] * b
    if l_pattern == "all_zero":
        return [0] * b
    if l_pattern == "with_zero":
        k = max(2, b // 2 + 1)
        return [0 if (i % 2 == 0) else k for i in range(b)]
    if l_pattern.startswith("ragged_"):
        lo, hi = (int(x) for x in l_pattern.split("_", 1)[1].split("-"))
        return [int(rng.randint(lo, hi + 1)) for _ in range(b)]
    raise ValueError(f"unknown l_pattern: {l_pattern}")


# Returns contiguous cpu tensors values, offsets and dense.


def _gen_inputs(b, d, l_pattern, dtype, seed_key):
    rng = random.Random(abs(hash(seed_key)) % 10_000_000)
    torch.manual_seed(abs(hash(("torch", seed_key))) % 10_000_000)

    lengths = _make_lengths(b, l_pattern, rng)
    sum_l = sum(lengths)

    offsets = torch.zeros((b + 1,), dtype=torch.int64)
    acc = 0
    for i, length in enumerate(lengths):
        acc += length
        offsets[i + 1] = acc

    if dtype == torch.int32:
        values = torch.randint(-1_000, 1_000, (sum_l, d), dtype=torch.int32)
        dense = torch.randint(-1_000, 1_000, (b, 1, d), dtype=torch.int32)
    else:
        values = torch.randn((sum_l, d)).to(dtype)
        dense = torch.randn((b, 1, d)).to(dtype)

    return values.contiguous(), offsets.contiguous(), dense.contiguous()


def _ref_nested_binary_op(values, offsets, dense, op_mode):
    b = dense.shape[0]
    result = values.clone()
    for i in range(b):
        rs = int(offsets[i])
        re = int(offsets[i + 1])
        if re <= rs:
            continue
        if op_mode == 0:
            result[rs:re] = values[rs:re] + dense[i, 0]
        else:
            result[rs:re] = values[rs:re] * dense[i, 0]
    return result


# Case matrix. Each shape row holds op_mode, batch, dim, pattern, allowed
# dtypes and a label. SMALL_SELECT picks one dtype per shape; the rest of the
# allowed dtypes fall into the large set.
ALL_DT = [torch.float32, torch.float16, torch.int32]

SMALL_FAST_SHAPES = [
    (0, 1, 1, "uniform_2", ALL_DT, "b1_d1_min"),
    (0, 2, 64, "uniform_4", ALL_DT, "fp32_vec_eq"),
    (1, 2, 64, "uniform_4", ALL_DT, "fp32_vec_eq_mul"),
    (0, 3, 63, "ragged_1-4", ALL_DT, "vec_minus1"),
    (0, 4, 128, "uniform_2", ALL_DT, "fp16_vec_eq"),
    (1, 4, 128, "uniform_2", ALL_DT, "fp16_vec_eq_mul"),
    (0, 2, 65, "uniform_3", ALL_DT, "vec_plus1_chunk"),
    (1, 2, 127, "uniform_2", ALL_DT, "fp16_vec_minus1"),
    (0, 2, 129, "uniform_2", ALL_DT, "fp16_vec_plus1_chunk"),
    (0, 4, 256, "ragged_1-3", ALL_DT, "chunked_d_ragged"),
    (1, 1, 64, "all_zero", ALL_DT, "empty_all_zero"),
    (0, 3, 64, "with_zero", ALL_DT, "with_zero_batch"),
    (1, 8, 64, "uniform_2", ALL_DT, "b8_uniform_mul"),
    (0, 2, 32, "ragged_1-8", ALL_DT, "small_d32_ragged"),
]

# Larger batch, dim and lengths for hardware-edge coverage on real NPU.
LARGE_ONLY_SHAPES = [
    (0, 16, 256, "ragged_1-16", [torch.float32], "b16_d256_fp32_add"),
    (1, 16, 256, "ragged_1-16", [torch.float32], "b16_d256_fp32_mul"),
    (0, 8, 512, "ragged_1-32", [torch.float16], "b8_d512_fp16_add"),
    (1, 8, 512, "ragged_1-32", [torch.float16], "b8_d512_fp16_mul"),
    (0, 4, 1024, "ragged_1-64", [torch.float32], "b4_d1024_fp32_add"),
    (1, 4, 1024, "ragged_1-64", [torch.int32], "b4_d1024_int32_mul"),
    (0, 32, 64, "uniform_8", [torch.float32], "b32_uniform"),
    (1, 64, 64, "ragged_1-4", [torch.float16], "b64_ragged_fp16_mul"),
]

# One dtype per small shape, balanced roughly evenly across fp32, fp16 and
# int32, plus mandatory vec-width edge coverage per dtype.
SMALL_SELECT = {
    "b1_d1_min": torch.float32,
    "fp32_vec_eq": torch.float32,
    "fp32_vec_eq_mul": torch.int32,
    "vec_minus1": torch.int32,
    "fp16_vec_eq": torch.float16,
    "fp16_vec_eq_mul": torch.float16,
    "vec_plus1_chunk": torch.float32,
    "fp16_vec_minus1": torch.float16,
    "fp16_vec_plus1_chunk": torch.float16,
    "chunked_d_ragged": torch.float32,
    "empty_all_zero": torch.float32,
    "with_zero_batch": torch.int32,
    "b8_uniform_mul": torch.float16,
    "small_d32_ragged": torch.int32,
}

CASES_SMALL = [
    Case(op_mode, b, d, l_pattern, SMALL_SELECT[label], label)
    for (op_mode, b, d, l_pattern, allowed_dt, label) in SMALL_FAST_SHAPES
]


def _build_cases_large():
    cases = []
    for op_mode, b, d, l_pattern, allowed_dt, label in SMALL_FAST_SHAPES:
        for dt in allowed_dt:
            if dt != SMALL_SELECT[label]:
                cases.append(Case(op_mode, b, d, l_pattern, dt, label))
    for op_mode, b, d, l_pattern, allowed_dt, label in LARGE_ONLY_SHAPES:
        for dt in allowed_dt:
            cases.append(Case(op_mode, b, d, l_pattern, dt, label))
    return cases


CASES_LARGE = _build_cases_large()


def _case_id(case):
    return f"{case.label}-{str(case.dtype).rsplit('.', 1)[-1]}"


def _run_case(case):
    seed_key = ("nbo", case.label, case.op_mode, case.b, case.d,
                case.l_pattern, str(case.dtype))
    values, offsets, dense = _gen_inputs(
        case.b, case.d, case.l_pattern, case.dtype, seed_key)

    expected = _ref_nested_binary_op(values, offsets, dense, case.op_mode)

    result_npu = torch.ops.ops_multimodal_fusion.nested_binary_op(
        values.npu(), offsets.npu(), dense.npu(), case.op_mode)
    result = result_npu.cpu()

    assert result.dtype == expected.dtype, (
        f"dtype mismatch (label={case.label}): "
        f"got {result.dtype}, want {expected.dtype}")
    assert result.shape == expected.shape, (
        f"shape mismatch (label={case.label}): "
        f"got {tuple(result.shape)}, want {tuple(expected.shape)}")

    tol = _TOL[case.dtype]
    if tol is None:
        assert torch.equal(result, expected), (
            f"int32 mismatch (label={case.label}, op_mode={case.op_mode}, "
            f"b={case.b}, d={case.d}, l_pattern={case.l_pattern})")
    else:
        assert torch.allclose(result, expected, **tol), (
            f"value mismatch (label={case.label}, op_mode={case.op_mode}, "
            f"b={case.b}, d={case.d}, l_pattern={case.l_pattern}, "
            f"dtype={case.dtype})\n"
            f"  max_abs_diff={(result - expected).abs().max().item():.6g}\n"
            f"  tol={tol}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_SMALL, ids=[c.label for c in CASES_SMALL])
def test_nested_binary_op_small(case):
    _run_case(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_LARGE, ids=[_case_id(c) for c in CASES_LARGE])
def test_nested_binary_op_large(case):
    _run_case(case)


# Negative-path tests: argument validation.


def _ok_inputs():
    """Build a valid values, offsets, dense and op_mode tuple on NPU."""
    b, d = 2, 8
    lengths = [2, 3]
    sum_l = sum(lengths)
    values = torch.randn((sum_l, d), dtype=torch.float32).npu()
    offsets = torch.tensor([0, lengths[0], lengths[0] + lengths[1]], dtype=torch.int64).npu()
    dense = torch.randn((b, 1, d), dtype=torch.float32).npu()
    return values, offsets, dense, 0


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_binary_op_values_1d_rejected():
    _, offsets, dense, om = _ok_inputs()
    values = torch.randn((5,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_binary_op(values, offsets, dense, om)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_binary_op_offsets_2d_rejected():
    values, _, dense, om = _ok_inputs()
    offsets = torch.tensor([[0, 2], [3, 5]], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_binary_op(values, offsets, dense, om)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_binary_op_dense_2d_rejected():
    values, offsets, _, om = _ok_inputs()
    dense = torch.randn((2, 8), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_binary_op(values, offsets, dense, om)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_binary_op_dense_size1_not_one():
    values, offsets, _, om = _ok_inputs()
    # The dense middle dimension must be one; here it is two.
    dense = torch.randn((2, 2, 8), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_binary_op(values, offsets, dense, om)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_binary_op_dim_mismatch():
    values, offsets, _, om = _ok_inputs()
    # The dense last dimension is four versus values last dimension eight.
    dense = torch.randn((2, 1, 4), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_binary_op(values, offsets, dense, om)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_binary_op_offsets_size_mismatch():
    values, _, dense, om = _ok_inputs()
    # With batch two the offsets length must be three; here it is four.
    offsets = torch.tensor([0, 2, 4, 5], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_binary_op(values, offsets, dense, om)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_binary_op_offsets_int32_rejected():
    values, _, dense, om = _ok_inputs()
    offsets = torch.tensor([0, 2, 5], dtype=torch.int32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_binary_op(values, offsets, dense, om)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_binary_op_offsets_nonzero_start():
    values, _, dense, om = _ok_inputs()
    offsets = torch.tensor([1, 3, 5], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_binary_op(values, offsets, dense, om)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_binary_op_offsets_not_monotonic():
    values, _, dense, om = _ok_inputs()
    # The third offset is smaller than the second.
    offsets = torch.tensor([0, 4, 2], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_binary_op(values, offsets, dense, om)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_binary_op_offsets_wrong_total():
    values, _, dense, om = _ok_inputs()
    # Values has five rows but the final offset claims seven.
    offsets = torch.tensor([0, 3, 7], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_binary_op(values, offsets, dense, om)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_binary_op_op_mode_invalid():
    values, offsets, dense, _ = _ok_inputs()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_binary_op(values, offsets, dense, 2)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_binary_op_dtype_mismatch():
    values, offsets, _, om = _ok_inputs()
    dense = torch.randn((2, 1, 8), dtype=torch.float16).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_binary_op(values, offsets, dense, om)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_binary_op_bf16_rejected():
    b, d = 2, 8
    values = torch.randn((5, d), dtype=torch.bfloat16).npu()
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64).npu()
    dense = torch.randn((b, 1, d), dtype=torch.bfloat16).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_binary_op(values, offsets, dense, 0)
