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

"""Tests for ops_multimodal_fusion.nested_add_pad.

This operator packs a ragged set of per-batch row groups back into a dense
padded tensor. It takes a two dimensional values tensor, a one dimensional
offsets tensor giving the cumulative row counts, a maximum length, and a
padding value; it returns a three dimensional padded tensor. For each batch the
valid rows are copied into the leading positions and the remaining positions
are filled with the padding value. It is the inverse of nested_remove_pad and
mirrors the dense conversion of a two dimensional nested tensor in PyTorch.

Supported dtypes are float32, float16, and int32. The trailing dimension width
in bytes must be a multiple of thirty-two, and each batch length must lie
between zero and the maximum length.

Because the operator is a pure copy and fill with no floating point arithmetic,
all comparisons are exact.

Test case counts:
  - test_nested_add_pad_small                            :  10 cases
  - test_nested_add_pad_large                            :  10 cases
  - test_nested_add_pad_interface_exist                  :   1
  - test_nested_add_pad_values_1d_rejected               :   1
  - test_nested_add_pad_values_3d_rejected               :   1
  - test_nested_add_pad_offsets_2d_rejected              :   1
  - test_nested_add_pad_offsets_int32_rejected           :   1
  - test_nested_add_pad_offsets_nonzero_start_rejected   :   1
  - test_nested_add_pad_offsets_non_monotonic_rejected   :   1
  - test_nested_add_pad_offsets_total_mismatch_rejected  :   1
  - test_nested_add_pad_max_l_below_max_li_rejected      :   1
  - test_nested_add_pad_max_l_negative_rejected          :   1
  - test_nested_add_pad_d_misaligned_rejected            :   1
  - test_nested_add_pad_bf16_rejected                    :   1
  - test_nested_add_pad_int64_rejected                   :   1
  - Total                                                :  33 cases
"""

import torch
import torch_npu  # noqa: F401
import pytest
import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "nested_add_pad"):
    pytest.skip(
        "ops_multimodal_fusion.nested_add_pad not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_nested_add_pad_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "nested_add_pad"),\
        "The 'nested_add_pad' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Input generator: produce the values tensor, the offsets tensor, and the
# expected padded tensor.
#
# Length-mode semantics (matched with nested_remove_pad for shared coverage):
#   uniform   : every batch has the same fixed length
#   ragged    : per-batch length drawn at random within the maximum, seeded
#   all_max   : every batch fills the maximum, leaving no padding region
#   all_zero  : every batch is empty, so the whole tensor is padding
#   with_zero : alternating empty and non-empty batches
# ---------------------------------------------------------------------------


def _make_lengths(batch, max_l, l_mode, rng):
    if l_mode.startswith("uniform_"):
        n = int(l_mode.split("_", 1)[1])
        return [n] * batch
    if l_mode == "all_max":
        return [max_l] * batch
    if l_mode == "all_zero":
        return [0] * batch
    if l_mode == "with_zero":
        k = max(2, max_l // 2)
        return [0 if (i % 2 == 0) else k for i in range(batch)]
    if l_mode == "ragged":
        return [int(rng.randint(1, max_l)) for _ in range(batch)]
    raise ValueError(f"unknown l_mode: {l_mode}")


def _gen_inputs(case, seed_key):
    batch, max_l, d, l_mode, dtype, padding_value = case
    import random
    rng = random.Random(abs(hash(seed_key)) % 10_000_000)
    torch.manual_seed(abs(hash(("torch", seed_key))) % 10_000_000)

    lengths_list = _make_lengths(batch, max_l, l_mode, rng)
    sum_l = sum(lengths_list)
    offsets_list = [0]
    acc = 0
    for l_i in lengths_list:
        acc += l_i
        offsets_list.append(acc)
    offsets = torch.tensor(offsets_list, dtype=torch.int64).contiguous()

    if dtype == torch.int32:
        values = torch.randint(-1_000, 1_000, (sum_l, d), dtype=torch.int32)
    else:
        values = torch.randn((sum_l, d)).to(dtype)
    values = values.contiguous()

    # Reference: per-batch valid copy + padding fill.
    if dtype == torch.int32:
        expected = torch.full((batch, max_l, d), int(padding_value), dtype=torch.int32)
    else:
        expected = torch.full((batch, max_l, d), float(padding_value)).to(dtype)
    for i in range(batch):
        rs = offsets_list[i]
        re = offsets_list[i + 1]
        l_i = re - rs
        if l_i > 0:
            expected[i, :l_i, :] = values[rs:re, :]

    return values, offsets, expected


# ---------------------------------------------------------------------------
# Case matrix. Each row carries batch count, maximum length, trailing width,
# length mode, dtype, padding value, and a label. A pure copy and fill compares
# exactly across dtypes.
# ---------------------------------------------------------------------------

CASES_SMALL = [
    # Small set keeps lengths uniform and modest; ragged, single-row,
    # alternating, and non-zero padding patterns are deferred to the large set.
    (1, 4, 8, "all_max", torch.float32, 0.0, "b1_L4_no_pad_fp32"),
    (1, 4, 16, "uniform_2", torch.float16, 0.0, "b1_L2_fp16"),
    (1, 4, 8, "uniform_4", torch.int32, 0, "b1_L4_int32"),
    (2, 4, 8, "uniform_4", torch.float32, 0.0, "b2_uniform_L4_fp32"),
    (2, 4, 16, "uniform_2", torch.float16, 0.0, "b2_uniform_L2_fp16"),
    (2, 4, 8, "uniform_4", torch.int32, 0, "b2_uniform_L4_int32"),
    (4, 8, 16, "uniform_4", torch.float32, 0.0, "b4_uniform_L4_fp32"),
    (4, 8, 32, "uniform_4", torch.float16, 0.0, "b4_uniform_L4_fp16"),
    (4, 4, 8, "all_max", torch.int32, 0, "b4_no_pad_int32"),
    (2, 4, 8, "all_zero", torch.float32, 0.0, "all_zero_short_fp32"),
]

CASES_LARGE = [
    # Ragged lengths (real-hardware functional testing only).
    (4, 16, 64, "ragged", torch.float32, 0.0, "b4_ragged_d64_fp32"),
    (8, 16, 64, "ragged", torch.float16, 0.0, "b8_ragged_d64_fp16"),
    (4, 16, 128, "ragged", torch.float16, 0.0, "b4_ragged_d128_fp16"),
    (16, 8, 16, "ragged", torch.float32, 0.0, "b16_ragged_d16_fp32"),
    # Single-row degenerate batches.
    (4, 8, 32, "uniform_1", torch.float16, 0.0, "b4_L1_fp16"),
    (2, 4, 8, "uniform_1", torch.float32, 0.0, "b2_L1_fp32"),
    # Alternating empty and non-empty batches.
    (3, 4, 16, "with_zero", torch.float16, 0.0, "with_zero_fp16"),
    # Non-zero padding value.
    (2, 4, 8, "uniform_2", torch.float32, -1.5, "padval_neg15_fp32"),
    (2, 4, 8, "uniform_2", torch.int32, 7, "padval_7_int32"),
    # Larger sizes.
    (4, 32, 32, "uniform_8", torch.float32, 0.0, "b4_uniform_L8_fp32"),
]


def _run_case(case):
    batch, max_l, d, l_mode, dtype, padding_value, label = case
    seed_key = ("nap", label, batch, max_l, d, l_mode, str(dtype), padding_value)
    values, offsets, expected = _gen_inputs(
        (batch, max_l, d, l_mode, dtype, padding_value), seed_key)

    # Compute on NPU, immediately copy to CPU, drop NPU refs.
    # Keeping NPU tensors as locals at assertion time can trigger a
    # torch._tensor_str segfault when pytest formats traceback locals.
    values_npu = values.npu()
    offsets_npu = offsets.npu()
    result = torch.ops.ops_multimodal_fusion.nested_add_pad(
        values_npu, offsets_npu, max_l, padding_value).cpu()
    del values_npu, offsets_npu

    got_dtype = result.dtype
    got_shape = tuple(result.shape)
    want_dtype = expected.dtype
    want_shape = tuple(expected.shape)
    assert got_dtype == want_dtype, (
        f"dtype mismatch (label={label}): got {got_dtype}, want {want_dtype}")
    assert got_shape == want_shape, (
        f"shape mismatch (label={label}): got {got_shape}, want {want_shape}")
    if result.numel() == 0:
        return
    eq = torch.equal(result, expected)
    if not eq:
        first_diff = (result != expected).nonzero(as_tuple=False)
        n_diff = first_diff.shape[0]
        sample = first_diff[:5].tolist() if n_diff > 0 else []
        assert False, (
            f"value mismatch (label={label}, B={batch}, max_L={max_l}, D={d}, "
            f"l_mode={l_mode}, dtype={dtype}, pad={padding_value}); "
            f"n_diff={n_diff} first={sample}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_SMALL, ids=[c[-1] for c in CASES_SMALL])
def test_nested_add_pad_small(case):
    _run_case(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_LARGE, ids=[c[-1] for c in CASES_LARGE])
def test_nested_add_pad_large(case):
    _run_case(case)


# ---------------------------------------------------------------------------
# Negative-path tests.
# ---------------------------------------------------------------------------


def _ok_inputs(dtype=torch.float32):
    # Two batches with lengths two and three, producing offsets that sum to five.
    values = torch.randn((5, 8)).to(dtype).npu()
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64).npu()
    return values, offsets, 4   # maximum length


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_add_pad_values_1d_rejected():
    values = torch.randn((10,), dtype=torch.float32).npu()
    offsets = torch.tensor([0, 4, 10], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_add_pad(values, offsets, 8, 0.0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_add_pad_values_3d_rejected():
    values = torch.randn((2, 4, 8), dtype=torch.float32).npu()
    offsets = torch.tensor([0, 1, 2], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_add_pad(values, offsets, 4, 0.0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_add_pad_offsets_2d_rejected():
    values, _, max_l = _ok_inputs()
    offsets = torch.tensor([[0], [2], [5]], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_add_pad(values, offsets, max_l, 0.0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_add_pad_offsets_int32_rejected():
    values, _, max_l = _ok_inputs()
    offsets = torch.tensor([0, 2, 5], dtype=torch.int32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_add_pad(values, offsets, max_l, 0.0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_add_pad_offsets_nonzero_start_rejected():
    values, _, max_l = _ok_inputs()
    offsets = torch.tensor([1, 3, 5], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_add_pad(values, offsets, max_l, 0.0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_add_pad_offsets_non_monotonic_rejected():
    values, _, max_l = _ok_inputs()
    offsets = torch.tensor([0, 3, 2], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_add_pad(values, offsets, max_l, 0.0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_add_pad_offsets_total_mismatch_rejected():
    values, _, max_l = _ok_inputs()
    # The final offset disagrees with the row count of values.
    offsets = torch.tensor([0, 2, 4], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_add_pad(values, offsets, max_l, 0.0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_add_pad_max_l_below_max_li_rejected():
    values, offsets, _ = _ok_inputs()   # longest batch has three rows
    # A maximum length below the longest batch must be rejected.
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_add_pad(values, offsets, 2, 0.0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_add_pad_max_l_negative_rejected():
    values, offsets, _ = _ok_inputs()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_add_pad(values, offsets, -1, 0.0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_add_pad_d_misaligned_rejected():
    # A trailing width whose byte size is not a multiple of thirty-two.
    values = torch.randn((5, 5), dtype=torch.float32).npu()
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_add_pad(values, offsets, 4, 0.0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_add_pad_bf16_rejected():
    values = torch.randn((5, 8), dtype=torch.bfloat16).npu()
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_add_pad(values, offsets, 4, 0.0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_add_pad_int64_rejected():
    values = torch.randint(-5, 5, (5, 8), dtype=torch.int64).npu()
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_add_pad(values, offsets, 4, 0.0)
