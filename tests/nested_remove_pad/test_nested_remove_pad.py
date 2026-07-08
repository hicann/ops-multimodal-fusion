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

"""Tests for ops_multimodal_fusion.nested_remove_pad.

This operator unpacks a dense padded tensor into the concatenation of its valid
per-batch rows. It takes a three dimensional padded tensor and a one
dimensional lengths tensor giving the valid row count per batch; it returns a
two dimensional tensor that concatenates the valid rows of every batch. Each
batch length must lie between zero and the padded maximum length.

Supported dtypes are float32, float16, and int32. The trailing dimension width
in bytes must be a multiple of thirty-two.

Because the operator is a pure copy with no floating point arithmetic, all
comparisons are exact.

Test case counts:
  - test_nested_remove_pad_small                       :  10 cases
  - test_nested_remove_pad_large                       :   8 cases
  - test_nested_remove_pad_interface_exist             :   1
  - test_nested_remove_pad_padded_2d_rejected          :   1
  - test_nested_remove_pad_padded_4d_rejected          :   1
  - test_nested_remove_pad_lengths_2d_rejected         :   1
  - test_nested_remove_pad_lengths_size_mismatch       :   1
  - test_nested_remove_pad_lengths_int32_rejected      :   1
  - test_nested_remove_pad_length_negative_rejected    :   1
  - test_nested_remove_pad_length_exceeds_max_rejected :   1
  - test_nested_remove_pad_d_misaligned_rejected       :   1
  - test_nested_remove_pad_bf16_rejected               :   1
  - test_nested_remove_pad_int64_rejected              :   1
  - Total                                              :  29 cases
"""

import torch
import torch_npu  # noqa: F401
import pytest
import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "nested_remove_pad"):
    pytest.skip(
        "ops_multimodal_fusion.nested_remove_pad not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_nested_remove_pad_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "nested_remove_pad"),\
        "The 'nested_remove_pad' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Input generator: produce the padded tensor, the lengths tensor, and the
# expected output.
#
# Length-mode semantics:
#   uniform   : every batch has the same fixed length
#   ragged    : per-batch length drawn at random within the maximum, seeded
#   all_max   : every batch fills the maximum, leaving no padding
#   all_zero  : every batch is empty
#   with_zero : alternating empty and non-empty batches
# ---------------------------------------------------------------------------


def _make_lengths(b, max_l, l_mode, rng):
    if l_mode.startswith("uniform_"):
        n = int(l_mode.split("_", 1)[1])
        return [n] * b
    if l_mode == "all_max":
        return [max_l] * b
    if l_mode == "all_zero":
        return [0] * b
    if l_mode == "with_zero":
        k = max(2, max_l // 2)
        return [0 if (i % 2 == 0) else k for i in range(b)]
    if l_mode == "ragged":
        return [int(rng.randint(1, max_l)) for _ in range(b)]
    raise ValueError(f"unknown l_mode: {l_mode}")


def _gen_inputs(case, seed_key):
    b, max_l, d, l_mode, dtype = case
    import random
    rng = random.Random(abs(hash(seed_key)) % 10_000_000)
    torch.manual_seed(abs(hash(("torch", seed_key))) % 10_000_000)

    lengths_list = _make_lengths(b, max_l, l_mode, rng)
    lengths = torch.tensor(lengths_list, dtype=torch.int64).contiguous()

    if dtype == torch.int32:
        padded = torch.randint(-1_000, 1_000, (b, max_l, d), dtype=torch.int32)
    else:
        padded = torch.randn((b, max_l, d)).to(dtype)

    # Reference: concat valid sub-slices.
    chunks = []
    for i in range(b):
        l_i = lengths_list[i]
        if l_i > 0:
            chunks.append(padded[i, :l_i, :])
    if not chunks:
        expected = torch.empty((0, d), dtype=dtype)
    else:
        expected = torch.cat(chunks, dim=0)

    return padded.contiguous(), lengths, expected


# ---------------------------------------------------------------------------
# Case matrix. Each row carries batch count, maximum length, trailing width,
# length mode, dtype, and a label. A pure copy compares exactly across dtypes.
# ---------------------------------------------------------------------------

CASES_SMALL = [
    # Small set keeps lengths uniform and modest; ragged, single-row, and
    # alternating patterns are deferred to the large set for hardware testing.
    (1, 4, 8, "all_max", torch.float32, "b1_L4_no_pad_fp32"),
    (1, 4, 16, "uniform_2", torch.float16, "b1_L2_fp16"),
    (1, 4, 8, "uniform_4", torch.int32, "b1_L4_int32"),
    (2, 4, 8, "uniform_4", torch.float32, "b2_uniform_L4_fp32"),
    (2, 4, 16, "uniform_2", torch.float16, "b2_uniform_L2_fp16"),
    (2, 4, 8, "uniform_4", torch.int32, "b2_uniform_L4_int32"),
    (4, 8, 16, "uniform_4", torch.float32, "b4_uniform_L4_fp32"),
    (4, 8, 32, "uniform_4", torch.float16, "b4_uniform_L4_fp16"),
    (4, 4, 8, "all_max", torch.int32, "b4_no_pad_int32"),
    (2, 4, 8, "all_zero", torch.float32, "all_zero_short_fp32"),
]

CASES_LARGE = [
    # Ragged lengths (real-hardware functional testing only).
    (4, 16, 64, "ragged", torch.float32, "b4_ragged_d64_fp32"),
    (8, 16, 64, "ragged", torch.float16, "b8_ragged_d64_fp16"),
    (4, 16, 128, "ragged", torch.float16, "b4_ragged_d128_fp16"),
    (16, 8, 16, "ragged", torch.float32, "b16_ragged_d16_fp32"),
    # Single-row degenerate batches (real-hardware only).
    (4, 8, 32, "uniform_1", torch.float16, "b4_L1_fp16"),
    (2, 4, 8, "uniform_1", torch.float32, "b2_L1_fp32"),
    # Alternating empty and non-empty batches.
    (3, 4, 16, "with_zero", torch.float16, "with_zero_fp16"),
    # Larger sizes.
    (4, 32, 32, "uniform_8", torch.float32, "b4_uniform_L8_fp32"),
]


def _run_case(case):
    b, max_l, d, l_mode, dtype, label = case
    seed_key = ("nrp", label, b, max_l, d, l_mode, str(dtype))
    padded, lengths, expected = _gen_inputs((b, max_l, d, l_mode, dtype), seed_key)

    # Compute on NPU, immediately copy to CPU, drop NPU refs.
    # Keeping NPU tensors as locals at assertion time triggers a
    # torch._tensor_str segfault when pytest formats traceback locals.
    padded_npu = padded.npu()
    lengths_npu = lengths.npu()
    result = torch.ops.ops_multimodal_fusion.nested_remove_pad(padded_npu, lengths_npu).cpu()
    del padded_npu, lengths_npu

    got_dtype = result.dtype
    got_shape = tuple(result.shape)
    want_dtype = expected.dtype
    want_shape = tuple(expected.shape)
    assert got_dtype == want_dtype, (
        f"dtype mismatch (label={label}): got {got_dtype}, want {want_dtype}")
    assert got_shape == want_shape, (
        f"shape mismatch (label={label}): got {got_shape}, want {want_shape}")
    if got_shape[0] == 0:
        # Empty result; torch.equal on empty tensors with matching shape/dtype
        # is True by definition.
        return
    eq = torch.equal(result, expected)
    if not eq:
        first_diff = (result != expected).nonzero(as_tuple=False)
        n_diff = first_diff.shape[0]
        sample = first_diff[:5].tolist() if n_diff > 0 else []
        assert False, (
            f"value mismatch (label={label}, B={b}, max_L={max_l}, D={d}, "
            f"l_mode={l_mode}, dtype={dtype}); n_diff={n_diff} first={sample}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_SMALL, ids=[c[-1] for c in CASES_SMALL])
def test_nested_remove_pad_small(case):
    _run_case(case)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_LARGE, ids=[c[-1] for c in CASES_LARGE])
def test_nested_remove_pad_large(case):
    _run_case(case)


# ---------------------------------------------------------------------------
# Negative-path tests.
# ---------------------------------------------------------------------------


def _ok_inputs(dtype=torch.float32):
    b, max_l, d = 2, 4, 8
    padded = torch.randn((b, max_l, d)).to(dtype).npu()
    lengths = torch.tensor([2, 3], dtype=torch.int64).npu()
    return padded, lengths


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_remove_pad_padded_2d_rejected():
    padded = torch.randn((4, 8), dtype=torch.float32).npu()
    lengths = torch.tensor([2, 3, 4, 1], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_remove_pad(padded, lengths)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_remove_pad_padded_4d_rejected():
    padded = torch.randn((2, 4, 4, 8), dtype=torch.float32).npu()
    lengths = torch.tensor([1, 2], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_remove_pad(padded, lengths)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_remove_pad_lengths_2d_rejected():
    padded, _ = _ok_inputs()
    lengths = torch.tensor([[2], [3]], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_remove_pad(padded, lengths)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_remove_pad_lengths_size_mismatch():
    padded, _ = _ok_inputs()
    # The lengths tensor has more entries than the padded batch count.
    lengths = torch.tensor([2, 3, 1], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_remove_pad(padded, lengths)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_remove_pad_lengths_int32_rejected():
    padded, _ = _ok_inputs()
    lengths = torch.tensor([2, 3], dtype=torch.int32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_remove_pad(padded, lengths)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_remove_pad_length_negative_rejected():
    padded, _ = _ok_inputs()
    lengths = torch.tensor([-1, 2], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_remove_pad(padded, lengths)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_remove_pad_length_exceeds_max_rejected():
    padded, _ = _ok_inputs()   # padded maximum length is four
    lengths = torch.tensor([5, 2], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_remove_pad(padded, lengths)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_remove_pad_d_misaligned_rejected():
    # A trailing width whose byte size is not a multiple of thirty-two.
    padded = torch.randn((2, 4, 5), dtype=torch.float32).npu()
    lengths = torch.tensor([1, 2], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_remove_pad(padded, lengths)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_remove_pad_bf16_rejected():
    padded = torch.randn((2, 4, 8), dtype=torch.bfloat16).npu()
    lengths = torch.tensor([1, 2], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_remove_pad(padded, lengths)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_remove_pad_int64_rejected():
    padded = torch.randint(-5, 5, (2, 4, 8), dtype=torch.int64).npu()
    lengths = torch.tensor([1, 2], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_remove_pad(padded, lengths)
