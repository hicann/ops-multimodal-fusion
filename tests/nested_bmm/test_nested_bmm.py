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

"""Tests for ops_multimodal_fusion.nested_bmm (RegBase vector-path outer-product,
nested / grouped GEMM).

Schema:
  nested_bmm Tensor a_values Tensor b_values Tensor sizes to Tensor

  a_values : 1-D flat sum_i M_i K_i contiguous dtype in fp32 fp16
  b_values : 1-D flat sum_i K_i N_i same dtype
  sizes : 2-D B 3 int64 per-row M_i K_i N_i
  out : 1-D flat sum_i M_i N_i same dtype

Constraints:
  - N_i * sizeof(dtype) must be a multiple of 32 bytes (fp32 N_i%8 == 0,
    fp16 N_i%16 == 0); Reg::LoadAlign / StoreAlign row-stride alignment.
  - M_i > 1 K_i > 1 N_i > 1.
  - Sum(M_i*K_i) and sum(K_i*N_i) must match a_values / b_values lengths.

Tolerances 大归约档 K-accumulation; atol scales with max_K :
  - fp32: rtol 1e-3 atol max 1e-3 max_K 1e-3
  - fp16: rtol 5e-3 atol max 5e-3 max_K 5e-3

Reference: CPU torch.bmm computed in fp64 then cast to dtype (avoids CPU fp16
sim divergence); for ragged shapes, computed per-batch and concatenated.

Test case counts:
  - test_nested_bmm_small                  :   8 cases  (M_i >= 2)
  - test_nested_bmm_large : 6 cases includes M_i 1 GEMV
  - test_nested_bmm_interface_exist        :   1
  - test_nested_bmm_a_2d_rejected          :   1
  - test_nested_bmm_b_2d_rejected          :   1
  - test_nested_bmm_sizes_1d_rejected      :   1
  - test_nested_bmm_sizes_wrong_cols       :   1
  - test_nested_bmm_sizes_int32_rejected   :   1
  - test_nested_bmm_size_mismatch_a        :   1
  - test_nested_bmm_size_mismatch_b        :   1
  - test_nested_bmm_dtype_mismatch         :   1
  - test_nested_bmm_bf16_rejected          :   1
  - test_nested_bmm_int32_rejected         :   1
  - test_nested_bmm_n_misaligned_rejected  :   1
  - Total                                  :  26 cases
"""

import torch
import torch_npu  # noqa: F401
import pytest
import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "nested_bmm"):
    pytest.skip(
        "ops_multimodal_fusion.nested_bmm not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_nested_bmm_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "nested_bmm"),\
        "The 'nested_bmm' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Tolerance.
# ---------------------------------------------------------------------------


def _tol_for(dtype, max_k):
    if dtype == torch.float32:
        return dict(rtol=1e-3, atol=max(1e-3, max_k * 1e-3))
    if dtype == torch.float16:
        return dict(rtol=5e-3, atol=max(5e-3, max_k * 5e-3))
    raise ValueError(f"unsupported dtype {dtype}")


# ---------------------------------------------------------------------------
# Input generator. batch_shapes list of M_i K_i N_i .
# ---------------------------------------------------------------------------


def _gen_nested(batch_shapes, dtype, seed_key):
    torch.manual_seed(abs(hash(seed_key)) % 10_000_000)
    a_chunks, b_chunks, c_chunks = [], [], []
    for (m, k, n) in batch_shapes:
        a = (torch.randn(m, k, dtype=torch.float32) * 0.5).to(dtype)
        b = (torch.randn(k, n, dtype=torch.float32) * 0.5).to(dtype)
        c = torch.mm(a.double(), b.double()).to(dtype)
        a_chunks.append(a.reshape(-1))
        b_chunks.append(b.reshape(-1))
        c_chunks.append(c.reshape(-1))
    a_values = torch.cat(a_chunks).contiguous()
    b_values = torch.cat(b_chunks).contiguous()
    expected = torch.cat(c_chunks).contiguous()
    sizes = torch.tensor(batch_shapes, dtype=torch.int64).contiguous()
    return a_values, b_values, sizes, expected


# ---------------------------------------------------------------------------
# Case matrix. Each row: batch_shapes_list dtype label .
# _small: M_i >= 2; fast set for routine functional validation.
# _large: includes M_i 1 GEMV cases deferred to real-hardware functional
#         testing) + larger sizes.
# ---------------------------------------------------------------------------

CASES_SMALL = [
    # Single-batch baseline (uniform).
    ([(4, 4, 8)], torch.float32, "single_4x4x8_fp32"),
    ([(2, 8, 16)], torch.float32, "single_2x8x16_fp32"),
    ([(4, 4, 16)], torch.float16, "single_4x4x16_fp16"),
    # Multi-batch uniform.
    ([(4, 4, 8), (4, 4, 8)], torch.float32, "uniform_b2_fp32"),
    ([(2, 4, 16), (2, 4, 16), (2, 4, 16)],
                                      torch.float16, "uniform_b3_fp16"),
    # Ragged M / N (same K).
    ([(2, 4, 8), (4, 4, 16), (3, 4, 8)],
                                      torch.float32, "ragged_MN_b3_fp32"),
    # Ragged all M K N each varies . N must be multiple of 16 for fp16.
    ([(2, 4, 16), (3, 8, 32), (4, 2, 16)],
                                      torch.float16, "ragged_all_b3_fp16"),
    # Multiple uniform batches stressing UB.
    ([(8, 8, 16)] * 4, torch.float32, "b4_8cube_fp32"),
]

CASES_LARGE = [
    # M_i 1 GEMV-degenerate cases real-hardware functional testing only .
    ([(1, 8, 8)], torch.float32, "single_M1_fp32"),
    ([(1, 8, 16), (1, 8, 32)], torch.float16, "M1_b2_fp16"),
    ([(1, 4, 8), (4, 4, 8), (1, 4, 16)],
                                      torch.float32, "ragged_with_M1_fp32"),
    # Larger sizes.
    ([(16, 16, 32)] * 2, torch.float32, "b2_16cube_fp32"),
    ([(32, 16, 32), (8, 16, 64)], torch.float16, "ragged_large_fp16"),
    ([(8, 8, 16), (16, 8, 16), (24, 8, 16), (32, 8, 16)],
                                      torch.float32, "b4_ragged_M_fp32"),
]


def _run_case(batch_shapes, dtype, label):
    seed_key = ("nested_bmm", label, tuple(map(tuple, batch_shapes)), str(dtype))
    a, b, sizes, expected = _gen_nested(batch_shapes, dtype, seed_key)

    max_k = max(k for (_, k, _) in batch_shapes)

    result_npu = torch.ops.ops_multimodal_fusion.nested_bmm(
        a.npu(), b.npu(), sizes.npu())
    result = result_npu.cpu()

    assert result.dtype == expected.dtype, (
        f"dtype mismatch (label={label}): got {result.dtype}, want {expected.dtype}")
    assert result.shape == expected.shape, (
        f"shape mismatch (label={label}): "
        f"got {tuple(result.shape)}, want {tuple(expected.shape)}")

    tol = _tol_for(dtype, max_k)
    assert torch.allclose(result, expected, **tol), (
        f"value mismatch (label={label}, batch_shapes={batch_shapes}, dtype={dtype})\n"
        f"  max_abs_diff={(result.float() - expected.float()).abs().max().item():.6g}\n"
        f"  tol={tol}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("batch_shapes,dtype,label", CASES_SMALL)
def test_nested_bmm_small(batch_shapes, dtype, label):
    _run_case(batch_shapes, dtype, label)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("batch_shapes,dtype,label", CASES_LARGE)
def test_nested_bmm_large(batch_shapes, dtype, label):
    _run_case(batch_shapes, dtype, label)


# ---------------------------------------------------------------------------
# Negative-path tests.
# ---------------------------------------------------------------------------


def _ok_inputs(dtype=torch.float32):
    batch_shapes = [(2, 4, 8), (4, 4, 8)]
    a, b, sizes, _ = _gen_nested(batch_shapes, dtype, "ok")
    return a.npu(), b.npu(), sizes.npu()


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_bmm_a_2d_rejected():
    _, b, sizes = _ok_inputs()
    a = torch.randn((8, 4), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_bmm(a, b, sizes)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_bmm_b_2d_rejected():
    a, _, sizes = _ok_inputs()
    b = torch.randn((4, 8), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_bmm(a, b, sizes)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_bmm_sizes_1d_rejected():
    a, b, _ = _ok_inputs()
    sizes = torch.tensor([2, 4, 8, 4, 4, 8], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_bmm(a, b, sizes)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_bmm_sizes_wrong_cols():
    a, b, _ = _ok_inputs()
    sizes = torch.tensor([[2, 4, 8, 0], [4, 4, 8, 0]], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_bmm(a, b, sizes)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_bmm_sizes_int32_rejected():
    a, b, _ = _ok_inputs()
    sizes = torch.tensor([[2, 4, 8], [4, 4, 8]], dtype=torch.int32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_bmm(a, b, sizes)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_bmm_size_mismatch_a():
    _, b, sizes = _ok_inputs()
    # sizes says sum M K 2 4 + 4 4 24 but provide 20 elements.
    a = torch.randn((20,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_bmm(a, b, sizes)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_bmm_size_mismatch_b():
    a, _, sizes = _ok_inputs()
    # sizes says sum K N 4 8 + 4 8 64 but provide 40 elements.
    b = torch.randn((40,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_bmm(a, b, sizes)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_bmm_dtype_mismatch():
    a, _, sizes = _ok_inputs(dtype=torch.float32)
    b = torch.randn((64,), dtype=torch.float16).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_bmm(a, b, sizes)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_bmm_bf16_rejected():
    batch_shapes = [(2, 4, 8)]
    a = torch.randn((8,), dtype=torch.bfloat16).npu()
    b = torch.randn((32,), dtype=torch.bfloat16).npu()
    sizes = torch.tensor(batch_shapes, dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_bmm(a, b, sizes)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_bmm_int32_rejected():
    batch_shapes = [(2, 4, 8)]
    a = torch.randint(-5, 5, (8,), dtype=torch.int32).npu()
    b = torch.randint(-5, 5, (32,), dtype=torch.int32).npu()
    sizes = torch.tensor(batch_shapes, dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_bmm(a, b, sizes)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nested_bmm_n_misaligned_rejected():
    # fp32 N 5 is not multiple of 8 to host TORCH_CHECK rejects.
    batch_shapes = [(2, 4, 5)]
    a = torch.randn((8,), dtype=torch.float32).npu()
    b = torch.randn((20,), dtype=torch.float32).npu()
    sizes = torch.tensor(batch_shapes, dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.nested_bmm(a, b, sizes)
