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
"""Tests for ops_multimodal_fusion.kthvalue.

Computes the k-th smallest value along ``dim`` and returns ``(values, indices)``.
Values and indices use strict ``torch.equal`` (values are an element-copy from
the input; indices are int64 positions). Inputs are generated with unique
values on the reduction axis so the index is unambiguous (PyTorch's tie-break
on kthvalue is not guaranteed to match the first-occurrence rule used here).

Test case counts:
  - test_kthvalue_small        : 84 cases (14 shape entries × 3 dtypes × 2 keepdim)
  - test_kthvalue_large        : 10 cases (5 shape entries × 2 dtypes × 1 keepdim)
  - test_kthvalue_interface_exist : 1
  - test_kthvalue_invalid_k       : 1
  - test_kthvalue_invalid_dim     : 1
  - test_kthvalue_int64_rejected  : 1
  - test_kthvalue_bf16_rejected   : 1
  - Total                  : 99 cases
"""

import pytest
import torch
import torch_npu

import ops_multimodal_fusion

if not hasattr(torch.ops.ops_multimodal_fusion, "kthvalue"):
    pytest.skip(
        "ops_multimodal_fusion.kthvalue not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_kthvalue_interface_exist():
    """The 'ops_multimodal_fusion.kthvalue' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "kthvalue"),\
        "The 'kthvalue' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Input generation that guarantees uniqueness along the reduction axis.
# ---------------------------------------------------------------------------


def _make_unique_along_dim(shape, dim, dtype, seed_key):
    """Return a tensor whose values along ``dim`` are pairwise distinct.

    For each reduction-pair we use ``torch.randperm(dim_size)``: integers
    [0, dim_size) are exactly representable in fp32 / int32 and in fp16
    up to dim_size <= 2048 (11-bit precision). Tests choose dim_size
    accordingly.

    ``seed_key`` makes every parametrized case deterministic so failures
    are reproducible across reruns.
    """
    torch.manual_seed(abs(hash(seed_key)) % 10_000_000)
    sizes = list(shape)
    ndim = len(sizes)
    dim_pos = dim if dim >= 0 else dim + ndim
    dim_size = sizes[dim_pos]
    outer = 1
    for i in range(dim_pos):
        outer *= sizes[i]
    inner = 1
    for i in range(dim_pos + 1, ndim):
        inner *= sizes[i]

    # (outer * inner) independent permutations, then reshape to (outer, dim_size, inner).
    perms = torch.stack(
        [torch.randperm(dim_size) for _ in range(outer * inner)], dim=0
    )
    # perms: (outer * inner, dim_size) -> (outer, inner, dim_size) -> (outer, dim_size, inner)
    perms = perms.view(outer, inner, dim_size).permute(0, 2, 1).contiguous()
    arr = perms.view(*sizes)
    return arr.to(dtype)


# ---------------------------------------------------------------------------
# Case matrix.
# Each row: (shape, dim, k). Tuned so dim_size stays in the fp16/bf16 exact
# integer range (<=128 for bf16, <=2048 for fp16) when those dtypes apply.
# ---------------------------------------------------------------------------

CASES_SMALL = [
    # (shape, dim, k) — dim_size small enough for every supported dtype
    ((64,), 0, 1),
    ((64,), 0, 32),
    ((64,), 0, 64),
    ((64,), -1, 16),
    ((8, 32), -1, 1),
    ((8, 32), -1, 16),
    ((8, 32), -1, 32),
    ((32, 8), 0, 16),       # innerSize > 1 strided load
    ((4, 64, 8), 1, 32),       # 3D middle dim
    ((4, 64, 8), 1, 1),
    ((4, 16, 64), 2, 32),
    ((128,), 0, 64),       # median (k = ceil(n/2))
    ((128, 4), 0, 64),       # multi-pair innerSize
    ((2, 3, 64), -1, 32),
]

# Cases with larger dim_size; only run for fp32 and int32 (fp16 / bf16
# cannot represent the integer permutations exactly past their mantissa
# width and would flake on tie comparisons).
CASES_LARGE_FP32_INT32 = [
    ((512,), 0, 256),        # full medium reduction, k = median
    ((1024,), 0, 1),
    ((1024,), 0, 1024),       # k = dimSize, worst-case algorithm
    ((4, 1023), -1, 512),        # odd dim_size
    ((1024, 8), 0, 512),        # large outerSize, innerSize > 1
]

DTYPES_ALL = [torch.float32, torch.float16, torch.int32]
DTYPES_LARGE = [torch.float32, torch.int32]


def _expected(x_cpu, k, dim, keepdim):
    """CPU reference. Inputs are guaranteed tie-free per
    ``_make_unique_along_dim``, so we can call torch.kthvalue directly
    on the typed tensor and trust the index match.
    """
    return torch.kthvalue(x_cpu, k, dim=dim, keepdim=keepdim)


def _run(x_cpu, k, dim, keepdim, dtype):
    expected_v, expected_i = _expected(x_cpu, k, dim, keepdim)
    rv_npu, ri_npu = torch.ops.ops_multimodal_fusion.kthvalue(
        x_cpu.npu(), int(k), int(dim), bool(keepdim)
    )
    result_v = rv_npu.cpu()
    result_i = ri_npu.cpu()

    assert result_v.dtype == dtype,\
        f"values dtype mismatch: got {result_v.dtype}, want {dtype}"
    assert result_i.dtype == torch.int64,\
        f"indices dtype mismatch: got {result_i.dtype}, want torch.int64"
    assert result_v.shape == expected_v.shape,\
        f"values shape mismatch: got {tuple(result_v.shape)}, want {tuple(expected_v.shape)}"
    assert result_i.shape == expected_i.shape,\
        f"indices shape mismatch: got {tuple(result_i.shape)}, want {tuple(expected_i.shape)}"

    # values are an exact element copy → strict equality.
    assert torch.equal(result_v, expected_v), (
        f"values mismatch (dtype={dtype}, shape={tuple(x_cpu.shape)}, "
        f"k={k}, dim={dim}, keepdim={keepdim})"
    )
    assert torch.equal(result_i, expected_i), (
        f"indices mismatch (dtype={dtype}, shape={tuple(x_cpu.shape)}, "
        f"k={k}, dim={dim}, keepdim={keepdim})"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape,dim,k", CASES_SMALL)
@pytest.mark.parametrize("dtype", DTYPES_ALL)
@pytest.mark.parametrize("keepdim", [False, True])
def test_kthvalue_small(shape, dim, k, dtype, keepdim):
    seed_key = ("small", tuple(shape), int(dim), int(k), str(dtype), bool(keepdim))
    x_cpu = _make_unique_along_dim(shape, dim, dtype, seed_key)
    _run(x_cpu, k, dim, keepdim, dtype)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape,dim,k", CASES_LARGE_FP32_INT32)
@pytest.mark.parametrize("dtype", DTYPES_LARGE)
@pytest.mark.parametrize("keepdim", [False])
def test_kthvalue_large(shape, dim, k, dtype, keepdim):
    seed_key = ("large", tuple(shape), int(dim), int(k), str(dtype), bool(keepdim))
    x_cpu = _make_unique_along_dim(shape, dim, dtype, seed_key)
    _run(x_cpu, k, dim, keepdim, dtype)


# Negative-path tests: argument validation.


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_kthvalue_invalid_k():
    x = torch.randn(8, 16).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.kthvalue(x, 0, -1, False)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.kthvalue(x, 17, -1, False)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_kthvalue_invalid_dim():
    x = torch.randn(8, 16).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.kthvalue(x, 1, 5, False)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_kthvalue_int64_rejected():
    """int64 inputs are deferred per the dav-3510 first-version note."""
    x = torch.randint(0, 1000, (8, 16), dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.kthvalue(x, 1, -1, False)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_kthvalue_bf16_rejected():
    """bf16 inputs are deferred per the dav-3510 first-version note
    (bisheng cannot select scalar `<` on __bf16).
    """
    x = torch.randn(8, 16, dtype=torch.bfloat16).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.kthvalue(x, 1, -1, False)
