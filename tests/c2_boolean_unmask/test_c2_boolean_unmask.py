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

"""Tests for ops_multimodal_fusion.c2_boolean_unmask.

Caffe2-style boolean unmask (inverse of c2_boolean_mask). Given a flat
alternating list inputs mask_0 values_0 mask_1 values_1 ... rebuild a
1-D output y of length maskSize:
    y i values_ j i r
where j(i) is the first mask true at position i (first-true-wins) and r is the
per-mask prefix count of positions assigned to mask j(i). Every output position
must be covered by at least one mask; the count assigned to mask j must equal
values_j.numel .

Golden reference: CPU first-true-wins reconstruction (_reference). Values are
copied verbatim (no arithmetic), so every dtype must match bytewise.

Test case counts:
  - test_c2_boolean_unmask_small             : 7 cases (minimal coverage per
                                               maskSize / dtype / mask mode)
  - test_c2_boolean_unmask_large             : 8 cases (large maskSize, more
                                               masks, overlap combos)
  - test_c2_boolean_unmask_interface_exist   : 1
  - test_c2_boolean_unmask_empty             : 1
  - test_c2_boolean_unmask_values_bf16_rejected   : 1
  - test_c2_boolean_unmask_values_int64_rejected  : 1
  - test_c2_boolean_unmask_mask_dtype_rejected    : 1
  - test_c2_boolean_unmask_odd_inputs_rejected    : 1
  - test_c2_boolean_unmask_too_few_inputs_rejected: 1
  - test_c2_boolean_unmask_mask_ndim_rejected     : 1
  - test_c2_boolean_unmask_mask_len_mismatch      : 1
  - test_c2_boolean_unmask_uncovered_position     : 1
  - test_c2_boolean_unmask_count_mismatch         : 1
  - Total                                    : 26 cases
"""

import torch
import torch_npu  # noqa: F401
import pytest
import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "c2_boolean_unmask"):
    pytest.skip(
        "ops_multimodal_fusion.c2_boolean_unmask not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_c2_boolean_unmask_interface_exist():
    """'ops_multimodal_fusion.c2_boolean_unmask' is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "c2_boolean_unmask"),\
        "The 'c2_boolean_unmask' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Input generators. All deterministic via seed_key.
# ---------------------------------------------------------------------------


def _seed(seed_key):
    torch.manual_seed(abs(hash(repr(seed_key))) % 10_000_000)


# Mask-set generators. Each returns a list of 1-D mask tensors (mask_dtype),
# all length maskSize, that together cover every position (first-true-wins).


def _make_masks(mask_size, mode, mask_dtype, sk):
    if mode == "single_all_true":
        return [torch.ones(mask_size, dtype=mask_dtype)]
    if mode == "two_complement":
        _seed(("complement", sk))
        m0 = torch.rand(mask_size) < 0.5
        m1 = ~m0
        return [m0.to(mask_dtype), m1.to(mask_dtype)]
    if mode == "three_partition":
        idx = torch.arange(mask_size)
        return [(idx % 3 == k).to(mask_dtype) for k in range(3)]
    if mode == "overlap":
        # mask_0 true on first half, mask_1 all true: overlap on first half ->
        # mask_0 wins there (first-true), mask_1 wins the second half.
        h = mask_size // 2
        m0 = torch.zeros(mask_size, dtype=torch.bool)
        m0[:h] = True
        m1 = torch.ones(mask_size, dtype=torch.bool)
        return [m0.to(mask_dtype), m1.to(mask_dtype)]
    raise ValueError(f"unknown mode {mode}")


def _assign_counts(masks):
    """Per-mask count of positions where that mask is the first true one."""
    mb = [m.cpu().to(torch.bool) for m in masks]
    n = mb[0].numel()
    counts = [0] * len(mb)
    for i in range(n):
        for j, m in enumerate(mb):
            if bool(m[i]):
                counts[j] += 1
                break
        else:
            raise AssertionError(f"position {i} not covered by any mask")
    return counts


def _make_values(counts, dtype, sk):
    vals = []
    for j, c in enumerate(counts):
        _seed(("values", sk, j))
        if dtype == torch.int32:
            vals.append(torch.randint(-(2 ** 20), 2 ** 20, (c,), dtype=dtype))
        else:
            vals.append(torch.randn(c).to(dtype))
    return vals


def _reference(masks, values):
    mb = [m.cpu().to(torch.bool) for m in masks]
    vc = [v.cpu() for v in values]
    n = mb[0].numel()
    out = torch.empty(n, dtype=values[0].dtype)
    ks = [0] * len(mb)
    for i in range(n):
        for j, m in enumerate(mb):
            if bool(m[i]):
                out[i] = vc[j][ks[j]]
                ks[j] += 1
                break
        else:
            raise AssertionError(f"position {i} not covered")
    return out


def _run(mask_size, dtype, mask_dtype, mode, label_str):
    sk = (label_str, mask_size, str(dtype), str(mask_dtype), mode)
    masks = _make_masks(mask_size, mode, mask_dtype, sk)
    counts = _assign_counts(masks)
    values = _make_values(counts, dtype, sk)
    expected = _reference(masks, values)

    inputs = []
    for m, v in zip(masks, values):
        inputs.append(m.npu())
        inputs.append(v.npu())
    result = torch.ops.ops_multimodal_fusion.c2_boolean_unmask(inputs).cpu()

    assert result.dtype == dtype,\
        f"[{label_str}] dtype: got {result.dtype}, want {dtype}"
    assert tuple(result.shape) == (mask_size,),\
        f"[{label_str}] shape: got {tuple(result.shape)}, want {(mask_size,)}"
    # Verbatim gather: every dtype must match exactly.
    assert torch.equal(result, expected), (
        f"[{label_str}] bytewise mismatch (maskSize={mask_size}, dtype={dtype}, "
        f"mode={mode})"
    )


# ---------------------------------------------------------------------------
# Case matrix. Labels avoid the substrings "small" / "large" so pytest -k
# filters do not accidentally collide.
# label maskSize dtype mask_dtype mode
# ---------------------------------------------------------------------------

CASES_SMALL = [
    # vec_width-aligned fp32 (lanes 64), complementary masks.
    ("fp32_m64_complement", 64, torch.float32, torch.bool, "two_complement"),
    # vec_width-aligned fp16 (lanes 128), single all-true identity.
    ("fp16_m128_identity", 128, torch.float16, torch.bool, "single_all_true"),
    # int32 + uint8 mask, 3-way partition, exact-integer output.
    ("int32_m9_partition_uint8", 9, torch.int32, torch.uint8, "three_partition"),
    # minimal case.
    ("fp32_m1_identity", 1, torch.float32, torch.bool, "single_all_true"),
    # fp16 overlap (first-true-wins) crossing lane width.
    ("fp16_m129_overlap", 129, torch.float16, torch.bool, "overlap"),
    # int32 partial-core maskSize 7 < 8 cores complementary exact integer.
    ("int32_m7_complement", 7, torch.int32, torch.bool, "two_complement"),
    # fp32 non-aligned tail, 3-way partition.
    ("fp32_m65_partition", 65, torch.float32, torch.bool, "three_partition"),
]

CASES_LARGE = [
    ("fp32_m1024_complement", 1024, torch.float32, torch.bool, "two_complement"),
    ("fp16_m8192_overlap", 8192, torch.float16, torch.bool, "overlap"),
    ("int32_m1024_partition_uint8", 1024, torch.int32, torch.uint8, "three_partition"),
    ("fp32_m2048_partition", 2048, torch.float32, torch.bool, "three_partition"),
    ("fp16_m4096_complement", 4096, torch.float16, torch.bool, "two_complement"),
    ("int32_m8192_overlap", 8192, torch.int32, torch.bool, "overlap"),
    ("fp32_m1000_complement", 1000, torch.float32, torch.bool, "two_complement"),
    ("fp16_m777_partition_uint8", 777, torch.float16, torch.uint8, "three_partition"),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("label_str,mask_size,dtype,mask_dtype,mode", CASES_SMALL)
def test_c2_boolean_unmask_small(label_str, mask_size, dtype, mask_dtype, mode):
    _run(mask_size, dtype, mask_dtype, mode, label_str)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("label_str,mask_size,dtype,mask_dtype,mode", CASES_LARGE)
def test_c2_boolean_unmask_large(label_str, mask_size, dtype, mask_dtype, mode):
    _run(mask_size, dtype, mask_dtype, mode, label_str)


# ---------------------------------------------------------------------------
# Edge / negative-path validation.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_boolean_unmask_empty():
    """maskSize == 0 returns an empty output without launching the kernel."""
    m0 = torch.zeros(0, dtype=torch.bool).npu()
    v0 = torch.zeros(0, dtype=torch.float32).npu()
    out = torch.ops.ops_multimodal_fusion.c2_boolean_unmask([m0, v0]).cpu()
    assert tuple(out.shape) == (0,)
    assert out.dtype == torch.float32


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_boolean_unmask_values_bf16_rejected():
    """bf16 values are deferred on this platform."""
    m0 = torch.ones(4, dtype=torch.bool).npu()
    v0 = torch.randn(4, dtype=torch.bfloat16).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_boolean_unmask([m0, v0])


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_boolean_unmask_values_int64_rejected():
    """int64 values are deferred on this platform."""
    m0 = torch.ones(4, dtype=torch.bool).npu()
    v0 = torch.arange(4, dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_boolean_unmask([m0, v0])


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_boolean_unmask_mask_dtype_rejected():
    """mask dtype must be bool or uint8 (int32 rejected)."""
    m0 = torch.ones(4, dtype=torch.int32).npu()
    v0 = torch.randn(4, dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_boolean_unmask([m0, v0])


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_boolean_unmask_odd_inputs_rejected():
    """inputs list must have even length."""
    m0 = torch.ones(4, dtype=torch.bool).npu()
    v0 = torch.randn(4, dtype=torch.float32).npu()
    m1 = torch.zeros(4, dtype=torch.bool).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_boolean_unmask([m0, v0, m1])


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_boolean_unmask_too_few_inputs_rejected():
    """inputs list must have at least one mask values pair."""
    m0 = torch.ones(4, dtype=torch.bool).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_boolean_unmask([m0])


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_boolean_unmask_mask_ndim_rejected():
    """each mask must be 1-D."""
    m0 = torch.ones(2, 2, dtype=torch.bool).npu()
    v0 = torch.randn(4, dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_boolean_unmask([m0, v0])


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_boolean_unmask_mask_len_mismatch():
    """all masks must share the same length (== mask_0)."""
    m0 = torch.tensor([True, False, True, False], dtype=torch.bool).npu()
    v0 = torch.randn(2, dtype=torch.float32).npu()
    m1 = torch.tensor([False, True, False], dtype=torch.bool).npu()  # wrong length
    v1 = torch.randn(2, dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_boolean_unmask([m0, v0, m1, v1])


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_boolean_unmask_uncovered_position():
    """a position covered by no mask is rejected (mirrors CUDA assert)."""
    # maskSize 3; position 1 is false in both masks. values lengths sum to 3.
    m0 = torch.tensor([True, False, False], dtype=torch.bool).npu()
    v0 = torch.randn(2, dtype=torch.float32).npu()
    m1 = torch.tensor([False, False, True], dtype=torch.bool).npu()
    v1 = torch.randn(1, dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_boolean_unmask([m0, v0, m1, v1])


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_boolean_unmask_count_mismatch():
    """values_j length must equal the count assigned to mask j."""
    m0 = torch.ones(4, dtype=torch.bool).npu()       # assigned all 4 positions
    v0 = torch.randn(3, dtype=torch.float32).npu()   # but only 3 values
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_boolean_unmask([m0, v0])
