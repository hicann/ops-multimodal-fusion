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

"""Tests for ops_multimodal_fusion.c2_bucketize.

Caffe2-style bucketize: given input `self` and a monotonically-increasing
`boundaries` float list, return int32 bucket indices
    out i # j : boundaries j < self i
 torch.bucketize self boundaries right False . Output shape self.shape.

Golden reference: dtype-faithful broadcast count (comparison done in self's
dtype, exactly as the kernel loads boundaries cast to self dtype and compares).

Test case counts:
  - test_c2_bucketize_small               : 7 cases
  - test_c2_bucketize_large               : 8 cases
  - test_c2_bucketize_interface_exist     : 1
  - test_c2_bucketize_empty_input         : 1
  - test_c2_bucketize_empty_boundaries    : 1
  - test_c2_bucketize_unsorted_rejected   : 1
  - test_c2_bucketize_dtype_int32_rejected: 1
  - test_c2_bucketize_dtype_bf16_rejected : 1
  - test_c2_bucketize_zerodim_rejected    : 1
  - test_c2_bucketize_m_exceeds_vecwidth  : 1
  - Total                                 : 23 cases
"""

import torch
import torch_npu  # noqa: F401
import pytest
import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "c2_bucketize"):
    pytest.skip(
        "ops_multimodal_fusion.c2_bucketize not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_c2_bucketize_interface_exist():
    """'ops_multimodal_fusion.c2_bucketize' is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "c2_bucketize"),\
        "The 'c2_bucketize' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


def _seed(seed_key):
    torch.manual_seed(abs(hash(repr(seed_key))) % 10_000_000)


def _make_bounds(m, dtype, sk):
    """Monotonically increasing boundaries of length M, as a python float list."""
    if m == 0:
        return []
    _seed(("bounds", sk))
    # Cumulative positive gaps -> strictly increasing; rounded to the dtype grid.
    gaps = torch.rand(m) + 0.1
    b = torch.cumsum(gaps, dim=0) - gaps[0]  # starts near 0
    b = b.to(dtype).to(torch.float64)
    # De-dup after dtype rounding to keep monotonic-nondecreasing (host check allows ==).
    return [float(v) for v in b.tolist()]


def _make_self(shape, dtype, dist, bounds, sk):
    _seed(("self", sk))
    n = 1
    for s in shape:
        n *= s
    if len(bounds) > 0:
        lo, hi = bounds[0], bounds[-1]
        span = max(hi - lo, 1.0)
    else:
        lo, hi, span = 0.0, 1.0, 1.0
    if dist == "span":
        x = torch.rand(n) * (span * 1.5) + (lo - span * 0.25)
    elif dist == "all_low":
        x = torch.full((n,), lo - span - 1.0)
    elif dist == "all_high":
        x = torch.full((n,), hi + span + 1.0)
    elif dist == "equal":
        # Mix exact boundary values with in-between values.
        x = torch.tensor([bounds[i % len(bounds)] if len(bounds) else 0.0
                          for i in range(n)], dtype=torch.float32)
    elif dist == "inf":
        x = torch.rand(n) * span + lo
        if n >= 2:
            x[0] = float("inf")
            x[1] = float("-inf")
    else:
        raise ValueError(dist)
    return x.reshape(shape).to(dtype)


def _reference(self_t, bounds_list):
    """out i # j : boundaries j < self i comparison in self's dtype."""
    s = self_t.cpu().reshape(-1)
    if len(bounds_list) == 0:
        return torch.zeros(self_t.shape, dtype=torch.int32)
    b = torch.tensor(bounds_list, dtype=self_t.dtype)
    cnt = (b.unsqueeze(0) < s.unsqueeze(1)).sum(dim=1).to(torch.int32)
    return cnt.reshape(self_t.shape)


def _run(shape, dtype, m, dist, label_str):
    sk = (label_str, tuple(shape), str(dtype), m, dist)
    bounds = _make_bounds(m, dtype, sk)
    self_t = _make_self(shape, dtype, dist, bounds, sk)
    expected = _reference(self_t, bounds)

    result = torch.ops.ops_multimodal_fusion.c2_bucketize(self_t.npu(), bounds).cpu()

    assert result.dtype == torch.int32,\
        f"[{label_str}] dtype: got {result.dtype}, want int32"
    assert tuple(result.shape) == tuple(shape),\
        f"[{label_str}] shape: got {tuple(result.shape)}, want {tuple(shape)}"
    assert torch.equal(result, expected), (
        f"[{label_str}] bucket index mismatch (shape={tuple(shape)}, dtype={dtype}, "
        f"M={m}, dist={dist})"
    )


# label shape dtype M dist
CASES_SMALL = [
    ("fp32_n64_m8_span", (64,), torch.float32, 8, "span"),
    ("fp16_n128_m8_span", (128,), torch.float16, 8, "span"),
    ("fp32_n8_m1_span", (8,), torch.float32, 1, "span"),
    ("fp32_n8_m0_empty", (8,), torch.float32, 0, "all_low"),
    ("fp16_n9_m64_equal", (9,), torch.float16, 64, "equal"),
    ("fp32_n7_m64_inf", (7,), torch.float32, 64, "inf"),
    ("fp32_4x4_m4_extremes", (4, 4), torch.float32, 4, "all_high"),
]

CASES_LARGE = [
    ("fp32_n4096_m32_span", (4096,), torch.float32, 32, "span"),
    ("fp16_n8192_m100_span", (8192,), torch.float16, 100, "span"),
    ("fp32_n1024_m64_span", (1024,), torch.float32, 64, "span"),
    ("fp16_n2048_m128_span", (2048,), torch.float16, 128, "span"),
    ("fp32_2x3x4_m16_equal", (2, 3, 4), torch.float32, 16, "equal"),
    ("fp16_n4096_m32_inf", (4096,), torch.float16, 32, "inf"),
    ("fp32_n9_m8_allhigh", (9,), torch.float32, 8, "all_high"),
    ("fp16_16x16_m50_span", (16, 16), torch.float16, 50, "span"),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("label_str,shape,dtype,m,dist", CASES_SMALL)
def test_c2_bucketize_small(label_str, shape, dtype, m, dist):
    _run(shape, dtype, m, dist, label_str)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("label_str,shape,dtype,m,dist", CASES_LARGE)
def test_c2_bucketize_large(label_str, shape, dtype, m, dist):
    _run(shape, dtype, m, dist, label_str)


# ---------------------------------------------------------------------------
# Edge / negative-path validation.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_bucketize_empty_input():
    """Empty self -> empty int32 output."""
    out = torch.ops.ops_multimodal_fusion.c2_bucketize(torch.empty(0).npu(), [1.0, 2.0]).cpu()
    assert tuple(out.shape) == (0,)
    assert out.dtype == torch.int32


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_bucketize_empty_boundaries():
    """Empty boundaries -> every element falls in bucket 0."""
    self_t = torch.randn(8)
    out = torch.ops.ops_multimodal_fusion.c2_bucketize(self_t.npu(), []).cpu()
    assert torch.equal(out, torch.zeros(8, dtype=torch.int32))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_bucketize_unsorted_rejected():
    """Non-monotonic boundaries are rejected."""
    self_t = torch.randn(8).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_bucketize(self_t, [1.0, 3.0, 2.0])


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_bucketize_dtype_int32_rejected():
    """int32 self is deferred on this platform."""
    self_t = torch.arange(8, dtype=torch.int32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_bucketize(self_t, [1.0, 2.0, 3.0])


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_bucketize_dtype_bf16_rejected():
    """bf16 self is deferred on this platform."""
    self_t = torch.randn(8, dtype=torch.bfloat16).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_bucketize(self_t, [1.0, 2.0, 3.0])


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_bucketize_zerodim_rejected():
    """0-D self is rejected (Caffe2 requires dim >= 1)."""
    self_t = torch.tensor(1.5).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_bucketize(self_t, [1.0, 2.0])


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_bucketize_m_exceeds_vecwidth():
    """M > vec_width fp32 64 is rejected in this version chunked deferred ."""
    self_t = torch.randn(8).npu()
    bounds = [float(i) for i in range(65)]  # 65 > 64
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_bucketize(self_t, bounds)
