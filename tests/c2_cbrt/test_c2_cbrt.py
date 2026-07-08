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

"""Tests for ops_multimodal_fusion.c2_cbrt.

Caffe2-style elementwise real cube root: y cbrt x sign x |x|^ 1/3 .
Output dtype/shape == input. fp32 / fp16.

Golden reference: numpy.cbrt on an fp64 promotion, cast back to self dtype
 real cube root: cbrt -8 -2 cbrt 0 0 cbrt +/-inf +/-inf nan to nan .

Tolerance: cbrt is computed via exp(ln/3), so the "transcendental" tier --
fp32 rtol atol 1e-2 fp16 rtol atol 1e-1 equal_nan .

Test case counts:
  - test_c2_cbrt_small               : 7 cases
  - test_c2_cbrt_large               : 6 cases
  - test_c2_cbrt_fp16_large_random   : 1
  - test_c2_cbrt_interface_exist     : 1
  - test_c2_cbrt_empty               : 1
  - test_c2_cbrt_dtype_int32_rejected: 1
  - test_c2_cbrt_dtype_bf16_rejected : 1
  - Total                            : 18 cases
"""

import numpy as np
import torch
import torch_npu  # noqa: F401
import pytest
import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "c2_cbrt"):
    pytest.skip(
        "ops_multimodal_fusion.c2_cbrt not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_c2_cbrt_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "c2_cbrt"),\
        "The 'c2_cbrt' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


def _seed(sk):
    torch.manual_seed(abs(hash(repr(sk))) % 10_000_000)


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def _make_input(shape, dtype, dist, sk):
    n = _numel(shape)
    if dist == "mixed":
        _seed(sk)
        x = torch.rand(n) * 20.0 - 10.0  # -10 10 spans pos/neg
        x[::17] = 0.0                    # sprinkle exact zeros
        return x.reshape(shape).to(dtype)
    if dist == "cubes":
        base_k = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        vals = [float(base_k[i % len(base_k)] ** 3) for i in range(n)]  # perfect cubes
        return torch.tensor(vals, dtype=torch.float32).reshape(shape).to(dtype)
    if dist == "special":
        base = [0.0, 1.0, -1.0, 8.0, -8.0, 27.0, -27.0,
                float("inf"), float("-inf"), float("nan")]
        vals = [base[i % len(base)] for i in range(n)]
        return torch.tensor(vals, dtype=torch.float32).reshape(shape).to(dtype)
    raise ValueError(dist)


def _run(shape, dtype, dist, label_str):
    sk = (label_str, tuple(shape), str(dtype), dist)
    x = _make_input(shape, dtype, dist, sk)
    ref = torch.from_numpy(np.cbrt(x.double().cpu().numpy())).to(dtype)

    res = torch.ops.ops_multimodal_fusion.c2_cbrt(x.npu()).cpu()

    assert res.dtype == dtype, f"[{label_str}] dtype: got {res.dtype}, want {dtype}"
    assert tuple(res.shape) == tuple(shape),\
        f"[{label_str}] shape: got {tuple(res.shape)}, want {tuple(shape)}"
    rtol, atol = (1e-2, 1e-2) if dtype == torch.float32 else (1e-1, 1e-1)
    torch.testing.assert_close(res, ref, rtol=rtol, atol=atol, equal_nan=True)


CASES_SMALL = [
    ("fp32_n64_mixed", (64,), torch.float32, "mixed"),
    ("fp16_n128_mixed", (128,), torch.float16, "mixed"),
    ("fp32_n65_cubes", (65,), torch.float32, "cubes"),
    ("fp32_special", (10,), torch.float32, "special"),
    ("fp16_special", (10,), torch.float16, "special"),
    ("fp16_4x16_mixed", (4, 16), torch.float16, "mixed"),
    ("fp32_2x3x4_cubes", (2, 3, 4), torch.float32, "cubes"),
]

CASES_LARGE = [
    ("fp32_n4096_mixed", (4096,), torch.float32, "mixed"),
    ("fp16_n8192_mixed", (8192,), torch.float16, "mixed"),
    ("fp32_n129_mixed", (129,), torch.float32, "mixed"),
    ("fp16_n256_cubes", (256,), torch.float16, "cubes"),
    ("fp32_8x8x8_mixed", (8, 8, 8), torch.float32, "mixed"),
    ("fp16_n1000_mixed", (1000,), torch.float16, "mixed"),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("label_str,shape,dtype,dist", CASES_SMALL)
def test_c2_cbrt_small(label_str, shape, dtype, dist):
    _run(shape, dtype, dist, label_str)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("label_str,shape,dtype,dist", CASES_LARGE)
def test_c2_cbrt_large(label_str, shape, dtype, dist):
    _run(shape, dtype, dist, label_str)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_cbrt_fp16_large_random():
    """fp16 random large magnitudes vs np.cbrt(fp64).

    ST fp16 uses Reg-kernel-matched golden with -10 10 inputs; this covers
    the gap where the device exp(ln/3) path may drift for large |x| in fp16.
    """
    torch.manual_seed(20260625)
    shape = (4096,)
    x = (torch.rand(shape) * 2000.0 - 1000.0).to(torch.float16)  # -1000 1000
    x[::127] = 0.0
    ref = torch.from_numpy(np.cbrt(x.double().cpu().numpy())).to(torch.float16)
    res = torch.ops.ops_multimodal_fusion.c2_cbrt(x.npu()).cpu()
    assert res.dtype == torch.float16
    assert tuple(res.shape) == shape
    torch.testing.assert_close(res, ref, rtol=1e-1, atol=1e-1, equal_nan=True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_cbrt_empty():
    """Empty input -> empty output, dtype preserved."""
    out = torch.ops.ops_multimodal_fusion.c2_cbrt(torch.empty(0, dtype=torch.float32).npu()).cpu()
    assert tuple(out.shape) == (0,)
    assert out.dtype == torch.float32


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_cbrt_dtype_int32_rejected():
    """int32 self is deferred on this platform."""
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_cbrt(torch.arange(8, dtype=torch.int32).npu())


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_cbrt_dtype_bf16_rejected():
    """bf16 self is deferred on this platform."""
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_cbrt(torch.randn(8, dtype=torch.bfloat16).npu())
