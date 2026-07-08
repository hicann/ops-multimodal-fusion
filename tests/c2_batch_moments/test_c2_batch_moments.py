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

"""Tests for ops_multimodal_fusion.c2_batch_moments.

Caffe2-style per-channel first and second moments in the forward NCHW layout.
The first moment is the mean of the channel values; the second moment is the
mean of the squared channel values, the raw expected square rather than the
centred variance.

Test case counts:
  - test_c2_batch_moments_small       : 7 cases
  - test_c2_batch_moments_large       : 12 cases
  - test_c2_batch_moments_interface_exist : 1
  - test_c2_batch_moments_invalid_ndim    : 1
  - test_c2_batch_moments_bf16_rejected   : 1
  - test_c2_batch_moments_int32_rejected  : 1
  - test_c2_batch_moments_int64_rejected  : 1
  - Total                             : 24 cases
"""

import torch
import torch_npu  # noqa: F401
import pytest
import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "c2_batch_moments"):
    pytest.skip(
        "ops_multimodal_fusion.c2_batch_moments not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_c2_batch_moments_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "c2_batch_moments"),\
        "The 'c2_batch_moments' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Input generators.
# ---------------------------------------------------------------------------


def _seed(seed_key):
    torch.manual_seed(abs(hash(repr(seed_key))) % 10_000_000)


def _gen_randn(shape, dtype, sk):
    _seed(sk)
    return torch.randn(shape).to(dtype)


def _gen_zeros(shape, dtype, sk):
    _seed(sk)
    return torch.zeros(shape, dtype=dtype)


def _gen_ones(shape, dtype, sk):
    _seed(sk)
    return torch.ones(shape, dtype=dtype)


def _gen_full(shape, dtype, sk, value=2.0):
    _seed(sk)
    return torch.full(shape, float(value), dtype=dtype)


def _gen_centered(shape, dtype, sk):
    """Subtract the mean from randn so the tensor straddles zero.

    Useful to verify that the second moment stays non-negative even when
    the inputs are negative.
    """
    _seed(sk)
    x = torch.randn(shape)
    return (x - x.mean()).to(dtype)


_GENERATORS = {
    "randn": _gen_randn,
    "zeros": _gen_zeros,
    "ones": _gen_ones,
    "full2": lambda s, d, sk: _gen_full(s, d, sk, value=2.0),
    "centered": _gen_centered,
}


def _gen_input(gen_name, shape, dtype, seed_key):
    return _GENERATORS[gen_name](shape, dtype, seed_key)


# ---------------------------------------------------------------------------
# Reference: PyTorch matching caffe2 BatchMomentsCUDAKernel NCHW.
# ---------------------------------------------------------------------------


def _reference(x: torch.Tensor) -> tuple:
    x_cpu = x.cpu()
    n, c = x_cpu.size(0), x_cpu.size(1)
    hxw = x_cpu.numel() // (n * c)
    x_flat = x_cpu.transpose(0, 1).contiguous().view(c, -1)
    mu = x_flat.mean(dim=1)
    var = (x_flat * x_flat).mean(dim=1)  # raw mean of squares, not centred variance
    return mu, var


def _run(case, x_cpu):
    (label_str, _shape, _dtype, _gen_name,
     rtol, atol, expect_equal, extra_var_nonneg) = case
    mu_ref, var_ref = _reference(x_cpu)
    mu_npu, var_npu = torch.ops.ops_multimodal_fusion.c2_batch_moments(x_cpu.npu())
    mu_res, var_res = mu_npu.cpu(), var_npu.cpu()

    assert mu_res.dtype == x_cpu.dtype,\
        f"[{label_str}] mu dtype: got {mu_res.dtype}, want {x_cpu.dtype}"
    assert var_res.dtype == x_cpu.dtype,\
        f"[{label_str}] var dtype: got {var_res.dtype}, want {x_cpu.dtype}"
    c = x_cpu.size(1)
    assert tuple(mu_res.shape) == (c,),\
        f"[{label_str}] mu shape: got {tuple(mu_res.shape)}, want ({c},)"
    assert tuple(var_res.shape) == (c,),\
        f"[{label_str}] var shape: got {tuple(var_res.shape)}, want ({c},)"

    if extra_var_nonneg:
        assert (var_res.to(torch.float32) >= -atol).all(),\
            f"[{label_str}] var has negative entry: min={var_res.min().item():.6g}"

    if expect_equal:
        assert torch.equal(mu_res, mu_ref),\
            f"[{label_str}] mu bytewise mismatch (shape={tuple(x_cpu.shape)})"
        assert torch.equal(var_res, var_ref),\
            f"[{label_str}] var bytewise mismatch (shape={tuple(x_cpu.shape)})"
    else:
        if not torch.allclose(mu_res, mu_ref, rtol=rtol, atol=atol):
            d = (mu_res.to(torch.float32) - mu_ref.to(torch.float32)).abs()
            raise AssertionError(
                f"[{label_str}] mu allclose failed (shape={tuple(x_cpu.shape)}, "
                f"dtype={x_cpu.dtype}, rtol={rtol}, atol={atol})\n"
                f"max abs diff={d.max().item():.6g} at c={torch.argmax(d).item()}"
            )
        if not torch.allclose(var_res, var_ref, rtol=rtol, atol=atol):
            d = (var_res.to(torch.float32) - var_ref.to(torch.float32)).abs()
            raise AssertionError(
                f"[{label_str}] var allclose failed (shape={tuple(x_cpu.shape)}, "
                f"dtype={x_cpu.dtype}, rtol={rtol}, atol={atol})\n"
                f"max abs diff={d.max().item():.6g} at c={torch.argmax(d).item()}"
            )


# ---------------------------------------------------------------------------
# Case matrix.
# Labels avoid the substrings "small" or "large" so the pytest keyword
# filters do not accidentally collide.
# ---------------------------------------------------------------------------

# Fast cases small UB footprint quick on simulator .
# label shape dtype gen_name rtol atol expect_equal extra_var_nonneg
CASES_FAST = [
    ("nchw_basic", (1, 4, 7, 7), torch.float32, "randn", 1e-3, 1e-3, False, False),
    ("vec_minus", (2, 4, 1, 63), torch.float32, "randn", 1e-3, 1e-3, False, False),
    ("vec_plus", (2, 4, 1, 65), torch.float32, "randn", 1e-3, 1e-3, False, False),
    ("fp16_vec", (4, 8, 1, 128), torch.float16, "randn", 1e-2, 1e-2, False, False),
    ("c_one", (4, 1, 32, 32), torch.float32, "randn", 1e-3, 1e-3, False, False),
    ("zeros", (2, 4, 8, 8), torch.float32, "zeros", 0.0, 0.0, True, False),
    ("ones", (2, 4, 8, 8), torch.float32, "ones", 0.0, 0.0, True, False),
]

# Heavy cases.
CASES_HEAVY = [
    ("resnet_mid", (8, 64, 28, 28), torch.float32, "randn", 1e-3, 1e-3, False, False),
    ("resnet_tail", (4, 1024, 7, 7), torch.float32, "randn", 1e-3, 1e-3, False, False),
    ("fp16_chunked", (4, 8, 16, 16), torch.float16, "randn", 1e-2, 1e-2, False, False),
    ("hw_one", (8, 16, 1, 1), torch.float32, "randn", 1e-3, 1e-3, False, False),
    ("n_one_batch", (1, 64, 28, 28), torch.float32, "randn", 1e-3, 1e-3, False, False),
    ("ncl_3d", (2, 8, 100), torch.float32, "randn", 1e-3, 1e-3, False, False),
    ("ncdhw_5d", (1, 4, 8, 16, 16), torch.float32, "randn", 1e-3, 1e-3, False, False),
    ("full_two", (2, 4, 8, 8), torch.float32, "full2", 1e-5, 1e-5, False, False),
    ("centered_neg", (2, 4, 16, 16), torch.float32, "centered", 1e-3, 1e-3, False, True),
    ("deep_n_acc", (128, 4, 8, 8), torch.float32, "randn", 1e-3, 1e-3, False, False),
    ("saturate_c", (4, 256, 16, 16), torch.float32, "randn", 1e-3, 1e-3, False, False),
    ("fp16_zeros", (4, 8, 16, 16), torch.float16, "zeros", 0.0, 0.0, True, False),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_FAST)
def test_c2_batch_moments_small(case):
    label_str, shape, dtype, gen_name = case[0], case[1], case[2], case[3]
    seed_key = ("fast", label_str, tuple(shape), str(dtype), gen_name)
    x = _gen_input(gen_name, shape, dtype, seed_key)
    _run(case, x)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_HEAVY)
def test_c2_batch_moments_large(case):
    label_str, shape, dtype, gen_name = case[0], case[1], case[2], case[3]
    seed_key = ("heavy", label_str, tuple(shape), str(dtype), gen_name)
    x = _gen_input(gen_name, shape, dtype, seed_key)
    _run(case, x)


# ---------------------------------------------------------------------------
# Negative-path tests: argument and dtype validation.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_batch_moments_invalid_ndim():
    """X must have at least 2 dims."""
    x = torch.randn(8).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_batch_moments(x)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_batch_moments_bf16_rejected():
    x = torch.randn(2, 4, 8, 8, dtype=torch.bfloat16).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_batch_moments(x)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_batch_moments_int32_rejected():
    x = torch.randint(0, 100, (2, 4, 8, 8), dtype=torch.int32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_batch_moments(x)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_batch_moments_int64_rejected():
    x = torch.randint(0, 100, (2, 4, 8, 8), dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_batch_moments(x)
