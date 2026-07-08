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

"""Tests for ops_multimodal_fusion.c2_affine_channel.

Caffe2-style per-channel affine transformation in the forward NCHW layout.
Each output element is the input scaled by its channel factor and then
shifted by its channel bias.

Test case counts:
  - test_c2_affine_channel_small        : 7 cases (one shape per fast geometry)
  - test_c2_affine_channel_large        : 11 cases at ResNet scale with three
                                                    and five dim variants and
                                                    special value scenarios
  - test_c2_affine_channel_interface_exist : 1
  - test_c2_affine_channel_invalid_ndim   : 1
  - test_c2_affine_channel_scale_shape    : 1
  - test_c2_affine_channel_dtype_mismatch : 1
  - test_c2_affine_channel_bf16_rejected  : 1
  - test_c2_affine_channel_int64_rejected : 1
  - Total                                : 24 cases
"""

import torch
import torch_npu  # noqa: F401
import pytest
import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "c2_affine_channel"):
    pytest.skip(
        "ops_multimodal_fusion.c2_affine_channel not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_c2_affine_channel_interface_exist():
    """'ops_multimodal_fusion.c2_affine_channel' is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "c2_affine_channel"),\
        "The 'c2_affine_channel' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Input generators. All deterministic via seed_key.
# ---------------------------------------------------------------------------


def _seed(seed_key):
    torch.manual_seed(abs(hash(repr(seed_key))) % 10_000_000)


def _gen_randn(shape, dtype, sk):
    """Random X with randn scale and bias, the common case."""
    _seed(sk)
    c = shape[1]
    x = torch.randn(shape).to(dtype)
    scale = torch.randn(c).to(dtype)
    bias = torch.randn(c).to(dtype)
    return x, scale, bias


def _gen_identity(shape, dtype, sk):
    """scale 1 bias 0; output must equal X bytewise."""
    _seed(sk)
    c = shape[1]
    x = torch.randn(shape).to(dtype)
    scale = torch.ones(c, dtype=dtype)
    bias = torch.zeros(c, dtype=dtype)
    return x, scale, bias


def _gen_scale_zero(shape, dtype, sk):
    """scale 0; output must equal bias broadcast over N C ."""
    _seed(sk)
    c = shape[1]
    x = torch.randn(shape).to(dtype)
    scale = torch.zeros(c, dtype=dtype)
    bias = torch.randn(c).to(dtype)
    return x, scale, bias


_GENERATORS = {
    "randn": _gen_randn,
    "identity": _gen_identity,
    "scale0": _gen_scale_zero,
}


def _gen_input(gen_name, shape, dtype, seed_key):
    return _GENERATORS[gen_name](shape, dtype, seed_key)


# ---------------------------------------------------------------------------
# Reference: PyTorch broadcast matching caffe2 math::AffineChannel NCHW.
# ---------------------------------------------------------------------------


def _reference(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    c = x.size(1)
    view_shape = [1, c] + [1] * (x.dim() - 2)
    return x.cpu() * scale.cpu().view(view_shape) + bias.cpu().view(view_shape)


def _run(case, tensors):
    label_str, _shape, _dtype, _gen_name, rtol, atol, expect_equal = case
    x_cpu, scale_cpu, bias_cpu = tensors
    expected = _reference(x_cpu, scale_cpu, bias_cpu)
    result_npu = torch.ops.ops_multimodal_fusion.c2_affine_channel(
        x_cpu.npu(), scale_cpu.npu(), bias_cpu.npu()
    )
    result = result_npu.cpu()

    assert result.dtype == x_cpu.dtype,\
        f"[{label_str}] dtype: got {result.dtype}, want {x_cpu.dtype}"
    assert result.shape == x_cpu.shape,\
        f"[{label_str}] shape: got {tuple(result.shape)}, want {tuple(x_cpu.shape)}"

    if expect_equal:
        assert torch.equal(result, expected), (
            f"[{label_str}] bytewise mismatch (shape={tuple(x_cpu.shape)}, "
            f"dtype={x_cpu.dtype})"
        )
    else:
        if not torch.allclose(result, expected, rtol=rtol, atol=atol):
            diff = (result.to(torch.float32) - expected.to(torch.float32)).abs()
            raise AssertionError(
                f"[{label_str}] allclose failed (shape={tuple(x_cpu.shape)}, "
                f"dtype={x_cpu.dtype}, rtol={rtol}, atol={atol})\n"
                f"max abs diff={diff.max().item():.6g} "
                f"at idx {torch.argmax(diff.flatten()).item()}"
            )


# ---------------------------------------------------------------------------
# Case matrix.
# Labels avoid the substrings "small" or "large" so the pytest keyword
# filters do not accidentally collide.
# ---------------------------------------------------------------------------

# Fast cases small UB footprint quick on simulator .
# label shape dtype gen_name rtol atol expect_equal
CASES_FAST = [
    ("nchw_basic", (1, 4, 7, 7), torch.float32, "randn", 1e-4, 1e-4, False),
    ("vec_minus", (2, 4, 1, 63), torch.float32, "randn", 1e-4, 1e-4, False),
    ("vec_plus", (2, 4, 1, 65), torch.float32, "randn", 1e-4, 1e-4, False),
    ("fp16_vec", (4, 8, 1, 128), torch.float16, "randn", 1e-3, 1e-3, False),
    ("c_one", (4, 1, 32, 32), torch.float32, "randn", 1e-4, 1e-4, False),
    ("identity", (2, 4, 8, 8), torch.float32, "identity", 0.0, 0.0, True),
    ("hw_one", (4, 16, 1, 1), torch.float32, "randn", 1e-4, 1e-4, False),
]

# Heavy cases with typical conv sizes, ndim variants and special values.
CASES_HEAVY = [
    ("resnet_mid", (8, 64, 56, 56), torch.float32, "randn", 1e-4, 1e-4, False),
    ("resnet_tail", (4, 256, 14, 14), torch.float32, "randn", 1e-4, 1e-4, False),
    ("resnet_deep", (2, 1024, 7, 7), torch.float32, "randn", 1e-4, 1e-4, False),
    ("fp16_chunked", (4, 8, 16, 16), torch.float16, "randn", 1e-3, 1e-3, False),
    ("ncl_3d", (2, 8, 100), torch.float32, "randn", 1e-4, 1e-4, False),
    ("ncdhw_5d", (1, 4, 8, 16, 16), torch.float32, "randn", 1e-4, 1e-4, False),
    ("scale_zero", (2, 4, 8, 8), torch.float32, "scale0", 0.0, 0.0, True),
    ("n_one_batch", (1, 64, 28, 28), torch.float32, "randn", 1e-4, 1e-4, False),
    ("c_three_rgb", (4, 3, 32, 32), torch.float32, "randn", 1e-4, 1e-4, False),
    ("large_hw", (1, 4, 112, 112), torch.float32, "randn", 1e-4, 1e-4, False),
    ("fp16_identity", (4, 8, 16, 16), torch.float16, "identity", 0.0, 0.0, True),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_FAST)
def test_c2_affine_channel_small(case):
    label_str, shape, dtype, gen_name = case[0], case[1], case[2], case[3]
    seed_key = ("fast", label_str, tuple(shape), str(dtype), gen_name)
    tensors = _gen_input(gen_name, shape, dtype, seed_key)
    _run(case, tensors)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_HEAVY)
def test_c2_affine_channel_large(case):
    label_str, shape, dtype, gen_name = case[0], case[1], case[2], case[3]
    seed_key = ("heavy", label_str, tuple(shape), str(dtype), gen_name)
    tensors = _gen_input(gen_name, shape, dtype, seed_key)
    _run(case, tensors)


# ---------------------------------------------------------------------------
# Negative-path tests: argument and dtype validation.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_affine_channel_invalid_ndim():
    """X must have at least 2 dims."""
    x = torch.randn(8).npu()
    scale = torch.randn(8).npu()
    bias = torch.randn(8).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_affine_channel(x, scale, bias)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_affine_channel_scale_shape():
    """The scale length must equal the channel dimension of X."""
    x = torch.randn(2, 4, 8, 8).npu()
    scale = torch.randn(8).npu()  # wrong C
    bias = torch.randn(4).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_affine_channel(x, scale, bias)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_affine_channel_dtype_mismatch():
    """scale dtype must match X dtype."""
    x = torch.randn(2, 4, 8, 8).npu()
    scale = torch.randn(4, dtype=torch.float16).npu()  # mismatch
    bias = torch.randn(4).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_affine_channel(x, scale, bias)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_affine_channel_bf16_rejected():
    """bf16 is deferred on this platform."""
    x = torch.randn(2, 4, 8, 8, dtype=torch.bfloat16).npu()
    scale = torch.randn(4, dtype=torch.bfloat16).npu()
    bias = torch.randn(4, dtype=torch.bfloat16).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_affine_channel(x, scale, bias)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_affine_channel_int64_rejected():
    """int types are not supported."""
    x = torch.randint(0, 10, (2, 4, 8, 8), dtype=torch.int64).npu()
    scale = torch.randint(0, 10, (4,), dtype=torch.int64).npu()
    bias = torch.randint(0, 10, (4,), dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_affine_channel(x, scale, bias)
