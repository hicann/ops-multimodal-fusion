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

from collections import namedtuple

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "depthwise_conv3d"):
    pytest.skip(
        "ops_multimodal_fusion.depthwise_conv3d not registered for current NPU_ARCH; "
        "skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float16, torch.float32]

# Convolution spatial config grouped so the reference/custom helpers take a
# single argument instead of three separate stride/padding/dilation triples.
ConvCfg = namedtuple("ConvCfg", "stride padding dilation")
_DEFAULT_CFG = ConvCfg((1, 1, 1), (0, 0, 0), (1, 1, 1))


def _make_tensor(shape, dtype, seed=0, low=-1.0, high=1.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(*shape, dtype=torch.float32).uniform_(low, high, generator=g)
    if x.numel() > 16:
        flat = x.flatten()
        flat[1::7] = 0.0
        flat[3::11] = -0.75
        flat[5::13] = 0.5
    return x.to(dtype)


def _reference(x, weight, bias, cfg):
    x_ref = x.float() if x.dtype == torch.float16 else x
    w_ref = weight.float() if weight.dtype == torch.float16 else weight
    b_ref = None if bias is None else (bias.float() if bias.dtype == torch.float16 else bias)
    y = F.conv3d(
        x_ref,
        w_ref,
        b_ref,
        stride=cfg.stride,
        padding=cfg.padding,
        dilation=cfg.dilation,
        groups=x.size(1),
    )
    return y.to(x.dtype)


def _custom(x, weight, bias=None, cfg=_DEFAULT_CFG):
    bias_npu = None if bias is None else bias.npu()
    return torch.ops.ops_multimodal_fusion.depthwise_conv3d(
        x.npu(), weight.npu(), bias_npu,
        list(cfg.stride), list(cfg.padding), list(cfg.dilation)
    ).cpu()


def _assert_close(actual, expected, label):
    assert actual.dtype == expected.dtype, (
        f"{label}: dtype mismatch {actual.dtype} vs {expected.dtype}"
    )
    assert actual.shape == expected.shape, (
        f"{label}: shape mismatch {actual.shape} vs {expected.shape}"
    )
    rtol, atol = (1.5e-2, 1.5e-2) if expected.dtype == torch.float16 else (1e-4, 1e-4)
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"{label}: max_abs="
        f"{(actual - expected).abs().max().item() if actual.numel() else 0.0} "
        f"actual={actual.flatten()[:8]} expected={expected.flatten()[:8]}"
    )


def test_depthwise_conv3d_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "depthwise_conv3d")


# One parametrized conv case, bundled so the test takes a single argument
# instead of a long positional list.
Case = namedtuple("Case", "shape multiplier kernel_size stride padding dilation with_bias")

CASES = [
    ((1, 1, 3, 3, 3), 1, (1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1), False),
    ((1, 2, 4, 4, 4), 1, (2, 2, 2), (1, 1, 1), (0, 0, 0), (1, 1, 1), False),
    ((1, 2, 5, 4, 6), 2, (2, 1, 3), (1, 2, 1), (0, 0, 1), (1, 1, 1), True),
    ((2, 2, 5, 5, 5), 1, (2, 2, 2), (2, 1, 2), (1, 0, 1), (1, 1, 1), True),
    ((1, 3, 5, 5, 6), 1, (2, 2, 2), (1, 1, 1), (0, 0, 0), (2, 1, 2), False),
    ((2, 2, 4, 5, 6), 2, (1, 2, 3), (1, 1, 2), (0, 1, 1), (1, 1, 1), True),
    ((1, 2, 6, 5, 7), 3, (2, 1, 3), (2, 1, 1), (1, 0, 2), (2, 1, 1), True),
    ((1, 1, 4, 4, 4), 3, (1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1), True),
    ((1, 2, 3, 4, 5), 1, (3, 4, 5), (1, 1, 1), (0, 0, 0), (1, 1, 1), False),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", [Case(*c) for c in CASES])
def test_depthwise_conv3d_matches_torch(dtype, case):
    c_in = case.shape[1]
    c_out = c_in * case.multiplier
    seed = sum(case.shape) + sum(case.kernel_size) * 17 + case.multiplier * 31
    x = _make_tensor(case.shape, dtype, seed=seed)
    weight = _make_tensor((c_out, 1, *case.kernel_size), dtype, seed=seed + 1)
    bias = _make_tensor((c_out,), dtype, seed=seed + 2) if case.with_bias else None

    cfg = ConvCfg(case.stride, case.padding, case.dilation)
    actual = _custom(x, weight, bias, cfg)
    expected = _reference(x, weight, bias, cfg)
    _assert_close(
        actual,
        expected,
        f"dtype={dtype} shape={case.shape} multiplier={case.multiplier} k={case.kernel_size}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_depthwise_conv3d_known_values_channel_mapping(dtype):
    x = torch.arange(1, 1 + 2 * 3 * 3 * 3, dtype=torch.float32).reshape(1, 2, 3, 3, 3)
    x = (x * 0.1).to(dtype)
    weight = torch.tensor(
        [
            [[[[1.0]]]],
            [[[[2.0]]]],
            [[[[-1.0]]]],
            [[[[0.5]]]],
        ],
        dtype=dtype,
    )
    bias = torch.tensor([0.0, 1.0, -1.0, 0.25], dtype=dtype)
    actual = _custom(x, weight, bias)
    expected = _reference(x, weight, bias, ConvCfg((1, 1, 1), (0, 0, 0), (1, 1, 1)))
    _assert_close(actual, expected, f"known-channel-mapping dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_depthwise_conv3d_padding_zero_behavior(dtype):
    x = torch.ones((1, 1, 2, 2, 2), dtype=dtype)
    weight = torch.ones((1, 1, 2, 2, 2), dtype=dtype)
    actual = _custom(x, weight, None, ConvCfg((1, 1, 1), (1, 1, 1), (1, 1, 1)))
    expected = _reference(x, weight, None, ConvCfg((1, 1, 1), (1, 1, 1), (1, 1, 1)))
    _assert_close(actual, expected, f"padding-zero dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_depthwise_conv3d_fp16_wider_accumulation():
    x = _make_tensor((1, 1, 4, 4, 4), torch.float16, seed=101, low=-2.0, high=2.0)
    weight = _make_tensor((1, 1, 3, 3, 3), torch.float16, seed=102, low=-2.0, high=2.0)
    bias = _make_tensor((1,), torch.float16, seed=103, low=-1.0, high=1.0)
    actual = _custom(x, weight, bias, ConvCfg((1, 1, 1), (1, 1, 1), (1, 1, 1)))
    expected = _reference(x, weight, bias, ConvCfg((1, 1, 1), (1, 1, 1), (1, 1, 1)))
    _assert_close(actual, expected, "fp16-wide-accumulation")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_depthwise_conv3d_empty_batch():
    x = torch.empty((0, 2, 4, 4, 4), dtype=torch.float32)
    weight = torch.randn(2, 1, 2, 2, 2)
    actual = _custom(x, weight)
    assert actual.dtype == torch.float32
    assert actual.shape == (0, 2, 3, 3, 3)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_depthwise_conv3d_invalid_input_rank():
    x = torch.randn(1, 2, 4, 4)
    weight = torch.randn(2, 1, 2, 2, 2)
    with pytest.raises(RuntimeError):
        _custom(x, weight)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_depthwise_conv3d_invalid_weight_rank():
    x = torch.randn(1, 2, 4, 4, 4)
    weight = torch.randn(2, 1, 2, 2)
    with pytest.raises(RuntimeError):
        _custom(x, weight)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_depthwise_conv3d_invalid_weight_channel_dim():
    x = torch.randn(1, 2, 4, 4, 4)
    weight = torch.randn(2, 2, 2, 2, 2)
    with pytest.raises(RuntimeError):
        _custom(x, weight)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_depthwise_conv3d_invalid_channel_multiplier():
    x = torch.randn(1, 2, 4, 4, 4)
    weight = torch.randn(3, 1, 2, 2, 2)
    with pytest.raises(RuntimeError):
        _custom(x, weight)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_depthwise_conv3d_invalid_bias_shape():
    x = torch.randn(1, 2, 4, 4, 4)
    weight = torch.randn(2, 1, 2, 2, 2)
    bias = torch.randn(3)
    with pytest.raises(RuntimeError):
        _custom(x, weight, bias)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "stride,padding,dilation",
    [
        ((0, 1, 1), (0, 0, 0), (1, 1, 1)),
        ((1, 1), (0, 0, 0), (1, 1, 1)),
        ((1, 1, 1), (-1, 0, 0), (1, 1, 1)),
        ((1, 1, 1), (0, 0), (1, 1, 1)),
        ((1, 1, 1), (0, 0, 0), (0, 1, 1)),
        ((1, 1, 1), (0, 0, 0), (1, 1)),
    ],
)


def test_depthwise_conv3d_invalid_attributes(stride, padding, dilation):
    x = torch.randn(1, 2, 4, 4, 4)
    weight = torch.randn(2, 1, 2, 2, 2)
    with pytest.raises(RuntimeError):
        _custom(x, weight, cfg=ConvCfg(stride, padding, dilation))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_depthwise_conv3d_invalid_output_size():
    x = torch.randn(1, 1, 2, 2, 2)
    weight = torch.randn(1, 1, 3, 3, 3)
    with pytest.raises(RuntimeError):
        _custom(x, weight)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_depthwise_conv3d_invalid_dtype():
    x = torch.randint(0, 4, (1, 1, 4, 4, 4), dtype=torch.int32)
    weight = torch.randint(0, 4, (1, 1, 2, 2, 2), dtype=torch.int32)
    with pytest.raises(RuntimeError):
        _custom(x, weight)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "x_dtype,weight_dtype,bias_dtype",
    [
        (torch.float32, torch.float16, torch.float32),
        (torch.float32, torch.float32, torch.float16),
    ],
)


def test_depthwise_conv3d_dtype_mismatch(x_dtype, weight_dtype, bias_dtype):
    x = torch.randn(1, 1, 4, 4, 4, dtype=x_dtype)
    weight = torch.randn(1, 1, 2, 2, 2, dtype=weight_dtype)
    bias = torch.randn(1, dtype=bias_dtype)
    with pytest.raises(RuntimeError):
        _custom(x, weight, bias)
