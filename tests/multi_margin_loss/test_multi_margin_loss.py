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


if not hasattr(torch.ops.ops_multimodal_fusion, "multi_margin_loss"):
    pytest.skip(
        "ops_multimodal_fusion.multi_margin_loss not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float16, torch.float32]
REDUCTIONS = [(0, "none"), (1, "mean"), (2, "sum")]
P_VALUES = [1, 2]
_REDUCTION_NAME = {0: "none", 1: "mean", 2: "sum"}

# Loss hyper-parameters grouped so the reference helper takes a single config
# argument instead of a long positional list.
LossCfg = namedtuple("LossCfg", "p margin weight reduction_name")


def _make_input(shape, dtype, seed, low=-2.0, high=2.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(*shape, dtype=torch.float32).uniform_(low, high, generator=g)
    if x.numel() > 8:
        flat = x.flatten()
        flat[1::7] = 0.0
        flat[3::11] = -1.25
        flat[5::13] = 1.75
    return x.to(dtype)


def _make_weight(classes, dtype):
    return torch.linspace(0.25, 2.25, steps=classes, dtype=torch.float32).to(dtype)


def _reference(x, target, cfg):
    x_ref = x.float() if x.dtype == torch.float16 else x
    weight_ref = None
    if cfg.weight is not None:
        weight_ref = cfg.weight.float() if cfg.weight.dtype == torch.float16 else cfg.weight
    return F.multi_margin_loss(
        x_ref, target, p=cfg.p, margin=cfg.margin, weight=weight_ref,
        reduction=cfg.reduction_name
    ).to(x.dtype)


def _assert_close(actual, expected, label):
    actual = actual.cpu()
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    rtol, atol = (8e-3, 8e-3) if expected.dtype == torch.float16 else (1e-5, 1e-5)
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"{label}: max_abs={(actual - expected).abs().max().item() if actual.numel() else 0.0} "
        f"actual={actual.flatten()[:8]} expected={expected.flatten()[:8]}"
    )


def test_multi_margin_loss_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "multi_margin_loss")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("p", P_VALUES)
@pytest.mark.parametrize("reduction,reduction_name", REDUCTIONS)
def test_multi_margin_loss_1d(dtype, p, reduction, reduction_name):
    x = torch.tensor([0.2, 1.0, -0.4, 0.7, -1.5], dtype=dtype)
    target = torch.tensor(1, dtype=torch.long)

    actual = torch.ops.ops_multimodal_fusion.multi_margin_loss(
        x.npu(), target.npu(), p, 1.0, None, reduction
    )
    expected = _reference(x, target, LossCfg(p, 1.0, None, reduction_name))

    _assert_close(actual, expected, f"1d dtype={dtype} p={p} reduction={reduction_name}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("p", P_VALUES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("with_weight", [False, True])
@pytest.mark.parametrize("shape", [(1, 3), (4, 7), (2, 17)])
def test_multi_margin_loss_2d(dtype, p, reduction, with_weight, shape):
    reduction_name = _REDUCTION_NAME[reduction]
    n, classes = shape
    x = _make_input(shape, dtype, seed=n * 100 + classes * 17 + p)
    target = (torch.arange(n, dtype=torch.long) * 3 + 1) % classes
    weight = _make_weight(classes, dtype) if with_weight else None

    actual = torch.ops.ops_multimodal_fusion.multi_margin_loss(
        x.npu(),
        target.npu(),
        p,
        0.75,
        None if weight is None else weight.npu(),
        reduction,
    )
    expected = _reference(x, target, LossCfg(p, 0.75, weight, reduction_name))

    _assert_close(
        actual,
        expected,
        f"2d dtype={dtype} p={p} reduction={reduction_name} "
        f"weight={with_weight} shape={shape}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("p", P_VALUES)
@pytest.mark.parametrize("reduction,reduction_name", REDUCTIONS)
@pytest.mark.parametrize("classes", [64, 65, 128, 129])
def test_multi_margin_loss_multi_reg_chunk_tail(dtype, p, reduction, reduction_name, classes):
    x = _make_input((2, classes), dtype, seed=classes * 991 + p, low=-1.0, high=1.0)
    target = torch.tensor([0, classes - 1], dtype=torch.long)
    weight = _make_weight(classes, dtype)

    actual = torch.ops.ops_multimodal_fusion.multi_margin_loss(
        x.npu(), target.npu(), p, 1.25, weight.npu(), reduction
    )
    expected = _reference(x, target, LossCfg(p, 1.25, weight, reduction_name))

    _assert_close(
        actual,
        expected,
        f"multi-chunk dtype={dtype} p={p} reduction={reduction_name} C={classes}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("p", P_VALUES)
@pytest.mark.parametrize("reduction,reduction_name", REDUCTIONS)
def test_multi_margin_loss_non_contiguous(dtype, p, reduction, reduction_name):
    padded = torch.empty((3, 12), dtype=torch.float32)
    values = _make_input((3, 6), dtype, seed=888).float()
    padded[:, 0::2] = values
    padded[:, 1::2] = -9.0
    x = torch.as_strided(padded.to(dtype), size=(3, 6), stride=(12, 2))

    target_padded = torch.tensor([0, 4, 2, 5, 1, 3], dtype=torch.long)
    target = torch.as_strided(target_padded, size=(3,), stride=(2,))

    weight_padded = torch.empty(12, dtype=torch.float32)
    weight_values = _make_weight(6, dtype).float()
    weight_padded[0::2] = weight_values
    weight_padded[1::2] = -7.0
    weight = torch.as_strided(weight_padded.to(dtype), size=(6,), stride=(2,))

    assert not x.is_contiguous()
    assert not target.is_contiguous()
    assert not weight.is_contiguous()

    x_npu = torch.as_strided(padded.to(dtype).npu(), size=(3, 6), stride=(12, 2))
    target_npu = torch.as_strided(target_padded.npu(), size=(3,), stride=(2,))
    weight_npu = torch.as_strided(weight_padded.to(dtype).npu(), size=(6,), stride=(2,))

    actual = torch.ops.ops_multimodal_fusion.multi_margin_loss(
        x_npu, target_npu, p, 0.5, weight_npu, reduction
    )
    expected = _reference(x, target, LossCfg(p, 0.5, weight, reduction_name))

    _assert_close(
        actual, expected,
        f"non-contiguous dtype={dtype} p={p} reduction={reduction_name}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_multi_margin_loss_empty_batch(dtype, reduction):
    x = torch.empty((0, 5), dtype=dtype)
    target = torch.empty((0,), dtype=torch.long)

    actual = torch.ops.ops_multimodal_fusion.multi_margin_loss(x.npu(), target.npu(), 1, 1.0, None, reduction)
    actual_cpu = actual.cpu()
    assert actual_cpu.dtype == dtype
    if reduction == 0:
        assert actual_cpu.shape == (0,)
    else:
        assert actual_cpu.shape == ()
        assert torch.equal(actual_cpu, torch.tensor(0.0, dtype=dtype))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_multi_margin_loss_invalid_p():
    x = torch.randn(2, 3)
    target = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.multi_margin_loss(x.npu(), target.npu(), 3, 1.0, None, 1)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_multi_margin_loss_invalid_fractional_p():
    x = torch.randn(2, 3)
    target = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises((RuntimeError, ValueError)):
        torch.ops.ops_multimodal_fusion.multi_margin_loss(x.npu(), target.npu(), 1.5, 1.0, None, 1)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_multi_margin_loss_invalid_reduction():
    x = torch.randn(2, 3)
    target = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.multi_margin_loss(x.npu(), target.npu(), 1, 1.0, None, 3)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_multi_margin_loss_invalid_target_dtype():
    x = torch.randn(2, 3)
    target = torch.tensor([0, 1], dtype=torch.int32)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.multi_margin_loss(x.npu(), target.npu(), 1, 1.0, None, 1)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_multi_margin_loss_invalid_target_shape_for_1d():
    x = torch.randn(3)
    target = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.multi_margin_loss(x.npu(), target.npu(), 1, 1.0, None, 1)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_multi_margin_loss_invalid_target_shape_for_2d():
    x = torch.randn(2, 3)
    target = torch.tensor([[0], [1]], dtype=torch.long)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.multi_margin_loss(x.npu(), target.npu(), 1, 1.0, None, 1)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_multi_margin_loss_invalid_weight_shape():
    x = torch.randn(2, 3)
    target = torch.tensor([0, 1], dtype=torch.long)
    weight = torch.ones(4)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.multi_margin_loss(x.npu(), target.npu(), 1, 1.0, weight.npu(), 1)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_multi_margin_loss_invalid_weight_dtype():
    x = torch.randn(2, 3)
    target = torch.tensor([0, 1], dtype=torch.long)
    weight = torch.ones(3, dtype=torch.float16)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.multi_margin_loss(x.npu(), target.npu(), 1, 1.0, weight.npu(), 1)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_multi_margin_loss_invalid_input_ndim():
    x = torch.randn(2, 3, 4)
    target = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.multi_margin_loss(x.npu(), target.npu(), 1, 1.0, None, 1)
