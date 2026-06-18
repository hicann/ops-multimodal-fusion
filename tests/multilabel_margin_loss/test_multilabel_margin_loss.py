#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "multilabel_margin_loss"):
    pytest.skip(
        "ops_multimodal_fusion.multilabel_margin_loss not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float16, torch.float32]
REDUCTIONS = [(0, "none"), (1, "mean"), (2, "sum")]


def _target(rows, classes):
    target = torch.full((len(rows), classes), -1, dtype=torch.long)
    for n, labels in enumerate(rows):
        if labels:
            target[n, : len(labels)] = torch.tensor(labels, dtype=torch.long)
    return target


def _make_input(shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(*shape, dtype=torch.float32).uniform_(-2.5, 2.5, generator=g)
    if x.numel() > 6:
        flat = x.flatten()
        flat[1::7] = 0.0
        flat[3::11] = -1.25
        flat[5::13] = 2.0
    return x.to(dtype)


def _reference(x, target, reduction_name):
    ref_input = x.float() if x.dtype == torch.float16 else x
    return F.multilabel_margin_loss(ref_input, target, reduction=reduction_name).to(x.dtype)


def _assert_close(actual, expected, label):
    actual = actual.cpu()
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    rtol, atol = (5e-3, 5e-3) if expected.dtype == torch.float16 else (1e-5, 1e-5)
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"{label}: max_abs={(actual - expected).abs().max().item() if actual.numel() else 0.0} "
        f"actual={actual.flatten()[:8]} expected={expected.flatten()[:8]}"
    )


def test_multilabel_margin_loss_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "multilabel_margin_loss")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduction,reduction_name", REDUCTIONS)
def test_multilabel_margin_loss_1d(dtype, reduction, reduction_name):
    x = torch.tensor([0.4, -1.2, 2.0, 0.1, -0.7], dtype=dtype)
    target = torch.tensor([2, 0, -1, -1, -1], dtype=torch.long)

    actual = torch.ops.ops_multimodal_fusion.multilabel_margin_loss(x.npu(), target.npu(), reduction)
    expected = _reference(x, target, reduction_name)

    _assert_close(actual, expected, f"1d dtype={dtype} reduction={reduction_name}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduction,reduction_name", REDUCTIONS)
@pytest.mark.parametrize(
    "classes,rows",
    [
        (1, [[0], [-1], [0]]),
        (2, [[0], [1], [0, 1]]),
        (7, [[0, 3], [1], [], [2, 5, 6], [0, 1, 2, 3, 4, 5, 6]]),
        (17, [[0, 8, 16], [1, 2], [], [3, 4, 5, 6], [7]]),
    ],
)
def test_multilabel_margin_loss_2d_shapes(dtype, reduction, reduction_name, classes, rows):
    x = _make_input((len(rows), classes), dtype, seed=classes * 113 + len(rows))
    target = _target(rows, classes)

    actual = torch.ops.ops_multimodal_fusion.multilabel_margin_loss(x.npu(), target.npu(), reduction)
    expected = _reference(x, target, reduction_name)

    _assert_close(
        actual,
        expected,
        f"2d dtype={dtype} reduction={reduction_name} classes={classes} rows={rows}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_multilabel_margin_loss_all_negative_targets(dtype):
    x = _make_input((4, 5), dtype, seed=991)
    target = torch.full((4, 5), -1, dtype=torch.long)

    actual = torch.ops.ops_multimodal_fusion.multilabel_margin_loss(x.npu(), target.npu(), 0)
    expected = _reference(x, target, "none")

    _assert_close(actual, expected, f"all-negative dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_multilabel_margin_loss_all_labels(dtype):
    x = _make_input((3, 6), dtype, seed=445)
    target = torch.stack([torch.randperm(6) for _ in range(3)]).long()

    actual = torch.ops.ops_multimodal_fusion.multilabel_margin_loss(x.npu(), target.npu(), 0)
    expected = _reference(x, target, "none")

    _assert_close(actual, expected, f"all-labels dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduction,reduction_name", REDUCTIONS)
def test_multilabel_margin_loss_non_contiguous_input_and_target(dtype, reduction, reduction_name):
    padded = torch.empty((4, 12), dtype=torch.float32)
    values = _make_input((4, 6), dtype, seed=777).float()
    padded[:, 0::2] = values
    padded[:, 1::2] = -99.0
    x = torch.as_strided(padded.to(dtype), size=(4, 6), stride=(12, 2))

    target_padded = torch.full((4, 12), -1, dtype=torch.long)
    target_values = _target([[0, 2], [1], [], [3, 5]], 6)
    target_padded[:, 0::2] = target_values
    target = torch.as_strided(target_padded, size=(4, 6), stride=(12, 2))

    assert not x.is_contiguous()
    assert not target.is_contiguous()

    x_npu = torch.as_strided(padded.to(dtype).npu(), size=(4, 6), stride=(12, 2))
    target_npu = torch.as_strided(target_padded.npu(), size=(4, 6), stride=(12, 2))
    assert not x_npu.is_contiguous()
    assert not target_npu.is_contiguous()

    actual = torch.ops.ops_multimodal_fusion.multilabel_margin_loss(x_npu, target_npu, reduction)
    expected = _reference(x, target, reduction_name)

    _assert_close(actual, expected, f"non-contiguous dtype={dtype} reduction={reduction_name}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduction,reduction_name", REDUCTIONS)
@pytest.mark.parametrize("classes", [64, 65, 128, 129])
def test_multilabel_margin_loss_multi_reg_chunk_tail(dtype, reduction, reduction_name, classes):
    x = _make_input((2, classes), dtype, seed=classes * 1009)
    rows = [
        [0, classes // 2, classes - 1],
        [1, classes // 3],
    ]
    target = _target(rows, classes)

    actual = torch.ops.ops_multimodal_fusion.multilabel_margin_loss(x.npu(), target.npu(), reduction)
    expected = _reference(x, target, reduction_name)

    _assert_close(
        actual,
        expected,
        f"multi-chunk dtype={dtype} reduction={reduction_name} classes={classes}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_multilabel_margin_loss_empty_batch(dtype, reduction):
    x = torch.empty((0, 5), dtype=dtype)
    target = torch.empty((0, 5), dtype=torch.long)

    actual = torch.ops.ops_multimodal_fusion.multilabel_margin_loss(x.npu(), target.npu(), reduction)
    actual_cpu = actual.cpu()
    assert actual_cpu.dtype == dtype
    if reduction == 0:
        assert actual_cpu.shape == (0,)
    else:
        assert actual_cpu.shape == ()
        assert torch.equal(actual_cpu, torch.tensor(0.0, dtype=dtype))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_multilabel_margin_loss_invalid_reduction_cpu_dispatch():
    x = torch.randn(2, 3)
    target = _target([[0], [1]], 3)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.multilabel_margin_loss(x.npu(), target.npu(), 3)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_multilabel_margin_loss_invalid_target_dtype_cpu_dispatch():
    x = torch.randn(2, 3)
    target = _target([[0], [1]], 3).int()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.multilabel_margin_loss(x.npu(), target.npu(), 1)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_multilabel_margin_loss_invalid_shape_cpu_dispatch():
    x = torch.randn(2, 3)
    target = torch.full((2, 4), -1, dtype=torch.long)
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.multilabel_margin_loss(x.npu(), target.npu(), 1)
