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
"""Tests for the ops_multimodal_fusion.cdist_backward operator, the x1 gradient of torch.cdist."""

import math
from collections import namedtuple

import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "cdist_backward"):
    pytest.skip(
        "ops_multimodal_fusion.cdist_backward not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float32, torch.float16]

# Case bundle keeps the run helper at a single argument so codecheck stays within five parameters.
Case = namedtuple("Case", ["x1", "x2", "grad", "p", "dtype", "label"])


def _expanded_shapes(x1, x2):
    batch = torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2])
    return (*batch, x1.shape[-2], x1.shape[-1]), (*batch, x2.shape[-2], x2.shape[-1])


def _make_distance(x1_cpu, x2_cpu, p, dtype):
    x1_shape, x2_shape = _expanded_shapes(x1_cpu, x2_cpu)
    r1 = x1_shape[-2]
    r2 = x2_shape[-2]
    degenerate = 0 in x1_shape[:-2] or r1 == 0 or r2 == 0
    if degenerate or x1_shape[-1] == 0:
        return torch.zeros((*x1_shape[:-2], r1, r2), dtype=dtype)
    x1_work = x1_cpu.expand(x1_shape).to(torch.float32)
    x2_work = x2_cpu.expand(x2_shape).to(torch.float32)
    return torch.cdist(x1_work, x2_work, p=p).to(dtype)


def _reference_dx1(x1_cpu, x2_cpu, grad_cpu, p, dtype):
    x1_shape, x2_shape = _expanded_shapes(x1_cpu, x2_cpu)
    if p == 0 or 0 in x1_shape:
        return torch.zeros(x1_shape, dtype=dtype)
    if x2_shape[-2] == 0:
        return torch.zeros(x1_shape, dtype=dtype)

    x1_req = (
        x1_cpu.expand(x1_shape)
        .clone()
        .detach()
        .to(torch.float32)
        .requires_grad_(True)
    )
    x2_work = x2_cpu.expand(x2_shape).clone().detach().to(torch.float32)
    grad_work = grad_cpu.to(torch.float32)
    dist = torch.cdist(x1_req, x2_work, p=p)
    (dx1,) = torch.autograd.grad(dist, x1_req, grad_outputs=grad_work)
    return dx1.to(dtype)


def _custom_dx1(x1_cpu, x2_cpu, grad_cpu, p, cdist_cpu):
    return torch.ops.ops_multimodal_fusion.cdist_backward(
        grad_cpu.npu(), x1_cpu.npu(), x2_cpu.npu(), float(p), cdist_cpu.npu()
    ).cpu()


def _assert_close(actual, expected, dtype, label):
    assert actual.shape == expected.shape, (
        f"{label}: shape mismatch {actual.shape} vs {expected.shape}"
    )
    assert actual.dtype == expected.dtype, (
        f"{label}: dtype mismatch {actual.dtype} vs {expected.dtype}"
    )
    assert torch.isfinite(actual.float()).all(), f"{label}: actual contains NaN/Inf"
    if dtype == torch.float16:
        assert torch.allclose(actual.float(), expected.float(), rtol=6e-2, atol=6e-2), (
            f"{label}: fp16 values mismatch actual={actual} expected={expected}"
        )
    else:
        assert torch.allclose(actual, expected, rtol=2e-4, atol=2e-4), (
            f"{label}: fp32 values mismatch actual={actual} expected={expected}"
        )


def _run_case(case):
    x1_typed = case.x1.to(case.dtype)
    x2_typed = case.x2.to(case.dtype)
    grad_typed = case.grad.to(case.dtype)
    cdist_cpu = _make_distance(x1_typed, x2_typed, case.p, case.dtype)
    expected = _reference_dx1(x1_typed, x2_typed, grad_typed, case.p, case.dtype)
    actual = _custom_dx1(x1_typed, x2_typed, grad_typed, case.p, cdist_cpu)
    _assert_close(actual, expected, case.dtype, case.label)


def _make_strided(shape, dtype=torch.float32, offset=0.0):
    base = torch.linspace(
        -0.35 + offset,
        0.45 + offset,
        steps=math.prod(shape) * 2,
        dtype=dtype,
    ).reshape(*shape[:-1], shape[-1] * 2)
    return base[..., ::2]


def _p_branch_tensors():
    x1 = torch.tensor(
        [[0.0, 0.5, -0.25], [1.0, -0.5, 0.75]],
        dtype=torch.float32,
    )
    x2 = torch.tensor(
        [[0.25, -0.5, 0.0], [1.0, -0.5, 0.75], [-0.75, 0.25, 0.5]],
        dtype=torch.float32,
    )
    grad = torch.tensor(
        [[1.0, -0.25, 0.5], [-0.75, 0.125, 1.25]],
        dtype=torch.float32,
    )
    return x1, x2, grad


def test_cdistbwd_interface_exists():
    assert hasattr(torch.ops.ops_multimodal_fusion, "cdist_backward")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("p", [0.5, 1.0, 1.5, 2.0, 3.0, float("inf")])
def test_cdistbwd_p_branches_fp32(p):
    x1, x2, grad = _p_branch_tensors()
    _run_case(Case(x1, x2, grad, p, torch.float32, f"p-branch p={p}"))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_cdistbwd_p_zero_fp32():
    x1, x2, grad = _p_branch_tensors()
    _run_case(Case(x1, x2, grad, 0.0, torch.float32, "p-zero"))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape1,shape2,p",
    [
        ((2, 4), (3, 4), 2.0),
        ((2, 2, 5), (2, 3, 5), 1.0),
        ((2, 1, 2, 4), (1, 3, 3, 4), 1.5),
        ((1, 3, 2, 16), (2, 1, 4, 16), 3.0),
        ((1, 1), (2, 1), float("inf")),
        ((1, 2, 64), (1, 3, 64), 2.0),
        ((1, 1, 2, 128), (1, 1, 3, 128), 0.5),
    ],
)
def test_cdistbwd_shape_and_dtype_variants(dtype, shape1, shape2, p):
    x1 = _make_strided(shape1, torch.float32, offset=0.0)
    x2 = _make_strided(shape2, torch.float32, offset=0.2)
    distance_shape = (*torch.broadcast_shapes(shape1[:-2], shape2[:-2]), shape1[-2], shape2[-2])
    grad = torch.linspace(-0.6, 0.8, steps=math.prod(distance_shape)).reshape(distance_shape)
    _run_case(Case(x1, x2, grad, p, dtype, f"shape-dtype dtype={dtype} p={p}"))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("p", [0.5, 2.0, 3.0])
def test_cdistbwd_zero_distance_no_nan(dtype, p):
    x1 = torch.tensor([[[0.0, 1.0], [2.0, -1.0]]], dtype=torch.float32)
    x2 = torch.tensor([[[0.0, 1.0], [2.0, -1.0], [0.5, 0.5]]], dtype=torch.float32)
    grad = torch.tensor([[[1.0, -0.5, 0.25], [0.75, -1.25, 0.0]]], dtype=torch.float32)
    _run_case(Case(x1, x2, grad, p, dtype, f"zero-distance dtype={dtype} p={p}"))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_cdistbwd_p_lt_one_diff_zero_dist_nonzero():
    x1 = torch.tensor([[1.0, 0.0, 2.0]], dtype=torch.float32)
    x2 = torch.tensor([[1.0, 2.0, -1.0], [0.0, 0.0, 2.0]], dtype=torch.float32)
    grad = torch.tensor([[0.75, -1.25]], dtype=torch.float32)
    _run_case(Case(x1, x2, grad, 0.5, torch.float32, "p-lt-one-diff-zero"))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_cdistbwd_inf_ties_known_values():
    x1 = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    x2 = torch.tensor([[2.0, -2.0, 1.0], [-3.0, 3.0, 0.5]], dtype=torch.float32)
    grad = torch.tensor([[1.5, -2.0]], dtype=torch.float32)
    cdist = _make_distance(x1, x2, float("inf"), torch.float32)
    actual = _custom_dx1(x1, x2, grad, float("inf"), cdist)
    expected = torch.tensor([[-3.5, 3.5, 0.0]], dtype=torch.float32)
    _assert_close(actual, expected, torch.float32, "inf-ties-known")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "x1_shape,x2_shape",
    [
        ((0, 3), (2, 3)),
        ((2, 0), (3, 0)),
        ((0, 2, 3), (0, 4, 3)),
    ],
)
def test_cdistbwd_empty_no_value_shortcuts(x1_shape, x2_shape):
    x1 = torch.empty(x1_shape, dtype=torch.float32)
    x2 = torch.empty(x2_shape, dtype=torch.float32)
    distance_shape = (*torch.broadcast_shapes(x1_shape[:-2], x2_shape[:-2]), x1_shape[-2], x2_shape[-2])
    grad = torch.empty(distance_shape, dtype=torch.float32)
    cdist = torch.empty(distance_shape, dtype=torch.float32)
    actual = _custom_dx1(x1, x2, grad, 2.0, cdist)
    expected = torch.zeros((*distance_shape[:-1], x1_shape[-1]), dtype=torch.float32)
    _assert_close(actual, expected, torch.float32, f"empty x1={x1_shape} x2={x2_shape}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_cdistbwd_empty_r2_zero_shortcut():
    x1_shape = (2, 3)
    x2_shape = (0, 3)
    x1 = torch.empty(x1_shape, dtype=torch.float32)
    x2 = torch.empty(x2_shape, dtype=torch.float32)
    distance_shape = (x1_shape[-2], x2_shape[-2])
    grad = torch.empty(distance_shape, dtype=torch.float32)
    cdist = torch.empty(distance_shape, dtype=torch.float32)
    actual = _custom_dx1(x1, x2, grad, 2.0, cdist)
    expected = torch.zeros((x1_shape[-2], x1_shape[-1]), dtype=torch.float32)
    _assert_close(actual, expected, torch.float32, "empty-r2-zero")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_cdistbwd_non_contiguous_inputs():
    x1 = _make_strided((2, 2, 4), torch.float32, offset=0.0)
    x2 = _make_strided((2, 3, 4), torch.float32, offset=0.1)
    assert not x1.is_contiguous()
    assert not x2.is_contiguous()
    grad_base = torch.linspace(-0.5, 0.7, steps=2 * 2 * 3 * 2).reshape(2, 2, 3, 2)
    grad = grad_base[..., 0]
    cdist = _make_distance(x1, x2, 2.0, torch.float32)
    cdist_nc = torch.stack((cdist, cdist + 1.0), dim=-1)[..., 0]
    assert not grad.is_contiguous()
    assert not cdist_nc.is_contiguous()
    expected = _reference_dx1(x1, x2, grad, 2.0, torch.float32)
    actual = _custom_dx1(x1, x2, grad, 2.0, cdist_nc)
    _assert_close(actual, expected, torch.float32, "non-contiguous")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_cdistbwd_higher_rank_broadcast_batch():
    x1 = _make_strided((2, 1, 1, 2, 4), torch.float32, offset=-0.1)
    x2 = _make_strided((1, 3, 2, 3, 4), torch.float32, offset=0.2)
    distance_shape = (*torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2]), 2, 3)
    grad = torch.linspace(-0.4, 0.5, steps=math.prod(distance_shape)).reshape(distance_shape)
    _run_case(Case(x1, x2, grad, 2.5, torch.float32, "higher-rank-broadcast"))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_cdistbwd_multi_tile_feature_dim():
    x1 = torch.linspace(-0.25, 0.35, steps=2304, dtype=torch.float32).reshape(1, 2304)
    x2 = torch.linspace(0.15, -0.45, steps=2304, dtype=torch.float32).reshape(1, 2304)
    grad = torch.tensor([[0.75]], dtype=torch.float32)
    _run_case(Case(x1, x2, grad, 2.0, torch.float32, "multi-tile-feature-dim"))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("p", [0.5, 1.0, 2.0, 3.0, float("inf")])
def test_cdistbwd_zero_grad_returns_zero(p):
    x1 = _make_strided((1, 2, 5), torch.float32, offset=0.0)
    x2 = _make_strided((1, 3, 5), torch.float32, offset=0.3)
    grad = torch.zeros(1, 2, 3, dtype=torch.float32)
    _run_case(Case(x1, x2, grad, p, torch.float32, f"zero-grad p={p}"))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("p", [0.25, 2.5, 4.5])
def test_cdistbwd_additional_fractional_p_values(p):
    x1 = torch.tensor(
        [[[0.1, -0.4, 0.6], [0.7, 0.2, -0.3]]],
        dtype=torch.float32,
    )
    x2 = torch.tensor(
        [[[-0.2, 0.3, 0.5], [0.9, -0.1, -0.7], [0.0, 0.8, -0.2]]],
        dtype=torch.float32,
    )
    grad = torch.tensor(
        [[[0.25, -0.75, 1.0], [-0.5, 0.125, 0.875]]],
        dtype=torch.float32,
    )
    _run_case(Case(x1, x2, grad, p, torch.float32, f"fractional-p p={p}"))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_cdistbwd_non_contiguous_fp16_inputs():
    x1 = _make_strided((1, 2, 6), torch.float16, offset=0.0)
    x2 = _make_strided((1, 3, 6), torch.float16, offset=0.2)
    grad_base = torch.linspace(
        -0.5,
        0.7,
        steps=1 * 2 * 3 * 2,
        dtype=torch.float16,
    ).reshape(1, 2, 3, 2)
    grad = grad_base[..., 0]
    cdist = _make_distance(x1, x2, 1.5, torch.float16)
    cdist_nc = torch.stack((cdist, cdist + 1.0), dim=-1)[..., 0]
    assert not x1.is_contiguous()
    assert not x2.is_contiguous()
    assert not grad.is_contiguous()
    assert not cdist_nc.is_contiguous()
    expected = _reference_dx1(x1, x2, grad, 1.5, torch.float16)
    actual = _custom_dx1(x1, x2, grad, 1.5, cdist_nc)
    _assert_close(actual, expected, torch.float16, "non-contiguous-fp16")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_cdistbwd_dx2_reuse_matches_autograd():
    x1 = torch.tensor([[[0.2, -0.4], [1.0, 0.5]]], dtype=torch.float32)
    x2 = torch.tensor([[[-0.2, 0.1], [0.7, 0.5], [1.2, -0.3]]], dtype=torch.float32)
    grad = torch.tensor([[[0.5, -1.0, 0.25], [1.5, -0.75, 0.125]]], dtype=torch.float32)
    p = 2.0

    x1_req = x1.clone().requires_grad_(True)
    x2_req = x2.clone().requires_grad_(True)
    dist = torch.cdist(x1_req, x2_req, p=p)
    _, expected_dx2 = torch.autograd.grad(dist, (x1_req, x2_req), grad_outputs=grad)

    cdist = dist.detach()
    actual_dx2 = _custom_dx1(
        x2,
        x1,
        grad.transpose(-1, -2).contiguous(),
        p,
        cdist.transpose(-1, -2).contiguous(),
    )
    _assert_close(actual_dx2, expected_dx2, torch.float32, "dx2-reuse")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_cdistbwd_invalid_args():
    x1 = torch.randn(2, 3, dtype=torch.float32)
    x2 = torch.randn(4, 3, dtype=torch.float32)
    grad = torch.randn(2, 4, dtype=torch.float32)
    cdist = _make_distance(x1, x2, 2.0, torch.float32)

    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.cdist_backward(
            grad.npu(), torch.randn(3).npu(), x2.npu(), 2.0, cdist.npu()
        )
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.cdist_backward(
            grad.npu(), x1.npu(), torch.randn(4, 2).npu(), 2.0, cdist.npu()
        )
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.cdist_backward(
            grad.npu(), x1.npu(), x2.npu(), -1.0, cdist.npu()
        )
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.cdist_backward(
            grad.npu(), x1.npu(), x2.npu(), float("nan"), cdist.npu()
        )
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.cdist_backward(
            grad.npu(), x1.npu(), x2.npu(), 2.0, cdist[:, :3].npu()
        )
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.cdist_backward(
            grad.to(torch.float16).npu(), x1.npu(), x2.npu(), 2.0, cdist.npu()
        )
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.cdist_backward(
            grad.to(torch.int32).npu(),
            x1.to(torch.int32).npu(),
            x2.to(torch.int32).npu(),
            2.0,
            cdist.to(torch.int32).npu(),
        )
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.cdist_backward(
            torch.randn(2, 3, 2, 4).npu(),
            torch.randn(2, 2, 3).npu(),
            torch.randn(3, 4, 3).npu(),
            2.0,
            torch.randn(2, 3, 2, 4).npu(),
        )
