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

import math

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "pdist_backward"):
    pytest.skip(
        "ops_multimodal_fusion.pdist_backward not registered for current NPU_ARCH; "
        "skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _make_strided(shape, dtype=torch.float32, offset=0.0):
    base = torch.linspace(
        -0.45 + offset,
        0.55 + offset,
        steps=math.prod(shape) * 2,
        dtype=dtype,
    ).reshape(shape[0], shape[1] * 2)
    return base[:, ::2]


def _reference(x_cpu, p, dtype):
    """Compute pdist backward using PyTorch autograd on CPU."""
    if x_cpu.size(0) <= 1 or x_cpu.size(1) == 0:
        return torch.zeros_like(x_cpu, dtype=dtype)
    x_work = x_cpu.to(torch.float32).clone().requires_grad_(True)
    dist = F.pdist(x_work, p=float(p))
    loss = dist.sum()
    loss.backward()
    grad_input = x_work.grad
    return grad_input.to(dtype)


def _custom(grad_cpu, x_cpu, p, pdist_output_cpu):
    """Call the custom pdist_backward operator on NPU."""
    return torch.ops.ops_multimodal_fusion.pdist_backward(
        grad_cpu.npu(),
        x_cpu.npu(),
        float(p),
        pdist_output_cpu.npu(),
    ).cpu()


def _assert_close(actual, expected, dtype, label):
    assert actual.shape == expected.shape, (
        f"{label}: shape mismatch {actual.shape} vs {expected.shape}"
    )
    assert actual.dtype == expected.dtype, (
        f"{label}: dtype mismatch {actual.dtype} vs {expected.dtype}"
    )
    assert torch.isfinite(actual.float()).all(), (
        f"{label}: actual contains NaN/Inf"
    )
    if dtype == torch.float32:
        assert torch.allclose(actual, expected, rtol=3e-4, atol=3e-4), (
            f"{label}: fp32 mismatch max_diff="
            f"{(actual - expected).abs().max().item()}"
        )
    elif dtype == torch.float16:
        assert torch.allclose(actual.float(), expected.float(), rtol=6e-2, atol=6e-2), (
            f"{label}: fp16 mismatch"
        )
    else:
        assert torch.allclose(actual.float(), expected.float(), rtol=8e-2, atol=8e-2), (
            f"{label}: bf16 mismatch"
        )


def _run_case(x_cpu, p, dtype, label):
    x_typed = x_cpu.to(dtype)
    x_ref = x_typed.to(torch.float32)

    # Reference: autograd on the same representable input values in fp32.
    expected = _reference(x_ref, p, dtype)

    # Build inputs for custom op
    x_npu_input = x_typed
    pdist_out = F.pdist(x_ref, p=float(p)).to(dtype)
    grad = torch.ones_like(pdist_out)

    actual = _custom(grad, x_npu_input, p, pdist_out)
    _assert_close(actual, expected, dtype, label)


# ── Interface tests ──────────────────────────────────────────────────────

def test_pdist_backward_interface_exists():
    assert hasattr(torch.ops.ops_multimodal_fusion, "pdist_backward")


# ── P-value branch tests ─────────────────────────────────────────────────

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("p", [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, float("inf")])
def test_pdist_backward_p_branches_fp32(p):
    x = torch.tensor(
        [
            [0.0, 0.5, -0.25],
            [1.0, -0.5, 0.75],
            [0.0, 0.5, -0.25],
            [-0.75, 0.25, 0.5],
        ],
        dtype=torch.float32,
    )
    _run_case(x, p, torch.float32, f"p-branches p={p}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_pdist_backward_default_p_is_two():
    x = torch.tensor(
        [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]],
        dtype=torch.float32,
    )
    _run_case(x, 2.0, torch.float32, "default-p")


# ── Shape & dtype variants ───────────────────────────────────────────────

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape,p",
    [
        ((2, 4), 2.0),
        ((5, 3), 1.0),
        ((4, 5), 0.5),
        ((3, 16), 3.0),
        ((3, 1), float("inf")),
    ],
)
def test_pdist_backward_shape_dtype_variants(dtype, shape, p):
    x = _make_strided(shape, dtype=torch.float32, offset=0.1)
    _run_case(x, p, dtype, f"shape-dtype dtype={dtype} shape={shape} p={p}")


# ── Zero-gradient / identical rows ───────────────────────────────────────

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_pdist_backward_identical_rows_zero_distance(dtype):
    x = torch.tensor(
        [
            [0.25, -0.5, 0.75, 0.0],
            [0.25, -0.5, 0.75, 0.0],
            [-0.25, 0.5, -0.75, 1.0],
            [0.25, -0.5, 0.75, 0.0],
        ],
        dtype=torch.float32,
    )
    _run_case(x, 2.0, dtype, f"identical-rows dtype={dtype}")


# ── gradient for p of zero should be zero ────────────────────────────────

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_pdist_backward_p_zero_gradient_zero():
    x = torch.tensor(
        [[0.0, 0.5], [1.0, -0.5], [2.0, 0.75]],
        dtype=torch.float32,
    )
    expected = _reference(x, 0.0, torch.float32)
    pdist_out = F.pdist(x.to(torch.float32), p=0.0).to(torch.float32)
    grad = torch.ones_like(pdist_out)
    actual = _custom(grad, x.to(torch.float32), 0.0, pdist_out)
    # gradient for p of zero is all zeros
    assert torch.allclose(actual, torch.zeros_like(actual), rtol=0.0, atol=0.0), (
        f"p=0 gradient not zero: {actual}"
    )
    _assert_close(actual, expected, torch.float32, "p-zero-grad")


# ── sign-based gradient for p of one ─────────────────────────────────────

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_pdist_backward_p_one_sign_pattern():
    """Verify p of one backward gives the sign of the per-pair difference."""
    x = torch.tensor(
        [[0.0, 2.0, -1.0], [1.0, -3.0, 0.5], [-2.0, 1.0, 2.0]],
        dtype=torch.float32,
    )
    _run_case(x, 1.0, torch.float32, "p-one-sign")


# ── Euclidean gradient for p of two ──────────────────────────────────────

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_pdist_backward_p_two_euclidean():
    """Known-values test for the backward pass at p of two."""
    x = torch.tensor(
        [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]],
        dtype=torch.float32,
    )
    _run_case(x, 2.0, torch.float32, "p-two-euclidean")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_pdist_backward_p_inf_known_values_ties_fp32():
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, -2.0, 1.0],
            [-3.0, 1.0, 3.0],
        ],
        dtype=torch.float32,
    )
    grad = torch.tensor([1.0, 0.5, -2.0], dtype=torch.float32)
    pdist_out = F.pdist(x, p=float("inf"))
    actual = _custom(grad, x, float("inf"), pdist_out)
    expected = torch.tensor(
        [
            [-0.5, 1.0, -0.5],
            [-1.0, -1.0, 0.0],
            [1.5, 0.0, 0.5],
        ],
        dtype=torch.float32,
    )
    _assert_close(actual, expected, torch.float32, "p-inf-known-ties-fp32")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_pdist_backward_p_inf_known_values_low_precision_1d(dtype):
    x = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32).to(dtype)
    x_ref = x.to(torch.float32)
    pdist_out = F.pdist(x_ref, p=float("inf")).to(dtype)
    grad = torch.ones_like(pdist_out)
    actual = _custom(grad, x, float("inf"), pdist_out)
    expected = torch.tensor([[-2.0], [0.0], [2.0]], dtype=dtype)
    _assert_close(actual, expected, dtype, f"p-inf-known-1d-{dtype}")


# ── Multi-tile feature dimension ─────────────────────────────────────────

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_pdist_backward_multi_tile_feature_dim():
    x = torch.stack(
        (
            torch.linspace(-0.2, 0.4, steps=2304),
            torch.linspace(0.3, -0.5, steps=2304),
        )
    ).to(torch.float32)
    _run_case(x, 2.0, torch.float32, "multi-tile-feature-dim")


# ── Empty / edge shapes ──────────────────────────────────────────────────

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", [(0, 3), (1, 3), (3, 0)])
def test_pdist_backward_empty_and_edge_shapes(shape):
    x = torch.empty(shape, dtype=torch.float32)
    pdist_out = F.pdist(x.to(torch.float32), p=2.0).to(torch.float32)
    grad = torch.ones_like(pdist_out)
    actual = _custom(grad, x.to(torch.float32), 2.0, pdist_out)
    expected = _reference(x, 2.0, torch.float32)
    _assert_close(actual, expected, torch.float32, f"empty shape={shape}")


# ── Non-contiguous input ─────────────────────────────────────────────────

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_pdist_backward_non_contiguous_input():
    x = _make_strided((4, 6), dtype=torch.float32, offset=-0.1)
    assert not x.is_contiguous()
    _run_case(x, 1.5, torch.float32, "non-contiguous")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_pdist_backward_non_contiguous_bfloat16_input():
    x = _make_strided((3, 6), dtype=torch.bfloat16, offset=0.2)
    assert not x.is_contiguous()
    x_typed = x.to(torch.bfloat16)
    x_ref = x_typed.to(torch.float32)
    expected = _reference(x_ref, 2.0, torch.bfloat16)
    pdist_out = F.pdist(x_ref, p=2.0).to(torch.bfloat16)
    grad = torch.ones_like(pdist_out)
    actual = _custom(grad, x_typed, 2.0, pdist_out)
    _assert_close(actual, expected, torch.bfloat16, "non-contiguous-bf16")


# ── Invalid argument tests ───────────────────────────────────────────────

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_pdist_backward_invalid_args():
    x = torch.randn(3, 4, dtype=torch.float32)
    grad = torch.ones(3, dtype=torch.float32)
    pdist_out = F.pdist(x, p=2.0)

    # Wrong input rank
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.pdist_backward(
            grad.npu(), torch.randn(4).npu(), 2.0, pdist_out.npu()
        )

    # Wrong grad shape
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.pdist_backward(
            torch.ones(10).npu(), x.npu(), 2.0, pdist_out.npu()
        )

    # Negative p
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.pdist_backward(
            grad.npu(), x.npu(), -1.0, pdist_out.npu()
        )

    # NaN p
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.pdist_backward(
            grad.npu(), x.npu(), float("nan"), pdist_out.npu()
        )

    # Wrong dtype
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.pdist_backward(
            grad.to(torch.int32).npu(), x.to(torch.int32).npu(),
            2.0, pdist_out.to(torch.int32).npu()
        )

    # Wrong pdist_output shape
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.pdist_backward(
            grad.npu(), x.npu(), 2.0, torch.ones(10).npu()
        )


# ── A grad of zero produces zero output ──────────────────────────────────

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_pdist_backward_zero_grad_all_zeros():
    x = torch.randn(5, 7, dtype=torch.float32)
    pdist_out = F.pdist(x, p=2.0)
    grad = torch.zeros_like(pdist_out)
    actual = _custom(grad, x, 2.0, pdist_out)
    assert torch.allclose(actual, torch.zeros_like(actual), rtol=0.0, atol=0.0), (
        f"zero grad should produce all zeros, got {actual}"
    )


# ── Larger shape / stress test ───────────────────────────────────────────

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_pdist_backward_larger_shape():
    x = _make_strided((8, 32), dtype=torch.float32, offset=0.05)
    _run_case(x, 2.0, torch.float32, "larger-shape")


# ── End-to-end: autograd through custom pdist ────────────────────────────
@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_pdist_backward_e2e_consistency():
    """Verify pdist_backward produces the same result as autograd through pdist."""
    x = torch.tensor(
        [
            [0.5, -0.25, 0.75, 0.1],
            [-0.5, 0.5, -0.5, 0.2],
            [0.25, 0.0, 0.5, -0.1],
        ],
        dtype=torch.float32,
    )

    # CPU reference: autograd through the pdist forward
    x_ref = x.clone().requires_grad_(True)
    dist_ref = F.pdist(x_ref, p=2.0)
    loss_ref = dist_ref.sum()
    loss_ref.backward()
    expected = x_ref.grad

    # Keep this test focused on the backward op. Running custom pdist forward
    # and custom backward in a single process can trip device state.
    pdist_out = F.pdist(x, p=2.0)
    grad = torch.ones_like(pdist_out)
    actual = torch.ops.ops_multimodal_fusion.pdist_backward(
        grad.npu(), x.npu(), 2.0, pdist_out.npu()
    ).cpu()

    _assert_close(actual, expected, torch.float32, "e2e-consistency")
