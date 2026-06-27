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
"""Tests for ops_multimodal_fusion.gru_cell."""

from collections import namedtuple

import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "gru_cell"):
    pytest.skip(
        "ops_multimodal_fusion.gru_cell not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float16, torch.float32]

# Bundle the six GRU operands so helper signatures stay within the parameter
# limit. bias_ih / bias_hh may be None when absent.
GruInputs = namedtuple(
    "GruInputs", ["input_", "hx", "weight_ih", "weight_hh", "bias_ih", "bias_hh"]
)
# Tolerance bound for an arbitrary float comparison.
Tol = namedtuple("Tol", ["rtol", "atol"])
# Parameters describing how to synthesise a GruInputs bundle. bias_flags selects
# whether bias_ih / bias_hh are present; bounds is the uniform sampling range.
BuildSpec = namedtuple(
    "BuildSpec",
    ["shape_input", "hidden_size", "dtype", "seed", "bias_flags", "bounds"],
)
BuildSpec.__new__.__defaults__ = ((True, True), (-0.4, 0.4))


def _make_tensor(shape, dtype, seed, bounds=(-0.4, 0.4)):
    low, high = bounds
    gen = torch.Generator().manual_seed(seed)
    return torch.empty(*shape, dtype=torch.float32).uniform_(low, high, generator=gen).to(dtype)


def _build_inputs(spec):
    """Construct a GruInputs bundle from a BuildSpec.

    spec.shape_input is the full input shape (1-D or 2-D). spec.bias_flags
    selects whether bias_ih / bias_hh are present.
    """
    shape_input = spec.shape_input
    hidden_size = spec.hidden_size
    dtype = spec.dtype
    seed = spec.seed
    bounds = spec.bounds
    input_size = shape_input[-1]
    hx_shape = shape_input[:-1] + (hidden_size,)
    bias_ih_present, bias_hh_present = spec.bias_flags
    input_ = _make_tensor(shape_input, dtype, seed, bounds)
    hx = _make_tensor(hx_shape, dtype, seed + 1, bounds)
    weight_ih = _make_tensor((3 * hidden_size, input_size), dtype, seed + 2, bounds)
    weight_hh = _make_tensor((3 * hidden_size, hidden_size), dtype, seed + 3, bounds)
    bias_ih = _make_tensor((3 * hidden_size,), dtype, seed + 4, bounds) if bias_ih_present else None
    bias_hh = _make_tensor((3 * hidden_size,), dtype, seed + 5, bounds) if bias_hh_present else None
    return GruInputs(input_, hx, weight_ih, weight_hh, bias_ih, bias_hh)


def _reference(gru):
    gi = gru.input_.float().matmul(gru.weight_ih.float().t())
    gh = gru.hx.float().matmul(gru.weight_hh.float().t())
    if gru.bias_ih is not None:
        gi = gi + gru.bias_ih.float()
    if gru.bias_hh is not None:
        gh = gh + gru.bias_hh.float()
    chunk_dim = 0 if gi.dim() == 1 else 1
    i_r, i_z, i_n = gi.chunk(3, dim=chunk_dim)
    h_r, h_z, h_n = gh.chunk(3, dim=chunk_dim)
    resetgate = torch.sigmoid(i_r + h_r)
    updategate = torch.sigmoid(i_z + h_z)
    newgate = torch.tanh(i_n + resetgate * h_n)
    hy = newgate + updategate * (gru.hx.float() - newgate)
    return hy.to(gru.input_.dtype)


def _custom(gru):
    hy = torch.ops.ops_multimodal_fusion.gru_cell(
        gru.input_.npu(),
        gru.hx.npu(),
        gru.weight_ih.npu(),
        gru.weight_hh.npu(),
        None if gru.bias_ih is None else gru.bias_ih.npu(),
        None if gru.bias_hh is None else gru.bias_hh.npu(),
    )
    return hy.cpu()


def _assert_close(actual, expected, label):
    assert actual.shape == expected.shape, (
        f"{label}: shape mismatch {actual.shape} vs {expected.shape}"
    )
    assert actual.dtype == expected.dtype, (
        f"{label}: dtype mismatch {actual.dtype} vs {expected.dtype}"
    )
    rtol, atol = (2e-2, 2e-2) if expected.dtype == torch.float16 else (2e-4, 2e-4)
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"{label}: max_abs={(actual - expected).abs().max().item()}"
    )


def _assert_allclose_with_tol(actual, expected, label, tol):
    assert actual.shape == expected.shape, (
        f"{label}: shape mismatch {actual.shape} vs {expected.shape}"
    )
    assert torch.allclose(actual.float(), expected.float(), rtol=tol.rtol, atol=tol.atol), (
        f"{label}: max_abs={(actual.float() - expected.float()).abs().max().item()}"
    )


def _run_and_compare(gru, label):
    """Run both the custom op and the reference, then assert close."""
    actual = _custom(gru)
    expected = _reference(gru)
    _assert_close(actual, expected, label)
    return actual


def _load_module(module, gru):
    """Copy a GruInputs bundle's weights/biases into a torch.nn.GRUCell."""
    with torch.no_grad():
        module.weight_ih.copy_(gru.weight_ih)
        module.weight_hh.copy_(gru.weight_hh)
        module.bias_ih.copy_(gru.bias_ih)
        module.bias_hh.copy_(gru.bias_hh)


def _assert_matches_module(shape_input, hidden_size, seed):
    gru = _build_inputs(BuildSpec(shape_input, hidden_size, torch.float32, seed))
    input_size = shape_input[-1]
    module = torch.nn.GRUCell(input_size, hidden_size)
    _load_module(module, gru)
    expected = module(gru.input_, gru.hx)
    actual = _custom(gru)
    _assert_close(actual, expected, f"torch.nn.GRUCell shape={shape_input} H={hidden_size}")


def test_gru_cell_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "gru_cell")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "batch,input_size,hidden_size,bias",
    [
        (1, 1, 1, True),
        (1, 3, 2, True),
        (2, 2, 3, True),
        (3, 4, 2, False),
        (2, 5, 4, False),
    ],
)
def test_gru_cell_matches_formula(dtype, batch, input_size, hidden_size, bias):
    seed = 100 + batch * 17 + input_size * 11 + hidden_size * 7 + int(bias)
    gru = _build_inputs(BuildSpec((batch, input_size), hidden_size, dtype, seed, (bias, bias)))
    _run_and_compare(gru, f"B={batch} I={input_size} H={hidden_size} dtype={dtype} bias={bias}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("input_size,hidden_size,bias", [(3, 2, True), (5, 4, False)])
def test_gru_cell_unbatched_matches_formula(dtype, input_size, hidden_size, bias):
    seed = 700 + input_size * 13 + hidden_size * 19 + int(bias)
    gru = _build_inputs(BuildSpec((input_size,), hidden_size, dtype, seed, (bias, bias)))
    _run_and_compare(gru, f"unbatched I={input_size} H={hidden_size} dtype={dtype} bias={bias}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "bias_ih_present,bias_hh_present",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_gru_cell_unbatched_bias_combinations(dtype, bias_ih_present, bias_hh_present):
    seed = 760 + int(bias_ih_present) * 17 + int(bias_hh_present) * 31
    gru = _build_inputs(BuildSpec((4,), 3, dtype, seed, (bias_ih_present, bias_hh_present)))
    _run_and_compare(
        gru,
        f"unbatched-bias I=4 H=3 dtype={dtype} ih={bias_ih_present} hh={bias_hh_present}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bias_ih_present,bias_hh_present", [(True, False), (False, True)])
def test_gru_cell_single_side_bias(dtype, bias_ih_present, bias_hh_present):
    seed = 850 + int(bias_ih_present) * 17 + int(bias_hh_present) * 29
    gru = _build_inputs(BuildSpec((2, 3), 3, dtype, seed, (bias_ih_present, bias_hh_present)))
    _run_and_compare(
        gru, f"single-side-bias dtype={dtype} ih={bias_ih_present} hh={bias_hh_present}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_gru_cell_known_values(dtype):
    input_ = torch.tensor([[0.25, -0.5]], dtype=dtype)
    hx = torch.tensor([[0.1, -0.2]], dtype=dtype)
    weight_ih = torch.tensor(
        [
            [0.2, -0.1], [0.05, 0.3],
            [-0.3, 0.4], [0.2, 0.1],
            [0.5, -0.2], [-0.1, 0.25],
        ],
        dtype=dtype,
    )
    weight_hh = torch.tensor(
        [
            [0.1, 0.2], [-0.2, 0.1],
            [0.3, -0.1], [0.05, 0.2],
            [-0.25, 0.15], [0.2, -0.3],
        ],
        dtype=dtype,
    )
    bias_ih = torch.linspace(-0.2, 0.2, 6, dtype=dtype)
    bias_hh = torch.linspace(0.1, -0.1, 6, dtype=dtype)
    gru = GruInputs(input_, hx, weight_ih, weight_hh, bias_ih, bias_hh)
    _run_and_compare(gru, f"known-values dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_gru_cell_saturated_gates(dtype):
    input_ = torch.tensor([[3.0, -3.0]], dtype=dtype)
    hx = torch.tensor([[2.0, -2.0]], dtype=dtype)
    weight_ih = _make_tensor((6, 2), dtype, 401, (-1.0, 1.0))
    weight_hh = _make_tensor((6, 2), dtype, 402, (-1.0, 1.0))
    bias_ih = torch.tensor([2.0, -2.0, 1.5, -1.5, 0.5, -0.5], dtype=dtype)
    bias_hh = torch.tensor([-1.0, 1.0, 2.0, -2.0, -0.5, 0.5], dtype=dtype)
    gru = GruInputs(input_, hx, weight_ih, weight_hh, bias_ih, bias_hh)
    _run_and_compare(gru, f"saturated dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_gru_cell_larger_hidden_offsets(dtype):
    gru = _build_inputs(BuildSpec((2, 6), 8, dtype, 950, (True, True), (-0.2, 0.2)))
    _run_and_compare(gru, f"larger-hidden B=2 I=6 H=8 dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "batch,input_size,hidden_size,seed",
    [(1, 1, 1, 351), (3, 2, 5, 361), (4, 5, 3, 371)],
)
def test_gru_cell_matches_torch_nn_module_float32_shapes(batch, input_size, hidden_size, seed):
    _assert_matches_module((batch, input_size), hidden_size, seed)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("input_size,hidden_size,seed", [(1, 1, 381), (5, 2, 391)])
def test_gru_cell_unbatched_matches_torch_nn_module_float32_shapes(input_size, hidden_size, seed):
    _assert_matches_module((input_size,), hidden_size, seed)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_gru_cell_matches_torch_nn_module_float32():
    _assert_matches_module((2, 3), 4, 301)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_gru_cell_unbatched_matches_torch_nn_module_float32():
    _assert_matches_module((3,), 4, 801)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("update_bias,target_name", [(10.0, "hx"), (-10.0, "new")])
def test_gru_cell_update_gate_isolation(dtype, update_bias, target_name):
    input_ = torch.zeros((1, 2), dtype=dtype)
    hx = torch.tensor([[0.35, -0.45]], dtype=dtype)
    weight_ih = torch.zeros((6, 2), dtype=dtype)
    weight_hh = torch.zeros((6, 2), dtype=dtype)
    bias_ih = torch.tensor([0.0, 0.0, update_bias, update_bias, 0.30, -0.40], dtype=dtype)
    bias_hh = torch.zeros(6, dtype=dtype)
    gru = GruInputs(input_, hx, weight_ih, weight_hh, bias_ih, bias_hh)
    actual = _run_and_compare(gru, f"update-gate-isolation {target_name} dtype={dtype}")

    newgate_target = torch.tanh(torch.tensor([[0.30, -0.40]], dtype=torch.float32)).to(dtype)
    target = hx if target_name == "hx" else newgate_target
    _assert_allclose_with_tol(
        actual, target, f"update-gate-target {target_name} dtype={dtype}", Tol(2e-3, 2e-3)
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_gru_cell_rejects_mixed_rank_state():
    input_ = torch.randn(3, dtype=torch.float32)
    hx = torch.randn(1, 2, dtype=torch.float32)
    weight_ih = torch.randn(6, 3, dtype=torch.float32)
    weight_hh = torch.randn(6, 2, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(GruInputs(input_, hx, weight_ih, weight_hh, None, None))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_gru_cell_rejects_batch_mismatch():
    input_ = torch.randn(2, 3, dtype=torch.float32)
    hx = torch.randn(3, 2, dtype=torch.float32)
    weight_ih = torch.randn(6, 3, dtype=torch.float32)
    weight_hh = torch.randn(6, 2, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(GruInputs(input_, hx, weight_ih, weight_hh, None, None))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_gru_cell_rejects_input_rank3():
    input_ = torch.randn(1, 2, 3, dtype=torch.float32)
    hx = torch.randn(1, 2, 2, dtype=torch.float32)
    weight_ih = torch.randn(6, 3, dtype=torch.float32)
    weight_hh = torch.randn(6, 2, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(GruInputs(input_, hx, weight_ih, weight_hh, None, None))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_gru_cell_rejects_hx_rank3():
    input_ = torch.randn(2, 3, dtype=torch.float32)
    hx = torch.randn(2, 1, 2, dtype=torch.float32)
    weight_ih = torch.randn(6, 3, dtype=torch.float32)
    weight_hh = torch.randn(6, 2, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(GruInputs(input_, hx, weight_ih, weight_hh, None, None))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_gru_cell_invalid_weight_shape():
    input_ = torch.randn(2, 3, dtype=torch.float32)
    hx = torch.randn(2, 2, dtype=torch.float32)
    weight_ih = torch.randn(5, 3, dtype=torch.float32)
    weight_hh = torch.randn(6, 2, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(GruInputs(input_, hx, weight_ih, weight_hh, None, None))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_gru_cell_invalid_weight_hh_hidden_dim():
    input_ = torch.randn(2, 3, dtype=torch.float32)
    hx = torch.randn(2, 2, dtype=torch.float32)
    weight_ih = torch.randn(6, 3, dtype=torch.float32)
    weight_hh = torch.randn(6, 3, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(GruInputs(input_, hx, weight_ih, weight_hh, None, None))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_gru_cell_invalid_weight_rank():
    input_ = torch.randn(2, 3, dtype=torch.float32)
    hx = torch.randn(2, 2, dtype=torch.float32)
    weight_ih = torch.randn(6, 3, 1, dtype=torch.float32)
    weight_hh = torch.randn(6, 2, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(GruInputs(input_, hx, weight_ih, weight_hh, None, None))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_gru_cell_invalid_bias_shape():
    input_ = torch.randn(2, 3, dtype=torch.float32)
    hx = torch.randn(2, 2, dtype=torch.float32)
    weight_ih = torch.randn(6, 3, dtype=torch.float32)
    weight_hh = torch.randn(6, 2, dtype=torch.float32)
    bias_ih = torch.randn(5, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(GruInputs(input_, hx, weight_ih, weight_hh, bias_ih, None))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_gru_cell_dtype_mismatch():
    input_ = torch.randn(2, 3, dtype=torch.float32)
    hx = torch.randn(2, 2, dtype=torch.float16)
    weight_ih = torch.randn(6, 3, dtype=torch.float32)
    weight_hh = torch.randn(6, 2, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(GruInputs(input_, hx, weight_ih, weight_hh, None, None))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_gru_cell_invalid_dtype():
    input_ = torch.randint(0, 5, (2, 3), dtype=torch.int32)
    hx = torch.randint(0, 5, (2, 2), dtype=torch.int32)
    weight_ih = torch.randint(0, 5, (6, 3), dtype=torch.int32)
    weight_hh = torch.randint(0, 5, (6, 2), dtype=torch.int32)
    with pytest.raises(RuntimeError):
        _custom(GruInputs(input_, hx, weight_ih, weight_hh, None, None))
