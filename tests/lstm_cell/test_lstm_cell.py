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

from collections import namedtuple

import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "lstm_cell"):
    pytest.skip(
        "ops_multimodal_fusion.lstm_cell not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float16, torch.float32]

# Full set of LSTMCell input tensors grouped together so the reference and
# device helpers stay within the five-parameter codecheck limit.
LstmInputs = namedtuple(
    "LstmInputs",
    ["input", "hx", "cx", "weight_ih", "weight_hh", "bias_ih", "bias_hh"],
)

# Specification for the parametrized batched case builder, kept as a single
# argument for the same reason.
CaseSpec = namedtuple(
    "CaseSpec",
    ["batch", "input_size", "hidden_size", "dtype", "seed", "bias", "low", "high"],
)


def _make_tensor(shape, dtype, seed, low=-0.4, high=0.4):
    gen = torch.Generator().manual_seed(seed)
    return torch.empty(*shape, dtype=torch.float32).uniform_(low, high, generator=gen).to(dtype)


def _reference(inputs):
    gates = inputs.input.float().matmul(inputs.weight_ih.float().t())
    gates = gates + inputs.hx.float().matmul(inputs.weight_hh.float().t())
    if inputs.bias_ih is not None:
        gates = gates + inputs.bias_ih.float()
    if inputs.bias_hh is not None:
        gates = gates + inputs.bias_hh.float()
    chunk_dim = 0 if gates.dim() == 1 else 1
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=chunk_dim)
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)
    cy = forgetgate * inputs.cx.float() + ingate * cellgate
    hy = outgate * torch.tanh(cy)
    return hy.to(inputs.input.dtype), cy.to(inputs.input.dtype)


def _custom(inputs):
    args = [
        inputs.input.npu(),
        inputs.hx.npu(),
        inputs.cx.npu(),
        inputs.weight_ih.npu(),
        inputs.weight_hh.npu(),
        None if inputs.bias_ih is None else inputs.bias_ih.npu(),
        None if inputs.bias_hh is None else inputs.bias_hh.npu(),
    ]
    hy, cy = torch.ops.ops_multimodal_fusion.lstm_cell(*args)
    return hy.cpu(), cy.cpu()


def _assert_pair_close(actual, expected, label):
    (hy, cy) = actual
    (ref_hy, ref_cy) = expected
    assert hy.shape == ref_hy.shape and cy.shape == ref_cy.shape, label
    assert hy.dtype == ref_hy.dtype and cy.dtype == ref_cy.dtype, label
    rtol, atol = (2e-2, 2e-2) if ref_hy.dtype == torch.float16 else (2e-4, 2e-4)
    assert torch.allclose(hy, ref_hy, rtol=rtol, atol=atol), (
        f"{label} hy mismatch max_abs={(hy - ref_hy).abs().max().item()}"
    )
    assert torch.allclose(cy, ref_cy, rtol=rtol, atol=atol), (
        f"{label} cy mismatch max_abs={(cy - ref_cy).abs().max().item()}"
    )


def _build_batched_inputs(spec):
    bias = spec.bias
    seed = spec.seed
    input_ = _make_tensor((spec.batch, spec.input_size), spec.dtype, seed, spec.low, spec.high)
    hx = _make_tensor((spec.batch, spec.hidden_size), spec.dtype, seed + 1, spec.low, spec.high)
    cx = _make_tensor((spec.batch, spec.hidden_size), spec.dtype, seed + 2, spec.low, spec.high)
    weight_ih = _make_tensor((4 * spec.hidden_size, spec.input_size), spec.dtype, seed + 3, spec.low, spec.high)
    weight_hh = _make_tensor((4 * spec.hidden_size, spec.hidden_size), spec.dtype, seed + 4, spec.low, spec.high)
    bias_ih = _make_tensor((4 * spec.hidden_size,), spec.dtype, seed + 5, spec.low, spec.high) if bias else None
    bias_hh = _make_tensor((4 * spec.hidden_size,), spec.dtype, seed + 6, spec.low, spec.high) if bias else None
    return LstmInputs(input_, hx, cx, weight_ih, weight_hh, bias_ih, bias_hh)


def _case(spec):
    inputs = _build_batched_inputs(spec)
    actual = _custom(inputs)
    expected = _reference(inputs)
    label = (
        f"B={spec.batch} I={spec.input_size} H={spec.hidden_size} "
        f"dtype={spec.dtype} bias={spec.bias}"
    )
    _assert_pair_close(actual, expected, label)


def _unbatched_case(input_size, hidden_size, dtype, seed, bias=True):
    input_ = _make_tensor((input_size,), dtype, seed)
    hx = _make_tensor((hidden_size,), dtype, seed + 1)
    cx = _make_tensor((hidden_size,), dtype, seed + 2)
    weight_ih = _make_tensor((4 * hidden_size, input_size), dtype, seed + 3)
    weight_hh = _make_tensor((4 * hidden_size, hidden_size), dtype, seed + 4)
    bias_ih = _make_tensor((4 * hidden_size,), dtype, seed + 5) if bias else None
    bias_hh = _make_tensor((4 * hidden_size,), dtype, seed + 6) if bias else None
    inputs = LstmInputs(input_, hx, cx, weight_ih, weight_hh, bias_ih, bias_hh)
    actual = _custom(inputs)
    expected = _reference(inputs)
    _assert_pair_close(
        actual,
        expected,
        f"unbatched I={input_size} H={hidden_size} dtype={dtype} bias={bias}",
    )


def test_lstm_cell_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "lstm_cell")


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
def test_lstm_cell_matches_formula(dtype, batch, input_size, hidden_size, bias):
    seed = 100 + batch * 17 + input_size * 11 + hidden_size * 7 + int(bias)
    _case(CaseSpec(batch, input_size, hidden_size, dtype, seed, bias, -0.4, 0.4))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("input_size,hidden_size,bias", [(3, 2, True), (5, 4, False)])
def test_lstm_cell_unbatched_matches_formula(dtype, input_size, hidden_size, bias):
    seed = 700 + input_size * 13 + hidden_size * 19 + int(bias)
    _unbatched_case(input_size, hidden_size, dtype, seed, bias=bias)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bias_ih_present,bias_hh_present", [(True, False), (False, True)])
def test_lstm_cell_single_side_bias(dtype, bias_ih_present, bias_hh_present):
    batch, input_size, hidden_size = 2, 3, 3
    seed = 850 + int(bias_ih_present) * 17 + int(bias_hh_present) * 29
    input_ = _make_tensor((batch, input_size), dtype, seed)
    hx = _make_tensor((batch, hidden_size), dtype, seed + 1)
    cx = _make_tensor((batch, hidden_size), dtype, seed + 2)
    weight_ih = _make_tensor((4 * hidden_size, input_size), dtype, seed + 3)
    weight_hh = _make_tensor((4 * hidden_size, hidden_size), dtype, seed + 4)
    bias_ih = _make_tensor((4 * hidden_size,), dtype, seed + 5) if bias_ih_present else None
    bias_hh = _make_tensor((4 * hidden_size,), dtype, seed + 6) if bias_hh_present else None
    inputs = LstmInputs(input_, hx, cx, weight_ih, weight_hh, bias_ih, bias_hh)
    actual = _custom(inputs)
    expected = _reference(inputs)
    _assert_pair_close(
        actual,
        expected,
        f"single-side-bias dtype={dtype} ih={bias_ih_present} hh={bias_hh_present}",
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_lstm_cell_large_cell_state(dtype):
    batch, input_size, hidden_size = 2, 2, 3
    input_ = _make_tensor((batch, input_size), dtype, 901, low=-0.2, high=0.2)
    hx = _make_tensor((batch, hidden_size), dtype, 902, low=-0.2, high=0.2)
    cx = torch.tensor([[4.0, -4.0, 3.5], [-3.5, 2.5, -2.5]], dtype=dtype)
    weight_ih = _make_tensor((4 * hidden_size, input_size), dtype, 903, low=-0.3, high=0.3)
    weight_hh = _make_tensor((4 * hidden_size, hidden_size), dtype, 904, low=-0.3, high=0.3)
    bias_ih = _make_tensor((4 * hidden_size,), dtype, 905, low=-0.2, high=0.2)
    bias_hh = _make_tensor((4 * hidden_size,), dtype, 906, low=-0.2, high=0.2)
    inputs = LstmInputs(input_, hx, cx, weight_ih, weight_hh, bias_ih, bias_hh)
    actual = _custom(inputs)
    expected = _reference(inputs)
    _assert_pair_close(actual, expected, f"large-cx dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_lstm_cell_larger_hidden_offsets(dtype):
    _case(CaseSpec(2, 6, 8, dtype, 950, True, -0.2, 0.2))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_lstm_cell_matches_torch_nn_module_float32():
    batch, input_size, hidden_size = 2, 3, 4
    dtype = torch.float32
    input_ = _make_tensor((batch, input_size), dtype, 301)
    hx = _make_tensor((batch, hidden_size), dtype, 302)
    cx = _make_tensor((batch, hidden_size), dtype, 303)
    weight_ih = _make_tensor((4 * hidden_size, input_size), dtype, 304)
    weight_hh = _make_tensor((4 * hidden_size, hidden_size), dtype, 305)
    bias_ih = _make_tensor((4 * hidden_size,), dtype, 306)
    bias_hh = _make_tensor((4 * hidden_size,), dtype, 307)
    module = torch.nn.LSTMCell(input_size, hidden_size)
    with torch.no_grad():
        module.weight_ih.copy_(weight_ih)
        module.weight_hh.copy_(weight_hh)
        module.bias_ih.copy_(bias_ih)
        module.bias_hh.copy_(bias_hh)
    expected = module(input_, (hx, cx))
    inputs = LstmInputs(input_, hx, cx, weight_ih, weight_hh, bias_ih, bias_hh)
    actual = _custom(inputs)
    _assert_pair_close(actual, expected, "torch.nn.LSTMCell")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_lstm_cell_unbatched_matches_torch_nn_module_float32():
    input_size, hidden_size = 3, 4
    dtype = torch.float32
    input_ = _make_tensor((input_size,), dtype, 801)
    hx = _make_tensor((hidden_size,), dtype, 802)
    cx = _make_tensor((hidden_size,), dtype, 803)
    weight_ih = _make_tensor((4 * hidden_size, input_size), dtype, 804)
    weight_hh = _make_tensor((4 * hidden_size, hidden_size), dtype, 805)
    bias_ih = _make_tensor((4 * hidden_size,), dtype, 806)
    bias_hh = _make_tensor((4 * hidden_size,), dtype, 807)
    module = torch.nn.LSTMCell(input_size, hidden_size)
    with torch.no_grad():
        module.weight_ih.copy_(weight_ih)
        module.weight_hh.copy_(weight_hh)
        module.bias_ih.copy_(bias_ih)
        module.bias_hh.copy_(bias_hh)
    expected = module(input_, (hx, cx))
    inputs = LstmInputs(input_, hx, cx, weight_ih, weight_hh, bias_ih, bias_hh)
    actual = _custom(inputs)
    _assert_pair_close(actual, expected, "torch.nn.LSTMCell unbatched")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_lstm_cell_known_values(dtype):
    input_ = torch.tensor([[0.25, -0.5]], dtype=dtype)
    hx = torch.tensor([[0.1, -0.2]], dtype=dtype)
    cx = torch.tensor([[0.3, -0.4]], dtype=dtype)
    weight_ih = torch.tensor(
        [
            [0.2, -0.1], [0.05, 0.3],
            [-0.3, 0.4], [0.2, 0.1],
            [0.5, -0.2], [-0.1, 0.25],
            [0.15, -0.35], [0.4, 0.05],
        ],
        dtype=dtype,
    )
    weight_hh = torch.tensor(
        [
            [0.1, 0.2], [-0.2, 0.1],
            [0.3, -0.1], [0.05, 0.2],
            [-0.25, 0.15], [0.2, -0.3],
            [0.35, 0.1], [-0.15, 0.25],
        ],
        dtype=dtype,
    )
    bias_ih = torch.linspace(-0.2, 0.2, 8, dtype=dtype)
    bias_hh = torch.linspace(0.1, -0.1, 8, dtype=dtype)
    inputs = LstmInputs(input_, hx, cx, weight_ih, weight_hh, bias_ih, bias_hh)
    actual = _custom(inputs)
    expected = _reference(inputs)
    _assert_pair_close(actual, expected, f"known-values dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_lstm_cell_saturated_gates(dtype):
    input_ = torch.tensor([[3.0, -3.0]], dtype=dtype)
    hx = torch.tensor([[2.0, -2.0]], dtype=dtype)
    cx = torch.tensor([[1.5, -1.5]], dtype=dtype)
    weight_ih = _make_tensor((8, 2), dtype, 401, low=-1.0, high=1.0)
    weight_hh = _make_tensor((8, 2), dtype, 402, low=-1.0, high=1.0)
    bias_ih = torch.tensor([2.0, -2.0, 1.5, -1.5, 0.5, -0.5, 3.0, -3.0], dtype=dtype)
    bias_hh = torch.tensor([-1.0, 1.0, 2.0, -2.0, -0.5, 0.5, -3.0, 3.0], dtype=dtype)
    inputs = LstmInputs(input_, hx, cx, weight_ih, weight_hh, bias_ih, bias_hh)
    actual = _custom(inputs)
    expected = _reference(inputs)
    _assert_pair_close(actual, expected, f"saturated dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_lstm_cell_rejects_mixed_rank_state():
    input_ = torch.randn(3, dtype=torch.float32)
    hx = torch.randn(1, 2, dtype=torch.float32)
    cx = torch.randn(1, 2, dtype=torch.float32)
    weight_ih = torch.randn(8, 3, dtype=torch.float32)
    weight_hh = torch.randn(8, 2, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(LstmInputs(input_, hx, cx, weight_ih, weight_hh, None, None))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_lstm_cell_invalid_hx_cx_shape():
    input_ = torch.randn(2, 3, dtype=torch.float32)
    hx = torch.randn(2, 2, dtype=torch.float32)
    cx = torch.randn(2, 3, dtype=torch.float32)
    weight_ih = torch.randn(8, 3, dtype=torch.float32)
    weight_hh = torch.randn(8, 2, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(LstmInputs(input_, hx, cx, weight_ih, weight_hh, None, None))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_lstm_cell_invalid_weight_shape():
    input_ = torch.randn(2, 3, dtype=torch.float32)
    hx = torch.randn(2, 2, dtype=torch.float32)
    cx = torch.randn(2, 2, dtype=torch.float32)
    weight_ih = torch.randn(7, 3, dtype=torch.float32)
    weight_hh = torch.randn(8, 2, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(LstmInputs(input_, hx, cx, weight_ih, weight_hh, None, None))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_lstm_cell_invalid_bias_shape():
    input_ = torch.randn(2, 3, dtype=torch.float32)
    hx = torch.randn(2, 2, dtype=torch.float32)
    cx = torch.randn(2, 2, dtype=torch.float32)
    weight_ih = torch.randn(8, 3, dtype=torch.float32)
    weight_hh = torch.randn(8, 2, dtype=torch.float32)
    bias_ih = torch.randn(7, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(LstmInputs(input_, hx, cx, weight_ih, weight_hh, bias_ih, None))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_lstm_cell_dtype_mismatch():
    input_ = torch.randn(2, 3, dtype=torch.float32)
    hx = torch.randn(2, 2, dtype=torch.float16)
    cx = torch.randn(2, 2, dtype=torch.float32)
    weight_ih = torch.randn(8, 3, dtype=torch.float32)
    weight_hh = torch.randn(8, 2, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(LstmInputs(input_, hx, cx, weight_ih, weight_hh, None, None))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_lstm_cell_invalid_dtype():
    input_ = torch.randint(0, 5, (2, 3), dtype=torch.int32)
    hx = torch.randint(0, 5, (2, 2), dtype=torch.int32)
    cx = torch.randint(0, 5, (2, 2), dtype=torch.int32)
    weight_ih = torch.randint(0, 5, (8, 3), dtype=torch.int32)
    weight_hh = torch.randint(0, 5, (8, 2), dtype=torch.int32)
    with pytest.raises(RuntimeError):
        _custom(LstmInputs(input_, hx, cx, weight_ih, weight_hh, None, None))
