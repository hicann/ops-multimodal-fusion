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

"""Tests for ops_multimodal_fusion.c2_accuracy.

Caffe2-style top-k classification accuracy. Per row, computes the one based
rank of the label score and marks the row correct when that rank is within
top_k; output is the mean correctness across rows as an fp32 scalar.

Tie rule from the CUDA formula in caffe2 operators accuracy_op:
    rank counts columns j over the row width where the prediction at j
    beats the prediction at the label column, or where it equals the label
    prediction and column j is below the label column.

Test case counts:
  - test_c2_accuracy_small : 7 cases one dtype top_k per fast shape
  - test_c2_accuracy_large         : 14 cases, full coverage with tie and boundary scenarios
  - test_c2_accuracy_interface_exist : 1
  - test_c2_accuracy_invalid_dim   : 1
  - test_c2_accuracy_label_mismatch: 1
  - test_c2_accuracy_topk_zero     : 1
  - test_c2_accuracy_bf16_rejected : 1
  - test_c2_accuracy_int64_rejected: 1
  - Total                          : 27 cases
"""

import torch
import torch_npu  # noqa: F401  (registers NPU dispatch key)
import pytest
import ops_multimodal_fusion  # noqa: F401  (registers torch.ops.ops_multimodal_fusion.c2_accuracy)


if not hasattr(torch.ops.ops_multimodal_fusion, "c2_accuracy"):
    pytest.skip(
        "ops_multimodal_fusion.c2_accuracy not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_c2_accuracy_interface_exist():
    """The 'ops_multimodal_fusion.c2_accuracy' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "c2_accuracy"),\
        "The 'c2_accuracy' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Input generators. All deterministic via seed_key for reproducibility.
# ---------------------------------------------------------------------------


def _seed(seed_key):
    torch.manual_seed(abs(hash(repr(seed_key))) % 10_000_000)


def _gen_randn(shape, dtype, sk, opts):
    _seed(sk)
    pred = torch.randn(shape).to(dtype)
    n = shape[0]
    d = shape[1]
    label = torch.randint(0, d, (n,), dtype=torch.int32)
    return pred, label


def _gen_scaled(shape, dtype, sk, opts):
    """Scale randn by the requested factor to widen value range.

    A wider range lowers the fp16 tie probability so randomised cases
    exercise the non-tie path rather than the tie path.
    """
    _seed(sk)
    scale = opts.get("scale", 100.0)
    pred = (torch.randn(shape) * scale).to(dtype)
    n = shape[0]
    d = shape[1]
    label = torch.randint(0, d, (n,), dtype=torch.int32)
    return pred, label


def _gen_const(shape, dtype, sk, opts):
    """Fill pred with one constant so every position ties.

    The label is fixed for every row, which verifies the tie-break that
    keeps a column when its index is at or below the label.
    """
    _seed(sk)
    pred = torch.full(shape, float(opts.get("value", 5.0))).to(dtype)
    label_t = torch.full((shape[0],), int(opts.get("label", 0)), dtype=torch.int32)
    return pred, label_t


def _gen_label_argmax(shape, dtype, sk, opts):
    """Set label to the argmax column so accuracy should reach 1.0."""
    _seed(sk)
    pred = torch.randn(shape).to(dtype)
    label = torch.argmax(pred, dim=1).to(torch.int32)
    return pred, label


def _gen_label_argmin(shape, dtype, sk, opts):
    """Set label to the argmin column.

    Accuracy should be 0.0 for top_k of one because the label sits at the
    bottom of the ranking.
    """
    _seed(sk)
    pred = torch.randn(shape).to(dtype)
    label = torch.argmin(pred, dim=1).to(torch.int32)
    return pred, label


def _gen_label_fixed(shape, dtype, sk, opts):
    """Random pred, all labels fixed at the given column.

    Probes label boundary positions such as the first and last column.
    """
    _seed(sk)
    pred = torch.randn(shape).to(dtype)
    label_t = torch.full((shape[0],), int(opts.get("label", 0)), dtype=torch.int32)
    return pred, label_t


def _gen_pred_top_run(shape, dtype, sk, opts):
    """Make the first run of columns equal and larger than the rest.

    Remaining columns are random and smaller than the run value, and label
    varies per row. This exercises the tie-break so rows whose label falls
    inside the run get a small rank while later labels rank below the run.
    """
    _seed(sk)
    run_count = opts.get("run_count", 5)
    run_value = opts.get("run_value", 1.0)
    n, d = shape
    assert run_count <= d
    rng = torch.randn(n, d)
    rng = rng - rng.max(dim=1, keepdim=True).values - 1.0  # all entries below zero
    pred = rng + float(run_value)
    pred[:, :run_count] = float(run_value)
    pred = pred.to(dtype)
    label = torch.randint(0, d, (n,), dtype=torch.int32)
    return pred, label


_GENERATORS = {
    "randn": _gen_randn,
    "scaled": _gen_scaled,
    "const": _gen_const,
    "argmax": _gen_label_argmax,
    "argmin": _gen_label_argmin,
    "fixed_label": _gen_label_fixed,
    "top_run": _gen_pred_top_run,
}


def _gen_input(gen_name, shape, dtype, seed_key, gen_kwargs):
    return _GENERATORS[gen_name](shape, dtype, seed_key, gen_kwargs)


# ---------------------------------------------------------------------------
# CPU reference following caffe2 AccuracyKernel.
# ---------------------------------------------------------------------------


def _reference(pred: torch.Tensor, label: torch.Tensor, top_k: int) -> torch.Tensor:
    pred_cpu = pred.cpu()
    label_cpu = label.cpu().to(torch.int64)
    n, d = pred_cpu.shape
    label_pred = pred_cpu.gather(1, label_cpu.unsqueeze(1)).squeeze(1)
    cols = torch.arange(d).unsqueeze(0).expand(n, d)
    label_b = label_cpu.unsqueeze(1)
    label_pred_b = label_pred.unsqueeze(1)
    mask_gt = pred_cpu > label_pred_b
    mask_eq = pred_cpu == label_pred_b
    mask_le = cols <= label_b
    combined = mask_gt | (mask_eq & mask_le)
    rank = combined.sum(dim=1)
    correct = (rank <= top_k).to(torch.int32).sum().to(torch.float32)
    return correct / float(n)


def _run(pred_cpu, label_cpu, top_k, label_str):
    expected = _reference(pred_cpu, label_cpu, top_k)
    result_npu = torch.ops.ops_multimodal_fusion.c2_accuracy(
        pred_cpu.npu(), label_cpu.npu(), int(top_k)
    )
    result = result_npu.cpu()

    assert result.dtype == torch.float32,\
        f"[{label_str}] output dtype: got {result.dtype}, want torch.float32"
    assert result.dim() == 0,\
        f"[{label_str}] output ndim: got {result.dim()}, want 0"
    # accuracy is the correct count over N, exact in fp32 when count and N
    # fit losslessly, which always holds here since N stays under four
    # thousand and count never exceeds N. The reference path uses the same
    # arithmetic, so byte-equal is the natural assertion.
    assert torch.equal(result, expected), (
        f"[{label_str}] accuracy mismatch (shape={tuple(pred_cpu.shape)}, "
        f"top_k={top_k}, dtype={pred_cpu.dtype})\n"
        f"expected={expected.item()}\nresult ={result.item()}"
    )


# ---------------------------------------------------------------------------
# Case matrix.
# Each tuple holds label_str, shape, pred_dtype, gen_name, gen_kwargs, top_k.
# Labels and generator names avoid the substrings "small" or "large" so that
# the pytest keyword filter for fast cases never collides.
# ---------------------------------------------------------------------------

CASES_FAST = [
    ("rand_n1", (1, 10), torch.float32, "randn", {}, 1),
    ("vec_eq", (8, 64), torch.float32, "randn", {}, 1),
    ("vec_minus", (16, 63), torch.float32, "randn", {}, 1),
    ("vec_plus", (16, 65), torch.float32, "randn", {}, 5),
    ("fp16_chunk", (16, 128), torch.float16, "scaled", {"scale": 100.0}, 5),
    ("tie_mid", (4, 16), torch.float32, "const", {"value": 5.0, "label": 7}, 8),
    ("topk_eq_D", (4, 32), torch.float32, "randn", {}, 32),
]

CASES_HEAVY = [
    ("imagenet_k1", (64, 1000), torch.float32, "randn", {}, 1),
    ("imagenet_k5", (64, 1000), torch.float32, "randn", {}, 5),
    ("saturate_n256", (256, 100), torch.float32, "randn", {}, 5),
    ("huge_n_4k", (4096, 100), torch.float32, "randn", {}, 5),
    ("fp16_chunk_big", (32, 256), torch.float16, "scaled", {"scale": 100.0}, 5),
    ("tie_end", (4, 16), torch.float32, "const", {"value": 5.0, "label": 15}, 1),
    ("topk_gt_D", (4, 32), torch.float32, "randn", {}, 100),
    ("label_eq_max", (4, 64), torch.float32, "argmax", {}, 1),
    ("label_eq_min", (4, 64), torch.float32, "argmin", {}, 1),
    ("label_zero", (8, 100), torch.float32, "fixed_label", {"label": 0}, 5),
    ("label_last", (8, 100), torch.float32, "fixed_label", {"label": 99}, 5),
    ("pred_top5_tie", (4, 10), torch.float32, "top_run", {"run_count": 5, "run_value": 1.0}, 3),
    ("fp16_k1", (8, 64), torch.float16, "scaled", {"scale": 100.0}, 1),
    ("vec_2d_k5", (8, 32), torch.float32, "randn", {}, 5),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_FAST)
def test_c2_accuracy_small(case):
    label_str, shape, dtype, gen_name, gen_kwargs, top_k = case
    seed_key = ("fast", label_str, tuple(shape), str(dtype), gen_name,
                tuple(sorted(gen_kwargs.items())), int(top_k))
    pred, label = _gen_input(gen_name, shape, dtype, seed_key, gen_kwargs)
    _run(pred, label, top_k, label_str)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", CASES_HEAVY)
def test_c2_accuracy_large(case):
    label_str, shape, dtype, gen_name, gen_kwargs, top_k = case
    seed_key = ("heavy", label_str, tuple(shape), str(dtype), gen_name,
                tuple(sorted(gen_kwargs.items())), int(top_k))
    pred, label = _gen_input(gen_name, shape, dtype, seed_key, gen_kwargs)
    _run(pred, label, top_k, label_str)


# ---------------------------------------------------------------------------
# Negative-path tests: argument and dtype validation.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_accuracy_invalid_dim():
    """Predictions must be 2 dim."""
    pred = torch.randn(8).npu()  # 1 dim
    label = torch.randint(0, 8, (8,), dtype=torch.int32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_accuracy(pred, label, 1)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_accuracy_label_mismatch():
    """Label count must equal the prediction row count."""
    pred = torch.randn(8, 16).npu()
    label = torch.randint(0, 16, (7,), dtype=torch.int32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_accuracy(pred, label, 1)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_accuracy_topk_zero():
    """top_k must be at least one."""
    pred = torch.randn(8, 16).npu()
    label = torch.randint(0, 16, (8,), dtype=torch.int32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_accuracy(pred, label, 0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_accuracy_bf16_rejected():
    """bf16 predictions are deferred on this platform."""
    pred = torch.randn(8, 16, dtype=torch.bfloat16).npu()
    label = torch.randint(0, 16, (8,), dtype=torch.int32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_accuracy(pred, label, 1)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_accuracy_int64_rejected():
    """int64 labels are deferred since the int64 cast binary is missing."""
    pred = torch.randn(8, 16).npu()
    label = torch.randint(0, 16, (8,), dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_accuracy(pred, label, 1)
