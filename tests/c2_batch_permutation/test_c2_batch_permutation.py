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

"""Tests for ops_multimodal_fusion.c2_batch_permutation.

Caffe2-style batch row permutation forward dim 0 gather :
    Y n X indices n
where indices is a 1-D int64 tensor of length N with values in [0, N).

Golden reference: ``torch.index_select X 0 indices `` semantically identical
to Caffe2 BatchPermutation forward).

Test case counts:
  - test_c2_batch_permutation_small        : 7 cases (minimal coverage per
                                              shape/dtype/idx-mode)
  - test_c2_batch_permutation_large        : 11 cases (3-D / 4-D shapes,
                                              large N or K, special idx modes)
  - test_c2_batch_permutation_interface_exist : 1
  - test_c2_batch_permutation_empty           : 1
  - test_c2_batch_permutation_invalid_ndim    : 1
  - test_c2_batch_permutation_size_mismatch   : 1
  - test_c2_batch_permutation_indices_dim     : 1
  - test_c2_batch_permutation_indices_dtype   : 1
  - test_c2_batch_permutation_oob_indices     : 1
  - test_c2_batch_permutation_bf16_rejected   : 1
  - Total                                   : 26 cases
"""

import torch
import torch_npu  # noqa: F401
import pytest
import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "c2_batch_permutation"):
    pytest.skip(
        "ops_multimodal_fusion.c2_batch_permutation not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_c2_batch_permutation_interface_exist():
    """'ops_multimodal_fusion.c2_batch_permutation' is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "c2_batch_permutation"),\
        "The 'c2_batch_permutation' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Input generators. All deterministic via seed_key.
# ---------------------------------------------------------------------------


def _seed(seed_key):
    torch.manual_seed(abs(hash(repr(seed_key))) % 10_000_000)


def _gen_x(shape, dtype, sk):
    _seed(sk)
    if dtype == torch.int32:
        return torch.randint(-(2**20), 2**20, shape, dtype=dtype)
    return torch.randn(shape).to(dtype)


def _idx_identity(n, sk):
    return torch.arange(n, dtype=torch.int64)


def _idx_reverse(n, sk):
    return torch.arange(n - 1, -1, -1, dtype=torch.int64)


def _idx_random(n, sk):
    _seed(("idx_random", sk))
    return torch.randperm(n, dtype=torch.int64)


def _idx_all_same(n, sk):
    # All output rows pull from row N-1.
    return torch.full((n,), n - 1, dtype=torch.int64)


def _idx_repeat_pairs(n, sk):
    # First half pulls from each row twice clamped to 0 N .
    base = torch.arange(n, dtype=torch.int64)
    return torch.clamp(base // 2, max=n - 1)


_IDX_GENERATORS = {
    "identity": _idx_identity,
    "reverse": _idx_reverse,
    "random": _idx_random,
    "all_same": _idx_all_same,
    "repeat_pairs": _idx_repeat_pairs,
}


def _gen_input(shape, dtype, idx_mode, seed_key):
    x = _gen_x(shape, dtype, seed_key)
    n = x.size(0)
    indices = _IDX_GENERATORS[idx_mode](n, seed_key)
    return x, indices


# ---------------------------------------------------------------------------
# Reference: torch.index_select on dim 0 -- semantically identical to
# Caffe2 BatchPermutation forward.
# ---------------------------------------------------------------------------


def _reference(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.index_select(x.cpu(), 0, indices.cpu())


def _run(x_cpu, indices_cpu, label_str):
    expected = _reference(x_cpu, indices_cpu)
    result_npu = torch.ops.ops_multimodal_fusion.c2_batch_permutation(
        x_cpu.npu(), indices_cpu.npu()
    )
    result = result_npu.cpu()

    assert result.dtype == x_cpu.dtype,\
        f"[{label_str}] dtype: got {result.dtype}, want {x_cpu.dtype}"
    assert result.shape == x_cpu.shape,\
        f"[{label_str}] shape: got {tuple(result.shape)}, want {tuple(x_cpu.shape)}"
    # Gather is bytewise: every dtype must match exactly.
    assert torch.equal(result, expected), (
        f"[{label_str}] bytewise mismatch (shape={tuple(x_cpu.shape)}, "
        f"dtype={x_cpu.dtype})"
    )


# ---------------------------------------------------------------------------
# Case matrix.
# Labels avoid the substrings "small" / "large" so pytest -k filters don't
# accidentally collide.
# ---------------------------------------------------------------------------

# Fast cases: minimal coverage per shape / dtype / vec_width / idx mode.
# label shape dtype idx_mode
CASES_FAST = [
    # vec_width-aligned fp32 (lanes 64), simple identity gather.
    ("fp32_n8_k64_identity", (8, 64), torch.float32, "identity"),
    # vec_width-aligned fp16 (lanes 128), reverse permutation.
    ("fp16_n8_k128_reverse", (8, 128), torch.float16, "reverse"),
    # int32 non-aligned K (63 < lanes), random permutation, N just over coreNum (8 cores).
    ("int32_n9_k63_random", (9, 63), torch.int32, "random"),
    # K 1 scalar gather single batch.
    ("fp32_n1_k1_identity", (1, 1), torch.float32, "identity"),
    # fp16 multi-row-per-core path, repeat-pairs index mode, K crosses 128.
    ("fp16_n128_k129_repeat", (128, 129), torch.float16, "repeat_pairs"),
    # int32 partial-core N 7 < 8 cores all-same idx K just over lanes.
    ("int32_n7_k65_all_same", (7, 65), torch.int32, "all_same"),
    # fp32 large K within UB, random.
    ("fp32_n32_k1024_random", (32, 1024), torch.float32, "random"),
]

# Heavy cases: 3-D / 4-D shapes, large N, idx edges.
CASES_HEAVY = [
    # 3-D N H W to K H W flatten.
    ("fp32_3d_n4_8x8_identity", (4, 8, 8), torch.float32, "identity"),
    ("fp16_3d_n8_16x16_random", (8, 16, 16), torch.float16, "random"),
    # 4-D RoI feature N C H W shape typical ResNet FPN scale.
    ("fp32_4d_n16_3x7x7_random", (16, 3, 7, 7), torch.float32, "random"),
    ("fp16_4d_n8_4x14x14_reverse", (8, 4, 14, 14), torch.float16, "reverse"),
    ("int32_4d_n4_2x4x4_random", (4, 2, 4, 4), torch.int32, "random"),
    # Large batch, small K.
    ("fp32_n512_k16_random", (512, 16), torch.float32, "random"),
    # K crosses 32B alignment boundary in non-trivial way.
    ("fp32_n16_k15_random", (16, 15), torch.float32, "random"),
    ("fp16_n16_k17_reverse", (16, 17), torch.float16, "reverse"),
    # All-same idx with large K (should produce N copies of a single row).
    ("fp32_n8_k512_all_same", (8, 512), torch.float32, "all_same"),
    # Repeat-pairs at moderate size.
    ("int32_n32_k128_repeat", (32, 128), torch.int32, "repeat_pairs"),
    # Single row N 1 with larger K.
    ("fp32_n1_k1024_identity", (1, 1024), torch.float32, "identity"),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("label_str,shape,dtype,idx_mode", CASES_FAST)
def test_c2_batch_permutation_small(label_str, shape, dtype, idx_mode):
    seed_key = ("fast", label_str, tuple(shape), str(dtype), idx_mode)
    x, indices = _gen_input(shape, dtype, idx_mode, seed_key)
    _run(x, indices, label_str)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("label_str,shape,dtype,idx_mode", CASES_HEAVY)
def test_c2_batch_permutation_large(label_str, shape, dtype, idx_mode):
    seed_key = ("heavy", label_str, tuple(shape), str(dtype), idx_mode)
    x, indices = _gen_input(shape, dtype, idx_mode, seed_key)
    _run(x, indices, label_str)


# ---------------------------------------------------------------------------
# Edge tests: empty / negative-path validation.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_batch_permutation_empty():
    """N == 0 returns empty_like(X) without launching the kernel."""
    x = torch.randn(0, 4).npu()
    indices = torch.empty(0, dtype=torch.int64).npu()
    result = torch.ops.ops_multimodal_fusion.c2_batch_permutation(x, indices).cpu()
    assert tuple(result.shape) == (0, 4)
    assert result.dtype == torch.float32


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_batch_permutation_invalid_ndim():
    """X must have at least 1 dim. Build a 0-D tensor via reshape."""
    x = torch.tensor(3.0).npu()
    indices = torch.tensor([0], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_batch_permutation(x, indices)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_batch_permutation_size_mismatch():
    """indices.size(0) must equal X.size(0)."""
    x = torch.randn(4, 8).npu()
    indices = torch.tensor([0, 1, 2], dtype=torch.int64).npu()  # wrong length
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_batch_permutation(x, indices)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_batch_permutation_indices_dim():
    """indices must be 1-D."""
    x = torch.randn(4, 8).npu()
    indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64).npu()  # 2 D
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_batch_permutation(x, indices)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_batch_permutation_indices_dtype():
    """indices dtype must be int64 (int32 must be rejected)."""
    x = torch.randn(4, 8).npu()
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_batch_permutation(x, indices)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_batch_permutation_oob_indices():
    """indices values must lie in [0, N); negative or >= N is rejected on host."""
    x = torch.randn(4, 8).npu()
    indices_high = torch.tensor([0, 1, 2, 4], dtype=torch.int64).npu()  # 4 == N
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_batch_permutation(x, indices_high)
    indices_neg = torch.tensor([0, 1, -1, 3], dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_batch_permutation(x, indices_neg)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_c2_batch_permutation_bf16_rejected():
    """bf16 / int64 / fp64 are deferred on this platform."""
    x = torch.randn(4, 8, dtype=torch.bfloat16).npu()
    indices = torch.arange(4, dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.c2_batch_permutation(x, indices)
