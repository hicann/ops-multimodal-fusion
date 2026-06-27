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
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "zeta"):
    pytest.skip(
        "ops_multimodal_fusion.zeta not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_zeta_interface_exist():
    """The 'ops_multimodal_fusion.zeta' operator must be registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "zeta"), \
        "The 'zeta' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


SHAPES = [
    (1,),
    (7,),
    (1024,),
    (10000,),
    (32, 32),
    (100, 100),
    (256, 512),
    (16, 32, 64),
    (1, 3, 32, 32),
    (100000,),
]

SPECIAL_SHAPES = [
    (8,),
    (9,),
    (15,),
    (16,),
    (17,),
    (3, 5),
    (2, 3, 7),
]

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _sample_x(shape, seed):
    """Sample x in a stable Hurwitz-zeta domain, away from the pole at unity."""
    g = torch.Generator().manual_seed(seed)
    return torch.empty(*shape, dtype=torch.float32).uniform_(1.25, 8.0, generator=g)


def _sample_q(shape, seed):
    """Sample a positive q, since positive q is the real-valued Hurwitz-zeta domain."""
    g = torch.Generator().manual_seed(seed)
    return torch.empty(*shape, dtype=torch.float32).uniform_(0.5, 12.0, generator=g)


def _assert_close(actual, expected, msg):
    if actual.dtype == torch.float32:
        rtol, atol = 5e-3, 5e-4
    else:
        rtol, atol = 1.5e-2, 2e-3
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"{msg}: max abs diff = {(actual - expected).abs().max().item()}, "
        f"max rel diff = "
        f"{((actual - expected) / expected.abs().clamp_min(1e-30)).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_zeta_operator(shape, dtype):
    """Compare NPU Hurwitz Zeta against torch.special.zeta on valid inputs."""
    seed = abs(hash((shape, dtype))) % 10_000_000
    x = _sample_x(shape, seed=seed).to(dtype)
    q = _sample_q(shape, seed=seed + 1).to(dtype)

    y = torch.ops.ops_multimodal_fusion.zeta(x.npu(), q.npu()).cpu()
    expected = torch.special.zeta(x.to(torch.float64), q.to(torch.float64)).to(dtype)

    assert y.dtype == dtype, f"dtype mismatch: {y.dtype} vs {dtype}"
    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
    _assert_close(y, expected, f"zeta mismatch for shape={shape}, dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_zeta_special_shapes(shape, dtype):
    """Cover 32-byte boundaries, tail padding, and non-aligned multi-dim tiles."""
    seed = abs(hash(("special", shape, dtype))) % 10_000_000
    x = _sample_x(shape, seed=seed).to(dtype)
    q = _sample_q(shape, seed=seed + 1).to(dtype)

    y = torch.ops.ops_multimodal_fusion.zeta(x.npu(), q.npu()).cpu()
    expected = torch.special.zeta(x.to(torch.float64), q.to(torch.float64)).to(dtype)

    assert y.dtype == dtype, f"dtype mismatch: {y.dtype} vs {dtype}"
    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
    _assert_close(y, expected, f"zeta special-shape mismatch for shape={shape}, dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_zeta_named_values():
    """Known values and stable spot checks."""
    x = torch.tensor([2.0, 2.0, 3.0, 4.0, 6.0, 8.0], dtype=torch.float32)
    q = torch.tensor([1.0, 2.0, 1.0, 0.5, 3.0, 10.0], dtype=torch.float32)

    y = torch.ops.ops_multimodal_fusion.zeta(x.npu(), q.npu()).cpu()
    expected = torch.special.zeta(x.to(torch.float64), q.to(torch.float64)).to(torch.float32)
    _assert_close(y, expected, "zeta named-value mismatch")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_zeta_dtype_roundtrip(dtype):
    """Zeta follows torch.special.zeta dtype behavior for supported floating dtypes."""
    x = torch.tensor([1.5, 2.0, 3.0, 4.0], dtype=dtype)
    q = torch.tensor([0.75, 1.0, 2.0, 8.0], dtype=dtype)

    y = torch.ops.ops_multimodal_fusion.zeta(x.npu(), q.npu()).cpu()
    expected = torch.special.zeta(x.to(torch.float64), q.to(torch.float64)).to(dtype)

    assert y.dtype == dtype
    _assert_close(y, expected, f"zeta dtype mismatch for dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_zeta_broadcast_inputs():
    """Broadcasted x/q inputs are handled by the zeta kernel path."""
    x = torch.tensor([[2.0], [3.0], [4.0]], dtype=torch.float32)
    q = torch.tensor([[1.0, 2.0, 4.0, 8.0]], dtype=torch.float32)

    y = torch.ops.ops_multimodal_fusion.zeta(x.npu(), q.npu()).cpu()
    expected = torch.special.zeta(x.to(torch.float64), q.to(torch.float64)).to(torch.float32)

    assert y.shape == (3, 4)
    _assert_close(y, expected, "zeta broadcast mismatch")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "x_shape,q_shape",
    [
        ((17, 1), (1, 9)),
        ((1, 17), (3, 17)),
    ],
)
def test_zeta_broadcast_special_shapes(x_shape, q_shape):
    """Broadcast cases with non-aligned rows/columns and scalar expansion."""
    seed = abs(hash(("broadcast", x_shape, q_shape))) % 10_000_000
    x = _sample_x(x_shape, seed=seed)
    q = _sample_q(q_shape, seed=seed + 1)

    y = torch.ops.ops_multimodal_fusion.zeta(x.npu(), q.npu()).cpu()
    expected = torch.special.zeta(x.to(torch.float64), q.to(torch.float64)).to(torch.float32)

    assert y.shape == expected.shape
    _assert_close(y, expected, f"zeta broadcast-special mismatch for x={x_shape}, q={q_shape}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_zeta_empty_tensor():
    """Empty tensors pass through with broadcasted shape and dtype."""
    x = torch.empty((0, 3), dtype=torch.float32).npu()
    q = torch.empty((1, 3), dtype=torch.float32).npu()
    y = torch.ops.ops_multimodal_fusion.zeta(x, q).cpu()
    assert y.shape == (0, 3)
    assert y.dtype == torch.float32


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_zeta_rejects_unsupported_dtype():
    """Only floating dtypes covered by torch.special.zeta on dav-3510 are accepted."""
    for unsupported in [torch.float64, torch.int32]:
        x = torch.full((4,), 2.0, dtype=unsupported).npu()
        q = torch.full((4,), 1.0, dtype=unsupported).npu()
        with pytest.raises(RuntimeError, match="zeta"):
            torch.ops.ops_multimodal_fusion.zeta(x, q)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_zeta_rejects_dtype_mismatch():
    """Mismatched dtypes must raise before launch."""
    x = torch.full((4,), 2.0, dtype=torch.float32).npu()
    q = torch.full((4,), 1.0, dtype=torch.float64).npu()
    with pytest.raises(RuntimeError, match="zeta"):
        torch.ops.ops_multimodal_fusion.zeta(x, q)
