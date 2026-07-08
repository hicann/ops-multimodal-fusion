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

import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "weight_norm"):
    pytest.skip(
        "ops_multimodal_fusion.weight_norm not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float16, torch.float32]


def _make_tensor(shape, dtype, seed=0, low=-1.0, high=1.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(*shape, dtype=torch.float32).uniform_(low, high, generator=g)
    if x.numel() > 8:
        flat = x.flatten()
        flat[1::5] = 0.25
        flat[3::7] = -0.5
    return x.to(dtype)


def _normalize_dim(dim, ndim):
    return dim + ndim if dim < 0 else dim


def _norm_shape(shape, dim):
    dim = _normalize_dim(dim, len(shape))
    out = [1] * len(shape)
    out[dim] = shape[dim]
    return tuple(out)


def _make_g(shape, dim, dtype, seed=100, one_dim=False):
    dim = _normalize_dim(dim, len(shape))
    c = shape[dim]
    g = _make_tensor((c,), dtype, seed=seed, low=0.25, high=1.5)
    return g if one_dim else g.reshape(_norm_shape(shape, dim))


def _reference(v, g, dim):
    raw_dim = dim
    dim = _normalize_dim(dim, v.dim())
    vf = v.float()
    if raw_dim == -1:
        # torch._weight_norm treats a literal dim of negative one as a global
        # full-tensor L2 norm, taking the norm over all elements rather than a
        # per-last-axis-channel norm. A positive last-axis index stays per-axis.
        norm = torch.sqrt(torch.sum(vf * vf))
    else:
        reduce_dims = tuple(i for i in range(v.dim()) if i != dim)
        if reduce_dims:
            norm = torch.sqrt(torch.sum(vf * vf, dim=reduce_dims, keepdim=True))
        else:
            norm = torch.sqrt(vf * vf)
    gf = g.float().reshape(_norm_shape(tuple(v.shape), dim)) if g.dim() == 1 else g.float()
    return (vf * gf / norm).to(v.dtype)


def _custom(v, g, dim):
    return torch.ops.ops_multimodal_fusion.weight_norm(v.npu(), g.npu(), dim).cpu()


def _assert_close(actual, expected, label):
    assert actual.dtype == expected.dtype, (
        f"{label}: dtype mismatch {actual.dtype} vs {expected.dtype}"
    )
    assert actual.shape == expected.shape, (
        f"{label}: shape mismatch {actual.shape} vs {expected.shape}"
    )
    rtol, atol = (1e-2, 1e-2) if expected.dtype == torch.float16 else (1e-4, 1e-4)
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True), (
        f"{label}: max_abs="
        f"{torch.nan_to_num(actual - expected).abs().max().item() if actual.numel() else 0.0} "
        f"actual={actual.flatten()[:8]} expected={expected.flatten()[:8]}"
    )


def test_weight_norm_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "weight_norm")


CASES = [
    ((4,), 0, False),
    ((2, 3), 0, False),
    ((2, 3), 0, True),
    ((2, 3), -1, True),
    ((2, 3, 4), 1, False),
    ((2, 3, 4), 1, True),
    ((2, 3, 4), -1, False),
    ((2, 3, 4, 5), 0, False),
    ((2, 3, 4, 5), 2, True),
    ((2, 3, 2, 4, 3), -2, False),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,dim,one_dim_g", CASES)
def test_weight_norm_matches_reference(dtype, shape, dim, one_dim_g):
    seed = sum(shape) * 13 + _normalize_dim(dim, len(shape)) * 17
    v = _make_tensor(shape, dtype, seed=seed)
    g = _make_g(shape, dim, dtype, seed=seed + 1, one_dim=one_dim_g)
    actual = _custom(v, g, dim)
    expected = _reference(v, g, dim)
    _assert_close(actual, expected, f"dtype={dtype} shape={shape} dim={dim}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_weight_norm_known_values_dim0(dtype):
    v = torch.tensor([[3.0, 4.0], [0.0, 5.0]], dtype=dtype)
    g = torch.tensor([[10.0], [2.0]], dtype=dtype)
    actual = _custom(v, g, 0)
    expected = torch.tensor([[6.0, 8.0], [0.0, 2.0]], dtype=dtype)
    _assert_close(actual, expected, f"known-values-dim0 dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_weight_norm_large_inner_multi_tile(dtype):
    v = _make_tensor((1, 2, 5000), dtype, seed=501, low=-0.5, high=0.5)
    g = _make_g(tuple(v.shape), 1, dtype, seed=502, one_dim=True)
    actual = _custom(v, g, 1)
    expected = _reference(v, g, 1)
    _assert_close(actual, expected, f"multi-tile dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_weight_norm_zero_norm_matches_formula_nan():
    v = torch.zeros((2, 3), dtype=torch.float32)
    g = torch.ones((2, 1), dtype=torch.float32)
    actual = _custom(v, g, 0)
    expected = _reference(v, g, 0)
    assert torch.isnan(expected).all()
    assert torch.isnan(actual).all()


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_weight_norm_invalid_empty_tensor():
    v = torch.empty((0, 3), dtype=torch.float32)
    g = torch.empty((0, 1), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(v, g, 0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_weight_norm_invalid_dim():
    v = torch.randn(2, 3, dtype=torch.float32)
    g = torch.randn(2, 1, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(v, g, 2)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_weight_norm_invalid_g_shape():
    v = torch.randn(2, 3, dtype=torch.float32)
    g = torch.randn(2, 2, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _custom(v, g, 0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_weight_norm_dtype_mismatch():
    v = torch.randn(2, 3, dtype=torch.float32)
    g = torch.randn(2, 1, dtype=torch.float16)
    with pytest.raises(RuntimeError):
        _custom(v, g, 0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_weight_norm_invalid_dtype():
    v = torch.randint(0, 5, (2, 3), dtype=torch.int32)
    g = torch.randint(0, 5, (2, 1), dtype=torch.int32)
    with pytest.raises(RuntimeError):
        _custom(v, g, 0)
