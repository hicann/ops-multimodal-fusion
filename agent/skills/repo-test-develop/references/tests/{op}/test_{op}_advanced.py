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

# TEMPLATE: 进阶模板——菜单式，融合两类写法：
# TEMPLATE:   Part A 带属性算子的属性参数化（参考 tests/avg_pool2d/test_avg_pool2d.py）
# TEMPLATE:   Part B 边界 / 异常用例（参考 tests/angle/test_angle.py）
# TEMPLATE: 复制到 tests/{op}/test_{op}.py，全局替换 {op}(snake) / {Op}(Pascal) / {OP}(UPPER)，
# TEMPLATE: 并按算子类型裁剪不需要的 Part。

import logging
import math

import pytest
import torch
import torch.nn.functional as F  # noqa: F401  # Part A（带属性算子）golden 用
import torch_npu  # noqa: F401

import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "{op}"):
    pytest.skip(
        "ops_multimodal_fusion.{op} not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_{op}_interface_exist():
    """Test that the 'ops_multimodal_fusion.{op}' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "{op}"),\
        "The '{op}' operator is not registered in the 'torch.ops.ops_multimodal_fusion' namespace."


DTYPES = [torch.float32, torch.float16]


def _dtype_tol(dtype):
    """Per-dtype tolerance vs the PyTorch CPU golden.
    f32: 1e-6 / 1e-6；f16: 5e-3 / 5e-3（半精度 + 舍入误差）。
    """
    if dtype == torch.float32:
        return dict(rtol=1e-6, atol=1e-6)
    return dict(rtol=5e-3, atol=5e-3)


# ---------------------------------------------------------------------------
# Part A：带属性算子的属性参数化（参考 avg_pool2d）。
# TEMPLATE: 纯 elementwise 无属性算子请整段删除 Part A。
# ---------------------------------------------------------------------------

# TEMPLATE: 每行 = (input_shape, attr1, attr2, ...)，按 Schema 参数顺序打包，覆盖各属性分支组合。
# TEMPLATE: 下例为 avg_pool2d 的 (shape, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)。
{OP}_CONFIGS = [
    # 基本非重叠
    ((1, 1, 4, 4), (2, 2), (2, 2), (0, 0), False, True, None),
    ((2, 16, 32, 32), (2, 2), (2, 2), (0, 0), False, True, None),
    # 带 padding / count_include_pad / ceil_mode / divisor_override / 非方核 / stride 默认(空)
    ((1, 3, 8, 8), (3, 3), (1, 1), (1, 1), False, False, None),
    ((1, 1, 5, 5), (3, 3), (2, 2), (0, 0), True, True, None),
    ((1, 1, 4, 4), (2, 2), (2, 2), (0, 0), False, True, 2),
    ((1, 1, 8, 8), (2, 3), (1, 1), (0, 0), False, True, None),
    ((1, 3, 8, 8), (2, 2), (), (0, 0), False, True, None),
    # 大 shape 超 UB 容量，强制多 tile
    ((8, 3, 128, 128), (2, 2), (2, 2), (0, 0), False, True, None),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", {OP}_CONFIGS)
def test_{op}_attributes(case):
    """Test the {op} operator (with attributes) against the PyTorch CPU reference."""
    (shape, kernel_size, stride, padding, ceil_mode,
     count_include_pad, divisor_override) = case
    a = torch.randn(*shape, dtype=torch.float32)

    # TEMPLATE: golden = 带属性算子的 PyTorch CPU 参考（如 F.avg_pool2d）。
    expected = F.avg_pool2d(
        a,
        kernel_size=kernel_size,
        stride=stride if stride else None,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    a_npu = a.npu()
    # TEMPLATE: 属性以 python list 传入 NPU 算子，顺序与 Schema 一致。
    result_npu = torch.ops.ops_multimodal_fusion.{op}(
        a_npu,
        list(kernel_size),
        list(stride),
        list(padding),
        ceil_mode,
        count_include_pad,
        divisor_override,
    )
    result = result_npu.cpu()

    assert result.shape == expected.shape,\
        f"Shape mismatch: got {result.shape}, expected {expected.shape}"
    # 累加 / 属性类容差放宽到 1e-3。
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3),\
        f"{Op} failed for shape {shape}. " \
        f"Max diff: {torch.max(torch.abs(result - expected)):.6f}"


# ---------------------------------------------------------------------------
# Part B：边界 / 异常用例（参考 angle）。适用于所有算子，按需裁剪。
# TEMPLATE: golden 一律用 PyTorch CPU 现算，示例用 torch.angle，请替换为本算子参考。
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_{op}_named_values(dtype):
    """Expected outputs at hand-picked signed / edge values within each dtype's resolution."""
    # TEMPLATE: 固定值覆盖 正 / 负 / 零 / 极大极小。
    values = [1.0, 2.0, 0.1, -1.0, -2.0, -0.1, 0.0, -0.0, 100.0, -100.0]
    xs = torch.tensor(values, dtype=dtype)
    expected = torch.angle(xs)  # TEMPLATE: 换成本算子的 PyTorch CPU 参考
    y = torch.ops.ops_multimodal_fusion.{op}(xs.npu()).cpu()

    assert torch.allclose(y, expected, **_dtype_tol(dtype)), (
        f"{op} named-values mismatch (dtype={dtype}); "
        f"got {y.tolist()} expected {expected.tolist()}"
    )
    logging.info(f"Named-values test passed (dtype={dtype}).")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_{op}_nan_propagates(dtype):
    """NaN input must produce NaN output."""
    xs = torch.tensor([float('nan'), 1.0, -1.0, float('nan')], dtype=dtype)
    y = torch.ops.ops_multimodal_fusion.{op}(xs.npu()).cpu()
    assert torch.isnan(y[0]).item() and torch.isnan(y[3]).item(),\
        f"NaN did not propagate (dtype={dtype}): got {y.tolist()}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_{op}_empty_tensor(dtype):
    """Empty tensor passes through with matching shape/dtype (no kernel launch)."""
    x = torch.empty((0,), dtype=dtype).npu()
    y = torch.ops.ops_multimodal_fusion.{op}(x).cpu()
    assert y.shape == (0,)
    assert y.dtype == dtype


@pytest.mark.skip(
    reason=(
        "torch_npu in the current CANN release lacks D2D strided copy support "
        "(aclnnInplaceCopy fails with error 561103 for any non-contiguous NPU "
        "tensor). The kernel only operates on contiguous buffers; callers holding "
        "a transposed/strided tensor must materialize it before invoking the op. "
        "Re-enable once torch_npu ships strided D2D."
    )
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_{op}_non_contiguous_input(dtype):
    """Non-contiguous tensors are handled (kernel makes a contiguous copy)."""
    base = torch.randn(32, 32, dtype=dtype)
    x = base.t()  # transpose -> non-contiguous
    assert not x.is_contiguous()

    y = torch.ops.ops_multimodal_fusion.{op}(x.npu()).cpu()
    expected = torch.angle(x.contiguous())  # TEMPLATE: 换成本算子的 PyTorch CPU 参考
    assert torch.allclose(y, expected, **_dtype_tol(dtype)), (
        f"non-contiguous mismatch (dtype={dtype}): "
        f"max abs diff = {(y - expected).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_{op}_rejects_unsupported_dtype():
    """float64 (and other unsupported dtypes) must error cleanly via TORCH_CHECK."""
    xs = torch.tensor([1.0, -1.0], dtype=torch.float64).npu()
    with pytest.raises(RuntimeError, match="{op}"):
        torch.ops.ops_multimodal_fusion.{op}(xs)
