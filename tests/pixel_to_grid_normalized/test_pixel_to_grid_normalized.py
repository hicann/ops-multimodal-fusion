#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# pixel_to_grid_normalized 算子黑盒测试（自包含单文件）。
#
# golden 来源:
#   1) 独立 Python 参考实现（与算子定义逐点对应）
#   2) 解析构造用例
# 运行（NPU）:
#   cd /tmp && pytest <repo>/tests/pixel_to_grid_normalized/ -v

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "applications" / "autonomous_driving" / "common"))

import ops  # noqa: E402,F401

DEVICE = "npu" if torch.npu.is_available() else "cpu"


def test_l0_interface_exists():
    assert hasattr(torch.ops.fusionrepo, "pixel_to_grid_normalized")


def test_l0_output_shape():
    px = torch.rand(2, 10, 2) * 100
    out = torch.ops.fusionrepo.pixel_to_grid_normalized(px.to(DEVICE), 100, 200)
    assert out.shape == (2, 10, 2)


def test_l0_invalid_last_dim():
    with pytest.raises(ValueError):
        torch.ops.fusionrepo.pixel_to_grid_normalized(torch.zeros(4, 3).to(DEVICE), 10, 10)


def test_l0_invalid_size():
    with pytest.raises(ValueError):
        torch.ops.fusionrepo.pixel_to_grid_normalized(torch.zeros(4, 2).to(DEVICE), 0, 10)


def test_l1_corner_mapping():
    # align_corners 语义: 像素 0 → -1, 末像素 → +1
    # 参数序: (pixel_coords, height, width); u 对应 width, v 对应 height
    px = torch.tensor([[0.0, 0.0], [99.0, 199.0]])
    out = torch.ops.fusionrepo.pixel_to_grid_normalized(px.to(DEVICE), 200, 100)
    # 第一点: u=0 (width=100 的首像素) → -1; v=0 (height=200 的首像素) → -1
    assert torch.allclose(out.cpu()[0], torch.tensor([-1.0, -1.0]))
    # 第二点: u=99 (width 末像素) → +1; v=199 (height=200 非末像素) → 199/199*2-1
    v_expect = 199.0 / (200 - 1) * 2 - 1
    assert torch.allclose(out.cpu()[1], torch.tensor([1.0, v_expect]))


def test_l1_center_mapping():
    # 中间像素 → 0
    px = torch.tensor([[50.0, 100.0]])
    out = torch.ops.fusionrepo.pixel_to_grid_normalized(px.to(DEVICE), 201, 101)
    assert torch.allclose(out.cpu()[0], torch.tensor([0.0, 0.0]))


def test_l1_formula_matches():
    g = torch.Generator().manual_seed(0)
    px = torch.rand(4, 50, 2, generator=g)
    px[..., 0] *= 199  # u
    px[..., 1] *= 99   # v
    height, width = 100, 200
    out = torch.ops.fusionrepo.pixel_to_grid_normalized(px.to(DEVICE), height, width)
    ref_u = px[..., 0] / (width - 1.0) * 2.0 - 1.0
    ref_v = px[..., 1] / (height - 1.0) * 2.0 - 1.0
    assert torch.allclose(out.cpu()[..., 0], ref_u, atol=1e-6)
    assert torch.allclose(out.cpu()[..., 1], ref_v, atol=1e-6)


def test_l1_single_pixel_axis():
    # 尺寸为 1 的轴: 归一化为 0 (不除零)
    px = torch.tensor([[5.0, 3.0]])
    out = torch.ops.fusionrepo.pixel_to_grid_normalized(px.to(DEVICE), 1, 10)
    # v 轴 (height=1) → 0
    assert out.cpu()[0, 1].item() == 0.0


def test_l1_deterministic():
    px = torch.rand(3, 20, 2) * 100
    a = torch.ops.fusionrepo.pixel_to_grid_normalized(px.to(DEVICE), 100, 100)
    b = torch.ops.fusionrepo.pixel_to_grid_normalized(px.to(DEVICE), 100, 100)
    assert torch.equal(a, b)
