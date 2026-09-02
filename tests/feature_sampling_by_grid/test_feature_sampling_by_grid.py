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
# feature_sampling_by_grid 算子黑盒测试（自包含单文件）。
#
# golden 来源:
#   1) 独立 Python 参考实现（与算子定义逐点对应）
#   2) 解析构造用例
# 运行（NPU）:
#   cd /tmp && pytest <repo>/tests/feature_sampling_by_grid/ -v

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "applications" / "autonomous_driving" / "common"))

import ops  # noqa: E402,F401

DEVICE = "npu" if torch.npu.is_available() else "cpu"


def test_l0_interface_exists():
    assert hasattr(torch.ops.fusionrepo, "feature_sampling_by_grid")


def test_l0_output_shape_4d_grid():
    fmap = torch.randn(2, 8, 20, 30)
    grid = torch.rand(2, 50, 2) * 2 - 1
    out = torch.ops.fusionrepo.feature_sampling_by_grid(fmap.to(DEVICE), grid.to(DEVICE))
    assert out.shape == (2, 8, 50)


def test_l0_output_shape_3d_grid():
    # grid (B, N, 2) 与 (B, N, 1, 2) 均支持
    fmap = torch.randn(1, 4, 10, 10)
    grid3 = torch.rand(1, 20, 2) * 2 - 1
    grid4 = grid3.unsqueeze(2)
    a = torch.ops.fusionrepo.feature_sampling_by_grid(fmap.to(DEVICE), grid3.to(DEVICE))
    b = torch.ops.fusionrepo.feature_sampling_by_grid(fmap.to(DEVICE), grid4.to(DEVICE))
    assert a.shape == (1, 4, 20) and torch.equal(a, b)


def test_l0_invalid_dim():
    with pytest.raises(ValueError):
        torch.ops.fusionrepo.feature_sampling_by_grid(torch.randn(10, 10).to(DEVICE),
                                                      torch.rand(5, 2).to(DEVICE))


def test_l1_identity_grid():
    # 网格恰为像素中心 (align_corners=True): 输出 = 该位置特征
    fmap = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    grid = torch.tensor([[[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]])
    out = torch.ops.fusionrepo.feature_sampling_by_grid(fmap.to(DEVICE), grid.to(DEVICE))
    # (-1,-1)=角(0,0), (1,-1)=(0,3), (-1,1)=(3,0), (1,1)=(3,3)
    assert torch.allclose(out.cpu()[0, 0], torch.tensor([0.0, 3.0, 12.0, 15.0]))


def test_l1_matches_grid_sample():
    g = torch.Generator().manual_seed(0)
    fmap = torch.randn(2, 6, 15, 20, generator=g)
    grid = torch.rand(2, 40, 2, generator=g) * 2 - 1
    ref = F.grid_sample(fmap, grid.unsqueeze(2), mode="bilinear", align_corners=True).squeeze(-1)
    out = torch.ops.fusionrepo.feature_sampling_by_grid(fmap.to(DEVICE), grid.to(DEVICE))
    assert torch.allclose(out.cpu(), ref, atol=1e-5)


def test_l1_bilinear_midpoint():
    # 网格中心 (0,0) → 4 像素均值 (2x2 图)
    fmap = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    grid = torch.tensor([[[0.0, 0.0]]])
    out = torch.ops.fusionrepo.feature_sampling_by_grid(fmap.to(DEVICE), grid.to(DEVICE))
    assert torch.allclose(out.cpu()[0, 0, 0], torch.tensor([2.5]))


def test_l1_deterministic():
    fmap = torch.randn(1, 4, 10, 10)
    grid = torch.rand(1, 30, 2) * 2 - 1
    a = torch.ops.fusionrepo.feature_sampling_by_grid(fmap.to(DEVICE), grid.to(DEVICE))
    b = torch.ops.fusionrepo.feature_sampling_by_grid(fmap.to(DEVICE), grid.to(DEVICE))
    assert torch.equal(a, b)
