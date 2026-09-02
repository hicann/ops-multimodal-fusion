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
# voxel_projection 算子黑盒测试（自包含单文件）。
#
# golden 来源:
#   1) 独立 Python 参考实现（与算子定义逐点对应）
#   2) 解析构造用例
# 运行（NPU）:
#   cd /tmp && pytest <repo>/tests/voxel_projection/ -v

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "applications" / "autonomous_driving" / "common"))

import ops  # noqa: E402,F401

DEVICE = "npu" if torch.npu.is_available() else "cpu"


def _proj_matrix():
    # 单位投影矩阵 (3x4): 像素坐标 = 齐次 xy
    return torch.eye(3, 4)


def test_l0_interface_exists():
    assert hasattr(torch.ops.fusionrepo, "voxel_projection")


def test_l0_output_shape():
    coords = torch.randint(0, 100, (50, 3))
    vs = torch.tensor([0.1, 0.1, 0.2])
    pcr = torch.tensor([0.0, -40.0, -3.0, 70.4, 40.0, 1.0])
    args = (coords.to(DEVICE), vs.to(DEVICE), pcr.to(DEVICE), _proj_matrix().to(DEVICE))
    out = torch.ops.fusionrepo.voxel_projection(*args)
    assert out.shape == (50, 2)


def test_l0_invalid_dim():
    with pytest.raises(ValueError):
        torch.ops.fusionrepo.voxel_projection(torch.zeros(5, 2).to(DEVICE), torch.ones(3).to(DEVICE),
                                              torch.zeros(6).to(DEVICE), _proj_matrix().to(DEVICE))


def test_l1_center_formula():
    # 体素中心 = (coor + 0.5) * vs + range_min; 单位投影 → 像素 = (x, y)
    coords = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.float32)
    vs = torch.tensor([0.5, 1.0, 2.0])
    pcr = torch.tensor([10.0, -20.0, 4.0, 100.0, 20.0, 8.0])
    args = (coords.to(DEVICE), vs.to(DEVICE), pcr.to(DEVICE), _proj_matrix().to(DEVICE))
    out = torch.ops.fusionrepo.voxel_projection(*args)
    # 中心: (x, y, z) = (10.25, -19.5, 5); 透视除法 (w=z): 像素 = (x/z, y/z)
    assert torch.allclose(out.cpu()[0], torch.tensor([2.05, -3.9]))
    # 第二体素: (10.75, -17.5, 11) → (0.977, -1.591)
    assert torch.allclose(out.cpu()[1], torch.tensor([10.75 / 11.0, -17.5 / 11.0]))


def _points_to_image_ref(centers, proj):
    """与 points_to_image 同口径的参考 (直接矩阵投影, 深度>eps)。"""
    ones = torch.ones(centers.shape[0], 1, dtype=centers.dtype)
    homo = torch.cat([centers, ones], dim=1)          # (N, 4)
    uvw = homo @ proj.t()                              # (N, 3)
    eps = 1e-6
    depth = uvw[:, 2:3].clamp(min=eps)
    return uvw[:, :2] / depth


@pytest.mark.parametrize("num_voxels,seed", [(10, 0), (100, 1), (1000, 2)])
def test_l1_matches_projection(num_voxels, seed):
    g = torch.Generator().manual_seed(seed)
    coords = torch.randint(0, 200, (num_voxels, 3), generator=g).float()
    vs = torch.tensor([0.05, 0.05, 0.1])
    pcr = torch.tensor([0.0, -40.0, -3.0, 70.4, 40.0, 1.0])
    proj = torch.randn(3, 4, generator=g)
    proj[:, 3] = 0  # 保证深度主要来自前方
    proj[2, :3] = 0
    proj[2, 2] = 1.0
    centers = (coords + 0.5) * vs + pcr[:3]
    ref = _points_to_image_ref(centers, proj)
    out = torch.ops.fusionrepo.voxel_projection(coords.to(DEVICE), vs.to(DEVICE), pcr.to(DEVICE), proj.to(DEVICE))
    assert torch.allclose(out.cpu(), ref, atol=1e-4)


def test_l1_deterministic():
    coords = torch.randint(0, 100, (50, 3)).float()
    vs = torch.tensor([0.1, 0.1, 0.2])
    pcr = torch.tensor([0.0, -40.0, -3.0, 70.4, 40.0, 1.0])
    args = (coords.to(DEVICE), vs.to(DEVICE), pcr.to(DEVICE), _proj_matrix().to(DEVICE))
    a = torch.ops.fusionrepo.voxel_projection(*args)
    b = torch.ops.fusionrepo.voxel_projection(*args)
    assert torch.equal(a, b)
