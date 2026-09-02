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
# compute_voxel_centroid 算子黑盒测试（自包含单文件）。
#
# golden 来源:
#   1) 独立 Python 参考实现（与算子定义逐点对应）
#   2) 解析构造用例
# 运行（NPU）:
#   cd /tmp && pytest <repo>/tests/compute_voxel_centroid/ -v

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "applications" / "autonomous_driving" / "common"))

import ops  # noqa: E402,F401

DEVICE = "npu" if torch.npu.is_available() else "cpu"


def test_l0_interface_exists():
    assert hasattr(torch.ops.fusionrepo, "compute_voxel_centroid")


def test_l0_output_shape():
    pts = torch.randn(5, 32, 3)
    npt = torch.randint(1, 33, (5,))
    c = torch.ops.fusionrepo.compute_voxel_centroid(pts.to(DEVICE), npt.to(DEVICE))
    assert c.shape == (5, 3)
    assert c.dtype == pts.dtype


def test_l1_single_point():
    # 单点体素: 质心 = 该点
    pts = torch.tensor([[[3.0, 4.0, 5.0], [0.0, 0.0, 0.0]]])
    npt = torch.tensor([1])
    c = torch.ops.fusionrepo.compute_voxel_centroid(pts.to(DEVICE), npt.to(DEVICE))
    assert torch.equal(c.cpu()[0], torch.tensor([3.0, 4.0, 5.0]))


def test_l1_two_points_mean():
    pts = torch.tensor([[[1.0, 1.0, 1.0], [3.0, 5.0, 7.0]]])
    npt = torch.tensor([2])
    c = torch.ops.fusionrepo.compute_voxel_centroid(pts.to(DEVICE), npt.to(DEVICE))
    assert torch.allclose(c.cpu()[0], torch.tensor([2.0, 3.0, 4.0]))


def test_l1_padded_zeros_semantics():
    # 语义约定 (与 mmcv 体素化输出配合): voxel_points 的 padding 位为 0,
    # 算子对全部 max_points 维求和后除以 num_points —— padding 为 0 时
    # 结果等于有效点的均值
    pts = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    npt = torch.tensor([2])
    c = torch.ops.fusionrepo.compute_voxel_centroid(pts.to(DEVICE), npt.to(DEVICE))
    assert torch.allclose(c.cpu()[0], torch.tensor([1.5, 0.0, 0.0]))


def test_l1_zero_count_no_nan():
    # num_points=0: clamp 到 1, 结果为全零和 / 1 (不产生 NaN)
    pts = torch.randn(2, 10, 3)
    npt = torch.tensor([0, 0])
    c = torch.ops.fusionrepo.compute_voxel_centroid(pts.to(DEVICE), npt.to(DEVICE))
    assert not torch.isnan(c).any().item()


@pytest.mark.parametrize("num_voxels,max_pts,seed", [(1, 16, 0), (8, 32, 1), (64, 5, 2)])
def test_l1_matches_reference(num_voxels, max_pts, seed):
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(num_voxels, max_pts, 3, generator=g)
    npt = torch.randint(1, max_pts + 1, (num_voxels,), generator=g)
    # padding 位清零, 模拟 mmcv 体素化输出
    for i in range(num_voxels):
        pts[i, npt[i]:] = 0.0
    ref = torch.stack([pts[i, :npt[i]].mean(dim=0) for i in range(num_voxels)])
    c = torch.ops.fusionrepo.compute_voxel_centroid(pts.to(DEVICE), npt.to(DEVICE))
    assert torch.allclose(c.cpu(), ref, atol=1e-5)


def test_l1_deterministic():
    pts = torch.randn(4, 20, 3)
    npt = torch.randint(1, 21, (4,))
    a = torch.ops.fusionrepo.compute_voxel_centroid(pts.to(DEVICE), npt.to(DEVICE))
    b = torch.ops.fusionrepo.compute_voxel_centroid(pts.to(DEVICE), npt.to(DEVICE))
    assert torch.equal(a, b)
