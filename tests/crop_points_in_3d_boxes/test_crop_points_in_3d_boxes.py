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
#
# crop_points_in_3d_boxes 算子黑盒测试（自包含单文件）。
#
# golden 来源:
#   1) 逐 ROI Python 循环参考实现（与轴对齐筛选定义逐点对应）
#   2) 解析构造: 手工放置盒内/盒外点, 精确核对特征与掩码
# 运行（NPU）:
#   cd /tmp && pytest <repo>/tests/crop_points_in_3d_boxes/ -v

import sys
from pathlib import Path

import math

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "applications" / "autonomous_driving" / "common"))

import ops  # noqa: E402,F401

DEVICE = "npu" if torch.npu.is_available() else "cpu"


def _reference(rois, points_xyz, points_feat):
    """逐 ROI Python 循环参考实现 (轴对齐筛选, 与算子语义一致)。"""
    batch_size, num_rois, _ = rois.shape
    num_points, num_channels = points_feat.shape
    feats = torch.zeros((batch_size, num_rois, num_points, num_channels), dtype=points_feat.dtype)
    mask = torch.zeros((batch_size, num_rois, num_points), dtype=torch.bool)
    for b in range(batch_size):
        for m in range(num_rois):
            cx, cy, cz = rois[b, m, 1:4].tolist()
            dx, dy, dz = rois[b, m, 4:7].tolist()
            inside = (
                (points_xyz[:, 0] >= cx - dx / 2) & (points_xyz[:, 0] <= cx + dx / 2) &
                (points_xyz[:, 1] >= cy - dy / 2) & (points_xyz[:, 1] <= cy + dy / 2) &
                (points_xyz[:, 2] >= cz - dz / 2) & (points_xyz[:, 2] <= cz + dz / 2)
            )
            mask[b, m] = inside
            feats[b, m] = points_feat * inside.unsqueeze(-1).to(points_feat.dtype)
    return feats, mask


def _rois(batch_size, num_rois, seed=0):
    g = torch.Generator().manual_seed(seed)
    rois = torch.zeros(batch_size, num_rois, 8)
    rois[..., 0] = 0                                  # batch idx 占位
    rois[..., 1] = torch.rand(batch_size, num_rois, generator=g) * 40
    rois[..., 2] = torch.rand(batch_size, num_rois, generator=g) * 20 - 10
    rois[..., 3] = 1.5
    rois[..., 4] = 1.6 + torch.rand(batch_size, num_rois, generator=g)
    rois[..., 5] = 3.9 + torch.rand(batch_size, num_rois, generator=g) * 2
    rois[..., 6] = 1.5
    rois[..., 7] = torch.rand(batch_size, num_rois, generator=g) * math.pi  # yaw 占位 (不参与筛选)
    return rois


# ------------------------------------------------------------
# L0 门槛
# ------------------------------------------------------------

def test_l0_interface_exists():
    assert hasattr(torch.ops.fusionrepo, "crop_points_in_3d_boxes")


def test_l0_output_shape_dtype():
    rois = _rois(2, 3)
    xyz = torch.randn(50, 3)
    feat = torch.randn(50, 4)
    gf, gm = torch.ops.fusionrepo.crop_points_in_3d_boxes(
        rois.to(DEVICE), xyz.to(DEVICE), feat.to(DEVICE))
    assert gf.shape == (2, 3, 50, 4)
    assert gm.shape == (2, 3, 50)
    assert gm.dtype == torch.bool
    assert gf.dtype == feat.dtype


@pytest.mark.parametrize("bad", ["rois_dim", "xyz_dim", "feat_dim", "n_mismatch"])
def test_l0_invalid_inputs(bad):
    rois = _rois(1, 2)
    xyz = torch.randn(10, 3)
    feat = torch.randn(10, 3)
    if bad == "rois_dim":
        args = (rois[0], xyz, feat)
    elif bad == "xyz_dim":
        args = (rois, xyz[0], feat)
    elif bad == "feat_dim":
        args = (rois, xyz, feat[0])
    else:
        args = (rois, xyz[:5], feat)
    with pytest.raises(ValueError):
        torch.ops.fusionrepo.crop_points_in_3d_boxes(
            *[a.to(DEVICE) for a in args])


# ------------------------------------------------------------
# L1 功能
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "batch_size,num_rois,num_points,seed",
    [(1, 1, 20, 0), (2, 3, 50, 1), (4, 8, 200, 2), (1, 60, 500, 3)]
)
def test_l1_matches_reference(batch_size, num_rois, num_points, seed):
    g = torch.Generator().manual_seed(seed)
    rois = _rois(batch_size, num_rois, seed)
    xyz = torch.rand(num_points, 3, generator=g) * torch.tensor([40.0, 20.0, 4.0]) - torch.tensor([0.0, 10.0, 2.0])
    feat = torch.randn(num_points, 4, generator=g)
    ref_f, ref_m = _reference(rois, xyz, feat)
    gf, gm = torch.ops.fusionrepo.crop_points_in_3d_boxes(
        rois.to(DEVICE), xyz.to(DEVICE), feat.to(DEVICE))
    assert torch.equal(gm.cpu(), ref_m)
    assert torch.equal(gf.cpu(), ref_f)


def test_l1_analytic_containment():
    # 手工构造: 盒 [2,0,1] 尺寸 [4,2,2] → x∈[0,4] y∈[-1,1] z∈[0,2]
    rois = torch.tensor([[[0.0, 2.0, 0.0, 1.0, 4.0, 2.0, 2.0, 0.7]]])
    xyz = torch.tensor([
        [2.0, 0.0, 1.0],     # 中心 → 内
        [0.0, -1.0, 0.0],    # 角落 → 内 (闭区间)
        [4.0, 1.0, 2.0],     # 另一角 → 内
        [4.5, 0.0, 1.0],     # x 超界 → 外
        [2.0, 1.5, 1.0],     # y 超界 → 外
        [2.0, 0.0, 2.5],     # z 超界 → 外
    ])
    feat = torch.arange(6, dtype=torch.float32).unsqueeze(1).expand(6, 3).contiguous()
    gf, gm = torch.ops.fusionrepo.crop_points_in_3d_boxes(
        rois.to(DEVICE), xyz.to(DEVICE), feat.to(DEVICE))
    assert gm.cpu()[0, 0].tolist() == [True, True, True, False, False, False]
    expect = feat * torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32).unsqueeze(1)
    assert torch.equal(gf.cpu()[0, 0], expect)


def test_l1_yaw_not_used():
    # 语义: 轴对齐筛选, yaw 不影响结果
    xyz = torch.tensor([[3.9, 0.0, 1.0]])
    feat = torch.tensor([[1.0]])
    r0 = torch.tensor([[[0.0, 2.0, 0.0, 1.0, 4.0, 2.0, 2.0, 0.0]]])
    r1 = r0.clone()
    r1[..., 7] = 1.8   # 改 yaw
    _, m0 = torch.ops.fusionrepo.crop_points_in_3d_boxes(r0.to(DEVICE), xyz.to(DEVICE), feat.to(DEVICE))
    _, m1 = torch.ops.fusionrepo.crop_points_in_3d_boxes(r1.to(DEVICE), xyz.to(DEVICE), feat.to(DEVICE))
    assert torch.equal(m0, m1)


def test_l1_empty_points():
    rois = _rois(2, 2)
    xyz = torch.zeros(0, 3)
    feat = torch.zeros(0, 5)
    gf, gm = torch.ops.fusionrepo.crop_points_in_3d_boxes(
        rois.to(DEVICE), xyz.to(DEVICE), feat.to(DEVICE))
    assert gf.shape == (2, 2, 0, 5)
    assert gm.shape == (2, 2, 0)


def test_l1_deterministic():
    rois = _rois(2, 4, seed=9)
    xyz = torch.randn(100, 3)
    feat = torch.randn(100, 3)
    a = torch.ops.fusionrepo.crop_points_in_3d_boxes(
        rois.to(DEVICE), xyz.to(DEVICE), feat.to(DEVICE))
    b = torch.ops.fusionrepo.crop_points_in_3d_boxes(
        rois.to(DEVICE), xyz.to(DEVICE), feat.to(DEVICE))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
