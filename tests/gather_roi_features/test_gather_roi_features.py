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
# gather_roi_features 算子黑盒测试（自包含单文件）。
#
# golden 来源:
#   1) 独立 Python 参考实现（与算子定义逐点对应）
#   2) 解析构造用例
# 运行（NPU）:
#   cd /tmp && pytest <repo>/tests/gather_roi_features/ -v

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "applications" / "autonomous_driving" / "common"))

import ops  # noqa: E402,F401

DEVICE = "npu" if torch.npu.is_available() else "cpu"


def _reference(rois, coords, feats):
    """逐 ROI Python 循环参考实现。"""
    num_rois = rois.shape[0]
    num_voxels = coords.shape[0]
    num_channels = feats.shape[1]
    gathered = torch.zeros(num_rois, num_voxels, num_channels, dtype=feats.dtype)
    counts = torch.zeros(num_rois, dtype=torch.long)
    for r in range(num_rois):
        cx, cy, cz, dx, dy, dz, yaw = rois[r].tolist()
        c, s = torch.cos(torch.tensor(yaw)).item(), torch.sin(torch.tensor(yaw)).item()
        for v in range(num_voxels):
            rx = coords[v, 0].item() - cx
            ry = coords[v, 1].item() - cy
            rz = coords[v, 2].item() - cz
            lx = rx * c + ry * s
            ly = -rx * s + ry * c
            if abs(lx) <= dx / 2 and abs(ly) <= dy / 2 and abs(rz) <= dz / 2:
                gathered[r, v] = feats[v]
                counts[r] += 1
    return gathered, counts


def _rois(num, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = torch.zeros(num, 7)
    r[:, 0] = torch.linspace(5, 60, num) + torch.randn(num, generator=g) * 0.1
    r[:, 1] = torch.randn(num, generator=g) * 4
    r[:, 2] = 1.5
    r[:, 3] = 1.6 + torch.rand(num, generator=g)
    r[:, 4] = 3.9 + torch.rand(num, generator=g) * 2
    r[:, 5] = 1.5
    r[:, 6] = torch.randn(num, generator=g) * 0.7
    return r


def _voxels(num, seed=1):
    g = torch.Generator().manual_seed(seed)
    v = torch.rand(num, 3, generator=g) * torch.tensor([70.0, 40.0, 4.0]) - torch.tensor([0.0, 20.0, 2.0])
    return v


def test_l0_interface_exists():
    assert hasattr(torch.ops.fusionrepo, "gather_roi_features")


def test_l0_output_shape():
    rois = _rois(3)
    coords = _voxels(100)
    feats = torch.randn(100, 8)
    g, c = torch.ops.fusionrepo.gather_roi_features(rois.to(DEVICE), coords.to(DEVICE), feats.to(DEVICE))
    assert g.shape == (3, 100, 8) and c.shape == (3,)


def test_l0_invalid_dim():
    bad_rois = torch.zeros(2, 3).to(DEVICE)
    with pytest.raises(Exception):
        torch.ops.fusionrepo.gather_roi_features(bad_rois, _voxels(10).to(DEVICE),
                                                 torch.randn(10, 4).to(DEVICE))


@pytest.mark.parametrize("num_rois,num_voxels,seed", [(1, 50, 0), (4, 200, 1), (16, 400, 2)])
def test_l1_matches_reference(num_rois, num_voxels, seed):
    rois = _rois(num_rois, seed)
    coords = _voxels(num_voxels, seed + 10)
    feats = torch.randn(num_voxels, 8, generator=torch.Generator().manual_seed(seed))
    rg, rc = _reference(rois, coords, feats)
    g, c = torch.ops.fusionrepo.gather_roi_features(rois.to(DEVICE), coords.to(DEVICE), feats.to(DEVICE))
    assert torch.equal(c.cpu(), rc)
    assert torch.equal(g.cpu(), rg)


def test_l1_all_outside():
    rois = torch.tensor([[10.0, 10.0, 1.0, 2.0, 2.0, 2.0, 0.0]])
    coords = torch.tensor([[100.0, 100.0, 100.0]])
    feats = torch.randn(1, 4)
    g, c = torch.ops.fusionrepo.gather_roi_features(rois.to(DEVICE), coords.to(DEVICE), feats.to(DEVICE))
    assert c.item() == 0 and (g == 0).all().item()


def test_l1_all_inside():
    rois = torch.tensor([[10.0, 10.0, 1.0, 4.0, 4.0, 4.0, 0.0]])
    coords = torch.tensor([[10.0, 10.0, 1.0]])
    feats = torch.randn(1, 4)
    g, c = torch.ops.fusionrepo.gather_roi_features(rois.to(DEVICE), coords.to(DEVICE), feats.to(DEVICE))
    assert c.item() == 1 and torch.equal(g[0, 0].cpu(), feats[0])


def test_l1_rotated_roi():
    # 45 度旋转 ROI: 局部坐标判定
    rois = torch.tensor([[10.0, 10.0, 1.0, 5.0, 1.0, 2.0, 0.7853981]])
    coords = torch.tensor([[12.0, 12.0, 1.0], [11.0, 9.0, 1.0]])
    feats = torch.randn(2, 4)
    g, c = torch.ops.fusionrepo.gather_roi_features(rois.to(DEVICE), coords.to(DEVICE), feats.to(DEVICE))
    # 旋转 ROI 的局部坐标判定由参考实现逐点定义, 此处仅对拍
    ref_g, ref_c = _reference(rois, coords, feats)
    assert torch.equal(c.cpu(), ref_c) and torch.equal(g.cpu(), ref_g)


def test_l1_deterministic():
    rois = _rois(8, seed=5)
    coords = _voxels(200, seed=6)
    feats = torch.randn(200, 8)
    a = torch.ops.fusionrepo.gather_roi_features(rois.to(DEVICE), coords.to(DEVICE), feats.to(DEVICE))
    b = torch.ops.fusionrepo.gather_roi_features(rois.to(DEVICE), coords.to(DEVICE), feats.to(DEVICE))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
