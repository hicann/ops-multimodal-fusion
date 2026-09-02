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
# generate_roi_grid_points 算子黑盒测试（自包含单文件）。
#
# golden 来源:
#   1) 逐 ROI 参考实现（Python for 循环, 与定义逐点对应, CPU fp64 语义）
#   2) 解析性质: 未旋转 ROI 网格 = 中心 + 缩放线性网格; 旋转满足正交性;
#      网格点全部落在旋转包围盒内; 点间间距 = 尺寸/(grid_size-1)
# 运行（NPU）:
#   cd /tmp && pytest <repo>/tests/generate_roi_grid_points/ -v

import math
import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "applications" / "autonomous_driving" / "common"))

import ops  # noqa: E402,F401

DEVICE = "npu" if torch.npu.is_available() else "cpu"


def _reference_per_roi(rois, grid_size):
    """逐 ROI Python 循环参考实现 (fp32 与被测同口径, 逐点定义一致)。"""
    lin = torch.linspace(-0.5, 0.5, grid_size, dtype=rois.dtype)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    base = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    out = torch.zeros((rois.shape[0], base.shape[0], 3), dtype=rois.dtype)
    for i in range(rois.shape[0]):
        x, y, z, dx, dy, dz, yaw = rois[i].tolist()
        pts = base.clone()
        pts[:, 0] *= dx
        pts[:, 1] *= dy
        pts[:, 2] *= dz
        c, s = math.cos(yaw), math.sin(yaw)
        xr = pts[:, 0] * c - pts[:, 1] * s
        yr = pts[:, 0] * s + pts[:, 1] * c
        pts[:, 0] = xr + x
        pts[:, 1] = yr + y
        pts[:, 2] = pts[:, 2] + z
        out[i] = pts
    return out


def _rois(num_rois, seed=0):
    g = torch.Generator().manual_seed(seed)
    rois = torch.zeros(num_rois, 7)
    rois[:, 0] = torch.linspace(5, 60, num_rois) + torch.randn(num_rois, generator=g) * 0.1
    rois[:, 1] = torch.randn(num_rois, generator=g) * 4
    rois[:, 2] = 1.5 + torch.randn(num_rois, generator=g) * 0.3
    rois[:, 3] = 1.6 + torch.rand(num_rois, generator=g)
    rois[:, 4] = 3.9 + torch.rand(num_rois, generator=g) * 2
    rois[:, 5] = 1.5 + torch.rand(num_rois, generator=g) * 0.5
    rois[:, 6] = torch.randn(num_rois, generator=g) * 0.7
    return rois


# ------------------------------------------------------------
# L0 门槛
# ------------------------------------------------------------

def test_l0_interface_exists():
    assert hasattr(torch.ops.fusionrepo, "generate_roi_grid_points")


def test_l0_output_shape():
    rois = _rois(5)
    gp = torch.ops.fusionrepo.generate_roi_grid_points(rois.to(DEVICE), 6)
    assert gp.shape == (5, 216, 3)
    assert gp.device.type == DEVICE.split(":")[0]
    assert gp.dtype == rois.dtype


def test_l0_grid_size_one():
    rois = _rois(3)
    gp = torch.ops.fusionrepo.generate_roi_grid_points(rois.to(DEVICE), 1)
    # grid_size=1 → linspace(-0.5, 0.5, 1) = [-0.5] → 网格点位于 ROI 角落缩放处
    assert gp.shape == (3, 1, 3)


def test_l0_invalid_inputs():
    rois = _rois(2)
    with pytest.raises(ValueError):
        torch.ops.fusionrepo.generate_roi_grid_points(rois.to(DEVICE), 0)
    with pytest.raises(ValueError):
        torch.ops.fusionrepo.generate_roi_grid_points(torch.zeros(2, 3).to(DEVICE), 4)


# ------------------------------------------------------------
# L1 功能 (vs 逐 ROI 参考实现)
# ------------------------------------------------------------

@pytest.mark.parametrize("num_rois,gs,seed", [(1, 6, 0), (3, 4, 1), (16, 6, 2), (60, 6, 3), (7, 2, 4)])
def test_l1_matches_reference(num_rois, gs, seed):
    rois = _rois(num_rois, seed)
    ref = _reference_per_roi(rois, gs)
    out = torch.ops.fusionrepo.generate_roi_grid_points(rois.to(DEVICE), gs).cpu()
    diff = (out.double() - ref.double()).abs()
    assert diff.max().item() < 5e-6, f"max_diff={diff.max().item():.2e}"


def test_l1_axis_aligned_no_rotation():
    # yaw=0: 网格 = 中心 + [(-0.5..0.5)*dim] 线性网格
    rois = torch.tensor([[10.0, 20.0, 5.0, 4.0, 2.0, 2.0, 0.0]])
    gp = torch.ops.fusionrepo.generate_roi_grid_points(rois.to(DEVICE), 3).cpu()[0]
    xs = sorted(set(gp[:, 0].tolist()))
    ys = sorted(set(gp[:, 1].tolist()))
    zs = sorted(set(gp[:, 2].tolist()))
    assert len(xs) == 3 and len(ys) == 3 and len(zs) == 3
    assert xs == [8.0, 10.0, 12.0]      # 10 ± 4*0.5
    assert ys == [19.0, 20.0, 21.0]     # 20 ± 2*0.5
    assert zs == [4.0, 5.0, 6.0]        # 5 ± 2*0.5


def test_l1_rotation_preserves_extent():
    # 旋转后所有网格点仍在旋转包围盒内, 且中心距不超过半对角线
    rois = _rois(8, seed=5)
    gp = torch.ops.fusionrepo.generate_roi_grid_points(rois.to(DEVICE), 6).cpu()
    for i in range(8):
        cx, cy, _, dx, dy, _, _ = rois[i].tolist()
        rel = gp[i].double() - torch.tensor([cx, cy, 0.0]).double()
        r = rel[:, :2].norm(dim=-1).max().item()
        r_max = math.sqrt(dx * dx + dy * dy) / 2 + 1e-4
        assert r <= r_max, f"roi {i}: {r:.3f} > {r_max:.3f}"


def test_l1_spacing_uniform_unrotated():
    # 未旋转时相邻网格点间距 = dim / (grid_size - 1)
    rois = torch.tensor([[0.0, 0.0, 0.0, 6.0, 3.0, 1.0, 0.0]])
    gp = torch.ops.fusionrepo.generate_roi_grid_points(rois.to(DEVICE), 7).cpu()[0]
    xs = sorted(set(gp[:, 0].tolist()))
    assert abs((xs[1] - xs[0]) - 1.0) < 1e-6     # 6 / 6
    ys = sorted(set(gp[:, 1].tolist()))
    assert abs((ys[1] - ys[0]) - 0.5) < 1e-6     # 3 / 6


def test_l1_deterministic():
    rois = _rois(10, seed=6)
    b = rois.to(DEVICE)
    g1 = torch.ops.fusionrepo.generate_roi_grid_points(b, 6)
    g2 = torch.ops.fusionrepo.generate_roi_grid_points(b, 6)
    assert torch.equal(g1, g2)


def test_l1_translation_invariance():
    rois = _rois(6, seed=7)
    rois_t = rois.clone()
    rois_t[:, 0] += 100.0
    rois_t[:, 1] -= 30.0
    g1 = torch.ops.fusionrepo.generate_roi_grid_points(rois.to(DEVICE), 4).cpu()
    g2 = torch.ops.fusionrepo.generate_roi_grid_points(rois_t.to(DEVICE), 4).cpu()
    assert torch.allclose(g1[..., 0] - g1[:, :1, 0], g2[..., 0] - g2[:, :1, 0], atol=1e-3)
    assert torch.allclose(g1[..., 1] - g1[:, :1, 1], g2[..., 1] - g2[:, :1, 1], atol=1e-3)


def test_l1_grid_points_order():
    # 点序: meshgrid indexing='ij' 行主序展开 — x 最慢, z 最快
    rois = torch.tensor([[0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 0.0]])
    gp = torch.ops.fusionrepo.generate_roi_grid_points(rois.to(DEVICE), 2).cpu()[0]
    # 2x2x2, 坐标 ∈ {-1.5, +1.5}
    assert abs(gp[0, 0].item() - (-1.5)) < 1e-6
    assert abs(gp[1, 2].item() - 1.5) < 1e-6      # 第 2 点: z 最快
    assert abs(gp[2, 1].item() - 1.5) < 1e-6      # 第 3 点: y 变化
    assert abs(gp[4, 0].item() - 1.5) < 1e-6      # 第 5 点: x 变化
