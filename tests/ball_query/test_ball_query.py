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
# ball_query 算子黑盒测试（自包含单文件）。
#
# golden 来源:
#   1) 逐中心 Python 循环参考实现（与 mmcv CUDA 语义一致: 按原始顺序
#      取前 sample_num 个半径内点, 不排序, 不足补 -1）
#   2) 解析构造: 手工布置球内/球外点
# 运行（NPU）:
#   cd /tmp && pytest <repo>/tests/ball_query/ -v

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "applications" / "autonomous_driving" / "common"))

import ops  # noqa: E402,F401

DEVICE = "npu" if torch.npu.is_available() else "cpu"


def _collect_valid(xyz_row, center, max_r2, min_r2, sample_num):
    """按原始顺序收集前 sample_num 个球内点索引。"""
    picked = []
    for n in range(xyz_row.shape[0]):
        if len(picked) >= sample_num:
            break
        d2 = ((center - xyz_row[n]) ** 2).sum().item()
        if min_r2 < d2 < max_r2:
            picked.append(n)
    return picked


def _reference(xyz, center_xyz, max_radius, min_radius, sample_num):
    """逐中心循环参考 (与 CUDA 语义一致)。"""
    batch_size, num_points, _ = xyz.shape
    _, num_centers, _ = center_xyz.shape
    out = torch.full((batch_size, num_centers, sample_num), -1, dtype=torch.int32)
    max_r2, min_r2 = max_radius ** 2, min_radius ** 2
    for b in range(batch_size):
        for m in range(num_centers):
            picked = _collect_valid(xyz[b], center_xyz[b, m], max_r2, min_r2, sample_num)
            for k, n in enumerate(picked):
                out[b, m, k] = n
    return out


def _points(batch_size, num_points, num_centers, seed=0):
    g = torch.Generator().manual_seed(seed)
    xyz = torch.randn(batch_size, num_points, 3, generator=g) * 5.0
    center = torch.randn(batch_size, num_centers, 3, generator=g) * 2.0
    return xyz, center


# ------------------------------------------------------------
# L0 门槛
# ------------------------------------------------------------

def test_l0_interface_exists():
    assert hasattr(torch.ops.fusionrepo, "ball_query")


def test_l0_output_shape_dtype():
    xyz, center = _points(2, 100, 10)
    idx = torch.ops.fusionrepo.ball_query(
        xyz.to(DEVICE), center.to(DEVICE), 2.0, 0.0, 16)
    assert idx.shape == (2, 10, 16)
    assert idx.dtype == torch.int32


@pytest.mark.parametrize("bad", ["xyz_dim", "batch_mismatch", "k0", "r0"])
def test_l0_invalid_inputs(bad):
    xyz, center = _points(2, 50, 5)
    if bad == "xyz_dim":
        with pytest.raises(ValueError):
            torch.ops.fusionrepo.ball_query(xyz[0].to(DEVICE), center.to(DEVICE), 2.0)
        return
    if bad == "batch_mismatch":
        with pytest.raises(ValueError):
            torch.ops.fusionrepo.ball_query(
                xyz.to(DEVICE), center[:1].to(DEVICE), 2.0)
        return
    if bad == "k0":
        with pytest.raises(ValueError):
            torch.ops.fusionrepo.ball_query(
                xyz.to(DEVICE), center.to(DEVICE), 2.0, 0.0, 0)
        return
    with pytest.raises(ValueError):
        torch.ops.fusionrepo.ball_query(
            xyz.to(DEVICE), center.to(DEVICE), 0.0)


# ------------------------------------------------------------
# L1 功能
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "batch_size,num_points,num_centers,sample_num,seed",
    [(1, 50, 5, 8, 0), (2, 200, 16, 16, 1), (4, 512, 32, 32, 2)]
)
def test_l1_matches_reference(batch_size, num_points, num_centers, sample_num, seed):
    xyz, center = _points(batch_size, num_points, num_centers, seed)
    ref = _reference(xyz, center, 2.0, 0.0, sample_num)
    out = torch.ops.fusionrepo.ball_query(
        xyz.to(DEVICE), center.to(DEVICE), 2.0, 0.0, sample_num).cpu()
    assert torch.equal(out, ref)


def test_l1_analytic_first_k_in_order():
    # 点 0..4 距中心 1,2,3,4,5；半径 4.5 → 前 3 个有效为 0,1,2
    # 注: 与 mmcv CUDA 一致, 距离须严格大于 min_radius (=0), 距离 0 不计入
    xyz = torch.tensor([[[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0], [5.0, 0, 0]]])
    center = torch.tensor([[[0.0, 0, 0]]])
    idx = torch.ops.fusionrepo.ball_query(
        xyz.to(DEVICE), center.to(DEVICE), 4.5, 0.0, 3).cpu()
    assert idx[0, 0].tolist() == [0, 1, 2]


def test_l1_fill_minus_one_when_insufficient():
    xyz = torch.tensor([[[1.0, 0, 0], [2.0, 0, 0], [50.0, 0, 0]]])
    center = torch.tensor([[[0.0, 0, 0]]])
    idx = torch.ops.fusionrepo.ball_query(
        xyz.to(DEVICE), center.to(DEVICE), 5.0, 0.0, 5).cpu()
    assert idx[0, 0].tolist() == [0, 1, -1, -1, -1]


def test_l1_min_radius():
    # 半径 (1, 5): 距离 1 的点被排除（严格大于内半径）
    xyz = torch.tensor([[[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]]])
    center = torch.tensor([[[0.0, 0, 0]]])
    idx = torch.ops.fusionrepo.ball_query(
        xyz.to(DEVICE), center.to(DEVICE), 5.0, 1.0, 5).cpu()
    assert idx[0, 0].tolist() == [1, 2, -1, -1, -1]


def test_l1_empty_ball_all_minus_one():
    xyz = torch.tensor([[[100.0, 0, 0]]])
    center = torch.tensor([[[0.0, 0, 0]]])
    idx = torch.ops.fusionrepo.ball_query(
        xyz.to(DEVICE), center.to(DEVICE), 2.0, 0.0, 4).cpu()
    assert idx[0, 0].tolist() == [-1, -1, -1, -1]


def test_l1_deterministic():
    xyz, center = _points(2, 300, 20, seed=8)
    a = torch.ops.fusionrepo.ball_query(xyz.to(DEVICE), center.to(DEVICE), 2.0, 0.0, 16)
    b = torch.ops.fusionrepo.ball_query(xyz.to(DEVICE), center.to(DEVICE), 2.0, 0.0, 16)
    assert torch.equal(a, b)
