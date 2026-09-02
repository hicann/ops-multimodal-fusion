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
# furthest_point_sample 算子黑盒测试（自包含单文件）。
#
# golden 来源:
#   1) CPU 同算法参考实现（随机数据无并列距离 → 索引逐位一致）
#   2) FPS 数学性质: 首点为 0 / 索引唯一 / 覆盖性（min-dist 单调不减语义）
# 运行（NPU）:
#   cd /tmp && pytest <repo>/tests/furthest_point_sample/ -v

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "applications" / "autonomous_driving" / "common"))

import ops  # noqa: E402,F401

DEVICE = "npu" if torch.npu.is_available() else "cpu"


def _reference(points_xyz, num_points):
    """CPU 同算法参考实现。"""
    batch_size, num_total, _ = points_xyz.shape
    xyz = points_xyz[:, :, :3]
    indices = torch.zeros((batch_size, num_points), dtype=torch.int32)
    min_dist = torch.full((batch_size, num_total), 1e10, dtype=xyz.dtype)
    farthest = xyz[:, 0:1, :]
    ar = torch.arange(batch_size)
    for i in range(1, num_points):
        dist = torch.sum((xyz - farthest) ** 2, dim=-1)
        min_dist = torch.minimum(min_dist, dist)
        idx = torch.argmax(min_dist, dim=-1)
        indices[:, i] = idx.to(torch.int32)
        farthest = xyz[ar, idx.long()].unsqueeze(1)
    return indices


def _points(batch_size, num_total, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(batch_size, num_total, 3, generator=g) * 10.0
    # 保证无并列距离（避免 argmax 平局歧义）
    return pts


# ------------------------------------------------------------
# L0 门槛
# ------------------------------------------------------------

def test_l0_interface_exists():
    assert hasattr(torch.ops.fusionrepo, "furthest_point_sample")


def test_l0_output_shape_dtype():
    pts = _points(2, 100)
    idx = torch.ops.fusionrepo.furthest_point_sample(pts.to(DEVICE), 16)
    assert idx.shape == (2, 16)
    assert idx.dtype == torch.int32


@pytest.mark.parametrize("bad", ["dim", "c3", "zero", "too_many"])
def test_l0_invalid_inputs(bad):
    if bad == "dim":
        args = (torch.randn(100, 3), 8)
    elif bad == "c3":
        args = (torch.randn(2, 100, 2), 8)
    elif bad == "zero":
        args = (torch.randn(2, 100, 3), 0)
    else:
        args = (torch.randn(2, 100, 3), 101)
    with pytest.raises(ValueError):
        if bad in ("zero", "too_many"):
            torch.ops.fusionrepo.furthest_point_sample(args[0].to(DEVICE), args[1])
        else:
            torch.ops.fusionrepo.furthest_point_sample(args[0].to(DEVICE), args[1])


# ------------------------------------------------------------
# L1 功能
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "batch_size,num_total,num_sampled,seed",
    [(1, 32, 8, 0), (2, 128, 16, 1), (4, 512, 32, 2), (1, 4096, 64, 3)]
)
def test_l1_matches_reference(batch_size, num_total, num_sampled, seed):
    pts = _points(batch_size, num_total, seed)
    ref = _reference(pts, num_sampled)
    out = torch.ops.fusionrepo.furthest_point_sample(pts.to(DEVICE), num_sampled).cpu()
    assert torch.equal(out, ref)


def test_l1_first_index_is_zero():
    pts = _points(3, 256, seed=4)
    idx = torch.ops.fusionrepo.furthest_point_sample(pts.to(DEVICE), 16).cpu()
    assert (idx[:, 0] == 0).all()


def test_l1_indices_unique():
    # 无重复点 → FPS 索引唯一
    pts = _points(2, 512, seed=5)
    idx = torch.ops.fusionrepo.furthest_point_sample(pts.to(DEVICE), 128).cpu()
    for b in range(2):
        assert len(set(idx[b].tolist())) == 128


def test_l1_num_points_one():
    pts = _points(2, 64, seed=6)
    idx = torch.ops.fusionrepo.furthest_point_sample(pts.to(DEVICE), 1).cpu()
    assert idx.tolist() == [[0], [0]]


def test_l1_uses_only_xyz():
    # C > 3 时额外列不参与距离计算: 追加噪声列结果不变
    pts3 = _points(2, 200, seed=7)
    g = torch.Generator().manual_seed(99)
    extra = torch.randn(2, 200, 4, generator=g)
    pts7 = torch.cat([pts3, extra], dim=-1)
    i3 = torch.ops.fusionrepo.furthest_point_sample(pts3.to(DEVICE), 32).cpu()
    i7 = torch.ops.fusionrepo.furthest_point_sample(pts7.to(DEVICE), 32).cpu()
    assert torch.equal(i3, i7)


def test_l1_deterministic():
    pts = _points(2, 256, seed=8)
    a = torch.ops.fusionrepo.furthest_point_sample(pts.to(DEVICE), 32)
    b = torch.ops.fusionrepo.furthest_point_sample(pts.to(DEVICE), 32)
    assert torch.equal(a, b)
