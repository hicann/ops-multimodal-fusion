# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
from torch.library import Library


def ball_query(
    xyz: torch.Tensor,
    center_xyz: torch.Tensor,
    max_radius: float,
    min_radius: float = 0.0,
    sample_num: int = 16
) -> torch.Tensor:
    """
    球查询（Ball Query）：为每个中心点收集半径内的点索引。

    功能：
        对每个中心点，在 xyz 中按原始顺序收集与中心点距离落在
        (min_radius, max_radius) 内的前 sample_num 个点的索引；
        不足 sample_num 个时以 -1 填充。
        与 mmcv CUDA ball_query 语义一致（不排序、按原始顺序取前 sample_num 个）。

    实现说明：
        全批量向量化——距离矩阵一次算出；「前 sample_num 个 True 的位置」用
        掩码前缀和（cumsum）+ searchsorted 求得，无逐中心 Python 循环、
        无 nonzero / 布尔索引（NPU 友好）。

    Args:
        xyz: (batch_size, num_points, 3)
            被查询的点云
        center_xyz: (batch_size, num_centers, 3)
            中心点（如 ROI 采样后的新质心）
        max_radius: float
            球半径（> 0）
        min_radius: float
            内半径（默认 0.0；距离须严格大于 min_radius）
        sample_num: int
            每个中心点最多收集的点数

    Returns:
        idx: (batch_size, num_centers, sample_num) int32
            点索引；无有效点的槽位为 -1
    """
    if xyz.dim() != 3 or xyz.shape[-1] != 3:
        raise ValueError("xyz must have shape (batch_size, num_points, 3)")
    if center_xyz.dim() != 3 or center_xyz.shape[-1] != 3:
        raise ValueError("center_xyz must have shape (batch_size, num_centers, 3)")
    if xyz.shape[0] != center_xyz.shape[0]:
        raise ValueError("xyz and center_xyz must have the same batch size")
    if sample_num <= 0:
        raise ValueError("sample_num must be positive")
    if max_radius <= 0:
        raise ValueError("max_radius must be positive")

    batch_size, num_points, _ = xyz.shape
    _, num_centers, _ = center_xyz.shape
    device = xyz.device

    # [1] 距离矩阵: (batch_size, num_centers, num_points)
    diff = center_xyz.unsqueeze(2) - xyz.unsqueeze(1)
    dist_sq = (diff * diff).sum(dim=-1)
    max_r_sq = max_radius * max_radius
    min_r_sq = min_radius * min_radius
    within = (dist_sq < max_r_sq) & (dist_sq > min_r_sq)

    # [2] 前 sample_num 个 True 的位置: 前缀和 + searchsorted
    cs = within.to(torch.int32).cumsum(dim=-1)               # (batch_size, num_centers, num_points) 非降
    targets = torch.arange(1, sample_num + 1, device=device,
                           dtype=torch.int32).view(1, 1, -1)
    total = cs[..., -1:]                                     # (batch_size, num_centers, 1)
    pos = torch.searchsorted(cs, targets.expand(batch_size, num_centers, sample_num))
    pos = pos.clamp(max=num_points - 1)
    valid = targets <= total
    idx = torch.where(valid, pos, torch.zeros_like(pos))
    idx = torch.where(valid, idx, torch.full_like(idx, -1)).to(torch.int32)

    return idx


# ============================================================
# 注册实现
# ============================================================
_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("ball_query", ball_query)


# ============================================================
# Meta 实现（等价 register_fake）
# ============================================================
def _ball_query_meta(xyz, center_xyz, max_radius, min_radius=0.0, sample_num=16):
    batch_size, num_centers, _ = center_xyz.shape
    return torch.empty((batch_size, num_centers, int(sample_num)), device="meta", dtype=torch.int32)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("ball_query", _ball_query_meta)
