# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
gather_roi_features

说明：
    该文件实现 “ROI内特征收集” 前置原语，
    供 operator_registry.py 中注册后的
    torch.ops.fusionrepo.gather_roi_features 调用。

功能：
    根据 ROI 范围，从 voxel / point 特征中收集每个 ROI 内部的局部特征。

    该算子只负责：
        1. 判断哪些点 / voxel 落在 ROI 内
        2. 将这些特征整理成固定张量
        3. 统计每个 ROI 的有效特征数量

    不负责：
        - mean / max / sum 聚合
        - 后续分类检测
        - 特征变换

输入：
    rois: (num_rois, 7)
        ROI 参数，格式为：
        (x, y, z, dx, dy, dz, yaw)

    voxel_coords: (num_voxels, 3)
        每个 voxel / point 的三维坐标
        格式为：
        (x, y, z)

    voxel_features: (num_voxels, num_channels)
        每个 voxel / point 对应的特征

输出：
    gathered_features: (num_rois, num_voxels, num_channels)
        每个 ROI 内收集到的特征，未命中的位置补 0

    valid_counts: (num_rois,)
        每个 ROI 内有效特征数量
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# 1) Python 实现（CompositeExplicitAutograd）
# ============================================================
def _gather_roi_features_impl(
    rois: torch.Tensor,
    voxel_coords: torch.Tensor,
    voxel_features: torch.Tensor
):
    """向量化实现（无 Python for 循环）。

    对输入做形状校验：非法输入给出清晰的 ValueError，而非晦涩的广播报错。
    """
    if rois.dim() != 2 or rois.shape[1] != 7:
        raise ValueError("rois must have shape (num_rois, 7)")

    if voxel_coords.dim() != 2 or voxel_coords.shape[1] != 3:
        raise ValueError("voxel_coords must have shape (num_voxels, 3)")

    if voxel_features.dim() != 2:
        raise ValueError("voxel_features must have shape (num_voxels, num_channels)")

    if voxel_coords.shape[0] != voxel_features.shape[0]:
        raise ValueError("voxel_coords and voxel_features must have the same first dim")

    device = voxel_features.device
    dtype = voxel_features.dtype

    rois = rois.to(device=device, dtype=dtype)
    voxel_coords = voxel_coords.to(device=device, dtype=dtype)

    num_rois = rois.shape[0]
    num_voxels = voxel_coords.shape[0]
    num_channels = voxel_features.shape[1]

    # 向量化: 所有 ROI 同时处理
    # 相对坐标广播: rel 形状 (num_rois, num_voxels, 3)
    rel = voxel_coords.unsqueeze(0) - rois[:, 0:3].unsqueeze(1)

    cos_yaw = torch.cos(rois[:, 6])  # (num_rois,)
    sin_yaw = torch.sin(rois[:, 6])
    zeros = torch.zeros_like(cos_yaw)
    ones = torch.ones_like(cos_yaw)
    # 旋转矩阵 (转置形式), 形状 (num_rois, 3, 3)
    rot_t = torch.stack([
        torch.stack([cos_yaw, -sin_yaw, zeros], dim=-1),
        torch.stack([sin_yaw, cos_yaw, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1),
    ], dim=-2)

    # 批量旋转: 局部坐标矩阵乘, 输出 (num_rois, num_voxels, 3)
    rel_local = torch.bmm(rel, rot_t)

    # 框内掩码, 形状 (num_rois, num_voxels)
    half_d = rois[:, 3:6] / 2.0
    mask = (
        (rel_local[..., 0].abs() <= half_d[:, 0].unsqueeze(1)) &
        (rel_local[..., 1].abs() <= half_d[:, 1].unsqueeze(1)) &
        (rel_local[..., 2].abs() <= half_d[:, 2].unsqueeze(1))
    )

    valid_counts = mask.sum(dim=1)  # (num_rois,)

    # 收集特征并置零盒外, 形状 (num_rois, num_voxels, num_channels)
    feat_expanded = voxel_features.unsqueeze(0).expand(num_rois, -1, -1)  # (num_rois, num_voxels, num_channels)
    mask_expanded = mask.unsqueeze(-1).expand_as(feat_expanded)
    gathered_features = feat_expanded * mask_expanded.to(dtype)

    return gathered_features, valid_counts


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("gather_roi_features", _gather_roi_features_impl)


# ============================================================
# 2) Meta 实现
# ============================================================
def _gather_roi_features_meta(
    rois: torch.Tensor,
    voxel_coords: torch.Tensor,
    voxel_features: torch.Tensor
):
    num_rois = rois.shape[0]
    num_voxels = voxel_coords.shape[0]
    num_channels = voxel_features.shape[1]

    gathered_features = torch.empty((num_rois, num_voxels, num_channels), device="meta", dtype=voxel_features.dtype)
    valid_counts = torch.empty((num_rois,), device="meta", dtype=torch.long)

    return gathered_features, valid_counts


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("gather_roi_features", _gather_roi_features_meta)


# ============================================================
# 3) Python 包装
# ============================================================
def gather_roi_features(
    rois: torch.Tensor,
    voxel_coords: torch.Tensor,
    voxel_features: torch.Tensor
):
    return torch.ops.fusionrepo.gather_roi_features(
        rois,
        voxel_coords,
        voxel_features
    )
