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


def crop_points_in_3d_boxes(
    rois: torch.Tensor,
    points_xyz: torch.Tensor,
    points_feat: torch.Tensor
):
    """
    判断点是否落在 3D ROI 内, 并按 ROI 对点特征分组。

    功能：
        对每个 ROI 建立轴对齐包围盒（中心 ± 尺寸/2），统计哪些点落在盒内，
        输出按 ROI 分组的特征张量（盒外置 0）与有效掩码。

        当前实现与原 MPCF 代码语义保持一致：只按轴对齐包围盒范围筛选，
        不使用 yaw 旋转（rois 第 8 列 yaw 仅作为格式占位）。

    实现说明：
        全批量广播——(batch_size, num_rois, 3) 包围盒边界与 (num_points, 3) 点坐标一次广播比较，
        无逐 ROI Python 循环。

    Args:
        rois: (batch_size, num_rois, 8)
            [batch_idx, x, y, z, dx, dy, dz, yaw]
        points_xyz: (num_points, 3)
            点坐标 [x, y, z]
        points_feat: (num_points, num_channels)
            点特征

    Returns:
        grouped_features: (batch_size, num_rois, num_points, num_channels)
            分组特征；点不在 ROI 内的位置为 0
        grouped_mask: (batch_size, num_rois, num_points)
            点是否落在对应 ROI 内
    """
    if rois.dim() != 3 or rois.size(-1) != 8:
        raise ValueError("rois must have shape (batch_size, num_rois, 8)")

    if points_xyz.dim() != 2 or points_xyz.size(-1) != 3:
        raise ValueError("points_xyz must have shape (num_points, 3)")

    if points_feat.dim() != 2:
        raise ValueError("points_feat must have shape (num_points, num_channels)")

    if points_xyz.size(0) != points_feat.size(0):
        raise ValueError("points_xyz and points_feat must have the same number of points")

    batch_size, num_rois, _ = rois.shape
    num_points, num_channels = points_feat.shape
    dtype = points_feat.dtype

    # [1] 包围盒边界: (batch_size, num_rois, 3)
    center = rois[..., 1:4]
    half = rois[..., 4:7] / 2
    bmin = center - half
    bmax = center + half

    # [2] 广播比较: (batch_size, num_rois, 1) vs (num_points,) → (batch_size, num_rois, num_points)
    px = points_xyz[:, 0]
    py = points_xyz[:, 1]
    pz = points_xyz[:, 2]
    inside = (
        (px.view(1, 1, num_points) >= bmin[..., 0:1]) & (px.view(1, 1, num_points) <= bmax[..., 0:1]) &
        (py.view(1, 1, num_points) >= bmin[..., 1:2]) & (py.view(1, 1, num_points) <= bmax[..., 1:2]) &
        (pz.view(1, 1, num_points) >= bmin[..., 2:3]) & (pz.view(1, 1, num_points) <= bmax[..., 2:3])
    )                                                        # (batch_size, num_rois, num_points) bool

    # [3] 分组特征: 盒内置原特征, 盒外置 0
    grouped_features = points_feat.unsqueeze(0).unsqueeze(0) \
        * inside.unsqueeze(-1).to(dtype)                     # (batch_size, num_rois, num_points, num_channels)
    grouped_mask = inside

    return grouped_features, grouped_mask


# ============================================================
# 注册实现
# ============================================================
_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("crop_points_in_3d_boxes", crop_points_in_3d_boxes)


# ============================================================
# Meta 实现（等价 register_fake）
# ============================================================
def _crop_points_in_3d_boxes_meta(rois, points_xyz, points_feat):
    if rois.dim() != 3 or rois.size(-1) != 8:
        raise ValueError("rois must have shape (batch_size, num_rois, 8)")
    if points_xyz.dim() != 2 or points_xyz.size(-1) != 3:
        raise ValueError("points_xyz must have shape (num_points, 3)")
    if points_feat.dim() != 2:
        raise ValueError("points_feat must have shape (num_points, num_channels)")
    batch_size, num_rois, _ = rois.shape
    num_points, num_channels = points_feat.shape
    return (torch.empty((batch_size, num_rois, num_points, num_channels), device="meta", dtype=points_feat.dtype),
            torch.empty((batch_size, num_rois, num_points), device="meta", dtype=torch.bool))


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("crop_points_in_3d_boxes", _crop_points_in_3d_boxes_meta)
