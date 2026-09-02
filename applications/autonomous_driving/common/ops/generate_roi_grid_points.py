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


def generate_roi_grid_points(
    rois: torch.Tensor,
    grid_size: int
) -> torch.Tensor:
    """
    为每个旋转 3D ROI 生成其包围盒内的规则网格采样点。

    功能：
        在 ROI 局部坐标系的 [-0.5, 0.5]^3 内生成 grid_size^3 个均匀网格点，
        按 ROI 尺寸缩放、绕 z 轴旋转 yaw 后平移到 ROI 中心，输出全局坐标。
        供 ROI 网格池化（grid point pooling）与位置编码使用。

    实现说明：
        全批量向量化——基础网格只生成一次，缩放/旋转/平移以广播完成，
        无逐 ROI Python 循环（旋转矩阵含两列零元素，直接使用 2D 旋转
        标量公式 x' = x·cos − y·sin, y' = x·sin + y·cos）。

    Args:
        rois: (num_rois, 7)
            ROI 参数 [x, y, z, dx, dy, dz, yaw]
        grid_size: int
            每维网格点数（>0），总点数 num_grid = grid_size ** 3

    Returns:
        grid_points: (num_rois, num_grid, 3)
            每个 ROI 的网格点全局坐标 [x, y, z]
    """
    if rois.dim() != 2 or rois.shape[1] != 7:
        raise ValueError("rois must have shape (num_rois, 7)")

    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    device = rois.device
    dtype = rois.dtype

    lin = torch.linspace(-0.5, 0.5, grid_size, device=device, dtype=dtype)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    base_grid = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)  # (num_grid, 3)

    # 向量化: 缩放 + 2D 旋转 + 平移一次性完成
    centers = rois[:, 0:3]          # (num_rois, 3)
    dims = rois[:, 3:6]             # (num_rois, 3)
    yaws = rois[:, 6]               # (num_rois,)

    scaled = base_grid.unsqueeze(0) * dims.unsqueeze(1)      # (num_rois, num_grid, 3)
    cos_yaw = torch.cos(yaws).view(-1, 1)                    # (num_rois, 1)
    sin_yaw = torch.sin(yaws).view(-1, 1)                    # (num_rois, 1)
    x = scaled[..., 0] * cos_yaw - scaled[..., 1] * sin_yaw  # (num_rois, num_grid)
    y = scaled[..., 0] * sin_yaw + scaled[..., 1] * cos_yaw  # (num_rois, num_grid)
    z = scaled[..., 2]                                       # (num_rois, num_grid)
    grid_points = torch.stack([x, y, z], dim=-1)             # (num_rois, num_grid, 3)
    grid_points = grid_points + centers.unsqueeze(1)         # (num_rois, num_grid, 3)

    return grid_points


# ============================================================
# 注册实现
# ============================================================
_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("generate_roi_grid_points", generate_roi_grid_points)


# ============================================================
# Meta 实现（等价 register_fake）
# ============================================================
def _generate_roi_grid_points_meta(rois, grid_size):
    num_rois = rois.shape[0]
    num_grid = int(grid_size) ** 3
    return torch.empty((num_rois, num_grid, 3), device="meta", dtype=rois.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("generate_roi_grid_points", _generate_roi_grid_points_meta)
