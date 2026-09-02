# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
feature_sampling_by_grid

说明：
    统一实现跨模型的特征采样算子。

注册算子：
    torch.ops.fusionrepo.feature_sampling_by_grid(feature_map, grid)

功能：
    根据连续采样坐标 grid，从特征图 feature_map 中进行双线性插值采样，
    返回对应位置的特征表示。

等价替代：
    avod::bev_crop_resize_rpn / det
    avod::image_crop_resize_rpn / det
    il_fusion::image_point_sampling
    li_fusion::image_sampler
    cvf::camera_to_bev_feature_projection

输入：
    feature_map : (batch_size , num_channels , H , W)

    grid :
        (batch_size , num_points , 2)   或
        (batch_size , num_points , 1 , 2)

        坐标范围 [-1 , 1]

输出：
    sampled_features :
        (batch_size , num_channels , num_points)
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.library import Library


# ============================================================
# 2) Implementation
# ============================================================

def _feature_sampling_by_grid_impl(
        feature_map: torch.Tensor,
        grid: torch.Tensor):

    if feature_map.dim() != 4:
        raise ValueError("feature_map must be (batch_size,num_channels,H,W)")

    if grid.dim() == 3:
        grid = grid.unsqueeze(2)   # (batch_size,num_points,1,2)

    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        align_corners=True
    )  # (batch_size,num_channels,num_points,1)

    sampled = sampled.squeeze(-1)  # (batch_size,num_channels,num_points)

    return sampled.contiguous()


_lib_impl = Library(
    "fusionrepo",
    "IMPL",
    "CompositeExplicitAutograd"
)

_lib_impl.impl(
    "feature_sampling_by_grid",
    _feature_sampling_by_grid_impl
)


# ============================================================
# 3) Meta 实现
# ============================================================

def _feature_sampling_by_grid_meta(
        feature_map: torch.Tensor,
        grid: torch.Tensor):

    batch_size, num_channels, _, _ = feature_map.shape

    if grid.dim() == 3:
        num_points = grid.shape[1]
    else:
        num_points = grid.shape[1]

    return torch.empty(
        (batch_size, num_channels, num_points),
        device="meta",
        dtype=feature_map.dtype
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")

_lib_meta.impl(
    "feature_sampling_by_grid",
    _feature_sampling_by_grid_meta
)


# ============================================================
# 4) Python wrapper
# ============================================================

def feature_sampling_by_grid(
        feature_map: torch.Tensor,
        grid: torch.Tensor):

    return torch.ops.fusionrepo.feature_sampling_by_grid(
        feature_map,
        grid
    )
