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


def furthest_point_sample(
    points_xyz: torch.Tensor,
    num_points: int
) -> torch.Tensor:
    """
    最远点采样（Farthest Point Sampling, FPS）。

    功能：
        从每个 batch 的 num_total 个点中迭代采样 num_points 个点：
        从第 0 个点出发，每步选取「到已选点集最小距离」最大的点，
        与 CUDA 实现（mmcv furthest_point_sample）语义一致。

    实现说明：
        迭代 num_points-1 步，每步以一次广播距离计算 +
        一次 min 归约 + 一次 argmax 完成（批量向量化，无逐点循环
        内的 CPU 同步）。距离仅使用前 3 列 xyz，与 CUDA 行为一致。

    Args:
        points_xyz: (batch_size, num_total, C)
            点云，C >= 3（仅使用前 3 列 xyz 参与距离计算）
        num_points: int
            采样点数（1 <= num_points <= num_total）

    Returns:
        indices: (batch_size, num_points) int32
            采样点的索引；indices[:, 0] 恒为 0

    已知限制：
        存在并列最大距离时，选取结果取决于 argmax 实现，
        可能与 CUDA 内核在并列情形下的具体索引不同（采样语义等价）。
    """
    if points_xyz.dim() != 3 or points_xyz.shape[-1] < 3:
        raise ValueError("points_xyz must have shape (batch_size, num_total, C) with C >= 3")
    num_total = points_xyz.shape[1]
    if not (1 <= num_points <= num_total):
        raise ValueError(f"num_points must be in [1, {num_total}], got {num_points}")

    device = points_xyz.device
    xyz = points_xyz[:, :, :3].contiguous()

    indices = torch.zeros((points_xyz.shape[0], num_points),
                          dtype=torch.int32, device=device)
    min_dist = torch.full((xyz.shape[0], num_total), 1e10,
                          device=device, dtype=xyz.dtype)

    farthest = xyz[:, 0:1, :]                                # (batch_size, 1, 3)
    batch_ar = torch.arange(xyz.shape[0], device=device)
    for i in range(1, num_points):
        dist = torch.sum((xyz - farthest) ** 2, dim=-1)      # (batch_size, num_total)
        min_dist = torch.minimum(min_dist, dist)
        idx = torch.argmax(min_dist, dim=-1)                 # (batch_size,)
        indices[:, i] = idx.to(torch.int32)
        farthest = xyz[batch_ar, idx.long()].unsqueeze(1)    # (batch_size, 1, 3)

    return indices


# ============================================================
# 注册实现
# ============================================================
_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("furthest_point_sample", furthest_point_sample)


# ============================================================
# Meta 实现（等价 register_fake）
# ============================================================
def _furthest_point_sample_meta(points_xyz, num_points):
    return torch.empty((points_xyz.shape[0], int(num_points)),
                       device="meta", dtype=torch.int32)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("furthest_point_sample", _furthest_point_sample_meta)
