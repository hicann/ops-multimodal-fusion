# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple

import torch
from torch.library import Library


# 采样网格划分数: 每框 4 x 5 = 20 个确定性分层采样点
_GRID_U = 4
_GRID_V = 5


class BoxParams(NamedTuple):
    """BEV 旋转框参数组（中心 / 尺寸 / 朝向及其三角函数）。"""
    cx: torch.Tensor
    cy: torch.Tensor
    w: torch.Tensor
    h: torch.Tensor
    angle: torch.Tensor
    cos_a: torch.Tensor
    sin_a: torch.Tensor


def _box_bev_params(boxes_bev):
    """从 (N, 7) 旋转框提取 BEV 参数组。"""
    return BoxParams(
        cx=boxes_bev[:, 0],
        cy=boxes_bev[:, 1],
        w=boxes_bev[:, 3],
        h=boxes_bev[:, 4],
        angle=boxes_bev[:, 6],
        cos_a=torch.cos(boxes_bev[:, 6]),
        sin_a=torch.sin(boxes_bev[:, 6]),
    )


def _sample_points(boxes_bev, params):
    """每框 4x5=20 个确定性分层网格采样点, 旋转到全局坐标系。

    返回 (num_boxes, 20) 的全局坐标 gx/gy。2D 面积采样（非共线），
    同一输入多次执行逐位一致，估计误差随采样数收敛。
    """
    gu = (torch.arange(_GRID_U, device=boxes_bev.device, dtype=boxes_bev.dtype) + 0.5) / _GRID_U
    gv = (torch.arange(_GRID_V, device=boxes_bev.device, dtype=boxes_bev.dtype) + 0.5) / _GRID_V
    su, sv = torch.meshgrid(gu, gv, indexing="ij")
    su = su.reshape(-1)                                        # (20,)
    sv = sv.reshape(-1)                                        # (20,)
    local_x = (su - 0.5) * params.w.unsqueeze(-1)              # (num_boxes, 20)
    local_y = (sv - 0.5) * params.h.unsqueeze(-1)              # (num_boxes, 20)
    gx = (local_x * params.cos_a.unsqueeze(-1) - local_y * params.sin_a.unsqueeze(-1)
          + params.cx.unsqueeze(-1))
    gy = (local_x * params.sin_a.unsqueeze(-1) + local_y * params.cos_a.unsqueeze(-1)
          + params.cy.unsqueeze(-1))
    return gx, gy


def _exact_areas(params):
    """各框 BEV 精确面积（角点鞋带公式）。"""
    hw = params.w / 2
    hh = params.h / 2
    corners = torch.stack([
        torch.stack([params.cx - hw * params.cos_a + hh * params.sin_a,
                     params.cy - hw * params.sin_a - hh * params.cos_a], dim=-1),
        torch.stack([params.cx + hw * params.cos_a + hh * params.sin_a,
                     params.cy + hw * params.sin_a - hh * params.cos_a], dim=-1),
        torch.stack([params.cx + hw * params.cos_a - hh * params.sin_a,
                     params.cy + hw * params.sin_a + hh * params.cos_a], dim=-1),
        torch.stack([params.cx - hw * params.cos_a - hh * params.sin_a,
                     params.cy - hw * params.sin_a + hh * params.cos_a], dim=-1),
    ], dim=1)                                                  # (num_boxes, 4, 2)
    return 0.5 * (
        (corners[..., 0] * torch.roll(corners[..., 1], -1, dims=-1)
         - torch.roll(corners[..., 0], -1, dims=-1) * corners[..., 1]).sum(dim=-1)
    ).abs()                                                    # (num_boxes,)


def _rotated_bev_iou_matrix(boxes_sorted):
    """蒙特卡洛（确定性面积采样）估计的旋转 BEV IoU 矩阵。

    inter(i, j) ≈ area_j × P(框 j 的采样点落入框 i)；
    union ≈ area_i + area_j − inter。
    """
    num_boxes = boxes_sorted.shape[0]
    params = _box_bev_params(boxes_sorted)
    gx, gy = _sample_points(boxes_sorted, params)

    # 框 j 的采样点落入框 i 的判定（点旋转到框 i 的局部坐标）:
    # gx/gy 的 0 轴是采样点所属框 (j), 中心/边界/旋转的 0 轴是被测框 (i)
    dx = gx.unsqueeze(0) - params.cx.view(num_boxes, 1, 1)     # [i, j, k]
    dy = gy.unsqueeze(0) - params.cy.view(num_boxes, 1, 1)
    ba = -params.angle.view(num_boxes, 1, 1)
    lx = dx * torch.cos(ba) - dy * torch.sin(ba)
    ly = dx * torch.sin(ba) + dy * torch.cos(ba)
    inside = (lx.abs() <= (params.w / 2).view(num_boxes, 1, 1)) & \
             (ly.abs() <= (params.h / 2).view(num_boxes, 1, 1))  # [i, j, k]

    areas = _exact_areas(params)
    inter_area = inside.float().mean(dim=-1) * areas.unsqueeze(0)
    union = areas.unsqueeze(1) + areas.unsqueeze(0) - inter_area
    return inter_area / union.clamp(min=1e-8)                  # (num_boxes, num_boxes)


def _greedy_suppress_fixpoint(iou, iou_threshold):
    """贪心 NMS 抑制的不动点迭代（抑制链安全）。

    框 j 被抑制 ⟺ 存在分数更高的**保留**框 i 与其 IoU 超阈值；
    迭代 suppress[j] ← ∃ i<j: iou(i,j) > thr 且 ¬suppress[i]
    直至收敛（典型 2~3 轮，每轮全张量化），收敛点即贪心 NMS 的
    不动点刻画，无逐框 Python 循环。
    """
    num_boxes = iou.shape[0]
    tri = torch.triu(torch.ones(num_boxes, num_boxes, device=iou.device,
                                dtype=torch.bool), diagonal=1)
    overlap = (iou > iou_threshold) & tri                      # [i, j] = i 抑制候选
    suppress = overlap.any(dim=0)                              # 一次性判定 (上界)
    for _ in range(num_boxes):
        new_suppress = (overlap & ~suppress.unsqueeze(1)).any(dim=0)
        if torch.equal(new_suppress, suppress):
            break
        suppress = new_suppress
    return suppress


def _extract_keep_indices(keep_mask, order):
    """keep 索引提取: 掩码前缀和 + searchsorted（无 nonzero/布尔索引）。"""
    device = order.device
    prefix = keep_mask.to(torch.int32).cumsum(0)
    num_keep = int(prefix[-1].item())
    if num_keep == 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    targets = torch.arange(1, num_keep + 1, device=device, dtype=torch.int32)
    keep_pos = torch.searchsorted(prefix, targets)
    return order[keep_pos]


def nms_bev(
    boxes_bev: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float
) -> torch.Tensor:
    """
    对旋转 BEV 框执行 NMS（Non-Maximum Suppression）。

    功能：
        按分数从高到低贪心保留，抑制与其旋转平面 IoU 超过阈值的其他框，
        抑制语义与标准贪心 NMS 一致（抑制链安全：被抑制的框不再参与
        对后续框的抑制）。
        与 nms3d（IoU3D）不同，本算子只计算 BEV 平面（俯视）旋转 IoU，
        适用于 z 方向重叠由任务隐含保证的场景（如 KITTI 同类目标）。

    实现流程（各步骤为模块级函数，全程张量化）：
        1. 按分数降序排序；
        2. `_rotated_bev_iou_matrix`: 确定性 4x5 分层网格面积采样估计
           旋转 IoU 矩阵（非随机数、非共线，多次执行逐位一致）；
        3. `_greedy_suppress_fixpoint`: 贪心抑制的不动点迭代
           （抑制链安全，典型 2~3 轮收敛）；
        4. `_extract_keep_indices`: 掩码前缀和 + searchsorted 提取
           keep 索引（无 nonzero / 布尔索引）。

        算子包含少量 CPU 同步（不动点收敛判断与结果计数）。

    Args:
        boxes_bev: (N, 7)
            旋转 3D 框（仅使用 BEV 分量）
            [cx, cy, cz, dx, dy, dz, yaw]
        scores: (N,)
            每个框的置信度分数
        iou_threshold: float
            IoU 阈值，大于该值则抑制

    Returns:
        keep_indices: (M,)
            保留下来的框索引（按分数从高到低排列），M <= N

    已知限制：
        - IoU 为面积采样估计（误差 O(1/采样数)），阈值附近的框对
          可能与精确多边形 IoU 的判定不同
        - 仅使用 BEV 平面分量，不做高度方向校验
    """
    if boxes_bev.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes_bev.device)

    order = torch.argsort(scores, descending=True)
    if boxes_bev.shape[0] == 1:
        return order

    boxes_sorted = boxes_bev[order].contiguous()
    iou = _rotated_bev_iou_matrix(boxes_sorted)
    suppress = _greedy_suppress_fixpoint(iou, iou_threshold)
    return _extract_keep_indices(~suppress, order)


# ============================================================
# 注册实现
# ============================================================
_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("nms_bev", nms_bev)


# ============================================================
# Meta 实现（等价 register_fake）
# ============================================================
def _nms_bev_meta(boxes_bev, scores, iou_threshold):
    # 输出 M 与数据相关（M <= N），此处给出 N 作为形状上界
    return torch.empty((boxes_bev.shape[0],), device="meta", dtype=torch.long)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("nms_bev", _nms_bev_meta)
