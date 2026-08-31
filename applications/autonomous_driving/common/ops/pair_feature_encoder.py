# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# utils/pair_feature_encoder.py 直接替换成这个版本
import torch
from torch.library import Library


def pair_feature_encoder(
    pair_indices: torch.Tensor,
    iou_matrix: torch.Tensor,
    boxes_3d: torch.Tensor,
    scores_2d: torch.Tensor,
    scores_3d: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fusionrepo.pair_feature_encoder(
        pair_indices, iou_matrix, boxes_3d, scores_2d, scores_3d
    )


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")


def pair_feature_encoder_impl(
    pair_indices: torch.Tensor,
    iou_matrix: torch.Tensor,
    boxes_3d: torch.Tensor,
    scores_2d: torch.Tensor,
    scores_3d: torch.Tensor,
) -> torch.Tensor:
    # 输出 (P,4): [iou2d, score3d, score2d, distance3d] —— 与原 CLOCS build_stage2_training 一致
    if pair_indices.numel() == 0:
        return torch.empty((0, 4), device=iou_matrix.device, dtype=iou_matrix.dtype)

    i = pair_indices[:, 0].long()
    j = pair_indices[:, 1].long()

    iou = iou_matrix[i, j].to(dtype=iou_matrix.dtype).view(-1, 1)          # (P,1)
    s2 = scores_2d[i].to(dtype=iou_matrix.dtype).view(-1, 1)               # (P,1)
    s3 = scores_3d[j].to(dtype=iou_matrix.dtype).view(-1, 1)               # (P,1)

    # distance: 原 CLOCS 用 3D box center 的 BEV (x,y) 距离 /82，不 clamp
    b3d = boxes_3d[j].to(dtype=iou_matrix.dtype)
    if b3d.dim() != 2:
        b3d = b3d.view(b3d.shape[0], -1)
    dist = torch.norm(b3d[:, :2], dim=1).view(-1, 1)
    dist = dist / 82.0

    # 特征顺序 [iou, s3, s2, dist]（与原 CLOCS 一致，s2/s3 位置不可颠倒）
    return torch.cat([iou, s3, s2, dist], dim=1)                            # (P,4)


_lib_impl.impl("pair_feature_encoder", pair_feature_encoder_impl)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")


def pair_feature_encoder_meta(
    pair_indices: torch.Tensor,
    iou_matrix: torch.Tensor,
    boxes_3d: torch.Tensor,
    scores_2d: torch.Tensor,
    scores_3d: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((pair_indices.shape[0], 4), device="meta", dtype=iou_matrix.dtype)


_lib_meta.impl("pair_feature_encoder", pair_feature_encoder_meta)
