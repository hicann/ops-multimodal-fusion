# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Rotated box IoU for BEV (camera x-z plane), CPU implementation.

Replaces the previous hand-written numba quadrilateral-intersection port, which
had float32 precision bugs (self-IoU=0 / negative IoU at certain angles).

This version uses shapely (robust polygon intersection) and matches the
official mmdet3d rotate_iou angle convention (rotation is clockwise when the
angle is positive). Interface is identical to mmdet3d's rotate_iou_gpu_eval:

    rotate_iou_cpu(boxes, query_boxes, criterion=-1) -> (N, K) float32

criterion: -1 means IoU, 0 means area_inter/area1, 1 means area_inter/area2,
           any other value means raw intersection area (used by d3_box_overlap).
"""
import numpy as np
from shapely.geometry import Polygon


def _rbox_to_polygon(rbox):
    """rbox [cx, cy, dim_x, dim_y, angle] -> shapely Polygon.

    Mirrors mmdet3d rotate_iou.rbbox_to_corners exactly: the four BEV corner
    offsets are (-x_d/2, -y_d/2), (-x_d/2, y_d/2), (x_d/2, y_d/2) and
    (x_d/2, -y_d/2), rotated by angle (cos/sin) and translated by (cx, cy).
    """
    cx, cy, x_d, y_d, angle = float(rbox[0]), float(rbox[1]), float(rbox[2]), \
        float(rbox[3]), float(rbox[4])
    a_cos = np.cos(angle)
    a_sin = np.sin(angle)
    lx = np.array([-x_d / 2.0, -x_d / 2.0, x_d / 2.0, x_d / 2.0])
    ly = np.array([-y_d / 2.0, y_d / 2.0, y_d / 2.0, -y_d / 2.0])
    x = a_cos * lx + a_sin * ly + cx
    y = -a_sin * lx + a_cos * ly + cy
    return Polygon(list(zip(x, y)))


def rotate_iou_cpu(boxes, query_boxes, criterion=-1):
    """CPU rotated-box IoU, robust shapely implementation."""
    boxes = np.asarray(boxes, dtype=np.float64)
    query_boxes = np.asarray(query_boxes, dtype=np.float64)
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return iou

    areas1 = boxes[:, 2] * boxes[:, 3]
    areas2 = query_boxes[:, 2] * query_boxes[:, 3]

    polys2 = [_rbox_to_polygon(q) for q in query_boxes]
    for i in range(N):
        p1 = _rbox_to_polygon(boxes[i])
        for j in range(K):
            inter = p1.intersection(polys2[j]).area
            a1 = areas1[i]
            a2 = areas2[j]
            if criterion == -1:
                iou[i, j] = inter / (a1 + a2 - inter + 1e-10)
            elif criterion == 0:
                iou[i, j] = inter / (a1 + 1e-10)
            elif criterion == 1:
                iou[i, j] = inter / (a2 + 1e-10)
            else:
                iou[i, j] = inter
    return iou
