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
# nms_bev 算子黑盒测试（自包含单文件，无 conftest / 外部数据）。
#
# golden 来源:
#   1) 精确凸多边形 IoU（Sutherland-Hodgman 裁剪, CPU, fp64）+ 贪心 NMS —— 精度/功能判定
#   2) 确定性：同一输入两次执行输出逐位一致
# 运行（NPU）:
#   cd /tmp && pytest <repo>/tests/nms_bev/test_nms_bev.py -v
#
# L0 门槛  6  = 接口存在性 + dtype/shape/空输入/单框
# L1 功能 12  = 抑制/保留/旋转/重复框/保序/确定性/与精确 IoU 宽裕一致
# L2 一致性 2  = MC IoU vs 精确多边形 IoU 的 MERE / MARE（采样点口径）

import math
import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "applications" / "autonomous_driving" / "common"))

import ops  # noqa: E402  触发 operator_registry (DEF) 与全部 IMPL 注册

DEVICE = "npu" if torch.npu.is_available() else "cpu"


# ------------------------------------------------------------
# golden: 精确凸多边形 IoU (Sutherland-Hodgman, fp64) + 贪心 NMS
# ------------------------------------------------------------

def _box_corners(box):
    cx, cy, dx, dy, yaw = box[0], box[1], box[3], box[4], box[6]
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = dx / 2, dy / 2
    # 逆时针序 (inside 判定约定)
    pts = [(hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)]
    return [(cx + x * c - y * s, cy + x * s + y * c) for (x, y) in pts]


def _point_inside(p, a, b):
    """点 p 是否在有向边 a->b 的内侧。"""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0


def _edge_intersect(a, b, p, q):
    """线段 pq 与有向边 a->b 所在直线的交点。"""
    d1 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    d2 = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
    t = d1 / (d1 - d2)
    return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))


def _clip_edge(input_pts, a, b):
    """以有向边 a->b 对多边形做一轮裁剪。"""
    output = []
    if not input_pts:
        return output
    prev = input_pts[-1]
    for cur in input_pts:
        cur_in = _point_inside(cur, a, b)
        prev_in = _point_inside(prev, a, b)
        if cur_in and not prev_in:
            output.append(_edge_intersect(a, b, prev, cur))
            output.append(cur)
        elif cur_in:
            output.append(cur)
        elif prev_in:
            output.append(_edge_intersect(a, b, prev, cur))
        prev = cur
    return output


def _clip(subject, clipper):
    output = list(subject)
    for edge_idx, a in enumerate(clipper):
        b = clipper[(edge_idx + 1) % len(clipper)]
        output = _clip_edge(output, a, b)
    return output


def _poly_area(poly):
    if len(poly) < 3:
        return 0.0
    area2 = 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        area2 += x1 * y2 - x2 * y1
    return abs(area2) / 2


def _exact_iou_matrix(boxes):
    """boxes: (num_boxes,7) numpy/tensor-like -> (num_boxes,num_boxes) fp64 exact BEV IoU."""
    num_boxes = len(boxes)
    polys = [_box_corners(b) for b in boxes]
    areas = [_poly_area(p) for p in polys]
    iou = [[0.0] * num_boxes for _ in range(num_boxes)]
    for i, poly_i in enumerate(polys):
        for j, poly_j in enumerate(polys):
            if i == j:
                iou[i][j] = 1.0
                continue
            inter = _poly_area(_clip(poly_i, poly_j))
            union = areas[i] + areas[j] - inter
            iou[i][j] = inter / union if union > 0 else 0.0
    return iou


def _suppress_overlapping(iou, chosen, thr, suppressed):
    for j, row in enumerate(iou):
        if j != chosen and row[chosen] > thr:
            suppressed[j] = True


def _greedy_nms_exact(boxes, scores, thr):
    iou = _exact_iou_matrix(boxes)
    num_boxes = len(boxes)
    order = sorted(range(num_boxes), key=lambda k: -float(scores[k]))
    keep, suppressed = [], [False] * num_boxes
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        _suppress_overlapping(iou, idx, thr, suppressed)
    return sorted(keep)


# ------------------------------------------------------------
# 用例生成工具
# ------------------------------------------------------------

def _rand_boxes(num_boxes, seed, spread=40.0):
    g = torch.Generator().manual_seed(seed)
    boxes = torch.zeros(num_boxes, 7)
    boxes[:, 0] = torch.rand(num_boxes, generator=g) * spread
    boxes[:, 1] = torch.rand(num_boxes, generator=g) * spread - spread / 2
    boxes[:, 2] = 1.0
    boxes[:, 3] = 1.4 + torch.rand(num_boxes, generator=g) * 0.8
    boxes[:, 4] = 3.6 + torch.rand(num_boxes, generator=g) * 1.2
    boxes[:, 5] = 1.5
    boxes[:, 6] = (torch.rand(num_boxes, generator=g) - 0.5) * math.pi
    scores = torch.rand(num_boxes, generator=g)
    return boxes, scores


def _make_boxes(offs, yaw=0.0):
    """一组平移错开的同尺寸框: 每个偏移一个框, x = 10 + off。"""
    rows = [[10.0 + off, 10.0, 1.0, 4.0, 2.0, 1.5, yaw] for off in offs]
    return torch.tensor(rows)


# ------------------------------------------------------------
# L0 门槛
# ------------------------------------------------------------

def test_l0_interface_exists():
    assert hasattr(torch.ops.fusionrepo, "nms_bev")


def test_l0_empty_input():
    boxes = torch.zeros(0, 7, device=DEVICE)
    scores = torch.zeros(0, device=DEVICE)
    keep = torch.ops.fusionrepo.nms_bev(boxes, scores, 0.5)
    assert keep.shape == (0,)
    assert keep.device.type == DEVICE.split(":")[0]


def test_l0_single_box():
    boxes, scores = _rand_boxes(1, seed=1)
    keep = torch.ops.fusionrepo.nms_bev(boxes.to(DEVICE), scores.to(DEVICE), 0.5)
    assert keep.tolist() == [0]


def test_l0_output_dtype_device():
    boxes, scores = _rand_boxes(8, seed=2)
    keep = torch.ops.fusionrepo.nms_bev(boxes.to(DEVICE), scores.to(DEVICE), 0.5)
    assert keep.dtype == torch.long
    assert keep.device.type == DEVICE.split(":")[0]
    assert keep.numel() >= 1


# ------------------------------------------------------------
# L1 功能
# ------------------------------------------------------------

def test_l1_identical_boxes_suppressed():
    # 完全相同的框: 只有最高分保留
    boxes = torch.tensor([[5.0, 5.0, 1.0, 4.0, 2.0, 1.5, 0.3]] * 4)
    scores = torch.tensor([0.1, 0.9, 0.5, 0.3])
    keep = torch.ops.fusionrepo.nms_bev(boxes.to(DEVICE), scores.to(DEVICE), 0.5)
    assert keep.tolist() == [1]


def test_l1_disjoint_boxes_all_kept():
    boxes = _make_boxes([0.0, 30.0, 60.0])
    scores = torch.tensor([0.2, 0.8, 0.5])
    keep = torch.ops.fusionrepo.nms_bev(boxes.to(DEVICE), scores.to(DEVICE), 0.5)
    assert sorted(keep.tolist()) == [0, 1, 2]


def test_l1_keep_sorted_by_score():
    boxes, scores = _rand_boxes(20, seed=3)
    keep = torch.ops.fusionrepo.nms_bev(boxes.to(DEVICE), scores.to(DEVICE), 0.5)
    vals = scores[keep.cpu()].tolist()
    assert vals == sorted(vals, reverse=True)
    assert len(set(keep.tolist())) == keep.numel()


def _axis_aligned_iou_off(off, w=4.0, h=2.0):
    """同尺寸轴对齐框, x 向偏移 off 时的精确 IoU。"""
    inter = max(0.0, w - off) * h
    return inter / (2 * w * h - inter)


def test_l1_heavy_overlap_suppressed():
    # 同尺寸轴对齐双框, x 偏移 0.8 → 精确 IoU ≈ 0.667 > 0.5 → 抑制
    iou_exact = _axis_aligned_iou_off(0.8)
    assert iou_exact > 0.65
    boxes = _make_boxes([0.0, 0.8])
    scores = torch.tensor([0.9, 0.8])
    keep = torch.ops.fusionrepo.nms_bev(boxes.to(DEVICE), scores.to(DEVICE), 0.5)
    assert keep.tolist() == [0]


def test_l1_light_overlap_kept():
    # IoU ≈ 0.23 < 0.5 → 保留
    iou_exact = _axis_aligned_iou_off(2.6)
    assert iou_exact < 0.4
    boxes = _make_boxes([0.0, 2.6])
    scores = torch.tensor([0.9, 0.8])
    keep = torch.ops.fusionrepo.nms_bev(boxes.to(DEVICE), scores.to(DEVICE), 0.5)
    assert sorted(keep.tolist()) == [0, 1]


def test_l1_rotated_boxes_vs_exact():
    # 旋转双框偏移扫描: 非歧义对 (|exact IoU - 0.5| > 0.15) 的
    # MC 判定必须与精确多边形 IoU 一致 (自标定, 不依赖手选 offset)
    for yaw in (0.6, 1.2):
        agreed = 0
        for k in range(10):
            off = 0.3 + k * 0.35
            boxes = _make_boxes([0.0, off], yaw=yaw).numpy()
            exact = _exact_iou_matrix(boxes)[0][1]
            if abs(exact - 0.5) <= 0.15:
                continue
            scores = torch.tensor([0.9, 0.8])
            keep = torch.ops.fusionrepo.nms_bev(
                torch.tensor(boxes).to(DEVICE), scores.to(DEVICE), 0.5).cpu().tolist()
            suppressed = 1 not in keep
            assert suppressed == (exact > 0.5), f"yaw={yaw} off={off:.2f} exact={exact:.3f}"
            agreed += 1
        assert agreed >= 3, f"yaw={yaw} 有效对不足"


def test_l1_matches_greedy_exact_wide_margin():
    # 随机场景: 采样 IoU 与精确 IoU 均远离阈值 (|IoU-0.5| > 0.15) 的框,
    # 其保留/抑制判定必须与精确贪心 NMS 一致
    boxes, scores = _rand_boxes(30, seed=7)
    iou = _exact_iou_matrix(boxes.numpy())
    num_boxes = len(boxes)
    order = sorted(range(num_boxes), key=lambda k: -float(scores[k]))
    rank = {v: i for i, v in enumerate(order)}
    ambiguous = set()
    for i in range(num_boxes):
        for j in range(num_boxes):
            if i != j and abs(iou[i][j] - 0.5) < 0.15:
                ambiguous.add(rank[i])
                ambiguous.add(rank[j])
    keep_op = set(torch.ops.fusionrepo.nms_bev(
        boxes.to(DEVICE), scores.to(DEVICE), 0.5).cpu().tolist())
    keep_exact = set(_greedy_nms_exact(boxes.numpy(), scores.tolist(), 0.5))
    for i in range(num_boxes):
        if rank[i] in ambiguous:
            continue
        assert (i in keep_op) == (i in keep_exact), f"box {i} 判定不一致"


def test_l1_duplicate_keeps_highest_score():
    boxes = torch.tensor([
        [1.0, 1.0, 1.0, 4.0, 2.0, 1.5, 0.0],
        [1.05, 1.0, 1.0, 4.0, 2.0, 1.5, 0.0],   # 近重复
        [1.0, 1.0, 1.0, 4.0, 2.0, 1.5, 0.0],    # 完全重复
    ])
    scores = torch.tensor([0.3, 0.95, 0.6])
    keep = torch.ops.fusionrepo.nms_bev(boxes.to(DEVICE), scores.to(DEVICE), 0.5)
    assert keep.tolist() == [1]


def test_l1_deterministic():
    boxes, scores = _rand_boxes(50, seed=11)
    b, s = boxes.to(DEVICE), scores.to(DEVICE)
    k1 = torch.ops.fusionrepo.nms_bev(b, s, 0.5)
    k2 = torch.ops.fusionrepo.nms_bev(b, s, 0.5)
    k3 = torch.ops.fusionrepo.nms_bev(b.clone(), s.clone(), 0.5)
    assert torch.equal(k1, k2) and torch.equal(k1, k3)


def test_l1_scale_invariance():
    # 整体平移后结果不变（平移不变性）
    boxes, scores = _rand_boxes(25, seed=13)
    boxes_t = boxes.clone()
    boxes_t[:, 0] += 100.0
    boxes_t[:, 1] -= 50.0
    k1 = torch.ops.fusionrepo.nms_bev(boxes.to(DEVICE), scores.to(DEVICE), 0.5)
    k2 = torch.ops.fusionrepo.nms_bev(boxes_t.to(DEVICE), scores.to(DEVICE), 0.5)
    assert k1.tolist() == k2.tolist()


def test_l1_threshold_sensitivity():
    boxes = _make_boxes([0.0, 1.6])   # IoU ≈ 0.48-0.52 区间附近
    scores = torch.tensor([0.9, 0.8])
    keep_strict = torch.ops.fusionrepo.nms_bev(boxes.to(DEVICE), scores.to(DEVICE), 0.3)
    keep_loose = torch.ops.fusionrepo.nms_bev(boxes.to(DEVICE), scores.to(DEVICE), 0.9)
    assert 0 in keep_strict.cpu().tolist()
    assert sorted(keep_loose.cpu().tolist()) == [0, 1]


# ------------------------------------------------------------
# L2 一致性: MC IoU vs 精确多边形 IoU (MERE / MARE, 采样点口径)
# ------------------------------------------------------------

def test_l2_mc_iou_accuracy_axis_aligned():
    # 轴对齐同尺寸双框: 精确 IoU 解析已知, 验证 MC 判定边界正确
    offs = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.6, 6.0]
    thr = 0.5
    for off in offs:
        exact = _axis_aligned_iou_off(off)
        boxes = _make_boxes([0.0, off])
        scores = torch.tensor([0.9, 0.8])
        keep = torch.ops.fusionrepo.nms_bev(boxes.to(DEVICE), scores.to(DEVICE), thr).cpu().tolist()
        suppressed = 1 not in keep
        # 远离阈值的用例判定必须一致
        if exact > thr + 0.1:
            assert suppressed, f"off={off} exact={exact:.3f} 应抑制"
        if exact < thr - 0.1:
            assert not suppressed, f"off={off} exact={exact:.3f} 应保留"


def test_l2_mc_iou_relative_error_bounded():
    # 旋转 ~30° 双框扫描: 精确 IoU 与 MC 判定边界方向一致
    yaw = 0.5
    offs = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
    checked = 0
    for off in offs:
        boxes = _make_boxes([0.0, off], yaw=yaw).numpy()
        exact = _exact_iou_matrix(boxes)[0][1]
        if exact < 0.2 or exact > 0.8:      # 只统计远离边界的对
            continue
        boxes_t = torch.tensor(boxes)
        scores = torch.tensor([0.9, 0.8])
        keep = torch.ops.fusionrepo.nms_bev(
            boxes_t.to(DEVICE), scores.to(DEVICE), 0.5).cpu().tolist()
        suppressed = 1 not in keep
        expect = exact > 0.5
        if abs(exact - 0.5) > 0.1:          # 阈值附近 ±0.1 内不判定
            assert suppressed == expect, f"off={off:.2f} exact={exact:.3f}"
            checked += 1
    assert checked >= 4


def _axis_iou_pair(w1, h1, w2, h2, off):
    """轴对齐双框 (中心同 y), 框 2 沿 x 偏移 off 的精确 IoU。"""
    x_lo = max(-w1 / 2, off - w2 / 2)
    x_hi = min(w1 / 2, off + w2 / 2)
    ix = max(0.0, x_hi - x_lo)
    iy = min(h1, h2)
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def test_l1_containment_small_inside_large_kept():
    # 小框完整包含于大框: 精确 IoU = a_small / a_large (远小于 1),
    # 面积因子若乘错框会把 IoU 放大并错误抑制小框 (P1 回归)
    big = [10.0, 10.0, 1.0, 10.0, 6.0, 1.5, 0.0]
    small = [10.0, 10.0, 1.0, 2.0, 1.2, 1.5, 0.0]
    exact = _axis_iou_pair(10.0, 6.0, 2.0, 1.2, 0.0)
    assert exact < 0.1, exact
    boxes = torch.tensor([big, small])
    scores = torch.tensor([0.9, 0.8])
    keep = torch.ops.fusionrepo.nms_bev(boxes.to(DEVICE), scores.to(DEVICE), 0.5).cpu().tolist()
    assert sorted(keep) == [0, 1], f"包含场景 IoU 精确值={exact:.3f}, keep={keep}"


# 不同尺寸框对用例: (w1, h1, w2, h2, off, thr)
_DIFF_SIZE_CASES = [
    (10.0, 6.0, 2.0, 1.2, 0.0, 0.5),
    (10.0, 6.0, 6.0, 4.0, 1.0, 0.5),
    (4.0, 2.0, 12.0, 3.0, 2.0, 0.5),
    (8.0, 8.0, 3.0, 3.0, 0.0, 0.5),
]


@pytest.mark.parametrize("case", _DIFF_SIZE_CASES)
def test_l1_diff_size_pairs_match_exact(case):
    w1, h1, w2, h2, off, thr = case
    boxes = torch.tensor([
        [10.0, 10.0, 1.0, w1, h1, 1.5, 0.0],
        [10.0 + off, 10.0, 1.0, w2, h2, 1.5, 0.0],
    ])
    scores = torch.tensor([0.9, 0.8])
    exact = _axis_iou_pair(w1, h1, w2, h2, off)
    keep = torch.ops.fusionrepo.nms_bev(boxes.to(DEVICE), scores.to(DEVICE), thr).cpu().tolist()
    if abs(exact - thr) > 0.1:
        expect_suppressed = exact > thr
        suppressed = 1 not in keep
        assert suppressed == expect_suppressed, f"exact={exact:.3f} thr={thr} keep={keep}"


def test_l1_suppression_chain_not_transitive():
    # A(0.9)-B(0.6) 重叠, B-C(0.5) 重叠, A-C 不重叠。
    # 贪心: 保留 A 抑制 B; C 未与保留框重叠 -> 保留 C。
    # 一次性三角抑制会错误地把 C 一并抑制 (P2-2 回归)
    boxes = torch.tensor([
        [10.0, 10.0, 1.0, 4.0, 2.0, 1.5, 0.0],
        [12.0, 10.0, 1.0, 4.0, 2.0, 1.5, 0.0],
        [14.2, 10.0, 1.0, 4.0, 2.0, 1.5, 0.0],
    ])
    scores = torch.tensor([0.9, 0.6, 0.5])
    iou_ab = _axis_iou_pair(4.0, 2.0, 4.0, 2.0, 2.0)
    iou_bc = _axis_iou_pair(4.0, 2.0, 4.0, 2.0, 2.2)
    iou_ac = _axis_iou_pair(4.0, 2.0, 4.0, 2.0, 4.2)
    thr = 0.25
    assert iou_ab > thr > iou_ac, (iou_ab, iou_ac)
    assert iou_bc > thr, iou_bc
    keep = torch.ops.fusionrepo.nms_bev(boxes.to(DEVICE), scores.to(DEVICE), thr).cpu().tolist()
    assert keep == [0, 2], f"贪心应保留 A 与 C, got {keep}"


def test_l1_chain_matches_exact_greedy_random():
    # 随机场景 (含抑制链): 与精确贪心 NMS 对拍 (排除阈值附近歧义框)
    boxes, scores = _rand_boxes(40, seed=21)
    keep_op = set(torch.ops.fusionrepo.nms_bev(
        boxes.to(DEVICE), scores.to(DEVICE), 0.3).cpu().tolist())
    keep_exact = set(_greedy_nms_exact(boxes.numpy(), scores.tolist(), 0.3))
    iou = _exact_iou_matrix(boxes.numpy())
    num_boxes = len(boxes)
    order = sorted(range(num_boxes), key=lambda k: -float(scores[k]))
    rank = {v: i for i, v in enumerate(order)}
    ambiguous = set()
    for i in range(num_boxes):
        for j in range(num_boxes):
            if i != j and abs(iou[i][j] - 0.3) < 0.1:
                ambiguous.add(rank[i])
                ambiguous.add(rank[j])
    for i in range(num_boxes):
        if rank[i] in ambiguous:
            continue
        assert (i in keep_op) == (i in keep_exact), f"box {i} 判定不一致"
