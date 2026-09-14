# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
eval_hmfi_official.py

用官方 mmdet3d KittiMetric（与 PointPillars run_official_eval.py / CLOCs eval_clocs_official.py
同一把尺子）评估 HMFI。复用 test_hmfi 的构建/前向/融合权重加载，把 final_results 的 3D 框
转 LiDARInstance3DBoxes 喂入同一个 KittiMetric（kitti_infos_val_v2.pkl）。

用法:
  cd <autonomous_driving 根目录> && python3 model/hmfi/eval_hmfi_official.py \
      --device npu:0 [--max-frames 0] [--out results/hmfi/official_eval_result]

说明:
  - --max-frames >0 且 < 全量时为冒烟模式（只前向，不 compute_metrics）
  - 依赖 mmcv 稀疏卷积纯 torch 替换 + rpn_head proposal（见 test_hmfi.py 顶部 import）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault('NUMBA_NUM_THREADS', '8')

# rotate_iou CUDA -> CPU stub（在 import kitti_eval 之前）
AD_ROOT = Path(__file__).resolve().parents[2]
_riu = 'mmdet3d.evaluation.functional.kitti_utils.rotate_iou'
if _riu not in sys.modules:
    import types
    _s = types.ModuleType(_riu)
    sys.path.insert(0, str(AD_ROOT / "common" / "kitti_eval"))
    from rotate_iou_cpu import rotate_iou_cpu
    _s.rotate_iou_gpu_eval = rotate_iou_cpu
    sys.modules[_riu] = _s

import torch

import mmdet3d
from mmdet3d.evaluation.metrics.kitti_metric import KittiMetric
from mmdet3d.structures import LiDARInstance3DBoxes
from mmengine import load

# 项目路径
sys.path.insert(0, str(AD_ROOT))
sys.path.insert(0, str(AD_ROOT / "model" / "pointpillars"))
sys.path.insert(0, str(AD_ROOT / "common" / "pcdet_stub"))

import common.ops  # noqa: F401
import common.patches.mmcv_sparse_conv  # noqa: F401  mmcv 稀疏卷积 NPU 替换

from model.hmfi import test_hmfi as T
from model.hmfi.initialization.build_hmfi_system import build_hmfi_system, test_step

DATA_ROOT = AD_ROOT / "data" / "kitti"
V2_PKL = DATA_ROOT / "kitti_infos_val_v2.pkl"


def main():
    args = T.build_argparser().parse_args()

    if args.device.startswith('npu'):
        import torch_npu  # noqa: F401

    infos = load(str(V2_PKL))
    data_list = infos['data_list']
    n_total = len(data_list)
    smoke = (args.max_frames > 0) and (args.max_frames < n_total)
    if smoke:
        data_list = data_list[:args.max_frames]
    n_run = len(data_list)
    print(f"[hmfi-official] device={args.device} frames={n_run}/{n_total} (smoke={smoke})")

    # ---- 构建系统（复刻 test_hmfi main 逻辑）----
    point_cloud_feat_index = None if args.point_cloud_feat_index < 0 else args.point_cloud_feat_index
    depth_range = tuple(float(x) for x in args.depth_range.split(","))
    voxel_size = tuple(float(x) for x in args.voxel_size.split(","))
    point_cloud_range = tuple(float(x) for x in args.point_cloud_range.split(","))
    code_weights = T.parse_float_list(args.code_weights, 7, "--code_weights")

    sys_ = build_hmfi_system(
        image_cfg=T.normalize_path(args.image_cfg),
        image_ckpt=T.normalize_path(args.image_ckpt),
        point_cloud_cfg=T.normalize_path(args.point_cloud_cfg),
        point_cloud_ckpt=T.normalize_path(args.point_cloud_ckpt),
        device=args.device,
        image_feat_index=args.image_feat_index,
        point_cloud_feat_index=point_cloud_feat_index,
        lidar_channels=args.lidar_channels,
        image_channels=args.image_channels,
        fusion_out_channels=args.fusion_out_channels,
        d_k=args.d_k,
        d_v=args.d_v,
        num_heads=args.num_heads,
        depth_mode=args.depth_mode,
        downsample_factor=args.downsample_factor,
        depth_range=depth_range,
        num_depth_bins=args.num_depth_bins,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        use_vfim=args.use_vfim,
        vfim_loss_weight=args.vfim_loss_weight,
        alpha=args.alpha,
        pos_iou_thr=args.pos_iou_thr,
        neg_iou_thr=args.neg_iou_thr,
        code_weights=code_weights,
        max_pos_weight=args.max_pos_weight,
        score_thr=args.score_thr,
        nms_thr=args.nms_thr,
        pre_nms_topk=args.pre_nms_topk,
        post_nms_topk=args.post_nms_topk,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        debug_base_loss=args.debug_base_loss,
        freeze_image_backbone=args.freeze_image_backbone,
        freeze_point_cloud_backbone=args.freeze_point_cloud_backbone,
    )
    T.load_fusion_checkpoint(sys_, T.normalize_path(args.fusion_ckpt), for_test=True)

    metric = KittiMetric(
        ann_file=str(V2_PKL),
        pcd_limit_range=[0, -40, -3, 70.4, 40, 0.0],
        prefix='Kitti metric',
    )
    metric.dataset_meta = {'classes': ['Car']}

    image_dir = DATA_ROOT / "training" / "image_2"
    velodyne_dir = DATA_ROOT / "training" / "velodyne"
    calib_dir = DATA_ROOT / "training" / "calib"

    t_start = time.time()
    total_dets = 0
    for idx, info in enumerate(data_list):
        sid = os.path.splitext(info['lidar_points']['lidar_path'])[0]
        img_path = str(image_dir / (sid + ".png"))
        lidar_path = str(velodyne_dir / (sid + ".bin"))
        calib_path = str(calib_dir / (sid + ".txt"))

        calib = T.read_kitti_calib(calib_path)
        batch = T.build_eval_batch(img_path, lidar_path, calib, args.device)
        out = test_step(sys_, batch)
        fr = out['final_results']
        if fr:
            r = fr[0]
            boxes = r['boxes_3d'].detach().cpu()      # (M,7) [x,y,z,l,w,h,yaw]
            scores = r['scores_3d'].detach().cpu().float().reshape(-1)
            labels = r['labels_3d'].detach().cpu().long().reshape(-1)
        else:
            boxes = torch.zeros((0, 7), dtype=torch.float32)
            scores = torch.zeros((0,), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        n_det = boxes.shape[0]
        total_dets += n_det
        if labels.shape[0] != n_det:
            labels = torch.zeros_like(scores, dtype=torch.long)

        # HMFI 解码框已是 mmdet3d [x,y,z,l,w,h,yaw]（RPN proposal 同约定）。
        # z 为框中心（Part-A2 Anchor3DHead 不应用 KITTI bottom-center 偏移）→ origin=(0.5,0.5,0.5)
        # （实测：origin=(0.5,0.5,0) 会把框下移 h/2，导致 3D IoU 全挂、BEV 正常）
        boxes3d = LiDARInstance3DBoxes(
            boxes.contiguous(), box_dim=7, origin=(0.5, 0.5, 0.5))

        sample = {
            'pred_instances_3d': {
                'bboxes_3d': boxes3d,
                'scores_3d': scores,
                'labels_3d': labels,
            },
            'pred_instances': {},
            'sample_idx': idx,
        }
        metric.process(None, [sample])

        if (idx + 1) % 200 == 0 or idx == 0 or idx == n_run - 1:
            el = time.time() - t_start
            print(f"[hmfi-official] {idx+1}/{n_run} dets累计={total_dets} "
                  f"耗时{el:.0f}s", flush=True)

    infer_time = time.time() - t_start
    print(f"[hmfi-official] 推理完成 {infer_time:.0f}s 检测={total_dets}")

    if smoke:
        print("[hmfi-official] 冒烟模式：跳过 compute_metrics（官方 KittiMetric 需全量）")
        return

    print("[hmfi-official] 开始 compute_metrics (官方 KittiMetric)...")
    t_eval = time.time()
    metric_dict = metric.compute_metrics(metric.results)
    print(f"[hmfi-official] kitti_eval 完成 {time.time()-t_eval:.0f}s")

    if args.out:
        out_pre = args.out
        out_txt = Path(out_pre + ".txt")
        out_json = Path(out_pre + ".json")
    else:
        out_dir = AD_ROOT / "results" / "hmfi"
        out_txt = out_dir / "official_eval_result.txt"
        out_json = out_dir / "official_eval_summary.json"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, 'w') as f:
        for k, v in metric_dict.items():
            f.write(f"{k}: {v}\n")
    with open(out_json, 'w') as f:
        json.dump({k: float(v) for k, v in metric_dict.items()}, f, indent=2)

    print("[hmfi-official] 结果指标:")
    for k, v in metric_dict.items():
        print(f"  {k} = {v:.4f}")
    print(f"[hmfi-official] 结果已存: {out_txt} / {out_json}")


if __name__ == "__main__":
    main()
