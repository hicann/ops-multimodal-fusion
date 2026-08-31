# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
eval_clocs_official.py

用官方 mmdet3d KittiMetric（与 PointPillars run_official_eval.py 同一把尺子）评估 CLOCS。

CLOCS 是独立框架（非 mmdet3d model），本脚本把 CLOCS test_step 输出的 3D 框转成
LiDARInstance3DBoxes，喂入同一个 KittiMetric（kitti_infos_val_v2.pkl），
得到与 PointPillars 逐项同口径的指标（3D/BEV/2D AP11/AP40 easy/moderate/hard）。

用法:
  cd <autonomous_driving 根目录> && python3 model/clocs/eval_clocs_official.py \
      --device npu:0 [--max-frames 0] [--out results/clocs/official_eval_result]

说明:
  - --max-frames >0 且 < 全量时为冒烟模式（只前向验证，不 compute_metrics；
    官方 KittiMetric 要求样本数 == data_list 长度）。
  - 全量 3769 帧 val 与 PointPillars run_official_eval.py 完全同源（同一 v2 pkl）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# numba 死锁防护：限制线程（与 run_official_eval.py 一致）
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

import torch  # noqa: E402

import mmdet  # noqa: E402
import mmdet3d  # noqa: E402
from mmdet3d.evaluation.metrics.kitti_metric import KittiMetric  # noqa: E402
from mmdet3d.structures import LiDARInstance3DBoxes  # noqa: E402
from mmengine import load  # noqa: E402

# 项目路径：让 model.clocs / common / pcdet_stub 可导入
sys.path.insert(0, str(AD_ROOT))
sys.path.insert(0, str(AD_ROOT / "common" / "pcdet_stub"))

import common.ops  # noqa: F401  (激活 fusionrepo 自定义算子注册)
from model.clocs.initialization.build_clocs_system import build_clocs_system, test_step  # noqa: E402
from model.clocs.test_script import read_kitti_calib  # noqa: E402

DATA_ROOT = AD_ROOT / "data" / "kitti"
V2_PKL = DATA_ROOT / "kitti_infos_val_v2.pkl"
CKPT_2D = AD_ROOT / "checkpoints" / "clocs" / \
    "faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth"
CKPT_3D = AD_ROOT / "checkpoints" / "pointpillars" / \
    "hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth"
CFG_2D = Path(mmdet.__file__).parent / ".mim" / "configs" / \
    "faster_rcnn" / "faster-rcnn_r50-caffe_fpn_1x_coco.py"
CFG_3D = Path(mmdet3d.__file__).parent / ".mim" / "configs" / \
    "pointpillars" / "pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py"


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="npu:0")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0=全量; >0 且<全量时为冒烟（仅前向，不 compute_metrics）")
    ap.add_argument("--out", type=str, default="",
                    help="结果输出前缀（不含扩展名）")
    ap.add_argument("--det2d_score_thr", type=float, default=0.05)
    ap.add_argument("--det3d_score_thr", type=float, default=0.05)
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

    infos = load(str(V2_PKL))
    data_list = infos['data_list']
    n_total = len(data_list)
    smoke = (args.max_frames > 0) and (args.max_frames < n_total)
    if smoke:
        data_list = data_list[:args.max_frames]
    n_run = len(data_list)
    print(f"[clocs-official] device={args.device} frames={n_run}/{n_total} "
          f"(smoke={smoke})")

    image_dir = DATA_ROOT / "training" / "image_2"
    velodyne_dir = DATA_ROOT / "training" / "velodyne"
    calib_dir = DATA_ROOT / "training" / "calib"

    sys_ = build_clocs_system(
        det2d_cfg=str(CFG_2D),
        det2d_ckpt=str(CKPT_2D),
        det3d_cfg=str(CFG_3D),
        det3d_ckpt=str(CKPT_3D),
        device=args.device,
        det2d_score_thr=args.det2d_score_thr,
        det3d_score_thr=args.det3d_score_thr,
        det2d_allowed_labels=(2,),
        det3d_allowed_labels=(0,),
        score_fuse_mode="weighted_sum",
        freeze_detectors=True,
    )

    metric = KittiMetric(
        ann_file=str(V2_PKL),
        pcd_limit_range=[0, -40, -3, 70.4, 40, 0.0],
        prefix='Kitti metric',
    )
    metric.dataset_meta = {'classes': ['Car']}

    t_start = time.time()
    total_dets = 0
    for idx, info in enumerate(data_list):
        sid = os.path.splitext(info['lidar_points']['lidar_path'])[0]  # 如 003712
        img_path = str(image_dir / (sid + ".png"))
        lidar_path = str(velodyne_dir / (sid + ".bin"))
        calib_path = str(calib_dir / (sid + ".txt"))
        assert os.path.exists(img_path), f"missing image: {img_path}"
        assert os.path.exists(lidar_path), f"missing lidar: {lidar_path}"
        assert os.path.exists(calib_path), f"missing calib: {calib_path}"

        calib = read_kitti_calib(calib_path)
        pred = test_step(sys_, {
            "image": img_path,    # test_step 内部取 batch["image"]
            "images": img_path,
            "points": lidar_path,
            "calib": calib,
        })

        boxes_lidar = pred["boxes_3d"].detach().cpu()  # (M,7) [x,y,z,w,l,h,yaw]
        scores = pred["scores_3d"].detach().cpu().to(torch.float32).reshape(-1)
        n_det = boxes_lidar.shape[0]
        total_dets += n_det
        labels = pred.get("labels_3d", None)
        if labels is None:
            labels = torch.zeros_like(scores, dtype=torch.long)
        else:
            labels = labels.detach().cpu().long().reshape(-1)
        if labels.shape[0] != n_det:
            labels = torch.zeros_like(scores, dtype=torch.long)

        # 接口 [w,l,h] -> mmdet3d LiDARInstance3DBoxes [l,w,h]
        # origin 必须用 (0.5,0.5,0)（相对原点，z 为底心，KITTI 约定）：若用 (0.5,0.5,0.5)
        # 会整体下移 h/2，导致 LiDAR->CAM 转换后 3D IoU 全挂（实测验证）
        if n_det > 0:
            boxes_mm = boxes_lidar[:, [0, 1, 2, 4, 3, 5, 6]].contiguous()
        else:
            boxes_mm = torch.zeros((0, 7), dtype=torch.float32)
        boxes3d = LiDARInstance3DBoxes(boxes_mm, box_dim=7, origin=(0.5, 0.5, 0.0))

        sample = {
            'pred_instances_3d': {
                'bboxes_3d': boxes3d,
                'scores_3d': scores,
                'labels_3d': labels,
            },
            'pred_instances': {},   # CLOCS 不输出 2D 检测，2D AP 恒为 0（符合预期）
            'sample_idx': idx,
        }
        metric.process(None, [sample])

        if (idx + 1) % 200 == 0 or idx == 0 or idx == n_run - 1:
            el = time.time() - t_start
            eta = (n_run - idx - 1) / max((idx + 1) / max(el, 1e-6), 1e-6)
            print(f"[clocs-official] {idx + 1}/{n_run} dets累计={total_dets} "
                  f"耗时{el:.0f}s ETA{eta:.0f}s", flush=True)

    infer_time = time.time() - t_start
    print(f"[clocs-official] 推理完成 {infer_time:.0f}s 检测={total_dets}")

    if smoke:
        print("[clocs-official] 冒烟模式：跳过 compute_metrics（官方 KittiMetric 需全量 3769 帧）")
        return

    print("[clocs-official] 开始 compute_metrics (官方 KittiMetric)...")
    t_eval = time.time()
    metric_dict = metric.compute_metrics(metric.results)
    eval_time = time.time() - t_eval
    print(f"[clocs-official] kitti_eval 完成 {eval_time:.0f}s")

    if args.out:
        out_txt = Path(args.out + ".txt")
        out_json = Path(args.out + ".json")
    else:
        out_dir = AD_ROOT / "results" / "clocs"
        out_txt = out_dir / "official_eval_result.txt"
        out_json = out_dir / "official_eval_summary.json"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, 'w') as f:
        for k, v in metric_dict.items():
            f.write(f"{k}: {v}\n")
    with open(out_json, 'w') as f:
        json.dump({k: float(v) for k, v in metric_dict.items()}, f, indent=2)

    print("[clocs-official] 结果指标:")
    for k, v in metric_dict.items():
        print(f"  {k} = {v:.4f}")
    print(f"[clocs-official] 结果已存: {out_txt} / {out_json}")


if __name__ == "__main__":
    main()
