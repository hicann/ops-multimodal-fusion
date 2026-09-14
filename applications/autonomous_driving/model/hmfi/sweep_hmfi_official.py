# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
sweep_hmfi_official.py

HMFI 后处理阈值扫描（score_thr / nms_thr）。

分两步，避免每个阈值都跑一遍全量前向（~23 min/遍）：

  python3 model/hmfi/sweep_hmfi_official.py dump \
      --device npu:0 --fusion_ckpt checkpoints/hmfi/hmfi_best.pth \
      --dump results/hmfi/sweep_dump.npz

  python3 model/hmfi/sweep_hmfi_official.py sweep \
      --dump results/hmfi/sweep_dump.npz \
      --score_thr_sweep 0.02,0.05,0.1,0.2,0.3 \
      --nms_thr_sweep 0.01,0.05,0.1,0.2 \
      --out results/hmfi/sweep_result.json

dump 步骤把 test_step 返回的 pre-NMS decoded boxes 与融合分数（postprocessor 的
真实输入）按帧存成 npz。sweep 步骤在 CPU 上离线复刻 HMFIPostProcessor 的
score-threshold + BEV-NMS + topk，再喂官方 KittiMetric compute_metrics，
得到与 eval_hmfi_official.py 完全同一把尺子的 AP11/AP40。

用法与 eval_hmfi_official.py 相同：需在 <autonomous_driving 根目录> 运行。
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

import mmdet3d  # noqa: F401
from mmdet3d.evaluation.metrics.kitti_metric import KittiMetric
from mmdet3d.structures import LiDARInstance3DBoxes
from mmengine import load

# 项目路径
sys.path.insert(0, str(AD_ROOT))
sys.path.insert(0, str(AD_ROOT / "model" / "pointpillars"))
sys.path.insert(0, str(AD_ROOT / "common" / "pcdet_stub"))

import common.ops  # noqa: F401
import common.patches.mmcv_sparse_conv  # noqa: F401

from model.hmfi import test_hmfi as T
from model.hmfi.initialization.build_hmfi_system import (
    build_hmfi_system,
    test_step,
    _decode_boxes,
    _xyzhwlr_to_bev_xyxy,
)

DATA_ROOT = AD_ROOT / "data" / "kitti"
V2_PKL = DATA_ROOT / "kitti_infos_val_v2.pkl"

KEY_3D_AP11_MOD_STRICT = "pred_instances_3d/KITTI/Car_3D_AP11_moderate_strict"
KEY_3D_AP40_MOD_STRICT = "pred_instances_3d/KITTI/Car_3D_AP40_moderate_strict"


def build_system(args):
    if args.device.startswith('npu'):
        import torch_npu  # noqa: F401

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
    return sys_


def collect_pre_nms(sys_, batch):
    """Return (boxes_np (M,7), scores_np (M,)) exactly as the postprocessor's input."""
    out = test_step(sys_, batch)
    pred = out.get('pred', {})
    if not pred or 'cls_score' not in pred:
        # fallback: use final_results if pre-NMS unavailable
        fr = out['final_results']
        if fr:
            r = fr[0]
            return r['boxes_3d'].detach().cpu().float().numpy(), \
                r['scores_3d'].detach().cpu().float().numpy()
        return np.zeros((0, 7), np.float32), np.zeros((0,), np.float32)

    cls_score = pred['cls_score']
    reg_offset = pred['reg_offset']
    flat_rois = pred['flat_rois']
    flat_roi_batch_indices = pred['flat_roi_batch_indices']
    roi_scores = pred.get('flat_roi_scores', None)

    if flat_rois is None or flat_rois.numel() == 0:
        return np.zeros((0, 7), np.float32), np.zeros((0,), np.float32)

    decoded = _decode_boxes(flat_rois, reg_offset)

    # replicate postprocessor score fusion
    if cls_score is not None and cls_score.numel() == decoded.shape[0]:
        scores = torch.sigmoid(cls_score.view(-1))
        if sys_.postprocessor.fuse_roi_score and roi_scores is not None and roi_scores.numel() == scores.numel():
            scores = torch.sqrt(scores.clamp_min(0.0) * roi_scores.to(scores.device).clamp_min(0.0))
    else:
        scores = torch.ones(decoded.shape[0], dtype=decoded.dtype)

    # only batch 0 frames in eval; keep all ROIs regardless of batch index
    if flat_roi_batch_indices is not None and flat_roi_batch_indices.numel() == decoded.shape[0]:
        mask = (flat_roi_batch_indices == 0)
        decoded = decoded[mask]
        scores = scores[mask]

    return decoded.detach().cpu().float().numpy(), scores.detach().cpu().float().numpy()


def cmd_dump(args):
    infos = load(str(V2_PKL))
    data_list = infos['data_list']
    n_total = len(data_list)
    step = getattr(args, 'every', 1) or 1
    n_sel = n_total if args.max_frames <= 0 else min(args.max_frames, n_total)
    positions = list(range(0, n_sel, step))
    print(f"[dump] device={args.device} frames={len(positions)}/{n_total} "
          f"(every={step}) ckpt={args.fusion_ckpt}")

    sys_ = build_system(args)
    image_dir = DATA_ROOT / "training" / "image_2"
    velodyne_dir = DATA_ROOT / "training" / "velodyne"
    calib_dir = DATA_ROOT / "training" / "calib"

    boxes_list = []
    scores_list = []
    idx_list = []
    t0 = time.time()
    for j, pos in enumerate(positions):
        info = data_list[pos]
        sid = os.path.splitext(info['lidar_points']['lidar_path'])[0]
        img_path = str(image_dir / (sid + ".png"))
        lidar_path = str(velodyne_dir / (sid + ".bin"))
        calib_path = str(calib_dir / (sid + ".txt"))
        calib = T.read_kitti_calib(calib_path)
        batch = T.build_eval_batch(img_path, lidar_path, calib, args.device)
        boxes, scores = collect_pre_nms(sys_, batch)
        boxes_list.append(boxes.astype(np.float32))
        scores_list.append(scores.astype(np.float32))
        idx_list.append(pos)
        if (j + 1) % 200 == 0 or j == 0 or j == len(positions) - 1:
            el = time.time() - t0
            print(f"[dump] {j+1}/{len(positions)} (pos={pos}) el={el:.0f}s", flush=True)

    dump_path = T.normalize_path(args.dump)
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    n_fr = len(boxes_list)
    box_obj = np.empty(n_fr, dtype=object)
    score_obj = np.empty(n_fr, dtype=object)
    for i in range(n_fr):
        box_obj[i] = np.ascontiguousarray(boxes_list[i], dtype=np.float32)
        score_obj[i] = np.ascontiguousarray(scores_list[i], dtype=np.float32)
    np.savez_compressed(dump_path, idx=np.asarray(idx_list), boxes=box_obj, scores=score_obj)
    print(f"[dump] saved {n_fr} frames -> {dump_path}")


def replicate_postprocessor(boxes_np, scores_np, score_thr, nms_thr, pre_nms_topk, post_nms_topk):
    """Offline replication of HMFIPostProcessor.forward for one frame (batch 0)."""
    if boxes_np.shape[0] == 0:
        return boxes_np, scores_np

    boxes = torch.from_numpy(boxes_np).float()
    scores = torch.from_numpy(scores_np).float()

    mask = scores >= score_thr
    boxes = boxes[mask]
    scores = scores[mask]
    if boxes.numel() == 0:
        return boxes.numpy().reshape(0, 7), scores.numpy()

    if pre_nms_topk > 0 and scores.numel() > pre_nms_topk:
        keep_topk = torch.topk(scores, k=pre_nms_topk).indices
        boxes = boxes[keep_topk]
        scores = scores[keep_topk]

    if nms_thr is not None and boxes.shape[0] > 0:
        bev = _xyzhwlr_to_bev_xyxy(boxes)
        keep = torch.ops.torchvision.nms(bev, scores, nms_thr) if hasattr(torch.ops, 'torchvision') else _nms_fallback(bev, scores, nms_thr)
        if post_nms_topk > 0:
            keep = keep[:post_nms_topk]
        boxes = boxes[keep]
        scores = scores[keep]

    return boxes.numpy().reshape(-1, 7), scores.numpy()


def _nms_fallback(bev_boxes, scores, iou_threshold):
    """Sort by score desc, greedy suppression (IoU in xyxy)."""
    from model.hmfi.initialization.build_hmfi_system import _box_iou_2d
    order = torch.argsort(scores, descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        ious = _box_iou_2d(bev_boxes[i:i+1], bev_boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long)


def cmd_sweep(args):
    dump_path = T.normalize_path(args.dump)
    if not os.path.exists(dump_path):
        raise FileNotFoundError(dump_path)
    print(f"[sweep] load dump {dump_path}")
    data = np.load(dump_path, allow_pickle=True)
    boxes_list = list(data['boxes'])
    scores_list = list(data['scores'])
    idx_list = list(data['idx'])

    def _fr_boxes(x):
        a = np.asarray(x)
        if a.dtype.kind == 'O':
            a = a.astype(np.float32)
        return a.reshape(-1, 7)

    def _fr_scores(x):
        a = np.asarray(x)
        if a.dtype.kind == 'O':
            a = a.astype(np.float32)
        return a.reshape(-1)
    n_run = len(boxes_list)
    print(f"[sweep] frames={n_run}")

    score_values = [float(x) for x in args.score_thr_sweep.split(",") if x.strip()]
    nms_values = [float(x) for x in args.nms_thr_sweep.split(",") if x.strip()]
    print(f"[sweep] score_thr={score_values} nms_thr={nms_values}")

    results = {}
    pre = int(args.pre_nms_topk)
    post = int(args.post_nms_topk)

    for nms_thr in nms_values:
        for score_thr in score_values:
            print(f"[sweep] score_thr={score_thr} nms_thr={nms_thr} ...", flush=True)
            total_dets = 0
            metric = KittiMetric(
                ann_file=str(V2_PKL),
                pcd_limit_range=[0, -40, -3, 70.4, 40, 0.0],
                prefix='Kitti metric',
            )
            metric.dataset_meta = {'classes': ['Car']}

            for f in range(n_run):
                boxes, scores = replicate_postprocessor(
                    _fr_boxes(boxes_list[f]),
                    _fr_scores(scores_list[f]),
                    score_thr, nms_thr, pre, post)
                n = boxes.shape[0]
                total_dets += n
                if n:
                    boxes3d = LiDARInstance3DBoxes(
                        torch.from_numpy(boxes).float().contiguous(), box_dim=7, origin=(0.5, 0.5, 0.5))
                    labels = torch.zeros(n, dtype=torch.long)
                    scores_t = torch.from_numpy(scores).float().reshape(-1)
                else:
                    boxes3d = LiDARInstance3DBoxes(
                        torch.zeros((0, 7), dtype=torch.float32), box_dim=7, origin=(0.5, 0.5, 0.5))
                    scores_t = torch.zeros((0,), dtype=torch.float32)
                    labels = torch.zeros((0,), dtype=torch.long)
                sample = {
                    'pred_instances_3d': {
                        'bboxes_3d': boxes3d,
                        'scores_3d': scores_t,
                        'labels_3d': labels,
                    },
                    'pred_instances': {},
                    'sample_idx': int(idx_list[f]),
                }
                metric.process(None, [sample])

            t_eval = time.time()
            metric_dict = metric.compute_metrics(metric.results)
            ap11 = float(metric_dict.get(KEY_3D_AP11_MOD_STRICT, -1.0))
            ap40 = float(metric_dict.get(KEY_3D_AP40_MOD_STRICT, -1.0))
            easy11 = float(metric_dict.get("pred_instances_3d/KITTI/Car_3D_AP11_easy_strict", -1.0))
            hard11 = float(metric_dict.get("pred_instances_3d/KITTI/Car_3D_AP11_hard_strict", -1.0))
            easy40 = float(metric_dict.get("pred_instances_3d/KITTI/Car_3D_AP40_easy_strict", -1.0))
            hard40 = float(metric_dict.get("pred_instances_3d/KITTI/Car_3D_AP40_hard_strict", -1.0))
            bev_mod40 = float(metric_dict.get("pred_instances_3d/KITTI/Car_BEV_AP40_moderate_strict", -1.0))
            bev_mod11 = float(metric_dict.get("pred_instances_3d/KITTI/Car_BEV_AP11_moderate_strict", -1.0))
            tag = f"s{score_thr:g}_n{nms_thr:g}"
            results[tag] = {
                "score_thr": score_thr, "nms_thr": nms_thr,
                "total_dets": int(total_dets),
                "AP11_mod": round(ap11, 4), "AP40_mod": round(ap40, 4),
                "AP11_easy": round(easy11, 4), "AP11_hard": round(hard11, 4),
                "AP40_easy": round(easy40, 4), "AP40_hard": round(hard40, 4),
                "BEV_AP11_mod": round(bev_mod11, 4), "BEV_AP40_mod": round(bev_mod40, 4),
            }
            print(f"[sweep] {tag} -> AP11_mod={ap11:.4f} AP40_mod={ap40:.4f} "
                  f"(kitti_eval {time.time()-t_eval:.0f}s)", flush=True)

    if args.out:
        out_path = T.normalize_path(args.out)
    else:
        out_path = str(AD_ROOT / "results" / "hmfi" / "sweep_result.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[sweep] done -> {out_path}")

    # top by AP11 moderate
    ranked = sorted(results.values(), key=lambda x: x["AP11_mod"], reverse=True)
    print("\n==== TOP 5 by AP11 moderate (strict) ====")
    for r in ranked[:5]:
        print(f"  score_thr={r['score_thr']:g} nms_thr={r['nms_thr']:g} "
              f"AP11_mod={r['AP11_mod']:.4f} AP40_mod={r['AP40_mod']:.4f}")


def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return
    cmd = argv[0]

    if cmd == 'dump':
        ap = T.build_argparser()
        ap.add_argument('--dump', type=str, default=str(AD_ROOT / "results" / "hmfi" / "sweep_dump.npz"))
        ap.add_argument('--every', type=int, default=1, help="stride over val pkl (subset dump, e.g. 4)")
        ap.set_defaults(device='npu:0')
        args = ap.parse_args(argv[1:])
        cmd_dump(args)
    elif cmd == 'sweep':
        p = argparse.ArgumentParser(description="offline threshold sweep on dumped pre-NMS preds")
        p.add_argument('--dump', type=str, default=str(AD_ROOT / "results" / "hmfi" / "sweep_dump.npz"))
        p.add_argument('--score_thr_sweep', type=str, default="0.02,0.05,0.1,0.2,0.3")
        p.add_argument('--nms_thr_sweep', type=str, default="0.01,0.05,0.1,0.2")
        p.add_argument('--pre_nms_topk', type=int, default=4096)
        p.add_argument('--post_nms_topk', type=int, default=500)
        p.add_argument('--out', type=str, default=str(AD_ROOT / "results" / "hmfi" / "sweep_result.json"))
        args = p.parse_args(argv[1:])
        cmd_sweep(args)
    else:
        print(f"unknown cmd: {cmd}\n{__doc__}")


if __name__ == "__main__":
    main()
