# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Phase 0 验证门：用官方 mmdet3d KittiMetric 端到端评估官方 CUDA checkpoint。
不经过我们自定义的 anno 转换，完全走 KittiMetric.convert_valid_bboxes / convert_annos_to_kitti_annos。

用法:
  cd <autonomous_driving 根目录> && python3 \
    npu_adaptation_workspace/eval_official/run_official_eval.py \
    [--max-frames 0] [--device npu:0] [--checkpoint <path>]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# numba 死锁防护：限制线程，避免 d3_box_overlap_kernel 并行区死锁
import os
os.environ.setdefault('NUMBA_NUM_THREADS', '8')

import torch  # noqa: E402

# rotate_iou CUDA -> CPU stub（在 import kitti_eval 之前）
AD_ROOT = Path(__file__).resolve().parents[3]
_riu = 'mmdet3d.evaluation.functional.kitti_utils.rotate_iou'
if _riu not in sys.modules:
    import types
    _s = types.ModuleType(_riu)
    sys.path.insert(0, str(AD_ROOT / "common" / "kitti_eval"))
    from rotate_iou_cpu import rotate_iou_cpu
    _s.rotate_iou_gpu_eval = rotate_iou_cpu
    sys.modules[_riu] = _s

import mmdet3d  # noqa: E402
from mmdet3d.apis import init_model  # noqa: E402
from mmdet3d.structures import Box3DMode, Det3DDataSample, LiDARInstance3DBoxes  # noqa: E402
from mmdet3d.utils import register_all_modules  # noqa: E402
from mmdet3d.evaluation.metrics.kitti_metric import KittiMetric  # noqa: E402
from mmengine import load  # noqa: E402

ROOT = AD_ROOT
DATA_ROOT = AD_ROOT / "data" / "kitti"
EVAL_DIR = AD_ROOT / "results" / "pointpillars" / "eval_official"
V2_PKL = AD_ROOT / "data" / "kitti" / "kitti_infos_val_v2.pkl"
CKPT = AD_ROOT / "checkpoints" / "pointpillars" / \
    "hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth"
CFG = Path(mmdet3d.__file__).parent / ".mim" / "configs" / "pointpillars" / \
    "pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py"


def log(msg):
    from datetime import datetime
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')
    print(f"[{ts}] {msg}", flush=True)
    (AD_ROOT / "results" / "pointpillars").mkdir(parents=True, exist_ok=True)
    with open(AD_ROOT / "results" / "pointpillars" / "eval_official.log", "a") as f:
        f.write(f"{ts} [eval_official] {msg}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--device', type=str, default='npu:0')
    parser.add_argument('--checkpoint', type=str, default=str(CKPT))
    parser.add_argument('--out', type=str, default=None,
                        help='结果输出前缀 (不含扩展名)。默认官方 official_eval_result.{txt,json}')
    args = parser.parse_args()

    if args.device.startswith('npu'):
        import torch_npu  # noqa

    log(f"Phase0 官方 KittiMetric 端到端评估开始 device={args.device} ckpt={args.checkpoint}")

    register_all_modules()
    log(f"init_model ...")
    ckpt_data = torch.load(args.checkpoint, map_location='cpu')
    if isinstance(ckpt_data, dict) and 'model_state_dict' in ckpt_data:
        # 自训练 checkpoint 格式：嵌套 model_state_dict
        from mmengine.config import Config as MConfig
        import copy
        from mmdet3d.registry import MODELS
        cfg_obj = MConfig.fromfile(str(CFG))
        cfg_obj.data_root = str(DATA_ROOT)
        model_cfg = copy.deepcopy(cfg_obj.model)
        for key in ['backend_args']:
            if isinstance(model_cfg, dict) and key in model_cfg:
                model_cfg.pop(key)
        dp = model_cfg.get('data_preprocessor', {})
        if isinstance(dp, dict) and 'backend_args' in dp:
            dp.pop('backend_args')
        model = MODELS.build(model_cfg)
        model.load_state_dict(ckpt_data['model_state_dict'])
        model.to(args.device)
        model.eval()
    else:
        model = init_model(str(CFG), args.checkpoint, device=args.device)
        model.eval()

    infos = load(str(V2_PKL))
    data_list = infos['data_list']
    n_total = len(data_list)
    if args.max_frames > 0:
        data_list = data_list[:args.max_frames]
        n_total = len(data_list)
    log(f"val 帧数={n_total}")

    metric = KittiMetric(
        ann_file=str(V2_PKL),
        pcd_limit_range=[0, -40, -3, 70.4, 40, 0.0],
        prefix='Kitti metric',
    )
    metric.dataset_meta = {'classes': ['Car']}

    velodyne_dir = DATA_ROOT / "training" / "velodyne"
    t_start = time.time()
    total_pts = 0
    total_dets = 0

    for idx, info in enumerate(data_list):
        lidar_path = velodyne_dir / info['lidar_points']['lidar_path']
        pts_np = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)
        total_pts += pts_np.shape[0]
        pts = torch.from_numpy(pts_np).to(args.device)

        data_sample = Det3DDataSample()
        data_sample.set_metainfo({
            'box_type_3d': LiDARInstance3DBoxes,
            'box_mode_3d': Box3DMode.LIDAR,
            'sample_idx': idx,
        })
        processed = model.data_preprocessor(
            {'inputs': {'points': [pts]}, 'data_samples': [data_sample]},
            training=False)
        with torch.no_grad():
            preds = model.predict(processed['inputs'], processed['data_samples'])
        preds[0].set_metainfo({'sample_idx': idx})
        inst = preds[0].pred_instances_3d
        n_det = 0 if not hasattr(inst, 'bboxes_3d') else len(inst.bboxes_3d)
        total_dets += n_det
        metric.process(None, [p.to_dict() for p in preds])

        if (idx + 1) % 50 == 0 and args.device.startswith('npu'):
            torch.npu.empty_cache()
        if (idx + 1) % 200 == 0 or idx == 0 or idx == n_total - 1:
            el = time.time() - t_start
            eta = (n_total - idx - 1) / max((idx + 1) / el, 1e-6)
            log(f"推理 {idx + 1}/{n_total} dets累计={total_dets} 耗时{el:.0f}s ETA{eta:.0f}s")

    infer_time = time.time() - t_start
    log(f"推理完成 {infer_time:.0f}s 点数={total_pts} 检测={total_dets}")

    log("开始 compute_metrics (kitti_eval) ...")
    t_eval = time.time()
    metric_dict = metric.compute_metrics(metric.results)
    eval_time = time.time() - t_eval
    log(f"kitti_eval 完成 {eval_time:.0f}s")

    # 落盘结果
    if args.out:
        out_txt = Path(args.out + ".txt")
        out_json = Path(args.out + ".json")
    else:
        out_txt = EVAL_DIR / "official_eval_result.txt"
        out_json = EVAL_DIR / "official_eval_summary.json"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, 'w') as f:
        for k, v in metric_dict.items():
            f.write(f"{k}: {v}\n")
    with open(out_json, 'w') as f:
        json.dump({k: float(v) for k, v in metric_dict.items()}, f, indent=2)

    log(f"结果指标:")
    for k, v in metric_dict.items():
        log(f"  {k} = {v:.4f}")
    log(f"结果已存: {out_txt} / {out_json}")
    log("Phase0 官方 KittiMetric 端到端评估结束")


if __name__ == '__main__':
    main()
