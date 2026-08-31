# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
PointPillars 单帧 CPU 前向 smoke test（迁移阶段一的最小验证）

目的（新人必读）：
    这是一个"最小冒烟测试"脚本 —— 它不验证精度、不验证与 GPU 结果是否一致，
    只验证一件事：整条前向链路在本地 CPU 上能否从头跑到尾：
        点云 -> 体素化 -> PillarEncoder -> Scatter -> SECFPN -> 检测头 -> 检测框

重要坑（已定位，新人务必理解）：
    mmdet3d 的 PillarFeatureNet 在 legacy 分支会"原地改写"输入 voxels 的 xyz
    （源码里 `f_center = features[:, :, :3]` 是输入张量的视图，随后就地相减）。
    后果：对同一份 voxels 先调 extract_feat、再调 predict，predict 内部第二次
    extract_feat 用的是被污染的数据 -> 检测框全丢（0 框）。
    本脚本规避方式：特征用接口拿（第一次调用，正确）；检测框用"重新体素化的
    新 batch"再 predict（保证 predict 内部也是第一次调用，正确）。
    这个坑在 GPU/服务器上同样存在，做迁移对齐时记住：一次前向只调一次 extract_feat。

用法：
    conda activate fusionrepo
    python run_pointpillars_forward.py [--bin path/to/velodyne.bin] [--device cpu]
"""

import argparse          # 解析命令行参数（--bin / --device）
from pathlib import Path  # 跨平台路径处理

import numpy as np       # 读点云 .bin 文件（KITTI 点云是 float32 二进制）
import torch             # PyTorch 主库

import mmdet3d                       # 3D 检测库（提供 PointPillars 模型）
from mmdet3d.apis import init_model  # 根据 config + 权重初始化一个检测器
from mmdet3d.structures import (     # 3D 数据结构
    Box3DMode,            # 3D 框的坐标模式枚举（LIDAR 等）
    Det3DDataSample,      # 一个 3D 检测样本的容器（存放 metainfo/GT/预测）
    LiDARInstance3DBoxes,  # 激光雷达坐标系下的 3D 框类型
)
from mmdet3d.utils import register_all_modules  # 注册 mmdet3d 里所有模型/算子类

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 让 backbone 包可导入

from backbone.point_cloud_feature_interface import PointCloudFeatureInterface  # 本仓库的点云侧统一接口  # noqa: E402

# AD_ROOT = autonomous_driving 应用根目录（数据/权重统一约定在其下）
# 用 __file__ 定位，任何目录下运行都能正确找到权重和数据
AD_ROOT = Path(__file__).resolve().parents[2]
ROOT = AD_ROOT


def main():
    # ---------- 0. 解析命令行参数 ----------
    parser = argparse.ArgumentParser()
    # --bin：可选，指定一帧点云文件；不给就用 data/kitti_small 里已备好的 000000.bin
    parser.add_argument("--bin", default="", help="单帧点云 .bin 路径，缺省用 data/kitti_small/training/velodyne/000000.bin")
    # --device：默认 cpu；就算你传 cuda:0，本机没 GPU 也会在下面自动回退成 cpu
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device
    if device.startswith("npu"):
        import torch_npu  # noqa: F401
    # 兜底：如果写了 cuda 但本机没有 CUDA，自动切成 cpu，避免报错。
    # 工程经验：任何"自动降级"都不能静默发生，必须打日志/告警，
    # 否则你会在完全不知情的情况下用错了设备，还查不到原因。
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] 你指定了 device={args.device}，但本机没有可用 CUDA，"
              f"已自动降级为 cpu（如需 GPU 请检查驱动 / torch 是否 CUDA 版）")
        device = "cpu"

    # ---------- 1. 加载模型 ----------
    # register_all_modules()：mmdet3d 用"注册机制"管理模型类。
    # config 文件里写的是字符串名字（如 type='PillarFeatureNet'），
    # 必须先执行注册，init_model 才能按名字找到对应类来构建。
    register_all_modules()

    # config 文件：描述模型的"骨架和参数"（用了哪些层、体素大小、检测头怎么配）。
    # 它不在本仓库里，而是随 mmdet3d 包安装（site-packages/mmdet3d/.mim/configs/...）。
    # 注：PointCloudFeatureInterface 里就是通过这个 config 推断 voxel_size / point_cloud_range 的。
    cfg_path = Path(mmdet3d.__file__).parent / ".mim" / "configs" / "pointpillars" / "pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py"
    # 权重文件：预训练好的模型参数（.pth）。模型"聪明不聪明"全看这些数值。
    ckpt_path = ROOT / "checkpoints" / "pointpillars" / "hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth"

    assert cfg_path.exists(), f"config not found: {cfg_path}"
    assert ckpt_path.exists(), f"checkpoint not found: {ckpt_path}"

    print("[1/5] 初始化 mmdet3d PointPillars 检测器 (device={})".format(device))
    # init_model = 读 config 搭出网络结构 + 把权重灌进去。这一步做完，模型就能用了。
    detector = init_model(str(cfg_path), str(ckpt_path), device=device)
    # eval()：切到"推理模式"（关闭 Dropout/BN 的统计更新），推理前必须调用。
    detector.eval()

    # ---------- 2. 包装本仓库接口 ----------
    print("[2/5] 包装 PointCloudFeatureInterface")
    # 本仓库所有融合模型的"点云侧"都走这个接口，它是你迁移要改造的目标代码。
    # 作用：把 detector 封装成"喂点云 -> 吐特征 + 检测框"的统一入口。
    backbone = PointCloudFeatureInterface(detector).to(device)

    # ---------- 3. 读一帧点云 ----------
    # KITTI 点云文件格式：连续 float32，(x, y, z, intensity)，每行 4 个数。
    # 所以先按 float32 整体读进来，再 reshape 成 (N, 4)，N = 这一帧的点数。
    bin_path = Path(args.bin) if args.bin else ROOT / "data" / "kitti_small" / "training" / "velodyne" / "000000.bin"
    assert bin_path.exists(), f"point cloud not found: {bin_path}"
    pts = torch.from_numpy(np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)).to(device)
    print(f"      points: {tuple(pts.shape)}")  # 例：(115384, 4) = 11.5 万个点，每点 4 维
    # 打印点云在三维空间的范围，方便你确认数据是合理的（车前方 0~70m、左右 ±40m 左右）
    print(f"      xyz range min={pts[:, :3].min(0).values.tolist()} max={pts[:, :3].max(0).values.tolist()}")

    # ---------- 4. 特征前向 ----------
    print("[3/5] 特征前向（首次调用，正确）：体素化 + PillarEncoder + Scatter + SECFPN")
    # torch.no_grad()：推理时不需要计算梯度，省内存、提速（训练时才要梯度）。
    # 前向链路的 5 步（对照你新人文档里的图）：
    #   ① 体素化：把散乱点云按 0.16x0.16m 格子分成体素(pillar)
    #   ② PillarEncoder：每个 pillar 内的点过一个 MLP，聚合成 pillar 特征
    #   ③ Scatter：把 pillar 特征"摊回"成稠密的 BEV 特征图 (C,H,W)
    #   ④ SECFPN：2D 卷积网络，提取多尺度 BEV 特征
    #   ⑤ 检测头：在 BEV 特征上出类别 + 3D 框
    with torch.no_grad():
        out = backbone(pts)
    # out 是 dict，key 含义见 PointCloudFeatureInterface 的 docstring。
    # pts_feats = (B*H*W, C)：BEV 特征图被展平成点序列，方便后续融合模块使用
    print("      pts_feats (B*H*W, C):", tuple(out["pts_feats"].shape))
    if out["voxels"] is not None:
        # voxels = (M, max_points, 3)：M=体素数，每个体素最多 max_points 个点（这里只留 xyz 3 维）
        print("      voxels (M, max_points, 4):", tuple(out["voxels"].shape))
    if out["voxel_coords"] is not None:
        # voxel_coords = (M, 4)：每个体素在 [batch, z, y, x] 网格上的整数坐标（前 3 列已经是中心化后的值）
        print("      voxel_coords (M, 4) [b,z,y,x]:", tuple(out["voxel_coords"].shape))

    # ---------- 5. 检测框（重新体素化后 predict） ----------
    print("[4/5] 检测框（重新体素化后 predict，避开 voxel 原地改写坑）")
    # mmdet3d 的接口约定：喂给模型的不只是张量，还有"data samples"（样本元信息）。
    # 这里手工构造 1 个空样本，告诉模型"3D 框用 LIDAR 坐标系、用 LiDARInstance3DBoxes 类型"。
    batch_data_samples = []
    for _ in range(1):  # batch_size = 1，只处理一帧
        ds = Det3DDataSample()
        ds.set_metainfo({
            "box_type_3d": LiDARInstance3DBoxes,  # 框的容器类型
            "box_mode_3d": Box3DMode.LIDAR,       # 坐标模式：LIDAR(x前y左z上)
        })
        batch_data_samples.append(ds)

    # 重新跑一次 data_preprocessor（体素化）。为什么？
    # 因为上一步 extract_feat 已经把上一份 voxels 的 xyz 原地改掉了（见文件头"重要坑"），
    # 必须拿一份"新的、干净的" voxels 给 predict，predict 内部的 extract_feat 才是正确的第一次调用。
    processed = detector.data_preprocessor(
        {"inputs": {"points": [pts]}, "data_samples": batch_data_samples}, training=False
    )
    with torch.no_grad():
        # predict = 前向 + 后处理（锚框解码、NMS 去重、按分数阈值筛选）
        preds = detector.predict(processed["inputs"], processed["data_samples"])

    # 解析预测结果：
    # preds[0] 是第 0 帧的 Det3DDataSample；pred_instances_3d 里装着最终检测框
    inst = preds[0].pred_instances_3d
    boxes = inst.bboxes_3d                        # 3D 框对象（带 .tensor 属性）
    boxes_t = boxes.tensor if hasattr(boxes, "tensor") else boxes  # 取出 (K, 7) 的数值张量
    # 分数：每个框的置信度；标签：类别 id（0=Car）
    if hasattr(inst, "scores_3d"):
        scores = inst.scores_3d
    else:
        scores = getattr(inst, "scores", None)
    if hasattr(inst, "labels_3d"):
        labels = inst.labels_3d
    else:
        labels = getattr(inst, "labels", None)

    print(f"      flat_rois: {tuple(boxes_t.shape)}  scores: {tuple(scores.shape)}")
    if boxes_t.numel():
        # 7 维 = [x, y, z, 长 l, 宽 w, 高 h, 朝向角 ry]（LIDAR 坐标系）
        for i in range(boxes_t.shape[0]):
            print(f"      [{i}] score={float(scores[i]):.4f} label={int(labels[i])} box={boxes_t[i].tolist()}")
    else:
        # 工程经验：0 框可能是"真没有目标"，也可能是某处出了静默错误
        # （本仓库的 PointCloudFeatureInterface 会把 predict 的异常吞掉并返回空框），
        # 所以这里显式打一条告警，而不是静默跳过。
        print("[WARN] 未检出框。原因排查：① 检查上一步是否踩了 voxel 原地改写坑；"
              "② config 里 test_cfg.score_thr 可能太高；③ 点云内容本身可能确实没车")

    print("[5/5] 完成")


# Python 入口约定：只有"直接运行本文件"时才执行 main()
# （被别的文件 import 时不会执行，避免副作用）
if __name__ == "__main__":
    main()
