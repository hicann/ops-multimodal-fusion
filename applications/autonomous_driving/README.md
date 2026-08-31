# 3D 目标检测（相机-点云多模态融合）

Ascend NPU 3D 目标检测应用，支持 PointPillars / CLOCS 多模型接入。

## 1. 目录结构

```
autonomous_driving/
├── common/                                # 跨模型共享
│   ├── kitti_eval/                        # KITTI 评估工具
│   │   ├── rotate_iou_cpu.py              #   旋转 IoU CPU 实现（shapely）
│   │   ├── generate_val_pkl.py            #   生成 val 标注 pkl
│   │   ├── kitti_converter.py             #   官方数据转换（OpenMMLab）
│   │   ├── kitti_data_utils.py            #   官方数据工具（OpenMMLab）
│   │   └── update_infos_to_v2.py          #   官方 v2 格式转换（OpenMMLab）
│   ├── ops/                                # 融合算子（operator_registry + 9 个算子）
│   ├── interfaces/                         # det2d/det3d 输出接口
│   ├── kitti_writer.py                     # KITTI 预测写盘
│   ├── pcdet_stub/                         # pcdet 依赖最小 stub
│   └── patches/                           # 第三方库 NPU 适配（路径对应 site-packages）
│       ├── mmdet3d/
│       │   ├── apis/inference.py
│       │   └── models/data_preprocessors/voxelize.py
│       └── mmcv/
│           ├── ops/nms.py
│           └── ops/roi_align.py
└── model/
    ├── pointpillars/                      # PointPillars 模型
    │   ├── backbone/
    │   │   └── point_cloud_feature_interface.py   # 点云特征提取接口
    │   ├── eval_official/
    │   │   └── run_official_eval.py               # 官方 KittiMetric 评估
    │   └── run_pointpillars_forward.py            # 单帧推理入口
    └── clocs/                             # CLOCS 模型（2D+3D 后融合）
        ├── clocs_framework.py
        ├── initialization/
        │   └── build_clocs_system.py
        ├── module/
        │   ├── fusion_mlp.py              # FusionMLP 融合网络块
        │   ├── fusion_score_module.py
        │   ├── match_pair_module.py
        │   └── aggregate_post_module.py
        └── test_script.py                 # 端到端 eval/train 入口
```

## 2. 环境配置

### 2.1 环境清单（实测）

| 组件 | 版本 |
|---|---|
| Python | 3.12.9 |
| torch | 2.7.1+cpu |
| torch_npu | 2.7.1.post8 |
| CANN | 9.0.0.beta2 |
| 芯片 | Ascend950PR ×1 |
| mmcv | 2.2.0 * |
| mmdet | 3.3.0 |
| mmdet3d | 1.4.0 |
| mmengine | 0.10.7 |
| setuptools | 69.5.1 |
| numpy | 1.26.4 |

> \* mmcv 无 NPU 预编译 wheel，需源码编译（`FORCE_NPU=1`，详见 2.2 安装步骤）；其余包均为 pip 直接安装。

> site-packages 指 Python 第三方库安装目录，下文以 `<site-packages>` 指代。
> 用 `python -c "import site; print(site.getsitepackages())"` 查询当前环境的实际路径。

### 2.2 安装步骤

```bash
# 1. mmcv 2.2.0 源码编译（NPU 模式，约 3 分钟）
#    需先把 mmcv 的 6 个 NPU C++ 源文件中 c10::SmallVector<int64_t, SIZE> 替换为 <int64_t, 5>
FORCE_NPU=1 MAX_JOBS=16 pip3 install mmcv-2.2.0 --no-build-isolation

# 2. mmdet / mmdet3d / mmengine
pip3 install mmdet==3.3.0 mmdet3d==1.4.0 --no-deps
pip3 install mmengine==0.10.7
#    放宽 mmdet/mmdet3d 的 mmcv_maximum_version：'2.2.0' → '2.3.0'

# 3. 运行时依赖
pip3 install pycocotools shapely terminaltables pyquaternion tensorboard numba \
    scikit-image scikit-learn trimesh plyfile networkx descartes nuscenes-devkit
```

> 说明：mmcv 源码编译依赖 setuptools 的 `pkg_resources`，需固定 ≤ 69.5.1（`pip3 install setuptools==69.5.1`，本机已固定）；open3d 在 aarch64 无 wheel，PointPillars 不依赖，跳过。

### 2.3 应用第三方库 patch

`common/patches/` 下的目录结构与 site-packages 中的相对路径一一对应，覆盖到 site-packages 即可：

```bash
cp common/patches/mmdet3d/apis/inference.py                     <site-packages>/mmdet3d/apis/inference.py
cp common/patches/mmdet3d/models/data_preprocessors/voxelize.py <site-packages>/mmdet3d/models/data_preprocessors/voxelize.py
cp common/patches/mmcv/ops/nms.py                               <site-packages>/mmcv/ops/nms.py
cp common/patches/mmcv/ops/roi_align.py                         <site-packages>/mmcv/ops/roi_align.py
```

## 3. 通用约定

- 数据放 `data/kitti/`，权重放 `checkpoints/`，结果放 `results/`
- 脚本用 `AD_ROOT = Path(__file__).resolve().parents[N]` 自动定位应用根目录

## 4. 模型

### 4.1 PointPillars

KITTI 3D Car 检测，使用 mmdet3d 官方 PointPillars（config + 官方权重），**模型网络结构零修改**。

#### 4.1.1 改动内容

| 文件 | 改动 |
|---|---|
| `common/patches/mmdet3d/apis/inference.py` | `init_model` 增加 npu 设备支持 |
| `common/patches/mmdet3d/models/data_preprocessors/voxelize.py` | Voxelization NPU→CPU 回退 |
| `common/patches/mmcv/ops/nms.py` | `nms_rotated` NPU→CPU 回退 |
| `model/pointpillars/eval_official/run_official_eval.py` | 新增：官方 KittiMetric 端到端评估入口 |

#### 4.1.2 推理验证

```bash
# 0. 下载官方权重
mkdir -p checkpoints/pointpillars
wget -O checkpoints/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth \
  https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth

# 1. 单帧推理（--bin 为 KITTI 点云文件路径，如 data/kitti/training/velodyne/000000.bin）
#    NPU
python model/pointpillars/run_pointpillars_forward.py --bin data/kitti/training/velodyne/000000.bin --device npu:0
#    CPU
python model/pointpillars/run_pointpillars_forward.py --bin data/kitti/training/velodyne/000000.bin --device cpu

# 2. 生成 val 标注 pkl（官方评估前置：KittiMetric 需要 v2 格式标注）
#    generate_val_pkl.py 从原始数据生成 v1 标注 → update_infos_to_v2.py 转成 v2 格式
python common/kitti_eval/generate_val_pkl.py
python common/kitti_eval/update_infos_to_v2.py --dataset kitti \
  --pkl-path data/kitti/kitti_infos_val.pkl --out-dir data/kitti/
mv data/kitti/kitti_infos_val.pkl data/kitti/kitti_infos_val_v2.pkl

# 3. 官方评估（3769 帧 val，官方 KittiMetric）
#    NPU
python model/pointpillars/eval_official/run_official_eval.py --device npu:0 --checkpoint <权重路径>
#    CPU
python model/pointpillars/eval_official/run_official_eval.py --device cpu --checkpoint <权重路径>
```

#### 4.1.3 官方权重结果

官方 CUDA 训练的 PointPillars 权重，KITTI Car 全量 3769 帧 val，官方 mmdet3d 1.4.0 KittiMetric 口径。

单帧推理（`000000.bin`）：NPU 与 CPU 均检出 8 个 Car（score 0.113~0.643），结果一致。

全量评估（strict IoU≥0.7）：

| 指标 | 设备 | easy | moderate | hard |
|---|---|---|---|---|
| 3D AP11 | NPU | 84.25 | 80.28 | 75.12 |
| 3D AP11 | CPU | 84.25 | 80.28 | 75.13 |
| 3D AP40 | NPU | 88.16 | 80.97 | 76.51 |
| 3D AP40 | CPU | 88.16 | 80.94 | 76.51 |
| BEV AP11 | NPU | 85.63 | 84.86 | 82.24 |
| BEV AP11 | CPU | 85.64 | 84.85 | 82.23 |
| BEV AP40 | NPU | 90.07 | 85.34 | 82.59 |
| BEV AP40 | CPU | 90.07 | 85.34 | 82.58 |

CPU 与 NPU 结果一致（3D AP11/AP40 moderate 差异 <0.1pp）。全量推理耗时：NPU 90s，CPU 934s。

对比 mmdet3d 官方 benchmark（3D AP40 moderate 77.6%）：本应用 NPU 实测 80.97%，达到并略超官方基准（同为 R40 口径）。

### 4.2 CLOCS

KITTI 3D Car 检测，2D（Faster R-CNN）+ 3D（PointPillars）后融合。2D/3D 检测器各用官方预训练权重并冻结，FusionMLP 对每个 2D-3D pair 打分、沿 2D 维 max 聚合后重排 3D 框，再经 3D NMS 输出。

> **算法来源**：CLOCS 融合算法参考论文 *CLOCs: Camera-LiDAR Object Candidates Fusion for 3D Object Detection*（Pang Su 等，CVPR 2021 Workshop）及官方实现 [pangsu0613/CLOCs](https://github.com/pangsu0613/CLOCs)（MIT License）。本应用的 `common/ops/`、`model/clocs/` 为按论文思想在 Ascend NPU 上的自研实现，不包含官方代码逐字拷贝；2D/3D 检测器使用 mmdet/mmdet3d 官方预训练权重。

#### 4.2.1 改动内容

| 文件 | 改动 |
|---|---|
| `model/clocs/test_script.py` | 修复硬编码 `calib_path="/root/kitti/..."`；`rotate_iou` 用 CPU 版替换（NPU 无 CUDA 驱动） |
| `common/patches/mmcv/ops/roi_align.py` | RoIAlign NPU→CPU 回退 |
| `model/clocs/initialization/build_clocs_system.py` | 修复 det3d 输出 w/l 互换：mmdet3d `[l,w,h]` → 接口约定 `[w,l,h]`，修复 3D 框投影/配对/iou3d 失真 |
| `common/kitti_writer.py` | 补上 docstring 声称却缺失的 w/l 交换，与 adapter 修复配套（保证写盘框尺寸正确） |
| `common/ops/pair_feature_encoder.py` | pair 特征顺序改 `[iou,s3,s2,dist]`、distance 改原 CLOCS 的 `norm(x,y)/82`（对齐原始 CLOCS 特征格式） |
| `model/clocs/eval_clocs_official.py` | 新增：官方 `KittiMetric` 评估脚本（与 4.1 PointPillars 同口径、同一 v2 pkl） |

#### 4.2.2 推理验证

权重：2D Faster R-CNN（167MB，需下载）与 3D PointPillars（复用 4.1 官方权重）。CLOCS 的 `fusion_trained.pth` 未随项目提供，脚本缺省用随机初始化 FusionMLP。

```bash
# 2D 检测器权重下载
mkdir -p checkpoints/clocs
wget -O checkpoints/clocs/faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth \
  https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco/faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth
```

```bash
# 生成 val split 文件（6 位零填充帧号，test_script 按 image_2 文件名匹配）
python - <<'EOF'
ids = [int(x.strip()) for x in open('data/kitti/ImageSets/val.txt') if x.strip()]
open('data/kitti/ImageSets/val_zero_padded.txt','w').writelines(f'{i:06d}\n' for i in ids)
EOF

# 官方 KittiMetric 评估（与 PointPillars 4.1 同口径；--max-frames>0 且<全量时为前向冒烟）
python model/clocs/eval_clocs_official.py --device npu:0            # 全量 3769 帧 NPU
python model/clocs/eval_clocs_official.py --device cpu              # 全量 3769 帧 CPU
python model/clocs/eval_clocs_official.py --device npu:0 --max-frames 5   # 冒烟

# （可选）test_script 生成 KITTI 预测文件（旧 kitti_eval 口径，仅用于对拍/调参）
python model/clocs/test_script.py --device npu:0 --split_file data/kitti/ImageSets/val_zero_padded.txt
```

#### 4.2.3 结果对比（官方 KittiMetric 口径）

官方 mmdet3d `KittiMetric`（与 4.1 PointPillars 同一把尺子、同一 `data/kitti/kitti_infos_val_v2.pkl`），Car，3769 帧 val，strict IoU≥0.7。当前 FusionMLP 为随机初始化，融合≈单调重排，结果反映「后融合管线 + 随机 FusionMLP」基线。

| 指标 | 设备 | easy | moderate | hard |
|---|---|---|---|---|
| 3D AP11 | NPU | 86.49 | 83.08 | 77.28 |
| 3D AP11 | CPU | 86.58 | 83.11 | 77.30 |
| 3D AP40 | NPU | 91.44 | 83.51 | 79.06 |
| 3D AP40 | CPU | 91.49 | 83.51 | 79.07 |
| BEV AP11 | NPU | 87.87 | 87.49 | 85.31 |
| BEV AP11 | CPU | 87.93 | 87.52 | 85.35 |
| BEV AP40 | NPU | 93.32 | 88.18 | 85.44 |
| BEV AP40 | CPU | 93.35 | 88.23 | 85.50 |
| 2D AP11 | NPU | 94.57 | 87.75 | 86.28 |
| 2D AP11 | CPU | 94.31 | 87.81 | 86.31 |

CPU 与 NPU 一致（全部指标差异 <0.1pp）。全量 3769 帧耗时：CPU 约 68 min（推理 ~66 min + 评估 ~2 min）。

对比：纯 PointPillars 底座（4.1）3D AP11 moderate = 80.28%、3D AP40 = 80.97%。CLOCS 官方口径 83.08%（AP11）/ 83.51%（AP40），略高于底座（+2~3pp，来自投影前向过滤与自研 `nms3d` 的后处理差异）。此前 README 的 69.21% 是旧版 `kitti_eval`（`test_script.py` 路径）实现差异低估 ~20pp 造成的假象，并非真实精度。

#### 4.2.4 已知问题与修复记录

| # | 状态 | 问题 | 位置 |
|---|---|---|---|
| 1 | 已修 | w/l 互换：mmdet3d `bboxes_3d.tensor` 为 `[l,w,h]`，接口约定 `[w,l,h]`，`MMDet3DAdapter.forward` 未换位 | `model/clocs/initialization/build_clocs_system.py`（+ `common/kitti_writer.py` 配套） |
| 2 | 已修 | distance 归一化与特征顺序与原 CLOCS 不一致（`[iou,s2,s3]` vs 原 `[iou,s3,s2]`；`/80`+clamp vs 原 `norm(x,y)/82`） | `common/ops/pair_feature_encoder.py` |
| 3 | 遗留 | FusionMLP 权重未随项目提供，缺省用随机初始化 → 融合未激活，当前结果 ≈ 纯 3D 底座 | 权重文件 |
| 4 | 遗留（结构性） | 候选集规模：post-NMS ~5-10 框（~1-13 pair）vs 论文 pre-NMS 70400 anchor（~90 万 pair） | 结构性 |
| 5 | 已知限制 | `update_infos_to_v2.py` 生成的 val 标注 classes 仅含主要类别（Pedestrian/Cyclist/Car/Van/Truck/Person_sitting/Tram/Misc），'Van'/'DontCare' 未纳入完整 label 处理 | `common/kitti_eval/update_infos_to_v2.py` |

> 关于源项目 README 的 "Ours 3D AP 85.39%"：该数字与原 CLOCs 作者报告值一致，属引用/声称值；对应训练好的 FusionMLP 权重未随项目交付，无法用本代码/权重复现，也不与本表口径直接对比。若需复现论文融合精度，需补 pre-NMS 候选 + 训练好 FusionMLP 权重。
