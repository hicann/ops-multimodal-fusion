---
name: repo-build-guide
description: ops-multimodal-fusion 仓 wheel 构建与验证命令参考。编译算子、运行测试的常用命令。触发：需要编译算子、跑测试、查看 build.sh 用法时。
---

# ops-multimodal-fusion 构建命令

本仓是基于 AscendC 的 PyTorch 自定义算子库，编译产物为单个 wheel 包 `ops_multimodal_fusion`（version 1.0.0）。

## 环境要求

- CANN 开发套件（含 Ascend C 编译器）
- Python >= 3.8、GCC >= 7.3.0、CMake >= 3.16.0（可用 `bash install_deps.sh` 安装）
- torch、torch_npu：须预装（在 `setup.py` 的 `install_requires` 中，`requirements.txt` 不含）

## 编译命令

### 编译全量

```bash
bash build.sh
```

不带参数即编译全部算子，默认 SoC 为 `ascend950`。

### 指定芯片编译

```bash
bash build.sh --soc={soc_version}
```

### 编译指定算子

```bash
bash build.sh --ops=add,rms_norm
```

### 按改动清单编译

```bash
bash build.sh --filelist={path}
```

参数说明：
- `--soc`：芯片版本号（小写），取值 `ascend910b`、`ascend910_93`、`ascend950`，默认 `ascend950`（也可用 `SOC` 环境变量设置）。
- `--ops`：算子目录名（snake_case），逗号分隔多算子，如 `--ops=add,rms_norm`。
- `--filelist`：读取一个「每行一个改动路径」的清单文件，自动识别受影响的算子与需要构建的 SoC（`arch35/` 改动 → `ascend950`；`arch22/` 改动 → `ascend910b`；算子公共文件 → 该算子支持的全部 SoC）。
- `--ops` 与 `--filelist` 互斥，不能同时使用。

> `build.sh` 仅支持 `--soc` / `--ops` / `--filelist` / `-h|--help`。传入其它参数会报错并退出。

## 常用芯片版本

| 芯片 | --soc 参数 | NPU_ARCH | 架构目录 |
|------|-----------|----------|----------|
| Ascend910B | ascend910b | dav-2201 | arch22 |
| Ascend910_93 | ascend910_93 | dav-2201 | arch22 |
| Ascend950 | ascend950 | dav-3510 | arch35 |

> 绝大多数算子只有 arch35 实现（对标 CANN 9.0.0），因此默认 SoC 为 `ascend950`。

## 输出目录与安装

| 产物 | 路径 | 说明 |
|------|------|------|
| wheel 包 | `dist/ops_multimodal_fusion-1.0.0+{soc}-cp{py}-...-linux_*.whl` | 编译产物 |
| 动态库 | wheel 内的 `libops_multimodal_fusion.so` | 打包进 wheel 的算子库 |

安装（**在源码目录外执行**，避免 import 到源码目录而非已安装包）：

```bash
cd /tmp && pip install {path_to_whl} --force-reinstall --no-deps
```

卸载：

```bash
pip uninstall ops_multimodal_fusion
```

## 测试运行

本仓测试与编译解耦，安装 wheel 后用独立 pytest 运行（无需通过 `build.sh`）：

```bash
# 全量测试
pytest tests/ -v

# 单算子测试
pytest tests/{op}/ -v
```

用例内部依次 `import torch` / `import torch_npu` / `import ops_multimodal_fusion`；若算子未注册到 `torch.ops.ops_multimodal_fusion`，模块级 `pytest.skip` 会跳过整个文件。golden 用 PyTorch CPU 计算，float 结果用 `torch.allclose` 比对，int 结果用 `torch.equal`。

## 测试失败诊断

### 快速排查

1. 检查编译日志确认无警告/报错
2. 检查 NPU 设备状态：`npu-smi info`
3. 调整运行时日志级别定位报错：`export ASCEND_GLOBAL_LOG_LEVEL=0`（0-debug、1-info、2-warning、3-error），并可加 `pytest tests/{op}/ -v -s` 查看打印
4. 对比 golden（PyTorch CPU）与 NPU 输出的最大误差和位置，float 型注意 `rtol/atol`（常用 1e-4 / 1e-3）

### 基线对比（判断是否为本次修改引入）

当测试用例执行失败时，需要判断是否为**本次修改引入的问题**：

1. **切换到上游基线分支**（如 `origin/master`），重新编译并安装、运行相同算子的测试：
   ```bash
   git checkout origin/master
   bash build.sh --ops={op_name} --soc={soc_version}
   cd /tmp && pip install {path_to_whl} --force-reinstall --no-deps
   pytest tests/{op_name}/ -v
   ```

2. **对比结果**：
   - 若基线分支上测试**通过** → 本次修改引入了问题，需要排查
   - 若基线分支上测试**同样失败** → 这是算子原有的问题，非本次修改导致

3. **切回开发分支**继续工作：
   ```bash
   git checkout {your_branch}
   ```

**注意**：切换分支前确保当前修改已 commit 或 stash，避免丢失工作进度。
