# build.sh 参数说明

## 简介

build.sh 是 ops-multimodal-fusion 项目的构建脚本，位于项目根目录下。该脚本通过配置不同参数实现多种功能，包括编译算子、生成 wheel 包等。

## 使用方法

### 1. 配置环境变量

在使用 build.sh 之前，需要先配置 CANN 和 Python 环境：

```bash
# 配置 CANN 环境变量（默认路径安装）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 验证环境变量
echo $ASCEND_HOME_PATH
```

### 2. 构建命令格式

```bash
./build.sh [OPTIONS]
```

## 参数说明

build.sh 支持多种功能，可通过 `--help` 参数查看所有选项：

```bash
./build.sh --help
```

| 参数 | 必选/可选 | 说明 |
|------|----------|------|
| `--soc=SOC` | 可选 | 指定目标 SoC 型号，支持：`ascend910b`、`ascend910_93`、`ascend950`。默认为 `ascend950`。 |
| `-j[N]` | 可选 | 指定编译线程数，默认为 CPU 核心数。 |
| `--clean` | 可选 | 清理构建产物后重新编译。 |
| `-h, --help` | 可选 | 显示帮助信息。 |

## 支持的 SoC 型号

| SoC 型号 | 架构代号 | 说明 |
|---------|----------|------|
| ascend910b | arch22 | Atlas A2 训练/推理系列 |
| ascend910_93 | arch22 | Atlas A3 训练/推理系列 |
| ascend950 | arch35 | 950 系列 |

> **说明**：不同架构对应不同的算子实现目录（arch22/arch35）。

## 使用示例

### 基本编译

```bash
# 编译项目（默认 SoC: ascend950）
bash build.sh

# 指定 SoC 类型编译
bash build.sh --soc=ascend910b

# 使用 16 线程编译
bash build.sh -j16

# 清理后重新编译
bash build.sh --clean
```

### 组合使用

```bash
# 为 Atlas A2 编译，使用 16 线程
bash build.sh --soc=ascend910b -j16
```

## 输出说明

### 编译输出

编译成功后，生成的 wheel 包位于 `dist/` 目录：

```
dist/
└── ops_multimodal_fusion-1.0.0-cp38-abi3-linux_*.whl
```

### wheel 包说明

- **命名格式**：`ops_multimodal_fusion-{version}-cp38-abi3-{platform}.whl`
- **Python ABI**：使用 Stable ABI (abi3)，兼容 Python 3.8+
- **平台支持**：根据编译环境生成对应的平台包

## 注意事项

1. **环境变量要求**：必须设置 `ASCEND_HOME_PATH` 环境变量，否则脚本会报错退出。

2. **Python 版本**：需要 Python >= 3.8，推荐 3.10。

3. **依赖要求**：需要安装 torch 和 torch_npu。

4. **SoC 选择**：根据实际硬件选择正确的 SoC 类型。

## 相关文档

- [环境部署](quick_install.md)
- [算子调用](../invocation/quick_op_invocation.md)
- [算子开发](../develop/operator_development_guide.md)