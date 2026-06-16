# ops_multimodal_fusion 接口文档

## 0. 头文件与模块结构

- **ops_multimodal_fusion Python 包**
  - 包路径：`ops_multimodal_fusion/`
  - 初始化文件：`ops_multimodal_fusion/__init__.py`
  - 算子动态库：`ops_multimodal_fusion/libops_multimodal_fusion_<op_name>.so`

- **PyTorch 扩展机制**
  - 通过 `torch.ops.ops_multimodal_fusion.<op_name>` 调用算子
  - 所有算子在 import 时自动加载

---

## 1. 模块简介

`ops_multimodal_fusion` 是基于 AscendC 的 PyTorch 自定义算子库，使用 fast kernel launch（`<<<>>>` 直调）方式实现高性能 NPU 算子，编译为 Python wheel 包供直接安装使用。

提供以下能力：
- PyTorch 扩展算子注册
- Meta 函数支持（shape/dtype 推断）
- NPU Kernel 实现
- 自动动态库加载

---

## 2. 算子接口

### 2.1 abs

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.abs(Tensor x) -> Tensor
```

#### 功能

对输入张量执行逐元素绝对值运算：`y = |x|`

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | Tensor | 输入张量，支持 FP32、FP16 数据类型 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 输出张量，shape 与输入相同，数据类型与输入相同 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |
| FP16 | `torch.float16` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | ✅ |
| Atlas A3 训练/推理系列 | arch22 | ✅ |
| 950 系列 | arch35 | ✅ |

#### 调用示例

```python
import torch
import torch_npu
import ops_multimodal_fusion

# 创建输入张量并移至 NPU
x = torch.randn(32, 64, dtype=torch.float32).npu()

# 调用 abs 算子
result = torch.ops.ops_multimodal_fusion.abs(x)

# 输出仍在 NPU 上
print(result.shape)   # torch.Size([32, 64])
print(result.dtype)   # torch.float32
print(result.device)  # npu
```

#### 源码位置

- 算子实现：`applications/llm/abs/arch22/abs.asc`
- 测试文件：`tests/abs/test_abs.py`

---

### 2.2 upsample_linear1d

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.upsample_linear1d(Tensor input, int output_size, bool align_corners=False, float scale=-1.) -> Tensor
```

#### 功能

对 3-D 张量 `[N, C, W]` 在最后一维（宽度）上做一维线性插值上/下采样，语义对齐 `torch.nn.functional.interpolate(mode="linear")`。

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | Tensor | 输入张量，形状 `[N, C, W]`，支持 FP32、FP16 数据类型 |
| `output_size` | int | 输出宽度 `OW`，必须为正 |
| `align_corners` | bool | 线性插值的 align_corners 标志，默认 `False` |
| `scale` | float | 缩放因子。默认 `-1.` 表示由 `output_size` 推导比例；传正值时按 PyTorch scale_factor 语义，且 `output_size` 须等于 `floor(W * scale)` |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 输出张量，形状 `[N, C, output_size]`，数据类型与输入相同 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |
| FP16 | `torch.float16` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 调用示例

```python
import torch
import torch_npu
import ops_multimodal_fusion

# 创建输入张量并移至 NPU
x = torch.randn(2, 3, 5, dtype=torch.float32).npu()

# 上采样到宽度 8
result = torch.ops.ops_multimodal_fusion.upsample_linear1d(x, 8, False)

print(result.shape)   # torch.Size([2, 3, 8])
print(result.dtype)   # torch.float32
```

#### 源码位置

- 算子实现：`applications/llm/upsample_linear1d/arch35/upsample_linear1d.asc`
- 测试文件：`tests/upsample_linear1d/test_upsample_linear1d.py`

---

## 3. 算子目录结构

每个算子的典型目录结构如下：

```
applications/llm/${op_name}/           # 算子目录
├── arch22/                            # Atlas A2/A3 实现
│   ├── ${op_name}.asc                 # 算子实现文件
│   └── CMakeLists.txt                 # 编译配置
└── arch35/                            # 950 系列实现（可选）
    ├── ${op_name}.asc
    └── CMakeLists.txt

tests/${op_name}/                      # 测试目录
└── test_${op_name}.py                 # pytest 测试文件
```

---

## 4. 算子实现文件说明

`.asc` 文件是算子的完整实现，包含 4 个必要部分：

| 部分 | 宏 / 关键代码 | 作用 |
|------|--------------|------|
| **Schema 注册** | `TORCH_LIBRARY_FRAGMENT(ops_multimodal_fusion, m)` | 向 PyTorch 声明算子签名 |
| **Meta 函数** | `TORCH_LIBRARY_IMPL(..., Meta, m)` | 推断输出 tensor 的 shape 和 dtype |
| **Kernel 实现** | `__global__ __aicore__ void ${op_name}_kernel(...)` | AscendC 设备端代码 |
| **NPU Dispatch** | `TORCH_LIBRARY_IMPL(..., PrivateUse1, m)` | Host 端调度 |

---

## 5. 调用流程

### 5.1 安装

```bash
pip install ops_multimodal_fusion-1.0.0-cp38-abi3-*.whl --force-reinstall
```

### 5.2 导入

```python
import ops_multimodal_fusion  # 自动加载所有算子动态库
```

### 5.3 调用

```python
import torch
import torch_npu

x = torch.randn(shape).npu()
result = torch.ops.ops_multimodal_fusion.<op_name>(x)
```

---

## 6. 更多帮助

- [快速入门](../QUICKSTART.md)
- [环境部署](install/quick_install.md)
- [算子开发指南](develop/operator_development_guide.md)
- [Ascend C API 参考](https://hiascend.com/document/redirect/CannCommunityAscendCApi)