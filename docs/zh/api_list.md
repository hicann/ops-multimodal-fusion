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
- 算子实现：`applications/llm/abs/arch35/abs.asc`
- 测试文件：`tests/abs/test_abs.py`

---
### 2.2 adaptive_avg_pool2d

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor
```

#### 功能

二维自适应平均池化，将任意输入空间尺寸平均池化到指定输出尺寸。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `self` | 输入张量，形状 [N, C, H, W] |
| `output_size` | 目标输出空间尺寸 [outH, outW] |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，形状 [N, C, outH, outW]，dtype 同输入 |

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

#### 源码位置

- 算子实现：`applications/llm/adaptive_avg_pool2d/arch35/adaptive_avg_pool2d.asc`
- 测试文件：`tests/adaptive_avg_pool2d/test_adaptive_avg_pool2d.py`

---
### 2.3 add

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.add(Tensor x, Tensor y) -> Tensor
```

#### 功能

逐元素计算两个张量之和 x + y。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 第一个输入张量 |
| `y` | 第二个输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐元素相加结果，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |
| FP16 | `torch.float16` | ✅ |
| INT32 | `torch.int32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/add/arch35/add.asc`
- 测试文件：`tests/add/test_add.py`

---
### 2.4 airy_ai

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.airy_ai(Tensor x) -> Tensor
```

#### 功能

逐元素计算艾里函数 Ai(x)。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 各元素的艾里函数值，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/airy_ai/arch35/airy_ai.asc`
- 测试文件：`tests/airy_ai/test_airy_ai.py`

---
### 2.5 angle

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.angle(Tensor input) -> Tensor
```

#### 功能

逐元素计算辐角（实数输入：非负元素为 0、负元素为 π）。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `input` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，逐元素辐角，dtype 同输入 |

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

#### 源码位置

- 算子实现：`applications/llm/angle/arch35/angle.asc`
- 测试文件：`tests/angle/test_angle.py`

---
### 2.6 any

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.any(Tensor x, int dim, bool keepdim) -> Tensor
```

#### 功能

沿指定维度做逻辑或归约，判断该维度上是否存在非零元素，输出布尔张量。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |
| `dim` | 归约维度 |
| `keepdim` | 是否保留被归约的维度 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 布尔张量（`torch.bool`），表示对应维度是否存在非零元素 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |
| FP16 | `torch.float16` | ✅ |
| INT32 | `torch.int32` | ✅ |
| BOOL | `torch.bool` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/any/arch35/any.asc`
- 测试文件：`tests/any/test_any.py`

---
### 2.7 avg_pool2d

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
```

#### 功能

对二维输入做平均池化，按滑动窗口对各区域取均值。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `self` | 输入张量 |
| `kernel_size` | 池化窗口大小 (kH, kW) |
| `stride` | 滑动步长，默认与 kernel_size 相同 |
| `padding` | 输入两侧的零填充大小 |
| `ceil_mode` | 计算输出尺寸时是否向上取整 |
| `count_include_pad` | 求均值时是否计入填充元素 |
| `divisor_override` | 指定时用作除数覆盖默认池化区域大小 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 平均池化后的输出张量，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/avg_pool2d/arch35/avg_pool2d.asc`
- 测试文件：`tests/avg_pool2d/test_avg_pool2d.py`

---
### 2.8 bessel_j0

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.bessel_j0(Tensor x) -> Tensor
```

#### 功能

逐元素计算第一类零阶贝塞尔函数 J0(x)。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/bessel_j0/arch35/bessel_j0.asc`
- 测试文件：`tests/bessel_j0/test_bessel_j0.py`

---
### 2.9 bessel_j1

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.bessel_j1(Tensor x) -> Tensor
```

#### 功能

逐元素计算第一类一阶贝塞尔函数 J1(x)。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/bessel_j1/arch35/bessel_j1.asc`
- 测试文件：`tests/bessel_j1/test_bessel_j1.py`

---
### 2.10 bessel_y0

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.bessel_y0(Tensor x) -> Tensor
```

#### 功能

逐元素计算第二类零阶贝塞尔函数 Y0(x)。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/bessel_y0/arch35/bessel_y0.asc`
- 测试文件：`tests/bessel_y0/test_bessel_y0.py`

---
### 2.11 bessel_y1

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.bessel_y1(Tensor x) -> Tensor
```

#### 功能

逐元素计算第二类一阶贝塞尔函数 Y1(x)。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/bessel_y1/arch35/bessel_y1.asc`
- 测试文件：`tests/bessel_y1/test_bessel_y1.py`

---
### 2.12 binomial

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.binomial(Tensor count, Tensor prob, int seed=0) -> Tensor
```

#### 功能

按二项分布 B(count, prob) 逐元素采样，count 为试验次数、prob 为单次成功概率。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `count` | 试验次数张量 |
| `prob` | 单次成功概率张量 |
| `seed` | 随机数种子 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 二项分布采样结果，dtype 同输入 |

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

#### 源码位置

- 算子实现：`applications/llm/binomial/arch35/binomial.asc`
- 测试文件：`tests/binomial/test_binomial.py`

---
### 2.13 bitwisenot

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.bitwisenot(Tensor x) -> Tensor
```

#### 功能

逐元素对整数张量做按位取反 ~x。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 整数输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 按位取反结果，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| INT32 | `torch.int32` | ✅ |
| INT16 | `torch.int16` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/bitwisenot/arch35/bitwisenot.asc`
- 测试文件：`tests/bitwisenot/test_bitwisenot.py`

---
### 2.14 c2_accuracy

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.c2_accuracy(Tensor predictions, Tensor labels, int top_k=1) -> Tensor
```

#### 功能

Caffe2 分类准确率算子，统计预测的 top_k 中命中真实标签的样本比例。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `predictions` | 各样本各类别的预测得分张量 |
| `labels` | 真实标签张量 |
| `top_k` | 取预测得分最高的前 k 个判定是否命中 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 标量准确率张量，dtype 同输入 |

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

#### 源码位置

- 算子实现：`applications/llm/c2_accuracy/arch35/c2_accuracy.asc`
- 测试文件：`tests/c2_accuracy/test_c2_accuracy.py`

---
### 2.15 c2_affine_channel

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.c2_affine_channel(Tensor X, Tensor scale, Tensor bias) -> Tensor
```

#### 功能

Caffe2 通道仿射变换，按通道对输入做 X * scale + bias。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `X` | 输入张量 |
| `scale` | 各通道的缩放系数 |
| `bias` | 各通道的偏置 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 通道仿射变换后的张量，dtype 同输入 |

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

#### 源码位置

- 算子实现：`applications/llm/c2_affine_channel/arch35/c2_affine_channel.asc`
- 测试文件：`tests/c2_affine_channel/test_c2_affine_channel.py`

---
### 2.16 c2_batch_moments

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.c2_batch_moments(Tensor X) -> (Tensor mu, Tensor var)
```

#### 功能

Caffe2 批量矩统计，沿批维度计算输入的均值 mu 与方差 var。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `X` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 元组 (mu, var)：批量均值与方差张量，dtype 同输入 |

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

#### 源码位置

- 算子实现：`applications/llm/c2_batch_moments/arch35/c2_batch_moments.asc`
- 测试文件：`tests/c2_batch_moments/test_c2_batch_moments.py`

---
### 2.17 c2_batch_permutation

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.c2_batch_permutation(Tensor X, Tensor indices) -> Tensor
```

#### 功能

Caffe2 批量重排，按 indices 给出的顺序沿批维度重新排列输入。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `X` | 输入张量 |
| `indices` | 批维度的重排索引张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 按索引重排后的张量，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |
| FP16 | `torch.float16` | ✅ |
| INT32 | `torch.int32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/c2_batch_permutation/arch35/c2_batch_permutation.asc`
- 测试文件：`tests/c2_batch_permutation/test_c2_batch_permutation.py`

---
### 2.18 c2_boolean_mask

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.c2_boolean_mask(Tensor data, Tensor mask) -> (Tensor masked_data, Tensor masked_indices)
```

#### 功能

Caffe2 布尔掩码选择，按 mask 为真的位置筛选 data，并返回被选元素及其原始索引。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `data` | 输入数据张量 |
| `mask` | 布尔掩码张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 元组 (masked_data, masked_indices)：被选中的数据及其原始索引 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |
| FP16 | `torch.float16` | ✅ |
| INT32 | `torch.int32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/c2_boolean_mask/arch35/c2_boolean_mask.asc`
- 测试文件：`tests/c2_boolean_mask/test_c2_boolean_mask.py`

---
### 2.19 c2_boolean_unmask

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.c2_boolean_unmask(Tensor[] inputs) -> Tensor
```

#### 功能

Caffe2 布尔反掩码，按交替给出的掩码与数据张量列表，将各数据散布回对应位置组装成完整输出。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `inputs` | 掩码与数据交替排列的张量列表 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 反掩码组装后的完整张量，dtype 同数据输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |
| FP16 | `torch.float16` | ✅ |
| INT32 | `torch.int32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/c2_boolean_unmask/arch35/c2_boolean_unmask.asc`
- 测试文件：`tests/c2_boolean_unmask/test_c2_boolean_unmask.py`

---
### 2.20 c2_bucketize

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.c2_bucketize(Tensor self, float[] boundaries) -> Tensor
```

#### 功能

Caffe2 分桶算子，按给定的有序边界计算每个输入元素所落入的桶序号。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `self` | 输入张量 |
| `boundaries` | 升序排列的分桶边界列表 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 各元素对应的桶序号张量 |

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

#### 源码位置

- 算子实现：`applications/llm/c2_bucketize/arch35/c2_bucketize.asc`
- 测试文件：`tests/c2_bucketize/test_c2_bucketize.py`

---
### 2.21 c2_cbrt

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.c2_cbrt(Tensor self) -> Tensor
```

#### 功能

Caffe2 立方根算子，逐元素计算输入的立方根 cbrt(x)。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `self` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 各元素立方根结果，dtype 同输入 |

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

#### 源码位置

- 算子实现：`applications/llm/c2_cbrt/arch35/c2_cbrt.asc`
- 测试文件：`tests/c2_cbrt/test_c2_cbrt.py`

---
### 2.22 cauchy

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.cauchy(Tensor x, float median=0.0, float sigma=1.0, int seed=0) -> Tensor
```

#### 功能

按柯西分布随机采样填充张量。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量（提供形状/dtype/device） |
| `median` | 分布位置参数，默认 0.0 |
| `sigma` | 分布尺度参数，默认 1.0 |
| `seed` | 随机种子，默认 0 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，形状/dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP16 | `torch.float16` | ✅ |
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/cauchy/arch35/cauchy.asc`
- 测试文件：`tests/cauchy/test_cauchy.py`

---
### 2.23 cdist_backward

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.cdist_backward(Tensor grad, Tensor x1, Tensor x2, float p, Tensor cdist) -> Tensor
```

#### 功能

计算 `torch.cdist` 关于 x1 的梯度，用于 p-范数成对距离的反向传播。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `grad` | 距离矩阵的上游梯度 |
| `x1` | 前向输入 x1，形状 [..., P, M] |
| `x2` | 前向输入 x2，形状 [..., R, M] |
| `p` | 范数阶数 p |
| `cdist` | 前向输出的成对距离矩阵 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | x1 的梯度，形状与 x1 相同，dtype 同输入 |

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

#### 源码位置

- 算子实现：`applications/llm/cdist_backward/arch35/cdist_backward.asc`
- 测试文件：`tests/cdist_backward/test_cdist_backward.py`

---
### 2.24 chebyshev_polynomial_t

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.chebyshev_polynomial_t(Tensor x, Tensor n) -> Tensor
```

#### 功能

逐元素计算第一类切比雪夫多项式 T_n(x)。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |
| `n` | 多项式阶数张量（内部按整数处理） |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 各元素的 T_n(x) 值，FP32 输出 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/chebyshev_polynomial_t/arch35/chebyshev_polynomial_t.asc`
- 测试文件：`tests/chebyshev_polynomial_t/test_chebyshev_polynomial_t.py`

---
### 2.25 chebyshev_polynomial_u

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.chebyshev_polynomial_u(Tensor x, Tensor n) -> Tensor
```

#### 功能

逐元素计算第二类切比雪夫多项式 U_n(x)。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |
| `n` | 多项式阶数张量（内部按整数处理） |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 各元素的 U_n(x) 值，FP32 输出 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/chebyshev_polynomial_u/arch35/chebyshev_polynomial_u.asc`
- 测试文件：`tests/chebyshev_polynomial_u/test_chebyshev_polynomial_u.py`

---
### 2.26 chebyshev_polynomial_v

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.chebyshev_polynomial_v(Tensor x, Tensor n) -> Tensor
```

#### 功能

逐元素计算第三类切比雪夫多项式 V_n(x)，n 为多项式阶数。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |
| `n` | 多项式阶数张量（与 x 可广播） |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐元素 V_n(x) 结果，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/chebyshev_polynomial_v/arch35/chebyshev_polynomial_v.asc`
- 测试文件：`tests/chebyshev_polynomial_v/test_chebyshev_polynomial_v.py`

---
### 2.27 chebyshev_polynomial_w

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.chebyshev_polynomial_w(Tensor x, Tensor n) -> Tensor
```

#### 功能

逐元素计算第四类切比雪夫多项式 W_n(x)，n 为多项式阶数。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |
| `n` | 多项式阶数张量（与 x 可广播） |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐元素 W_n(x) 结果，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/chebyshev_polynomial_w/arch35/chebyshev_polynomial_w.asc`
- 测试文件：`tests/chebyshev_polynomial_w/test_chebyshev_polynomial_w.py`

---
### 2.28 complex

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.complex(Tensor real, Tensor imag) -> Tensor
```

#### 功能

由实部与虚部张量构造复数张量。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `real` | 实部张量 |
| `imag` | 虚部张量，形状/dtype 同 real |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，复数张量 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP16 | `torch.float16` | ✅ |
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/complex/arch35/complex.asc`
- 测试文件：`tests/complex/test_complex.py`

---
### 2.29 conjphysical

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.conjphysical(Tensor x) -> Tensor
```

#### 功能

逐元素物理共轭。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP16 | `torch.float16` | ✅ |
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/conjphysical/arch35/conjphysical.asc`
- 测试文件：`tests/conjphysical/test_conjphysical.py`

---
### 2.30 copysign

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.copysign(Tensor a, Tensor b) -> Tensor
```

#### 功能

逐元素将 b 的符号复制到 a 的数值上，返回幅值取自 a、符号取自 b 的结果（含符号零、NaN 按 IEEE-754 处理）。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `a` | 提供幅值的输入张量 |
| `b` | 提供符号的输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 幅值同 a、符号同 b 的结果，dtype 同输入 |

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

#### 源码位置

- 算子实现：`applications/llm/copysign/arch35/copysign.asc`
- 测试文件：`tests/copysign/test_copysign.py`

---
### 2.31 cummax

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.cummax(Tensor x, int dim) -> (Tensor values, Tensor indices)
```

#### 功能

沿指定维度计算累积最大值及其索引。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |
| `dim` | 归约维度 |

#### 返回值

| 类型 | 说明 |
|------|------|
| (Tensor, Tensor) | (values, indices)：累积最大值与对应索引 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP16 | `torch.float16` | ✅ |
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/cummax/arch35/cummax.asc`
- 测试文件：`tests/cummax/test_cummax.py`

---
### 2.32 cumprod

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.cumprod(Tensor x, int dim) -> Tensor
```

#### 功能

沿指定维度计算累积乘积。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |
| `dim` | 累乘维度 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP16 | `torch.float16` | ✅ |
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/cumprod/arch35/cumprod.asc`
- 测试文件：`tests/cumprod/test_cumprod.py`

---
### 2.33 depthwise_conv3d

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.depthwise_conv3d(Tensor input, Tensor weight, Tensor? bias, int[3] stride, int[3] padding, int[3] dilation) -> Tensor
```

#### 功能

三维逐通道（depthwise）卷积。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `input` | 输入张量 [N, C, D, H, W] |
| `weight` | 逐通道卷积核 |
| `bias` | 可选偏置，可为 None |
| `stride` | 三维步长 [sd, sh, sw] |
| `padding` | 三维填充 [pd, ph, pw] |
| `dilation` | 三维膨胀 [dd, dh, dw] |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，卷积输出 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP16 | `torch.float16` | ✅ |
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/depthwise_conv3d/arch35/depthwise_conv3d.asc`
- 测试文件：`tests/depthwise_conv3d/test_depthwise_conv3d.py`

---
### 2.34 dequant_swiglu_quant

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.dequant_swiglu_quant(Tensor x, Tensor? weight_scale, Tensor? act_scale) -> (Tensor, Tensor)
```

#### 功能

融合反量化、SwiGLU 激活与动态量化三步：先按 weight_scale/act_scale 反量化输入，再执行 SwiGLU（silu(右半) * 左半），最后做按行动态量化输出 int8 及对应缩放因子。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量（最后一维为偶数，前一半与后一半参与 SwiGLU） |
| `weight_scale` | 可选权重反量化缩放因子（int32 输入时使用） |
| `act_scale` | 可选激活反量化缩放因子（int32 输入时使用） |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | int8 量化输出（最后一维减半） |
| Tensor | 按行的 float32 量化缩放因子 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| INT32 | `torch.int32` | ✅ |
| FP16 | `torch.float16` | ✅ |
| BF16 | `torch.bfloat16` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/dequant_swiglu_quant/arch35/dequant_swiglu_quant.asc`
- 测试文件：`tests/dequant_swiglu_quant/test_dequant_swiglu_quant.py`

---
### 2.35 digamma

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.digamma(Tensor x) -> Tensor
```

#### 功能

逐元素计算 digamma 函数（lnΓ 的一阶导数）。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/digamma/arch35/digamma.asc`
- 测试文件：`tests/digamma/test_digamma.py`

---
### 2.36 dirichlet

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.dirichlet(Tensor alpha, int seed=0) -> Tensor
```

#### 功能

按浓度参数 alpha 从狄利克雷分布采样，最后一维为单纯形轴：每行 K 个元素采样并归一化为概率单纯形上一点（非负且行和为 1）。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `alpha` | 浓度参数张量，最后一维为事件/单纯形轴 |
| `seed` | 随机数种子，默认 0 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 与 alpha 同形状同 dtype 的采样结果，每行落在概率单纯形上 |

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

#### 源码位置

- 算子实现：`applications/llm/dirichlet/arch35/dirichlet.asc`
- 测试文件：`tests/dirichlet/test_dirichlet.py`

---
### 2.37 dynamic_quant

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.dynamic_quant(Tensor x, bool symmetric=True) -> (Tensor, Tensor)
```

#### 功能

对输入按行做动态量化，输出 int8 量化张量及对应的按行 float32 缩放因子，支持对称与非对称两种模式。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |
| `symmetric` | 是否对称量化，默认 True |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | int8 量化输出 |
| Tensor | 按行的 float32 量化缩放因子 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP16 | `torch.float16` | ✅ |
| BF16 | `torch.bfloat16` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/dynamic_quant/arch35/dynamic_quant.asc`
- 测试文件：`tests/dynamic_quant/test_dynamic_quant.py`

---
### 2.38 entr

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.entr(Tensor x) -> Tensor
```

#### 功能

逐元素计算熵函数 entr(x) = -x·ln(x)（x>0），x=0 取 0，x<0 取 -inf。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP16 | `torch.float16` | ✅ |
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/entr/arch35/entr.asc`
- 测试文件：`tests/entr/test_entr.py`

---
### 2.39 erfcx

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.erfcx(Tensor x) -> Tensor
```

#### 功能

逐元素计算缩放互补误差函数 erfcx(x) = exp(x²)·erfc(x)。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/erfcx/arch35/erfcx.asc`
- 测试文件：`tests/erfcx/test_erfcx.py`

---
### 2.40 exp

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.exp(Tensor x) -> Tensor
```

#### 功能

逐元素计算自然指数 exp(x)。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐元素 exp(x) 结果，dtype 同输入 |

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

#### 源码位置

- 算子实现：`applications/llm/exp/arch35/exp.asc`
- 测试文件：`tests/exp/test_exp.py`

---
### 2.41 exp2

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.exp2(Tensor x) -> Tensor
```

#### 功能

逐元素计算以 2 为底的指数：`y = 2^x`。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，shape 同输入，dtype 同输入 |

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

#### 源码位置

- 算子实现：`applications/llm/exp2/arch35/exp2.asc`
- 测试文件：`tests/exp2/test_exp2.py`

---
### 2.42 exponential

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.exponential(Tensor x, float lambd=1.0, int seed=0) -> Tensor
```

#### 功能

按指数分布 `Exponential(lambd)` 随机采样填充张量，等价于 `torch.Tensor.exponential_`；输入仅提供形状/dtype/device。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量（仅提供形状/dtype/device，数据不读取） |
| `lambd` | 指数分布率参数，默认 1.0 |
| `seed` | 随机种子，默认 0 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，形状/dtype 同输入，填充指数分布样本 |

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

#### 源码位置

- 算子实现：`applications/llm/exponential/arch35/exponential.asc`
- 测试文件：`tests/exponential/test_exponential.py`

---
### 2.43 fft_conj_symmetry

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.fft_conj_symmetry(Tensor input, int dim, int out_size) -> Tensor
```

#### 功能

利用 Hermitian 共轭对称将单边（onesided）FFT 结果沿指定维度还原为完整双边谱。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `input` | 复数输入张量，沿 `dim` 长度为 `out_size // 2 + 1` |
| `dim` | 应用共轭对称的维度 |
| `out_size` | 沿 `dim` 的完整输出长度（须 ≥ 2） |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 复数输出张量，沿 `dim` 长度为 `out_size`，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| Complex64 | `torch.complex64` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/fft_conj_symmetry/arch35/fft_conj_symmetry.asc`
- 测试文件：`tests/fft_conj_symmetry/test_fft_conj_symmetry.py`

---
### 2.44 foreach_ceil

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.foreach_ceil(Tensor[] tensors) -> Tensor[]
```

#### 功能

对张量列表逐元素向上取整。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `tensors` | 输入张量列表 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor[] | Tensor[]，逐张量逐元素 ceil 结果 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP16 | `torch.float16` | ✅ |
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/foreach_ceil/arch35/foreach_ceil.asc`
- 测试文件：`tests/foreach_ceil/test_foreach_ceil.py`

---
### 2.45 foreach_floor

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.foreach_floor(Tensor[] tensors) -> Tensor[]
```

#### 功能

对张量列表中每个张量逐元素向下取整（floor），返回结果张量列表。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `tensors` | 输入张量列表 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐张量 floor 后的结果张量列表，dtype 同输入 |

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

#### 源码位置

- 算子实现：`applications/llm/foreach_floor/arch35/foreach_floor.asc`
- 测试文件：`tests/foreach_floor/test_foreach_floor.py`

---
### 2.46 foreach_frac

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.foreach_frac(Tensor[] tensors) -> Tensor[]
```

#### 功能

对张量列表中每个张量逐元素取小数部分（x - trunc(x)），返回结果张量列表。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `tensors` | 输入张量列表 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐张量取小数部分后的结果张量列表，dtype 同输入 |

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

#### 源码位置

- 算子实现：`applications/llm/foreach_frac/arch35/foreach_frac.asc`
- 测试文件：`tests/foreach_frac/test_foreach_frac.py`

---
### 2.47 foreach_lgamma

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.foreach_lgamma(Tensor[] tensors) -> Tensor[]
```

#### 功能

对张量列表中每个张量逐元素计算 log-Gamma（lgamma = log|Γ(x)|），返回结果张量列表。仅支持 float32。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `tensors` | 输入张量列表，元素须为 float32 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐张量 lgamma 后的结果张量列表 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/foreach_lgamma/arch35/foreach_lgamma.asc`
- 测试文件：`tests/foreach_lgamma/test_foreach_lgamma.py`

---
### 2.48 foreach_trunc

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.foreach_trunc(Tensor[] tensors) -> Tensor[]
```

#### 功能

对张量列表中每个张量逐元素向零取整（trunc），返回结果张量列表。支持 float32 / float16。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `tensors` | 输入张量列表，元素须为 float32 或 float16 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐张量 trunc 后的结果张量列表，dtype 同输入 |

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

#### 源码位置

- 算子实现：`applications/llm/foreach_trunc/arch35/foreach_trunc.asc`
- 测试文件：`tests/foreach_trunc/test_foreach_trunc.py`

---
### 2.49 fractional_max_pool2d

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.fractional_max_pool2d(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples) -> (Tensor values, Tensor indices)
```

#### 功能

二维分数最大池化，按随机采样确定滑动窗口起点并在窗口内取最大值，语义对齐 `torch.nn.functional.fractional_max_pool2d`。

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `self` | Tensor | 输入张量，形状 `[C, H, W]` 或 `[N, C, H, W]`，支持 FP32、FP16 |
| `kernel_size` | int[2] | 池化窗口大小 `[kH, kW]` |
| `output_size` | int[2] | 输出空间尺寸 `[outH, outW]` |
| `random_samples` | Tensor | 随机采样张量，形状 `[N, C, 2]` 或 `[C, 2]`，float32 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 池化输出值，dtype 与输入相同 |
| Tensor | 池化索引，int64，shape 与输出值相同 |

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

#### 源码位置

- 算子实现：`applications/llm/fractional_max_pool2d/arch35/fractional_max_pool2d.asc`
- 测试文件：`tests/fractional_max_pool2d/test_fractional_max_pool2d.py`

---
### 2.50 fractional_max_pool3d

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.fractional_max_pool3d(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples) -> (Tensor values, Tensor indices)
```

#### 功能

三维分数最大池化，按随机采样确定滑动窗口起点并在窗口内取最大值，语义对齐 `torch.nn.functional.fractional_max_pool3d`。

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `self` | Tensor | 输入张量，形状 `[C, D, H, W]` 或 `[N, C, D, H, W]`，支持 FP32、FP16 |
| `kernel_size` | int[3] | 池化窗口大小 `[kD, kH, kW]` |
| `output_size` | int[3] | 输出空间尺寸 `[outD, outH, outW]` |
| `random_samples` | Tensor | 随机采样张量，形状 `[N, C, 3]` 或 `[C, 3]`，float32 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 池化输出值，dtype 与输入相同 |
| Tensor | 池化索引，int64，shape 与输出值相同 |

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

#### 源码位置

- 算子实现：`applications/llm/fractional_max_pool3d/arch35/fractional_max_pool3d.asc`
- 测试文件：`tests/fractional_max_pool3d/test_fractional_max_pool3d.py`

---
### 2.51 frexp

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.frexp(Tensor x) -> (Tensor mantissa, Tensor exponent)
```

#### 功能

将输入按 IEEE-754 分解为尾数与指数，满足 `x = mantissa * 2^exponent`，尾数取值范围 `(-1, -0.5] ∪ {0} ∪ [0.5, 1)`。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| (Tensor, Tensor) | (mantissa, exponent)：尾数 dtype 同输入，指数为 int32 |

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

#### 源码位置

- 算子实现：`applications/llm/frexp/arch35/frexp.asc`
- 测试文件：`tests/frexp/test_frexp.py`

---
### 2.52 gamma

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.gamma(Tensor alpha, int seed=0) -> Tensor
```

#### 功能

按浓度参数 alpha 从伽马分布逐元素采样（Marsaglia-Tsang 方法）。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `alpha` | 伽马分布浓度（形状）参数张量 |
| `seed` | 随机数种子，默认 0 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 与 alpha 同形状同 dtype 的伽马采样结果 |

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

#### 源码位置

- 算子实现：`applications/llm/gamma/arch35/gamma.asc`
- 测试文件：`tests/gamma/test_gamma.py`

---
### 2.53 gelu

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.gelu(Tensor x) -> Tensor
```

#### 功能

逐元素计算 GELU 激活函数。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐元素 GELU(x) 结果，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/gelu/arch35/gelu.asc`
- 测试文件：`tests/gelu/test_gelu.py`

---
### 2.54 geometric

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.geometric(Tensor x, float p, int seed=0) -> Tensor
```

#### 功能

按几何分布 `Geometric(p)`（支持集 {1,2,3,...}）随机采样填充张量，等价于 `torch.Tensor.geometric_`；输入仅提供形状/dtype/device。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量（仅提供形状/dtype/device，数据不读取） |
| `p` | 成功概率参数 |
| `seed` | 随机种子，默认 0 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，形状/dtype 同输入，填充几何分布样本 |

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

#### 源码位置

- 算子实现：`applications/llm/geometric/arch35/geometric.asc`
- 测试文件：`tests/geometric/test_geometric.py`

---
### 2.55 get_indice_pairs_subm_lookup

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.get_indice_pairs_subm_lookup(Tensor coords, int[] spatial_shape, int[] ksize, int[] dilation, Tensor sorted_enc, Tensor sorted_row) -> Tensor
```

#### 功能

稀疏卷积索引生成（`get_indice_pairs`）subm 路径的「邻居查重」半场（Ascend C，SIMT 并行查找）：

对每个 (输入 voxel `i`, kernel offset `k`) 计算候选坐标 `cand = coords[i] + (koff - center) * dilation`；
若候选落在 `[0, spatial_shape)` 且命中活跃 voxel，则输出该邻居的输入行号，否则输出 `-1`。
用于替代 NPU 上昂贵的 `(N, K)` 次 `torch.searchsorted`；排序与配对打包由 host 侧完成。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `coords` | 输入 voxel 坐标，形状 `(N, ndim)`，int64，连续 |
| `spatial_shape` / `ksize` / `dilation` | 长度均为 `ndim` 的 int 列表（取值 ≥ 1） |
| `sorted_enc` | 输入坐标线性编码的升序结果，形状 `(N,)`，int64，连续（host 侧 `torch.sort` 产出） |
| `sorted_row` | 每个排序位置对应的原始输入行号，形状 `(N,)`，int32，连续；值域 `[0, N)` |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 形状 `(N, K)`，int32（`K = prod(ksize)`）；`neighbor[i*K+k]` 为邻居输入行号，无邻居为 `-1` |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| int64 / int32 | `torch.int64` / `torch.int32` | ✅（`coords`/`sorted_enc` 为 int64，`sorted_row` 为 int32，输出 int32） |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/autonomous_driving/get_indice_pairs/arch35/get_indice_pairs.asc`

---
### 2.56 gru_cell

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.gru_cell(Tensor input, Tensor hx, Tensor weight_ih, Tensor weight_hh, Tensor? bias_ih=None, Tensor? bias_hh=None) -> Tensor
```

#### 功能

单步 GRU 单元前向，门顺序遵循 PyTorch（reset, update, new），支持 1-D 非批和 2-D 批输入。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `input` | 输入张量，[I] 或 [B, I] |
| `hx` | 上一时刻隐藏状态，[H] 或 [B, H] |
| `weight_ih` | 输入到隐藏的权重，[3H, I] |
| `weight_hh` | 隐藏到隐藏的权重，[3H, H] |
| `bias_ih` | 可选输入偏置 [3H]，可为 None |
| `bias_hh` | 可选隐藏偏置 [3H]，可为 None |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 新隐藏状态 hy，形状同 hx，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP16 | `torch.float16` | ✅ |
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/gru_cell/arch35/gru_cell.asc`
- 测试文件：`tests/gru_cell/test_gru_cell.py`

---
### 2.57 hermite_polynomial_h

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.hermite_polynomial_h(Tensor x, Tensor n) -> Tensor
```

#### 功能

逐元素计算物理学家厄米多项式 H_n(x)，n 为多项式阶数。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |
| `n` | 多项式阶数张量（与 x 可广播） |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐元素 H_n(x) 结果，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/hermite_polynomial_h/arch35/hermite_polynomial_h.asc`
- 测试文件：`tests/hermite_polynomial_h/test_hermite_polynomial_h.py`

---
### 2.58 hermite_polynomial_he

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.hermite_polynomial_he(Tensor x, Tensor n) -> Tensor
```

#### 功能

逐元素计算概率学家厄米多项式 He_n(x)，n 为多项式阶数。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |
| `n` | 多项式阶数张量（与 x 可广播） |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐元素 He_n(x) 结果，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/hermite_polynomial_he/arch35/hermite_polynomial_he.asc`
- 测试文件：`tests/hermite_polynomial_he/test_hermite_polynomial_he.py`

---
### 2.59 hypot

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.hypot(Tensor x, Tensor y) -> Tensor
```

#### 功能

逐元素计算直角三角形斜边 sqrt(x^2 + y^2)。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 第一条直角边输入张量 |
| `y` | 第二条直角边输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐元素 sqrt(x^2 + y^2) 结果，dtype 同输入 |

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

#### 源码位置

- 算子实现：`applications/llm/hypot/arch35/hypot.asc`
- 测试文件：`tests/hypot/test_hypot.py`

---
### 2.60 igamma

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.igamma(Tensor a, Tensor x) -> Tensor
```

#### 功能

逐元素计算正则化下不完全伽马函数 P(a, x)。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `a` | 形状参数张量 |
| `x` | 自变量张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐元素 P(a, x) 结果，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/igamma/arch35/igamma.asc`
- 测试文件：`tests/igamma/test_igamma.py`

---
### 2.61 igammac

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.igammac(Tensor a, Tensor x) -> Tensor
```

#### 功能

逐元素计算正则化上不完全伽马函数 Q(a, x) = 1 - P(a, x)。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `a` | 形状参数张量 |
| `x` | 自变量张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐元素 Q(a, x) 结果，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/igammac/arch35/igammac.asc`
- 测试文件：`tests/igammac/test_igammac.py`

---
### 2.62 index_copy

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor
```

#### 功能

按索引沿指定维度拷贝：对每个 j，`result[index[j]] = source[j]`，未被索引的位置保留 self 原值。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `self` | 目标张量（提供底值） |
| `dim` | 索引作用的维度 |
| `index` | 索引张量（int64） |
| `source` | 待写入的源数据 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，形状/dtype 同 self |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |
| FP16 | `torch.float16` | ✅ |
| INT32 | `torch.int32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/index_copy/arch35/index_copy.asc`
- 测试文件：`tests/index_copy/test_index_copy.py`

---
### 2.63 index_reduce

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.index_reduce(Tensor self, int dim, Tensor index, Tensor source, str reduce, bool include_self=True) -> Tensor
```

#### 功能

按索引将 source 沿 dim 归约聚合到 self。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `self` | 目标张量 |
| `dim` | 聚合维度 |
| `index` | 索引张量 |
| `source` | 源张量 |
| `reduce` | 归约方式（prod/mean/amax/amin 等） |
| `include_self` | 是否将 self 原值纳入归约，默认 True |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，聚合结果，dtype 同 self |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |
| FP16 | `torch.float16` | ✅ |
| INT32 | `torch.int32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/index_reduce/arch35/index_reduce.asc`
- 测试文件：`tests/index_reduce/test_index_reduce.py`

---
### 2.64 indice_conv

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.indice_conv(Tensor gathered, Tensor filters, Tensor k_idx, Tensor s_idx, int P) -> Tensor
```

#### 功能

Ascend C 稀疏卷积 per-pair matmul 算子（稀疏卷积 host-gather + kernel-compute 的 kernel 侧）：

```text
partial[s_idx[p], :] = gathered[p, :] @ filters[k_idx[p], :, :]
```

host 侧将稀疏卷积的有效 (input, output, kernel-offset) 三元组按 k-major 拍平并 gather 输入行，
本算子逐 pair 计算 `gathered[p] @ filters[k_idx[p]]`，再由 host 按输出 voxel scatter（index_add）累加
到输出张量。subm 中心偏移由 host 用稠密 mm 计算，不进入本算子。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `gathered` | gather 后的输入行，形状 `(P, C_in)`，FP32/FP16，连续 |
| `filters` | 卷积权重，形状 `(K, C_in, C_out)`，dtype 同 `gathered`，连续 |
| `k_idx` | 每个 pair 的 kernel-offset 索引，形状 `(P,)`，int32，连续；值域 `[0, K)`（K = `filters.size(0)`），越界报错 |
| `s_idx` | 每个 pair 在 partial 中的目标行号，形状 `(P,)`，int32，连续；值域 `[0, P)`，越界报错；**必须无重复**（见下方前置条件） |
| `P` | pair 数，须等于 `gathered.size(0)` |

> **前置条件（违反即未定义行为 / 报错）**
> 1. `k_idx[p] ∈ [0, K)`（K = `filters.size(0)`）且 `s_idx[p] ∈ [0, P)`：越界会在 host 侧经 on-device 归约校验后抛 `RuntimeError`，不会静默越界读写 GM。
> 2. **`s_idx` 必须无重复（单射）**：kernel 由 SIMT 线程并行写 `partial[s_idx[p]]`；若两个 pair 的 `s_idx` 相同，同一输出行被多线程竞争写（非原子、甚至非确定 last-writer-wins），结果为**未定义**。算子只保证模型侧传 identity（`build_k` 产生的 `0..P-1` 无重复排列）安全，其余调用方须自行保证。

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，形状 `(P, C_out)`，dtype 同 `gathered`；`partial[s_idx[p], :] = gathered[p, :] @ filters[k_idx[p], :, :]` |

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

#### 源码位置

- 算子实现：`applications/autonomous_driving/indice_conv/arch35/indice_conv.asc`
- 测试文件：`tests/indice_conv/test_indice_conv.py`

---

### 2.65 int_repr

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.int_repr(Tensor x) -> Tensor
```

#### 功能

取量化张量的底层整数表示。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入（量化）张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，整数表示 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| INT32 | `torch.int32` | ✅ |
| INT8 | `torch.int8` | ✅ |
| UINT8 | `torch.uint8` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/int_repr/arch35/int_repr.asc`
- 测试文件：`tests/int_repr/test_int_repr.py`

---
### 2.66 kaiserwindow

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.kaiserwindow(Tensor x, float beta, int window_length, bool periodic) -> Tensor
```

#### 功能

按形状参数 beta 计算凯泽窗（Kaiser window），支持周期与对称两种窗形。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 窗内位置索引输入张量 |
| `beta` | 凯泽窗形状参数 |
| `window_length` | 窗长 |
| `periodic` | 是否为周期窗（True 周期，False 对称） |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 凯泽窗系数，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/kaiserwindow/arch35/kaiserwindow.asc`
- 测试文件：`tests/kaiserwindow/test_kaiserwindow.py`

---
### 2.67 kthvalue

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.kthvalue(Tensor x, int k, int dim, bool keepdim) -> (Tensor values, Tensor indices)
```

#### 功能

沿指定维度取第 k 小的值及其索引。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |
| `k` | 第 k 小（1-based） |
| `dim` | 归约维度 |
| `keepdim` | 是否保留归约维度 |

#### 返回值

| 类型 | 说明 |
|------|------|
| (Tensor, Tensor) | (values, indices)：第 k 小值与对应索引 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP16 | `torch.float16` | ✅ |
| FP32 | `torch.float32` | ✅ |
| INT32 | `torch.int32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/kthvalue/arch35/kthvalue.asc`
- 测试文件：`tests/kthvalue/test_kthvalue.py`

---
### 2.68 laguerre_polynomial_l

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.laguerre_polynomial_l(Tensor x, Tensor n) -> Tensor
```

#### 功能

计算拉盖尔多项式 L_n(x)，n 为阶数（按元素广播）。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量，多项式自变量 |
| `n` | 阶数张量，与 x 广播；按元素取整作为多项式阶 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐元素的拉盖尔多项式值，形状为 x 与 n 广播后的形状 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/laguerre_polynomial_l/arch35/laguerre_polynomial_l.asc`
- 测试文件：`tests/laguerre_polynomial_l/test_laguerre_polynomial_l.py`

---
### 2.69 layer_norm

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.layer_norm(Tensor x, Tensor gamma, Tensor beta, float eps=1e-6) -> Tensor
```

#### 功能

层归一化，对最后一维做归一化后施加可学习的缩放与偏移：gamma * (x - mean) / sqrt(var + eps) + beta。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量，2-D（行数, hidden_size） |
| `gamma` | 1-D 缩放参数，长度等于 hidden_size |
| `beta` | 1-D 偏移参数，长度等于 hidden_size |
| `eps` | 数值稳定项，默认 1e-6 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 与 x 形状、dtype 相同的归一化结果 |

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

#### 源码位置

- 算子实现：`applications/llm/layer_norm/arch35/layer_norm.asc`
- 测试文件：`tests/layer_norm/test_layer_norm.py`

---
### 2.70 lcm

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.lcm(Tensor a, Tensor b) -> Tensor
```

#### 功能

逐元素计算两个整数张量的最小公倍数。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `a` | 输入张量 a，与 b 形状、dtype 相同 |
| `b` | 输入张量 b，与 a 形状、dtype 相同 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 与输入形状、dtype 相同的逐元素最小公倍数 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| INT32 | `torch.int32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/lcm/arch35/lcm.asc`
- 测试文件：`tests/lcm/test_lcm.py`

---
### 2.71 leftshift

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.leftshift(Tensor a, int b) -> Tensor
```

#### 功能

逐元素按位左移，z = a << b（b 为标量，采用二进制补码回绕语义）。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `a` | 输入张量 |
| `b` | 左移位数，标量整数，取值范围 [0, 31] |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 与 a 形状、dtype 相同的左移结果 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| INT32 | `torch.int32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/leftshift/arch35/leftshift.asc`
- 测试文件：`tests/leftshift/test_leftshift.py`

---
### 2.72 legendre_polynomial_p

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.legendre_polynomial_p(Tensor x, Tensor n) -> Tensor
```

#### 功能

计算勒让德多项式 P_n(x)，n 为阶数（按元素广播）。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量，多项式自变量 |
| `n` | 阶数张量，与 x 广播；按元素取整作为多项式阶 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐元素的勒让德多项式值，形状为 x 与 n 广播后的形状 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/legendre_polynomial_p/arch35/legendre_polynomial_p.asc`
- 测试文件：`tests/legendre_polynomial_p/test_legendre_polynomial_p.py`

---
### 2.73 log_add_exp2

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.log_add_exp2(Tensor a, Tensor b) -> Tensor
```

#### 功能

逐元素计算 log2(2^a + 2^b)，采用数值稳定算法（提取较大项）避免上溢/下溢。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `a` | 输入张量 a，与 b 形状、dtype 相同 |
| `b` | 输入张量 b，与 a 形状、dtype 相同 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 与输入形状、dtype 相同的逐元素结果 |

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

#### 源码位置

- 算子实现：`applications/llm/log_add_exp2/arch35/log_add_exp2.asc`
- 测试文件：`tests/log_add_exp2/test_log_add_exp2.py`

---
### 2.74 log_normal

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.log_normal(Tensor x, float mean=1.0, float std=2.0, int seed=0) -> Tensor
```

#### 功能

按对数正态分布随机采样填充张量。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量（提供形状/dtype/device） |
| `mean` | 底层正态均值，默认 1.0 |
| `std` | 底层正态标准差，默认 2.0 |
| `seed` | 随机种子，默认 0 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，形状/dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP16 | `torch.float16` | ✅ |
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/log_normal/arch35/log_normal.asc`
- 测试文件：`tests/log_normal/test_log_normal.py`

---
### 2.75 logcumsumexp

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.logcumsumexp(Tensor x, int dim) -> Tensor
```

#### 功能

沿指定维度计算累积 log-sum-exp：`y[..., i, ...] = log(sum(exp(x[..., 0:i+1, ...])))`。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |
| `dim` | 扫描维度 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，形状/dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP16 | `torch.float16` | ✅ |
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/logcumsumexp/arch35/logcumsumexp.asc`
- 测试文件：`tests/logcumsumexp/test_logcumsumexp.py`

---
### 2.76 logicalxor

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.logicalxor(Tensor x, Tensor y) -> Tensor
```

#### 功能

逐元素逻辑异或，输出为 bool 张量；输入须为 bool 类型。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入 bool 张量，与 y 形状、dtype 相同 |
| `y` | 输入 bool 张量，与 x 形状、dtype 相同 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 与输入形状相同的 bool 异或结果 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| BOOL | `torch.bool` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/logicalxor/arch35/logicalxor.asc`
- 测试文件：`tests/logicalxor/test_logicalxor.py`

---
### 2.77 logit

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.logit(Tensor x, float? eps=None) -> Tensor
```

#### 功能

logit（对数几率）函数，logit(x) = ln(x / (1 - x))；给定 eps 时先将 x 截断到 [eps, 1-eps]。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量，取值通常在 (0, 1) 区间 |
| `eps` | 可选截断阈值，默认 None（不截断） |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 与 x 形状、dtype 相同的 logit 结果 |

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

#### 源码位置

- 算子实现：`applications/llm/logit/arch35/logit.asc`
- 测试文件：`tests/logit/test_logit.py`

---
### 2.78 logndtr

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.logndtr(Tensor x) -> Tensor
```

#### 功能

逐元素计算标准正态 CDF 的对数 log Φ(x)。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/logndtr/arch35/logndtr.asc`
- 测试文件：`tests/logndtr/test_logndtr.py`

---
### 2.79 lstm_cell

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.lstm_cell(Tensor input, Tensor hx, Tensor cx, Tensor weight_ih, Tensor weight_hh, Tensor? bias_ih=None, Tensor? bias_hh=None) -> (Tensor hy, Tensor cy)
```

#### 功能

单步 LSTM 单元前向，支持 1-D 非批和 2-D 批输入，输出新的隐藏状态与细胞状态。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `input` | 输入张量，[I] 或 [B, I] |
| `hx` | 上一时刻隐藏状态，[H] 或 [B, H] |
| `cx` | 上一时刻细胞状态，[H] 或 [B, H] |
| `weight_ih` | 输入到隐藏的权重，[4H, I] |
| `weight_hh` | 隐藏到隐藏的权重，[4H, H] |
| `bias_ih` | 可选输入偏置 [4H]，可为 None |
| `bias_hh` | 可选隐藏偏置 [4H]，可为 None |

#### 返回值

| 类型 | 说明 |
|------|------|
| (Tensor, Tensor) | (hy, cy)：新隐藏状态与新细胞状态，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP16 | `torch.float16` | ✅ |
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/lstm_cell/arch35/lstm_cell.asc`
- 测试文件：`tests/lstm_cell/test_lstm_cell.py`

---
### 2.80 make_per_tensor_quantized

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.make_per_tensor_quantized(Tensor x, float scale, int zero_point) -> Tensor
```

#### 功能

按 per-tensor 的 scale 与 zero_point 构造量化张量的整数表示。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |
| `scale` | 量化 scale |
| `zero_point` | 量化 zero_point |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，量化整数表示 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| INT32 | `torch.int32` | ✅ |
| INT8 | `torch.int8` | ✅ |
| UINT8 | `torch.uint8` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/make_per_tensor_quantized/arch35/make_per_tensor_quantized.asc`
- 测试文件：`tests/make_per_tensor_quantized/test_make_per_tensor_quantized.py`

---
### 2.81 matrix_exp_util

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.matrix_exp_util(Tensor input, Tensor coefficients) -> Tensor
```

#### 功能

矩阵指数辅助算子，对输入沿首维做线性组合：output[k, ...] = sum_j coefficients[k, j] * input[j, ...]。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `input` | 输入张量 [T, *S]，首维为待组合的项 |
| `coefficients` | 系数张量 [N, T]，与 input 同 dtype |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 形状为 [N, *S]、与 input 同 dtype 的线性组合结果 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP16 | `torch.float16` | ✅ |
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/matrix_exp_util/arch35/matrix_exp_util.asc`
- 测试文件：`tests/matrix_exp_util/test_matrix_exp_util.py`

---
### 2.82 max

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.max_dim(Tensor x, int dim, bool keepdim) -> (Tensor values, Tensor indices)
```

#### 功能

沿指定维度做最大值归约，返回最大值及其首次出现位置的索引。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |
| `dim` | 归约维度，支持负数索引 |
| `keepdim` | 是否保留被归约维度 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | values：沿 dim 的最大值；indices：对应的 argmax 索引（int64） |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/max/arch35/max.asc`
- 测试文件：`tests/max/test_max.py`

---
### 2.83 max_unpool2d

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.max_unpool2d(Tensor input, Tensor indices, int[2] kernel_size, int[2] stride, int[2] padding, int[2] output_size) -> Tensor
```

#### 功能

二维最大反池化，将池化输出按索引还原到更大空间尺寸，语义对齐 `torch.nn.functional.max_unpool2d`。

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | Tensor | 池化输出张量 |
| `indices` | Tensor | 与 `input` 同 shape 的 int64 索引 |
| `kernel_size` | int[2] | 最大池化窗口大小 |
| `stride` | int[2] | 步长 |
| `padding` | int[2] | 填充 |
| `output_size` | int[2] | 目标输出空间尺寸 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 反池化输出张量，dtype 与输入相同 |

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

#### 源码位置

- 算子实现：`applications/llm/max_unpool2d/arch35/max_unpool2d.asc`
- 测试文件：`tests/max_unpool2d/test_max_unpool2d.py`

---
### 2.84 max_unpool3d

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.max_unpool3d(Tensor input, Tensor indices, int[3] kernel_size, int[] stride, int[] padding, int[] output_size) -> Tensor
```

#### 功能

三维最大反池化，将池化输出按索引还原到更大空间尺寸，语义对齐 `torch.nn.functional.max_unpool3d`。

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | Tensor | 池化输出张量 |
| `indices` | Tensor | 与 `input` 同 shape 的 int64 索引 |
| `kernel_size` | int[3] | 最大池化窗口大小 |
| `stride` | int[] | 步长，可为空表示默认 |
| `padding` | int[] | 填充，可为空表示默认 |
| `output_size` | int[] | 目标输出空间尺寸，可为空表示默认 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 反池化输出张量，dtype 与输入相同 |

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

#### 源码位置

- 算子实现：`applications/llm/max_unpool3d/arch35/max_unpool3d.asc`
- 测试文件：`tests/max_unpool3d/test_max_unpool3d.py`

---
### 2.85 mean

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.mean(Tensor x, int dim, bool keepdim) -> Tensor
```

#### 功能

沿指定维度求均值，mean = reduce_sum(x, dim) / dimSize。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |
| `dim` | 归约维度，支持负数索引 |
| `keepdim` | 是否保留被归约维度 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 沿 dim 求均值后的张量 |

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

#### 源码位置

- 算子实现：`applications/llm/mean/arch35/mean.asc`
- 测试文件：`tests/mean/test_mean.py`

---
### 2.86 mode

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.mode(Tensor x, int dim, bool keepdim) -> (Tensor values, Tensor indices)
```

#### 功能

沿指定维度取众数及其索引。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |
| `dim` | 归约维度 |
| `keepdim` | 是否保留归约维度 |

#### 返回值

| 类型 | 说明 |
|------|------|
| (Tensor, Tensor) | (values, indices)：众数与对应索引 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP16 | `torch.float16` | ✅ |
| FP32 | `torch.float32` | ✅ |
| INT32 | `torch.int32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/mode/arch35/mode.asc`
- 测试文件：`tests/mode/test_mode.py`

---
### 2.87 modified_bessel_k0

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.modified_bessel_k0(Tensor x) -> Tensor
```

#### 功能

第二类零阶修正贝塞尔函数 K0(x)。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 与 x 形状、dtype 相同的 K0 结果 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/modified_bessel_k0/arch35/modified_bessel_k0.asc`
- 测试文件：`tests/modified_bessel_k0/test_modified_bessel_k0.py`

---
### 2.88 modified_bessel_k1

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.modified_bessel_k1(Tensor x) -> Tensor
```

#### 功能

第二类一阶修正贝塞尔函数 K1(x)。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 与 x 形状、dtype 相同的 K1 结果 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/modified_bessel_k1/arch35/modified_bessel_k1.asc`
- 测试文件：`tests/modified_bessel_k1/test_modified_bessel_k1.py`

---
### 2.89 mul

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.mul(Tensor x, Tensor y) -> Tensor
```

#### 功能

逐元素乘法，z = x * y。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 x，与 y 形状相同 |
| `y` | 输入张量 y，与 x 形状相同 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 与输入形状、dtype 相同的逐元素乘积 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |
| FP16 | `torch.float16` | ✅ |
| INT32 | `torch.int32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/mul/arch35/mul.asc`
- 测试文件：`tests/mul/test_mul.py`

---
### 2.90 multi_margin_loss

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.multi_margin_loss(Tensor input, Tensor target, Scalar p=1, Scalar margin=1.0, Tensor? weight=None, int reduction=1) -> Tensor
```

#### 功能

多分类 margin 损失（multi-class hinge loss）。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `input` | 输入分数张量 [N, C] |
| `target` | 类别标签 [N] |
| `p` | margin 幂次，默认 1 |
| `margin` | margin 值，默认 1.0 |
| `weight` | 可选逐类权重，可为 None |
| `reduction` | 归约方式 0=none/1=mean/2=sum，默认 1 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，损失值 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP16 | `torch.float16` | ✅ |
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/multi_margin_loss/arch35/multi_margin_loss.asc`
- 测试文件：`tests/multi_margin_loss/test_multi_margin_loss.py`

---
### 2.91 multilabel_margin_loss

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.multilabel_margin_loss(Tensor input, Tensor target, int reduction=1) -> Tensor
```

#### 功能

多标签 margin 损失（multilabel hinge loss）。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `input` | 输入分数张量 [N, C] |
| `target` | 多标签目标 [N, C] |
| `reduction` | 归约方式 0=none/1=mean/2=sum，默认 1 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，损失值 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP16 | `torch.float16` | ✅ |
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/multilabel_margin_loss/arch35/multilabel_margin_loss.asc`
- 测试文件：`tests/multilabel_margin_loss/test_multilabel_margin_loss.py`

---
### 2.92 multinomial

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.multinomial(Tensor x, int num_samples, bool replacement=False, int seed=0) -> Tensor
```

#### 功能

多项分布采样，按每行的非负权重作为概率分布抽取类别索引。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 每行非负权重，1-D (K,) 或 2-D (numDist, K)，最后一维为类别轴 |
| `num_samples` | 每个分布抽取的样本数，须 > 0 |
| `replacement` | 是否有放回采样，默认 False |
| `seed` | 随机数种子，默认 0 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | int64 类别索引；1-D 输入返回 (num_samples,)，2-D 输入返回 (numDist, num_samples) |

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

#### 源码位置

- 算子实现：`applications/llm/multinomial/arch35/multinomial.asc`
- 测试文件：`tests/multinomial/test_multinomial.py`

---
### 2.93 nested_add_pad

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.nested_add_pad(Tensor values, Tensor offsets, int max_L, Scalar padding_value=0) -> Tensor
```

#### 功能

嵌套张量加 padding：将压缩格式 (sum_L, D) 按 offsets 还原为零填充的稠密张量 (B, max_L, D)。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `values` | 2-D (sum_L, D) 压缩值，contiguous |
| `offsets` | 1-D (B+1,) int64 偏移，非递减，offsets[0]=0、offsets[B]=sum_L |
| `max_L` | 输出第二维长度，须 >= 各 batch 的最大有效长度 |
| `padding_value` | 填充标量值，默认 0，转换为 values 的 dtype |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 3-D (B, max_L, D)、与 values 同 dtype 的零填充稠密张量 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |
| FP16 | `torch.float16` | ✅ |
| INT32 | `torch.int32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/nested_add_pad/arch35/nested_add_pad.asc`
- 测试文件：`tests/nested_add_pad/test_nested_add_pad.py`

---
### 2.94 nested_binary_op

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.nested_binary_op(Tensor values, Tensor offsets, Tensor dense, int op_mode) -> Tensor
```

#### 功能

对嵌套（不规则）张量与 dense 张量执行逐元素二元广播运算，语义对齐 PyTorch CUDA `op_dense_esuhm`。设批次数为 B，`offsets` 给出每个批次在 `values` 行方向的起止；对每个批次 i，将 `dense[i, 0, :]` 广播到该批次的所有行：`op_mode == 0` 为逐元素加，`op_mode == 1` 为逐元素乘。

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `values` | Tensor | 嵌套张量的值，2-D `[sum_L, D]`，支持 FP32、FP16、INT32 |
| `offsets` | Tensor | 批次行偏移，1-D `[B+1]`，int64，非递减，`offsets[0]=0`、`offsets[B]=sum_L` |
| `dense` | Tensor | dense 张量，3-D `[B, 1, D]`，dtype 与 `values` 相同 |
| `op_mode` | int | 运算模式：0 = 逐元素加，1 = 逐元素乘 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 输出张量，shape 与 `values` 相同 `[sum_L, D]`，dtype 与输入相同 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |
| FP16 | `torch.float16` | ✅ |
| INT32 | `torch.int32` | ✅ |

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

# 2 个批次，行数分别为 2、3，特征维 D=4
values = torch.randn(5, 4, dtype=torch.float32).npu()
offsets = torch.tensor([0, 2, 5], dtype=torch.int64).npu()
dense = torch.randn(2, 1, 4, dtype=torch.float32).npu()

# op_mode=0：逐批次将 dense 广播加到该批次所有行
result = torch.ops.ops_multimodal_fusion.nested_binary_op(values, offsets, dense, 0)
print(result.shape)   # torch.Size([5, 4])
```

#### 源码位置

- 算子实现：`applications/llm/nested_binary_op/arch35/nested_binary_op.asc`
- 测试文件：`tests/nested_binary_op/test_nested_binary_op.py`

---
### 2.95 nested_bmm

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.nested_bmm(Tensor a_values, Tensor b_values, Tensor sizes) -> Tensor
```

#### 功能

对嵌套张量执行批量矩阵乘（batched matrix multiply），按 sizes 描述的每个子矩阵分块计算 a 与 b 的乘积。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `a_values` | 左操作数的展平 values 张量 |
| `b_values` | 右操作数的展平 values 张量 |
| `sizes` | 描述各子矩阵形状的尺寸张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 各子矩阵乘积拼接而成的展平 values 张量 |

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

#### 源码位置

- 算子实现：`applications/llm/nested_bmm/arch35/nested_bmm.asc`
- 测试文件：`tests/nested_bmm/test_nested_bmm.py`

---
### 2.96 nested_remove_pad

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.nested_remove_pad(Tensor padded, Tensor lengths) -> Tensor
```

#### 功能

将带 padding 的批量张量按各样本的实际长度移除 padding，得到展平的嵌套张量 values。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `padded` | 含 padding 的输入张量 |
| `lengths` | 各样本实际有效长度 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 移除 padding 后的展平张量 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |
| FP16 | `torch.float16` | ✅ |
| INT32 | `torch.int32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/nested_remove_pad/arch35/nested_remove_pad.asc`
- 测试文件：`tests/nested_remove_pad/test_nested_remove_pad.py`

---
### 2.97 nextafter

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.nextafter(Tensor a, Tensor b) -> Tensor
```

#### 功能

逐元素返回从 a 朝 b 方向的下一个可表示浮点数。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `a` | 起始值张量 |
| `b` | 目标方向张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐元素的下一个浮点数，shape 与输入相同 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/nextafter/arch35/nextafter.asc`
- 测试文件：`tests/nextafter/test_nextafter.py`

---
### 2.98 norm

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.norm(Tensor x, Scalar p, int dim, bool keepdim) -> Tensor
```

#### 功能

沿指定维度计算 p 范数归约。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |
| `p` | 范数阶数 |
| `dim` | 归约维度 |
| `keepdim` | 是否保留归约维度 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 沿 dim 归约后的范数张量 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/norm/arch35/norm.asc`
- 测试文件：`tests/norm/test_norm.py`

---
### 2.99 pdist

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.pdist(Tensor input, float p=2.0) -> Tensor
```

#### 功能

计算输入各行之间的成对 p-范数距离，输出顺序对齐 `torch.nn.functional.pdist`（上三角行主序）。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `input` | 输入张量，形状 [N, D] |
| `p` | 范数阶数，默认 2.0 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，形状 [N * (N - 1) / 2]，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |
| FP16 | `torch.float16` | ✅ |
| BF16 | `torch.bfloat16` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/pdist/arch35/pdist.asc`
- 测试文件：`tests/pdist/test_pdist.py`

---
### 2.100 pdist_backward

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.pdist_backward(Tensor grad, Tensor input, float p, Tensor pdist_output) -> Tensor
```

#### 功能

计算 `pdist` 关于输入的梯度。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `grad` | 距离向量的上游梯度，[N * (N - 1) / 2] |
| `input` | 前向输入，形状 [N, D] |
| `p` | 范数阶数 |
| `pdist_output` | 前向输出的成对距离，[N * (N - 1) / 2] |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | input 的梯度，形状 [N, D]，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |
| FP16 | `torch.float16` | ✅ |
| BF16 | `torch.bfloat16` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/pdist_backward/arch35/pdist_backward.asc`
- 测试文件：`tests/pdist_backward/test_pdist_backward.py`

---
### 2.101 poisson

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.poisson(Tensor x, int seed=0) -> Tensor
```

#### 功能

以输入逐元素作为速率 lambda，按泊松分布 `Poisson(x)` 随机采样，语义等价 `torch.poisson`。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 速率张量 lambda（须非负） |
| `seed` | 随机种子，默认 0 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，形状/dtype 同输入，填充泊松分布样本 |

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

#### 源码位置

- 算子实现：`applications/llm/poisson/arch35/poisson.asc`
- 测试文件：`tests/poisson/test_poisson.py`

---
### 2.102 polar

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.polar(Tensor abs, Tensor angle) -> Tensor
```

#### 功能

由极坐标的模与辐角构造复数张量：`out = abs * (cos(angle) + sin(angle) * 1j)`。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `abs` | 模张量（非负实数） |
| `angle` | 辐角张量（弧度） |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 复数张量；fp16 输入 → complex32，fp32 输入 → complex64 |

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

#### 源码位置

- 算子实现：`applications/llm/polar/arch35/polar.asc`
- 测试文件：`tests/polar/test_polar.py`

---
### 2.103 polygamma

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.polygamma(Tensor x, int n) -> Tensor
```

#### 功能

逐元素计算 n 阶多伽马函数。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |
| `n` | 阶数（n≥0） |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/polygamma/arch35/polygamma.asc`
- 测试文件：`tests/polygamma/test_polygamma.py`

---
### 2.104 put

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.put(Tensor self, Tensor index, Tensor source, bool accumulate=False) -> Tensor
```

#### 功能

将 self 视为一维平铺，按线性索引写入或累加 source：`accumulate=False` 时赋值，`accumulate=True` 时累加。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `self` | 目标张量（任意维度，按平铺处理） |
| `index` | 线性索引张量（int64） |
| `source` | 待写入的源数据 |
| `accumulate` | 是否累加，默认 False |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，形状/dtype 同 self |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |
| FP16 | `torch.float16` | ✅ |
| INT32 | `torch.int32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/put/arch35/put.asc`
- 测试文件：`tests/put/test_put.py`

---
### 2.105 quantized_relu

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.quantized_relu(Tensor x, int zero_point) -> Tensor
```

#### 功能

对量化整型张量执行 ReLU，按 zero_point 截断（小于 zero_point 的值置为 zero_point）。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 量化整型输入张量 |
| `zero_point` | 量化零点 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 量化 ReLU 结果，dtype 与输入相同 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| INT8 | `torch.int8` | ✅ |
| UINT8 | `torch.uint8` | ✅ |
| INT32 | `torch.int32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/quantized_relu/arch35/quantized_relu.asc`
- 测试文件：`tests/quantized_relu/test_quantized_relu.py`

---
### 2.106 rms_norm_gated

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.rms_norm_gated(Tensor hidden_states, Tensor gate, Tensor gamma, float epsilon=1e-6) -> Tensor
```

#### 功能

带门控的 RMS 归一化：先对 hidden_states 做 RMS 归一化并乘以 gamma，再用 gate 进行门控。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `hidden_states` | 待归一化的输入张量 |
| `gate` | 门控张量 |
| `gamma` | 缩放权重 |
| `epsilon` | 数值稳定项，默认 1e-6 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 门控 RMS 归一化结果 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |
| FP16 | `torch.float16` | ✅ |
| BF16 | `torch.bfloat16` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/rms_norm_gated/arch35/rms_norm_gated.asc`
- 测试文件：`tests/rms_norm_gated/test_rms_norm_gated.py`

---
### 2.107 rsqrt

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.rsqrt(Tensor x) -> Tensor
```

#### 功能

逐元素计算平方根的倒数：`y = 1 / sqrt(x)`。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐元素平方根倒数，shape 与输入相同 |

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

#### 源码位置

- 算子实现：`applications/llm/rsqrt/arch35/rsqrt.asc`
- 测试文件：`tests/rsqrt/test_rsqrt.py`

---
### 2.108 scaled_modified_bessel_k0

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.scaled_modified_bessel_k0(Tensor x) -> Tensor
```

#### 功能

逐元素计算缩放的第二类零阶修正贝塞尔函数：`y = exp(x) * K0(x)`。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐元素结果，shape 与输入相同 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/scaled_modified_bessel_k0/arch35/scaled_modified_bessel_k0.asc`
- 测试文件：`tests/scaled_modified_bessel_k0/test_scaled_modified_bessel_k0.py`

---
### 2.109 scaled_modified_bessel_k1

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.scaled_modified_bessel_k1(Tensor x) -> Tensor
```

#### 功能

逐元素计算缩放的第二类一阶修正贝塞尔函数：`y = exp(x) * K1(x)`。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐元素结果，shape 与输入相同 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/scaled_modified_bessel_k1/arch35/scaled_modified_bessel_k1.asc`
- 测试文件：`tests/scaled_modified_bessel_k1/test_scaled_modified_bessel_k1.py`

---
### 2.110 searchsorted

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.searchsorted(Tensor sorted_sequence, Tensor values, bool out_int32=False, bool right=False) -> Tensor
```

#### 功能

在有序序列中二分查找各 value 的插入位置。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `sorted_sequence` | 升序有序序列 |
| `values` | 待查找的值 |
| `out_int32` | 输出索引是否用 int32，默认 False(int64) |
| `right` | 相等时取右侧边界，默认 False |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，插入位置索引 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| INT64 | `torch.int64` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/searchsorted/arch35/searchsorted.asc`
- 测试文件：`tests/searchsorted/test_searchsorted.py`

---
### 2.111 shifted_chebyshev_polynomial_t

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.shifted_chebyshev_polynomial_t(Tensor x, Tensor n) -> Tensor
```

#### 功能

逐元素计算移位第一类切比雪夫多项式 T*_n(x)。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 自变量张量 |
| `n` | 阶数张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/shifted_chebyshev_polynomial_t/arch35/shifted_chebyshev_polynomial_t.asc`
- 测试文件：`tests/shifted_chebyshev_polynomial_t/test_shifted_chebyshev_polynomial_t.py`

---
### 2.112 shifted_chebyshev_polynomial_u

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.shifted_chebyshev_polynomial_u(Tensor x, Tensor n) -> Tensor
```

#### 功能

逐元素计算移位第二类切比雪夫多项式 U*_n(x)。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 自变量张量 |
| `n` | 阶数张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/shifted_chebyshev_polynomial_u/arch35/shifted_chebyshev_polynomial_u.asc`
- 测试文件：`tests/shifted_chebyshev_polynomial_u/test_shifted_chebyshev_polynomial_u.py`

---
### 2.113 shifted_chebyshev_polynomial_v

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.shifted_chebyshev_polynomial_v(Tensor x, Tensor n) -> Tensor
```

#### 功能

逐元素计算移位第三类切比雪夫多项式 V*_n(x)。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 自变量张量 |
| `n` | 阶数张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/shifted_chebyshev_polynomial_v/arch35/shifted_chebyshev_polynomial_v.asc`
- 测试文件：`tests/shifted_chebyshev_polynomial_v/test_shifted_chebyshev_polynomial_v.py`

---
### 2.114 shifted_chebyshev_polynomial_w

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.shifted_chebyshev_polynomial_w(Tensor x, Tensor n) -> Tensor
```

#### 功能

逐元素计算移位第四类切比雪夫多项式 W*_n(x)。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 自变量张量 |
| `n` | 阶数张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/shifted_chebyshev_polynomial_w/arch35/shifted_chebyshev_polynomial_w.asc`
- 测试文件：`tests/shifted_chebyshev_polynomial_w/test_shifted_chebyshev_polynomial_w.py`

---
### 2.115 sigmoid

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.sigmoid(Tensor x) -> Tensor
```

#### 功能

逐元素 Sigmoid 激活：`y = 1 / (1 + exp(-x))`。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Sigmoid 结果，shape 与输入相同 |

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

#### 源码位置

- 算子实现：`applications/llm/sigmoid/arch35/sigmoid.asc`
- 测试文件：`tests/sigmoid/test_sigmoid.py`

---
### 2.116 sin

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.sin(Tensor x) -> Tensor
```

#### 功能

逐元素计算正弦：`y = sin(x)`。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量（弧度） |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐元素正弦，shape 与输入相同 |

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

#### 源码位置

- 算子实现：`applications/llm/sin/arch35/sin.asc`
- 测试文件：`tests/sin/test_sin.py`

---
### 2.117 sinc

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.sinc(Tensor x) -> Tensor
```

#### 功能

逐元素计算归一化 sinc 函数：`sinc(x) = sin(πx) / (πx)`，且 `sinc(0) = 1`，语义对齐 `torch.sinc`。

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | Tensor | 输入张量，支持 FP32、FP16 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 输出张量，shape 与 dtype 同输入 |

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

#### 源码位置

- 算子实现：`applications/llm/sinc/arch35/sinc.asc`
- 测试文件：`tests/sinc/test_sinc.py`

---
### 2.118 spherical_bessel_j0

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.spherical_bessel_j0(Tensor x) -> Tensor
```

#### 功能

逐元素计算零阶球贝塞尔函数：`y = sin(x) / x`（x=0 时为 1）。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐元素结果，shape 与输入相同 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/spherical_bessel_j0/arch35/spherical_bessel_j0.asc`
- 测试文件：`tests/spherical_bessel_j0/test_spherical_bessel_j0.py`

---
### 2.119 sqrt

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.sqrt(Tensor x) -> Tensor
```

#### 功能

逐元素计算平方根：`y = sqrt(x)`。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 逐元素平方根，shape 与输入相同 |

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

#### 源码位置

- 算子实现：`applications/llm/sqrt/arch35/sqrt.asc`
- 测试文件：`tests/sqrt/test_sqrt.py`

---
### 2.120 sum

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.sum(Tensor x, int dim, bool keepdim) -> Tensor
```

#### 功能

沿指定维度求和归约。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量 |
| `dim` | 归约维度 |
| `keepdim` | 是否保留归约维度 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 沿 dim 求和后的张量 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |
| FP16 | `torch.float16` | ✅ |
| INT32 | `torch.int32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/sum/arch35/sum.asc`
- 测试文件：`tests/sum/test_sum.py`

---
### 2.121 swi_glu

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.swi_glu(Tensor x) -> Tensor
```

#### 功能

SwiGLU 激活：将输入沿最后一维对半切分为 a、b，计算 `y = silu(a) * b`。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 输入张量，最后一维为偶数 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | SwiGLU 结果，最后一维为输入的一半 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |
| FP16 | `torch.float16` | ✅ |
| BF16 | `torch.bfloat16` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/swi_glu/arch35/swi_glu.asc`
- 测试文件：`tests/swi_glu/test_swi_glu.py`

---
### 2.122 take

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.take(Tensor self, Tensor index) -> Tensor
```

#### 功能

将 self 视为一维平铺，按线性索引取值：`result[i] = self.view(-1)[index[i]]`，输出形状与 index 相同。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `self` | 输入张量（任意维度，按平铺处理） |
| `index` | 线性索引张量（int64） |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，形状同 index，dtype 同 self |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |
| FP16 | `torch.float16` | ✅ |
| INT32 | `torch.int32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/take/arch35/take.asc`
- 测试文件：`tests/take/test_take.py`

---
### 2.123 tril_indices

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.tril_indices(int row, int col, int offset, bool out_int32) -> Tensor
```

#### 功能

返回 row×col 矩阵下三角部分元素的行列索引。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `row` | 矩阵行数 |
| `col` | 矩阵列数 |
| `offset` | 对角线偏移 |
| `out_int32` | 输出索引是否用 int32 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，形状 [2, K] 的行列索引 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| INT32 | `torch.int32` | ✅ |
| INT64 | `torch.int64` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/tril_indices/arch35/tril_indices.asc`
- 测试文件：`tests/tril_indices/test_tril_indices.py`

---
### 2.124 triu_indices

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.triu_indices(int row, int col, int offset, bool out_int32) -> Tensor
```

#### 功能

返回 row×col 矩阵上三角部分元素的行列索引。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `row` | 矩阵行数 |
| `col` | 矩阵列数 |
| `offset` | 对角线偏移 |
| `out_int32` | 输出索引是否用 int32 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，形状 [2, K] 的行列索引 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| INT32 | `torch.int32` | ✅ |
| INT64 | `torch.int64` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/triu_indices/arch35/triu_indices.asc`
- 测试文件：`tests/triu_indices/test_triu_indices.py`

---
### 2.125 unpack_pivots

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.unpack_pivots(Tensor pivots, int perm_size) -> Tensor
```

#### 功能

将 LU 分解的 pivot 索引展开为完整的行置换序列。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `pivots` | LU 分解的 pivot 索引张量 |
| `perm_size` | 置换序列长度 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | 展开后的行置换索引（INT64） |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| INT32 | `torch.int32` | ✅ |
| INT64 | `torch.int64` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/unpack_pivots/arch35/unpack_pivots.asc`
- 测试文件：`tests/unpack_pivots/test_unpack_pivots.py`

---
### 2.126 upsample_linear1d

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
### 2.127 upsample_nearest1d

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.upsample_nearest1d(Tensor input, int output_size, float scale=-1.) -> Tensor
```

#### 功能

对 3-D 张量 `[N, C, W]` 在宽度维上做一维最近邻上/下采样：`output[n, c, ow] = input[n, c, floor(ow * source_scale)]`。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `input` | 输入张量，形状 [N, C, W] |
| `output_size` | 输出宽度 OW |
| `scale` | 缩放因子，默认 -1.（由 output_size 推导） |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，形状 [N, C, output_size]，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP16 | `torch.float16` | ✅ |
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/upsample_nearest1d/arch35/upsample_nearest1d.asc`
- 测试文件：`tests/upsample_nearest1d/test_upsample_nearest1d.py`

---
### 2.128 upsample_trilinear3d

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.upsample_trilinear3d(Tensor input, int[3] output_size, bool align_corners=False, float scales_d=-1., float scales_h=-1., float scales_w=-1.) -> Tensor
```

#### 功能

三维三线性插值上/下采样，语义对齐 torch.nn.functional.interpolate(mode='trilinear')。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `input` | 输入张量 [N, C, D, H, W] |
| `output_size` | 目标输出空间尺寸 [outD, outH, outW] |
| `align_corners` | align_corners 标志，默认 False |
| `scales_d` | D 维缩放因子，默认 -1.（由 output_size 推导） |
| `scales_h` | H 维缩放因子，默认 -1. |
| `scales_w` | W 维缩放因子，默认 -1. |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，形状 [N, C, outD, outH, outW]，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP16 | `torch.float16` | ✅ |
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/upsample_trilinear3d/arch35/upsample_trilinear3d.asc`
- 测试文件：`tests/upsample_trilinear3d/test_upsample_trilinear3d.py`

---
### 2.129 weight_norm

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.weight_norm(Tensor v, Tensor g, int dim) -> Tensor
```

#### 功能

权重归一化：沿除 `dim` 外的维度求 v 的范数，重标定为 `out = v * g / norm`。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `v` | 方向张量 |
| `g` | 幅度张量，长度为 `v.size(dim)` |
| `dim` | 保留（不归约）的维度 |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，形状/dtype 同 v |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP16 | `torch.float16` | ✅ |
| FP32 | `torch.float32` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/weight_norm/arch35/weight_norm.asc`
- 测试文件：`tests/weight_norm/test_weight_norm.py`

---
### 2.130 zeta

#### 接口签名

```python
torch.ops.ops_multimodal_fusion.zeta(Tensor x, Tensor q) -> Tensor
```

#### 功能

逐元素计算 Hurwitz zeta 函数：`y = zeta(x, q) = sum_{k=0}^{inf} (q + k)^(-x)`，有效域 `x > 1` 且 `q > 0`。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `x` | 指数张量（x > 1） |
| `q` | 偏移张量（q > 0） |

#### 返回值

| 类型 | 说明 |
|------|------|
| Tensor | Tensor，dtype 同输入 |

#### 支持的数据类型

| 数据类型 | PyTorch 类型 | 支持状态 |
|----------|--------------|----------|
| FP32 | `torch.float32` | ✅ |
| FP16 | `torch.float16` | ✅ |
| BF16 | `torch.bfloat16` | ✅ |

#### 支持的芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

#### 源码位置

- 算子实现：`applications/llm/zeta/arch35/zeta.asc`
- 测试文件：`tests/zeta/test_zeta.py`

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
