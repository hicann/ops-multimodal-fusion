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
### 2.3 angle

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
### 2.4 bessel_j0

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
### 2.5 bessel_j1

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
### 2.6 bessel_y0

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
### 2.7 bessel_y1

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
### 2.8 cauchy

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
### 2.9 cdist_backward

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
### 2.10 complex

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
### 2.11 conjphysical

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
### 2.12 cummax

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
### 2.13 cumprod

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
### 2.14 depthwise_conv3d

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
### 2.15 digamma

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
### 2.16 entr

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
### 2.17 erfcx

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
### 2.18 exp2

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
### 2.19 exponential

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
### 2.20 fft_conj_symmetry

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
### 2.21 foreach_ceil

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
### 2.22 fractional_max_pool2d

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
### 2.23 fractional_max_pool3d

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
### 2.24 frexp

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
### 2.25 geometric

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
### 2.26 gru_cell

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
### 2.27 index_copy

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
### 2.28 index_reduce

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
### 2.29 int_repr

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
### 2.30 kthvalue

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
### 2.31 log_normal

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
### 2.32 logcumsumexp

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
### 2.33 logndtr

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
### 2.34 lstm_cell

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
### 2.35 make_per_tensor_quantized

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
### 2.36 max_unpool2d

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
### 2.37 max_unpool3d

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
### 2.38 mode

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
### 2.39 multi_margin_loss

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
### 2.40 multilabel_margin_loss

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
### 2.41 nested_binary_op

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
### 2.42 pdist

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
### 2.43 pdist_backward

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
### 2.44 poisson

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
### 2.45 polar

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
### 2.46 polygamma

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
### 2.47 put

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
### 2.48 searchsorted

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
### 2.49 shifted_chebyshev_polynomial_t

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
### 2.50 shifted_chebyshev_polynomial_u

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
### 2.51 shifted_chebyshev_polynomial_v

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
### 2.52 shifted_chebyshev_polynomial_w

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
### 2.53 take

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
### 2.54 tril_indices

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
### 2.55 triu_indices

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
### 2.56 upsample_linear1d

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
### 2.57 upsample_nearest1d

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
### 2.58 upsample_trilinear3d

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
### 2.59 weight_norm

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
### 2.60 zeta

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
