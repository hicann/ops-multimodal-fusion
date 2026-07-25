# addmv

> 面向使用者的算子文档。依据算子代码 `applications/llm/addmv/arch35/addmv.asc` 与需求真值源（`1.1-需求分析.md` v8.6）编写。

## 算子概述

`addmv` 是 BLAS Level-2 `gemv`（矩阵-向量乘 + 偏置累加）等价算子，按行对 `mat` 与 `vec` 做点积并与偏置向量 `self` 做轴向缩放加：

```
out = beta * self + alpha * (mat @ vec)

out[i] = beta * self[i] + alpha * Σ_{j=0}^{K-1} ( mat[i][j] * vec[j] ) ,  i ∈ [0, M)
```

其中 `self` 形状 `(M,)`、`mat` 形状 `(M, K)`、`vec` 形状 `(K,)`、`out` 形状 `(M,)`。

与 cuBLAS `gemv`（`y = alpha * op(A) * x + beta * y`）功能等价：本算子取 `op(A) = A`（不转置），并以**非 in-place** 形式返回新张量（`torch.addmv` 口径）。

> **重要**：本算子对齐 **BLAS/cuBLAS `gemv` 规范**，而非 `torch.addmv` 的实现行为，**不保证与 `torch.addmv` 逐位等价**。在所有场景下本算子结果 **≥ `torch.addmv`**（见[与 `torch.addmv` 的差异声明](#与-torchaddmv-的差异声明d1d4)：D3/D4 上本算子更精确），且全部差异仅在特定条件下可观测。常规有限值输入（不触发 D1–D4）下，结果与 `torch.addmv` 在精度阈值内一致。

## 接口列表

| 接口名 | 说明 |
|--------|------|
| `torch.ops.ops_multimodal_fusion.addmv` | 矩阵-向量乘 + 偏置累加（BLAS `gemv` 等价），含 K 维归约 |

## 产品支持情况

| 芯片系列 | 支持情况 |
|---------|---------|
| Ascend910B | ❌ |
| Ascend910_93 | ❌ |
| Ascend950 | ✅ |

> 单架构交付：仅 `arch35`（dav-3510 / Ascend950），不出 `arch22`。

## 接口详情

### `torch.ops.ops_multimodal_fusion.addmv`

#### 函数原型

```python
torch.ops.ops_multimodal_fusion.addmv(self, mat, vec, beta=1, alpha=1) -> Tensor
```

Schema（与 `.asc` 内 `TORCH_LIBRARY_FRAGMENT` 声明一致）：

```
addmv(Tensor self, Tensor mat, Tensor vec, Scalar beta=1, Scalar alpha=1) -> Tensor
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| `self` | 输入 | Tensor | 偏置向量 `(M,)`；或可广播到 `(M,)` 的三种 stride-0 形态：`(1,)`、0-dim、`(M,)` 的 expand view | NPU |
| `mat` | 输入 | Tensor | 矩阵 `(M, K)` | NPU |
| `vec` | 输入 | Tensor | 向量 `(K,)`，要求 `vec.size(0) == mat.size(1) == K` | NPU |
| `beta` | 输入 | Scalar，默认 `1` | `self` 的缩放系数；**精确为 0** 时忽略 `self`（不读取其值） | — |
| `alpha` | 输入 | Scalar，默认 `1` | 矩阵向量积的缩放系数；**精确为 0** 时不引用 `mat`/`vec`（BLAS 语义） | — |
| `out` | 输出 | Tensor | 输出向量 `(M,)`，dtype/device 与输入一致，contiguous | NPU |

#### 支持的数据类型

`self`、`mat`、`vec` 三者 dtype 必须一致，且属于下列支持列表：

| 数据类型 | PyTorch 类型 | 支持状态 | 说明 |
|----------|--------------|----------|------|
| FP32 | `torch.float32` | ✅ | 基础必备 |
| FP16 | `torch.float16` | ✅ | 内部走 fp32 中间累加以控制 K 维累加误差 |
| BF16 | `torch.bfloat16` | ✅ | 内部走 fp32 中间累加 |
| INT32 | `torch.int32` | ✅ | 位精确；`beta`/`alpha` 按向零截断取整（见约束说明） |

#### 支持的 shape

| 维度 | 上界 |
|------|------|
| `M` | ≤ 65536 |
| `K` | ≤ 65536 |
| 元素总量 | `M × K ≤ 2^26` |

- **广播 `self`**：支持 `(1,)`、0-dim、`(M,)` 的 expand view 三种 stride-0 形态，等价于 `self` 的唯一元素作用到全部 M 行。
- **空张量**：`M == 0` 时输出为空张量 `(0,)`；`K == 0` 时矩阵向量积为 0，`out = beta * self`。
- **超上界不拦截**：超出上表范围的输入按同一路径执行，但**不承诺其正确性与性能**（属"未验证范围"而非"非法输入"，不做 `TORCH_CHECK` 拦截，以避免与 `torch.addmv` 制造行为差异）。

#### 约束说明

| 约束 | 说明 |
|------|------|
| dtype 一致 | `self`、`mat`、`vec` 三者 dtype 必须相同，否则 host 侧 `TORCH_CHECK` 拒绝 |
| device | 三者必须同为 NPU 设备（`PrivateUse1`），否则 host 侧 `TORCH_CHECK` 拒绝 |
| 形状 | `mat` 为 2-D；`vec` 为 1-D 且 `vec.size(0) == mat.size(1)`；`self` 为 0-D 或 1-D 且 `numel` 为 `1` 或 `M` |
| int32 越界标量 | `beta`/`alpha` 超出 `[INT32_MIN, INT32_MAX]` 时 host 侧 `TORCH_CHECK` **拒绝**。精确谓词：设 `v = scalar.to<double>()`，当 `v < -2147483648.0 \|\| v > 2147483647.0` 时拒绝（比较在向零截断**之前**、于 double 域完成，闭区间接受）。`2147483647.5` 这类既非整数又越界的值走**拒绝**（非截断）；`INT32_MIN = -2147483648` 为合法值 |
| int32 in-range 非整数标量 | **不报错**，按向零截断取整（`0.5→0`、`2.5→2`、`-0.5→0`），对齐 `torch.addmv` 实测行为 |
| int32 算术溢出 | `alpha * (mat @ vec)` 及累加溢出按**二补数回绕**，不做饱和也不做检测（与 `torch.addmv` 一致） |
| 标量精度 | `beta`/`alpha` **按计算精度转换**：fp16/bf16 输出时按 **fp32** 承载、fp32/int32 输出时按其自身，**不先降到输出 dtype**（保证不在标量这一处丢失全流程 fp32 累加所保的精度） |
| 判零口径 | 是否短路只看用户传入的**原始 Scalar 是否精确为 0**，与 dtype、shape 均无关（见差异声明 D1/D2） |
| 输出 | 新建 Tensor（非 in-place），shape `(M,)`，contiguous |

## 与 `torch.addmv` 的差异声明（D1–D4）

本算子对齐 **BLAS/cuBLAS `gemv` 规范**而非 `torch.addmv` 的实现行为，以下 4 条差异均为主动选择、非缺陷。

- **总述**：不得宣称与 `torch.addmv` 逐位等价；但**在所有场景 ≥ `torch.addmv`**（D3/D4 上本算子更精确），全部差异仅在特定条件下可观测。
- **共同保证**：本算子是否短路**只取决于用户传入的标量是否精确为 0**，与 dtype、shape 均无关。
- **可观测性分两类**：D1、D2 仅在相关张量含 NaN/Inf 时可观测；**D3、D4 在全有限值输入下即可观测**，且这两条上本算子与高精度参考一致、`torch.addmv` 不一致（本算子是更准的一方）。

| # | 触发条件 | 本算子 | `torch.addmv` | 可观测性 |
|---|---------|--------|--------------|---------|
| **D1** | `alpha` **精确为 0** 且 `mat`/`vec` 含 NaN/Inf | 统一短路，`out = beta * self`，不引用 `mat`/`vec`，NaN/Inf **不传播** | 行为随 dtype/shape 不稳定：fp32 短路、fp16 不短路、bf16 以 `M×K = 4096` 为界翻转。**且 `bf16 ∧ M×K > 4096 ∧ alpha=0` 一格 `torch` 返回非确定性未初始化内存（连跑多次得不同值），不可信、不可作参考基准** | 仅含 NaN/Inf 时 |
| **D2** | 标量非 0 但 `\|scalar\| ≤ 2^-150`（fp32 下亦量化为 0，`≈ 7.0065e-46`，闭区间）且对应张量（`beta`→`self`，`alpha`→`mat`/`vec`）含 NaN/Inf | 按**原始 Scalar** 判零，故**不短路**，NaN/Inf 经 `0 * NaN` **传播** | 按其工作精度下的量化值判零，故**短路、不传播** | 仅含 NaN/Inf 时 |
| **D3** | 标量落 `torch` 工作精度吞为 0、但 fp32 下非 0 的区间：fp16 为 `(2^-150, 2^-25]`（`≈ (7.0065e-46, 2.98023e-8]`）；bf16 为 `(2^-150, 2^-134]`（`≈ (7.0065e-46, 4.59177e-41]`）且 `M×K ≤ 4096` | 标量按 fp32 承载、**真实生效**，与高精度参考一致。**本算子更精确** | 标量被吞为 0，该项整体归零 | **有限值输入即可观测** |
| **D4** | `bf16 ∧ M×K > 4096` 且 fp32 真值落入 bf16 次正规区 `[2^-133, 2^-126)`（`≈ [9.18e-41, 1.18e-38)`） | 保留次正规值（**已上板确认：NPU 降位 `Cast<bf16,float>` 不 flush 次正规**）。**本算子更精确** | 内核对 bf16 输出做**次正规冲零**，得 `0` | **有限值输入即可观测** |

> D4 的**不适用**范围（勿写成 bf16 通用或跨 dtype 通用）：fp16 任意 shape、fp32 任意 shape、**bf16 且 `M×K ≤ 4096`** 均**不冲零**（本算子与 `torch` 一致）。

## 环境限制说明

- **非连续（非 contiguous）输入**：本算子按正确语义处理——Host 侧对 `mat`/`vec`/非 stride-0 的 `self` 做 `.contiguous()` 归一后再下发。但当前环境 `torch_npu` **缺 D2D strided copy**（触发 `RuntimeError ... error code is 561103`），故非连续路径**暂无法上板验证**，相关测试用例标记 `skip`。**平台修复 D2D strided copy 后，算子侧无需任何改动即自动生效**。因此**不得宣称"非连续输入不报错"**。
- **广播 `self` 不受影响**：三种 stride-0 广播形态（`(1,)`、0-dim、`(M,)` 的 expand view）走 kernel 内 stride-0 直读，**无 D2D 拷贝**，不受 561103 影响，可正常上板。
- 构造 `(M,)` 的 expand view 时须在 **NPU 侧** expand（如 `x.npu().expand(M)`）；`x.expand(M).npu()` 会在 CPU 侧把视图物化为连续张量。

## 精度对齐说明

- **标杆（golden）**：PyTorch CPU `torch.addmv`。
- fp16/bf16 内部走 fp32 中间累加，NPU 输出可能比**同 dtype** CPU 路径更精确；数值精度判据以**高精度参考**（如 float64 计算）为准，不以同 dtype CPU 输出为准。
- 精度阈值（`ops-precision-standard`）：

  | dtype | 判定方式 | Threshold |
  |-------|---------|-----------|
  | float32 | MERE/MARE | `2^-13 ≈ 1.22e-4` |
  | float16 | MERE/MARE | `2^-10 ≈ 9.77e-4` |
  | bfloat16 | MERE/MARE | `2^-7 ≈ 7.81e-3` |
  | int32 | 位精确 | 二进制一致 |

## 调用示例

```python
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401  加载 libops_multimodal_fusion.so

M, K = 128, 256
self_ = torch.randn(M)
mat = torch.randn(M, K)
vec = torch.randn(K)

# 1. 在 NPU 上计算 out = beta*self + alpha*(mat @ vec)
out = torch.ops.ops_multimodal_fusion.addmv(
    self_.npu(), mat.npu(), vec.npu(), beta=1, alpha=1
)

# 2. 取回结果并与 PyTorch CPU 语义比对（常规有限值输入下在精度阈值内一致）
expected = torch.addmv(self_, mat, vec, beta=1, alpha=1)
print(torch.allclose(out.cpu(), expected, rtol=1e-4, atol=1e-3))  # True
```

自定义 `beta` / `alpha`：

```python
out = torch.ops.ops_multimodal_fusion.addmv(
    self_.npu(), mat.npu(), vec.npu(), beta=0.5, alpha=2.0
)
```

广播 `self`（标量偏置作用到全部 M 行）：

```python
# (1,) 形态：
bias = torch.randn(1)
out = torch.ops.ops_multimodal_fusion.addmv(bias.npu(), mat.npu(), vec.npu())

# 0-dim 形态：
bias0 = torch.randn(())
out = torch.ops.ops_multimodal_fusion.addmv(bias0.npu(), mat.npu(), vec.npu())

# (M,) expand view 形态：须在 NPU 侧 expand
biasM = torch.randn(1).npu().expand(M)
out = torch.ops.ops_multimodal_fusion.addmv(biasM, mat.npu(), vec.npu())
```

## 支持芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

- 目标芯片：`ascend950`（dav-3510 / arch35）。
- 算子实现：`applications/llm/addmv/arch35/addmv.asc`
- 测试文件：`tests/addmv/test_addmv.py`
