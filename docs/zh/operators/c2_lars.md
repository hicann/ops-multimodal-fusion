# c2_lars

> 面向使用者的算子文档。依据算子代码 `applications/llm/c2_lars/arch35/c2_lars.asc` 与需求真值源（`1.1-需求分析.md` v2）、开发方案（v5）、CP3 功能验收报告、CP4 性能验收报告编写。

## 算子概述

`c2_lars` 是 Caffe2 `Lars` 算子（**L**ayer-wise **A**daptive **R**ate **S**caling）的等价实现：给定一层的参数张量 `X` 与其梯度 `dX`，按两者的**全张量 L2 范数之比**重标定学习率，并用 `lr_max` / `lr_min` 双端截断，输出一个标量学习率。

```
X_norm  = ||X||_2                                         // 全张量 L2 范数（对所有元素）
dX_norm = ||dX||_2                                        // 全张量 L2 范数

val = (X_norm > 0) ? trust / (dX_norm / X_norm + wd + offset) : 1.0

lr_rescaled = max( min(val, lr_max), lr_min )
```

- 输入 `X` / `dX` 支持 fp32 / fp16 / bf16；`wd` / `trust` / `lr_max` 为**恒 float32** 的 1 元素标量 Tensor。
- 低精度输入在 kernel 内先展宽到 fp32 再做平方与累加（**禁止 fp16 累加**），**输出恒为 0-D float32 标量**。

## 接口列表

| 接口名 | 说明 |
|--------|------|
| `torch.ops.ops_multimodal_fusion.c2_lars` | Caffe2 LARS 学习率重标定：双路全张量 L2 归约 + 标量 epilogue（分支 + 双端 clamp） |

## 产品支持情况

| 芯片系列 | 支持情况 |
|---------|---------|
| Ascend910B | ❌ |
| Ascend910_93 | ❌ |
| Ascend950 | ✅ |

> 单架构交付：仅 `arch35`（dav-3510 / Ascend950），**不支持 `arch22`**（不产出 arch22 实现）。

## 接口详情

### `torch.ops.ops_multimodal_fusion.c2_lars`

#### 函数原型

```python
torch.ops.ops_multimodal_fusion.c2_lars(X, dX, wd, trust, lr_max, offset=0.5, lr_min=0.02) -> Tensor
```

Schema（与 `.asc` 内 `TORCH_LIBRARY_FRAGMENT` 声明一致）：

```
c2_lars(Tensor X, Tensor dX, Tensor wd, Tensor trust, Tensor lr_max, float offset=0.5, float lr_min=0.02) -> Tensor
```

> 默认值 `offset=0.5` / `lr_min=0.02` 取自 Caffe2 `LarsOp` 的参数默认值。

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 | 内存位置 |
|--------|---------|---------|------|---------|
| `X` | 输入 | Tensor | 参数张量，任意 shape / rank，参与 `X_norm = \|\|X\|\|_2`；dtype ∈ {fp32, fp16, bf16} | NPU |
| `dX` | 输入 | Tensor | 梯度张量，shape 与 dtype 均须与 `X` 完全一致，参与 `dX_norm = \|\|dX\|\|_2` | NPU |
| `wd` | 输入 | Tensor | weight decay，1 元素标量 Tensor，**恒 float32**；直接进入分母。**无非负约束**（与 Caffe2 一致，可为负） | NPU |
| `trust` | 输入 | Tensor | trust coefficient，1 元素标量 Tensor，**恒 float32**；作分子。无值域约束 | NPU |
| `lr_max` | 输入 | Tensor | 结果上界，1 元素标量 Tensor，**恒 float32**；参与 `min(val, lr_max)`。无值域约束 | NPU |
| `offset` | 输入 | float，默认 `0.5` | host 侧属性，分母偏置；要求 `offset >= 0` | — |
| `lr_min` | 输入 | float，默认 `0.02` | host 侧属性，结果下界；要求 `lr_min >= 0` | — |
| `lr_rescaled` | 输出 | Tensor | **0-D 标量** Tensor，1 元素，dtype **恒 float32** | NPU |

`wd` / `trust` / `lr_max` 的 **0-D 与 shape `(1,)` 两种形态均接受**（只要 `numel == 1`）；输出形态不受其影响，恒为 0-D。

#### 支持的数据类型

`X` 与 `dX` 的 dtype 必须相同，且属于下表：

| 数据类型 | PyTorch 类型 | 支持状态 | 说明 |
|----------|--------------|----------|------|
| FP32 | `torch.float32` | ✅ | 基线 dtype |
| FP16 | `torch.float16` | ✅ | kernel 内先 cast 到 fp32 再平方归约（避免 `\|x\| > ~256` 时 `x²` 即溢出 fp16） |
| BF16 | `torch.bfloat16` | ✅ | 同样 cast 到 fp32 累加；仅 8 位尾数，容差单独放宽（见[精度对齐说明](#精度对齐说明)） |
| INT32 | `torch.int32` | ❌ | LARS 语义为浮点学习率重标定，无整型场景，host 侧 `TORCH_CHECK` 拒绝 |

`wd` / `trust` / `lr_max` **与 `X` 的 dtype 解耦，恒为 float32**，传入其它 dtype 一律拒绝。

#### 支持的 shape

| 维度 | 约束 |
|------|------|
| `X` 的 shape / rank | 任意（0-D / 1-D / 2-D / 3-D / 4-D / 5-D 均已实测），归约结果与 shape 无关，只与元素总数有关 |
| `X.numel()` | `1 <= numel <= 2^31 - 1`（与 int32 索引上限对齐） |
| `dX` | `dX.sizes() == X.sizes()`（**完全一致**，不仅是同 numel） |
| `wd` / `trust` / `lr_max` | `numel == 1`（0-D 或 `(1,)`） |

- **空张量被拒绝**：`X.numel() == 0` 报错，而非短路返回（见约束 C1）。
- **实测覆盖范围**：功能与性能验收实跑覆盖 `numel` 为 1 / 1e3 / 1e5 / 1e7 四个量级（含质数等强非对齐尾块）及 L2 溢出档 2e8。`2^31 - 1` 是**声明上界**，未实跑（显存不允许）。

#### 约束说明

| # | 约束 | 说明 | 口径来源 |
|---|------|------|---------|
| C1 | `X` 非空且 dtype 合法 | `X.numel() >= 1`；dtype ∈ {fp32, fp16, bf16}；device 为 NPU。否则 host 侧 `TORCH_CHECK` 拒绝 | **本仓收窄**：Caffe2 未显式禁止空张量，本仓判定 `numel == 0` 为非法输入而非退化情形 |
| C2 | `dX` 与 `X` 同形同类型 | `dX.sizes() == X.sizes()` 且 `dX.dtype() == X.dtype()` | **本仓收窄**：Caffe2 原实现**只校验 `numel` 相等**（同 numel 异 shape 可通过），本仓收窄为 **shape 完全一致**，与仓内二元算子口径对齐 |
| C3 | 标量 Tensor 规格 | `wd` / `trust` / `lr_max` 各为 1 元素 Tensor（0-D 或 `(1,)`），dtype **恒 float32**，device 与 `X` 相同 | 本仓口径：标量 dtype 与 `X` 解耦（参照 `c2_accuracy` 固定 `labels` 为 int32 的先例） |
| C4 | `offset >= 0` | 否则 host 侧 `TORCH_CHECK` 拒绝（`offset = NaN` 亦被拒绝：`NaN >= 0` 求值为假） | 对齐 Caffe2 `CAFFE_ENFORCE_GE(offset_, 0)` |
| C5 | `lr_min >= 0` | 否则 host 侧 `TORCH_CHECK` 拒绝（`lr_min = NaN` 亦被拒绝，同上） | 对齐 Caffe2 `CAFFE_ENFORCE_GE(lr_min_, 0)` |
| C6 | `X.numel() <= 2^31 - 1` | 否则 host 侧 `TORCH_CHECK` 拒绝 | 本仓口径，与 int32 索引上限对齐 |
| C7 | 输出恒为 0-D float32 | 输出为 **0-D 标量** Tensor，dtype **恒 float32**，**不随输入 dtype 变化** | **本仓口径**：Caffe2 `LarsOp` 仅注册 float 实例，fp16/bf16 输入下 Caffe2 无对应定义；shape 取 0-D 而非 Caffe2 的 `[1]` |

- 全部校验在 **host 侧同步**完成（`TORCH_CHECK`），失败抛 `RuntimeError`，不下发任何 device 任务。
- **不校验 `lr_max` 与 `lr_min` 的大小关系**，也不校验 `wd` / `trust` / `lr_max` 的取值范围（见下节）。
- Meta 路径（`torch.compile` / meta 设备）执行同一套 C1~C6 校验（device 检查除外）。

## 数值边界行为

**总原则：一律以上述公式的浮点求值结果为准，不对任何特殊值做额外特判。** 下表为典型情形的行为说明（非穷尽枚举），均已由 CP3 的特殊值用例（SV-01~SV-14）实跑覆盖。

| 情形 | 触发条件 | 行为 |
|------|---------|------|
| **分母为 0，`trust > 0`** | `dX_norm/X_norm + wd + offset == 0`（如 `dX` 全零 ∧ `wd == 0` ∧ `offset == 0`，或 `wd` 取负值恰好抵消其余两项） | `val = +inf` → 输出 `lr_max`（当 `lr_max >= lr_min`） |
| **分母为 0，`trust < 0`** | 同上 | `val = -inf` → 输出 `lr_min` |
| **分母为 0，`trust == 0`** | 同上 | `val = NaN`，且 **NaN 被传播到输出**（与 PyTorch CPU 的 `torch.minimum` / `torch.clamp` 一致；kernel 内显式做了 NaN 对齐） |
| **分母为负** | `wd` 取足够小的负值 | 公式**照常求值、不报错**，`val` 与 `trust` 异号，输出经 clamp 落在 `[min(lr_max, lr_min), max(lr_max, lr_min)]` 区间内 |
| **`X_norm == 0`（X 全零）** | `X` 全体元素为 0 | 走 `val = 1.0` 分支，输出 `max(min(1.0, lr_max), lr_min)` |
| **`X` 含 NaN** | `X` 任一元素为 NaN | `X_norm = NaN` 使谓词 `(X_norm > 0)` 为**假** → 走 `val = 1.0` 分支，输出 `max(min(1.0, lr_max), lr_min)`（**不是** NaN） |
| **`X` 含 ±inf** | `X` 任一元素为 `+inf` 或 `-inf`（`(-inf)² = +inf`） | `X_norm = +inf` → `dX_norm / X_norm = 0` → `val = trust / (wd + offset)`，再按公式 clamp |
| **`lr_max < lr_min`** | 调用方自行传入 | 公式中 `max` 后置，**结果恒为 `lr_min`**；本算子**不校验**二者大小关系 |

## 已知限制与未定义行为

### 1. `lr_max = NaN`：已知未测边界，行为不作承诺

kernel 的 NaN 对齐**只处理 `val` 自身为 NaN 的情形**（即 `trust == 0` 且分母为 0，或输入张量使范数为 NaN）。若 **`lr_max` 本身被传入 NaN**：

- 本实现的比较指令对 NaN 操作数返回**另一侧**操作数（`min(val, NaN)` 取 `val`），最终得 `max(val, lr_min)`；
- 而 PyTorch CPU 的 `torch.minimum` / `torch.clamp` 会**传播 NaN**。

`lr_min` 为 NaN 时被 C5 校验拒绝（`NaN >= 0` 求值为假），**不可达**；但 `lr_max` 是**无值域约束的 device tensor，可达且当前无用例覆盖**。该情形不在需求枚举的边界行为内，**属未定义行为，调用方不应依赖其结果**。如需将其转为强约束，须走需求变更流程。

### 2. fp32 平方下溢导致的真实语义分歧（G1）

当 `X` 的元素量级极小时（全体满足 `|x| < 约 2.65e-23`，即 `√(½ × fp32 最小非规格数 1.4e-45)`），`x²` 在 fp32 中舍入为 **0**，kernel 得 `X_norm == 0` → 走 **`val = 1.0` 分支**；而无限精度（fp64）语义下 `X_norm > 0`，应走除法分支。**这是一处真实分歧，不是精度误差。**

CP3 实测（SV-10，`X` 全体 = `1e-30`）：

| 档位 | NPU 实测输出 | fp32 语义期望 | fp64 参考（golden） |
|------|-------------|--------------|-------------------|
| fp32 | **1.0** | 1.0 | 0.019999999552965164 |
| bf16 | **1.0** | 1.0 | 0.019999999552965164 |

即：**极小量级参数张量下，本算子按 fp32 下溢语义退化到 `val = 1.0` 分支**，与 fp64 参考实现结果不同。该分歧对 fp32 与 bf16 可达（bf16 最小非规格数约 `9.22e-41`）；fp16 的最小非规格数约 `5.96e-8`，远大于该阈值，故 fp16 输入下不可达。

### 3. fp32 平方和上溢（G2）：说明性提示，输出上不可观测

当元素量级极大时（如全体 = `1e20`，`x² = 1e40` 溢出 fp32），fp32 侧 `X_norm = +inf` 使 `dX_norm / X_norm = 0`；fp64 侧 `X_norm` 虽有限但极大，该比值同样下溢至 ~0。**两条路径殊途同归到 `val = trust / (wd + offset)`。**

CP3 实测（SV-11）：fp32 与 bf16 档 NPU 输出、fp32 语义期望、fp64 golden **三者同为 `0.8333333134651184`**。故 G2 属「设计期识别、实际不可观测」的分歧，**不构成与常规语义不同的行为承诺**。

### 4. 非连续（非 contiguous）输入当前不支持

算子 host 侧已对 `X` / `dX` 做 `.contiguous()` 归一，语义上是正确处理路径；但当前环境 `torch_npu` **缺 D2D strided copy**（触发 `RuntimeError ... error code is 561103`），故非连续输入**无法上板执行**，对应测试用例标记 `skip`。

- 调用方当前须自行传入连续张量（或先在 CPU 侧 `.contiguous()` 后再 `.npu()`）。
- **平台修复 D2D strided copy 后，算子侧无需任何改动即自动生效**；该限制随 CANN 版本升级复检。
- 因此**不得宣称"非连续输入可用"**。

### 5. 当前环境的原生算子覆盖不全（非本算子问题）

本机 CANN 环境的原生 kernel 覆盖残缺（CP4 已记录：`copy_` / `clone` / `mul` / `sum` / `linalg.vector_norm` 等 9 条原生路径均失败）。直接 `print` 一个 NPU 张量会内部调用 `torch.isfinite` 而报 `561103`，故本文示例一律先 `.cpu()` / `.item()` 取回再打印。该现象与 `c2_lars` 自身无关。

## 精度对齐说明

- **标杆（golden）**：PyTorch CPU，按上述公式做 **fp64 五步直算**（不调 `torch.linalg.vector_norm`，避免其内部防溢出缩放偏离朴素语义）。
- **容差按输入 dtype 分档**，而非按输出 dtype。原因：本算子输出恒 fp32，若套用「按输出 dtype 取阈值」的通用口径，bf16 输入也会被按 fp32 阈值判定，而 bf16 经平方归约后有效精度仅 2~3 位十进制，该判定不可达。
- 判据为 `torch.allclose`，即 `|actual − golden| <= atol + rtol·|golden|`。

| 输入 dtype | rtol | atol | CP3 实测最大相对误差 | CP3 实测最大绝对误差 | 容差占用 |
|-----------|------|------|--------------------|--------------------|---------|
| float32 | `1e-4` | `1e-5` | **6.519e-07** | 5.960e-08 | 0.11%（裕度 904x） |
| float16 | `1e-3` | `1e-3` | **1.904e-07** | 5.960e-08 | <0.01%（裕度 27622x） |
| bfloat16 | `2e-4` | `1e-8` | **8.926e-06** | 9.537e-07 | 4.32%（裕度 23x） |

- bf16 的 `atol=1e-8` 仅为 `|golden| < 5e-5` 的极小输出兜底；常规输出量级（1e-3 ~ 1e1）下判据由 rtol 项主导。
- fp16 / bf16 输入在 kernel 内全程 fp32 累加，实测误差主要来自跨 chunk 的串行累加链，而非输入量化。
- **SV-10 / SV-11（G1 / G2）不适用上表**：该两例的 fp64 参考与 NPU 被设计成分歧（见上文），其偏差不是精度缺陷。

## 性能特征

以下为 CP4 在 `Ascend950PR_9579`（AIV = 56、UB 248 KB、L2 128 MB）上的实测结论，供使用者预估用。

| 规模区间 | 主 Bound | 特征 |
|---------|---------|------|
| 大 shape（工作集溢出 L2） | **MTE2 BOUND（访存受限）** | `aiv_mte2_ratio = 0.992`（fp32 @ numel=5e7）；搬入量恰为理论下界 `2 × numel × sizeof(dtype)`，`X` / `dX` 各读一遍无重复读入 |
| 小 numel（≤ 262144） | **固定启动成本主导** | 时延 **≈ 15–19 µs 且与规模无关**（numel 从 1 到 262144 跨 5 个数量级，median 无增长趋势） |
| 拐点 | — | 平台期拐点在 **numel ≈ 2e6 ~ 4e6**（fp32）；越过后搬运开始主导 |

- **稳态读吞吐**：本机实测约 **1610 GB/s**（numel = 5e7 / 1e8 / 2e8 分别为 1578 / 1606 / 1611 GB/s，跨轮极差 0.1~0.2%）。该值是本算子在本机达成的稳态字节吞吐；受环境限制，无法在本机证实其等于 HBM 硬件上限。
- **L2 加持**：工作集小于 L2（128 MB）时读带宽显著更高（numel = 1e7 fp32 达 ~3100 GB/s，partial 段可达 3960 GB/s），故 1e7 量级的带宽数字**不代表 HBM 稳态**。
- **多核**：`numel >= 262144` 起 56 核全起；访存主导区（`numel >= 2e6`）负载不均衡度 ≤ 1.25%。
- **跨核归约收口**：实现为「多核 partial + 单核 finalize」两段 launch，finalize kernel 恒为 **1.80–2.35 µs 的有界常量**（跨 numel 1024 → 5e7、跨 dtype 均如此），大 shape 下仅占 device 时间 **0.9%**。
- **尾块无惩罚**：非对齐尾块由 VF 层 `MaskReg` 处理；`aiv_mte2_ratio = 0.992` 将全部非 MTE2 开销（含尾块掩码、tile 内归约、标量链、双缓冲切换）合计上界锁定在 **0.8%**。
- **采集环境为共享 NPU**，小 numel 的 median 受邻居进程干扰波动较大（跨轮极差 38~89%），上表小 numel 数字应按量级读取。

## 调用示例

```python
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401  加载 libops_multimodal_fusion.so

torch.manual_seed(0)

X = torch.randn(1024, 512)          # 参数张量
dX = torch.randn_like(X) * 1e-2     # 梯度张量，shape / dtype 与 X 完全一致
wd = torch.tensor(1e-4)             # 1 元素 float32 标量 Tensor
trust = torch.tensor(0.1)
lr_max = torch.tensor(1.0)
offset, lr_min = 0.5, 0.02

# 1. 在 NPU 上计算重标定后的学习率
out = torch.ops.ops_multimodal_fusion.c2_lars(
    X.npu(), dX.npu(), wd.npu(), trust.npu(), lr_max.npu(), offset, lr_min
)
print(out.cpu(), out.shape, out.dtype, out.dim())
# tensor(0.1960) torch.Size([]) torch.float32 0        ← 0-D fp32 标量

# 2. 取回结果并与 PyTorch CPU 语义（fp64 直算）比对
x64, d64 = X.double(), dX.double()
X_norm, dX_norm = x64.pow(2).sum().sqrt(), d64.pow(2).sum().sqrt()
val = torch.where(
    X_norm > 0,
    trust.double() / (dX_norm / X_norm + wd.double() + offset),
    torch.ones_like(X_norm),
)
expected = torch.maximum(torch.minimum(val, lr_max.double()),
                         torch.tensor(lr_min, dtype=torch.float64))
print(torch.allclose(out.cpu().double(), expected, rtol=1e-4, atol=1e-5))  # True
```

使用默认属性（`offset=0.5` / `lr_min=0.02`），标量也可传 shape `(1,)` 形态：

```python
out = torch.ops.ops_multimodal_fusion.c2_lars(
    X.npu(), dX.npu(), wd.reshape(1).npu(), trust.reshape(1).npu(), lr_max.reshape(1).npu()
)
assert out.dim() == 0 and out.dtype == torch.float32   # 输出形态不随标量形态变化
```

低精度输入（fp16 / bf16）——`X` 与 `dX` 的 dtype 必须一致，输出仍为 fp32：

```python
Xh, dXh = X.half().npu(), dX.half().npu()
out_h = torch.ops.ops_multimodal_fusion.c2_lars(Xh, dXh, wd.npu(), trust.npu(), lr_max.npu())
assert out_h.dtype == torch.float32
```

`X` 全零时走 `val = 1.0` 分支：

```python
Z = torch.zeros(1024).npu()
out_z = torch.ops.ops_multimodal_fusion.c2_lars(
    Z, torch.randn(1024).npu(), wd.npu(), trust.npu(), lr_max.npu(), 0.5, 0.02
)
# out_z == max(min(1.0, lr_max), lr_min) == 1.0
```

## 支持芯片

| 芯片类型 | 架构代号 | 支持状态 |
|----------|----------|----------|
| Atlas A2 训练/推理系列 | arch22 | - |
| Atlas A3 训练/推理系列 | arch22 | - |
| 950 系列 | arch35 | ✅ |

- 目标芯片：`ascend950`（dav-3510 / **arch35**）；构建参数 `--soc=ascend950`。**不支持 arch22**。
- 算子实现：`applications/llm/c2_lars/arch35/c2_lars.asc`
- 测试文件：`tests/c2_lars/test_c2_lars.py`、`tests/c2_lars/test_c2_lars_whitebox.py`
- 验证环境：CANN 9.1.0 / torch 2.7.1 + torch_npu 2.7.1.post4 / Python 3.12。
