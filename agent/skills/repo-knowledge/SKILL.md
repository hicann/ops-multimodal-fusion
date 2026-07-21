---
name: repo-knowledge
description: 仓库领域知识，提供本仓算子涉及的领域标准、概念与背景（PyTorch 自定义算子体系、多模态融合算子库 API 规范）。
---

# ops-multimodal-fusion 库规范

ops-multimodal-fusion 是一个基于 AscendC 的 **PyTorch 自定义算子库**，采用 fast kernel launch（`<<<>>>` 直调）实现，编译为单个 wheel 包 `ops_multimodal_fusion`（version 1.0.0，对标 CANN 9.0.0）。组织原则是「**一个 PyTorch 算子 = 一个单文件 AscendC `.asc` kernel**」，语义与精度对标 PyTorch 原生算子。

开发新算子时，一个 `.asc` 文件内联 **5 段**，按职责各就其位：

| 段 | 宏 / 函数 | 职责 |
|----|-----------|------|
| **1. Schema** | `TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)` + `m.def(...)` | 声明算子签名，注册到 `torch.ops.ops_multimodal_fusion` |
| **2. Meta** | `{op}_meta(...)` + `TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)` | 无计算的形状 / dtype 推断 |
| **3. Tiling** | host 侧 `calc_tiling_params(int64_t totalLength)` | 用 `platform_ascendc` 取硬件参数，算出 scalar 切分参数 |
| **4. Kernel** | `template <typename T> __global__ __aicore__ __vector__ void {op}_kernel(...)` | AscendC 计算主体 |
| **5. NPU Dispatch** | `{op}_npu(...)` + `TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)` | 直调 kernel 并经 `OpCommand::RunOpApi` 提交 |

---

## 一、本仓定位

| 维度 | 说明 |
|------|------|
| 库形态 | 基于 AscendC 的 PyTorch 自定义算子库，编译为单个 wheel 包 `ops_multimodal_fusion`（version 1.0.0） |
| 执行方式 | fast kernel launch —— host 侧用 `kernel<<<numBlocks, nullptr, stream>>>(...)` **直调** device kernel（无算子工程 kernel 模板注册流程） |
| 规模 | 共 **130 个算子**，覆盖 16 类 PyTorch 语义 |
| 对标 | CANN 9.0.0；每个算子与 **PyTorch 原生同名算子的语义 / 精度对齐** |
| 调用面 | `torch.ops.ops_multimodal_fusion.{op}(...)` |
| 加载方式 | `import torch; import torch_npu; import ops_multimodal_fusion`（加载 `libops_multimodal_fusion.so`） |

**依赖**：`torch`、`torch_npu` 由 `setup.py` 的 `install_requires` 声明，须**预装**（`install_deps.sh` 只装 python>=3.8 / gcc>=7.3.0 / cmake>=3.16.0，不装 torch/torch_npu）。`requirements.txt` 只含 build/pyyaml/numpy<2/pytest 及 PyTorch CPU 的 `--extra-index-url`。

---

## 二、PyTorch 自定义算子三段注册模型

一个算子通过 **三次注册** 接入 PyTorch dispatcher，全部内联在同一个 `.asc` 文件，`EXTENSION_MODULE_NAME` 即库命名空间 `ops_multimodal_fusion`：

| 注册宏 | 分派键 | 职责 | 触发时机 |
|--------|--------|------|----------|
| `TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)` | — | `m.def("op(...)->...")` 声明 schema | 库加载时 |
| `TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)` | `Meta` | 绑定 `{op}_meta`，做形状 / dtype 推断（无计算） | `torch.compile` / autograd / meta 设备走 |
| `TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)` | `PrivateUse1`（NPU） | 绑定 `{op}_npu`，真正在 NPU 上执行 | tensor 在 NPU 设备上时走 |

**Schema 示例**（见 `applications/llm/add/arch35/add.asc`）：

```cpp
TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("add(Tensor x, Tensor y) -> Tensor");
}
```

**Meta 示例**（推断输出 shape / dtype，通常 `torch::empty_like`）：

```cpp
torch::Tensor add_meta(const torch::Tensor &x, const torch::Tensor &y)
{
    TORCH_CHECK(x.sizes() == y.sizes(), "The shapes of x and y must be the same.");
    return torch::empty_like(x);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("add", add_meta);
}
```

> Meta 与 NPU Dispatch 共用同一份形状推断：`{op}_npu` 内部先调用 `{op}_meta` 拿到输出 tensor，再启动 kernel 写入。

**命名空间约定**：`namespace ops_multimodal_fusion::{Op}`，其中 `{Op}` 为算子名的 PascalCase（如 `add` → `Add`）。

---

## 三、直调执行模型（fast kernel launch）

NPU Dispatch 函数 `{op}_npu` 的执行链路：

| 步骤 | 代码 | 作用 |
|------|------|------|
| 1. device guard | `const c10::OptionalDeviceGuard guard(x.device());` | 绑定当前 NPU 设备 |
| 2. 形状推断 | `auto z = {op}_meta(...);` | 复用 Meta 得到输出 tensor |
| 3. 空 tensor 短路 | `if (x.numel() == 0) return z;` | 不启动 kernel，直接返回 |
| 4. 取流 | `auto stream = c10_npu::getCurrentNPUStream().stream(false);` | 当前 NPU stream |
| 5. tiling | `std::tie(numBlocks, blockLength, tileSize) = calc_tiling_params(totalLength);` | host 侧算切分 |
| 6. 取指针 | `auto x_ptr = (GM_ADDR)x.data_ptr();` | tensor → GM 地址 |
| 7. 按 dtype 分派 + 直调 | `AT_DISPATCH_SWITCH(...) { {op}_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(...); }` | 见下 |
| 8. 提交 | `at_npu::native::OpCommand::RunOpApi("Name", acl_call);` | 把 lambda 交给运行时下发 |

**dtype 分派 + 直调**（`AT_DISPATCH_SWITCH` / `AT_DISPATCH_CASE`）：

```cpp
auto acl_call = [=]() -> int {
    AT_DISPATCH_SWITCH(
        x.scalar_type(), "add_npu",
        AT_DISPATCH_CASE(torch::kFloat32, [&] {
            using scalar_t = float;
            add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength, blockLength, tileSize);
        })
        AT_DISPATCH_CASE(torch::kFloat16, [&] {
            using scalar_t = half;   // kFloat16 对应 AscendC 的 half
            add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength, blockLength, tileSize);
        })
        AT_DISPATCH_CASE(torch::kInt32, [&] {
            using scalar_t = int32_t;
            add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength, blockLength, tileSize);
        })
    );
    return 0;
};
at_npu::native::OpCommand::RunOpApi("Add", acl_call);
```

**要点**：

- `<<<numBlocks, nullptr, stream>>>` 三参数分别是核数、tiling（scalar 直调时为 `nullptr`）、stream。
- `RunOpApi` 的第一个参数是算子名字符串（如 `"Add"`），第二个是包裹 kernel launch 的 lambda。
- 参数校验一律用 `TORCH_CHECK`。

---

## 四、AscendC kernel 编程模型

kernel 为模板函数，签名形如：

```cpp
template <typename T>
__global__ __aicore__ __vector__ void {op}_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                  int64_t totalLength, int64_t blockLength, uint32_t tileSize);
```

内部编程要素：

| 要素 | 用法 | 说明 |
|------|------|------|
| `AscendC::TPipe pipe;` | 声明流水管理器 | 管理 UB 内存与 Que |
| `AscendC::TQue<QuePosition::VECIN, DEPTH>` / `VECOUT` | 输入 / 输出队列 | double buffer 深度由模板参数控制 |
| `pipe.InitBuffer(que, DEPTH, tileSize)` | 分配 UB buffer | tileSize 由 tiling 传入 |
| `AscendC::GlobalTensor<T>` + `SetGlobalBuffer(...)` | 绑定 GM | 见下多核偏移 |
| `AscendC::GetBlockIdx()` | 当前核序号 | 用于多核地址偏移 |
| `AllocTensor` / `EnQue` / `DeQue` / `FreeTensor` | 搬入 → 计算 → 搬出 | 队列生命周期 |
| `AscendC::DataCopyPad(...)` + `DataCopyExtParams` / `DataCopyPadExtParams<T>` | GM↔UB 搬运 | 处理非对齐尾块 |
| `AscendC::Add / Abs / ...` | 计算指令 | 与 PyTorch 算子语义对应 |

**多核地址偏移**：每个核只处理自己那段数据，SetGlobalBuffer 时按核偏移：

```cpp
xGm.SetGlobalBuffer((__gm__ T *)x + blockLength * AscendC::GetBlockIdx());
```

尾块由 `currentBlockLength = totalLength - GetBlockIdx() * blockLength` 计算，配合 `tileNum` / `tailTileElementNum` 分主循环块 + 尾块两段搬运计算。

### Tiling（host 侧 `platform_ascendc`）

tiling 在 host 侧完成，返回 **scalar** 参数（不打包成 struct，直调时随 `<<<>>>` 逐个传入）：

```cpp
std::tuple<int64_t, int64_t, int64_t> calc_tiling_params(int64_t totalLength)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);  // UB 容量
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();                          // AIV 核数
    TORCH_CHECK(coreNum > 0, "coreNum must be positive.");
    int64_t numBlocks   = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    int64_t tileSize    = ubSize / PIPELINE_DEPTH / BUFFER_NUM;
    return std::make_tuple(numBlocks, blockLength, tileSize);
}
```

- **核数**：`GetCoreNumAiv()`；**UB 容量**：`GetCoreMemSize(CoreMemType::UB, ubSize)`。
- 返回 `numBlocks`（起多少核）/ `blockLength`（每核处理长度）/ `tileSize`（单次搬运字节数）。

### `.asc` 固定 include 集合

```cpp
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include <type_traits>
```

---

## 五、算子谱与命名 / 交付约定

### 算子谱（16 类）

| 类 | 说明 | 示例算子 |
|----|------|---------|
| elementwise 数学 | 逐元素算术 / 初等函数 | add / abs / mul / exp / sqrt |
| 特殊 / 科学函数 | Bessel / 正交多项式 / 特殊函数 | bessel_j0 / chebyshev_polynomial_t / airy_ai / digamma / zeta |
| RNG | 随机分布采样 | binomial / cauchy / poisson |
| pooling | 池化 | avg_pool2d / adaptive_avg_pool2d |
| normalization | 归一化 | layer_norm / rms_norm_gated / weight_norm |
| quantization | 量化 | dynamic_quant / make_per_tensor_quantized |
| RNN cell | 循环单元 | gru_cell / lstm_cell |
| reductions / scans | 归约与扫描 | sum / mean / cumprod / logcumsumexp |
| indexing / sort | 索引与排序 | index_copy / kthvalue / searchsorted |
| interpolation / upsample | 插值上采样 | upsample_linear1d / upsample_nearest1d |
| foreach | 批量逐张量 | foreach_ceil / foreach_floor |
| nested-tensor | 嵌套张量 | nested_add_pad / nested_bmm |
| fusion | 融合算子 | swi_glu |
| complex | 复数 | angle / polar |
| Caffe2 | Caffe2 兼容 | c2_accuracy / c2_boolean_mask |
| 损失 / 距离 | 损失与距离度量 | multi_margin_loss / multilabel_margin_loss / pdist |

### 命名 / 交付约定

| 约定 | 规则 |
|------|------|
| 算子名 | `{op}` 用 snake_case（`add` / `layer_norm` / `avg_pool2d`），与 PyTorch 同名对齐 |
| 命名空间 | `ops_multimodal_fusion::{Op}`，`{Op}` 为 PascalCase |
| 交付形态 | 单文件 `applications/llm/{op}/arch35/{op}.asc` + 同目录 `CMakeLists.txt` + `tests/{op}/test_{op}.py` |
| `CMakeLists.txt` | 固定内容，正文仅一行 `add_sources()` |
| 校验 / 日志 | 参数校验一律 `TORCH_CHECK`（无其他自定义日志宏 / 状态码枚举） |

### 变体（复杂算子）

| 变体 | 参考 | 特征 |
|------|------|------|
| class 化 kernel | `applications/llm/layer_norm/arch35/layer_norm.asc` | `KernelLayerNorm<T>` 类（`Init` / `Process`）+ 薄 `__global__` launcher |
| 多头文件复杂融合 | `applications/llm/swi_glu/arch35/swi_glu.asc` | `.asc` 只做 host / 注册薄层，kernel 与 tiling 拆到同目录头文件（`swi_glu_apt.h` / `swi_glu_host_tiling.h` 等）；`<<<>>>` 传 tiling struct |

---

## 六、arch22 / arch35 目标与 SoC 映射

`build.sh` 只认三种 SoC，映射到 NPU_ARCH 与源码目录：

| `--soc` | NPU_ARCH | 目录 | wheel 后缀 |
|---------|----------|------|-----------|
| `ascend910b` | `dav-2201` | `arch22` | `+ascend910b` |
| `ascend910_93` | `dav-2201` | `arch22` | `+ascend910_93` |
| `ascend950`（默认，可用 `SOC` 环境变量覆盖） | `dav-3510` | `arch35` | `+ascend950` |

- **arch35 为近乎唯一目标**：130 个算子均落在 `arch35`；`arch22` 仅 `abs` 一个算子有（供 910b / 910_93 使用）。
- 产物：`dist/ops_multimodal_fusion-1.0.0+<soc>-cp<py>-...-linux_*.whl`。
- 安装：在源码目录**外**执行 `pip install <whl> --force-reinstall --no-deps`。
- 测试：`pytest tests/ -v`（或 `pytest tests/{op}/ -v`）。

---

## 七、多模态融合算子库领域补充规则

> 以下规则为 ops-multimodal-fusion 仓特有的领域约束，补充上述库规范。详细内容见 [references/domain-rules.md](references/domain-rules.md)。

| 主题 | 说明 |
|------|------|
| PyTorch 原生语义 / 精度对齐 | 每个算子的输出与 PyTorch 同名算子在语义与精度上对齐，golden 用 PyTorch CPU 计算 |
| Meta 形状 / dtype 推断正确性 | Meta 函数必须准确推断输出 shape / dtype，支撑 `torch.compile` 与 autograd |
| dtype 支持与 AT_DISPATCH 分派 | 支持 dtype 通过 `AT_DISPATCH_CASE` 显式枚举（float32/float16/int32），`kFloat16` 对应 AscendC `half` |
| 空 tensor 短路 | `numel() == 0` 时不启动 kernel，直接返回 Meta 推断出的空输出 |
| PrivateUse1 分派与 device guard | NPU 实现走 `PrivateUse1` 键，入口用 `OptionalDeviceGuard` 绑定设备 |
