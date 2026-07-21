# 本仓 .asc/直调编码约定

> 本文件收录 ops-multimodal-fusion 仓特有的编码约束，补充 AscendC 通用编码规则（R1-R10）和 MR 安全规则之外的本仓 .asc/直调领域编码约定。
>
> 组织原则：**一个 PyTorch 算子 = 一个单文件 AscendC `.asc` kernel**，与 PyTorch 原生语义/精度对齐。真实范例见 `applications/llm/add/arch35/add.asc`。

---

## 1. 单 .asc 每算子一文件

- 每个算子只有一个 `applications/llm/{op}/arch35/{op}.asc` + 同目录一个 `CMakeLists.txt`（内容固定，仅一行 `add_sources()`）。
- **不拆** `host.cpp` / `kernel.cpp` / `{op}_kernel.h`：Schema、Meta、Tiling、Kernel、NPU Dispatch 全部内联于同一个 `.asc` 文件。
- 简单算子无需任何本地头文件；只有多头文件复杂融合算子（如 `swi_glu`）才把 kernel/tiling 拆到同目录 `{op}_apt.h` / `{op}_host_tiling.h` 等头文件，`.asc` 只做 host/注册薄层。

```
applications/llm/add/arch35/
├── add.asc          # 全部逻辑内联于此
└── CMakeLists.txt   # 固定：add_sources()
```

---

## 2. 5 段式结构（全部内联于单 .asc）

`.asc` 从上到下固定 5 段，段间用 `// 1.` ~ `// 5.` 注释分隔：

| 段 | 内容 | 关键 API |
|----|------|---------|
| (1) Schema | 注册算子 schema | `TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m){ m.def("op(...)->..."); }` |
| (2) Meta | 推断输出 shape/dtype，无计算 | `{op}_meta(...)`（`torch::empty_like` 等）+ `TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)` |
| (3) Tiling | host 侧算 tiling scalar | `std::tuple<...> calc_tiling_params(int64_t totalLength)` |
| (4) Kernel | device 侧计算 | `template <typename T> __global__ __aicore__ __vector__ void {op}_kernel(...)` |
| (5) NPU Dispatch | 组装并直调 kernel | `{op}_npu(...)` + `TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)` |

- Meta 中即做输入合法性校验（`TORCH_CHECK`），Meta 被 NPU Dispatch 复用（`auto z = {op}_meta(x, y);`）。
- 空 tensor 早返回：`if (x.numel() == 0) { return z; }`。

---

## 3. `<<<>>>` 直调（fast kernel launch）

kernel 通过 `<<<>>>` 语法直调，包在 `AT_DISPATCH_SWITCH` 的 lambda 内，再交给 `OpCommand::RunOpApi`：

```cpp
auto acl_call = [=]() -> int {
    AT_DISPATCH_SWITCH(
        x.scalar_type(), "{op}_npu",
        AT_DISPATCH_CASE(torch::kFloat32, [&] {
            using scalar_t = float;
            {op}_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength, blockLength, tileSize);
        })
        AT_DISPATCH_CASE(torch::kFloat16, [&] {
            using scalar_t = half;
            {op}_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength, blockLength, tileSize);
        })
        AT_DISPATCH_CASE(torch::kInt32, [&] {
            using scalar_t = int32_t;
            {op}_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength, blockLength, tileSize);
        })
    );
    return 0;
};
at_npu::native::OpCommand::RunOpApi("{Op}", acl_call);
```

- `<<<numBlocks, nullptr, stream>>>`：第 1 参核数（来自 tiling），第 2 参恒为 `nullptr`，第 3 参 `c10_npu::getCurrentNPUStream().stream(false)`。
- 数据指针一律 `(GM_ADDR)tensor.data_ptr()`；scalar tiling（`totalLength`/`blockLength`/`tileSize`）按值随 `<<<>>>` 传入。
- `RunOpApi` 第 1 参是算子名字符串（PascalCase，如 `"Add"`）。
- 变体：class 化 kernel（如 `layer_norm`）由薄 `__global__` launcher 转调 `Kernel{Op}<T>::Init/Process`；多头文件融合算子（如 `swi_glu`）`<<<>>>` 传的是 tiling struct 而非 scalar。

---

## 4. 命名规范

| 类别 | 风格 | 示例 |
|------|------|------|
| 目录名（算子名） | snake_case | `add`、`abs`、`layer_norm`、`avg_pool2d` |
| 文件名 | `{op}.asc` | `add.asc`、`layer_norm.asc` |
| namespace | `ops_multimodal_fusion::{Op}` | `ops_multimodal_fusion::Add`、`ops_multimodal_fusion::LayerNorm` |
| kernel 函数 | `{op}_kernel<T>` | `add_kernel<T>`、`abs_kernel<T>` |
| host meta | `{op}_meta` | `add_meta`、`abs_meta` |
| host dispatch | `{op}_npu` | `add_npu`、`abs_npu` |
| schema 名串 | snake_case（`m.def` 内） | `"add(Tensor x, Tensor y) -> Tensor"` |
| RunOpApi 名串 | PascalCase | `"Add"`、`"Abs"`、`"LayerNorm"` |

- `{op}`=snake_case，`{Op}`=PascalCase，二者贯穿全文一致对应。

---

## 5. EXTENSION_MODULE_NAME 宏

- 所有注册宏的库名参数统一用 **`EXTENSION_MODULE_NAME` 宏**，**禁止**写死 `ops_multimodal_fusion` 字面量：
  - `TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)`（Schema）
  - `TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)`（Meta 后端）
  - `TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)`（NPU 后端）
- 该宏由构建系统统一注入为 wheel 包名 `ops_multimodal_fusion`，保证 schema 库名与 Python 侧 `torch.ops.ops_multimodal_fusion.{op}` 对齐。

---

## 6. Tiling：platform_ascendc 取核数/UB，scalar 传参

- host 侧 `calc_tiling_params` 用 `platform_ascendc::PlatformAscendCManager::GetInstance()` 动态取硬件参数，**禁止**硬编码核数（见 R2）：
  - 核数：`ascendcPlatform->GetCoreNumAiv()`
  - UB 大小：`ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize)`
- 返回 **scalar tuple**（`numBlocks`/`blockLength`/`tileSize`），**不构造 `TilingData` struct**、不做 H2D 拷贝 tiling。

```cpp
std::tuple<int64_t, int64_t, int64_t> calc_tiling_params(int64_t totalLength)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    TORCH_CHECK(coreNum > 0, "coreNum must be positive.");
    int64_t numBlocks = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    int64_t tileSize = ubSize / PIPELINE_DEPTH / BUFFER_NUM;
    return std::make_tuple(numBlocks, blockLength, tileSize);
}
```

- kernel 侧多核偏移：`SetGlobalBuffer((__gm__ T *)x + blockLength * AscendC::GetBlockIdx())`。
- kernel 内用 `AscendC::TPipe pipe` / `TQue<VECIN|VECOUT>` / `InitBuffer` / `DataCopyPad`（批量搬移，见 R1），`TPipe` 为局部变量（见 R3）。
- 复杂融合算子例外：`swi_glu` 在头文件里构造 tiling struct 并随 `<<<>>>` 传入。

---

## 7. 参数校验：一律 TORCH_CHECK

- 所有 host 侧参数校验（shape 一致、dtype 合法、非空等）统一用 `TORCH_CHECK(cond, "msg")`，通常放在 Meta 里：

```cpp
torch::Tensor add_meta(const torch::Tensor &x, const torch::Tensor &y)
{
    TORCH_CHECK(x.sizes() == y.sizes(), "The shapes of x and y must be the same.");
    return torch::empty_like(x);
}
```

- **禁止**任何 `OP_LOG` / `dlog` / `printf` 风格返回码校验；本仓无独立日志族，详见 `references/validation-and-logging.md`。

---

## 8. NPU Dispatch 组装顺序

`{op}_npu` 固定按此顺序组装：

```cpp
torch::Tensor {op}_npu(const torch::Tensor &x, ...)
{
    const c10::OptionalDeviceGuard guard(x.device());        // 1. 设备守卫
    auto z = {op}_meta(x, ...);                              // 2. 复用 meta 得输出
    if (x.numel() == 0) { return z; }                        // 3. 空 tensor 早返回
    auto stream = c10_npu::getCurrentNPUStream().stream(false); // 4. 取当前流
    int64_t totalLength = x.numel(), numBlocks, blockLength, tileSize;
    std::tie(numBlocks, blockLength, tileSize) = calc_tiling_params(totalLength); // 5. tiling
    auto x_ptr = (GM_ADDR)x.data_ptr();                      // 6. data_ptr -> GM_ADDR
    auto z_ptr = (GM_ADDR)z.data_ptr();
    auto acl_call = [=]() -> int { /* AT_DISPATCH_SWITCH + <<<>>> */ return 0; }; // 7.
    at_npu::native::OpCommand::RunOpApi("{Op}", acl_call);   // 8. 提交
    return z;
}
```

---

## 9. include 清单

`.asc` 顶部固定引入以下头文件（顺序照 `add.asc`），**禁止**引入冗余头文件：

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

- 复杂融合算子在此之后追加同目录本地头文件（如 `#include "swi_glu_apt.h"`）。

---

## 10. arch22/arch35 布局

- SoC→arch 映射：`ascend910b|ascend910_93` → `NPU_ARCH=dav-2201` / 目录 `arch22`；`ascend950` → `dav-3510` / 目录 `arch35`。
- **arch35 为近乎唯一目标**（130 个算子）；**arch22 仅 `abs` 一个算子有**（供 910b/910_93 用）。
- 新增算子默认只放 `arch35/`；仅当明确要覆盖 910b/910_93 时才另建 `arch22/{op}.asc`。

```
applications/llm/abs/
├── arch22/{ abs.asc, CMakeLists.txt }   # 仅 abs 有 arch22
└── arch35/{ abs.asc, CMakeLists.txt }
```

---

## 11. OAT 量化指标

| 指标 | 阈值 | 说明 |
|------|------|------|
| 圈复杂度 | ≤ 20 | 每个函数 |
| 函数深度 | ≤ 5 | 嵌套层级 |
| NBNC (Non-Blank Non-Comment lines) | ≤ 50 | 有效代码行 |
| 除零风险 | 0 | 不允许无保护的除法（tiling 中 `coreNum`/`numBlocks` 须先 `TORCH_CHECK > 0`） |

---

## 12. License Header

`.asc` 与同目录 `CMakeLists.txt` 必须带 CANN 头（`.asc` 用 `/*! ... */`，CMakeLists 用 `#` 注释），照 `applications/llm/add/arch35/add.asc`：

```cpp
/*!
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
```

紧随其后加文件说明块：

```cpp
/*!
 * \file {op}.asc
 * \brief {Op} operator implementation for Ascend NPU
 */
```
