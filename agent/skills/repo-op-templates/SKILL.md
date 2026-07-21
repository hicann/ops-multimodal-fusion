---
name: repo-op-templates
description: 算子代码模板库，提供 .asc 单文件骨架与模板选择规则，作为算子代码开发起点。
---

# Skill: repo-op-templates

ops-multimodal-fusion 算子代码模板库，为本仓「一个 PyTorch 算子 = 一个单文件 AscendC `.asc` kernel」的形态提供标准化代码骨架。Agent 在开发新算子时，应先按算子特征选定模板类型，以对应模板为起点，再按 `// TEMPLATE:` 注释填充业务逻辑。

---

## 本仓算子形态

ops-multimodal-fusion 是基于 AscendC 的 PyTorch 自定义算子库，采用 fast kernel launch（`<<<>>>` 直调）实现，编译为单个 wheel 包 `ops_multimodal_fusion`。每个算子由 **一个单文件 `.asc`** + **同目录 `CMakeLists.txt`** 构成：

```
applications/llm/{op}/
  arch35/                 -- 近乎唯一目标（ascend950 / dav-3510），130 个算子均在此
    {op}.asc              -- 单文件算子：schema + meta + tiling + kernel + dispatch 全内联
    CMakeLists.txt        -- 固定内容，仅一行 add_sources()
  arch22/                 -- 仅少数算子需要（当前仅 abs）；ascend910b / ascend910_93 / dav-2201 用
    {op}.asc              -- 同一份 .asc 骨架放到 arch22/ 目录即可
    CMakeLists.txt
```

- `{op}` 为 **snake_case** 算子名（如 `add`、`abs`、`layer_norm`、`swi_glu`）
- namespace 约定 `ops_multimodal_fusion::{Op}`，`{Op}` 为 **PascalCase**（如 `Add` / `LayerNorm`）
- 调用约定：`import torch; import torch_npu; import ops_multimodal_fusion` 后 `torch.ops.ops_multimodal_fusion.{op}(x.npu()).cpu()`
- CMakeLists.txt 内容固定（Huawei License 头 + 单行 `add_sources()`），无需为算子改动

---

## .asc 五段结构

单文件 `.asc` 把 host 与 device 代码全部内联，固定分为 5 段（参见 `applications/llm/add/arch35/add.asc`）：

| 段 | 内容 | 关键 API |
|----|------|---------|
| 1. Schema | 注册算子签名 | `TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m){ m.def("{op}(...)->..."); }` |
| 2. Meta | 推断输出 shape/dtype（不计算） | `{op}_meta(...)` + `torch::empty_like` + `TORCH_LIBRARY_IMPL(..., Meta, m)` |
| 3. Tiling | host 侧切核/切块参数 | `calc_tiling_params(...)`，用 `PlatformAscendCManager::GetInstance()` 取 UB（`GetCoreMemSize`）与核数（`GetCoreNumAiv`） |
| 4. Kernel | device 侧向量计算 | `template<typename T> __global__ __aicore__ __vector__ void {op}_kernel(...)`，内含 `TPipe` / `TQue<VECIN\|VECOUT>` / `InitBuffer` / `SetGlobalBuffer`(带多核偏移) / `DataCopyPad` |
| 5. Dispatch | NPU 直调入口 + 注册 | `{op}_npu(...)`：`OptionalDeviceGuard` → `getCurrentNPUStream` → `calc_tiling_params` → `data_ptr()` → `AT_DISPATCH_SWITCH`/`AT_DISPATCH_CASE` 内 `{op}_kernel<T><<<numBlocks, nullptr, stream>>>(...)` → `OpCommand::RunOpApi("{Op}", acl_call)` + `TORCH_LIBRARY_IMPL(..., PrivateUse1, m)` |

固定 includes：`<ATen/Operators.h>` `<torch/all.h>` `<torch/library.h>` `"torch_npu/csrc/core/npu/NPUStream.h"` `"torch_npu/csrc/framework/OpCommand.h"` `"kernel_operator.h"` `"platform/platform_ascendc.h"` `<type_traits>`。参数校验一律用 `TORCH_CHECK`。

> **`<<<>>>` 直调**：本仓不走注册式自定义算子工程，kernel 由 host 侧 `{op}_kernel<T><<<numBlocks, nullptr, stream>>>(...)` 直接拉起，`numBlocks` 为核数，`stream` 为当前 NPU stream。

---

## 目标架构

| 架构目录 | SOC_VERSION | NPU_ARCH | 说明 |
|---------|------------|----------|------|
| **arch35** | **ascend950** | **dav-3510** | 近乎唯一目标，130 个算子全部在此（默认） |
| arch22 | ascend910b / ascend910_93 | dav-2201 | 仅少数算子需要（当前仅 `abs`）；同一份 `.asc` 骨架放到 `arch22/` 目录即可 |

> 绝大多数算子只需写 `arch35/{op}.asc`。仅当算子确需在 910b/910_93 上提供时，才把同一骨架另置一份到 `arch22/{op}.asc`（`abs` 即此例）。

---

## 模板选型

按算子特征三选一：

| 选项 | 模板 | 适用场景 | 参照真实算子 |
|------|------|---------|-------------|
| ① **单文件 elementwise 骨架**（默认） | [references/applications/llm/{op}/arch35/{op}.asc](references/applications/llm/%7Bop%7D/arch35/%7Bop%7D.asc) + [CMakeLists.txt](references/applications/llm/%7Bop%7D/arch35/CMakeLists.txt) | 绝大多数算子：逐元素数学/科学函数，一读一写、按 tile 流水 | `add`（双输入）、`abs`（单输入） |
| ② **class 化 kernel 变体** | [references/variants/class-kernel-{op}.asc](references/variants/class-kernel-%7Bop%7D.asc) | 需行归约或多执行路径：normalization 类，kernel 封装成模板类 `Kernel{Op}<T>`（Init/Process），`__global__` 只做薄 launcher | `layer_norm` |
| ③ **复杂融合算子多头文件变体** | [references/variants/multi-header-{op}.asc](references/variants/multi-header-%7Bop%7D.asc) + [multi-header-{op}_impl.h](references/variants/multi-header-%7Bop%7D_impl.h) | tiling 复杂 / 多套模板实例的融合算子：`.asc` 只做 host 与注册薄层，kernel 与 tiling 拆到同目录头文件，`<<<>>>` 按值传 tiling struct | `swi_glu` |

选型要点：
- 默认选 ①。逐元素、无跨元素依赖的算子都用它。
- 出现「按行/按最后一维归约」「同精度需要多条执行路径」时选 ②。
- tiling 需精细求解、kernel 有多种类型/buffer 实例、单文件放不下时选 ③；`.asc` 变薄，复杂度进头文件。

---

## 命名与占位符

| 占位符 | 含义 | 示例 |
|-------|------|------|
| `{op}` | snake_case 算子名（目录名、文件名、schema、函数名） | `layer_norm` |
| `{Op}` | PascalCase（namespace、类名、`RunOpApi` 名） | `LayerNorm` |
| `{OP}` | UPPER（头文件 include guard） | `LAYER_NORM` |

---

## 使用方法

1. 按上表选定模板类型，复制对应模板到 `applications/llm/{op}/arch35/`（需要 arch22 时再复制一份到 `arch22/`）。
2. 全局替换占位符 `{op}` / `{Op}` / `{OP}` 为真实算子名。
3. 按模板中的 `// TEMPLATE:` 注释逐段填充：schema 参数、meta 校验与输出推断、tiling 的 `BUFFER_NUM`、kernel 的核心向量指令、dispatch 的 dtype 分支与 `RunOpApi` 名。
4. 同目录放固定内容的 `CMakeLists.txt`（Huawei 头 + `add_sources()`）。
5. 编译验证：`bash build.sh --ops={op}`（arch22 目标加 `--soc=ascend910b`），产出 whl 后 `pip install <whl> --force-reinstall --no-deps`，再 `pytest tests/{op}/ -v`。

---

## 参考资源

| 资源 | 路径 | 说明 |
|------|------|------|
| 单文件 elementwise 模板 | [references/applications/llm/{op}/arch35/{op}.asc](references/applications/llm/%7Bop%7D/arch35/%7Bop%7D.asc) | 规范五段单文件骨架（基于 `add.asc`），默认起点 |
| 固定 CMakeLists | [references/applications/llm/{op}/arch35/CMakeLists.txt](references/applications/llm/%7Bop%7D/arch35/CMakeLists.txt) | Huawei 头 + `add_sources()`，内容固定 |
| class 化 kernel 变体 | [references/variants/class-kernel-{op}.asc](references/variants/class-kernel-%7Bop%7D.asc) | `Kernel{Op}<T>` Init/Process + 薄 `__global__` launcher（基于 `layer_norm.asc`） |
| 多头文件融合变体 | [references/variants/multi-header-{op}.asc](references/variants/multi-header-%7Bop%7D.asc) | 薄 `.asc`，`#include` 同目录头 + `CalcTiling`→config + `<<<>>>` 传 tiling struct（基于 `swi_glu.asc`） |
| 融合变体伴随头 | [references/variants/multi-header-{op}_impl.h](references/variants/multi-header-%7Bop%7D_impl.h) | 示意 kernel/tiling 拆分（TilingData 结构体 / host CalcTiling / kernel 类 / launcher 薄壳） |
