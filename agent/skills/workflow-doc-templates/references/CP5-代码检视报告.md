# {算子名称} 代码检视报告

> CP5 代码检视产出。对全部变更文件做多维度检视。不通过时回退算子开发（3.1）。

## 检视摘要

| 维度 | 结果 | 详情 |
|------|------|------|
| 代码规范 | {通过/有问题} | 引用 repo-coding-rules 条款 |
| 设计一致性 | {通过/有问题} | 与开发方案/Spec 是否一致 |
| 潜在风险 | {通过/有问题} | 越界、精度、并发等 |
| 冗余清理 | {通过/有问题} | |

## 变更范围

| 文件 | 变更类型 | 说明 |
|------|----------|------|

## OAT 合规检查

| 函数 | 圈复杂度 | 深度 | NCNB | 除零 | extern | 达标 |
|------|---------|------|------|------|--------|------|
| | ≤20 | ≤5 | ≤50 | 0 | 0 | |

## 问题清单

### 发现问题（HIGH）

| 序号 | 文件 | 行号 | 问题描述 | 修复建议 |
|------|------|------|---------|---------|

### 冗余代码（HIGH，零容忍）

| 序号 | 类型 | 位置 | 详情 |
|------|------|------|------|
| | 未使用 include | | |
| | 未调用函数 | | |

### 需关注（MED）

| 序号 | 文件 | 行号 | 问题描述 | 建议 |
|------|------|------|---------|------|

### 疑似（LOW）

| 序号 | 文件 | 行号 | 问题描述 |
|------|------|------|---------|

## 本仓 .asc / 直调专项检查

### .asc 结构规范

| 检查项 | 状态 |
|--------|------|
| 单 .asc 5 段完整（Schema / Meta / Tiling / Kernel / NPU Dispatch） | ✅/❌ |
| Schema 用 `TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)` 注册 | ✅/❌ |
| Meta 用 `TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)` 推断输出 shape/dtype | ✅/❌ |
| Dispatch 用 `TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)` 注册 | ✅/❌ |
| namespace `ops_multimodal_fusion::{Op}` 命名规范 | ✅/❌ |

### 参数校验

| 检查项 | 状态 |
|--------|------|
| 输入合法性一律用 `TORCH_CHECK` 校验（shape/dtype/属性） | ✅/❌ |
| 无残留 printf/std::cout | ✅/❌ |

### 直调 Kernel 规范

| 检查项 | 状态 |
|--------|------|
| `<<<numBlocks, nullptr, stream>>>` 直调参数正确 | ✅/❌ |
| `calc_tiling_params` 返回 scalar（numBlocks/blockLength/tileSize） | ✅/❌ |
| `SetGlobalBuffer` 多核地址偏移正确（`blockLength * GetBlockIdx()`） | ✅/❌ |
| `AT_DISPATCH_SWITCH`/`CASE` 按 scalar_type 分派，覆盖声明的 dtype | ✅/❌ |
| OptionalDeviceGuard + getCurrentNPUStream + RunOpApi 调用链完整 | ✅/❌ |
| DataCopyPad 搬入/搬出对齐正确 | ✅/❌ |

### include 精简

| 检查项 | 状态 |
|--------|------|
| 仅保留必要 include，无冗余头文件 | ✅/❌ |

## 文档检视

| 检查项 | 状态 | 说明 |
|--------|------|------|
| README 按模板结构完整 | ✅/❌ | |
| 参数表标注内存位置 | ✅/❌ | |
| 调用示例可运行 | ✅/❌ | |
| 产品支持情况完整 | ✅/❌ | |

## 总结

{整体质量评价}

## 检视结论

**结论**：{通过 / 不通过}

{不通过时汇总结构化修改意见，指明回退 3.1 的修改点}

## 附录：修订记录

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| v1.0 | {YYYY-MM-DD} | 初始检视 |
