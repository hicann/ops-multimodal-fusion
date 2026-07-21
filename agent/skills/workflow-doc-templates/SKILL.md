---
name: workflow-doc-templates
description: 交付件模板，提供设计文档、验收报告等中间交付件的格式模板。触发：产出需求/Spec/方案/验收报告/算子文档/开发日志/Issue 等交付件时，先加载对应模板作为格式基准。
---

# 交付件模板索引（ops-multimodal-fusion 专用）

本仓交付件的格式模板集中在 `references/` 下，文件名按工作流统一流程表编号 + 中文标题命名。产出对应交付件时，先读取模板作为格式基准，按占位符 `{...}` 填充实际内容。

> 本文件只做路由。模板正文在 `references/` 中按需加载，不在此内联。

## 模板索引

| 编号 | 交付件 | 模板 | 说明 |
|------|--------|------|------|
| 0 | 开发环境信息 | [references/0-环境信息.md](references/0-环境信息.md) | 环境检查、NPU 设备、CANN 版本、SOC 确认 |
| 1.1 | 需求分析 | [references/1.1-需求分析.md](references/1.1-需求分析.md) | 算子数学定义、PyTorch 原生语义对标、参数规格、支持 dtype/shape、约束 |
| 1.2 | Spec | [references/1.2-Spec.md](references/1.2-Spec.md) | spec.yaml 生成指引（机器可校验的 L0 数学契约） |
| 2.1 | 测试方案设计 | [references/2.1-测试方案设计.md](references/2.1-测试方案设计.md) | pytest + PyTorch CPU golden 方案、shape/dtype 覆盖、边界/异常用例 |
| 2.2 | 开发方案设计 | [references/2.2-开发方案设计.md](references/2.2-开发方案设计.md) | .asc 单文件 kernel 设计、Tiling 策略、Meta 与 NPU Dispatch |
| CP3 | 功能验收报告 | [references/CP3-功能验收报告.md](references/CP3-功能验收报告.md) | 编译验证、精度统计、pytest 通过率、测试代码完整性 |
| CP4 | 性能验收报告 | [references/CP4-性能验收报告.md](references/CP4-性能验收报告.md) | 性能数据、瓶颈分析、shape/dtype 维度 |
| CP5 | 代码检视报告 | [references/CP5-代码检视报告.md](references/CP5-代码检视报告.md) | OAT 合规、TORCH_CHECK 校验、冗余代码、文档检视 |
| 6.1 | 算子文档 | [references/6.1-算子文档.md](references/6.1-算子文档.md) | README 模板（torch.ops.ops_multimodal_fusion.{op} 调用示例、支持 dtype/shape、产品支持） |
| 7.1 | 开发报告 | [references/7.1-开发报告.md](references/7.1-开发报告.md) | 交付物清单、开发过程、关键指标 |
| 7.2 | 经验总结 | [references/7.2-经验总结.md](references/7.2-经验总结.md) | 有效经验、踩坑记录、可复用产物 |
| — | Issue 问题记录 | [references/Issue-问题记录.md](references/Issue-问题记录.md) | 开发中的问题独立成文（Background/问题描述/根因分析/解决过程/影响与预防） |
| — | 开发日志 | [references/LOG-开发日志.md](references/LOG-开发日志.md) | 全程状态跟踪、进度跟踪（按统一流程编号） |
