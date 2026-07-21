---
name: repo-coding-rules
description: |
  ops-multimodal-fusion 仓 AscendC 编码规范 + MR 安全编码规则 + .asc/直调编码约定 + 校验与调试日志速查索引。
  作为编码与代码检视的对照清单，按需读取，不常驻上下文。
  触发：编写/修复算子或测试代码后自查、代码检视逐条核对规则、修复 codecheck 告警时加载。
  详细规则内容位于 references/ 目录，由调用方按需加载相关文档。
---

# AscendC 编码规范速查索引

本规则为**自查/检视清单**，不加载为常驻上下文。按场景由 agent 按需调用（本 skill 不感知自己在工作流中的位置，仅按场景提供规则）：

| 场景 | 加载方式 |
|------|---------|
| 编码/修复后自查 | 加载本 skill，按 `references/checklist.md` 8 步流程自查一次；发现违规查 `references/fix-guide.md` |
| 代码检视（逐条核对） | 对照 `references/mr-rules-essential.md`（严重/致命，须零违规）+ `references/mr-rules-general.md`（深度审查）+ `references/ascendc-r5-r10.md`（质量指标）做多维度检视 |
| codecheck 告警修复 | 按告警定位违规规则，依 `references/fix-guide.md` + `references/mr-rules-*.md` 修复 |
| 本仓 .asc/直调编码约定 | 查 `references/asc-coding-conventions.md`（单 .asc 一文件、5 段式结构、`<<<>>>` 直调、命名表、tiling scalar 传参、include 清单、OAT 指标） |
| 校验与调试日志 | 查 `references/validation-and-logging.md`（TORCH_CHECK 校验模板 + `ASCEND_GLOBAL_LOG_LEVEL` 运行时日志 + pytest -s / msprof / DumpTensor） |

## references 索引

### AscendC 编码规则（R1-R10）

- 每条规则独立成一个 reference 文件，包含错误/正确示例：

| 编号 | 规则 | reference |
|------|------|-----------|
| R1 | 禁止逐元素操作 | `references/R1-禁止逐元素操作.md` |
| R2 | 动态获取 CoreNum（禁止硬编码） | `references/R2-动态获取CoreNum.md` |
| R3 | TPipe 禁止作为成员变量 | `references/R3-TPipe禁止成员变量.md` |
| R4 | TilingData 禁止使用数组做核间分配 | `references/R4-TilingData禁止数组.md` |
| R5-R10 | 圈复杂度/嵌套深度/函数行数/除零防御/许可证头/extern 引用 | `references/ascendc-r5-r10.md` |

### MR 安全编码规则

- 按**严重等级**分组，方便代码检视时按优先级逐条核对：

| reference | 包含内容 | 适用场景 |
|-----------|---------|---------|
| `references/mr-rules-essential.md` | 严重/致命级规则（G.PRE.05、G.INC.*、G.FUU.09/10/12/13/15、G.MEM.04、G.STD.*、OAT 等 ~18 条） | MR 提交前必查 |
| `references/mr-rules-general.md` | 一般/建议级规则（G.EXP.*、G.CTL.03、G.AST.03、G.FUU.11/14、CQ.*、CIP.01 等 ~17 条） | 代码检视深度审查 |

### 检查流程与修复指南

| reference | 内容 | 适用场景 |
|-----------|------|---------|
| `references/checklist.md` | 8 步检查流程（文件级 → 头文件 → 函数级 → 表达式 → 安全函数 → 内存 → 标准库 → 冗余告警） | 提交前系统自查 |
| `references/fix-guide.md` | 14 种常见违规的修复方法对照表 | 发现违规后查修复方案 |

### 本仓 .asc/直调编码约定

| reference | 内容 | 适用场景 |
|-----------|------|---------|
| `references/asc-coding-conventions.md` | 单 .asc 每算子一文件、5 段式结构（Schema/Meta/Tiling/Kernel/NPU Dispatch 内联）、`{op}_kernel<T><<<numBlocks, nullptr, stream>>>` 直调、命名表、EXTENSION_MODULE_NAME 宏、platform_ascendc 取核数/UB 且 scalar 传参、TORCH_CHECK 校验、arch22/arch35 布局、include 清单、License Header | 编写/检视 .asc 时查本仓特定编码约束 |

### 校验与调试日志约定

| reference | 内容 | 适用场景 |
|-----------|------|---------|
| `references/validation-and-logging.md` | 参数校验用 `TORCH_CHECK(cond, "msg")`（shape/dtype/空 tensor 校验模板）+ 运行时调试日志靠 `ASCEND_GLOBAL_LOG_LEVEL`/`ASCEND_GLOBAL_EVENT_ENABLE` + pytest `-s` 打印 / msprof / DumpTensor；本仓不使用 OP_LOG/dlog/独立 host.cpp | 写参数校验、定位运行时问题、开启调试日志时查 |

## 使用示例

### 场景 1：编码自查

代码/测试编写或修复完成后调用本 skill 执行自检：

```
1. 加载 `references/checklist.md`，按 8 步流程逐条检查代码
2. 发现 R6 嵌套深度超标 → 加载 `references/ascendc-r5-r10.md` 查看修复方法
3. 发现 G.FUU.09 使用了 realloc → 加载 `references/mr-rules-essential.md` 确认级别与修复方案
4. 检查本仓 .asc 特有约束 → 加载 `references/asc-coding-conventions.md` 核对 5 段式结构 / `<<<>>>` 直调 / tiling scalar 传参 / include 清单
5. 修复完成后再次执行 checklist.md 直至通过
```

### 场景 2：代码检视

对变更文件做逐条核对时，加载：

```
1. `references/mr-rules-essential.md` 对照严重/致命规则做检视（必须零违规）
2. `references/mr-rules-general.md` 做深度审查（建议修复）
3. `references/ascendc-r5-r10.md` 检查代码质量指标（圈复杂度/嵌套深度/行数）
4. `references/asc-coding-conventions.md` 检查本仓 .asc 特有编码约束
```

### 场景 3：校验与调试日志

.asc 需要写参数校验、或定位运行时问题时，加载：

```
1. `references/validation-and-logging.md` 查 `TORCH_CHECK` 校验模板（shape/dtype/空 tensor）
2. 运行时定位问题时按其中说明设置 `ASCEND_GLOBAL_LOG_LEVEL` / 用 pytest `-s` 打印
3. 需要性能/中间结果时查 msprof / DumpTensor 用法
```

---

**注意**：本 skill 的 SKILL.md 仅作为索引与触发说明。Agent 在执行自查/检视时，应直接读取相应的 references 文件获取完整规则，**不要**在开发过程中常驻本 skill 的全部内容。
