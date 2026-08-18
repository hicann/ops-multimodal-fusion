# 贡献指南

本项目欢迎广大开发者体验并参与贡献，在参与社区贡献之前，请参见[cann-community](https://gitcode.com/cann/community)了解行为准则，进行CLA协议签署，了解源码仓的贡献流程。

开发者准备本地代码与提交PR时需要重点关注如下几点：

1. 提交PR时，请按照PR模板仔细填写本次PR的业务背景、目的、方案等信息。
2. 若您的修改不是简单的bug修复，而是涉及到新增特性、新增接口、新增配置参数或者修改代码流程等，请务必先通过Issue进行方案讨论，以避免您的代码被拒绝合入。若您不确定本次修改是否可被归为"简单的bug修复"，亦可通过提交Issue进行方案讨论。

开发者贡献场景主要包括：

## 一、贡献新算子

本项目使用 fast kernel launch（`<<<>>>` 直调）方式实现算子，每个算子以单文件 `.asc` 形式提供。完整的贡献过程如下：

### 1. 创建Issue需求

新建 `Requirement|需求建议` 类Issue，并阐明新增算子的设计方案。Issue一般需包含以下内容：

- **背景信息**
- **价值/作用**
- **设计方案**

请在提交的Issue中评论`/assign @yourself` 认领该任务。

### 2. 需求评审

Sig组将指派Committer对您提交的Issue进行评审并反馈修改意见。请在完成修改后，于Issue中@对应Committer。

若需求被接纳，请将贡献算子提交至 `applications/llm/<op_name>/<arch>/` 对应目录。

### 3. 代码开发

需求评审通过后，推荐使用 CANNBot Agent 辅助完成算子开发与验证。Agent 会依据仓库内置的算子模板、编码规范与测试规范自动生成算子实现与测试代码，并完成编译与功能验证。

#### 3.1 初始化算子开发 workflow

在仓库根目录执行初始化脚本，准备 Agent 工作环境：

```bash
bash agent/init.sh <target>
```

其中 `<target>` 指定所使用的 CLI 工具，可选 `opencode` 或 `claude`。脚本会自动拉取 [cannbot-skills](https://gitcode.com/cann/cannbot-skills) 并完成工作区配置（如需使用本地 cannbot-skills，可追加 `--repo cannbot-skills:<path>` 参数）。

#### 3.2 启动 Agent 生成算子

按照脚本执行结束输出的 Quick Start 提示，启动对应的 CLI（`opencode` 或 `claude`），然后向 CANNBot 描述算子需求，Agent 将自动生成 `.asc` 算子实现、`CMakeLists.txt` 及测试文件，并完成编译与功能验证。例如：

```text
帮我开发一个 abs 算子，支持 float16，shape 主要是 [1,128]、[4,2048]
```

开发与验证通过后，即可按第 4 步要求整理交付件并提交 PR。

### 4. PR提交

算子交付件如下：

```text
applications/llm/${op_name}/                         # 算子名
├── arch22/                                         # 架构目录（按实际SoC选择）
│   ├── ${op_name}.asc                             # 算子实现（Schema注册 + Meta + Kernel + NPU Dispatch）
│   └── CMakeLists.txt                             # 编译配置（固定内容：add_sources()）
tests/${op_name}/                                   # 测试目录
│   └── test_${op_name}.py                         # 算子测试文件
```

PR上库要求：

- 代码交付件：需提供算子 `.asc` 实现文件、`CMakeLists.txt` 编译配置、测试文件。
- 精度要求：新贡献算子需满足精度标准，具体请参见[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)。无论是否使用 Agent，均须在 PR 描述中附上精度测试报告：
  - 若算子由第 3 步的 Agent 生成，请在 PR 描述中附上 Agent 输出的精度测试报告。
  - 若未使用 Agent，请借助 CANNBot 的[黑盒测试 skill](https://gitcode.com/cann/cannbot-skills/tree/master/ops/ascendc-st-design)、[白盒测试 skill](https://gitcode.com/cann/cannbot-skills/tree/master/ops/ascendc-whitebox-design) 生成用例进行验证，并在 PR 描述中附上精度测试报告。
  - 精度测试报告请参考[精度验收报告模版](docs/zh/develop/precision_acceptance_template.md)撰写。

- 合规检查：
  - 代码是否符合《[C++ 编程规范](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)》
  - 代码是否编译通过
  - Markdown文档语法是否符合规范
- PR提交：通过 `git` 命令提交目标分支PR，检查PR标题是否清晰、PR描述是否规范（指明更改内容和原因、是否关联对应Issue）、是否签署CLA。

### 5. CI门禁

通过评论 `compile` 指令触发开源仓门禁，并依据CI检测结果进行修改，目前CI门禁包含以下检查项：

- 代码编译
- 静态检查（如涉及codecheck误报，请提交给sig成员屏蔽）
- UT测试

门禁通过后，请在关联的Issue中@指派的Committer。

### 6. Committer检视

Committer检视后将反馈检视意见，请根据意见修改，完成后@指派的Committer。

### 7. Maintainer合入

Committer检视通过后，标注 `/lgtm`标签。Maintainer将在1天内进行最终审核，确认无问题后，将标注 `/approve` 标签合入PR。

## 二、算子Bug修复

如果您在本项目中发现了某些算子Bug，希望对其进行修复，欢迎您新建Issue进行反馈和跟踪处理。

您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#%E6%8F%90%E4%BA%A4Issue%E5%A4%84%E7%90%86Issue%E4%BB%BB%E5%8A%A1)指引新建 `Bug-Report|缺陷反馈` 类Issue对Bug进行描述，然后在评论框中输入"/assign"或"/assign @yourself"，将该Issue分配给您进行处理。

## 三、算子优化

如果您对本项目中某些算子实现有泛化性增强/性能优化思路，希望着手实现这些优化点，欢迎您对算子进行优化贡献。

您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#%E6%8F%90%E4%BA%A4Issue%E5%A4%84%E7%90%86Issue%E4%BB%BB%E5%8A%A1)指引新建 `Requirement|需求建议` 类Issue对优化点进行说明，并提供您的设计方案，然后在评论框中输入"/assign"或"/assign @yourself"，将该Issue分配给您进行跟踪优化。

## 四、帮助解决他人Issue

如果社区中他人遇到的问题您有合适的解决方法，欢迎您在Issue中发表评论交流，帮助他人解决问题和痛点，共同优化易用性。

如果对应Issue需要进行代码修改，您可以在Issue评论框中输入"/assign"或"/assign @yourself"，将该Issue分配给您，跟踪协助解决问题。
