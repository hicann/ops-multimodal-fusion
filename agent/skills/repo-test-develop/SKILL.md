---
name: repo-test-develop
description: |
  仓库测试开发指导：为算子编写 pytest test_{op}.py，用 PyTorch CPU 作为 golden 参考。
  自包含单文件用例（无 conftest / pytest.ini / CSV），交付于 tests/{op}/test_{op}.py。
  触发：为新算子补精度用例、编写带属性 / 边界 / 异常用例、执行 pytest 精度测试时加载。
  代码模板位于 references/ 目录，由调用方按需复制。
---

# ops-multimodal-fusion 算子测试开发技能

ops-multimodal-fusion 精度测试采用 **pytest 参数化 + 自包含单文件** 组织，使用 **PyTorch CPU（`torch` / `torch.nn.functional`）** 作为唯一 CPU golden 参考。开发新算子测试时，以 `references/tests/{op}/` 下的模板为起点，复制到 `tests/{op}/` 后按算子语义填充。

## 测试标杆体系（单层 PyTorch CPU Golden）

```
                      ┌─────────────────────────────┐
                      │   NPU 算子执行               │
                      │   torch.ops.               │
                      │   ops_multimodal_fusion.{op} │
                      └──────────┬──────────────────┘
                                 │ NPU 结果 (.cpu())
                                 ▼
                      ┌─────────────────────────────┐
                      │       精度验证               │
                      │  allclose / equal            │
                      └──────────┬──────────────────┘
                                 │
                      ┌──────────▼──────────────────┐
                      │   PyTorch CPU Golden         │
                      │  （唯一 CPU 参考基准）        │
                      └─────────────────────────────┘
```

| 实现 | 用途 | 依赖 |
|------|------|------|
| PyTorch CPU（`torch.*` / `torch.nn.functional.*`） | 唯一 CPU golden 参考，与 PyTorch 原生语义 / 精度对齐 | `torch`（CPU，requirements 已含） |

**PyTorch CPU 作为测试标杆的原因**：算子实现即以「一个 PyTorch 算子 = 一个 AscendC kernel」对标 CANN 9.0.0，golden 与被测语义同源；无额外构建依赖（`requirements.txt` 已含 torch CPU wheel）；直接调 `torch.abs` / `a + b` / `F.avg_pool2d` 现算即得参考，无需外部数据表；社区公认、精度对齐 PyTorch 原生。

## 测试交付件目录结构

一个算子的测试交付件位于 `tests/{op}/`，**仅 1 个自包含文件**（无 `conftest.py` / `pytest.ini` / CSV / golden 头文件）：

```
tests/{op}/
└── test_{op}.py          -- 自包含单文件用例（golden 内联现算，不共享 fixture、不读外部数据表）
```

一个算子一个测试文件；用例自带 SHAPES / DTYPES，golden 在用例内用 PyTorch CPU 现算。

## 测试文件结构（5 段）

| 段 | 内容 |
|----|------|
| ① 4 行 import | `torch` / `torch_npu`(noqa) / `pytest` / `ops_multimodal_fusion`(noqa)。带属性算子加 `import torch.nn.functional as F`；边界用例可加 `import math` / `import logging`。 |
| ② 模块级 NPU_ARCH 守卫 | `if not hasattr(torch.ops.ops_multimodal_fusion, "{op}"): pytest.skip(..., allow_module_level=True)`。当前 SoC/arch 未注册该算子时跳过整模块。 |
| ③ interface_exist | `test_{op}_interface_exist()`：`hasattr` 断言算子已注册到 `torch.ops.ops_multimodal_fusion` 命名空间。 |
| ④ 模块级常量 | `SHAPES`（含超 UB 容量的大 shape，强制多 tile）与 `DTYPES`（算子实际支持的 dtype）。 |
| ⑤ 参数化用例 | `@pytest.mark.skipif(not torch.npu.is_available())` + `@pytest.mark.parametrize("shape", SHAPES)` + `@pytest.mark.parametrize("dtype", DTYPES)`（双参数化）。 |

调用约定：`torch.ops.ops_multimodal_fusion.{op}(x.npu()).cpu()`。

## Golden 与对比容差

golden 一律用 PyTorch CPU 现算，随算子而定：

| 算子 | golden 参考 |
|------|-------------|
| add | `expected = a + b` |
| abs | `expected = torch.abs(a)` |
| avg_pool2d | `expected = F.avg_pool2d(a, ...)` |
| angle | `expected = torch.angle(x)` |

对比容差按 dtype / 算子类型选择——**float 用 `torch.allclose`，int 用 `torch.equal`**：

| NPU dtype / 类型 | 比对方式 | rtol / atol | 说明 |
|------------------|----------|-------------|------|
| float32 | `torch.allclose` | 1e-4 / 1e-4 | 逐元素数学 elementwise |
| float16 | `torch.allclose` | 1e-4 / 1e-4 | 半精度；角度 / 特殊函数类按 dtype 放宽到 5e-3 |
| 带属性 / 累加类（如 avg_pool2d） | `torch.allclose` | 1e-3 / 1e-3 | 多步累加，容差放宽 |
| int32 | `torch.equal` | 位精确 | 整数运算，不用 allclose |

角度 / 特殊函数按 dtype 定制容差（见进阶模板 `_dtype_tol`）。

## 代码模板与模板选型

模板目录 `references/tests/{op}/`，文件用 `{op}`（snake_case）/ `{Op}`（PascalCase）/ `{OP}`（UPPER）占位，需替换处以 `# TEMPLATE:` 注释标注：

| 交付文件 | 模板 | 适用 |
|---------|------|------|
| `test_{op}.py` | [references/tests/{op}/test_{op}.py](references/tests/{op}/test_{op}.py) | ① 基础 elementwise（add / abs 类） |
| `test_{op}.py`（进阶） | [references/tests/{op}/test_{op}_advanced.py](references/tests/{op}/test_{op}_advanced.py) | ② 带属性 + 边界 / 异常（avg_pool2d + angle 类） |

**模板选型**：

- **① 基础 elementwise**：无属性、逐元素、float / int 混合（add 双输入、abs 单输入）→ 复制 `test_{op}.py`，改 op 名、`DTYPES`、golden 一行即可。
- **② 带属性 / 边界**：算子带 `kernel_size` / `stride` 等属性，或需覆盖命名值 / NaN / 空 tensor / 非连续 / 非法 dtype → 复制 `test_{op}_advanced.py`，按需裁剪 Part A（属性）/ Part B（边界异常）。

## 带属性算子写法要点（参考 avg_pool2d）

- 用一个 `CONFIGS` 列表把 `(input_shape, attr1, attr2, ...)` 按 Schema 参数顺序打包，单 `@pytest.mark.parametrize("case", CONFIGS)`。
- golden 调 `F.xxx(a, **attrs)`；NPU 调 `torch.ops.ops_multimodal_fusion.{op}(a_npu, list(attr), ...)`（属性以 python list 传入，顺序与 Schema 一致）。
- **先断言 `result.shape == expected.shape`，再 `torch.allclose(rtol=1e-3, atol=1e-3)`**。
- CONFIGS 覆盖各属性分支组合：padding / ceil_mode / count_include_pad / divisor_override / 非方核 / stride 默认（传空）/ 大 shape 多 tile。

## 边界 / 异常写法要点（参考 angle）

- **命名值** `test_{op}_named_values`：`torch.tensor` 固定值（正 / 负 / 零 / 极大极小）+ PyTorch CPU 参考。
- **特殊输入**：NaN 传播（`torch.isnan`）、空 tensor（shape `(0,)` 直通、无 kernel launch）、非连续 tensor（`.t()` 转置）。当前 CANN 版本缺 D2D strided copy（error 561103），非连续用例可 `@pytest.mark.skip`，调用方自行 `contiguous()`。
- **非法 dtype 拒绝**：`with pytest.raises(RuntimeError, match="{op}"): torch.ops.ops_multimodal_fusion.{op}(bad)`——如 `float64` 必须干净报错（算子内 `TORCH_CHECK`）。
- 按 dtype 定制容差 helper `_dtype_tol(dtype)`。

## 开发流程

1. **确认对标**：算子已注册（`interface_exist`）、与 PyTorch 对标语义（哪个 `torch.*` / `F.*` 是 golden）。
2. **选模板**：基础 elementwise → `test_{op}.py`；带属性 / 边界 → `test_{op}_advanced.py`。复制到 `tests/{op}/test_{op}.py`。
3. **填充**：替换 `{op}` / `{Op}` / `{OP}` 占位；改 `DTYPES`、golden 一行；带属性补 `CONFIGS`。
4. **验证**：`pytest tests/{op}/ -v` 全绿。

## 执行与调试

```bash
pytest tests/ -v            # 全量
pytest tests/{op}/ -v       # 单算子
pytest tests/{op}/ -v -s    # 打开 stdout（看 logging / print）
```

前置：已 `pip install <whl> --force-reinstall --no-deps` 安装当前 SoC 的 wheel，且预装 `torch` / `torch_npu`。

- 无 NPU 环境：`skipif(not torch.npu.is_available())` 自动跳过用例；算子未注册：模块级 `pytest.skip` 跳整模块。
- 日志：env `ASCEND_GLOBAL_LOG_LEVEL`（0 debug / 1 info / 2 warning / 3 error）、`ASCEND_GLOBAL_EVENT_ENABLE`。
- 性能：`msprof --output=./prof_out python your_test.py`；Tensor 落盘 `DumpTensor`（env `ASCEND_WORK_PATH`）。

## 常见问题

| 现象 | 处理 |
|------|------|
| 整模块被 skip | 当前 SoC/arch 未注册该算子（NPU_ARCH 守卫命中）；确认 `build.sh --soc=<...> --ops={op}` 已编入并重装 whl |
| 全部用例 skip | `torch.npu.is_available()` 为 False；确认 `torch_npu` 与 NPU 驱动就绪 |
| float 精度 fail | 放宽 rtol/atol（累加 / 属性类用 1e-3；半精度按 dtype 定制）；打印 `max abs diff` 定位 |
| int 用例 fail | 整数必须 `torch.equal` 位精确，勿用 `allclose` |
| 非连续用例报 561103 | 当前 CANN 版本缺 D2D strided copy；该用例 `@pytest.mark.skip`，调用方自行 `contiguous()` |
| `import ops_multimodal_fusion` 失败 | 未安装 wheel 或 `torch` / `torch_npu` 未预装；重装 whl（`--force-reinstall --no-deps`） |
