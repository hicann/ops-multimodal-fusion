# 多模态融合算子库领域补充规则

> 本文件收录 ops-multimodal-fusion 仓特有的领域约束，补充 `repo-knowledge` SKILL.md 中库规范未覆盖的部分。这些规则从工作流硬规则中提取，去除工作流步骤引用后保留领域本质。

---

## 1. PyTorch 原生算子语义 / 精度对齐原则

新增算子必须与 **PyTorch 同名原生算子** 严格对齐，按算子类型确定对齐方式：

| 维度 | 对齐要求 |
|------|---------|
| **语义** | 输入 / 输出个数、shape 规则、边界行为（NaN 传播、符号约定等）与 PyTorch 同名算子一致 |
| **golden** | 参考值用 **PyTorch CPU** 计算（如 `a + b` / `torch.abs(x)` / `F.avg_pool2d(...)` / `torch.angle(x)`） |
| **精度** | float 用 `torch.allclose(result, expected, rtol=, atol=)`（典型 `1e-4` / `1e-3`）；int 用 `torch.equal` |
| **签名** | schema 的入参 / 属性（如 `avg_pool2d` 的 kernel_size / stride）与 PyTorch 对应 |

> 算子的语义归属在需求分析阶段确定（对标哪个 PyTorch 算子），后续开发严格遵循该算子的语义与精度基线。

---

## 2. Meta 函数形状 / dtype 推断正确性

Meta 函数（`{op}_meta`）不做计算，只推断输出的 **shape 与 dtype**，其正确性直接支撑 `torch.compile` 与 autograd：

- **无副作用**：只构造输出 tensor（`torch::empty_like` / `torch::empty(...)` 等），禁止访问数据、禁止启动 kernel。
- **形状推断必须准确**：输出 shape 要与 NPU 实际写入的一致，否则 `torch.compile` 追踪 / autograd 反传会拿到错误形状。
- **dtype 推断必须准确**：输出 dtype 要与 kernel 实际产出的一致（同 dtype 或按算子语义变换，如复数 → 实部）。
- **参数校验**：在 Meta 中用 `TORCH_CHECK` 做形状 / 属性校验（如 `x.sizes() == y.sizes()`），失败给出清晰信息。
- **单一来源**：`{op}_npu` 复用 `{op}_meta` 得到输出 tensor，禁止在两处重复维护形状推断逻辑。

**示例**：

```cpp
torch::Tensor add_meta(const torch::Tensor &x, const torch::Tensor &y)
{
    TORCH_CHECK(x.sizes() == y.sizes(), "The shapes of x and y must be the same.");
    return torch::empty_like(x);   // 形状 / dtype 与输入一致
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("add", add_meta);
}
```

---

## 3. dtype 支持与 AT_DISPATCH 分派

算子支持的数据类型通过 `AT_DISPATCH_SWITCH` / `AT_DISPATCH_CASE` **显式枚举**，每个 case 内实例化对应类型的 kernel：

- **支持面显式**：`AT_DISPATCH_CASE(torch::kFloat32, ...)` / `torch::kFloat16` / `torch::kInt32` 各自一个 case，未列出的 dtype 不支持。
- **类型映射**：
  | PyTorch dtype | case 内 `scalar_t` |
  |---------------|--------------------|
  | `torch::kFloat32` | `float` |
  | `torch::kFloat16` | `half`（AscendC half，**非** `float`） |
  | `torch::kInt32` | `int32_t` |
- **不支持的 dtype 必须干净报错**：未匹配到 case 时由 `AT_DISPATCH_SWITCH` 抛 `RuntimeError`（测试可用 `pytest.raises(RuntimeError, ...)` 覆盖，如对 `float64` 报错）。
- **kernel 模板化**：`{op}_kernel<scalar_t><<<...>>>` 以 `scalar_t` 实例化，同一份 kernel 覆盖多精度，禁止为每种精度手写重复 kernel。

---

## 4. NPU Dispatch：device guard 与空 tensor 短路

`{op}_npu`（`PrivateUse1` 键实现）的入口约束：

- NPU 实现必须注册到 **`PrivateUse1`** 分派键（`TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)`）。
- 入口用 `const c10::OptionalDeviceGuard guard(x.device());` 绑定当前 NPU 设备，保证后续 stream / 内存操作落在正确设备。
- **空 tensor 短路**：`if (x.numel() == 0) return z;` —— 直接返回 Meta 推断出的空输出，**不** 计算 tiling、**不** 启动 kernel。
- stream 取自 `c10_npu::getCurrentNPUStream().stream(false)`，随 `<<<numBlocks, nullptr, stream>>>` 下发。
- tensor 数据指针经 `(GM_ADDR)x.data_ptr()` 转为 device 地址传入 kernel；host 侧标量（tiling 参数等）作为 scalar 直接传参。
- 不允许在 kernel 侧访问 host 内存；host↔device 只通过 `GM_ADDR` 与 scalar 传递。

---

## 5. 测试对齐：结果搬回 CPU 与 golden 对比

`tests/{op}/test_{op}.py` 自包含（无 conftest / pytest.ini / CSV），验证套路固定：

- **接口存在性**：模块级 `if not hasattr(torch.ops.ops_multimodal_fusion, "{op}"): pytest.skip(..., allow_module_level=True)`，并有 `test_{op}_interface_exist` 断言。
- **设备门控**：`@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")`。
- **参数化**：模块级 `SHAPES` / `DTYPES`，用 `@pytest.mark.parametrize` 展开（含超 UB 容量的大 shape 以触发多 tile）。
- **执行 → 搬回对比**：`x.npu()` 上算，`result = torch.ops.ops_multimodal_fusion.{op}(x_npu).cpu()` 搬回 CPU，与 PyTorch CPU golden 对比。
- **精度判定**：float 用 `torch.allclose(result, expected, rtol=, atol=)`；int 用 `torch.equal(result, expected)`。
- **边界 / 异常**：空 tensor（`numel()==0`）、NaN 传播、非法 dtype（`pytest.raises`）等按算子语义补充覆盖。
