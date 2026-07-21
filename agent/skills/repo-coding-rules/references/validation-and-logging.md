# 校验与调试日志约定

> 本文件收录 ops-multimodal-fusion 仓的参数校验与调试日志约定。
>
> **本仓不使用 `OP_LOG` / `dlog` 宏，也没有独立 `host.cpp` 日志层。** 参数校验一律走 PyTorch 的 `TORCH_CHECK`；运行时调试靠 CANN 环境变量与 pytest。

---

## 1. 参数校验：TORCH_CHECK

所有 host 侧输入合法性校验（shape、dtype、空 tensor 等）统一用 `TORCH_CHECK(cond, "msg")`，通常放在 Meta 函数里（Meta 会被 NPU Dispatch 复用，校验只写一次）。校验失败会抛出可读的 Python 异常。

```cpp
TORCH_CHECK(cond, "human-readable message");
```

- 第 1 参：布尔条件，为假即抛异常。
- 第 2 参起：错误信息，可用 `,` 拼接多个片段（`c10::str` 风格，支持流式拼接变量）。
- 不返回错误码、不打印日志——直接抛异常，PyTorch 侧 `pytest.raises` 可捕获。

### 模板：shape 校验

```cpp
// 取材 add.asc 的 add_meta
torch::Tensor add_meta(const torch::Tensor &x, const torch::Tensor &y)
{
    TORCH_CHECK(x.sizes() == y.sizes(), "The shapes of x and y must be the same.");
    return torch::empty_like(x);
}
```

### 模板：dtype 校验

```cpp
TORCH_CHECK(x.scalar_type() == torch::kFloat32 ||
            x.scalar_type() == torch::kFloat16 ||
            x.scalar_type() == torch::kInt32,
            "unsupported dtype: ", x.scalar_type());
```

### 模板：空 tensor / 维度校验

```cpp
TORCH_CHECK(x.dim() == 4, "input must be 4-D (N,C,H,W), got dim=", x.dim());
```

- 空 tensor 一般**不报错**，而是在 `{op}_npu` 中早返回：`if (x.numel() == 0) { return z; }`。仅当空输入语义非法时才 `TORCH_CHECK(x.numel() > 0, "input must not be empty.")`。

### 测试侧对应

`tests/{op}/test_{op}.py` 用 `pytest.raises` 验证非法输入被拒（真实见 `tests/angle/test_angle.py`）：

```python
with pytest.raises(Exception):
    torch.ops.ops_multimodal_fusion.{op}(bad_input.npu())
```

---

## 2. 运行时调试日志：环境变量

本仓不在算子内打日志；运行时问题靠 CANN 全局日志开关定位。参见 `docs/zh/debug/op_debug_prof.md`。

| 环境变量 | 说明 | 示例 |
|---------|------|------|
| `ASCEND_GLOBAL_LOG_LEVEL` | 全局日志级别：`0`=debug / `1`=info / `2`=warning / `3`=error | `export ASCEND_GLOBAL_LOG_LEVEL=0` |
| `ASCEND_GLOBAL_EVENT_ENABLE` | 事件日志开关（0 关 / 1 开） | `export ASCEND_GLOBAL_EVENT_ENABLE=0` |

```bash
# 开发调试：开到 debug 级
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_GLOBAL_EVENT_ENABLE=0

# 只看错误
export ASCEND_GLOBAL_LOG_LEVEL=3
```

---

## 3. pytest -s 打印调试

在测试里直接 `print` 中间结果，配合 `pytest -s`（`-s` 关闭输出捕获）查看：

```bash
pytest tests/{op}/ -v -s
```

```python
def test_{op}_debug():
    x = torch.randn(32, 64).npu()
    print(f"Input shape: {x.shape}, dtype: {x.dtype}")
    result = torch.ops.ops_multimodal_fusion.{op}(x)
    print(f"Output (first 10): {result.cpu().flatten()[:10]}")
```

---

## 4. msprof 性能分析

```bash
msprof --output=./prof_out python your_test.py
```

- 生成 profiling 数据到 `./prof_out`，用于分析 Task Duration、核利用率等指标。

---

## 5. DumpTensor 导出中间结果

通过环境变量开启算子中间结果落盘，用于比对 golden：

```bash
export ASCEND_WORK_PATH=./dump_out
export ASCEND_GLOBAL_LOG_LEVEL=0
```

---

## 6. 明确不使用的手段

| 手段 | 状态 | 本仓替代 |
|------|------|---------|
| `OP_LOG*` 日志宏 | ❌ 不使用 | `TORCH_CHECK`（校验）+ 环境变量日志（运行时） |
| `dlog` 日志族 | ❌ 不使用 | 同上 |
| 独立 `host.cpp` 日志层 | ❌ 无此文件 | 全部内联在单 `.asc` |
| 返回错误码（如 `*_STATUS_*`） | ❌ 不使用 | `TORCH_CHECK` 抛异常 |
