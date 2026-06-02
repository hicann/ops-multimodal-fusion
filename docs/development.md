# 开发方式

wheel 构建与算子编译解耦：新增/删除算子无需改动 Python 代码，也不需要重新链接单体 `_C.so`。

以在 `llm`（大模型算子）分类下添加名为 `foo` 的算子为例，需要提供以下交付件：

## 1. 算子实现：`applications/llm/foo/arch22/foo.asc`

单文件包含算子的完整实现，由 4 个必要部分组成：

| 部分 | 宏 / 关键代码 | 作用 |
|------|--------------|------|
| **Schema 注册** | `TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)` | 向 PyTorch 声明算子签名（名称、输入输出类型），使其可通过 `torch.ops.ops_multimodal_fusion.foo` 调用 |
| **Meta 函数** | `TORCH_LIBRARY_IMPL(..., Meta, m)` | 推断输出 tensor 的 shape 和 dtype，不执行实际计算；支撑 `torch.compile` 和 AutoGrad |
| **Kernel 实现** | `__global__ __aicore__ void foo_kernel(...)` | AscendC 设备端代码，在 AI Core 上执行实际计算逻辑 |
| **NPU Dispatch** | `TORCH_LIBRARY_IMPL(..., PrivateUse1, m)` | Host 端调度：分配输出 tensor、计算 tiling 参数、按 dtype 分发调用 kernel |

## 2. 编译配置：`applications/llm/foo/arch22/CMakeLists.txt`

内容固定一行：

```cmake
add_sources()
```

作用：调用 `cmake/func.cmake` 中的宏，使用 Bisheng 编译器以 AscendC 模式（`--npu-arch=${NPU_ARCH}`）编译该目录下的 `.asc` 文件，产出独立的 `libops_multimodal_fusion_foo.so` 直接输出到 `ops_multimodal_fusion/` 目录。

## 3. 测试用例：`tests/foo/test_foo.py`

验证算子的正确性，典型结构：

- **接口测试**：验证 `torch.ops.ops_multimodal_fusion.foo` 已注册可用
- **精度测试**：参数化多种 shape 和 dtype，将 NPU 计算结果与 CPU PyTorch 参考实现对比

## 交付件总结

```
新增文件：
  applications/llm/foo/arch22/foo.asc             # 算子实现（必须）
  applications/llm/foo/arch22/CMakeLists.txt      # 编译配置（必须）
  tests/foo/test_foo.py                            # 测试用例（必须）

无需修改任何已有文件 —— 构建系统会自动扫描 applications/ 下的新算子目录。
```
