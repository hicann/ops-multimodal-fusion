# 算子开发指南

## 目录结构

开发一个算子需要以下文件：

```
applications/llm/${op_name}/
├── arch22/                        # Atlas A2/A3 系列实现
│   ├── ${op_name}.asc             # 算子实现文件（Schema注册 + Meta + Kernel + NPU Dispatch）
│   └── CMakeLists.txt             # 编译配置（固定内容：add_sources()）
└── arch35/                        # 950 系列实现（可选）
    ├── ${op_name}.asc
    └── CMakeLists.txt

tests/${op_name}/                  # 测试目录（必须）
└── test_${op_name}.py             # 测试文件
```

**说明**：
- 所有算子统一放在 `applications/llm/` 目录下
- `${op_name}` 为算子名的小写下划线形式
- `arch22` 和 `arch35` 对应不同架构，可根据需要只实现其中一个
- wheel 构建与算子编译解耦：新增/删除算子无需改动 Python 代码

---

## 算子实现文件结构

### `${op_name}.asc` 文件

单文件包含算子的完整实现，由 4 个必要部分组成：

| 部分 | 宏 / 关键代码 | 作用 |
|------|--------------|------|
| **Schema 注册** | `TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)` | 向 PyTorch 声明算子签名（名称、输入输出类型），使其可通过 `torch.ops.ops_multimodal_fusion.${op_name}` 调用 |
| **Meta 函数** | `TORCH_LIBRARY_IMPL(..., Meta, m)` | 推断输出 tensor 的 shape 和 dtype，不执行实际计算；支撑 `torch.compile` 和 AutoGrad |
| **Kernel 实现** | `__global__ __aicore__ void ${op_name}_kernel(...)` | AscendC 设备端代码，在 AI Core 上执行实际计算逻辑 |
| **NPU Dispatch** | `TORCH_LIBRARY_IMPL(..., PrivateUse1, m)` | Host 端调度：分配输出 tensor、计算 tiling 参数、按 dtype 分发调用 kernel |

### 基本结构示例

```cpp
#include <torch/extension.h>
#include <torch/script.h>
#include "kernel_operator.h"

// ========== Schema 注册 ==========
TORCH_LIBRARY_FRAGMENT(ops_multimodal_fusion, m) {
    m.def("${op_name}(Tensor x) -> Tensor");
}

// ========== Meta 函数 ==========
TORCH_LIBRARY_IMPL(ops_multimodal_fusion, Meta, m) {
    m.impl("${op_name}", [](const torch::Tensor& x) {
        return torch::empty_like(x);
    });
}

// ========== Kernel 实现 ==========
using namespace AscendC;

extern "C" __global__ __aicore__ void ${op_name}_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR tiling) {
    // Kernel 类型声明
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    
    // 解析 Tiling 数据
    auto tilingData = (${OpName}TilingData*)tiling;
    
    // 核心计算逻辑
    // ...
}

// ========== NPU Dispatch ==========
TORCH_LIBRARY_IMPL(ops_multimodal_fusion, PrivateUse1, m) {
    m.impl("${op_name}", [](const torch::Tensor& x) {
        // 1. 分配输出 tensor
        torch::Tensor y = torch::empty_like(x);
        
        // 2. 计算 tiling 参数
        // ...
        
        // 3. 调用 kernel
        ${op_name}_kernel<<<...>>>(x.data_ptr(), y.data_ptr(), tiling);
        
        return y;
    });
}
```

---

## 编译配置

### `CMakeLists.txt` 文件

内容固定一行：

```cmake
add_sources()
```

作用：调用 `cmake/func.cmake` 中的宏，使用 Bisheng 编译器以 AscendC 模式编译该目录下的 `.asc` 文件，产出独立的 `libops_multimodal_fusion_${op_name}.so`。

---

## 测试文件

参见 [测试编写指南](test_writing_guide.md)。

**测试文件结构**：

```python
# tests/${op_name}/test_${op_name}.py

import torch
import torch_npu
import ops_multimodal_fusion
import pytest

def test_${op_name}_interface():
    """接口测试：验证算子已注册可用"""
    x = torch.randn(32, 64).npu()
    result = torch.ops.ops_multimodal_fusion.${op_name}(x)
    assert result.shape == x.shape

def test_${op_name}_accuracy():
    """精度测试：与 CPU 参考实现对比"""
    x = torch.randn(32, 64)
    x_npu = x.npu()
    
    # NPU 计算
    result_npu = torch.ops.ops_multimodal_fusion.${op_name}(x_npu)
    
    # CPU 参考计算
    result_cpu = torch.${op_name}(x)  # 或自定义参考实现
    
    # 对比结果
    assert torch.allclose(result_npu.cpu(), result_cpu, rtol=1e-5, atol=1e-5)
```

---

## 开发流程

### 步骤 1：创建目录和文件

```bash
# 创建算子目录
mkdir -p applications/llm/${op_name}/arch22
mkdir -p tests/${op_name}

# 创建文件
touch applications/llm/${op_name}/arch22/${op_name}.asc
touch applications/llm/${op_name}/arch22/CMakeLists.txt
touch tests/${op_name}/test_${op_name}.py
```

### 步骤 2：编写算子实现

在 `${op_name}.asc` 中实现 4 个部分：
1. Schema 注册
2. Meta 函数
3. Kernel 实现
4. NPU Dispatch

### 步骤 3：配置编译

在 `CMakeLists.txt` 中添加固定内容：
```cmake
add_sources()
```

### 步骤 4：编写测试

参考 [测试编写指南](test_writing_guide.md)。

### 步骤 5：编译验证

```bash
bash build.sh
pytest tests/${op_name}/ -v
```

---

## 关键概念

### PyTorch 扩展机制

| 层面 | 运行位置 | 职责 |
|------|---------|------|
| **Schema** | CPU | 声明算子签名，注册到 PyTorch |
| **Meta** | CPU | 推断输出 shape/dtype，支持编译和自动求导 |
| **NPU Dispatch** | CPU (Host) | 内存管理、Tiling 计算、调用 Kernel |
| **Kernel** | NPU AI Core | 实际计算逻辑 |

### Tiling

**目的**：将大任务切分成适合 NPU 执行的小块

**关键参数**：
- `totalLength` - 总数据量
- `usedCoreNum` - 使用多少个 AI Core
- `blockLength` - 每次迭代处理多少数据

### <<<>>> 核函数调用

```cpp
${op_name}_kernel<<<numBlocks, workspace, stream>>>(args...);
```

**参数说明**：
- `numBlocks` - 使用多少个 AI Core（Block）
- `workspace` - 共享内存指针，通常设置为 `nullptr`
- `stream` - ACL 执行流

---

## 完整示例

参见 `applications/llm/abs/` 目录：
- `arch22/abs.asc` - 算子完整实现
- `arch22/CMakeLists.txt` - 编译配置
- `tests/abs/test_abs.py` - 单元测试

---

## 相关文档

- [测试编写指南](test_writing_guide.md)
- [build 参数说明](../install/build.md)
- [项目目录](../install/dir_structure.md)