# 算子调试调优

本文档介绍 ops-multimodal-fusion 项目中常见的算子调试和调优方法。

## 概述

在算子开发过程中，可能会遇到以下问题：

- 算子功能异常，输出结果不正确
- 算子性能不达标，需要优化
- 算子运行时报错

针对这些问题，本文档提供以下调试和调优方法：

- [日志调试](#日志调试)：通过日志定位问题
- [性能调优](#性能调优)：优化算子性能

## 日志调试

### 编译时日志

使用 `-v` 参数查看 pytest 详细输出：

```bash
pytest tests/ -v -s
```

### 运行时日志

CANN 提供了多种日志级别，可通过环境变量配置：

```bash
# 设置日志级别
export ASCEND_GLOBAL_LOG_LEVEL=3  # 0-debug, 1-info, 2-warning, 3-error

# 设置日志路径
export ASCEND_GLOBAL_EVENT_ENABLE=0
```

### Python 调试

在测试中添加打印语句：

```python
def test_${op_name}_debug():
    x = torch.randn(32, 64).npu()
    print(f"Input shape: {x.shape}, dtype: {x.dtype}")
    
    result = torch.ops.ops_multimodal_fusion.${op_name}(x)
    print(f"Output shape: {result.shape}, dtype: {result.dtype}")
    
    # 打印部分结果
    print(f"Output (first 10): {result.cpu()[:10]}")
```

## 性能调优

### 1. Tiling 优化

合理设置 Tiling 参数可以提高算子性能：

- **Block 切分**：根据核数均匀分配数据
- **UB 切分**：充分利用 Unified Buffer 空间
- **对齐要求**：确保数据地址和大小满足硬件对齐要求

### 2. 内存优化

- 减少内存拷贝次数
- 使用双缓冲（Double Buffer）技术
- 合理规划 workspace 大小

### 3. 计算优化

- 向量化计算：使用 Ascend C 向量指令
- 流水并行：合理使用多队列
- 指令融合：减少中间结果存储

## 调试工具

### msProf 性能分析

使用 msProf 工具进行性能分析：

```bash
msprof --output=./prof_out python your_test.py
```

### DumpTensor 数据导出

导出算子中间结果进行调试：

```bash
export ASCEND_WORK_PATH=./dump_out
export ASCEND_GLOBAL_LOG_LEVEL=0
```

## 性能测试示例

```python
import torch
import torch_npu
import time
import ops_multimodal_fusion

def benchmark_${op_name}():
    """性能测试"""
    x = torch.randn(1024, 1024).npu()
    
    # 预热
    for _ in range(10):
        _ = torch.ops.ops_multimodal_fusion.${op_name}(x)
    
    # 计时
    torch_npu.synchronize()
    start = time.time()
    
    for _ in range(100):
        _ = torch.ops.ops_multimodal_fusion.${op_name}(x)
    
    torch_npu.synchronize()
    end = time.time()
    
    avg_time = (end - start) / 100
    print(f"Average time: {avg_time * 1000:.2f} ms")
```

## 更多帮助

- [CANN 开发文档](https://www.hiascend.com/document)
- [Ascend C 性能优化指南](https://hiascend.com/document/redirect/CannCommunityAscendCPerf)
- [msProf 工具使用指南](https://hiascend.com/document/redirect/CannCommunityMsprof)