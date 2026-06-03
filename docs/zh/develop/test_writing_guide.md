# 算子测试编写指南

## 文件结构

写一个算子的测试需要 **1 个测试文件**：

```
tests/${op_name}/
└── test_${op_name}.py    # 测试文件（必须）
```

**说明**：使用 pytest 测试框架，测试文件命名为 `test_${op_name}.py`。

---

## 测试文件结构

### 必须包含的 2 个部分

#### 1. 接口测试

验证算子已注册可用，调用方式正确：

```python
def test_${op_name}_interface():
    """接口测试：验证算子已注册可用"""
    import torch
    import torch_npu
    import ops_multimodal_fusion
    
    x = torch.randn(32, 64).npu()
    result = torch.ops.ops_multimodal_fusion.${op_name}(x)
    
    # 验证输出形状
    assert result.shape == x.shape
    # 验证输出在 NPU 上
    assert result.device.type == 'npu'
```

#### 2. 精度测试

将 NPU 计算结果与 CPU 参考实现对比：

```python
@pytest.mark.parametrize("shape", [(32, 64), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_${op_name}_accuracy(shape, dtype):
    """精度测试：与 CPU 参考实现对比"""
    import torch
    import torch_npu
    import ops_multimodal_fusion
    
    # 创建测试数据
    x = torch.randn(shape, dtype=dtype)
    x_npu = x.npu()
    
    # NPU 计算
    result_npu = torch.ops.ops_multimodal_fusion.${op_name}(x_npu)
    
    # CPU 参考计算（使用 PyTorch 内置函数或自定义实现）
    with torch.no_grad():
        result_cpu = torch.abs(x)  # 对于 abs 算子
    
    # 对比结果
    rtol = 1e-5 if dtype == torch.float32 else 1e-3
    atol = 1e-5 if dtype == torch.float32 else 1e-3
    assert torch.allclose(result_npu.cpu(), result_cpu, rtol=rtol, atol=atol)
```

---

## 完整示例

**test_abs.py**

```python
import torch
import torch_npu
import pytest

# 确保算子库已加载
import ops_multimodal_fusion


class TestAbs:
    """abs 算子测试类"""
    
    def test_abs_interface(self):
        """接口测试：验证算子已注册可用"""
        x = torch.randn(32, 64).npu()
        result = torch.ops.ops_multimodal_fusion.abs(x)
        
        assert result.shape == x.shape
        assert result.device.type == 'npu'
    
    @pytest.mark.parametrize("shape", [
        (32, 64),
        (128, 256),
        (1024, 1024),
        (1, 1),
        (10,),
    ])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_abs_accuracy(self, shape, dtype):
        """精度测试：与 CPU 参考实现对比"""
        # 创建测试数据
        x = torch.randn(shape, dtype=dtype)
        x_npu = x.npu()
        
        # NPU 计算
        result_npu = torch.ops.ops_multimodal_fusion.abs(x_npu)
        
        # CPU 参考计算
        result_cpu = torch.abs(x)
        
        # 对比结果
        rtol = 1e-5 if dtype == torch.float32 else 1e-3
        atol = 1e-5 if dtype == torch.float32 else 1e-3
        assert torch.allclose(result_npu.cpu(), result_cpu, rtol=rtol, atol=atol)
    
    def test_abs_negative_values(self):
        """边界测试：验证负数处理"""
        x = torch.tensor([-1.0, -2.0, -3.0]).npu()
        result = torch.ops.ops_multimodal_fusion.abs(x)
        expected = torch.tensor([1.0, 2.0, 3.0])
        
        assert torch.allclose(result.cpu(), expected)
    
    def test_abs_zero(self):
        """边界测试：验证零值处理"""
        x = torch.zeros(10).npu()
        result = torch.ops.ops_multimodal_fusion.abs(x)
        
        assert torch.allclose(result.cpu(), torch.zeros(10))
```

---

## pytest 常用功能

### 参数化测试

```python
@pytest.mark.parametrize("shape", [(32, 64), (128, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_${op_name}(shape, dtype):
    # 测试逻辑
```

### 测试类组织

```python
class Test${OpName}:
    def test_interface(self):
        # 接口测试
    
    def test_accuracy(self):
        # 精度测试
    
    def test_boundary(self):
        # 边界测试
```

### 标记测试

```python
@pytest.mark.skip(reason="待实现")
def test_future_feature():
    pass

@pytest.mark.slow
def test_large_shape():
    pass
```

---

## 运行测试

### 运行单个算子测试

```bash
pytest tests/${op_name}/ -v
```

### 运行所有测试

```bash
pytest tests/ -v
```

### 运行特定测试

```bash
pytest tests/${op_name}/test_${op_name}.py::Test${OpName}::test_accuracy -v
```

### 带详细输出

```bash
pytest tests/ -v -s
```

---

## 测试最佳实践

1. **接口测试**：验证算子可调用，输出形状正确
2. **精度测试**：使用参数化覆盖多种 shape 和 dtype
3. **边界测试**：测试特殊值（零、负数、极大/极小值）
4. **性能测试**：可选，测量算子执行时间

---

## 参考示例

完整示例：`tests/abs/test_abs.py`