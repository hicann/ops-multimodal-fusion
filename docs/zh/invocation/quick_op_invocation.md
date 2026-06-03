# 算子调用

## 前提条件

- 环境部署：调用项目算子之前，请先参考[环境部署](../install/quick_install.md)完成基础环境搭建。
- 调用算子列表：项目可调用的算子参见[接口列表](../api_list.md)。

## 编译执行

基于社区版 CANN 包对算子源码修改时，可采用如下方式进行源码编译：

### 编译 wheel 包

1. **编译 ops-multimodal-fusion 包**

    进入项目根目录，执行如下编译命令：

    ```bash
    # 编译项目
    bash build.sh
    ```

    若提示如下信息，说明编译成功。

    ```bash
    Build wheel success: dist/ops_multimodal_fusion-1.0.0-cp38-abi3-*.whl
    ```

    编译成功后，wheel 包存放于项目根目录的 `dist/` 目录下。

    如需指定 SoC 类型：

    ```bash
    bash build.sh --soc=${soc_version}
    ```

    SoC 类型取值：
    - Atlas A2 训练/推理系列：`ascend910b`
    - Atlas A3 训练/推理系列：`ascend910_93`
    - 950 系列：`ascend950`

2. **安装 ops-multimodal-fusion 包**

    ```bash
    pip install dist/ops_multimodal_fusion-1.0.0-cp38-abi3-*.whl --force-reinstall --no-deps
    ```

3. **（可选）卸载包**

    ```bash
    pip uninstall ops_multimodal_fusion
    ```

## 本地验证

通过 pytest 测试框架，可快速验证算子功能是否正常。

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行指定算子测试
pytest tests/${op_name}/ -v
```

执行测试后会打印执行结果，以 abs 算子为例：

```
tests/abs/test_abs.py::TestAbs::test_abs_interface PASSED
tests/abs/test_abs.py::TestAbs::test_abs_accuracy[shape0-float32] PASSED
tests/abs/test_abs.py::TestAbs::test_abs_accuracy[shape0-float16] PASSED
...
```

## 算子调用示例

### abs 算子

```python
import torch
import torch_npu
import ops_multimodal_fusion

# 创建输入张量并移至 NPU
x = torch.randn(32, 64).npu()

# 调用算子
result = torch.ops.ops_multimodal_fusion.abs(x)

# 输出仍在 NPU 上
print(result.shape)  # torch.Size([32, 64])
print(result.device)  # npu
```

### 调用方式说明

所有算子通过 PyTorch 扩展机制注册，调用方式统一：

```python
torch.ops.ops_multimodal_fusion.${op_name}(input_tensor)
```

**参数说明**：
- `${op_name}`：算子名称，如 `abs`
- `input_tensor`：输入张量，需先移至 NPU（`.npu()`）

### 返回值

算子返回结果张量，设备为 NPU。

如需将结果移至 CPU：

```python
result_cpu = result.cpu()
```

## 更多帮助

- [CANN 开发文档](https://www.hiascend.com/document)
- [Ascend C API 参考](https://hiascend.com/document/redirect/CannCommunityAscendCApi)
- [PyTorch 扩展文档](https://pytorch.org/docs/stable/notes/extending.html)