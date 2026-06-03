# 快速入门：基于 ops-multimodal-fusion 仓

## 使用须知

本指南旨在帮助您快速上手 CANN 和 `ops-multimodal-fusion` 算子仓的使用。为方便快速了解算子开发全流程，将以 **abs** 算子为实践对象，其源文件位于 `ops-multimodal-fusion/applications/llm/abs`，具体操作流程如下：

1. **[环境部署](zh/install/quick_install.md)**：完成软件包安装和源码下载，此处不再赘述。快速入门场景下，**推荐 CANNLab 或 Docker 环境**，安装操作简单。

   > **说明**：当前 CANNLab 或 Docker 环境默认最新商发版 CANN 包；如需体验 master 分支最新能力，可手动安装 CANN 包，注意软件与源码版本配套。

2. **[编译运行](#一编译运行)**：编译自定义算子包并安装，实现快速调用算子。

3. **[算子调用](#二算子调用)**：体验算子的调用方式。

4. **[算子调试](#三算子调试)**：掌握算子打印和性能采集方法。

## 一、编译运行

本阶段目的是**快速体验项目标准流程**，验证环境能否成功进行算子源码编译、打包、安装和运行。

### 1. 编译 abs 算子

环境准备好后（注意软件与源码版本配套），进入环境并访问项目源码根目录，编译算子。

通用编译命令格式：`bash build.sh --soc=<芯片版本>`。以 abs 算子为例，编译命令如下：

```bash
bash build.sh --soc=${soc_version}
```

$\{soc\_version\} 设置方法如下：

访问 [CANN下载中心](https://www.hiascend.com/cann/download)，根据页面提示复制硬件查询命令，在当前环境中执行，返回芯片 ID 信息，再回填到官网按 Enter 键获取产品名，产品名对应的 $\{soc\_version\} 取值如下，请按实际场景传参。

- Atlas A2 训练系列产品/Atlas A2 推理系列产品：取值为 `ascend910b`
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：取值为 `ascend910_93`
- 950 系列产品：取值为 `ascend950`

<img src="./zh/figures/socInfo.png" alt="芯片版本" width="800px" height="160px">

若提示如下信息，说明编译成功。

```bash
Build wheel success: dist/ops_multimodal_fusion-1.0.0-cp38-abi3-*.whl
```

编译成功后，wheel 包存放于项目根目录的 `dist/` 目录下。

### 2. 安装 ops-multimodal-fusion 包

```bash
pip install dist/ops_multimodal_fusion-1.0.0-cp38-abi3-*.whl --force-reinstall --no-deps
```

### 3. 快速验证：运行算子样例

通用的运行命令格式：`pytest tests/${op_name}/ -v`。

以 abs 算子为例，其提供了简单算子样例 `tests/abs/test_abs.py`，运行该样例验证算子功能是否正常。

```bash
pytest tests/abs/ -v
```

预期输出：打印算子 `abs` 的测试结果，表明算子已成功部署并正确执行。

```
tests/abs/test_abs.py::TestAbs::test_abs_interface PASSED
tests/abs/test_abs.py::TestAbs::test_abs_accuracy[...] PASSED
...
```

> **提示**：若已完成编译且仅需重新验证功能，可直接运行已编译的 pytest 测试，无需再次编译。

## 二、算子调用

### 调用方式

所有算子通过 PyTorch 扩展机制注册，调用方式统一：

```python
import torch
import torch_npu
import ops_multimodal_fusion

x = torch.randn(32, 64).npu()
result = torch.ops.ops_multimodal_fusion.abs(x)
```

### abs 算子示例

```python
import torch
import torch_npu
import ops_multimodal_fusion

# 创建输入张量并移至 NPU
x = torch.randn(32, 64, dtype=torch.float32).npu()

# 调用 abs 算子
result = torch.ops.ops_multimodal_fusion.abs(x)

# 输出仍在 NPU 上
print(result.shape)   # torch.Size([32, 64])
print(result.dtype)   # torch.float32
print(result.device)  # npu
```

## 三、算子调试

本阶段以 abs 算子为例，介绍算子调试和性能采集方法，以便后续问题分析定位。

### 1. 打印调试

算子如果出现执行失败、精度异常等问题，可在测试中添加打印进行问题分析和定位。

在 `tests/abs/test_abs.py` 中添加打印：

```python
def test_abs_debug():
    x = torch.randn(32, 64).npu()
    print(f"Input shape: {x.shape}, dtype: {x.dtype}")
    
    result = torch.ops.ops_multimodal_fusion.abs(x)
    print(f"Output shape: {result.shape}, dtype: {result.dtype}")
    
    # 打印部分结果
    print(f"Output (first 10): {result.cpu()[:10]}")
```

### 2. 性能采集

当算子功能验证正确后，可通过 `msprof` 工具采集算子性能数据。

- **运行 pytest 测试**

   ```bash
   pytest tests/abs/ -v
   ```

- **采集性能数据**

   ```bash
   msprof --output=./prof_out pytest tests/abs/test_abs.py
   ```

采集结果在项目目录下，msprof 命令执行完后会自动解析并导出性能数据结果文件，详细内容请参见 [msprof](https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/T&ITools/Profiling/atlasprofiling_16_0110.html)。

## 四、更多帮助

- [环境部署](zh/install/quick_install.md)
- [接口列表](zh/api_list.md)
- [算子开发指南](zh/develop/operator_development_guide.md)
- [算子调试调优](zh/debug/op_debug_prof.md)
- [CANN 开发文档](https://www.hiascend.com/document)
- [Ascend C API 参考](https://hiascend.com/document/redirect/CannCommunityAscendCApi)