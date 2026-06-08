# ops-multimodal-fusion

基于 AscendC 的 PyTorch 自定义算子库，使用 fast kernel launch（`<<<>>>` 直调）方式实现高性能 NPU 算子，编译为 Python wheel 包供直接安装使用。

## 仓编译、安装方式

### 环境部署

详见[环境部署](docs/zh/install/quick_install.md)。

### 编译

```bash
bash build.sh
```

该脚本依次执行：安装构建依赖 → 清理旧构建 → 编译 wheel 包。**不自动安装，也不跑测试。** 结束时打印 wheel 的绝对路径，以及可直接拷贝的安装与测试命令。

如需指定 SoC 类型（默认 `ascend950`）：

访问[CANN下载中心](https://www.hiascend.com/cann/download)，根据页面提示复制硬件查询命令，在当前环境中执行，返回芯片ID信息，再回填到官网按Enter键获取产品名，产品名对应的 `${soc_version}` 取值如下，请按实际场景传参。

- Atlas A2 训练系列产品/Atlas A2 推理系列产品：取值为 `ascend910b`
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：取值为 `ascend910_93`
- 950系列产品：取值为 `ascend950`

```bash
bash build.sh --soc=${soc_version}
```

编译产物为 `dist/ops_multimodal_fusion-1.0.0+<soc_version>-cp<编译环境Python版本>-cp<编译环境Python版本>-linux_*.whl`（如 `ops_multimodal_fusion-1.0.0+ascend950-cp310-cp310-linux_aarch64.whl`）。

### 安装

> **注意**：请勿在源码目录内执行 pip install，否则可能导致卸载异常。

```bash
cd /tmp && pip install /path/to/dist/ops_multimodal_fusion-*.whl --force-reinstall --no-deps
```

## 测试

```bash
pytest tests/ -v
```

## 接口使用

以 abs 算子为例：

```python
import torch
import torch_npu
import ops_multimodal_fusion

x = torch.randn(32, 64).npu()
result = torch.ops.ops_multimodal_fusion.abs(x)
```

## 接口支持清单

详见[接口支持清单](docs/zh/op_support_list.md)。

## 开发方式

详见[开发方式](docs/development.md)。

## 相关信息

- [快速入门](docs/QUICKSTART.md)
- [目录结构](docs/dir_structure.md)
- [接口列表](docs/zh/api_list.md)
- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-basic)

-----

- **问题反馈**：通过GitCode [Issues](https://gitcode.com/cann/ops-multimodal-fusion/issues) 提交问题。
- **社区互动**：通过GitCode [讨论](https://gitcode.com/cann/ops-multimodal-fusion/discussions) 参与交流。
