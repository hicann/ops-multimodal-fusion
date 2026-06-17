# ops-multimodal-fusion

基于 AscendC 的 PyTorch 自定义算子库，使用 fast kernel launch（`<<<>>>` 直调）方式实现高性能 NPU 算子，编译为 Python wheel 包供直接安装使用。

## 版本配套

当前仓库已验证通过的 CANN 版本如下：

| CANN 版本 | 发布时间 | 分支 |
| --- | --- | --- |
| [CANN 9.0.0](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0) | `2026/04/30` | master |

请根据实际 CPU 架构和产品型号，从上述链接目录中自行选择对应的 `.run` 安装包。本仓库需要安装 **Toolkit 包** 和 **Ops 包**。

安装包文件名格式如下：

- Toolkit 包：
  - `Ascend-cann-toolkit_${cann_version}_linux-aarch64.run`
  - `Ascend-cann-toolkit_${cann_version}_linux-x86_64.run`
- Ops 包：
  - `Ascend-cann-${soc_name}-ops_${cann_version}_linux-aarch64.run`
  - `Ascend-cann-${soc_name}-ops_${cann_version}_linux-x86_64.run`

1. **安装 CANN Toolkit 包**

    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
    ```
    - `${cann_version}`：表示安装包版本号，此处为 `9.0.0`。
    - `${arch}`：表示 CPU 架构，如 `aarch64`、`x86_64`。
    - `${install_path}`：表示指定安装路径，默认安装在 `/usr/local/Ascend` 目录。

2. **安装 CANN Ops 包**

    Ops 包需与 Toolkit 包安装到同一目录下，请根据实际产品型号选择对应的 Ops 包。

    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
    ```
    - `${soc_name}`：表示 NPU 产品型号名称，请根据实际硬件选择，如 `950`、`910b` 等。

3. **配置环境变量**

   安装完成后，请执行：

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

   请将 `${install_path}` 替换为 Toolkit 包的实际安装目录，例如 `/usr/local/Ascend` 或 `${HOME}/Ascend`。

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
