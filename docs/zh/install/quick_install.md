# 环境部署

您在学习 QuickStart 或各类教程操作之前，请先参考下面步骤完成基础环境搭建，确保已安装 NPU 驱动、固件和 CANN 软件（`Ascend-cann-toolkit`）等。

## 环境安装

本项目提供多种搭建昇腾环境的方式，请按需选择。

> **说明**：本文提到的编译态和运行态含义如下，请根据实际情况选择。
>
> - **编译态**：针对仅编译本项目不运行的场景，只需安装 CANN toolkit 包。
> - **运行态**：针对运行本项目的场景（编译运行或纯运行），需安装驱动与固件、CANN toolkit 包。

| 安装方式 | 使用说明 | 使用场景 |
| :--- | :--- | :--- |
| CANNLab | 一站式开发平台，提供在线直接运行的昇腾环境，无需手动安装。<br>当前可提供单机算力，**默认安装最新商发版 CANN 包**。 | 适用于没有昇腾设备的开发者。 |
| Docker | Docker 镜像是一种高效部署方式，已预集成 CANN 包和必备依赖。<br>当前适用于 Atlas A2/A3 和 950 系列产品，OS 支持 Ubuntu。<br>**默认安装最新商发版 CANN 包**。 | 适用有昇腾设备，需要快速搭建环境的开发者。 |
| 手动安装 | 手动安装 CANN 包和基础依赖，灵活性高。 | 适用有昇腾设备，想体验手动安装 CANN 包或体验最新 master 分支能力的开发者。 |

### 方式1：CANNLab

对于无昇腾设备的开发者，可直接使用 CANNLab 云开发环境，即"**一站式开发平台**"，该平台为您提供在线可直接运行的昇腾环境，环境中已安装必备的驱动固件、软件包和依赖，无需手动安装。

> **说明**：环境默认安装最新商发版 CANN 包，源码下载时注意与软件配套。更多关于开发平台的介绍请参考 [CANNLab 指导](https://gitcode.com/org/cann/discussions/54)。

1. 进入开源项目，单击 "CANNLab" 按钮，使用已认证过的华为云账号登录。若未注册或认证，请根据页面提示进行注册和认证。

   <img src="../figures/cloudIDE.png" alt="云平台"  width="750px" height="85px">

2. 根据页面提示创建 NPU 环境并配置规格，启动云开发环境后，单击 "连接 > WebIDE" 进入一站式开发平台。

   当前开源项目资源默认在 `/mnt/workspace/gitCode/${gitCode_id}` 目录下，$\{gitCode\_id\} 表示开发者个人 gitCode 账号。

   <img src="../figures/webIDE.png" alt="云平台"  width="1000px" height="150px">

### 方式2：Docker 部署

对于有昇腾设备的开发者，若您想快速搭建昇腾环境，可使用 Docker 镜像部署。

> **说明**：
>
> - 镜像文件比较大，下载需要一定时间，请您耐心等待。
> - 环境默认安装最新商发版 CANN 包，源码下载时注意与软件配套。

1. **安装驱动与固件（运行态依赖）**

   宿主机上昇腾驱动与固件的下载和安装操作请参考《[CANN 软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)》中"准备软件包"和"安装 NPU 驱动和固件"章节。驱动与固件是运行态依赖，若仅编译算子，可以不安装。

2. **下载镜像**

   - 步骤1：以 root 用户登录宿主机。确保宿主机已安装 Docker 引擎（版本 1.11.2 及以上）。
   - 步骤2：从昇腾镜像仓库拉取已预集成 CANN 软件包及所需依赖的镜像。命令如下，根据实际架构选择：

   ```bash
   # 示例：拉取 ARM 架构的 CANN 开发镜像
   docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
   # 示例：拉取 X86 架构的 CANN 开发镜像
   docker pull --platform=amd64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
   ```

3. **运行 Docker**

   拉取镜像后，需要以特定参数启动容器，以便容器内能访问宿主的昇腾设备。

   ```bash
   docker run --name cann_container --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -it swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops bash
   ```

   | 参数 | 说明 | 注意事项 |
   | :--- | :--- | :--- |
   | `--name cann_container` | 为容器指定名称，便于管理。 | 可自定义。 |
   | `--device /dev/davinci0` | 核心：将宿主机的 NPU 设备卡映射到容器内。 | 必须根据实际情况调整：`davinci0` 对应第 0 张 NPU 卡。请先执行 `npu-smi info` 查看设备号。 |
   | `--device /dev/davinci_manager` | 映射 NPU 设备管理接口。 | - |
   | `--device /dev/devmm_svm` | 映射设备内存管理接口。 | - |
   | `--device /dev/hisi_hdc` | 映射主机与设备间的通信接口。 | - |
   | `-v /usr/local/dcmi:/usr/local/dcmi` | 挂载设备管理接口相关工具和库。 | - |
   | `-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi` | 挂载 `npu-smi` 工具。 | 使容器内可查询 NPU 状态。 |
   | `-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/` | 关键：将宿主机 NPU 驱动库映射到容器内。 | - |
   | `-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info` | 挂载驱动版本信息文件。 | - |
   | `-v /etc/ascend_install.info:/etc/ascend_install.info` | 挂载 CANN 软件安装信息文件。 | - |
   | `-it` | `-i`（交互式）和 `-t`（分配伪终端）组合参数。 | - |
   | 镜像名称 | 指定要运行的 Docker 镜像。 | 请确保与 `docker pull` 的镜像名和标签一致。 |
   | `bash` | 容器启动后执行的命令。 | - |

### 方式3：手动安装

对于有昇腾设备的开发者，若您想手动搭建昇腾环境，请参考下述步骤。

#### 安装软件

- **场景1：体验 master 版本能力或基于 master 版本进行开发**

   1. **安装驱动与固件（运行态依赖）**

      下载和安装操作请参考《[CANN 软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)》中"准备软件包"和"安装 NPU 驱动和固件"章节。驱动与固件是运行态依赖，若仅编译算子，可以不安装。

   2. **安装 CANN toolkit 包**

      请单击 [下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)，选择最新时间版本，并根据产品型号和环境架构下载对应包。

      ```bash
      # 确保安装包具有可执行权限
      chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
      # 安装命令
      ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --quiet --install-path=${install_path}
      ```

      - $\{cann\_version\}：表示 CANN 包版本号。
      - $\{arch\}：表示 CPU 架构，如 aarch64、x86_64。
      - $\{install\_path\}：表示指定安装路径，root 用户默认安装在 `/usr/local/Ascend` 目录。

- **场景2：体验已发布版本能力或基于已发布版本进行开发**

   请访问 [CANN 官网下载中心](https://www.hiascend.com/cann/download)，选择发布版本，并根据产品型号和环境架构下载对应包，最后参考网页提供的命令完成安装。

#### 安装基础依赖

本项目基础依赖如下，注意满足版本号要求。

- Python >= 3.8（建议版本 <= 3.10）
- PyTorch >= 2.1.0
- torch_npu
- pip >= 20.0

安装命令：

```bash
pip install torch torch_npu
pip install -r requirements.txt
```

## 环境验证

安装完 CANN 包后，需验证环境和驱动是否正常。

- **检查 NPU 设备**

   ```bash
   # 运行 npu-smi，若能正常显示设备信息，则驱动正常
   npu-smi info
   ```

- **检查 CANN 版本**

   ```bash
   # 查看 CANN toolkit 包版本信息（默认路径安装）
   cat /usr/local/Ascend/ascend-toolkit/latest/opp/version.info
   ```

## 环境变量配置

按需选择合适的命令使环境变量生效。

```bash
# 默认路径安装，以 root 用户为例（非 root 用户，将 /usr/local 替换为 ${HOME}）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 指定路径安装
# source ${install_path}/ascend-toolkit/set_env.sh
```

## 源码下载

下载与 CANN 版本配套的分支源码，命令如下，$\{tag\_version\} 替换为分支标签名。

```bash
git clone -b ${tag_version} https://gitcode.com/cann/ops-multimodal-fusion.git
```