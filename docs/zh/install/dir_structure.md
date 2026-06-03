# 项目目录

> 本章罗列的部分目录或文件为可选，请以实际交付件为准。另请注意：
>
> - **算子目录**：`applications/llm/${op_name}/` 下承载算子所有交付件，目录介绍参见本文档。
> - **算子目录命名**：`${op_name}` 为算子名的小写下划线形式（如 `abs`）。
> - **架构子目录**：`arch22` 对应 Atlas A2/A3 系列，`arch35` 对应 950 系列，由 SOC 配置决定是否编译。
> - 若需补充算子或文档，欢迎参考 [贡献指南](../../../CONTRIBUTING.md)。

项目全量目录层级介绍如下：

```text
├── cmake                                               # 工程辅助 CMake 脚本
│   ├── ascend.cmake                                    # 检测 Ascend 工具链，设置 Bisheng 编译器
│   ├── func.cmake                                      # 提供 add_sources 宏
│   ├── python.cmake                                    # 检测 Python 解释器和开发库
│   ├── torch.cmake                                     # 检测 PyTorch
│   └── torch_npu.cmake                                 # 检测 torch_npu
├── applications                                        # 算子源码目录
│   └── llm                                             # 大模型算子分类
│       └── ${op_name}                                  # 算子目录（如 abs）
│           ├── arch22                                  # Atlas A2/A3 实现
│           │   ├── ${op_name}.asc                      # 算子实现文件
│           │   └── CMakeLists.txt                      # 编译配置
│           └── arch35                                  # 950 系列实现（可选）
│               ├── ${op_name}.asc
│               └── CMakeLists.txt
├── ops_multimodal_fusion                               # Python 包目录
│   └── __init__.py                                     # 包初始化，自动加载 so
├── tests                                               # 算子测试目录
│   └── ${op_name}                                      # 算子测试子目录
│       └── test_${op_name}.py                          # pytest 测试文件
├── docs                                                # 项目文档
│   ├── QUICKSTART.md                                   # 快速入门
│   ├── development.md                                  # 开发方式说明
│   ├── dir_structure.md                                # 目录结构说明
│   ├── README.md                                       # 文档入口
│   └── zh                                              # 中文文档
│       ├── api_list.md                                 # 接口列表
│       ├── figures                                     # 图片资源
│       │   ├── cloudIDE.png                            # CANNLab 云平台截图
│       │   ├── socInfo.png                             # 芯片版本查询截图
│       │   └── webIDE.png                              # WebIDE 截图
│       ├── install                                     # 安装部署文档
│       │   ├── dir_structure.md
│       │   └── quick_install.md
│       ├── debug                                       # 调试文档
│       │   └── op_debug_prof.md
│       ├── develop                                     # 开发文档
│       │   ├── operator_development_guide.md
│       │   └── test_writing_guide.md
│       └── invocation                                  # 调用文档
│           └── quick_op_invocation.md
├── CMakeLists.txt                                      # 工程入口 CMakeLists
├── CHANGELOG.md                                        # 变更日志
├── CONTRIBUTING.md                                     # 贡献指南
├── LICENSE                                             # 许可证
├── OAT.xml                                             # OAT 配置
├── README.md                                           # 项目说明
├── SECURITY.md                                         # 安全声明
├── Third_Party_Open_Source_Software_List.yaml          # 第三方开源列表
├── Third_Party_Open_Source_Software_Notice             # 第三方开源声明
├── build.sh                                            # 编译脚本
├── requirements.txt                                    # Python 依赖
├── setup.py                                            # wheel 构建入口
├── .clang-format                                       # 代码风格配置
└── .gitignore                                          # Git 忽略规则
```

## 目录说明

### 核心目录

| 目录/文件 | 说明 |
| :--- | :--- |
| `applications/` | 算子源码目录，包含所有算子实现 |
| `applications/llm/` | 大模型算子分类目录 |
| `ops_multimodal_fusion/` | Python 包目录 |
| `cmake/` | CMake 编译配置模块 |

### 文档目录

| 目录/文件 | 说明 |
| :--- | :--- |
| `docs/` | 项目文档目录 |
| `docs/QUICKSTART.md` | 快速入门指南 |
| `docs/zh/` | 中文文档目录 |
| `docs/zh/api_list.md` | 接口列表 |
| `docs/zh/install/` | 安装部署文档 |
| `docs/zh/develop/` | 开发相关文档 |
| `docs/zh/debug/` | 调试调优文档 |
| `docs/zh/invocation/` | 算子调用文档 |

### 测试目录

| 目录/文件 | 说明 |
| :--- | :--- |
| `tests/` | 测试目录 |
| `tests/${op_name}/` | 各算子测试用例目录 |

### 构建相关

| 目录/文件 | 说明 |
| :--- | :--- |
| `build.sh` | 编译脚本 |
| `setup.py` | Python wheel 构建入口 |
| `CMakeLists.txt` | CMake 配置入口 |

### 配置文件

| 文件 | 说明 |
| :--- | :--- |
| `.clang-format` | 代码风格配置 |
| `.gitignore` | Git 忽略规则 |
| `requirements.txt` | Python 依赖列表 |
| `OAT.xml` | OAT 配置 |