# 项目目录

项目全量目录层级介绍如下：

```text
├── cmake                                               # 项目工程编译模块
│   ├── ascend.cmake                                    # 检测 Ascend 工具链路径，设置 Bisheng 编译器
│   ├── func.cmake                                      # 提供 recursive_add_subdirectory / add_sources 宏
│   ├── python.cmake                                    # 检测 Python 解释器和开发库
│   ├── torch.cmake                                     # 检测 PyTorch
│   └── torch_npu.cmake                                 # 检测 torch_npu
├── applications                                        # 算子源码目录（按分类组织）
│   ├── llm                                             # 大模型相关算子分类
│       └── ${op_name}                                  # 算子工程目录，${op_name}表示算子名（小写下划线形式）
│           ├── arch22                                  # 架构目录，对应 Atlas A2/A3 系列产品
│           │   ├── ${op_name}.asc                      # 算子实现文件（Schema注册 + Meta + Kernel + NPU Dispatch）
│           │   └── CMakeLists.txt                      # 算子编译配置文件（固定内容：add_sources()）
│           └── arch35                                  # 架构目录，对应 950 系列产品
│               ├── ${op_name}.asc
│               └── CMakeLists.txt
│   └── autonomous_driving                              # 自动驾驶相关算子分类
│       ├── CMakeLists.txt                              # 类别编译入口（recursive_add_subdirectory）
│       ├── ops/                                        # AscendC kernel 算子容器（编入 whl）
│       │   └── ${op_name}                              # 算子工程目录（与 llm 同构）
│       │       └── arch35                              # 架构目录，对应 950 系列产品
│       │           ├── ${op_name}.asc                  # 算子实现文件（Schema注册 + Meta + Kernel + NPU Dispatch）
│       │           └── CMakeLists.txt                  # 算子编译配置文件（固定内容：add_sources()）
│       ├── common/ops/${op_name}.py                    # PyTorch 复合算子（fusionrepo 命名空间，源码 import）
│       └── model/                                      # 模型应用（PointPillars/CLOCS/HMFI 等）
├── ops_multimodal_fusion                               # Python 包目录
│   └── __init__.py                                     # import 时 glob + torch.ops.load_library 加载所有 so
├── tests                                               # 测试目录（按算子名称组织）
│   └── ${op_name}                                      # 算子测试目录
│       └── test_${op_name}.py                          # 算子测试文件
├── docs                                                # 项目相关文档目录
│   └── dir_structure.md                                # 目录结构说明文档
├── CMakeLists.txt                                      # 项目工程CMakeList入口
├── CONTRIBUTING.md                                     # 项目贡献指南文件
├── LICENSE                                             # 项目开源许可证信息
├── OAT.xml                                             # 配置脚本，代码仓工具使用，用于检查License是否规范
├── README.md                                           # 项目工程总介绍文档
├── SECURITY.md                                         # 项目安全声明文件
├── build.sh                                            # 项目工程编译脚本
├── requirements.txt                                    # 项目的第三方依赖包
└── setup.py                                            # Python wheel 构建入口
```
