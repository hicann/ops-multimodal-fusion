# 项目文档

## 目录说明

关键目录结构如下：

```
docs/
├── QUICKSTART.md                        # 快速入门
├── development.md                       # 开发方式说明
├── dir_structure.md                     # 目录结构说明
├── README.md                            # 文档总入口
└── zh/                                  # 中文文档目录
    ├── api_list.md                      # 接口列表
    ├── figures/                         # 图片资源目录
    │   ├── cloudIDE.png                 # CANNLab 云平台截图
    │   ├── socInfo.png                  # 芯片版本查询截图
    │   └── webIDE.png                   # WebIDE 截图
    ├── install/                         # 安装部署目录
    │   ├── dir_structure.md             # 项目目录结构
    │   └── quick_install.md             # 环境部署指南
    ├── debug/                           # 调试调优文档目录
    │   └── op_debug_prof.md             # 算子调试调优
    ├── develop/                         # 开发文档目录
    │   ├── operator_development_guide.md    # 算子开发指南
    │   └── test_writing_guide.md        # 测试编写指南
    └── invocation/                      # 算子调用文档目录
        └── quick_op_invocation.md       # 算子调用指南
```

## 文档列表

| 文档 | 说明 |
| :--- | :--- |
| [快速入门](QUICKSTART.md) | 从零开始快速体验项目核心基础能力 |
| [开发方式](development.md) | 介绍项目的开发方式和算子交付件 |
| [目录结构](dir_structure.md) | 介绍项目完整的目录结构 |
| [实现状态](implementation.md) | 介绍项目 API 实现进度 |
| [接口列表](zh/api_list.md) | 介绍项目包含的所有算子接口 |
| [环境部署](zh/install/quick_install.md) | 介绍项目的基础环境搭建 |
| [项目目录](zh/install/dir_structure.md) | 介绍项目完整的目录结构和各目录/文件的作用 |
| [算子开发指南](zh/develop/operator_development_guide.md) | 介绍如何从零开发一个新算子 |
| [测试编写指南](zh/develop/test_writing_guide.md) | 介绍如何快速编写算子测试文件 |
| [算子调用指南](zh/invocation/quick_op_invocation.md) | 介绍算子编译、安装和调用方法 |
| [算子调试调优](zh/debug/op_debug_prof.md) | 介绍常见的算子调试、调优方法 |