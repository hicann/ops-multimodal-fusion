# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Ops Multimodal Fusion"""
__version__ = "1.0.0"

import os

import torch

_pkg_dir = os.path.dirname(__file__)
_so_path = os.path.join(_pkg_dir, "libops_multimodal_fusion.so")
if not os.path.exists(_so_path):
    raise ImportError(
        f"libops_multimodal_fusion.so not found under {_pkg_dir}. "
        "Please make sure `ops_multimodal_fusion` is properly installed."
    )
torch.ops.load_library(_so_path)
