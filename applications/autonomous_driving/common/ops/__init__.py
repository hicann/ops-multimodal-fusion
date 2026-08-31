# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from . import operator_registry  # 算子 schema (DEF)，必须先于 impl

import importlib
import pkgutil
import pathlib

_pkg_dir = pathlib.Path(__file__).parent
for module in pkgutil.iter_modules([str(_pkg_dir)]):
    name = module.name
    if name.startswith("_") or name == "operator_registry":
        continue
    importlib.import_module(f"{__name__}.{name}")
