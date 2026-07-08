/*!
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DYNAMIC_QUANT_STORY_LOCAL_DEPS_H
#define DYNAMIC_QUANT_STORY_LOCAL_DEPS_H

#include "kernel_operator.h"
#include "dynamic_quant_tiling.h"

// Framework constants normally set by build system.
// Since this standalone story doesn't use int4 output, ORIG_DTYPE_Y != DT_INT4 always.
#ifndef DT_INT4
#define DT_INT4 29
#endif
#ifndef ORIG_DTYPE_Y
#define ORIG_DTYPE_Y 2
#endif

// Template mode constants (from dynamic_quant_struct.h, simplified)
#ifndef TPL_USE_DB_FALSE
#define TPL_USE_DB_FALSE 0
#define TPL_USE_DB_TRUE 1
#define TPL_COMMON_FULL_LOAD 0
#define TPL_COMMON_LARGE_SHAPE 1
#define TPL_MOE_FULL_LOAD 2
#define TPL_MOE_LARGE_SHAPE 3
#define TPL_EMPTY_TENSOR 6
#define TPL_HAS_SMOOTH_FALSE 0
#define TPL_HAS_SMOOTH_TRUE 1
#define TPL_IS_SYMMERTRICAL_FALSE 0
#define TPL_IS_SYMMERTRICAL_TRUE 1
#endif

#endif // DYNAMIC_QUANT_STORY_LOCAL_DEPS_H
