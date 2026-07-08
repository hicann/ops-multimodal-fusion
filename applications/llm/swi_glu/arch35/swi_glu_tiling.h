/*!
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SWI_GLU_STORY_TILING_H
#define SWI_GLU_STORY_TILING_H

#include <cstdint>

struct SwiGluTilingData {
    uint32_t is32BAligned = 0;
    uint32_t isDoubleBuffer = 0;
    uint64_t rowLen = 0;
    uint64_t colLen = 0;
    uint32_t baseRowLen = 0;
    uint32_t baseColLen = 0;
};

#endif // SWI_GLU_STORY_TILING_H
