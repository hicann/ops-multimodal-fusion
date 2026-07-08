/*!
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DYNAMIC_QUANT_STORY_TILING_H
#define DYNAMIC_QUANT_STORY_TILING_H

#include <cstdint>

struct DynamicQuantTilingData {
    uint32_t coreNum = 0;
    uint32_t rowLen = 0;
    uint32_t headCoreNum = 0;
    uint32_t rowPerHeadCore = 0;
    uint32_t rowPerTailCore = 0;
    uint32_t multiRowNumHeadCore = 0;
    uint32_t multiRowNumTailCore = 0;
    uint32_t innerLoopEle = 0;
    uint32_t innerLoopTimes = 0;
    uint32_t innerLoopTail = 0;
    uint32_t groupNum = 0;
    uint32_t alignGroupNum = 0;
    uint32_t hasSmooth = 0;
};

#endif // DYNAMIC_QUANT_STORY_TILING_H
