/*!
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEQUANT_SWIGLU_QUANT_TILING_DATA_H
#define DEQUANT_SWIGLU_QUANT_TILING_DATA_H

#include <cstdint>

struct DequantSwigluQuantV35BaseTilingData {
    int64_t inDimx = 0;
    int64_t inDimy = 0;
    int64_t outDimy = 0;
    int64_t UbFactorDimx = 0;
    int64_t UbFactorDimy = 0;
    int64_t usedCoreNum = 0;
    int64_t maxCoreNum = 0;
    int64_t inGroupNum = 0;
    int64_t quantMode = 0;
    int64_t actRight = 0;
    int64_t dstType = 0;
    int64_t roundMode = 0;
    int64_t activateDim = 0;
    int64_t loopTimesPerRow = 0;
    int64_t tailPerRow = 0;
    int64_t swiGluMode = 0;
    int64_t biasMode = 0;
    int64_t groupIndexMode = 0;
    int64_t quantIsOne = 0;
    int64_t speGroupType = 0;
    int64_t isSpecialCoreCut = 0;
    float clampLimit = 7.0f;
    float gluAlpha = 1.702f;
    float gluBias = 1.0f;
};

struct DequantSwigluQuantV35NlastTilingData {
    int64_t inDim0 = 0;
    int64_t inDim1 = 0;
    int64_t inDim2 = 0;
    int64_t outDim1 = 0;
    int64_t blockNum0 = 0;
    int64_t blockNum1 = 0;
    int64_t blockFormer0 = 0;
    int64_t blockFormer1 = 0;
    int64_t ubFormer0 = 0;
    int64_t ubFormer1 = 0;
    int64_t ubLoopOfFormerBlock0 = 0;
    int64_t ubLoopOfFormerBlock1 = 0;
    int64_t ubLoopOfTailBlock0 = 0;
    int64_t ubLoopOfTailBlock1 = 0;
    int64_t ubTailOfFormerBlock0 = 0;
    int64_t ubTailOfFormerBlock1 = 0;
    int64_t ubTailOfTailBlock0 = 0;
    int64_t ubTailOfTailBlock1 = 0;
    int64_t actRight = 0;
    int64_t roundMode = 0;
};

#endif // DEQUANT_SWIGLU_QUANT_TILING_DATA_H
