/*!
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_quant_host_tiling.h
 * \brief Standalone host-side tiling calculator for the dynamic_quant kernel.
 *        Adapted from dynamic_quant_tiling_arch35.cpp - no gert/ge dependencies.
 */
#ifndef DYNAMIC_QUANT_STORY_HOST_TILING_H
#define DYNAMIC_QUANT_STORY_HOST_TILING_H

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

#include "dynamic_quant_tiling.h"
#include "platform/platform_ascendc.h"

// Template parameter constants (from dynamic_quant_struct.h)
#ifndef TPL_COMMON_FULL_LOAD
#define TPL_COMMON_FULL_LOAD      0
#define TPL_COMMON_LARGE_SHAPE    1
#define TPL_MOE_FULL_LOAD         2
#define TPL_MOE_LARGE_SHAPE       3
#endif

namespace DynamicQuantHost {

// ---- AlignUp helper, matching the original exactly ----
template <uint32_t base, typename T = uint32_t>
static inline T AlignUp(T a)
{
    return (a + base - 1) / base * base;
}

// ---- Launch configuration returned to the caller ----
struct DynamicQuantLaunchConfig {
    uint32_t blockDim = 0;
    uint32_t quantMode = 0; // TPL_COMMON_FULL_LOAD, etc.
    bool useDb = false;
    DynamicQuantTilingData tiling;
};

// ---- Constants from the original tiling (preserved exactly) ----
static constexpr uint32_t RESERVED_LENGTH       = 1024;
static constexpr uint32_t COMPARE_INT           = 255;
static constexpr uint32_t FLOAT_NUM_ONE_RPT     = 128;
static constexpr uint64_t SMOOTH_BYTES_SIZE     = 2;
static constexpr uint64_t REQUIRED_BYTES_SIZE   = 7;
static constexpr uint64_t OFFSET_BYTES_SIZE     = 4;
static constexpr uint64_t DB_OFFSET_BYTES_SIZE  = 8;
static constexpr uint64_t DB_SMOOTH_BYTES_SIZE  = 4;
static constexpr uint64_t DB_REQUIRED_BYTES_SIZE = 14;

// ---- Main entry point ----
inline DynamicQuantLaunchConfig CalcDynamicQuantTiling(
    uint32_t rowNum, uint32_t rowLen,
    bool hasSmooth, uint32_t groupNum, bool isSymmetrical)
{
    // --- (a) Get platform info ---
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t vectorCoreNum = ascendcPlatform->GetCoreNumAiv();

    if (vectorCoreNum == 0 || ubSize == 0) {
        throw std::runtime_error("DynamicQuantHost: failed to get platform info - "
                                 "vectorCoreNum or ubSize is 0");
    }

    // --- (b) CalculateCoreNum logic ---
    constexpr uint32_t ONE = 1;

    uint32_t coreNum      = std::max(std::min(vectorCoreNum, rowNum), ONE);
    uint32_t headCoreNum  = rowNum % vectorCoreNum;
    uint32_t rowPerHeadCore = (rowNum + vectorCoreNum - 1U) / vectorCoreNum;
    uint32_t rowPerTailCore = rowNum / vectorCoreNum;

    // --- (c) CalculateTilingData logic ---
    uint32_t alignedRowLen = AlignUp<16>(rowLen);
    uint64_t maxUseUbSize  = ubSize - RESERVED_LENGTH;
    uint32_t smoothBuffer  = hasSmooth ? 1U : 0U;
    uint32_t offsetBuffer  = 1U; // always present (asymmetrical path)

    uint64_t calcSize = static_cast<uint64_t>(alignedRowLen) *
                        (smoothBuffer * SMOOTH_BYTES_SIZE + REQUIRED_BYTES_SIZE +
                         offsetBuffer * OFFSET_BYTES_SIZE);

    uint64_t calcDbSize = static_cast<uint64_t>(alignedRowLen) *
                          (smoothBuffer * DB_SMOOTH_BYTES_SIZE + DB_REQUIRED_BYTES_SIZE +
                           offsetBuffer * DB_OFFSET_BYTES_SIZE);

    bool     useDb = false;
    uint32_t quantMode = TPL_COMMON_FULL_LOAD;
    uint32_t multiRowNumHeadCore = 0;
    uint32_t multiRowNumTailCore = 0;
    uint32_t innerLoopEle   = 0;
    uint32_t innerLoopTimes = 0;
    uint32_t innerLoopTail  = 0;

    // Three scenarios (preserved exactly from original):
    if (calcSize > maxUseUbSize) {
        // Scenario 1: UB can't fit one row -> large_shape mode (innerLoop split)
        uint64_t calcLargeBuf =
            2UL * (static_cast<uint64_t>(smoothBuffer) * 2UL + 4UL); // for inQueue,outQueue&smoothQueue DB
        maxUseUbSize -= RESERVED_LENGTH;                               // for scaleQueue DB
        useDb = true;
        innerLoopEle = static_cast<uint32_t>(maxUseUbSize) /
                       static_cast<uint32_t>(calcLargeBuf) /
                       FLOAT_NUM_ONE_RPT * FLOAT_NUM_ONE_RPT;
        innerLoopTimes = rowLen / innerLoopEle;
        innerLoopTail  = rowLen % innerLoopEle;
        multiRowNumHeadCore = std::min({COMPARE_INT, ONE, rowPerHeadCore});
        multiRowNumTailCore = std::min({COMPARE_INT, ONE, rowPerTailCore});

        quantMode = groupNum > 0 ? TPL_MOE_LARGE_SHAPE : TPL_COMMON_LARGE_SHAPE;
    } else if (calcDbSize > maxUseUbSize) {
        // Scenario 2: UB fits one row but not double buffer -> no-db full_load
        useDb = false;
        uint32_t ubAvail = static_cast<uint32_t>(maxUseUbSize) / static_cast<uint32_t>(calcSize);
        multiRowNumHeadCore = std::min({ubAvail, COMPARE_INT, rowPerHeadCore});
        multiRowNumTailCore = std::min({ubAvail, COMPARE_INT, rowPerTailCore});

        quantMode = groupNum > 0 ? TPL_MOE_FULL_LOAD : TPL_COMMON_FULL_LOAD;
    } else {
        // Scenario 3: UB fits double buffer -> db full_load
        useDb = true;
        uint32_t ubAvail = static_cast<uint32_t>(maxUseUbSize) / static_cast<uint32_t>(calcDbSize);
        multiRowNumHeadCore = std::min({ubAvail, COMPARE_INT, rowPerHeadCore});
        multiRowNumTailCore = std::min({ubAvail, COMPARE_INT, rowPerTailCore});

        quantMode = groupNum > 0 ? TPL_MOE_FULL_LOAD : TPL_COMMON_FULL_LOAD;
    }

    // --- (d) Fill result struct ---
    DynamicQuantLaunchConfig cfg;
    cfg.blockDim  = coreNum;
    cfg.quantMode = quantMode;
    cfg.useDb     = useDb;

    DynamicQuantTilingData& t = cfg.tiling;
    t.coreNum             = coreNum;
    t.rowLen              = rowLen;
    t.headCoreNum         = headCoreNum;
    t.rowPerHeadCore      = rowPerHeadCore;
    t.rowPerTailCore      = rowPerTailCore;
    t.multiRowNumHeadCore = multiRowNumHeadCore;
    t.multiRowNumTailCore = multiRowNumTailCore;
    t.innerLoopEle        = innerLoopEle;
    t.innerLoopTimes      = innerLoopTimes;
    t.innerLoopTail       = innerLoopTail;
    t.groupNum            = groupNum;
    t.alignGroupNum       = AlignUp<16>(groupNum);
    t.hasSmooth           = hasSmooth ? 1U : 0U;

    return cfg;
}

} // namespace DynamicQuantHost

#endif // DYNAMIC_QUANT_STORY_HOST_TILING_H
