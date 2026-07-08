/*!
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEQUANT_SWIGLU_QUANT_HOST_TILING_H
#define DEQUANT_SWIGLU_QUANT_HOST_TILING_H

#include "dequant_swiglu_quant_tiling_data.h"
#include "platform/platform_ascendc.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace DsqHostTiling {

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t BLOCK_ELEM_B32 = BLOCK_SIZE / static_cast<int64_t>(sizeof(float));
constexpr int64_t BLOCK_ELEM_B16 = BLOCK_SIZE / static_cast<int64_t>(sizeof(int16_t));
constexpr int64_t BLOCK_ELEM_B8 = BLOCK_SIZE / static_cast<int64_t>(sizeof(int8_t));
constexpr int64_t UB_RESERVE = 1024;
constexpr int64_t SWI_FACTOR = 2;
constexpr int64_t Y_LAST_DIM_FULL_LOAD_MAX = 5120;
constexpr size_t SYS_WORKSPACE_SIZE = 16ULL * 1024 * 1024;

// Tiling key factors for full-load
constexpr int64_t ACTIVATE_DIM_FACTOR = 100000;
constexpr int64_t INPUT_X_FACTOR = 10000;
constexpr int64_t BIAS_FACTOR = 1000;
constexpr int64_t ACT_SCALE_FACTOR = 100;
constexpr int64_t QUANT_SCALE_FACTOR = 10;
constexpr int64_t GROUP_INDEX_FACTOR = 1;

// Tiling key factors for not-full-load
constexpr int64_t PLACEHOLDER = 1000000;
constexpr int64_t QUANT_MODE_FACTOR = 100000;
constexpr int64_t BIAS_FACTOR_NF = 10000;
constexpr int64_t ACT_FACTOR_NF = 1000;
constexpr int64_t QS_FACTOR_NF = 100;
constexpr int64_t QO_FACTOR_NF = 10;

constexpr int64_t SPECIAL_GROUP_64 = 64;
constexpr int64_t SPECIAL_GROUP_32 = 32;
constexpr int64_t SPECIAL_GROUP_16 = 16;

enum class XDtype { INT32, FP16, BF16 };
enum class YDtype { INT8, FP8_E4M3, FP8_E5M2, FP4_E2M1, FP4_E1M2, HIFLOAT8 };
enum class BiasDtype { NONE, INT32, FP32, FP16, BF16 };

struct TilingInput {
    int64_t inDimx;
    int64_t inDimy;   // x last dim (= 2 * outDimy)
    int64_t outDimy;  // y last dim

    uint64_t coreNum;
    uint64_t ubSize;

    XDtype xDtype;
    YDtype yDtype;
    BiasDtype biasDtype;
    bool hasWeightScale;
    bool hasActScale;
    bool hasQuantScale;
    bool hasQuantOffset;
    bool hasGroupIndex;
    int64_t groupNum;

    int64_t quantMode;  // 0=static, 1=dynamic
    int64_t activateDim; // dimension index for split (last dim = xDimNum-1)
    int64_t xDimNum;
    int64_t dstType;
    int64_t roundMode;
    int64_t swigluMode; // 0=left-right, 1=interleaved
    float clampLimit;
    float gluAlpha;
    float gluBias;
    int64_t actRight;
    int64_t speGroupType;
    int64_t biasMode;       // 0=none,1=int32,2=int64
    int64_t groupIndexMode; // 0=none,1=int32,2=int64
    int64_t quantIsOne;
};

struct LaunchConfig {
    int64_t tilingKey;
    uint32_t blockDim;
    size_t workspaceSize;
    DequantSwigluQuantV35BaseTilingData tiling;
};

struct NlastLaunchConfig {
    int64_t tilingKey;
    uint32_t blockDim;
    size_t workspaceSize;
    DequantSwigluQuantV35NlastTilingData tiling;
};

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    int64_t divisor = (b == 0) ? 1 : b;
    return (a + divisor - 1) / divisor;
}

inline int64_t BiasDtypeToValue(BiasDtype bd)
{
    switch (bd) {
        case BiasDtype::NONE:  return 0;
        case BiasDtype::INT32: return 1;
        case BiasDtype::FP32:  return 2;
        case BiasDtype::FP16:  return 3;
        case BiasDtype::BF16:  return 4;
    }
    return 0;
}

inline size_t XBytesPerElem(XDtype d) { return d == XDtype::INT32 ? 4 : 2; }

inline bool IsYFp4(YDtype d) { return d == YDtype::FP4_E2M1 || d == YDtype::FP4_E1M2; }

// Compute bias contribution to UB denominator
inline int64_t BiasUbCost(BiasDtype bd, int64_t xUbAlign32B, int64_t swigluMode, int64_t inDimy)
{
    if (bd == BiasDtype::NONE) return 0;
    int64_t biasMemory = xUbAlign32B * static_cast<int64_t>(sizeof(int16_t));
    int64_t biasTailSupply = CeilDiv(inDimy, 128) * 128 - inDimy;
    int64_t cost = 0;
    switch (bd) {
        case BiasDtype::INT32:
            cost = biasMemory * 2;
            if (swigluMode == 1) cost += biasTailSupply * 4;
            break;
        case BiasDtype::FP32:
            cost = biasMemory * SWI_FACTOR;
            if (swigluMode == 1) cost += biasTailSupply * 4;
            break;
        case BiasDtype::FP16:
        case BiasDtype::BF16:
            cost = biasMemory;
            if (swigluMode == 1) cost += biasTailSupply * 2;
            break;
        default: break;
    }
    return cost;
}

inline LaunchConfig CalcFullLoadTiling(const TilingInput &in)
{
    LaunchConfig cfg{};
    auto &td = cfg.tiling;
    int64_t outDimy = in.outDimy;
    int64_t inDimy = in.inDimy;
    int64_t inDimx = in.inDimx;
    size_t xBits = XBytesPerElem(in.xDtype);

    int64_t xUbAlign32B = CeilDiv(outDimy, BLOCK_ELEM_B32) * BLOCK_ELEM_B32;
    int64_t xUbAlign = (in.xDtype == XDtype::INT32) ? xUbAlign32B :
                        CeilDiv(outDimy, BLOCK_ELEM_B16) * BLOCK_ELEM_B16;
    int64_t aScaleAlign = BLOCK_ELEM_B32;
    int64_t yAlign8B = CeilDiv(outDimy, BLOCK_ELEM_B8) * BLOCK_ELEM_B8;

    int64_t ubAvail = static_cast<int64_t>(in.ubSize) - UB_RESERVE -
                      (xUbAlign32B * SWI_FACTOR + xUbAlign32B) * static_cast<int64_t>(sizeof(float));

    int64_t denom = 2 * (xUbAlign * SWI_FACTOR * static_cast<int64_t>(xBits) + aScaleAlign * 4) +
                    2 * yAlign8B * 1 + aScaleAlign * 4 + xUbAlign32B * 4;

    if (in.swigluMode == 1) {
        int64_t tailSupply = CeilDiv(inDimy, 128) * 128 - inDimy;
        denom += tailSupply * static_cast<int64_t>(xBits) + tailSupply * 4;
    }

    denom += BiasUbCost(in.biasDtype, xUbAlign32B, in.swigluMode, inDimy);

    if (in.hasQuantOffset) {
        denom += xUbAlign32B * 4;
    }

    int64_t ubFactorDimx = ubAvail / denom;
    ubFactorDimx = std::min(ubFactorDimx, inDimx);
    if (ubFactorDimx < 1) {
        throw std::runtime_error("x last dim too large for full load UB");
    }

    int64_t maxCore = std::min(static_cast<int64_t>(in.coreNum), CeilDiv(inDimx, ubFactorDimx));

    int64_t isSpecialCoreCut = 0;
    if ((in.speGroupType == 1) && (in.groupNum >= SPECIAL_GROUP_64) &&
        (inDimx / in.groupNum <= SPECIAL_GROUP_16) &&
        (in.activateDim == in.xDimNum - 1)) {
        isSpecialCoreCut = 1;
    }
    if ((in.speGroupType == 0) && (in.groupNum >= SPECIAL_GROUP_32) &&
        (inDimx / in.groupNum <= SPECIAL_GROUP_16) &&
        (in.activateDim == in.xDimNum - 1) && (in.biasDtype == BiasDtype::NONE) && (!in.hasQuantScale)) {
        isSpecialCoreCut = 1;
    }
    if (isSpecialCoreCut == 1) {
        maxCore = std::min(static_cast<int64_t>(in.coreNum), inDimx);
    }

    // Compute tiling key
    int64_t biasDtypeVal = BiasDtypeToValue(in.biasDtype);
    int64_t hasXInt = (in.quantMode == 1) ? 0 : 1;
    int64_t hasAScale = in.hasActScale ? 1 : 0;
    int64_t hasQScale = in.hasQuantScale ? 1 : 0;
    int64_t hasGIndex = in.hasGroupIndex ? 1 : 0;
    int64_t hasActivateDim = (in.activateDim != in.xDimNum - 1) ? 1 : 0;

    cfg.tilingKey = hasActivateDim * ACTIVATE_DIM_FACTOR + hasXInt * INPUT_X_FACTOR +
                    biasDtypeVal * BIAS_FACTOR + hasAScale * ACT_SCALE_FACTOR +
                    hasQScale * QUANT_SCALE_FACTOR + hasGIndex * GROUP_INDEX_FACTOR;

    td.inDimx = inDimx;
    td.inDimy = inDimy;
    td.outDimy = outDimy;
    td.UbFactorDimx = ubFactorDimx;
    td.UbFactorDimy = outDimy;
    td.usedCoreNum = maxCore;
    td.maxCoreNum = maxCore;
    td.inGroupNum = in.groupNum;
    td.quantMode = in.quantMode;
    td.actRight = in.actRight;
    td.dstType = in.dstType;
    td.roundMode = in.roundMode;
    td.activateDim = in.activateDim;
    td.swiGluMode = in.swigluMode;
    td.biasMode = in.biasMode;
    td.groupIndexMode = in.groupIndexMode;
    td.quantIsOne = in.quantIsOne;
    td.speGroupType = in.speGroupType;
    td.isSpecialCoreCut = isSpecialCoreCut;
    td.clampLimit = in.clampLimit;
    td.gluAlpha = in.gluAlpha;
    td.gluBias = in.gluBias;

    cfg.blockDim = static_cast<uint32_t>(maxCore);
    cfg.workspaceSize = SYS_WORKSPACE_SIZE;
    return cfg;
}

inline LaunchConfig CalcNotFullLoadTiling(const TilingInput &in)
{
    LaunchConfig cfg{};
    auto &td = cfg.tiling;
    int64_t outDimy = in.outDimy;
    int64_t inDimy = in.inDimy;
    int64_t inDimx = in.inDimx;
    bool isFp4 = IsYFp4(in.yDtype);

    int64_t ubAvail = static_cast<int64_t>(in.ubSize) - UB_RESERVE - BLOCK_SIZE;
    int64_t ubFactorDimy = 1;

    if (isFp4) {
        int64_t n = 1;
        n += (in.xDtype == XDtype::INT32) ? 16 : 8;
        n += in.hasWeightScale ? 16 : 0;
        n += 8; // quant_scale
        n += in.hasQuantOffset ? 8 : 0;
        if (in.biasDtype != BiasDtype::NONE) {
            n += (in.biasDtype == BiasDtype::INT32 || in.biasDtype == BiasDtype::FP32) ? 16 : 8;
        }
        n += 1;
        n *= 2; // double buffer
        n += 16; // tmp
        int64_t ySize = (ubAvail / n) / BLOCK_SIZE * BLOCK_SIZE;
        ubFactorDimy = ySize * 2;
    } else {
        int64_t n = 1;
        n += (in.xDtype == XDtype::INT32) ? 8 : 4;
        n += in.hasWeightScale ? 8 : 0;
        n += 4; // quant_scale
        n += in.hasQuantOffset ? 4 : 0;
        if (in.biasDtype != BiasDtype::NONE) {
            n += (in.biasDtype == BiasDtype::INT32 || in.biasDtype == BiasDtype::FP32) ? 8 : 4;
        }
        n *= 2; // double buffer
        n += 8; // tmp
        int64_t ySize = (ubAvail / n) / BLOCK_SIZE * BLOCK_SIZE;
        ubFactorDimy = ySize;
    }

    if (ubFactorDimy > outDimy) {
        int64_t numPerBlock = isFp4 ? BLOCK_SIZE * 2 : BLOCK_SIZE;
        ubFactorDimy = CeilDiv(outDimy, numPerBlock) * numPerBlock;
    }

    int64_t loopTimesPerRow = CeilDiv(outDimy, ubFactorDimy);
    int64_t tailPerRow = outDimy - (loopTimesPerRow - 1) * ubFactorDimy;
    int64_t maxCore = std::min(static_cast<int64_t>(in.coreNum), inDimx);

    // Compute not-full tiling key
    int64_t biasDtypeVal = BiasDtypeToValue(in.biasDtype);
    if (in.quantMode == 0) {
        // Static
        cfg.tilingKey = PLACEHOLDER + in.quantMode * QUANT_MODE_FACTOR +
                        (in.biasDtype != BiasDtype::NONE ? 1 : 0) * BIAS_FACTOR_NF +
                        (in.hasActScale ? 1 : 0) * ACT_FACTOR_NF +
                        (in.hasQuantScale ? 1 : 0) * QS_FACTOR_NF +
                        (in.hasQuantOffset ? 1 : 0) * QO_FACTOR_NF +
                        (in.hasGroupIndex ? 1 : 0);
    } else {
        // Dynamic
        cfg.tilingKey = PLACEHOLDER + in.quantMode * QUANT_MODE_FACTOR +
                        biasDtypeVal * BIAS_FACTOR_NF +
                        (in.hasActScale ? 1 : 0) * ACT_FACTOR_NF +
                        (in.hasQuantScale ? 1 : 0) * QS_FACTOR_NF +
                        (in.hasQuantOffset ? 1 : 0) * QO_FACTOR_NF +
                        (in.hasGroupIndex ? 1 : 0);
    }

    td.inDimx = inDimx;
    td.inDimy = inDimy;
    td.outDimy = outDimy;
    td.UbFactorDimx = 1;
    td.UbFactorDimy = ubFactorDimy;
    td.usedCoreNum = maxCore;
    td.maxCoreNum = static_cast<int64_t>(in.coreNum);
    td.inGroupNum = in.groupNum;
    td.quantMode = in.quantMode;
    td.actRight = in.actRight;
    td.dstType = in.dstType;
    td.roundMode = in.roundMode;
    td.activateDim = in.activateDim;
    td.loopTimesPerRow = loopTimesPerRow;
    td.tailPerRow = tailPerRow;
    td.swiGluMode = in.swigluMode;
    td.biasMode = in.biasMode;
    td.groupIndexMode = in.groupIndexMode;
    td.quantIsOne = in.quantIsOne;
    td.speGroupType = in.speGroupType;
    td.isSpecialCoreCut = 0;
    td.clampLimit = in.clampLimit;
    td.gluAlpha = in.gluAlpha;
    td.gluBias = in.gluBias;

    cfg.blockDim = static_cast<uint32_t>(maxCore);
    size_t usrSize = (in.quantMode == 1) ? static_cast<size_t>(maxCore) * outDimy * sizeof(float) : 0;
    cfg.workspaceSize = SYS_WORKSPACE_SIZE + usrSize;
    return cfg;
}

inline LaunchConfig CalcTiling(const TilingInput &in)
{
    if (in.outDimy > Y_LAST_DIM_FULL_LOAD_MAX) {
        return CalcNotFullLoadTiling(in);
    }
    return CalcFullLoadTiling(in);
}

inline TilingInput BuildDefaultInput()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t totalCore = ascendcPlatform->GetCoreNumAiv();

    TilingInput in{};
    in.coreNum = totalCore;
    in.ubSize = ubSize;
    in.clampLimit = 7.0f;
    in.gluAlpha = 1.702f;
    in.gluBias = 1.0f;
    return in;
}

} // namespace DsqHostTiling

#endif // DEQUANT_SWIGLU_QUANT_HOST_TILING_H
