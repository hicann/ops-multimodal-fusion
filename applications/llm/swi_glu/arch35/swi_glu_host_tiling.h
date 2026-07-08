/*!
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SWI_GLU_STORY_HOST_TILING_H
#define SWI_GLU_STORY_HOST_TILING_H

#include "swi_glu_tiling.h"
#include "platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace SwiGluHost {

constexpr int32_t DT_FLOAT = 0;
constexpr int32_t DT_FLOAT16 = 1;
constexpr int32_t DT_BF16 = 27;

constexpr uint32_t UB_MIN_BLOCK_SIZE = 32;
constexpr uint32_t L2_CACHE_LINE_SIZE = 512;
constexpr uint16_t DISCONTINE_COPY_MAX_BLOCKCNT = 4095;
constexpr uint16_t DISCONTINE_COPY_MAX_BLOCKLEN = 65535;
constexpr uint16_t DISCONTINE_COPY_MAX_STRIDE = 65535;

constexpr uint32_t XXGLU_TQUE_NUM = 3;
constexpr uint32_t SWIGLU_TBUF_NUM_HALF = 2;
constexpr uint32_t SWIGLU_TBUF_NUM_BF16 = 2;
constexpr uint32_t SWIGLU_TBUF_NUM_FLOAT = 1;

struct SwiGluLaunchConfig {
    uint32_t blockDim = 0;
    SwiGluTilingData tiling;
};

namespace detail {

template<typename T>
inline T AlignUp(T num, T rnd) { return (rnd == 0) ? 0 : ((num + rnd - 1) / rnd * rnd); }

template<typename T>
inline T AlignDown(T num, T rnd) { return ((rnd == 0) || (num < rnd)) ? 0 : (num / rnd * rnd); }

template<typename T>
inline T DivCeil(T num, T div) { return (div == 0) ? 0 : ((num + div - 1) / div); }

inline bool GetLengthByType(int32_t dtype, uint32_t &dsize)
{
    switch (dtype) {
        case DT_FLOAT16:
        case DT_BF16:
            dsize = 2;
            return true;
        case DT_FLOAT:
            dsize = 4;
            return true;
        default:
            return false;
    }
}

struct OptParam {
    uint32_t maxTileLen = 0;
    uint32_t optBaseRowLen = 0;
    uint32_t optBaseColLen = 0;
    uint64_t optTotalTileNum = 0;
    uint64_t optBaseSize = 0;
    uint64_t optBaseTileNum = 0;
    uint32_t totalUsedCoreNum = 0;
    uint64_t tileNumPerCore = 0;
};

class Calculator {
public:
    SwiGluTilingData tiling;

    SwiGluLaunchConfig Calc(uint64_t rowLen, uint64_t colLen, int32_t dtype)
    {
        auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
        uint64_t ubSize = 0;
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        totalAvailableCore = ascendcPlatform->GetCoreNumAiv();

        tiling.rowLen = rowLen;
        tiling.colLen = colLen;

        if (!GetLengthByType(dtype, inputDTypeLen)) {
            throw std::runtime_error("Unsupported dtype for SwiGlu tiling");
        }
        ubMinBlockLen = UB_MIN_BLOCK_SIZE / inputDTypeLen;
        cacheLineLen = L2_CACHE_LINE_SIZE / inputDTypeLen;
        alignPackLen = cacheLineLen;

        tiling.is32BAligned = (colLen % ubMinBlockLen == 0) ? 1 : 0;

        tiling.isDoubleBuffer = 1;
        OptParam optDb;
        if (!CalcOptTiling<2>(ubSize, dtype, optDb)) {
            throw std::runtime_error("CalcOptTiling with double buffer failed");
        }

        OptParam *opt = &optDb;
        if (optDb.tileNumPerCore == 2) {
            OptParam optNoDb;
            if (CalcOptTiling<1>(ubSize, dtype, optNoDb) &&
                optNoDb.tileNumPerCore == 1 &&
                optNoDb.totalUsedCoreNum >= optDb.totalUsedCoreNum) {
                opt = &optNoDb;
                tiling.isDoubleBuffer = 0;
            }
        }

        tiling.baseRowLen = opt->optBaseRowLen;
        tiling.baseColLen = opt->optBaseColLen;

        SwiGluLaunchConfig config;
        config.blockDim = opt->totalUsedCoreNum;
        config.tiling = tiling;
        return config;
    }

private:
    uint32_t inputDTypeLen = 2;
    uint32_t ubMinBlockLen = 0;
    uint32_t cacheLineLen = 0;
    uint32_t alignPackLen = 0;
    uint32_t totalAvailableCore = 0;

    template<uint16_t bufferNum>
    bool CalcOptTiling(uint64_t ubSize, int32_t dtype, OptParam &opt)
    {
        if (!CalcUbMaxTileLen<bufferNum>(ubSize, dtype, opt)) return false;
        if (!CalcOptBaseShape(opt)) return false;
        return true;
    }

    template<uint16_t bufferNum>
    bool CalcUbMaxTileLen(uint64_t ubSize, int32_t dtype, OptParam &opt)
    {
        uint32_t singleDataSize = 0;
        if (dtype == DT_FLOAT16) {
            singleDataSize = static_cast<uint32_t>(
                static_cast<uint64_t>(bufferNum) * XXGLU_TQUE_NUM * sizeof(int16_t) +
                SWIGLU_TBUF_NUM_HALF * sizeof(int32_t));
        } else if (dtype == DT_BF16) {
            singleDataSize = static_cast<uint32_t>(
                static_cast<uint64_t>(bufferNum) * XXGLU_TQUE_NUM * sizeof(int16_t) +
                SWIGLU_TBUF_NUM_BF16 * sizeof(int32_t));
        } else {
            singleDataSize = static_cast<uint32_t>(
                static_cast<uint64_t>(bufferNum) * XXGLU_TQUE_NUM * sizeof(int32_t) +
                SWIGLU_TBUF_NUM_FLOAT * sizeof(int32_t));
        }
        if (singleDataSize == 0) return false;
        uint64_t maxTileLenPerUB = ubSize / singleDataSize;
        opt.maxTileLen = static_cast<uint32_t>(AlignDown<uint64_t>(maxTileLenPerUB, ubMinBlockLen));
        return true;
    }

    void SaveOptBaseShape(uint32_t baseRowLen_, uint32_t baseColLen_, OptParam &opt)
    {
        if (baseRowLen_ == 0 || baseColLen_ == 0) return;
        uint64_t totalTileNum = DivCeil<uint64_t>(tiling.rowLen, baseRowLen_) *
                                DivCeil<uint64_t>(tiling.colLen, baseColLen_);
        uint64_t baseSize = static_cast<uint64_t>(baseRowLen_) * baseColLen_;
        uint64_t baseTileNum = (tiling.rowLen / baseRowLen_) * (tiling.colLen / baseColLen_);
        uint32_t usedCores = static_cast<uint32_t>(
            std::min(totalTileNum, static_cast<uint64_t>(totalAvailableCore)));

        if ((opt.optTotalTileNum == 0) ||
            (usedCores > opt.totalUsedCoreNum) ||
            ((usedCores == opt.totalUsedCoreNum) && (totalTileNum < opt.optTotalTileNum)) ||
            ((usedCores == opt.totalUsedCoreNum) && (totalTileNum == opt.optTotalTileNum) &&
             (baseSize > opt.optBaseSize)) ||
            ((usedCores == opt.totalUsedCoreNum) && (totalTileNum == opt.optTotalTileNum) &&
             (baseSize == opt.optBaseSize) && (baseTileNum > opt.optBaseTileNum))) {
            opt.optBaseRowLen = baseRowLen_;
            opt.optBaseColLen = baseColLen_;
            opt.optTotalTileNum = totalTileNum;
            opt.optBaseSize = baseSize;
            opt.optBaseTileNum = baseTileNum;
            opt.totalUsedCoreNum = usedCores;
            opt.tileNumPerCore = DivCeil<uint64_t>(totalTileNum, usedCores);
        }
    }

    uint32_t getBaseColLenUpBound(OptParam &opt)
    {
        uint32_t upBound = static_cast<uint32_t>(
            std::min(tiling.colLen, static_cast<uint64_t>(opt.maxTileLen)));
        if (tiling.is32BAligned == 1) {
            upBound = std::min(upBound, static_cast<uint32_t>(DISCONTINE_COPY_MAX_BLOCKLEN));
        } else {
            upBound = std::min(upBound, static_cast<uint32_t>(DISCONTINE_COPY_MAX_BLOCKLEN / inputDTypeLen));
        }
        if (upBound < tiling.colLen && upBound > cacheLineLen) {
            return AlignDown<uint32_t>(upBound, cacheLineLen);
        }
        return upBound;
    }

    uint32_t getBaseRowLenUpBound()
    {
        return static_cast<uint32_t>(
            std::min(tiling.rowLen, static_cast<uint64_t>(DISCONTINE_COPY_MAX_BLOCKCNT)));
    }

    bool MustBeSingleBaseRowLen(uint32_t baseColLen_)
    {
        if (tiling.is32BAligned == 1) {
            return ((tiling.colLen * 2 - baseColLen_) >
                    (static_cast<uint64_t>(DISCONTINE_COPY_MAX_STRIDE) * ubMinBlockLen));
        } else {
            return (((tiling.colLen * 2 - baseColLen_) * inputDTypeLen) > DISCONTINE_COPY_MAX_STRIDE);
        }
    }

    bool isInvalidBaseShape(uint32_t baseRowLen_, uint32_t baseColLen_)
    {
        return (baseRowLen_ < 1 || (baseRowLen_ > 1 && MustBeSingleBaseRowLen(baseColLen_)));
    }

    bool isValidTailCol(uint32_t baseRowLen_, uint32_t baseColLen_)
    {
        if (baseColLen_ == 0) return false;
        uint32_t tailColLen_ = tiling.colLen % baseColLen_;
        return !(baseRowLen_ > 1 && MustBeSingleBaseRowLen(tailColLen_));
    }

    bool CalcOptBaseShape(OptParam &opt)
    {
        uint32_t baseColLen_ = getBaseColLenUpBound(opt);
        if (MustBeSingleBaseRowLen(baseColLen_)) {
            SaveOptBaseShape(1, baseColLen_, opt);
            return true;
        }

        while (baseColLen_ > 0) {
            uint32_t baseRowLen_ = std::min(
                opt.maxTileLen / AlignUp<uint32_t>(baseColLen_, ubMinBlockLen),
                getBaseRowLenUpBound());
            if (isInvalidBaseShape(baseRowLen_, baseColLen_)) {
                return (opt.optTotalTileNum > 0);
            }
            if (isValidTailCol(baseRowLen_, baseColLen_)) {
                SaveOptBaseShape(baseRowLen_, baseColLen_, opt);
            }
            if (baseColLen_ <= alignPackLen || baseRowLen_ >= getBaseRowLenUpBound()) {
                return true;
            }
            if (baseColLen_ % alignPackLen == 0) {
                baseColLen_ -= alignPackLen;
            } else {
                baseColLen_ = AlignDown<uint32_t>(baseColLen_, alignPackLen);
            }
        }
    }
};

} // namespace detail

inline SwiGluLaunchConfig CalcSwiGluTiling(uint64_t rowLen, uint64_t colLen, int32_t dtype)
{
    detail::Calculator calc;
    return calc.Calc(rowLen, colLen, dtype);
}

} // namespace SwiGluHost

#endif // SWI_GLU_STORY_HOST_TILING_H
