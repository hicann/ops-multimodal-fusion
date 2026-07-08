/*!
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SWI_GLU_STORY_GLU_TILING_KERNEL_HPP
#define SWI_GLU_STORY_GLU_TILING_KERNEL_HPP

#include "glu_tiling.hpp"
#include "swi_glu_tiling.h"

constexpr uint32_t DEFAULT_MIN_BLOCK_SIZE = 32;

struct SwiGluSingleTileOffsetParam {
    uint64_t splitVecGmOffset1 = 0;
    uint64_t splitVecGmOffset2 = 0;
    uint64_t indepVecGmoffset = 0;
};

struct SwiGluCopyParam {
    uint16_t blockCount = 0;
    uint16_t blockLen = 0;
    uint32_t stride = 0;
};

struct SwiGluSinlgeTileCopyParam {
    SwiGluCopyParam splitVecCopyParam;
    SwiGluCopyParam indepVecCopyParam;
};

struct SwigluSingleTilingKernel {
    uint32_t is32BAligned = 1;
    uint64_t totalBlockLen = 0;
    uint64_t combColLen = 0;
    uint64_t colLen = 0;
    uint64_t rowLen = 0;

    uint32_t baseRowLen = 0;
    uint32_t baseColLen = 0;
    uint32_t tailRowLen = 0;
    uint32_t tailColLen = 0;

    uint32_t tileLength = 0;

    uint64_t rowTileNum = 0;
    uint64_t colTileNum = 0;
    uint64_t totalTileNum = 0;

    uint64_t baseRowTileNum = 0;
    uint64_t baseColTileNum = 0;

    uint64_t baseRowBaseColCalLen = 0;
    uint64_t baseRowTailColCalLen = 0;
    uint64_t tailRowBaseColCalLen = 0;
    uint64_t tailRowTailColCalLen = 0;
    SwiGluSinlgeTileCopyParam baseRowBaseColCopyParam;
    SwiGluSinlgeTileCopyParam baseRowTailColCopyParam;
    SwiGluSinlgeTileCopyParam tailRowBaseColCopyParam;
    SwiGluSinlgeTileCopyParam tailRowTailColCopyParam;

    uint64_t curCalLen;
    SwiGluSingleTileOffsetParam offsetParam;
    SwiGluSinlgeTileCopyParam *curTileCopyParam = nullptr;

    __aicore__ void GetTilingAndOffset(const SwiGluTilingData &tempTilingGm, uint32_t inputDTypeLen) {
        is32BAligned = tempTilingGm.is32BAligned;
        rowLen = tempTilingGm.rowLen;
        colLen = tempTilingGm.colLen;
        combColLen = colLen * 2;
        totalBlockLen = rowLen * combColLen;

        uint32_t minInputDTypeLen = 2;
        uint32_t isZero = 1;

        baseRowLen = tempTilingGm.baseRowLen;
        baseColLen = tempTilingGm.baseColLen;
        tileLength = (is32BAligned == 1) ? (baseRowLen * baseColLen) :
            baseRowLen * AlignUp<uint32_t>(baseColLen,
            ISMAX<uint32_t>(isZero, static_cast<uint32_t>(DEFAULT_MIN_BLOCK_SIZE / (inputDTypeLen == 0 ? minInputDTypeLen : inputDTypeLen))));

        baseRowTileNum = rowLen / baseRowLen;
        baseColTileNum = colLen / baseColLen;
        tailRowLen = rowLen % baseRowLen;
        tailColLen = colLen % baseColLen;
        rowTileNum = (tailRowLen > 0) ? (baseRowTileNum + 1) : baseRowTileNum;
        colTileNum = (tailColLen > 0) ? (baseColTileNum + 1) : baseColTileNum;
        totalTileNum = rowTileNum * colTileNum;

        CaclTileCopyParams(inputDTypeLen);
    }

    __aicore__ inline void CaclOneTileCopyParam(uint64_t calRowLen, uint64_t calColLen, uint32_t inputDTypeLen, SwiGluSinlgeTileCopyParam &SwiGluCopyParam)
    {
        uint16_t blockUnit = (is32BAligned == 1) ? DEFAULT_MIN_BLOCK_SIZE : 1;
        SwiGluCopyParam.splitVecCopyParam.blockCount = calRowLen;
        SwiGluCopyParam.splitVecCopyParam.blockLen = calColLen * inputDTypeLen / blockUnit;
        SwiGluCopyParam.splitVecCopyParam.stride =
            calRowLen == 1 ? 0 : ((combColLen - calColLen) * inputDTypeLen / blockUnit);

        SwiGluCopyParam.indepVecCopyParam.blockCount = calRowLen;
        SwiGluCopyParam.indepVecCopyParam.blockLen = calColLen * inputDTypeLen / blockUnit;
        SwiGluCopyParam.indepVecCopyParam.stride =
            calRowLen == 1 ? 0 : ((colLen - calColLen) * inputDTypeLen / blockUnit);
    }

    __aicore__ inline void CaclTileCopyParams(uint32_t inputDTypeLen) {
        uint32_t minInputDTypeLen = 2;
        uint32_t isZero = 1;
        baseRowBaseColCalLen = (is32BAligned == 1) ? (baseRowLen * baseColLen) :
          (baseRowLen * AlignUp<uint32_t>(baseColLen,
          ISMAX<uint32_t>(isZero, static_cast<uint32_t>(DEFAULT_MIN_BLOCK_SIZE / (inputDTypeLen == 0 ? minInputDTypeLen : inputDTypeLen)))));
        CaclOneTileCopyParam(baseRowLen, baseColLen, inputDTypeLen, baseRowBaseColCopyParam);

        baseRowTailColCalLen = (is32BAligned == 1) ? (baseRowLen * tailColLen) :
          baseRowLen * AlignUp<uint32_t>(tailColLen,
          ISMAX<uint32_t>(isZero, static_cast<uint32_t>(DEFAULT_MIN_BLOCK_SIZE / (inputDTypeLen == 0 ? minInputDTypeLen : inputDTypeLen))));
        CaclOneTileCopyParam(baseRowLen, tailColLen, inputDTypeLen, baseRowTailColCopyParam);

        tailRowBaseColCalLen = (is32BAligned == 1) ? (tailRowLen * baseColLen) :
          tailRowLen * AlignUp<uint32_t>(baseColLen,
          ISMAX<uint32_t>(isZero, static_cast<uint32_t>(DEFAULT_MIN_BLOCK_SIZE / (inputDTypeLen == 0 ? minInputDTypeLen : inputDTypeLen))));
        CaclOneTileCopyParam(tailRowLen, baseColLen, inputDTypeLen, tailRowBaseColCopyParam);

        tailRowTailColCalLen = (is32BAligned == 1) ? (tailRowLen * tailColLen) :
          tailRowLen * AlignUp<uint32_t>(tailColLen,
          ISMAX<uint32_t>(isZero, static_cast<uint32_t>(DEFAULT_MIN_BLOCK_SIZE / (inputDTypeLen == 0 ? minInputDTypeLen : inputDTypeLen))));
        CaclOneTileCopyParam(tailRowLen, tailColLen, inputDTypeLen, tailRowTailColCopyParam);
    }

    __aicore__ inline void CaclOneTileOffsetParam(uint64_t gmRowOffset, uint64_t colIdx)
    {
        offsetParam.splitVecGmOffset1 = gmRowOffset * combColLen + colIdx * baseColLen;
        offsetParam.splitVecGmOffset2 = offsetParam.splitVecGmOffset1 + colLen;
        offsetParam.indepVecGmoffset = gmRowOffset * colLen + colIdx * baseColLen;
    }

    __aicore__ inline void CaclOneTileParam(uint64_t tileIdx)
    {
        uint64_t rowTileIdx = tileIdx / colTileNum;
        uint64_t colTileIdx = tileIdx % colTileNum;
        CaclOneTileOffsetParam(rowTileIdx * baseRowLen, colTileIdx);
        if (rowTileIdx < baseRowTileNum) {
            if (colTileIdx < baseColTileNum) {
                curCalLen = baseRowBaseColCalLen;
                curTileCopyParam = &baseRowBaseColCopyParam;
            } else {
                curCalLen = baseRowTailColCalLen;
                curTileCopyParam = &baseRowTailColCopyParam;
            }
        } else {
            if (colTileIdx < baseColTileNum) {
                curCalLen = tailRowBaseColCalLen;
                curTileCopyParam = &tailRowBaseColCopyParam;
            } else {
                curCalLen = tailRowTailColCalLen;
                curTileCopyParam = &tailRowTailColCopyParam;
            }
        }
    }
};

#define SWIGLU_SINGLE_PROCESS_TILE(offsetParam, SwiGluCopyParam, calLen) \
do {                                \
    CopyIn(offsetParam, SwiGluCopyParam); \
    this->Compute(calLen); \
    CopyOut(offsetParam, SwiGluCopyParam); \
} while (0)

#define SWIGLU_SINGLE_PROCESS(kernelTiling) \
do {                                       \
    uint64_t blockNum = GetBlockNum();       \
    for(uint64_t tileIdx = AscendC::GetBlockIdx(); tileIdx < (kernelTiling).totalTileNum; tileIdx += blockNum) { \
        (kernelTiling).CaclOneTileParam(tileIdx); \
        SWIGLU_SINGLE_PROCESS_TILE((kernelTiling).offsetParam, *((kernelTiling).curTileCopyParam), (kernelTiling).curCalLen); \
    } \
} while(0)

#define SWIGLU_SINGLE_PROCESS_TILE_NON32BALIGNED(offsetParam, SwiGluCopyParam, calLen) \
do {                                              \
    CopyIn_Non32BAligned(offsetParam, SwiGluCopyParam); \
    this->Compute(calLen); \
    CopyOut_Non32BAligned(offsetParam, SwiGluCopyParam); \
} while(0)

#define SWIGLU_SINGLE_PROCESS_NON32BALIGNED(kernelTiling) \
do {                                       \
    uint64_t blockNum = GetBlockNum();       \
    for(uint64_t tileIdx = AscendC::GetBlockIdx(); tileIdx < (kernelTiling).totalTileNum; tileIdx += blockNum) { \
        (kernelTiling).CaclOneTileParam(tileIdx); \
        SWIGLU_SINGLE_PROCESS_TILE_NON32BALIGNED((kernelTiling).offsetParam, *((kernelTiling).curTileCopyParam), (kernelTiling).curCalLen); \
    } \
} while(0)

#endif // SWI_GLU_STORY_GLU_TILING_KERNEL_HPP
