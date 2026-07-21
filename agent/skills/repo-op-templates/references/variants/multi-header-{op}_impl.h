/*!
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * // TEMPLATE: 本头是「多头文件变体」的伴随示意头，被 {op}.asc 以 #include "{op}_impl.h" 引入。
 * // TEMPLATE: 它把复杂融合算子在真实工程里分散于多个同目录头的职责合并成 4 个 section 展示；
 * //           落地时按 swi_glu 的方式拆细即可：
 * //             - Section A 结构体 -> {op}_tiling.h        （host + kernel 共享的 TilingData）
 * //             - Section B host   -> {op}_host_tiling.h   （host 侧 CalcTiling，返回 blockDim + tiling）
 * //             - Section C kernel -> {op}_impl.hpp        （kernel 计算类，Init/Process）
 * //             - Section D 薄壳   -> {op}_apt.h           （__global__ launcher，按值收 TilingData）
 */

#ifndef {OP}_IMPL_H
#define {OP}_IMPL_H

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include <algorithm>
#include <cstdint>

// ============================================================================
// Section A: TilingData 结构体（host 计算、kernel 按值接收，通过 <<<>>> 传入）
// TEMPLATE: 真实算子放到 {op}_tiling.h；字段按算子 tiling 需求增删。
// ============================================================================
struct {Op}TilingData {
    uint32_t isDoubleBuffer = 0;
    uint64_t rowLen = 0;
    uint64_t colLen = 0;
    uint32_t baseRowLen = 0;
    uint32_t baseColLen = 0;
};

// ============================================================================
// Section B: host 侧 tiling 计算（返回 blockDim + TilingData）
// TEMPLATE: 真实算子放到 {op}_host_tiling.h；用 PlatformAscendCManager 取 UB / 核数，
//           推导 baseRow/baseCol 分块与所用核数，最终填 {Op}LaunchConfig。
// ============================================================================
namespace {Op}Host {

constexpr int32_t DT_FLOAT = 0;
constexpr int32_t DT_FLOAT16 = 1;
constexpr int32_t DT_BF16 = 27;

struct {Op}LaunchConfig {
    uint32_t blockDim = 0;
    {Op}TilingData tiling;
};

inline {Op}LaunchConfig Calc{Op}Tiling(uint64_t rowLen, uint64_t colLen, int32_t dtype)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t coreNum = ascendcPlatform->GetCoreNumAiv();

    uint32_t dsize = (dtype == DT_FLOAT) ? 4u : 2u;

    {Op}LaunchConfig config;
    config.tiling.rowLen = rowLen;
    config.tiling.colLen = colLen;
    config.tiling.isDoubleBuffer = 1;
    // TEMPLATE: 下面是占位分块 —— 真实算子按 UB 容量 / 对齐 / L2 CacheLine 精细求解 baseRow/baseCol。
    config.tiling.baseColLen = static_cast<uint32_t>(std::min<uint64_t>(colLen, ubSize / (dsize * 4)));
    config.tiling.baseRowLen = 1;
    uint64_t totalTiles = (config.tiling.baseColLen == 0)
        ? 1 : (colLen + config.tiling.baseColLen - 1) / config.tiling.baseColLen * rowLen;
    config.blockDim = static_cast<uint32_t>(std::min<uint64_t>(totalTiles, coreNum));
    if (config.blockDim == 0) config.blockDim = 1;
    return config;
}

} // namespace {Op}Host

// ============================================================================
// Section C: kernel 计算类（模板化，按类型实例化不同精度路径）
// TEMPLATE: 真实算子放到 {op}_impl.hpp；Init 负责地址偏移 + UB 分配，Process 负责搬入/计算/搬出。
// ============================================================================
template <typename inType, typename outType, uint16_t bufferNum>
class Kernel{Op} {
public:
    __aicore__ inline Kernel{Op}() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR /*workspace*/, GM_ADDR output,
                                const {Op}TilingData &tiling)
    {
        rowLen_ = tiling.rowLen;
        colLen_ = tiling.colLen;
        baseColLen_ = tiling.baseColLen;
        // TEMPLATE: 多核地址偏移按 baseRow/baseCol 与 GetBlockIdx() 计算；此处仅示意整块绑定。
        xGm_.SetGlobalBuffer((__gm__ inType *)input);
        zGm_.SetGlobalBuffer((__gm__ outType *)output);
        pipe_.InitBuffer(inQueue_, bufferNum, baseColLen_ * sizeof(inType));
        pipe_.InitBuffer(outQueue_, bufferNum, baseColLen_ * sizeof(outType));
    }

    __aicore__ inline void Process()
    {
        // TEMPLATE: 按 tile 遍历 —— CopyIn -> Compute（本算子核心融合逻辑）-> CopyOut。
        //           融合算子的核心计算（如 silu(a) * b 的向量指令链）在这里实现。
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, bufferNum> inQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, bufferNum> outQueue_;
    AscendC::GlobalTensor<inType> xGm_;
    AscendC::GlobalTensor<outType> zGm_;
    uint64_t rowLen_ = 0;
    uint64_t colLen_ = 0;
    uint32_t baseColLen_ = 0;
};

// ============================================================================
// Section D: __global__ 薄壳 launcher（按值收 TilingData，被 .asc 用 <<<>>> 直调）
// TEMPLATE: 真实算子放到 {op}_apt.h；按 isDoubleBuffer 等标志实例化不同 bufferNum / 类型组合。
// ============================================================================
__global__ __aicore__ void {op}_fp32(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                     const {Op}TilingData tilingData)
{
#if !defined(__NPU_HOST__)
    (void)workspace;
    if (tilingData.isDoubleBuffer == 1) {
        Kernel{Op}<float, float, 2> op;
        op.Init(input, nullptr, output, tilingData);
        op.Process();
    } else {
        Kernel{Op}<float, float, 1> op;
        op.Init(input, nullptr, output, tilingData);
        op.Process();
    }
#endif
}

__global__ __aicore__ void {op}_fp16(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                     const {Op}TilingData tilingData)
{
#if !defined(__NPU_HOST__)
    (void)workspace;
    Kernel{Op}<half, half, 2> op;
    op.Init(input, nullptr, output, tilingData);
    op.Process();
#endif
}

__global__ __aicore__ void {op}_bf16(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                     const {Op}TilingData tilingData)
{
#if !defined(__NPU_HOST__)
    (void)workspace;
    Kernel{Op}<bfloat16_t, bfloat16_t, 2> op;
    op.Init(input, nullptr, output, tilingData);
    op.Process();
#endif
}

#endif // {OP}_IMPL_H
