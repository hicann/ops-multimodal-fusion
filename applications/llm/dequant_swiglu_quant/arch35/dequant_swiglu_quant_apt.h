/*!
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEQUANT_SWIGLU_QUANT_APT_H
#define DEQUANT_SWIGLU_QUANT_APT_H

#include "dequant_swiglu_quant_local_deps.h"
#include "dequant_swiglu_quant.h"
#include "dequant_swiglu_quant_static.h"
#include "dequant_swiglu_quant_static_not_full.h"
#include "dequant_swiglu_quant_dynamic_not_full.h"
#include "dequant_swiglu_quant_nlast.h"

// Enable all dtype combos with DSQ_ENABLE_ALL_DTYPES
#ifdef DSQ_ENABLE_ALL_DTYPES
#define DSQ_ENABLE_INT32_INT8
#define DSQ_ENABLE_BF16_INT8
#define DSQ_ENABLE_FP16_INT8
#define DSQ_ENABLE_INT32_FP8E4M3
#define DSQ_ENABLE_INT32_FP8E5M2
#define DSQ_ENABLE_BF16_FP8E4M3
#define DSQ_ENABLE_BF16_FP8E5M2
#endif

#ifdef DSQ_ENABLE_INT32_INT8
// ===== int32_t -> int8_t =====
__global__ __aicore__ void dsq_base_nb_act1_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act1_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act1_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act1_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act1_qs1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act1_qs0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act0_qs1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act0_qs0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, int32_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act1_qs1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act1_qs0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act0_qs1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act0_qs0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act1_qs1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act1_qs0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act0_qs1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act0_qs0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, half, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act1_qs1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act1_qs0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act0_qs1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act0_qs0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, bfloat16_t, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act0_qs0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act0_qs1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act1_qs0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act1_qs1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, bool, float, bool, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, bool, float, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, bool, float, float, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, bool, float, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, bool, float, bool, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, bool, float, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, bool, float, float, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, bool, float, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, float, float, bool, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, float, float, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, float, float, float, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, float, float, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off0_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, float, float, bool, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off0_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, float, float, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off1_gi0_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, float, float, float, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off1_gi1_int32_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, float, float, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

#endif // DSQ_ENABLE_INT32_INT8

#ifdef DSQ_ENABLE_BF16_INT8
// ===== bfloat16_t -> int8_t =====
__global__ __aicore__ void dsq_base_nb_act1_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act1_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act1_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act1_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act1_qs1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act1_qs0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act0_qs1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act0_qs0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, int32_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act1_qs1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act1_qs0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act0_qs1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act0_qs0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, float, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act1_qs1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act1_qs0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act0_qs1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act0_qs0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, half, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act1_qs1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act1_qs0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act0_qs1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act0_qs0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, bfloat16_t, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act0_qs0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act0_qs1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act1_qs0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act1_qs1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, bool, bfloat16_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, bool, float, bool, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, bool, float, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, bool, float, float, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, bool, float, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, bool, float, bool, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, bool, float, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, bool, float, float, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, bool, float, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, float, float, bool, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, float, float, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, float, float, float, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, float, float, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off0_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, float, float, bool, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off0_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, float, float, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off1_gi0_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, float, float, float, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off1_gi1_bf16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, float, float, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

#endif // DSQ_ENABLE_BF16_INT8

#ifdef DSQ_ENABLE_FP16_INT8
// ===== half -> int8_t =====
__global__ __aicore__ void dsq_base_nb_act1_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act1_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act1_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act1_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act1_qs1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act1_qs0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act0_qs1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act0_qs0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, int32_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act1_qs1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act1_qs0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act0_qs1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act0_qs0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, float, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act1_qs1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act1_qs0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act0_qs1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act0_qs0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, half, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act1_qs1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act1_qs0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act0_qs1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act0_qs0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, bfloat16_t, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act0_qs0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act0_qs1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act1_qs0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act1_qs1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, bool, half, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<half, bool, bool, float, bool, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<half, bool, bool, float, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<half, bool, bool, float, float, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<half, bool, bool, float, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<half, float, bool, float, bool, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<half, float, bool, float, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<half, float, bool, float, float, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<half, float, bool, float, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<half, bool, float, float, bool, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<half, bool, float, float, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<half, bool, float, float, float, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<half, bool, float, float, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off0_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<half, float, float, float, bool, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off0_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<half, float, float, float, bool, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off1_gi0_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<half, float, float, float, float, bool, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off1_gi1_fp16_int8(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<half, float, float, float, float, int32_t, int8_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

#endif // DSQ_ENABLE_FP16_INT8

#ifdef DSQ_ENABLE_INT32_FP8E4M3
// ===== int32_t -> fp8_e4m3fn_t =====
__global__ __aicore__ void dsq_base_nb_act1_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act1_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act1_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act1_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act1_qs1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act1_qs0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act0_qs1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act0_qs0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, int32_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act1_qs1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act1_qs0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act0_qs1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act0_qs0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act1_qs1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act1_qs0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act0_qs1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act0_qs0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, half, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act1_qs1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act1_qs0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act0_qs1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act0_qs0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act0_qs0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act0_qs1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act1_qs0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act1_qs1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, bool, float, bool, bool, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, bool, float, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, bool, float, float, bool, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, bool, float, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, bool, float, bool, bool, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, bool, float, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, bool, float, float, bool, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, bool, float, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, float, float, bool, bool, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, float, float, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, float, float, float, bool, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, float, float, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off0_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, float, float, bool, bool, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off0_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, float, float, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off1_gi0_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, float, float, float, bool, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off1_gi1_int32_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, float, float, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

#endif // DSQ_ENABLE_INT32_FP8E4M3

#ifdef DSQ_ENABLE_INT32_FP8E5M2
// ===== int32_t -> fp8_e5m2_t =====
__global__ __aicore__ void dsq_base_nb_act1_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act1_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act1_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act1_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act1_qs1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act1_qs0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act0_qs1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act0_qs0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, int32_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act1_qs1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act1_qs0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act0_qs1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act0_qs0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act1_qs1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act1_qs0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act0_qs1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act0_qs0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, half, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act1_qs1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act1_qs0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act0_qs1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act0_qs0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act0_qs0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act0_qs1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act1_qs0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act1_qs1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, bool, float, bool, bool, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, bool, float, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, bool, float, float, bool, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, bool, float, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, bool, float, bool, bool, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, bool, float, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, bool, float, float, bool, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, bool, float, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, float, float, bool, bool, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, float, float, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, float, float, float, bool, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, bool, float, float, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off0_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, float, float, bool, bool, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off0_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, float, float, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off1_gi0_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, float, float, float, bool, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off1_gi1_int32_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<int32_t, float, float, float, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

#endif // DSQ_ENABLE_INT32_FP8E5M2

#ifdef DSQ_ENABLE_BF16_FP8E4M3
// ===== bfloat16_t -> fp8_e4m3fn_t =====
__global__ __aicore__ void dsq_base_nb_act1_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act1_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act1_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act1_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act1_qs1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act1_qs0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act0_qs1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act0_qs0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, int32_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act1_qs1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act1_qs0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act0_qs1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act0_qs0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, float, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act1_qs1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act1_qs0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act0_qs1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act0_qs0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, half, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act1_qs1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act1_qs0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act0_qs1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act0_qs0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, bfloat16_t, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act0_qs0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act0_qs1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act1_qs0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act1_qs1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, bool, bfloat16_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, bool, float, bool, bool, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, bool, float, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, bool, float, float, bool, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, bool, float, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, bool, float, bool, bool, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, bool, float, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, bool, float, float, bool, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, bool, float, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, float, float, bool, bool, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, float, float, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, float, float, float, bool, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, float, float, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off0_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, float, float, bool, bool, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off0_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, float, float, bool, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off1_gi0_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, float, float, float, bool, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off1_gi1_bf16_fp8e4m3(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, float, float, float, int32_t, fp8_e4m3fn_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

#endif // DSQ_ENABLE_BF16_FP8E4M3

#ifdef DSQ_ENABLE_BF16_FP8E5M2
// ===== bfloat16_t -> fp8_e5m2_t =====
__global__ __aicore__ void dsq_base_nb_act1_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act1_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act1_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act1_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_nb_act0_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act1_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bi32_act0_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act1_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp32_act0_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act1_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bfp16_act0_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act1_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_base_bbf16_act0_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act1_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_nb_act0_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act1_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bi32_act0_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act1_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp32_act0_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act1_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bfp16_act0_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, int64_t, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, float, bool, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, int64_t, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act1_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<float, bool, bool, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, int64_t, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, float, bool, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, int64_t, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_stat_bbf16_act0_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBaseStatic<bool, bool, bool, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act0_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_nb_act1_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act0_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bi32_act1_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act0_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp32_act1_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act0_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bfp16_act1_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, bool, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, bool, int32_t, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, bool, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act0_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<bool, float, int32_t, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, bool, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, bool, int32_t, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, bool, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_dynf_bbf16_act1_qs1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantDynamicNotFull<float, float, int32_t, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, ws, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act1_qs1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act1_qs0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act0_qs1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bi32_act0_qs0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, int32_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act1_qs1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act1_qs0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act0_qs1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp32_act0_qs0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, float, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act1_qs1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act1_qs0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act0_qs1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bfp16_act0_qs0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, half, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act1_qs1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act1_qs0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act0_qs1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_bbf16_act0_qs0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, bfloat16_t, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act0_qs0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act0_qs1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act1_qs0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_nlst_nb_act1_qs1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35NlastTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, bool, bfloat16_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, bool, float, bool, bool, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, bool, float, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, bool, float, float, bool, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act0_off1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, bool, float, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, bool, float, bool, bool, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, bool, float, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, bool, float, float, bool, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b0_act1_off1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, bool, float, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, float, float, bool, bool, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, float, float, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, float, float, float, bool, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act0_off1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, bool, float, float, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off0_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, float, float, bool, bool, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off0_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, float, float, bool, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off1_gi0_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, float, float, float, bool, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

__global__ __aicore__ void dsq_snf_b1_act1_off1_gi1_bf16_fp8e5m2(
    GM_ADDR x, GM_ADDR wScale, GM_ADDR aScale, GM_ADDR bias, GM_ADDR qScale, GM_ADDR qOffset,
    GM_ADDR gIdx, GM_ADDR y, GM_ADDR scale, GM_ADDR ws, const DequantSwigluQuantV35BaseTilingData td)
{
#if !defined(__NPU_HOST__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)AscendC::GetUserWorkspace(ws);
    if (g_coreType != AscendC::AIV) return;

    AscendC::TPipe pipe;
    DequantSwigluQuantV35Ops::DequantSwigluQuantStaticNotFull<bfloat16_t, float, float, float, float, int32_t, fp8_e5m2_t> op(&pipe);
    op.Init(x, wScale, aScale, bias, qScale, qOffset, gIdx, y, scale, &td);
    op.Process();
#endif
}

#endif // DSQ_ENABLE_BF16_FP8E5M2

// Total possible entry functions: 1092
#endif // DEQUANT_SWIGLU_QUANT_APT_H
