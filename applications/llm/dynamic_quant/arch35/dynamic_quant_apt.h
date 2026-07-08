/*!
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_quant_apt.h
 * \brief Non-template __global__ entry functions for the dynamic_quant standalone story.
 *
 * 24 entry functions = 4 quantModes x 2 xDtype x 3 yDtype.
 * Each entry dispatches at runtime on (hasSmooth, isSymmetrical, useDb).
 */
#ifndef DYNAMIC_QUANT_STORY_APT_H
#define DYNAMIC_QUANT_STORY_APT_H

#include "kernel_operator.h"
#include "dynamic_quant_regbase_full_load.h"
#include "dynamic_quant_regbase_large_shape_db.h"
#include "dynamic_quant_regbase_moe_full_load.h"
#include "dynamic_quant_regbase_moe_large_shape.h"

// ===========================================================================
// Part 1: Inner dispatch - device-pass real implementation / host-pass stubs
// ===========================================================================

#if !defined(__NPU_HOST__)

using namespace AscendC;

template<uint64_t V>
using UIntAsBool = std::integral_constant<bool, V != 0>;

// ---- COMMON_FULL_LOAD ----
template <typename xDtype, typename yDtype>
__aicore__ inline void dq_impl_fullload(
    GM_ADDR x, GM_ADDR smooth_scales,
    GM_ADDR y, GM_ADDR scale, GM_ADDR offset, GM_ADDR workspace,
    const DynamicQuantTilingData* tilingData,
    uint32_t hasSmooth, uint32_t isSymmetrical, uint32_t useDb)
{
    TPipe pipe;
#define FL_CASE(HS, SYM, DB)                                                                   \
    do {                                                                                        \
        DynamicQuantNDOpt::DynamicQuantRegbaseFullLoad<                                         \
            xDtype, yDtype, UIntAsBool<HS>::value, static_cast<uint32_t>((DB) + 1), (SYM)> op(&pipe); \
        op.Init(x, smooth_scales, y, scale, (SYM) ? nullptr : offset, workspace, tilingData);  \
        op.Process();                                                                           \
    } while (0)

    if (hasSmooth) {
        if (isSymmetrical) {
            if (useDb) { FL_CASE(1, true, 1); } else { FL_CASE(1, true, 0); }
        } else {
            if (useDb) { FL_CASE(1, false, 1); } else { FL_CASE(1, false, 0); }
        }
    } else {
        if (isSymmetrical) {
            if (useDb) { FL_CASE(0, true, 1); } else { FL_CASE(0, true, 0); }
        } else {
            if (useDb) { FL_CASE(0, false, 1); } else { FL_CASE(0, false, 0); }
        }
    }
#undef FL_CASE
}

// ---- COMMON_LARGE_SHAPE ----
template <typename xDtype, typename yDtype>
__aicore__ inline void dq_impl_largeshape(
    GM_ADDR x, GM_ADDR smooth_scales,
    GM_ADDR y, GM_ADDR scale, GM_ADDR offset, GM_ADDR workspace,
    const DynamicQuantTilingData& tilingData,
    uint32_t hasSmooth, uint32_t isSymmetrical)
{
    TPipe pipe;
#define LS_CASE(HS, SYM)                                                                       \
    do {                                                                                        \
        DynamicQuantNDOpt2::DynamicQuantLargeShapeDb<                                           \
            xDtype, yDtype, static_cast<int64_t>(HS), (SYM)> op(&pipe);                        \
        op.Init(x, smooth_scales, y, scale, (SYM) ? nullptr : offset, workspace, tilingData);  \
        op.Process();                                                                           \
    } while (0)

    if (hasSmooth) {
        if (isSymmetrical) { LS_CASE(1, true); } else { LS_CASE(1, false); }
    } else {
        if (isSymmetrical) { LS_CASE(0, true); } else { LS_CASE(0, false); }
    }
#undef LS_CASE
}

// ---- MOE_FULL_LOAD ----
template <typename xDtype, typename yDtype>
__aicore__ inline void dq_impl_moefullload(
    GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR group_index,
    GM_ADDR y, GM_ADDR scale, GM_ADDR offset, GM_ADDR workspace,
    const DynamicQuantTilingData* tilingData,
    uint32_t hasSmooth, uint32_t isSymmetrical, uint32_t useDb)
{
    TPipe pipe;
#define MFL_CASE(HS, SYM, DB)                                                                         \
    do {                                                                                               \
        DynamicQuantV2Op::DynamicQuantRegbaseFullLoadMOE<                                              \
            xDtype, yDtype, UIntAsBool<HS>::value, static_cast<uint32_t>((DB) + 1), (SYM)> op(&pipe); \
        op.Init(x, smooth_scales, group_index, y, scale,                                               \
                (SYM) ? nullptr : offset, workspace, tilingData);                                      \
        op.Process();                                                                                  \
    } while (0)

    if (hasSmooth) {
        if (isSymmetrical) {
            if (useDb) { MFL_CASE(1, true, 1); } else { MFL_CASE(1, true, 0); }
        } else {
            if (useDb) { MFL_CASE(1, false, 1); } else { MFL_CASE(1, false, 0); }
        }
    } else {
        if (isSymmetrical) {
            if (useDb) { MFL_CASE(0, true, 1); } else { MFL_CASE(0, true, 0); }
        } else {
            if (useDb) { MFL_CASE(0, false, 1); } else { MFL_CASE(0, false, 0); }
        }
    }
#undef MFL_CASE
}

// ---- MOE_LARGE_SHAPE ----
template <typename xDtype, typename yDtype>
__aicore__ inline void dq_impl_moelargeshape(
    GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR group_index,
    GM_ADDR y, GM_ADDR scale, GM_ADDR offset, GM_ADDR workspace,
    const DynamicQuantTilingData& tilingData,
    uint32_t hasSmooth, uint32_t isSymmetrical)
{
    TPipe pipe;
#define MLS_CASE(HS, SYM)                                                                      \
    do {                                                                                        \
        DynamicQuantRegBase::DynamicQuantLargeShapeMOE<                                         \
            xDtype, yDtype, static_cast<int64_t>(HS), (SYM)> op(&pipe);                        \
        op.Init(x, smooth_scales, group_index, y, scale,                                        \
                (SYM) ? nullptr : offset, workspace, tilingData);                               \
        op.Process();                                                                           \
    } while (0)

    if (hasSmooth) {
        if (isSymmetrical) { MLS_CASE(1, true); } else { MLS_CASE(1, false); }
    } else {
        if (isSymmetrical) { MLS_CASE(0, true); } else { MLS_CASE(0, false); }
    }
#undef MLS_CASE
}

#else // __NPU_HOST__ - empty stubs so __global__ signatures compile in host pass

template <typename xDtype, typename yDtype>
__aicore__ inline void dq_impl_fullload(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    const DynamicQuantTilingData*, uint32_t, uint32_t, uint32_t) {}

template <typename xDtype, typename yDtype>
__aicore__ inline void dq_impl_largeshape(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    const DynamicQuantTilingData&, uint32_t, uint32_t) {}

template <typename xDtype, typename yDtype>
__aicore__ inline void dq_impl_moefullload(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    const DynamicQuantTilingData*, uint32_t, uint32_t, uint32_t) {}

template <typename xDtype, typename yDtype>
__aicore__ inline void dq_impl_moelargeshape(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    const DynamicQuantTilingData&, uint32_t, uint32_t) {}

#endif // !defined(__NPU_HOST__)

// ===========================================================================
// Part 2: Overflow mode save/restore (Arch35 only, no-op otherwise)
// ===========================================================================

#ifndef FLOAT_OVERFLOW_MODE_CTRL
#define FLOAT_OVERFLOW_MODE_CTRL 60
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
  #define __DQ_SAVE_OVERFLOW__   int64_t oriOvf = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
  #define __DQ_RESTORE_OVERFLOW__ AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOvf);
#else
  #define __DQ_SAVE_OVERFLOW__
  #define __DQ_RESTORE_OVERFLOW__
#endif

// ===========================================================================
// Part 3: __global__ entry macros - ALWAYS visible (host sees empty body via stubs)
// ===========================================================================

#define DQ_ENTRY_FULL_LOAD(SUFFIX, X_TYPE, Y_TYPE)                                              \
__global__ __aicore__ void dq_fullload_##SUFFIX(                                                \
    GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR group_index,                                      \
    GM_ADDR y, GM_ADDR scale, GM_ADDR offset, GM_ADDR workspace,                                \
    const DynamicQuantTilingData tilingData,                                                     \
    uint32_t hasSmooth, uint32_t isSymmetrical, uint32_t useDb)                                  \
{                                                                                               \
    (void)group_index;                                                                          \
    __DQ_SAVE_OVERFLOW__                                                                        \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                             \
    dq_impl_fullload<X_TYPE, Y_TYPE>(                                                           \
        x, smooth_scales, y, scale, offset, workspace,                                          \
        &tilingData, hasSmooth, isSymmetrical, useDb);                                          \
    __DQ_RESTORE_OVERFLOW__                                                                     \
}

#define DQ_ENTRY_LARGE_SHAPE(SUFFIX, X_TYPE, Y_TYPE)                                            \
__global__ __aicore__ void dq_largeshape_##SUFFIX(                                              \
    GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR group_index,                                      \
    GM_ADDR y, GM_ADDR scale, GM_ADDR offset, GM_ADDR workspace,                                \
    const DynamicQuantTilingData tilingData,                                                     \
    uint32_t hasSmooth, uint32_t isSymmetrical, uint32_t useDb)                                  \
{                                                                                               \
    (void)group_index; (void)useDb;                                                             \
    __DQ_SAVE_OVERFLOW__                                                                        \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                             \
    dq_impl_largeshape<X_TYPE, Y_TYPE>(                                                         \
        x, smooth_scales, y, scale, offset, workspace,                                          \
        tilingData, hasSmooth, isSymmetrical);                                                   \
    __DQ_RESTORE_OVERFLOW__                                                                     \
}

#define DQ_ENTRY_MOE_FULL_LOAD(SUFFIX, X_TYPE, Y_TYPE)                                          \
__global__ __aicore__ void dq_moefullload_##SUFFIX(                                             \
    GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR group_index,                                      \
    GM_ADDR y, GM_ADDR scale, GM_ADDR offset, GM_ADDR workspace,                                \
    const DynamicQuantTilingData tilingData,                                                     \
    uint32_t hasSmooth, uint32_t isSymmetrical, uint32_t useDb)                                  \
{                                                                                               \
    __DQ_SAVE_OVERFLOW__                                                                        \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                             \
    dq_impl_moefullload<X_TYPE, Y_TYPE>(                                                        \
        x, smooth_scales, group_index, y, scale, offset, workspace,                             \
        &tilingData, hasSmooth, isSymmetrical, useDb);                                          \
    __DQ_RESTORE_OVERFLOW__                                                                     \
}

#define DQ_ENTRY_MOE_LARGE_SHAPE(SUFFIX, X_TYPE, Y_TYPE)                                        \
__global__ __aicore__ void dq_moelargeshape_##SUFFIX(                                           \
    GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR group_index,                                      \
    GM_ADDR y, GM_ADDR scale, GM_ADDR offset, GM_ADDR workspace,                                \
    const DynamicQuantTilingData tilingData,                                                     \
    uint32_t hasSmooth, uint32_t isSymmetrical, uint32_t useDb)                                  \
{                                                                                               \
    (void)useDb;                                                                                \
    __DQ_SAVE_OVERFLOW__                                                                        \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                             \
    dq_impl_moelargeshape<X_TYPE, Y_TYPE>(                                                      \
        x, smooth_scales, group_index, y, scale, offset, workspace,                             \
        tilingData, hasSmooth, isSymmetrical);                                                   \
    __DQ_RESTORE_OVERFLOW__                                                                     \
}

// ===========================================================================
// Part 4: Instantiate all 24 entries
// ===========================================================================

#define DQ_ENTRIES_FOR_MODE(MODE_MACRO)                     \
    MODE_MACRO(half_int8,       half, int8_t)               \
    MODE_MACRO(half_fp8e4m3,    half, fp8_e4m3fn_t)         \
    MODE_MACRO(half_fp8e5m2,    half, fp8_e5m2_t)           \
    MODE_MACRO(bf16_int8,       bfloat16_t, int8_t)         \
    MODE_MACRO(bf16_fp8e4m3,    bfloat16_t, fp8_e4m3fn_t)   \
    MODE_MACRO(bf16_fp8e5m2,    bfloat16_t, fp8_e5m2_t)

DQ_ENTRIES_FOR_MODE(DQ_ENTRY_FULL_LOAD)
DQ_ENTRIES_FOR_MODE(DQ_ENTRY_LARGE_SHAPE)
DQ_ENTRIES_FOR_MODE(DQ_ENTRY_MOE_FULL_LOAD)
DQ_ENTRIES_FOR_MODE(DQ_ENTRY_MOE_LARGE_SHAPE)

#endif // DYNAMIC_QUANT_STORY_APT_H
