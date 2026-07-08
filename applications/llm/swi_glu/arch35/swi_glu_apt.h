/*!
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SWI_GLU_STORY_APT_H
#define SWI_GLU_STORY_APT_H

#include "swi_glu_impl.hpp"
#include "swi_glu_bf16.hpp"
#include "swi_glu_single.hpp"

__global__ __aicore__ void swi_glu_fp32(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                        const SwiGluTilingData tilingData)
{
#if !defined(__NPU_HOST__)
    (void)workspace;
    if (tilingData.isDoubleBuffer == 1) {
        SwigluSingle<SwigluVector<float, float, 2>, float, float> op;
        op.Init(input, nullptr, output, tilingData);
        op.Process();
    } else {
        SwigluSingle<SwigluVector<float, float, 1>, float, float> op;
        op.Init(input, nullptr, output, tilingData);
        op.Process();
    }
#endif
}

__global__ __aicore__ void swi_glu_fp16(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                        const SwiGluTilingData tilingData)
{
#if !defined(__NPU_HOST__)
    (void)workspace;
    if (tilingData.isDoubleBuffer == 1) {
        SwigluSingle<SwigluVectorBF16<half, float, half, 2>, half, half> op;
        op.Init(input, nullptr, output, tilingData);
        op.Process();
    } else {
        SwigluSingle<SwigluVectorBF16<half, float, half, 1>, half, half> op;
        op.Init(input, nullptr, output, tilingData);
        op.Process();
    }
#endif
}

__global__ __aicore__ void swi_glu_bf16(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                        const SwiGluTilingData tilingData)
{
#if !defined(__NPU_HOST__)
    (void)workspace;
    if (tilingData.isDoubleBuffer == 1) {
        SwigluSingle<SwigluVectorBF16<bfloat16_t, float, bfloat16_t, 2>, bfloat16_t, bfloat16_t> op;
        op.Init(input, nullptr, output, tilingData);
        op.Process();
    } else {
        SwigluSingle<SwigluVectorBF16<bfloat16_t, float, bfloat16_t, 1>, bfloat16_t, bfloat16_t> op;
        op.Init(input, nullptr, output, tilingData);
        op.Process();
    }
#endif
}

#endif // SWI_GLU_STORY_APT_H
