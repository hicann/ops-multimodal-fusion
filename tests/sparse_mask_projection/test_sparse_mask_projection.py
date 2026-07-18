#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Tianjin University Ltd
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "sparse_mask_projection"):
    pytest.skip("sparse_mask_projection not registered", allow_module_level=True)

DTYPES = [torch.float32, torch.float16]
CASES = [
    (5, 5, 5), (10, 5, 8), (8, 3, 6), (16, 8, 12), (32, 16, 20),
    (64, 32, 40), (100, 50, 60), (8, 0, 4), (16, 16, 0), (1, 1, 1),
    (32, 0, 0), (64, 64, 32), (128, 64, 80), (256, 128, 100),
    (50, 25, 40), (10, 10, 10), (20, 5, 15), (30, 10, 25),
    (40, 20, 30), (60, 30, 45), (80, 40, 60), (15, 8, 12),
    (25, 12, 20), (35, 18, 28), (45, 22, 36), (55, 28, 44),
    (70, 35, 55), (85, 42, 68), (95, 48, 76), (120, 60, 90),
]


def _golden(sp_idx, sp_val, mask_idx):
    out = np.zeros(len(mask_idx), dtype=np.float32)
    si = 0
    for i, mi in enumerate(mask_idx):
        while si < len(sp_idx) and sp_idx[si] < mi:
            si += 1
        if si < len(sp_idx) and sp_idx[si] == mi:
            out[i] = sp_val[si]
            si += 1
    return out


def _run(sp_idx, sp_val, mask_idx, dtype):
    si = torch.from_numpy(sp_idx.astype(np.int32)).npu()
    sv = torch.from_numpy(sp_val.astype(np.float32)).to(dtype).npu()
    mi = torch.from_numpy(mask_idx.astype(np.int32)).npu()
    return torch.ops.ops_multimodal_fusion.sparse_mask_projection(
        si, sv, mi
    ).cpu()


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not found")
@pytest.mark.parametrize(
    "sp_len,val_len,mask_len", CASES, ids=[f"sp{a}_m{c}" for a, b, c in CASES]
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_projection(sp_len, val_len, mask_len, dtype):
    rng = np.random.default_rng(hash((sp_len, mask_len)) % (10**7))
    if val_len > 0:
        sp_idx = np.sort(
            rng.choice(np.arange(0, sp_len * 2), size=val_len, replace=False)
        ).astype(np.int32)
    else:
        sp_idx = np.array([], dtype=np.int32)
    sp_val = rng.standard_normal(val_len).astype(np.float32)
    if mask_len > 0:
        mask_idx = np.sort(
            rng.choice(np.arange(0, sp_len * 2), size=mask_len, replace=False)
        ).astype(np.int32)
    else:
        mask_idx = np.array([], dtype=np.int32)
    actual = _run(sp_idx, sp_val, mask_idx, dtype)
    expected = torch.from_numpy(_golden(sp_idx, sp_val, mask_idx)).to(dtype)
    assert torch.equal(actual, expected) or torch.allclose(
        actual, expected, equal_nan=True
    )


def test_interface_exists():
    assert hasattr(torch.ops.ops_multimodal_fusion, "sparse_mask_projection")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not found")
def test_reject_int64():
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.sparse_mask_projection(
            torch.tensor([0], dtype=torch.int64).npu(),
            torch.randn(1).npu(),
            torch.tensor([0], dtype=torch.int32).npu(),
        )
