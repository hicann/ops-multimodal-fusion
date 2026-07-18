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

if not hasattr(torch.ops.ops_multimodal_fusion, "sparse_mask_intersection"):
    pytest.skip("sparse_mask_intersection not registered", allow_module_level=True)

DTYPES = [torch.float32, torch.float16]
SIZES = [
    (8, 1), (8, 3), (8, 5), (16, 4), (16, 8), (32, 10),
    (64, 32), (100, 50), (128, 64), (256, 128), (1000, 100),
    (5000, 500), (8, 0), (1, 1), (16, 16), (32, 0),
    (1024, 512), (2048, 1024), (50, 50), (200, 200), (8, 8),
    (64, 64), (128, 1), (10, 10), (1000, 1), (512, 256),
    (4, 4), (2048, 2048), (3000, 500), (16, 7), (256, 1),
]


def _golden(src, mask):
    out = np.zeros(len(mask), dtype=src.dtype)
    for i, idx in enumerate(mask):
        if 0 <= idx < len(src):
            out[i] = src[idx]
    return out


def _run(src, mask, dtype):
    s = torch.from_numpy(
        src.astype(np.float32 if dtype == torch.float32 else np.float16)
    ).to(dtype).npu()
    m = torch.from_numpy(mask.astype(np.int32)).npu()
    return torch.ops.ops_multimodal_fusion.sparse_mask_intersection(s, m).cpu()


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not found")
@pytest.mark.parametrize("src_len,mask_len", SIZES, ids=[f"s{a}_m{b}" for a, b in SIZES])
@pytest.mark.parametrize("dtype", DTYPES)
def test_intersection(src_len, mask_len, dtype):
    rng = np.random.default_rng(hash((src_len, mask_len)) % (10**7))
    src = rng.standard_normal(src_len)
    if mask_len == 0:
        mask = np.array([], dtype=np.int32)
    else:
        mask = rng.integers(0, max(1, src_len), size=mask_len, dtype=np.int32)
    actual = _run(src, mask, dtype)
    expected = torch.from_numpy(_golden(src, mask)).to(dtype)
    assert torch.equal(actual, expected) or torch.allclose(
        actual, expected, equal_nan=True
    )


def test_interface_exists():
    assert hasattr(torch.ops.ops_multimodal_fusion, "sparse_mask_intersection")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not found")
def test_reject_int64_indices():
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.sparse_mask_intersection(
            torch.randn(5).npu(),
            torch.tensor([0, 1], dtype=torch.int64).npu(),
        )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not found")
def test_reject_2d_source():
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.sparse_mask_intersection(
            torch.randn(5, 5).npu(),
            torch.tensor([0], dtype=torch.int32).npu(),
        )
