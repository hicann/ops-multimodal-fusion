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

if not hasattr(torch.ops.ops_multimodal_fusion, "sparse_binary_intersect"):
    pytest.skip("sparse_binary_intersect not registered", allow_module_level=True)

DTYPES = [torch.float32, torch.float16]
CASES = [
    (5, 5), (10, 5), (8, 8), (16, 8), (32, 16), (64, 32), (100, 50),
    (0, 5), (5, 0), (0, 0), (1, 1), (16, 16), (32, 32), (64, 64),
    (10, 20), (20, 10), (50, 50), (100, 100), (3, 7), (7, 3),
    (15, 25), (25, 15), (40, 60), (60, 40), (8, 3), (3, 8),
    (12, 12), (24, 24), (48, 48), (200, 100),
]


def _golden(ia, va, ib, vb):
    out_idx, out_val = [], []
    ai = bi = 0
    while ai < len(ia) and bi < len(ib):
        if ia[ai] == ib[bi]:
            out_idx.append(ia[ai])
            out_val.append(va[ai] * vb[bi])
            ai += 1
            bi += 1
        elif ia[ai] < ib[bi]:
            ai += 1
        else:
            bi += 1
    return (
        np.array(out_idx, dtype=np.int32),
        np.array(out_val, dtype=np.float32),
    )


def _run(ia, va, ib, vb, dtype):
    ia_t = torch.from_numpy(ia.astype(np.int32)).npu()
    va_t = torch.from_numpy(va.astype(np.float32)).to(dtype).npu()
    ib_t = torch.from_numpy(ib.astype(np.int32)).npu()
    vb_t = torch.from_numpy(vb.astype(np.float32)).to(dtype).npu()
    oi, ov = torch.ops.ops_multimodal_fusion.sparse_binary_intersect(
        ia_t, va_t, ib_t, vb_t
    )
    return oi.cpu(), ov.cpu()


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not found")
@pytest.mark.parametrize("a_len,b_len", CASES, ids=[f"a{a}_b{b}" for a, b in CASES])
@pytest.mark.parametrize("dtype", DTYPES)
def test_intersect(a_len, b_len, dtype):
    rng = np.random.default_rng(hash((a_len, b_len)) % (10**7))
    max_idx = max(a_len, b_len, 10)
    if a_len > 0:
        ia = np.sort(
            rng.choice(np.arange(0, max_idx * 2), size=a_len, replace=False)
        ).astype(np.int32)
    else:
        ia = np.array([], dtype=np.int32)
    if b_len > 0:
        ib = np.sort(
            rng.choice(np.arange(0, max_idx * 2), size=b_len, replace=False)
        ).astype(np.int32)
    else:
        ib = np.array([], dtype=np.int32)
    va = rng.standard_normal(a_len).astype(np.float32)
    vb = rng.standard_normal(b_len).astype(np.float32)
    oi, ov = _run(ia, va, ib, vb, dtype)
    gi, gv = _golden(ia, va, ib, vb)
    assert len(oi) == len(gi), f"length mismatch {len(oi)} vs {len(gi)}"
    if len(gi) > 0:
        assert torch.equal(oi, torch.from_numpy(gi))
        assert torch.allclose(
            ov, torch.from_numpy(gv).to(dtype), rtol=1e-3, atol=1e-3
        )


def test_interface_exists():
    assert hasattr(torch.ops.ops_multimodal_fusion, "sparse_binary_intersect")
