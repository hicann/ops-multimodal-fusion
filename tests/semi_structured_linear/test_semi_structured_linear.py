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

if not hasattr(torch.ops.ops_multimodal_fusion, "semi_structured_linear"):
    pytest.skip("semi_structured_linear not registered", allow_module_level=True)

VALID_MASKS = [0b0011, 0b0101, 0b0110, 0b1001, 0b1010, 0b1100]
CASES = [
    (1, 4, 4), (2, 4, 8), (1, 8, 4), (4, 4, 4), (2, 8, 8),
    (1, 16, 8), (4, 8, 8), (8, 4, 8), (2, 4, 16), (4, 4, 16),
    (1, 4, 16), (2, 16, 16), (4, 8, 16), (8, 8, 16), (16, 4, 8),
    (1, 32, 16), (2, 8, 32), (4, 16, 32), (8, 4, 16), (16, 8, 8),
    (1, 1, 4), (3, 6, 12), (2, 10, 20), (5, 5, 20), (6, 4, 24),
    (2, 2, 8), (10, 2, 8), (1, 8, 32), (4, 4, 8), (3, 3, 12),
    (7, 2, 16), (2, 12, 8),
]


def _make_2x4(m_dim, k_dim, rng):
    w_dense = np.zeros((m_dim, k_dim), dtype=np.float32)
    wc = np.zeros((m_dim, k_dim // 2), dtype=np.float32)
    md = np.zeros((m_dim, k_dim // 4), dtype=np.uint8)
    for m in range(m_dim):
        for g in range(k_dim // 4):
            mask = VALID_MASKS[rng.integers(0, 6)]
            md[m][g] = mask
            pos = [i for i in range(4) if mask & (1 << i)]
            vals = rng.standard_normal(2).astype(np.float32)
            wc[m][2 * g:2 * g + 2] = vals
            w_dense[m][4 * g + pos[0]] = vals[0]
            w_dense[m][4 * g + pos[1]] = vals[1]
    return w_dense, wc, md


def _run(inputs, m_dim, k_dim):
    x_np, wc_np, md_np, bias_np = inputs
    x = torch.from_numpy(x_np).npu()
    wc = torch.from_numpy(wc_np).npu()
    md = torch.from_numpy(md_np).npu()
    b = torch.from_numpy(bias_np).npu()
    return torch.ops.ops_multimodal_fusion.semi_structured_linear(
        x, wc, md, b, m_dim, k_dim
    ).cpu()


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not found")
@pytest.mark.parametrize("batch,m_dim,k_dim", CASES, ids=[f"b{a}_m{b}_k{c}" for a, b, c in CASES])
def test_linear(batch, m_dim, k_dim):
    rng = np.random.default_rng(hash((batch, m_dim, k_dim)) % (10**7))
    x = rng.standard_normal((batch, k_dim)).astype(np.float32)
    w_dense, wc, md = _make_2x4(m_dim, k_dim, rng)
    bias = rng.standard_normal(m_dim).astype(np.float32)
    actual = _run((x, wc, md, bias), m_dim, k_dim)
    expected = torch.from_numpy(x @ w_dense.T + bias)
    assert torch.allclose(
        actual, expected, rtol=1e-3, atol=1e-3
    ), f"max diff {(actual - expected).abs().max()}"


def test_interface_exists():
    assert hasattr(torch.ops.ops_multimodal_fusion, "semi_structured_linear")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not found")
def test_reject_k_not_div4():
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.semi_structured_linear(
            torch.randn(1, 5).npu(),
            torch.randn(4, 2).npu(),
            torch.zeros(4, 1, dtype=torch.uint8).npu(),
            torch.randn(4).npu(),
            4,
            5,
        )
