#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tests for ops_multimodal_fusion.triu_indices.

torch.triu_indices(row, col, offset=0, *, dtype) -> Tensor(2, N): row-major
coordinates of the upper triangle {(r,c): c >= r+offset}. Output is exact
integer; reference is torch.triu_indices on CPU and the assertion is strict
torch.equal (no tolerance) for both shape and values.

Dispatch scope:
  - dtype int32 / int64 only (out_int32 selects), matching
    OpInfo('triu_indices', dtypes=(int32, int64))
  - row, col >= 0; triu_size and max(row,col) <= INT32_MAX

Test case counts:
  - test_triu_indices_small             : 7 cases  (one (struct, dtype) per fast shape)
  - test_triu_indices_large             : 23 cases (flipped-dtype fallout + large-only shapes x2 dtypes)
  - test_triu_indices_interface_exist   : 1
  - test_triu_indices_negative_row_rejected : 1
  - test_triu_indices_negative_col_rejected : 1
  - Total                               : 33 cases
"""

import pytest
import torch
import torch_npu  # noqa: F401  (registers NPU dispatch key)

import ops_multimodal_fusion  # noqa: F401  (registers torch.ops.ops_multimodal_fusion.triu_indices)

if not hasattr(torch.ops.ops_multimodal_fusion, "triu_indices"):
    pytest.skip(
        "ops_multimodal_fusion.triu_indices not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_triu_indices_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "triu_indices"), \
        "The 'triu_indices' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Case matrix. Each row: (row, col, offset, out_int32, label).
# Labels avoid the substrings 'small'/'large' so `-k "not large"` filtering
# never collides with a case id.
# ---------------------------------------------------------------------------

# Fast shapes (quick set). Minimal coverage: one (struct, dtype) each.
#   - dtype mix: int64 x4 / int32 x3
#   - both empty regimes (col zero, row zero) covered
#   - offset sign: zero / positive / negative (incl. full-row rectangle)
#   - vec_width (int32=64) boundary: single-chunk (N=36) and multi-chunk (N=78)
CASES_SMALL = [
    (3, 3, 0, False, "tri_diag_i64"),     # official; square; default int64
    (4, 3, -1, True, "tri_neg_off_i32"),  # official; -offset; full-row rect; int32 direct
    (4, 3, 1, False, "tri_pos_off_i64"),  # official; +offset; some rows empty
    (20, 0, 0, True, "tri_col0_empty"),   # empty (2,0) via col=0
    (0, 0, 0, False, "tri_row0_empty"),   # empty (2,0) via row=0
    (8, 8, 0, True, "tri_sq_veq_i32"),   # square; triu_size=36, single chunk
    (12, 12, 0, False, "tri_sq_vplus_i64"), # triu_size=78 > 64, multi-chunk
]

# Large-only shapes: multi-core / multi-chunk / big offsets, both dtypes.
LARGE_ONLY = [
    (0, 20, 0, "tri_row0_col20"),     # official empty (0, col)
    (20, 0, 5, "tri_col0_off"),       # empty with offset
    (1, 1, 0, "tri_min1"),           # minimal non-empty (N=1)
    (64, 64, 0, "tri_sq64"),           # square, multi-core
    (128, 96, -10, "tri_negoff_rect"),    # -offset + full-row rectangle, larger
    (100, 100, 50, "tri_bigpos_off"),     # large +offset (many rows empty)
    (300, 256, 0, "tri_tall_rect"),      # tall rectangular, multi-core+chunk
    (2, 68435455, 68435453, "tri_widecol"),  # wide-col, tiny N=3 (iota stays small)
]


def _build_cases_large():
    cases = [(r, c, off, (not i32), lab + "_flip")
             for (r, c, off, i32, lab) in CASES_SMALL]
    for (r, c, off, lab) in LARGE_ONLY:
        for i32 in (False, True):
            cases.append((r, c, off, i32, lab))
    return cases


CASES_LARGE = _build_cases_large()


def _run_case(row, col, offset, out_int32, label):
    ref_dtype = torch.int32 if out_int32 else torch.int64
    expected = torch.triu_indices(row, col, offset, dtype=ref_dtype)
    result = torch.ops.ops_multimodal_fusion.triu_indices(row, col, offset, out_int32).cpu()
    assert result.dtype == ref_dtype, (
        f"dtype mismatch: got {result.dtype}, want {ref_dtype} (label={label})"
    )
    assert result.shape == expected.shape, (
        f"shape mismatch: got {tuple(result.shape)}, want {tuple(expected.shape)} "
        f"(label={label}, row={row}, col={col}, offset={offset})"
    )
    assert torch.equal(result, expected), (
        f"value mismatch (label={label}, row={row}, col={col}, offset={offset}, "
        f"out_int32={out_int32})\n"
        f"  expected: {expected.flatten().tolist()[:32]}\n"
        f"  got     : {result.flatten().tolist()[:32]}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("row,col,offset,out_int32,label", CASES_SMALL)
def test_triu_indices_small(row, col, offset, out_int32, label):
    _run_case(row, col, offset, out_int32, label)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("row,col,offset,out_int32,label", CASES_LARGE)
def test_triu_indices_large(row, col, offset, out_int32, label):
    _run_case(row, col, offset, out_int32, label)


# Negative-path tests: argument validation.

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_triu_indices_negative_row_rejected():
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.triu_indices(-1, 5, 0, False)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_triu_indices_negative_col_rejected():
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.triu_indices(5, -1, 0, False)
