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
"""Tests for ops_multimodal_fusion.mode.

Computes the most-frequent value along ``dim`` and returns ``(values, indices)``.
Values: strict equality with torch.mode (ties broken to the smallest value).
Indices: verified via gather (x[idx] == mode_value); the exact idx among
multiple valid positions is implementation-defined.

Test case counts:
  - test_mode_small        : 12 cases (one (dtype, keepdim) per fast shape)
  - test_mode_large        : 72 cases (60 dtype/keepdim combos for fast shapes
                                       not picked by small + 12 large-only shape combos)
  - test_mode_interface_exist     : 1
  - test_mode_invalid_dim         : 1
  - test_mode_int64_rejected      : 1
  - test_mode_bf16_rejected       : 1
  - Total                  : 88 cases
"""

from collections import namedtuple
from itertools import product

import pytest
import torch
import torch_npu  # noqa: F401  (registers NPU dispatch key)

import ops_multimodal_fusion  # noqa: F401  (registers torch.ops.ops_multimodal_fusion.mode)

if not hasattr(torch.ops.ops_multimodal_fusion, "mode"):
    pytest.skip(
        "ops_multimodal_fusion.mode not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_mode_interface_exist():
    """The 'ops_multimodal_fusion.mode' operator is registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "mode"), \
        "The 'mode' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ---------------------------------------------------------------------------
# Input generators.
#
# Mode tests need three input regimes:
#   1. All-unique along dim: every value from zero to dim_size minus one appears
#      once; the mode is the smallest value at its randperm position.
#   2. All-equal along dim: every value is the same constant; the mode is that
#      constant and the last index is dim_size minus one.
#   3. Controlled ties or repetitions: for example half ones and half twos, or
#      random draws from a small alphabet so several values share the max count.
#
# All generators are deterministic via seed_key so re-runs are reproducible.
# ---------------------------------------------------------------------------

def _seed(seed_key):
    # Hash the repr so seeding stays deterministic even when seed_key holds
    # list-valued gen_kwargs such as an alphabet list; hashing the raw tuple would
    # raise an unhashable-type error when it recurses into the list.
    torch.manual_seed(abs(hash(repr(seed_key))) % 10_000_000)


def _make_unique_along_dim(shape, dim, dtype, seed_key):
    """Independent randperm(dim_size) along each (outer, inner) slice."""
    _seed(seed_key)
    sizes = list(shape)
    ndim = len(sizes)
    dim_pos = dim if dim >= 0 else dim + ndim
    dim_size = sizes[dim_pos]
    outer = 1
    for i in range(dim_pos):
        outer *= sizes[i]
    inner = 1
    for i in range(dim_pos + 1, ndim):
        inner *= sizes[i]
    perms = torch.stack(
        [torch.randperm(dim_size) for _ in range(outer * inner)], dim=0
    )
    perms = perms.view(outer, inner, dim_size).permute(0, 2, 1).contiguous()
    arr = perms.view(*sizes)
    return arr.to(dtype)


def _make_constant_along_dim(shape, dim, dtype, fill_value, seed_key):
    """Every element is fill_value (e.g., 7). mode = fill_value, idx =
    last position along dim.
    """
    _seed(seed_key)
    arr = torch.full(shape, fill_value)
    return arr.to(dtype)


def _make_repeated_run_along_dim(shape, dim, dtype, run_kwargs, seed_key):
    """Construct rows where ``run_value`` appears ``run_count`` times and
    the remaining ``dim_size - run_count`` positions are random distinct
    integers from ``alphabet`` (excluding ``run_value``). Guarantees a
    unique mode = run_value when run_count > dim_size - run_count.
    Positions of the run are randomly permuted per slice.
    """
    _seed(seed_key)
    run_value = run_kwargs["run_value"]
    run_count = run_kwargs["run_count"]
    alphabet = run_kwargs["alphabet"]
    sizes = list(shape)
    ndim = len(sizes)
    dim_pos = dim if dim >= 0 else dim + ndim
    dim_size = sizes[dim_pos]
    assert run_count <= dim_size
    outer = 1
    for i in range(dim_pos):
        outer *= sizes[i]
    inner = 1
    for i in range(dim_pos + 1, ndim):
        inner *= sizes[i]

    n_pairs = outer * inner
    rest = dim_size - run_count
    # pool of non-run_value alphabet values
    pool = [v for v in alphabet if v != run_value]
    rows = []
    for _ in range(n_pairs):
        # random distinct values from pool for the rest positions; allow
        # duplicates since rest may exceed pool size — but ensure no
        # individual value reaches run_count copies.
        if rest <= len(pool):
            # sample without replacement
            idxs = torch.randperm(len(pool))[:rest].tolist()
            row_rest = [pool[i] for i in idxs]
        else:
            # rare path: sample with replacement; harmless because
            # run_count > rest already guarantees mode is run_value.
            row_rest = [pool[i.item() % len(pool)]
                        for i in torch.randint(0, len(pool), (rest,))]
        row = [run_value] * run_count + row_rest
        # shuffle so run_value occurs at varied positions
        perm = torch.randperm(dim_size).tolist()
        row = [row[i] for i in perm]
        rows.append(row)
    arr = torch.tensor(rows)  # (n_pairs, dim_size)
    arr = arr.view(outer, inner, dim_size).permute(0, 2, 1).contiguous()
    return arr.view(*sizes).to(dtype)


def _make_low_alphabet_along_dim(shape, dim, dtype, alphabet_size, seed_key):
    """Each lane drawn uniformly from {0, .., alphabet_size-1}. Typical
    "mode" workload where dim_size >> alphabet_size and the most-frequent
    value is well-defined (with high probability).
    """
    _seed(seed_key)
    arr = torch.randint(0, alphabet_size, shape)
    return arr.to(dtype)


# ---------------------------------------------------------------------------
# Case matrix.
#
# Shapes are split into "fast" (SMALL_FAST_SHAPES) and "large" (LARGE_ONLY_SHAPES).
# SMALL_SELECT picks ONE (dtype, keepdim) per fast shape for the small test
# function; every other (dtype, keepdim) combination of the fast shapes plus
# all large-only shape combos go into the large test function.
# ---------------------------------------------------------------------------

# Fast shapes (small UB footprint, quick to run).
# columns: shape, dim, gen_name, gen_kwargs, label
SMALL_FAST_SHAPES = [
    ((64,), 0, "unique", {}, "1D_unique_vec_eq"),
    ((64,), -1, "unique", {}, "1D_unique_dim_neg"),
    ((64,), -1, "const", {"fill_value": 7}, "1D_const7"),
    ((32,), -1, "run", {"run_value": 5, "run_count": 20,
                               "alphabet": list(range(20))}, "1D_run5"),
    ((63,), -1, "unique", {}, "1D_vec_minus1"),
    ((65,), -1, "unique", {}, "1D_vec_plus1"),
    ((8, 64), -1, "unique", {}, "2D_last_unique"),
    ((4, 16), -1, "run", {"run_value": 3, "run_count": 9,
                               "alphabet": list(range(10))}, "2D_run3"),
    ((4, 32), -1, "unique", {}, "2D_keepdim_unique"),
    ((8, 32), -2, "unique", {}, "2D_negdim_minus2"),
    ((8, 1), -1, "unique", {}, "2D_dim1"),
    ((3, 33), -1, "run", {"run_value": 2, "run_count": 18,
                               "alphabet": list(range(8))}, "2D_run_vec_minus"),
]

# Large-only shapes (UB-tight, slow, or innerSize>1 strided paths).
LARGE_ONLY_SHAPES = [
    ((128, 8), 0, "unique", {}, "L1_strided_first"),
    ((4, 256, 8), 1, "unique", {}, "L2_3D_middle"),
    ((4096, 64), -1, "unique", {}, "L3_outer_4k"),
    ((16, 1023), -1, "run", {"run_value": 7, "run_count": 600,
                                  "alphabet": list(range(64))}, "L4_big_dim_run"),
    ((64, 512), -1, "low_alph", {"alphabet_size": 10}, "L5_low_alphabet"),
    ((32, 1024), -1, "unique", {}, "L6_worst_unique"),
]

DTYPES_ALL = [torch.float32, torch.float16, torch.int32]
DTYPES_LARGE_SHAPES = [torch.float32, torch.int32]  # large-only shapes skip fp16 (mantissa precision on integer perms)

# Minimum-coverage selection for the small test: ONE (dtype, keepdim) per fast shape.
# Selection notes:
#   - dtype distribution roughly 1:1:1 across fp32 / fp16 / int32
#   - keepdim=True exercised on exactly one shape (host-side meta only)
#   - vec_width boundary shapes use the dtype whose vec_width matches the boundary
#   - ties (const, run) explicitly covered
#   - int32 picked on shapes with exact-integer outputs (avoids float rounding ambiguity)
SMALL_SELECT = {
    "1D_unique_vec_eq": (torch.int32, False),  # dim=0 main path; int32 = vec_width-exact
    "1D_unique_dim_neg": (torch.float32, False),  # negative dim main path
    "1D_const7": (torch.float32, False),  # const ties + fp32 (validates middle-of-run idx)
    "1D_run5": (torch.int32, False),  # 1D run ties + int32
    "1D_vec_minus1": (torch.float32, False),  # vec_width-1 boundary (fp32)
    "1D_vec_plus1": (torch.float32, False),  # vec_width+1 chunked boundary (fp32)
    "2D_last_unique": (torch.float16, False),  # 2D last-dim + fp16 trailing-fill (dim_size<128)
    "2D_run3": (torch.float16, False),  # 2D run ties + fp16
    "2D_keepdim_unique": (torch.float32, True),   # ONLY keepdim=True coverage
    "2D_negdim_minus2": (torch.float32, False),  # negative dim != -1 main path
    "2D_dim1": (torch.int32, False),  # dim_size=1 boundary + int32
    "2D_run_vec_minus": (torch.float16, False),  # fp16 cross-chunk + run ties (33 lanes)
}

# One test case's full parameter set, bundled so case-driven helpers take a
# single argument instead of a long positional list.
Case = namedtuple("Case", "shape dim gen_name gen_kwargs label dtype keepdim")


def _case_id(case):
    return f"{case.label}-{str(case.dtype).split('.')[-1]}-kd{int(case.keepdim)}"


# Fast-set cases — 12 explicit septuples (shape, dim, gen_name, gen_kwargs, label, dtype, keepdim).
CASES_SMALL = [
    (*shape_entry, *SMALL_SELECT[shape_entry[-1]])
    for shape_entry in SMALL_FAST_SHAPES
]


# Large cases: every fast-shape combo not selected for SMALL plus all large-only
# shape combos (72 total).
def _build_cases_large():
    cases = []
    for shape_entry in SMALL_FAST_SHAPES:
        for dtype, keepdim in product(DTYPES_ALL, [False, True]):
            if (dtype, keepdim) != SMALL_SELECT[shape_entry[-1]]:
                cases.append((*shape_entry, dtype, keepdim))
    for shape_entry in LARGE_ONLY_SHAPES:
        for dtype in DTYPES_LARGE_SHAPES:
            cases.append((*shape_entry, dtype, False))
    return cases


CASES_LARGE = _build_cases_large()


_GENERATORS = {
    "unique": _make_unique_along_dim,
    "const": lambda shape, dim, dtype, sk, **kw: _make_constant_along_dim(
                    shape, dim, dtype, kw["fill_value"], sk),
    "run": lambda shape, dim, dtype, sk, **kw: _make_repeated_run_along_dim(
                    shape, dim, dtype, kw, sk),
    "low_alph": lambda shape, dim, dtype, sk, **kw: _make_low_alphabet_along_dim(
                    shape, dim, dtype, kw["alphabet_size"], sk),
}


def _gen_input(case, seed_key):
    gen = _GENERATORS[case.gen_name]
    if case.gen_name == "unique":
        return gen(case.shape, case.dim, case.dtype, seed_key)
    return gen(case.shape, case.dim, case.dtype, seed_key, **case.gen_kwargs)


def _expected(x_cpu, dim, keepdim):
    """CPU reference. torch.mode returns smallest value on tie; index policy is
    implementation-defined and may differ between CPU/CUDA. Tests verify our
    indices via gather (see _run) instead of strict-equal.
    """
    return torch.mode(x_cpu, dim=dim, keepdim=keepdim)


def _run(x_cpu, dim, keepdim, dtype, label):
    expected_v, _ = _expected(x_cpu, dim, keepdim)
    rv_npu, ri_npu = torch.ops.ops_multimodal_fusion.mode(
        x_cpu.npu(), int(dim), bool(keepdim)
    )
    result_v = rv_npu.cpu()
    result_i = ri_npu.cpu()

    assert result_v.dtype == dtype, \
        f"[{label}] values dtype mismatch: got {result_v.dtype}, want {dtype}"
    assert result_i.dtype == torch.int64, \
        f"[{label}] indices dtype mismatch: got {result_i.dtype}, want torch.int64"
    assert result_v.shape == expected_v.shape, \
        f"[{label}] values shape: got {tuple(result_v.shape)}, want {tuple(expected_v.shape)}"

    # values: strict equal (mode value is deterministic — tie-break = smallest).
    assert torch.equal(result_v, expected_v), (
        f"[{label}] values mismatch (dtype={dtype}, shape={tuple(x_cpu.shape)}, "
        f"dim={dim}, keepdim={keepdim})\n"
        f"expected={expected_v}\nresult ={result_v}"
    )

    # indices: gather x_cpu at result_i and verify it equals mode value. This
    # tolerates implementation-defined idx policy on ties (PyTorch CPU mode_kernel_impl
    # uses sort-based middle-of-run; our kernel uses N-pass middle-of-occurrences).
    # Both are valid mode indices for the same mode value.
    dim_norm = dim if dim >= 0 else dim + x_cpu.dim()
    if keepdim:
        gathered = x_cpu.gather(dim_norm, result_i.long()).squeeze(dim_norm)
        ref = expected_v.squeeze(dim_norm)
    else:
        gathered = x_cpu.gather(dim_norm, result_i.long().unsqueeze(dim_norm)).squeeze(dim_norm)
        ref = expected_v
    assert torch.equal(gathered, ref), (
        f"[{label}] indices point to wrong value (dtype={dtype}, shape={tuple(x_cpu.shape)}, "
        f"dim={dim}, keepdim={keepdim})\n"
        f"result_i ={result_i}\nx[result_i]={gathered}\nmode value={ref}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", [Case(*c) for c in CASES_SMALL], ids=_case_id)
def test_mode_small(case):
    seed_key = ("small", tuple(case.shape), int(case.dim), case.gen_name,
                tuple(sorted(case.gen_kwargs.items())), str(case.dtype), bool(case.keepdim))
    x_cpu = _gen_input(case, seed_key)
    _run(x_cpu, case.dim, case.keepdim, case.dtype, case.label)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("case", [Case(*c) for c in CASES_LARGE], ids=_case_id)
def test_mode_large(case):
    seed_key = ("large", tuple(case.shape), int(case.dim), case.gen_name,
                tuple(sorted(case.gen_kwargs.items())), str(case.dtype), bool(case.keepdim))
    x_cpu = _gen_input(case, seed_key)
    _run(x_cpu, case.dim, case.keepdim, case.dtype, case.label)


# ---------------------------------------------------------------------------
# Negative-path tests: argument and dtype validation.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_mode_invalid_dim():
    x = torch.randn(8, 16).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.mode(x, 5, False)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_mode_int64_rejected():
    """int64 inputs are deferred on this platform."""
    x = torch.randint(0, 1000, (8, 16), dtype=torch.int64).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.mode(x, -1, False)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_mode_bf16_rejected():
    """bf16 inputs are deferred (bisheng cannot select scalar `<` on __bf16)."""
    x = torch.randn(8, 16, dtype=torch.bfloat16).npu()
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.mode(x, -1, False)
