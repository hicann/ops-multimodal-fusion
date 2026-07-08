#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Tianjin University Ltd
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Test suite for foreach_ceil — golden reference: elementwise torch.ceil per tensor."""

import pytest
import torch
import torch_npu  # noqa: F401

import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "foreach_ceil"):
    pytest.skip(
        "ops_multimodal_fusion.foreach_ceil not registered for current NPU_ARCH; "
        "skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float32, torch.float16]


# ── Helpers ──────────────────────────────────────────────────────────────


def _reference(tensors):
    """Golden: elementwise ceil on each CPU tensor (public torch.ceil)."""
    return [torch.ceil(t.cpu()) for t in tensors]


def _custom(tensors):
    """Call custom foreach_ceil on NPU."""
    npu_tensors = [t.npu() for t in tensors]
    out = torch.ops.ops_multimodal_fusion.foreach_ceil(npu_tensors)
    return [t.cpu() for t in out]


def _assert_close(actual_list, expected_list, dtype, label, check_signed_zero=False):
    assert len(actual_list) == len(expected_list), (
        f"{label}: length mismatch {len(actual_list)} vs {len(expected_list)}"
    )
    for i, (a, e) in enumerate(zip(actual_list, expected_list)):
        assert a.shape == e.shape, (
            f"{label}[{i}]: shape mismatch {a.shape} vs {e.shape}"
        )
        assert a.dtype == e.dtype, (
            f"{label}[{i}]: dtype mismatch {a.dtype} vs {e.dtype}"
        )
        actual_f32 = a.float()
        expected_f32 = e.float()
        if dtype == torch.float32 or a.dtype == torch.float32:
            # CAST_CEIL -> CAST_NONE is bit-exact for float32
            rtol, atol = 0.0, 0.0
        else:
            # float16 may have cast-back rounding for large values
            rtol, atol = 1e-3, 1e-3
        assert torch.allclose(
            actual_f32, expected_f32, rtol=rtol, atol=atol, equal_nan=True
        ), (
            f"{label}[{i}]: mismatch actual={a} expected={e}"
        )
        if check_signed_zero:
            zero_mask = expected_f32 == 0
            assert torch.equal(
                torch.signbit(actual_f32)[zero_mask],
                torch.signbit(expected_f32)[zero_mask],
            ), f"{label}[{i}]: signed zero mismatch actual={a} expected={e}"


def _run_case(tensors, dtype, label, check_signed_zero=False):
    typed = [t.to(dtype) for t in tensors]
    expected = _reference(typed)
    actual = _custom(typed)
    _assert_close(actual, expected, dtype, label, check_signed_zero)


def _make_tensors(shapes, seed=42):
    """Create list of random float32 tensors with given shapes."""
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(*s, generator=gen, dtype=torch.float32) for s in shapes]


# ── Interface tests ──────────────────────────────────────────────────────


def test_foreach_ceil_interface_exists():
    assert hasattr(torch.ops.ops_multimodal_fusion, "foreach_ceil")


# ── Basic functionality ──────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_ceil_single_tensor(dtype):
    tensors = _make_tensors([(4,)])
    _run_case(tensors, dtype, "single-tensor")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_ceil_two_tensors(dtype):
    tensors = [
        torch.randn(4, dtype=torch.float32),
        torch.randn(2, 3, dtype=torch.float32),
    ]
    _run_case(tensors, dtype, "two-tensors")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_ceil_list_len(dtype):
    tensors = _make_tensors([(1,), (3,), (2, 2), (2, 3, 4)])
    typed = [t.to(dtype) for t in tensors]
    out = torch.ops.ops_multimodal_fusion.foreach_ceil([t.npu() for t in typed])
    actual = [t.cpu() for t in out]
    expected = _reference(typed)
    assert len(actual) == len(tensors)
    _assert_close(actual, expected, dtype, "list-len")


# ── Known values ─────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_ceil_known_values(dtype):
    """Verify exact ceil results for hand-picked values."""
    x = torch.tensor([
        -0.9, -0.6, -0.1, -1.0e-3, -0.0, 0.0,
        0.2, 1.2, -1.0, 3.0, -2.5
    ], dtype=torch.float32)
    tensors = [x]
    typed = [x.to(dtype)]
    actual = _custom(typed)
    expected = [torch.ceil(typed[0])]
    _assert_close(actual, expected, dtype, "known-values",
                  check_signed_zero=True)


# ── Special values ───────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_ceil_special_values_fp32():
    """NaN, Inf, -Inf, -0.0 should match the elementwise ceil reference."""
    x = torch.tensor([float('nan'), float('inf'), float('-inf'), -0.0, 0.0],
                     dtype=torch.float32)
    tensors = [x]
    actual = _custom(tensors)
    expected = _reference(tensors)
    _assert_close(actual, expected, torch.float32, "special-values-fp32",
                  check_signed_zero=True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_ceil_special_values_fp16():
    """NaN, Inf, -Inf, -0.0 should match the elementwise ceil reference for fp16."""
    x = torch.tensor([float('nan'), float('inf'), float('-inf'), -0.0, 0.0],
                     dtype=torch.float16)
    tensors = [x]
    actual = _custom(tensors)
    expected = _reference(tensors)
    _assert_close(actual, expected, torch.float16, "special-values-fp16",
                  check_signed_zero=True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_ceil_neg_zero(dtype):
    """-0.0 results should not become +0.0."""
    x = torch.tensor([-0.0, 0.0, -0.9, -0.1, -1.0e-3, 1.0, -1.0],
                     dtype=torch.float32).to(dtype)
    tensors = [x]
    actual = _custom(tensors)[0]
    expected = torch.ceil(x)
    _assert_close([actual], [expected], dtype, "negative-zero-results",
                  check_signed_zero=True)


# ── Large tensors (multi-tile / multi-block) ─────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_ceil_large_tensor(dtype):
    x = torch.linspace(-100.0, 100.0, steps=65536, dtype=torch.float32)
    tensors = [x]
    _run_case(tensors, dtype, "large-tensor")


# ── Non-contiguous input ─────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_ceil_non_contiguous(dtype):
    base = torch.randn(10, 20, dtype=torch.float32)
    x = base[:, ::2]  # stride > 1
    assert not x.is_contiguous()
    tensors = [x]
    _run_case(tensors, dtype, "non-contiguous")


# ── Empty tensor ─────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_ceil_empty_tensor(dtype):
    x = torch.empty(0, dtype=torch.float32)
    tensors = [x]
    typed = [x.to(dtype)]
    actual = _custom(typed)
    assert len(actual) == 1
    assert actual[0].numel() == 0


# ── Mixed dtypes in list ─────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_ceil_mixed_dtypes():
    tensors = [
        torch.tensor([-0.6, 1.2, 3.0], dtype=torch.float32),
        torch.tensor([2.3, -0.1, -2.5, -0.0], dtype=torch.float16),
    ]
    actual = _custom(tensors)
    expected = _reference(tensors)
    _assert_close(actual, expected, None, "mixed-dtypes",
                  check_signed_zero=True)


# ── Invalid argument tests ───────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_ceil_invalid_args():
    # Empty list
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_ceil([])

    # Non-float dtype
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_ceil([torch.randint(0, 10, (3,)).npu()])

    # Non-NPU tensor
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_ceil([torch.randn(3, 4)])

    # Mixed NPU and CPU
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_ceil([
            torch.randn(3, 4).npu(),
            torch.randn(5,),
        ])


# ── Multi-tensor list semantics ──────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape,dtype", [
    ((3,), torch.float32),
    ((2, 2), torch.float32),
    ((5,), torch.float16),
])


def test_foreach_ceil_shape_preserved(shape, dtype):
    """Output shape/dtype matches input for varied shapes."""
    x = torch.randn(*shape, dtype=torch.float32).to(dtype).npu()
    out = torch.ops.ops_multimodal_fusion.foreach_ceil([x])
    assert len(out) == 1
    assert out[0].shape == shape
    assert out[0].dtype == dtype


def test_foreach_ceil_empty_list_rejected():
    """Empty list should raise RuntimeError."""
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_ceil([])


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_ceil_output_not_alias():
    """Output must be a new tensor, not an alias of the input."""
    x = torch.randn(4, dtype=torch.float32).npu()
    out = torch.ops.ops_multimodal_fusion.foreach_ceil([x])
    assert out[0].data_ptr() != x.data_ptr(), "output aliases input storage"


# ── Large boundary values ───────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_ceil_large_boundary():
    """Values near int32 boundary: ceil should work within [-2^31+1, 2^31-1]."""
    # Within safe range
    x = torch.tensor([2.0**30, -(2.0**30), 2.0**24 + 0.5, -2.0**24 - 0.3],
                     dtype=torch.float32)
    actual = torch.ops.ops_multimodal_fusion.foreach_ceil([x.npu()])[0].cpu()
    expected = torch.ceil(x)
    assert torch.equal(actual, expected), (
        f"large-boundary mismatch: actual={actual} expected={expected}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_ceil_large_magnitude_fp32():
    """fp32 values with |x| >= 2^23 (incl. beyond int32 range) are already
    integers, so ceil(x) == x and must be returned bit-exact.

    This exercises the int32-cast overflow region (|x| >= 2^31) that the
    earlier marker-based restore could not handle.
    """
    x = torch.tensor([
        2.0**23,                 # smallest all-integer fp32 magnitude
        2.0**24,
        2.0**31,                 # exactly int32 max+1 (cast overflow region)
        2.0**31 + 2.0**8,        # just above int32 range
        1.0e10, 1.0e20, 3.0e30, 3.4e38,
        -(2.0**23), -(2.0**24), -(2.0**31),
        -1.0e10, -1.0e20, -3.0e30, -3.4e38,
    ], dtype=torch.float32)
    actual = torch.ops.ops_multimodal_fusion.foreach_ceil([x.npu()])[0].cpu()
    expected = torch.ceil(x)  # identity for all these (already integers)
    assert torch.equal(actual, expected), (
        f"large-magnitude mismatch: actual={actual} expected={expected}"
    )
    # Belt-and-suspenders: every value is already an integer == itself.
    assert torch.equal(actual, x), (
        f"large-magnitude not identity: actual={actual} input={x}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_ceil_large_magnitude_with_inf_nan_fp32():
    """Large finite magnitudes interleaved with Inf/NaN/-0 in one tensor."""
    x = torch.tensor([
        1.0e20, float('inf'), -3.4e38, float('-inf'),
        float('nan'), -0.0, -0.5, 5.0e9,
    ], dtype=torch.float32)
    actual = torch.ops.ops_multimodal_fusion.foreach_ceil([x.npu()])[0].cpu()
    expected = torch.ceil(x)
    # NaN compares unequal, so check separately.
    finite_or_inf = ~torch.isnan(expected)
    assert torch.equal(actual[finite_or_inf], expected[finite_or_inf]), (
        f"mismatch: actual={actual} expected={expected}"
    )
    assert torch.isnan(actual[~finite_or_inf]).all(), (
        f"NaN lane not preserved: actual={actual}"
    )
    # -0.0 sign preserved at index 5.
    assert torch.signbit(actual[5]).item(), f"-0.0 sign lost: {actual}"


# ── Long list (8 tensors, varied shapes) ─────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_ceil_eight_tensors(dtype):
    tensors = _make_tensors(
        [(1,), (3,), (2, 2), (2, 3, 4), (7,), (4, 5), (1, 1, 9), (16,)],
        seed=2026,
    )
    typed = [t.to(dtype) for t in tensors]
    out = torch.ops.ops_multimodal_fusion.foreach_ceil([t.npu() for t in typed])
    actual = [t.cpu() for t in out]
    expected = _reference(typed)
    assert len(actual) == 8
    _assert_close(actual, expected, dtype, "eight-tensors", check_signed_zero=True)
