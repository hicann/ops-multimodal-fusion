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

"""Test suite for foreach_frac with the torch foreach frac golden reference.

The foreach frac operator returns, per tensor, the value minus its truncated
part. Not-a-number stays not-a-number, an infinite input becomes
not-a-number, and zero results carry a positive sign because the reference
frac never returns a negative zero whether the input is a negative zero or
an integer.
"""

import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "foreach_frac"):
    pytest.skip(
        "ops_multimodal_fusion.foreach_frac not registered for current NPU_ARCH; "
        "skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float32, torch.float16]


# ── Helpers ──────────────────────────────────────────────────────────────


def _reference(tensors):
    """Golden: x minus trunc(x) on each tensor on CPU."""
    cpu_tensors = [t.cpu() for t in tensors]
    return [t - torch.trunc(t) for t in cpu_tensors]


def _custom(tensors):
    """Call custom foreach_frac on NPU."""
    npu_tensors = [t.npu() for t in tensors]
    out = torch.ops.ops_multimodal_fusion.foreach_frac(npu_tensors)
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
            # Subtracting the exact integer part is bit-exact for single
            # precision and matches the reference frac path.
            rtol, atol = 0.0, 0.0
        else:
            # In half precision, computing in single precision and casting
            # back may differ by one unit in the last place from the native
            # half-precision path.
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


def test_foreach_frac_interface_exists():
    assert hasattr(torch.ops.ops_multimodal_fusion, "foreach_frac")


# ── Basic functionality ──────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_frac_single_tensor(dtype):
    tensors = _make_tensors([(4,)])
    _run_case(tensors, dtype, "single-tensor")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_frac_two_tensors(dtype):
    tensors = [
        torch.randn(4, dtype=torch.float32),
        torch.randn(2, 3, dtype=torch.float32),
    ]
    _run_case(tensors, dtype, "two-tensors")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_frac_list_len(dtype):
    tensors = _make_tensors([(1,), (3,), (2, 2), (2, 3, 4)])
    typed = [t.to(dtype) for t in tensors]
    out = torch.ops.ops_multimodal_fusion.foreach_frac([t.npu() for t in typed])
    actual = [t.cpu() for t in out]
    expected = _reference(typed)
    assert len(actual) == len(tensors)
    _assert_close(actual, expected, dtype, "list-len")


# ── Known values ─────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_frac_known_values(dtype):
    """Verify frac results for hand-picked values (positive and negative)."""
    x = torch.tensor([
        3.7, -3.7, 5.0, -5.0, 0.5, -0.5, 1000.25, -1000.25,
        -0.0, 0.0, 1.0, -1.0,
    ], dtype=torch.float32)
    tensors = [x]
    typed = [x.to(dtype)]
    actual = _custom(typed)
    expected = [torch.frac(typed[0])]
    _assert_close(actual, expected, dtype, "known-values",
                  check_signed_zero=True)


# ── Special values ───────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_frac_special_values_fp32():
    """Infinite and not-a-number inputs give not-a-number while signed zeros
    keep their sign, matching the reference frac.
    """
    x = torch.tensor([float('nan'), float('inf'), float('-inf'), -0.0, 0.0],
                     dtype=torch.float32)
    tensors = [x]
    actual = _custom(tensors)
    expected = _reference(tensors)
    _assert_close(actual, expected, torch.float32, "special-values-fp32",
                  check_signed_zero=True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_frac_special_values_fp16():
    """Infinite and not-a-number inputs give not-a-number while signed zeros
    keep their sign, for half precision.
    """
    x = torch.tensor([float('nan'), float('inf'), float('-inf'), -0.0, 0.0],
                     dtype=torch.float16)
    tensors = [x]
    actual = _custom(tensors)
    expected = _reference(tensors)
    _assert_close(actual, expected, torch.float16, "special-values-fp16",
                  check_signed_zero=True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_frac_neg_zero(dtype):
    """A negative-zero input yields a positive zero per the reference frac;
    verify the sign of zero matches the reference.
    """
    x = torch.tensor([-0.0, 0.0, -0.9, -0.1, -1.0e-3, 1.0, -1.0],
                     dtype=torch.float32).to(dtype)
    tensors = [x]
    actual = _custom(tensors)[0]
    expected = torch.frac(x)
    _assert_close([actual], [expected], dtype, "negative-zero-results",
                  check_signed_zero=True)


# ── Large tensors (multi-tile / multi-block) ─────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_frac_large_tensor(dtype):
    x = torch.linspace(-100.0, 100.0, steps=65536, dtype=torch.float32)
    tensors = [x]
    _run_case(tensors, dtype, "large-tensor")


# ── Non-contiguous input ─────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_frac_non_contiguous(dtype):
    base = torch.randn(10, 20, dtype=torch.float32)
    x = base[:, ::2]  # take every other column to break contiguity
    assert not x.is_contiguous()
    tensors = [x]
    _run_case(tensors, dtype, "non-contiguous")


# ── Empty tensor ─────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_frac_empty_tensor(dtype):
    x = torch.empty(0, dtype=torch.float32)
    tensors = [x]
    typed = [x.to(dtype)]
    actual = _custom(typed)
    assert len(actual) == 1
    assert actual[0].numel() == 0


# ── Mixed dtypes in list ─────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_frac_mixed_dtypes():
    tensors = [
        torch.tensor([3.7, -2.5, 5.0], dtype=torch.float32),
        torch.tensor([2.3, -0.1, -2.5, -0.0], dtype=torch.float16),
    ]
    actual = _custom(tensors)
    expected = _reference(tensors)
    _assert_close(actual, expected, None, "mixed-dtypes",
                  check_signed_zero=True)


# ── Invalid argument tests ───────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_frac_invalid_args():
    # Empty list
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_frac([])

    # Non-float dtype
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_frac([torch.randint(0, 10, (3,)).npu()])

    # Non-NPU tensor
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_frac([torch.randn(3, 4)])

    # Mixed NPU and CPU
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_frac([
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


def test_foreach_frac_shape_preserved(shape, dtype):
    """Output shape/dtype matches input for varied shapes."""
    x = torch.randn(*shape, dtype=torch.float32).to(dtype).npu()
    out = torch.ops.ops_multimodal_fusion.foreach_frac([x])
    assert len(out) == 1
    assert out[0].shape == shape
    assert out[0].dtype == dtype


def test_foreach_frac_empty_list_rejected():
    """Empty list should raise RuntimeError."""
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_frac([])


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_frac_output_not_alias():
    """Output must be a new tensor, not an alias of the input."""
    x = torch.randn(4, dtype=torch.float32).npu()
    out = torch.ops.ops_multimodal_fusion.foreach_frac([x])
    assert out[0].data_ptr() != x.data_ptr(), "output aliases input storage"


# ── Large boundary values ────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_frac_large_boundary():
    """Values near the signed 32-bit boundary stay correct within the cast
    range.

    Below the all-integer threshold the fractional part is meaningful; at or
    above it every finite single-precision value is already an integer so the
    fraction is zero.
    """
    x = torch.tensor([
        2.0**30, -(2.0**30),        # integers within the cast range, fraction is zero
        1000.5, -1000.5,            # values with a meaningful fractional part
        2.0**23,                    # smallest all-integer magnitude, fraction is zero
    ], dtype=torch.float32)
    actual = torch.ops.ops_multimodal_fusion.foreach_frac([x.npu()])[0].cpu()
    expected = torch.frac(x)
    assert torch.allclose(actual, expected, equal_nan=True), (
        f"large-boundary mismatch: actual={actual} expected={expected}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_frac_large_magnitude_fp32():
    """Large single-precision magnitudes are already integers, so their
    fraction is zero.

    This exercises the cast overflow region beyond the signed 32-bit range
    that the big-magnitude self-subtract path handles.
    """
    x = torch.tensor([
        2.0**23,                 # smallest all-integer single-precision magnitude
        2.0**24,
        2.0**31,                 # just past the signed 32-bit cast overflow point
        2.0**31 + 2.0**8,        # just above the signed 32-bit range
        1.0e10, 1.0e20, 3.0e30, 3.4e38,
        -(2.0**23), -(2.0**24), -(2.0**31),
        -1.0e10, -1.0e20, -3.0e30, -3.4e38,
    ], dtype=torch.float32)
    actual = torch.ops.ops_multimodal_fusion.foreach_frac([x.npu()])[0].cpu()
    expected = torch.frac(x)  # zero for all these already-integer values
    assert torch.equal(actual, expected), (
        f"large-magnitude mismatch: actual={actual} expected={expected}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_frac_large_magnitude_with_inf_nan_fp32():
    """Large finite magnitudes interleaved with infinities, not-a-number and
    negative zero in one tensor.

    A large finite value gives zero, an infinity gives not-a-number,
    not-a-number stays not-a-number, and a negative zero gives a positive
    zero because the reference frac never returns a negative zero. The
    comparison is by value so the sign of zero on the negative-zero lane is
    not a hard failure here; the dedicated special-value and negative-zero
    tests cover the sign of zero against the reference.
    """
    x = torch.tensor([
        1.0e20, float('inf'), -3.4e38, float('-inf'),
        float('nan'), -0.0, -0.5, 5.0e9,
    ], dtype=torch.float32)
    actual = torch.ops.ops_multimodal_fusion.foreach_frac([x.npu()])[0].cpu()
    expected = torch.frac(x)
    # A not-a-number value never compares equal, so the finite lanes are
    # checked by value and the not-a-number lane is checked separately.
    is_nan_lane = torch.isnan(expected)
    assert torch.allclose(
        actual[~is_nan_lane], expected[~is_nan_lane], equal_nan=True
    ), f"mismatch: actual={actual} expected={expected}"
    assert torch.isnan(actual[is_nan_lane]).all(), (
        f"NaN lane not produced: actual={actual}"
    )


# ── Long list 8 tensors varied shapes ─────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_frac_eight_tensors(dtype):
    tensors = _make_tensors(
        [(1,), (3,), (2, 2), (2, 3, 4), (7,), (4, 5), (1, 1, 9), (16,)],
        seed=2026,
    )
    typed = [t.to(dtype) for t in tensors]
    out = torch.ops.ops_multimodal_fusion.foreach_frac([t.npu() for t in typed])
    actual = [t.cpu() for t in out]
    expected = _reference(typed)
    assert len(actual) == 8
    _assert_close(actual, expected, dtype, "eight-tensors", check_signed_zero=True)
