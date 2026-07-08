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

"""Test suite for foreach_floor - golden reference: torch.floor on each tensor."""

import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "foreach_floor"):
    pytest.skip(
        "ops_multimodal_fusion.foreach_floor not registered for current NPU_ARCH; "
        "skipping module",
        allow_module_level=True,
    )


DTYPES = [torch.float32, torch.float16]


# ── Helpers ──────────────────────────────────────────────────────────────


def _reference(tensors):
    """Golden: torch.floor on each tensor on CPU."""
    cpu_tensors = [t.cpu() for t in tensors]
    return [torch.floor(t) for t in cpu_tensors]


def _custom(tensors):
    """Call custom foreach_floor on NPU."""
    npu_tensors = [t.npu() for t in tensors]
    out = torch.ops.ops_multimodal_fusion.foreach_floor(npu_tensors)
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
            # The floor cast path is bit-exact for single precision.
            rtol, atol = 0.0, 0.0
        else:
            # Half precision may round when casting back for large values.
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


def test_foreach_floor_interface_exists():
    assert hasattr(torch.ops.ops_multimodal_fusion, "foreach_floor")


# ── Basic functionality ──────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_floor_single_tensor(dtype):
    tensors = _make_tensors([(4,)])
    _run_case(tensors, dtype, "single-tensor")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_floor_two_tensors(dtype):
    tensors = [
        torch.randn(4, dtype=torch.float32),
        torch.randn(2, 3, dtype=torch.float32),
    ]
    _run_case(tensors, dtype, "two-tensors")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_floor_list_len(dtype):
    tensors = _make_tensors([(1,), (3,), (2, 2), (2, 3, 4)])
    typed = [t.to(dtype) for t in tensors]
    out = torch.ops.ops_multimodal_fusion.foreach_floor([t.npu() for t in typed])
    actual = [t.cpu() for t in out]
    expected = _reference(typed)
    assert len(actual) == len(tensors)
    _assert_close(actual, expected, dtype, "list-len")


# ── Known values ─────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_floor_known_values(dtype):
    """Verify exact floor results for hand-picked values."""
    x = torch.tensor([
        -0.9, -0.6, -0.1, -1.0e-3, -0.0, 0.0,
        0.2, 1.2, -1.0, 3.0, -2.5
    ], dtype=torch.float32)
    tensors = [x]
    typed = [x.to(dtype)]
    actual = _custom(typed)
    expected = [torch.floor(typed[0])]
    _assert_close(actual, expected, dtype, "known-values",
                  check_signed_zero=True)


# ── Special values ───────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_floor_special_values_fp32():
    """Not-a-number, the infinities and a negative zero should match the
    reference floor.
    """
    x = torch.tensor([float('nan'), float('inf'), float('-inf'), -0.0, 0.0],
                     dtype=torch.float32)
    tensors = [x]
    actual = _custom(tensors)
    expected = _reference(tensors)
    _assert_close(actual, expected, torch.float32, "special-values-fp32",
                  check_signed_zero=True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_floor_special_values_fp16():
    """Not-a-number, the infinities and a negative zero should match the
    reference floor for half precision.
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
def test_foreach_floor_neg_zero(dtype):
    """A negative-zero result should keep its sign and not become a positive zero."""
    x = torch.tensor([-0.0, 0.0, -0.9, -0.1, -1.0e-3, 1.0, -1.0],
                     dtype=torch.float32).to(dtype)
    tensors = [x]
    actual = _custom(tensors)[0]
    expected = torch.floor(x)
    _assert_close([actual], [expected], dtype, "negative-zero-results",
                  check_signed_zero=True)


# ── Large tensors (multi-tile / multi-block) ─────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_floor_large_tensor(dtype):
    x = torch.linspace(-100.0, 100.0, steps=65536, dtype=torch.float32)
    tensors = [x]
    _run_case(tensors, dtype, "large-tensor")


# ── Non-contiguous input ─────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_floor_non_contiguous(dtype):
    base = torch.randn(10, 20, dtype=torch.float32)
    x = base[:, ::2]  # take every other column to break contiguity
    assert not x.is_contiguous()
    tensors = [x]
    _run_case(tensors, dtype, "non-contiguous")


# ── Empty tensor ─────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_floor_empty_tensor(dtype):
    x = torch.empty(0, dtype=torch.float32)
    tensors = [x.to(dtype)]
    actual = _custom(tensors)
    expected = _reference(tensors)
    assert len(actual) == 1
    assert actual[0].numel() == 0
    assert actual[0].shape == expected[0].shape
    assert actual[0].dtype == expected[0].dtype


# ── Mixed dtypes in list ─────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_floor_mixed_dtypes():
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
def test_foreach_floor_invalid_args():
    # Empty list
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_floor([])

    # Non-float dtype
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_floor([torch.randint(0, 10, (3,)).npu()])

    # Non-NPU tensor
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_floor([torch.randn(3, 4)])

    # Mixed NPU and CPU
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_floor([
            torch.randn(3, 4).npu(),
            torch.randn(5,),
        ])

    # Unsupported dtypes
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_floor([
            torch.randn(3, 4, dtype=torch.bfloat16).npu(),
        ])
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_floor([
            torch.randn(3, 4, dtype=torch.float64).npu(),
        ])
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_floor([
            torch.tensor([True, False]).npu(),
        ])


# ── Multi-tensor list semantics ──────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape,dtype", [
    ((3,), torch.float32),
    ((2, 2), torch.float32),
    ((5,), torch.float16),
])


def test_foreach_floor_shape_preserved(shape, dtype):
    """Output shape/dtype matches input and values match golden."""
    x = torch.randn(*shape, dtype=torch.float32).to(dtype)
    _run_case([x], dtype, f"shape-preserved-{shape}")
    out = torch.ops.ops_multimodal_fusion.foreach_floor([x.npu()])
    assert len(out) == 1
    assert out[0].shape == shape
    assert out[0].dtype == dtype


def test_foreach_floor_empty_list_rejected():
    """Empty list should raise RuntimeError."""
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.foreach_floor([])


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_floor_output_not_alias():
    """Output must be a new tensor, not an alias of the input."""
    x = torch.randn(4, dtype=torch.float32).npu()
    out = torch.ops.ops_multimodal_fusion.foreach_floor([x])
    assert out[0].data_ptr() != x.data_ptr(), "output aliases input storage"


# ── Large boundary values ───────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_floor_large_boundary():
    """Values near the signed 32-bit boundary still floor correctly within
    the safe cast range.
    """
    # Within the safe range.
    x = torch.tensor([2.0**30, -(2.0**30), 2.0**24 + 0.5, -2.0**24 - 0.3],
                     dtype=torch.float32)
    actual = torch.ops.ops_multimodal_fusion.foreach_floor([x.npu()])[0].cpu()
    expected = torch.floor(x)
    assert torch.equal(actual, expected), (
        f"large-boundary mismatch: actual={actual} expected={expected}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_floor_large_magnitude_fp32():
    """Large single-precision magnitudes are already integers, so flooring
    leaves them unchanged and must return them bit-exact.

    This exercises the cast overflow region beyond the signed 32-bit range
    that the earlier marker-based restore could not handle.
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
    actual = torch.ops.ops_multimodal_fusion.foreach_floor([x.npu()])[0].cpu()
    expected = torch.floor(x)  # identity for all these already-integer values
    assert torch.equal(actual, expected), (
        f"large-magnitude mismatch: actual={actual} expected={expected}"
    )
    # Belt and suspenders: every value is already an integer equal to itself.
    assert torch.equal(actual, x), (
        f"large-magnitude not identity: actual={actual} input={x}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_floor_large_magnitude_with_inf_nan_fp32():
    """Large finite magnitudes interleaved with infinities, not-a-number and
    negative zero in one tensor.
    """
    x = torch.tensor([
        1.0e20, float('inf'), -3.4e38, float('-inf'),
        float('nan'), -0.0, -0.5, 5.0e9,
    ], dtype=torch.float32)
    actual = torch.ops.ops_multimodal_fusion.foreach_floor([x.npu()])[0].cpu()
    expected = torch.floor(x)
    # A not-a-number value never compares equal, so it is checked separately.
    finite_or_inf = ~torch.isnan(expected)
    assert torch.equal(actual[finite_or_inf], expected[finite_or_inf]), (
        f"mismatch: actual={actual} expected={expected}"
    )
    assert torch.isnan(actual[~finite_or_inf]).all(), (
        f"NaN lane not preserved: actual={actual}"
    )
    # The negative zero lane keeps its sign.
    assert torch.signbit(actual[5]).item(), f"-0.0 sign lost: {actual}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_floor_fp23_boundary_fp32():
    """Values just below the all-integer threshold must floor, while values
    at or above it are returned unchanged through the restore path.
    """
    x = torch.tensor([
        8388607.0,           # one below the threshold, still in the floored region
        8388607.5,           # half below the threshold
        8388608.0,           # at the threshold, returned unchanged
        -8388607.5,
        -8388608.0,
    ], dtype=torch.float32)
    actual = torch.ops.ops_multimodal_fusion.foreach_floor([x.npu()])[0].cpu()
    expected = torch.floor(x)
    assert torch.equal(actual, expected), (
        f"2^23 boundary mismatch: actual={actual} expected={expected}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_foreach_floor_large_magnitude_fp16():
    """fp16 large magnitudes via cast-to-fp32 floor path."""
    x = torch.tensor([
        2048.0, 4096.0, 65504.0, -2048.0, -65504.0,
        1.5, -2.7, 0.0, -0.0,
    ], dtype=torch.float16)
    actual = _custom([x])
    expected = _reference([x])
    _assert_close(actual, expected, torch.float16, "large-magnitude-fp16",
                  check_signed_zero=True)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("numel", [32767, 32768, 32769, 65535])
def test_foreach_floor_tile_boundaries(numel):
    """Deterministic sizes around the maximum tile element count and the
    multi-block tails.
    """
    gen = torch.Generator().manual_seed(32768 + numel)
    x = torch.randn(numel, generator=gen, dtype=torch.float32)
    _run_case([x], torch.float32, f"tile-boundary-{numel}")


# ── Long list 8 tensors varied shapes ─────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_floor_eight_tensors(dtype):
    tensors = _make_tensors(
        [(1,), (3,), (2, 2), (2, 3, 4), (7,), (4, 5), (1, 1, 9), (16,)],
        seed=2026,
    )
    typed = [t.to(dtype) for t in tensors]
    out = torch.ops.ops_multimodal_fusion.foreach_floor([t.npu() for t in typed])
    actual = [t.cpu() for t in out]
    expected = _reference(typed)
    assert len(actual) == 8
    _assert_close(actual, expected, dtype, "eight-tensors", check_signed_zero=True)
