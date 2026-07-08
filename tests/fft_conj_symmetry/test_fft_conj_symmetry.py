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
"""Tests for ops_multimodal_fusion.fft_conj_symmetry (onesided to twosided)."""

import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401


if not hasattr(torch.ops.ops_multimodal_fusion, "fft_conj_symmetry"):
    pytest.skip(
        "ops_multimodal_fusion.fft_conj_symmetry not registered for current NPU_ARCH; "
        "skipping module",
        allow_module_level=True,
    )


# complex32 (ComplexHalf) is experimental in PyTorch and unsupported on the Ascend
# device (host-to-device copy of a complex32 tensor faults), so only complex64 is exercised.
DTYPES = [torch.complex64]


def _reference(x_cpu, dim, out_size):
    """CPU reference: apply conjugate symmetry along `dim`.

    The half size is out_size floor-divided by two plus one. For each index
    k along the dim, indices below the half size copy the input directly,
    while indices at or above the half size take the conjugate of the input
    element at out_size minus k.
    """
    half_size = out_size // 2 + 1
    assert x_cpu.size(dim) == half_size, (
        f"Input size along dim {dim} must be {half_size}, got {x_cpu.size(dim)}"
    )

    out_shape = list(x_cpu.shape)
    out_shape[dim] = out_size
    result = torch.zeros(out_shape, dtype=x_cpu.dtype, device="cpu")

    # Build slices for the direct region
    direct_slices = [slice(None)] * x_cpu.ndim
    direct_slices[dim] = slice(0, half_size)
    result[tuple(direct_slices)] = x_cpu[tuple(direct_slices)]

    # Fill conjugate mirror region
    for k in range(half_size, out_size):
        src_k = out_size - k
        if src_k < half_size:
            src_slices = [slice(None)] * x_cpu.ndim
            src_slices[dim] = slice(src_k, src_k + 1)
            dst_slices = [slice(None)] * x_cpu.ndim
            dst_slices[dim] = slice(k, k + 1)
            result[tuple(dst_slices)] = torch.conj(x_cpu[tuple(src_slices)])
        # src_k >= half_size cannot happen by construction since k >= half_size
        # implies out_size - k <= half_size

    return result


def _custom(x_cpu, dim, out_size):
    """Call the custom fft_conj_symmetry operator on NPU."""
    return torch.ops.ops_multimodal_fusion.fft_conj_symmetry(
        x_cpu.npu(), dim, out_size,
    ).cpu()


def _assert_close(actual, expected, dtype, label):
    assert actual.shape == expected.shape, (
        f"{label}: shape mismatch {actual.shape} vs {expected.shape}"
    )
    assert actual.dtype == expected.dtype, (
        f"{label}: dtype mismatch {actual.dtype} vs {expected.dtype}"
    )
    assert torch.isfinite(actual.real.float()).all(), (
        f"{label}: actual real part contains NaN/Inf"
    )
    assert torch.isfinite(actual.imag.float()).all(), (
        f"{label}: actual imag part contains NaN/Inf"
    )
    # FFTConjSymmetry does only sign-flip (negate imaginary part) and
    # exact DataCopyPad.  Sign flip is bit-exact in IEEE 754; the direct
    # half is a bitwise copy.  Casting between half and float is exact
    # because IEEE half is a subset of IEEE float and sign-flipped values
    # are still exactly representable in half.  Therefore both dtypes
    # produce bit-exact results.
    # The torch equal and allclose-with-zero-tolerance helpers do not
    # support ComplexHalf natively.  Compare real and imag lanes separately;
    # taking the float of a complex tensor would discard the imaginary part.
    if dtype == torch.complex32:
        actual_real = actual.real.float()
        actual_imag = actual.imag.float()
        expected_real = expected.real.float()
        expected_imag = expected.imag.float()
        real_diff = (actual_real - expected_real).abs()
        imag_diff = (actual_imag - expected_imag).abs()
        assert actual_real.equal(expected_real) and actual_imag.equal(expected_imag), (
            f"{label}: bit-exact mismatch (complex32 real/imag)\n"
            f"max_real_diff={float(real_diff.max().item()):.6e} "
            f"max_imag_diff={float(imag_diff.max().item()):.6e}"
        )
    else:
        assert torch.equal(actual, expected), (
            f"{label}: bit-exact mismatch\n"
            f"max_diff={float((actual - expected).abs().max().item()):.6e}"
        )


def _assert_real_imag_exact(actual, expected, label, *, equal_nan=False):
    torch.testing.assert_close(
        actual.real.float(), expected.real.float(),
        rtol=0, atol=0, equal_nan=equal_nan,
        msg=f"{label}: real part mismatch",
    )
    torch.testing.assert_close(
        actual.imag.float(), expected.imag.float(),
        rtol=0, atol=0, equal_nan=equal_nan,
        msg=f"{label}: imag part mismatch",
    )


def _run_case(x_cpu, dim, out_size, dtype, label):
    """Run a single test case."""
    x_typed = x_cpu.to(dtype)
    expected = _reference(x_typed, dim, out_size)
    actual = _custom(x_typed, dim, out_size)
    _assert_close(actual, expected, dtype, label)


def _make_complex_input(shape, seed=42):
    """Create a complex tensor with non-trivial real and imaginary parts."""
    gen = torch.Generator().manual_seed(seed)
    real = torch.randn(*shape, generator=gen, dtype=torch.float32)
    imag = torch.randn(*shape, generator=gen, dtype=torch.float32)
    return torch.complex(real, imag)


# ── Interface tests ──────────────────────────────────────────────────────


def test_fft_conj_symmetry_interface_exists():
    assert hasattr(torch.ops.ops_multimodal_fusion, "fft_conj_symmetry")


# ── Basic functionality ──────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("out_size", [4, 8, 16, 32])
def test_fft_conj_symmetry_basic_1d(dtype, out_size):
    half_size = out_size // 2 + 1
    x = _make_complex_input((half_size,), seed=out_size)
    _run_case(x, 0, out_size, dtype, f"basic-1d out_size={out_size}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fft_conj_symmetry_basic_2d_dim0(dtype):
    out_size = 10
    half_size = out_size // 2 + 1  # 6
    x = _make_complex_input((half_size, 4), seed=123)
    _run_case(x, 0, out_size, dtype, "basic-2d-dim0")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fft_conj_symmetry_basic_2d_dim1(dtype):
    out_size = 10
    half_size = out_size // 2 + 1  # 6
    x = _make_complex_input((3, half_size), seed=456)
    _run_case(x, 1, out_size, dtype, "basic-2d-dim1")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fft_conj_symmetry_basic_3d(dtype):
    out_size = 8
    half_size = out_size // 2 + 1  # 5
    x = _make_complex_input((2, half_size, 3), seed=789)
    _run_case(x, 1, out_size, dtype, "basic-3d")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fft_conj_symmetry_odd_out_size(dtype):
    out_size = 9
    half_size = out_size // 2 + 1  # 5
    x = _make_complex_input((2, half_size, 3), seed=909)
    _run_case(x, 1, out_size, dtype, "odd-out-size")


# ── Hermitian property tests ─────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fft_conj_symmetry_hermitian_property(dtype):
    """Verify output index k equals the conjugate of output index N minus k.

    DC and Nyquist are skipped: they must be real-only for the input to
    produce a valid Hermitian full spectrum.
    """
    out_size = 8  # half size is five
    half_size = out_size // 2 + 1
    # Build a Hermitian-compatible half spectrum where the DC and Nyquist bins are real
    x = _make_complex_input((half_size,), seed=101)
    x[0] = x[0].real + 0j       # DC bin made purely real
    x[-1] = x[-1].real + 0j     # Nyquist bin made purely real
    actual = _custom(x.to(dtype), 0, out_size)

    actual_real = actual.real.float()
    actual_imag = actual.imag.float()
    for k in range(1, out_size):
        mirror = out_size - k
        assert torch.equal(actual_real[k], actual_real[mirror]), (
            f"Hermitian real part violated at k={k}: "
            f"{actual_real[k]} vs {actual_real[mirror]}"
        )
        assert torch.equal(actual_imag[k], -actual_imag[mirror]), (
            f"Hermitian imag part violated at k={k}: "
            f"{actual_imag[k]} vs {-actual_imag[mirror]}"
        )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fft_conj_symmetry_dc_nyquist_real():
    """Verify the direct half is copied verbatim.

    The DC and Nyquist components should be purely real for a real-signal
    FFT, but this operator just copies or conjugates the input.
    """
    out_size = 8
    half_size = out_size // 2 + 1  # 5
    x = _make_complex_input((half_size,), seed=202)
    actual_fp32 = _custom(x.to(torch.complex64), 0, out_size)

    # First half_size elements should match input
    assert torch.allclose(
        actual_fp32[:half_size], x.to(torch.complex64), rtol=1e-5, atol=1e-5
    ), "Direct region should match input verbatim"


# ── Multi-dim with large inner sizes ─────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fft_conj_symmetry_large_inner(dtype):
    """Test with inner dimension large enough to span multiple tiles."""
    out_size = 16
    half_size = out_size // 2 + 1  # 9
    x = _make_complex_input((3, half_size, 128), seed=303)
    _run_case(x, 1, out_size, dtype, "large-inner")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fft_conj_symmetry_multi_tile_inner(dtype):
    """Inner dimension exceeds half of kMaxTileElems complex values."""
    out_size = 10
    half_size = out_size // 2 + 1
    x = _make_complex_input((2, half_size, 1100), seed=313)
    _run_case(x, 1, out_size, dtype, "multi-tile-inner")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fft_conj_symmetry_large_outer(dtype):
    """Test with many outer slices."""
    out_size = 8
    half_size = out_size // 2 + 1  # 5
    x = _make_complex_input((64, half_size, 3), seed=404)
    _run_case(x, 1, out_size, dtype, "large-outer")


# ── Minimum sizes ────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fft_conj_symmetry_min_out_size_2(dtype):
    """For an out_size of two the half size is two and there is no mirror region."""
    x = _make_complex_input((2,), seed=505)
    _run_case(x, 0, 2, dtype, "min-out-size-2")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fft_conj_symmetry_out_size_3(dtype):
    """For an out_size of three the half size is two with one mirror element at index two."""
    x = _make_complex_input((2,), seed=606)
    _run_case(x, 0, 3, dtype, "out-size-3")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fft_conj_symmetry_empty_outer(dtype):
    out_size = 8
    half_size = out_size // 2 + 1
    x = torch.empty((0, half_size, 3), dtype=torch.complex64)
    _run_case(x, 1, out_size, dtype, "empty-outer")


# ── Non-contiguous input ─────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fft_conj_symmetry_non_contiguous_input():
    """Non-contiguous input should still produce correct result."""
    out_size = 8
    half_size = out_size // 2 + 1  # 5
    base = _make_complex_input((half_size, 10), seed=707)
    x = base[:, ::2]  # stride 2 in last dim makes it non-contiguous
    assert not x.is_contiguous()
    _run_case(x, 0, out_size, torch.complex64, "non-contiguous")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fft_conj_symmetry_non_contiguous_complex32_input():
    out_size = 8
    half_size = out_size // 2 + 1
    base = _make_complex_input((4, half_size, 8), seed=717)
    x = base[::2, :, ::2]
    assert not x.is_contiguous()
    _run_case(x, 1, out_size, torch.complex32, "non-contiguous-complex32")


# ── Edge case: single-element inner dim ───────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fft_conj_symmetry_single_inner(dtype):
    out_size = 8
    half_size = out_size // 2 + 1  # 5
    x = _make_complex_input((half_size, 1), seed=808)
    _run_case(x, 0, out_size, dtype, "single-inner")


# ── Invalid argument tests ───────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fft_conj_symmetry_invalid_args():
    x = _make_complex_input((3, 5))  # complex64 input

    # Non-complex input
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.fft_conj_symmetry(
            torch.randn(3, 5).npu(), 0, 8
        )

    # Invalid dim
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.fft_conj_symmetry(x.npu(), 2, 8)

    # out_size too small
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.fft_conj_symmetry(x.npu(), 0, 1)

    # Mismatched half size: the input provides three but six is required
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.fft_conj_symmetry(
            _make_complex_input((3,)).npu(), 0, 10
        )

    # Wrong dtype
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.fft_conj_symmetry(
            torch.randn(5, dtype=torch.float32).npu(), 0, 8
        )

    # Unsupported complex dtype
    with pytest.raises(RuntimeError):
        torch.ops.ops_multimodal_fusion.fft_conj_symmetry(
            torch.ones(5, dtype=torch.complex128).npu(), 0, 8
        )


# ── Known value test ─────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fft_conj_symmetry_known_values():
    """Verify with known complex values."""
    # the half size is four when out_size is six
    # the input row holds the four leading elements
    # the output appends the conjugate of the third and second elements after the direct half
    a0 = complex(1.0, 2.0)
    a1 = complex(-3.0, 4.0)
    a2 = complex(5.0, -6.0)
    a3 = complex(-7.0, -8.0)
    x = torch.tensor([a0, a1, a2, a3], dtype=torch.complex64)

    actual = _custom(x, 0, 6)

    expected = torch.tensor([
        a0, a1, a2, a3,
        a2.conjugate(),  # mirror at index four pulls from input index two
        a1.conjugate(),  # mirror at index five pulls from input index one
    ], dtype=torch.complex64)

    _assert_close(actual, expected, torch.complex64, "known-values")


# ── Real-valued input ────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fft_conj_symmetry_real_valued(dtype):
    """Real-valued complex input should produce real-valued output."""
    out_size = 8
    half_size = out_size // 2 + 1  # 5
    real_part = torch.linspace(-0.5, 0.5, steps=half_size, dtype=torch.float32)
    x = torch.complex(real_part, torch.zeros_like(real_part))
    actual = _custom(x.to(dtype), 0, out_size)
    expected = _reference(x.to(dtype), 0, out_size)
    _assert_close(actual, expected, dtype, "real-valued")


# ── Pure imaginary input ─────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fft_conj_symmetry_pure_imaginary(dtype):
    """Pure imaginary input has its conjugate equal to its negation."""
    out_size = 8
    half_size = out_size // 2 + 1  # 5
    imag_part = torch.linspace(-0.5, 0.5, steps=half_size, dtype=torch.float32)
    x = torch.complex(torch.zeros_like(imag_part), imag_part)
    _run_case(x, 0, out_size, dtype, "pure-imaginary")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
def test_fft_conj_symmetry_special_values(dtype):
    out_size = 6
    real = torch.tensor([0.0, float("inf"), float("nan"), -2.0], dtype=torch.float32)
    imag = torch.tensor([-0.0, 1.0, float("-inf"), float("nan")], dtype=torch.float32)
    x = torch.complex(real, imag).to(dtype)
    expected = _reference(x, 0, out_size)
    actual = _custom(x, 0, out_size)
    _assert_real_imag_exact(actual, expected, "special-values", equal_nan=True)
    assert torch.equal(
        torch.signbit(actual.real.float()),
        torch.signbit(expected.real.float()),
    )
    assert torch.equal(
        torch.signbit(actual.imag.float()),
        torch.signbit(expected.imag.float()),
    )


# ── Zero input ───────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,dim", [
    ((5,), 0),
    ((3, 5), 0), ((3, 5), 1),
    ((2, 5, 3), 0), ((2, 5, 3), 1), ((2, 5, 3), 2),
])


def test_fft_conj_symmetry_zero_input(dtype, shape, dim):
    """All-zero input should produce all-zero output."""
    out_size = shape[dim] * 2
    half_size = out_size // 2 + 1
    in_shape = list(shape)
    in_shape[dim] = half_size
    x = torch.zeros(in_shape, dtype=dtype)
    out_shape = list(shape)
    out_shape[dim] = out_size
    expected = torch.zeros(out_shape, dtype=dtype)
    actual = torch.ops.ops_multimodal_fusion.fft_conj_symmetry(x.npu(), dim, out_size).cpu()
    _assert_close(actual, expected, dtype, f"zero-input shape={shape} dim={dim}")


# ── complex32 known values (exact bit-pattern round-trip) ───────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fft_conj_symmetry_known_values_complex32():
    """Verify complex32 conjugate symmetry with exact half-precision values.

    Half can represent ±0, ±1, ±2, ±0.5, ±0.25 ... exactly.  This test
    uses hand-picked values that survive the half→float→half round-trip
    with zero error so the bit-exact tolerance is meaningful.
    """
    # the half size is four when out_size is six
    # indices zero through three copy directly; the mirror at index five is the
    # conjugate of input index one
    a0 = complex(1.0, -2.0)      # exactly representable as half
    a1 = complex(-3.0, 4.0)      # exactly representable as half
    a2 = complex(0.5, -0.25)     # exactly representable as half
    a3 = complex(0.0, 0.0)       # zero
    x = torch.tensor([a0, a1, a2, a3], dtype=torch.complex32)

    expected = torch.tensor([
        a0, a1, a2, a3,                       # direct region
        a2.conjugate(),                        # mirror at index four, conjugate of input index two
        a1.conjugate(),                        # mirror at index five, conjugate of input index one
    ], dtype=torch.complex32)

    actual = torch.ops.ops_multimodal_fusion.fft_conj_symmetry(x.npu(), 0, 6).cpu()

    assert actual.dtype == torch.complex32
    assert actual.shape == expected.shape
    _assert_close(actual, expected, torch.complex32, "complex32 known-values")


# ── Stress test: large multi-dimensional ─────────────────────────────────


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_fft_conj_symmetry_large_multi_dim():
    out_size = 32
    half_size = out_size // 2 + 1  # 17
    x = _make_complex_input((4, half_size, 8, 8), seed=999)
    _run_case(x, 1, out_size, torch.complex64, "large-multi-dim")
