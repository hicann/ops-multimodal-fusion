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


import torch
import torch_npu  # noqa: F401
import pytest
import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "nextafter"):
    pytest.skip(
        "ops_multimodal_fusion.nextafter not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_nextafter_interface_exist():
    """The 'ops_multimodal_fusion.nextafter' operator must be registered in torch.ops."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "nextafter"),\
        "The 'nextafter' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


SHAPES = [
    (1,),
    (7,),
    (1024,),
    (10000,),
    (10, 10),
    (32, 32),
    (100, 100),
    (10, 100),
    (256, 512),
    (16, 32, 64),
    (1, 3, 32, 32),
    (4, 3, 64, 64),
    # Large shapes that exceed UB capacity, forcing multi-tile processing.
    (100000,),
    (1000000,),
    (2048, 2048),
    (64, 128, 256),
]

# torch.nextafter only supports float32 / float64 natively; we ship fp32 only.
DTYPES = [torch.float32]


def _sample(shape, dtype, seed=0, low=-100.0, high=100.0):
    g = torch.Generator().manual_seed(seed)
    return torch.empty(*shape, dtype=dtype).uniform_(low, high, generator=g)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_nextafter_operator(shape, dtype):
    """nextafter is a discrete bit-level op; must match torch.nextafter exactly."""
    seed = abs(hash((shape, dtype))) % 10_000_000
    a = _sample(shape, dtype, seed=seed)
    b = _sample(shape, dtype, seed=seed + 1)
    # Avoid NaN inputs (out of scope) and ensure no zeros sneak in to keep
    # a clean general-case test; signed-zero is exercised separately.
    a = torch.where(a == 0, torch.tensor(1e-30, dtype=dtype), a)
    b = torch.where(b == 0, torch.tensor(1e-30, dtype=dtype), b)

    a_npu = a.npu()
    b_npu = b.npu()
    c_npu = torch.ops.ops_multimodal_fusion.nextafter(a_npu, b_npu)
    c = c_npu.cpu()

    expected = torch.nextafter(a, b)

    assert c.dtype == dtype, f"dtype mismatch: {c.dtype} vs {dtype}"
    assert c.shape == a.shape, f"shape mismatch: {c.shape} vs {a.shape}"
    assert torch.equal(c, expected), (
        f"nextafter mismatch (dtype={dtype}, shape={shape}): "
        f"first 5 mismatches at "
        f"{(c != expected).nonzero(as_tuple=False)[:5].tolist()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nextafter_named_values():
    """A few closed-form cases verifiable by inspection."""
    # ulp 1.0f 2^-23 1.1920929e-07
    # ulp 2.0f 2^-22
    # nextafter 1.0 2.0 1.0 + 2^-23
    # nextafter 1.0 0.0 1.0 - 2^-24 gap to next-smaller is half of ulp at 1
    a = torch.tensor([1.0, 1.0, -1.0, -1.0, 2.0, -2.0], dtype=torch.float32)
    b = torch.tensor([2.0, 0.0, -2.0, 0.0, -1.0, 1.0], dtype=torch.float32)
    expected = torch.nextafter(a, b)

    c = torch.ops.ops_multimodal_fusion.nextafter(a.npu(), b.npu()).cpu()
    assert torch.equal(c, expected), (
        f"named-values mismatch; got {c.tolist()} expected {expected.tolist()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nextafter_equality_returns_b():
    """When a b IEEE specifies result b preserves sign of b for ±0 ."""
    a = torch.tensor([1.0, -3.5, 0.0, -0.0], dtype=torch.float32)
    b = torch.tensor([1.0, -3.5, 0.0, -0.0], dtype=torch.float32)
    expected = torch.nextafter(a, b)

    c = torch.ops.ops_multimodal_fusion.nextafter(a.npu(), b.npu()).cpu()
    assert torch.equal(c, expected), (
        f"equality mismatch; got {c.tolist()} expected {expected.tolist()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nextafter_signed_zero_to_nonzero():
    """nextafter ±0 ±x must produce ±denorm_min in direction of x."""
    a = torch.tensor([0.0, 0.0, -0.0, -0.0], dtype=torch.float32)
    b = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float32)
    expected = torch.nextafter(a, b)
    # Reference values: ±denorm_min ±1.401298e-45 bit pattern 0x00000001 / 0x80000001

    c = torch.ops.ops_multimodal_fusion.nextafter(a.npu(), b.npu()).cpu()
    assert torch.equal(c, expected), (
        f"zero-to-nonzero mismatch; got {c.tolist()} expected {expected.tolist()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nextafter_at_powers_of_two():
    """At 2^k, ulp transitions; the bit-level kernel must still be exact."""
    a = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, -1.0, -2.0, -4.0],
                     dtype=torch.float32)
    # Step toward zero at each, then step away from zero at each.
    b_toward = torch.zeros_like(a)
    b_away = torch.full_like(a, float("inf"))
    b_away = torch.copysign(b_away, a)  # +inf for positive a, -inf for negative a

    for direction_name, b in [("toward", b_toward), ("away", b_away)]:
        expected = torch.nextafter(a, b)
        c = torch.ops.ops_multimodal_fusion.nextafter(a.npu(), b.npu()).cpu()
        assert torch.equal(c, expected), (
            f"power-of-two ({direction_name}) mismatch; "
            f"got {c.tolist()} expected {expected.tolist()}"
        )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nextafter_finite_to_infinity():
    """nextafter ±MAX ±inf returns ±inf."""
    finfo = torch.finfo(torch.float32)
    a = torch.tensor([finfo.max, -finfo.max], dtype=torch.float32)
    b = torch.tensor([float("inf"), -float("inf")], dtype=torch.float32)
    expected = torch.nextafter(a, b)

    c = torch.ops.ops_multimodal_fusion.nextafter(a.npu(), b.npu()).cpu()
    assert torch.equal(c, expected), (
        f"finite->inf mismatch; got {c.tolist()} expected {expected.tolist()}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nextafter_idempotent_on_self():
    """nextafter a a a b in our IEEE-aligned implementation ."""
    a = _sample((4096,), torch.float32, seed=7, low=-50.0, high=50.0)
    c = torch.ops.ops_multimodal_fusion.nextafter(a.npu(), a.npu()).cpu()
    assert torch.equal(c, a)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nextafter_one_step_only():
    """|c - a| should be exactly one ulp(a) - no double-stepping or no-op."""
    a = _sample((4096,), torch.float32, seed=11, low=0.1, high=100.0)
    # b strictly above a so we always step "up".
    b = a + 1.0
    c = torch.ops.ops_multimodal_fusion.nextafter(a.npu(), b.npu()).cpu()
    expected = torch.nextafter(a, b)

    assert torch.equal(c, expected)
    # And c > a strictly, by exactly one representable step.
    assert (c > a).all(), "c must be strictly greater than a when stepping up"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nextafter_empty_tensor():
    """Empty tensor passes through with matching shape/dtype (no kernel launch)."""
    a = torch.empty((0,), dtype=torch.float32).npu()
    b = torch.empty((0,), dtype=torch.float32).npu()
    c = torch.ops.ops_multimodal_fusion.nextafter(a, b).cpu()
    assert c.shape == (0,)
    assert c.dtype == torch.float32


@pytest.mark.skip(
    reason=(
        "torch_npu in the current CANN release lacks D2D strided copy support "
        "(aclnnInplaceCopy fails with error 561103 for any non-contiguous NPU "
        "tensor, regardless of dtype). The kernel itself only operates on "
        "contiguous buffers; callers holding a transposed/strided tensor must "
        "materialize it themselves before invoking the op. "
        "Re-enable once torch_npu ships strided D2D."
    )
)


def test_nextafter_non_contiguous_input():
    """Non-contiguous tensors are handled (kernel makes a contiguous copy)."""
    base_a = _sample((32, 32), torch.float32, seed=19)
    base_b = _sample((32, 32), torch.float32, seed=20)
    a = base_a.t()
    b = base_b.t()
    assert not a.is_contiguous() and not b.is_contiguous()

    c = torch.ops.ops_multimodal_fusion.nextafter(a.npu(), b.npu()).cpu()
    expected = torch.nextafter(a.contiguous(), b.contiguous())
    assert torch.equal(c, expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nextafter_rejects_unsupported_dtype():
    """float64 / float16 are not in the supported-dtype list per issue #95."""
    for unsupported in [torch.float64, torch.float16]:
        a = torch.tensor([1.0, -1.0], dtype=unsupported).npu()
        b = torch.tensor([1.0, 1.0], dtype=unsupported).npu()
        with pytest.raises(RuntimeError, match="nextafter"):
            torch.ops.ops_multimodal_fusion.nextafter(a, b)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nextafter_rejects_shape_mismatch():
    """Mismatched input shapes must raise."""
    a = torch.zeros((4,), dtype=torch.float32).npu()
    b = torch.zeros((5,), dtype=torch.float32).npu()
    with pytest.raises(RuntimeError, match="nextafter"):
        torch.ops.ops_multimodal_fusion.nextafter(a, b)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_nextafter_rejects_dtype_mismatch():
    """Mismatched input dtypes must raise."""
    a = torch.zeros((4,), dtype=torch.float32).npu()
    b = torch.zeros((4,), dtype=torch.float64).npu()
    with pytest.raises(RuntimeError, match="nextafter"):
        torch.ops.ops_multimodal_fusion.nextafter(a, b)
