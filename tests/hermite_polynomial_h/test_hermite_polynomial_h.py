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

import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401
if not hasattr(torch.ops.ops_multimodal_fusion, "hermite_polynomial_h"):
    pytest.skip(
        "ops_multimodal_fusion.hermite_polynomial_h not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def reference_manual(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    # Keep manual model consistent with operator: n is first cast to int32.
    n_int = n.to(torch.int32)
    n = n_int.to(torch.float32)
    out = torch.empty_like(x, dtype=torch.float32)

    nan_mask = torch.isnan(x)
    neg_mask = n_int < 0
    zero_mask = n_int == 0
    one_mask = n_int == 1
    limit_mask = n_int > 128
    recur_mask = ~(nan_mask | neg_mask | zero_mask | one_mask | limit_mask)

    out[nan_mask] = torch.nan
    out[neg_mask] = 0.0
    out[zero_mask] = 1.0
    out[one_mask] = x[one_mask] + x[one_mask]
    out[limit_mask] = torch.nan

    if recur_mask.any():
        x_sel = x[recur_mask]
        n_sel = n_int[recur_mask]
        p = torch.ones_like(x_sel)
        q = x_sel + x_sel
        r = torch.zeros_like(x_sel)
        two_x = x_sel + x_sel
        for k in range(2, 257, 2):
            active = (n_sel + n_sel) > k
            if active.any():
                r_new = two_x[active] * q[active] - float(k) * p[active]
                r[active] = r_new
                p[active] = q[active]
                q[active] = r_new
        out[recur_mask] = r

    return out


def reference(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    # Match operator front-end semantics: n is normalized as int32 before kernel compute.
    n_norm = n.to(torch.int32).to(torch.float32)
    return torch.special.hermite_polynomial_h(x, n_norm)


def run_op(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return torch.ops.ops_multimodal_fusion.hermite_polynomial_h(x.npu(), n.npu()).cpu()


def assert_close(actual: torch.Tensor, expected: torch.Tensor, *, rtol=2e-4, atol=2e-4):
    actual = actual.float()
    expected = expected.float()
    finite_mask = torch.isfinite(actual) & torch.isfinite(expected)
    if finite_mask.any():
        max_diff = (actual[finite_mask] - expected[finite_mask]).abs().max().item()
        assert torch.allclose(actual[finite_mask], expected[finite_mask], rtol=rtol, atol=atol), max_diff
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isposinf(actual), torch.isposinf(expected))
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))


def tolerance_by_degree(degree: float) -> tuple[float, float]:
    degree = max(0.0, float(degree))
    # Random-coverage cases still need a tolerance rule with slack for float32 recurrence
    # accumulation and input-distribution effects that are not fully captured by a small
    # number of deterministic probes. Keep this helper as a conservative empirical rule
    # for broad random samples, and rely on dedicated anchor tests for strict high-degree
    # regression detection.
    rtol = 3e-4 + degree * 1.5e-5
    atol = 2e-4
    return rtol, atol


SHAPES_2D = [
    (1, 128), (64, 64), (256, 128), (128, 256), (1024, 128), (4096, 128),
    (32, 8), (32, 16), (64, 32), (16, 64), (128, 512),
    (32, 1024), (8, 2048), (4, 4096), (2, 8192),
    (32, 9), (32, 10), (64, 17), (32, 31), (32, 33),
    (128, 65), (64, 100), (16, 127), (16, 129), (32, 255),
    (16, 257), (8, 1000), (4, 3333),
    (1, 128), (3, 128), (7, 64), (17, 128), (63, 256), (65, 256),
    (8192, 128), (16384, 64), (32768, 32), (65537, 16), (4096, 256),
    (8192, 65), (16384, 33),
    (2, 13000), (2, 16384), (1, 20000), (4, 10000), (2, 25000), (1, 32768),
    (2, 13001), (1, 16383), (2, 20001), (1, 9999), (2, 10007),
    (1, 16000),
    (4, 8), (4, 16), (4, 24), (4, 32), (4, 64), (4, 128),
    (4, 256), (4, 512), (4, 1024), (4, 2048), (4, 4096),
]

SHAPES_3D = [
    (1, 128, 128), (2, 64, 128), (4, 256, 128),
    (1, 1, 128), (3, 7, 65), (1, 1, 13000),
    (16, 64, 128), (32, 32, 128), (64, 16, 128), (128, 8, 128),
    (256, 4, 128), (512, 4, 64),
    (4, 128, 128), (4, 256, 128), (8, 512, 128), (2, 1024, 64),
    (32, 64, 128), (64, 32, 64), (128, 128, 32),
    (1024, 8, 128), (2048, 4, 128), (4096, 2, 128),
    (1024, 32, 64), (2048, 16, 64), (8192, 4, 64),
    (4, 1024, 128), (8, 2048, 64), (16, 4096, 32),
    (2, 2048, 128), (4, 4096, 64),
    (512, 256, 64), (1024, 128, 32), (256, 512, 32),
    (128, 1024, 32), (512, 512, 16),
    (16, 16, 65), (32, 8, 257), (8, 32, 1000),
    (256, 32, 65), (512, 16, 33),
    (8, 16, 13000), (4, 32, 10007), (16, 64, 16384), (8, 128, 10000),
]

SHAPES_4D = [
    (2, 3, 4, 128),
    (4, 8, 16, 128), (8, 4, 16, 128), (16, 8, 4, 128),
    (2, 16, 32, 128), (32, 4, 8, 128), (8, 16, 32, 64),
    (2, 4, 8, 1024), (4, 4, 4, 2048),
    (256, 4, 8, 128), (512, 2, 4, 128), (1024, 2, 2, 64),
    (4, 256, 8, 128), (2, 512, 4, 128), (4, 1024, 2, 64),
    (4, 8, 256, 128), (2, 4, 512, 128), (4, 2, 1024, 64),
    (64, 64, 16, 128), (128, 32, 8, 128), (32, 128, 16, 64),
    (256, 16, 16, 64), (64, 64, 64, 32),
    (4, 4, 4, 65), (2, 3, 5, 257), (64, 16, 8, 65), (32, 32, 4, 33),
]


def test_hermite_polynomial_h_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "hermite_polynomial_h")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_hermite_polynomial_h_empty():
    x = torch.empty((0,), dtype=torch.float32)
    n = torch.empty((0,), dtype=torch.float32)
    out = run_op(x, n)
    assert out.shape == x.shape
    assert out.dtype == torch.float32
    assert out.numel() == 0


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", [(1,), (7,), (17,), (128,), (2, 17), (4, 33), (3, 5, 7)])
@pytest.mark.parametrize("degree", [-3.0, -1.0, 0.0, 1.0, 2.0, 5.0, 7.0, 15.0, 32.0, 64.0, 128.0])
def test_hermite_polynomial_h_regular_random(shape, degree):
    x = (torch.rand(*shape, dtype=torch.float32) - 0.5) * 6.0
    n = torch.full(shape, degree, dtype=torch.float32)
    expected = reference(x, n)
    out = run_op(x, n)
    assert out.shape == x.shape
    assert out.dtype == torch.float32
    rtol, atol = tolerance_by_degree(degree)
    assert_close(out, expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES_2D)
def test_hermite_polynomial_h_regbase_2d_shapes(shape):
    x = (torch.rand(*shape, dtype=torch.float32) - 0.5) * 6.0
    n = torch.full(shape, 2.0, dtype=torch.float32)
    expected = reference(x, n)
    out = run_op(x, n)
    assert out.shape == x.shape
    assert_close(out, expected, rtol=6e-4, atol=4e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES_3D)
def test_hermite_polynomial_h_regbase_3d_shapes(shape):
    x = (torch.rand(*shape, dtype=torch.float32) - 0.5) * 6.0
    n = torch.full(shape, 2.0, dtype=torch.float32)
    expected = reference(x, n)
    out = run_op(x, n)
    assert out.shape == x.shape
    assert_close(out, expected, rtol=6e-4, atol=4e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES_4D)
def test_hermite_polynomial_h_regbase_4d_shapes(shape):
    x = (torch.rand(*shape, dtype=torch.float32) - 0.5) * 6.0
    n = torch.full(shape, 2.0, dtype=torch.float32)
    expected = reference(x, n)
    out = run_op(x, n)
    assert out.shape == x.shape
    assert_close(out, expected, rtol=6e-4, atol=4e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_hermite_polynomial_h_broadcast():
    x = (torch.rand(3, 4, 5, dtype=torch.float32) - 0.5) * 4.0
    n = torch.tensor([0.0, 1.0, 2.0, 9.0, 17.0], dtype=torch.float32)
    expected = reference(x, n)
    out = run_op(x, n)
    assert out.shape == expected.shape
    assert_close(out, expected, rtol=4e-4, atol=4e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_hermite_polynomial_h_boundary_samples():
    x = torch.tensor(
        [-3.0, -1.5, -1.0, -0.25, 0.0, 0.25, 1.0, 1.5, 3.0],
        dtype=torch.float32,
    )
    n = torch.tensor(
        [-3.0, -1.0, 0.0, 1.0, 2.0, 7.0, 15.0, 128.0, 129.0],
        dtype=torch.float32,
    )
    expected = reference(x, n)
    out = run_op(x, n)
    assert_close(out, expected, rtol=5e-4, atol=5e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_hermite_polynomial_h_non_contiguous_inputs():
    # Keep this case focused on non-contiguous layout handling. Avoid sampling into
    # the high-degree / large-magnitude region where classification can legitimately
    # flip to NaN and turn this path test into a special-value stress test.
    base_x = (torch.rand(9, 17, dtype=torch.float32) - 0.5) * 2.0
    base_n = torch.randint(-3, 33, (9, 17), dtype=torch.int32).to(torch.float32)
    x = base_x[:, ::2]
    n = base_n[:, ::2]
    assert not x.is_contiguous()
    assert not n.is_contiguous()
    out = run_op(x, n)
    expected = reference(x.contiguous(), n.contiguous())
    assert out.shape == x.shape
    max_degree = float(n.to(torch.int32).max().item()) if n.numel() else 0.0
    rtol, atol = tolerance_by_degree(max_degree)
    assert_close(out, expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_hermite_polynomial_h_fractional_n_matches_truncation():
    x = torch.tensor([-1.5, -0.2, 0.2, 1.5, 2.0], dtype=torch.float32)
    n = torch.tensor([7.9, 6.2, 1.9, -2.7, 128.9], dtype=torch.float32)
    truncated = n.to(torch.int32).to(torch.float32)
    out = run_op(x, n)
    expected = reference(x, truncated)
    assert_close(out, expected, rtol=4e-4, atol=4e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_hermite_polynomial_h_exact_small_orders():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    n = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    expected = torch.tensor([1.0, -2.0, -2.0, -4.0, 76.0], dtype=torch.float32)
    out = run_op(x, n)
    assert_close(out, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_hermite_polynomial_h_dense_manual_recurrence():
    x = torch.linspace(-2.0, 2.0, steps=4096, dtype=torch.float32)
    n = torch.full_like(x, 17.0)
    expected = reference_manual(x, n)
    out = run_op(x, n)
    assert_close(out, expected, rtol=5e-4, atol=5e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_hermite_polynomial_h_limit_behavior():
    x = torch.tensor([-0.75, 0.25, 1.5, -2.0], dtype=torch.float32)
    n = torch.tensor([128.0, 129.0, 130.0, 256.0], dtype=torch.float32)
    expected = reference(x, n)
    out = run_op(x, n)
    assert_close(out, expected, rtol=5e-4, atol=5e-4)
    # Classification should exactly follow reference after n normalization.
    assert torch.equal(torch.isnan(out), torch.isnan(expected))
    assert torch.equal(torch.isinf(out), torch.isinf(expected))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_hermite_polynomial_h_special_values():
    x = torch.tensor(
        [0.0, -1.0, float("nan"), float("inf"), float("-inf"), 0.5, 1.5],
        dtype=torch.float32,
    )
    n = torch.tensor(
        [3.0, 4.0, 5.0, 2.0, 3.0, -1.0, float("nan")],
        dtype=torch.float32,
    )
    expected = reference(x, n)
    out = run_op(x, n)
    assert_close(out, expected, rtol=5e-4, atol=5e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    ("x_value", "degree", "expected_value"),
    [
        (0.0, 32.0, 1.2576278315534066e22),
        (0.5, 32.0, -9.009368863861478e21),
        (1.0, 32.0, -3.891124151796944e21),
        (1.5, 32.0, 3.3471655402342814e22),
        (2.0, 32.0, -9.146296532112032e22),
    ],
)


def test_hermite_polynomial_h_high_degree_anchor_values(x_value, degree, expected_value):
    x = torch.tensor([x_value], dtype=torch.float32)
    n = torch.tensor([degree], dtype=torch.float32)
    expected = torch.tensor([expected_value], dtype=torch.float32)
    out = run_op(x, n)
    assert_close(out, expected, rtol=0.0, atol=0.0)


def test_reference_manual_matches_torch_special():
    x = torch.tensor(
        [-3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0, float("inf"), float("-inf"), float("nan")],
        dtype=torch.float32,
    )
    n = torch.tensor(
        [-3.0, -1.0, 0.0, 1.0, 2.0, 5.0, 32.0, 3.0, 7.0, float("nan")],
        dtype=torch.float32,
    )
    expected = reference(x, n)
    manual = reference_manual(x, n)
    finite_mask = torch.isfinite(expected) & torch.isfinite(manual)
    if finite_mask.any():
        max_diff = (expected[finite_mask] - manual[finite_mask]).abs().max().item()
        assert torch.allclose(expected[finite_mask], manual[finite_mask], rtol=1e-5, atol=1e-5), max_diff
    assert torch.equal(torch.isnan(expected), torch.isnan(manual))
    assert torch.equal(torch.isposinf(expected), torch.isposinf(manual))
    assert torch.equal(torch.isneginf(expected), torch.isneginf(manual))
