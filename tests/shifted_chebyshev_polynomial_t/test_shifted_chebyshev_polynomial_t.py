#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest
import torch
import torch_npu

import ops_multimodal_fusion


if not hasattr(torch.ops.ops_multimodal_fusion, "shifted_chebyshev_polynomial_t"):
    pytest.skip(
        "ops_multimodal_fusion.shifted_chebyshev_polynomial_t not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def reference(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return torch.special.shifted_chebyshev_polynomial_t(x, n)


def run_op(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return torch.ops.ops_multimodal_fusion.shifted_chebyshev_polynomial_t(x.npu(), n.npu()).cpu()


def assert_close(actual: torch.Tensor, expected: torch.Tensor, *, rtol=2e-4, atol=2e-4):
    max_diff = (actual.float() - expected.float()).abs().max().item() if actual.numel() else 0.0
    assert torch.allclose(actual.float(), expected.float(), rtol=rtol, atol=atol, equal_nan=True), max_diff


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


def test_shifted_chebyshev_polynomial_t_interface_exist():
    assert hasattr(torch.ops.ops_multimodal_fusion, "shifted_chebyshev_polynomial_t")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_shifted_chebyshev_polynomial_t_empty():
    x = torch.empty((0,), dtype=torch.float32)
    n = torch.empty((0,), dtype=torch.float32)
    out = run_op(x, n)
    assert out.shape == x.shape
    assert out.dtype == torch.float32
    assert out.numel() == 0


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", [(1,), (7,), (17,), (128,), (2, 17), (4, 33), (3, 5, 7)])
@pytest.mark.parametrize("degree", [-3.0, -1.0, 0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 16.0, 32.0])
def test_shifted_chebyshev_polynomial_t_regular_random(shape, degree):
    x = torch.rand(*shape, dtype=torch.float32) * 2.0
    n = torch.full(shape, degree, dtype=torch.float32)
    expected = reference(x, n)
    out = run_op(x, n)
    assert out.shape == x.shape
    assert out.dtype == torch.float32
    assert_close(out, expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES_2D)
def test_shifted_chebyshev_polynomial_t_regbase_2d_shapes(shape):
    x = torch.rand(*shape, dtype=torch.float32) * 2.0
    n = torch.full(shape, 17.0, dtype=torch.float32)
    expected = reference(x, n)
    out = run_op(x, n)
    assert out.shape == x.shape
    assert_close(out, expected, rtol=3e-4, atol=3e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES_3D)
def test_shifted_chebyshev_polynomial_t_regbase_3d_shapes(shape):
    x = torch.rand(*shape, dtype=torch.float32) * 2.0
    n = torch.full(shape, 17.0, dtype=torch.float32)
    expected = reference(x, n)
    out = run_op(x, n)
    assert out.shape == x.shape
    assert_close(out, expected, rtol=3e-4, atol=3e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES_4D)
def test_shifted_chebyshev_polynomial_t_regbase_4d_shapes(shape):
    x = torch.rand(*shape, dtype=torch.float32) * 2.0
    n = torch.full(shape, 17.0, dtype=torch.float32)
    expected = reference(x, n)
    out = run_op(x, n)
    assert out.shape == x.shape
    assert_close(out, expected, rtol=3e-4, atol=3e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_shifted_chebyshev_polynomial_t_broadcast():
    x = torch.rand(3, 4, 5, dtype=torch.float32) * 2.0
    n = torch.tensor([0.0, 1.0, 2.0, 7.0, 15.0], dtype=torch.float32)
    expected = reference(x, n)
    out = run_op(x, n)
    assert out.shape == expected.shape
    assert_close(out, expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_shifted_chebyshev_polynomial_t_boundary_samples():
    x = torch.tensor(
        [
            -0.5,
            -1e-4,
            0.0,
            1e-4,
            0.25,
            0.5,
            0.75,
            0.9999,
            1.0,
            1.0001,
            1.5,
        ],
        dtype=torch.float32,
    )
    n = torch.tensor(
        [
            -3.0,
            7.0,
            8.0,
            9.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
        ],
        dtype=torch.float32,
    )
    expected = reference(x, n)
    out = run_op(x, n)
    assert_close(out, expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_shifted_chebyshev_polynomial_t_dense_inner_interval():
    x = torch.linspace(1e-4, 0.9999, steps=4096, dtype=torch.float32)
    n = torch.full_like(x, 17.0)
    expected = reference(x, n)
    out = run_op(x, n)
    assert_close(out, expected, rtol=3e-4, atol=3e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_shifted_chebyshev_polynomial_t_non_contiguous_inputs():
    base_x = torch.rand(9, 17, dtype=torch.float32) * 2.0
    base_n = torch.randint(-3, 20, (9, 17), dtype=torch.int32).to(torch.float32)
    x = base_x[:, ::2]
    n = base_n[:, ::2]
    assert not x.is_contiguous()
    assert not n.is_contiguous()
    out = run_op(x, n)
    expected = reference(x.contiguous(), n.contiguous())
    assert out.shape == x.shape
    assert_close(out, expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_shifted_chebyshev_polynomial_t_special_values():
    # The NaN input is excluded from the golden comparison: the torch reference is
    # undefined at NaN and its value there varies with batch layout, so it is not a
    # stable baseline. NaN propagation is checked on its own below.
    x = torch.tensor(
        [0.0, 1.0, float("inf"), float("-inf"), 0.5, 1.5],
        dtype=torch.float32,
    )
    n = torch.tensor(
        [3.0, 4.0, 2.0, 3.0, -1.0, 8.0],
        dtype=torch.float32,
    )
    expected = reference(x, n)
    out = run_op(x, n)
    assert_close(out, expected, rtol=5e-4, atol=5e-4)

    # A NaN input propagates to a NaN output (the polynomial of NaN is NaN).
    x_nan = torch.tensor([float("nan")], dtype=torch.float32)
    n_nan = torch.tensor([5.0], dtype=torch.float32)
    assert torch.isnan(run_op(x_nan, n_nan)).all()


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_shifted_chebyshev_polynomial_t_exact_endpoints():
    x = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    n = torch.tensor([7.0, 8.0, 9.0, 7.0, 8.0, 9.0], dtype=torch.float32)
    expected = reference(x, n)
    out = run_op(x, n)
    assert_close(out, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_shifted_chebyshev_polynomial_t_large_shifted_abs_x_high_degree():
    x = torch.tensor([-10.0, -7.5, -7.0, 8.0, 8.5, 10.0], dtype=torch.float32)
    n = torch.tensor([7.0, 8.0, 9.0, 7.0, 8.0, 9.0], dtype=torch.float32)
    expected = reference(x, n)
    out = run_op(x, n)
    assert_close(out, expected, rtol=5e-6, atol=5e-2)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_shifted_chebyshev_polynomial_t_near_endpoints_high_degree():
    x = torch.tensor(
        [1e-6, 1e-4, 1e-3, 0.999, 0.9999, 0.999999],
        dtype=torch.float32,
    )
    n = torch.tensor([7.0, 7.0, 7.0, 7.0, 7.0, 7.0], dtype=torch.float32)
    expected = reference(x, n)
    out = run_op(x, n)
    assert_close(out, expected, rtol=5e-4, atol=5e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_shifted_chebyshev_polynomial_t_fractional_n_matches_truncation():
    x = torch.tensor([0.1, 0.3, 0.7, 0.9], dtype=torch.float32)
    n = torch.tensor([7.9, 6.2, 1.9, -2.7], dtype=torch.float32)
    truncated = n.to(torch.int32).to(torch.float32)
    out = run_op(x, n)
    expected = reference(x, truncated)
    assert_close(out, expected)
