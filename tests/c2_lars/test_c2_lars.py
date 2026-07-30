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
#
# c2_lars operator precision / functional test suite.
#
# Implements the frozen L0/L1/L2 case matrix of `.cannbot/C2_LARS/2.1-测试方案设计.md` against the
# truth source `.cannbot/C2_LARS/1.1-需求分析.md`.  213 pytest nodes:
#     L0  12  = 1 interface + 11 门槛用例                                   (方案 §4.1)
#     L1 181  = A 54 + B 9 + C 24 + D 15 + E 15 + F 21 + G 2 + S 41         (方案 §4.2)
#     L2  20  = 19 rejection + 1 skipped (non-contiguous, CANN D2D 缺失)     (方案 §4.3)
#
# Self-contained single file (repo convention: no conftest / pytest.ini / CSV / external data table).
# MUST be run from OUTSIDE the source tree so the installed wheel is not shadowed (需求 §4):
#     cd /tmp && pytest /workspace/asc-ops/ops-multimodal-fusion-tpc/tests/c2_lars/ -v
# 分级筛选：pytest -k "l0_"  /  -k "l1_"  /  -k "l2_"
#
# Golden design (方案 §2.2):
#   五步直算，全程 fp64 累加；不调用 torch.linalg.vector_norm（内部防溢出缩放会偏离需求 §1 的朴素语义）。
#   golden 消费的是**送入 NPU 的同一个量化后张量**，故 bf16/fp16 的输入量化误差对两侧共模抵消。
#   host 属性 offset / lr_min 先经 fp32 再进 fp64（方案 §2.2 注），否则「输出恰等于 lr_min」的用例会带
#   约 1.5e-08 的固定相对偏移。
#
# X / dX 数据族配对口径 (方案 §2.2.1，锁定项)：每个用例显式指定 X 与 dX 各自的数据族，二者独立取样
#   (seed / seed ^ 0x5A5A5A5A)；禁止 dX = X 或同族同 seed。
#
# 断言口径 (方案 §6)：数值比对一律 torch.allclose，禁止 torch.equal / == 位精确断言（输出依赖运行期
#   numBlocks 与 SKU）；该禁令同样适用于期望值确定的用例。唯一例外是结构 / 谓词断言
#   (dtype / dim / numel / torch.isnan)。

import logging
import zlib

import pytest
import torch
import torch_npu  # noqa: F401

import ops_multimodal_fusion  # noqa: F401

# ── module-level NPU_ARCH guard (方案 §6) ──────────────────────────────────
if not hasattr(torch.ops.ops_multimodal_fusion, "c2_lars"):
    pytest.skip(
        "ops_multimodal_fusion.c2_lars not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )

requires_npu = pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")

F32, F16, BF16 = torch.float32, torch.float16, torch.bfloat16
DTYPES = [F32, F16, BF16]
_DN = {F32: "fp32", F16: "fp16", BF16: "bf16"}

NAN = float("nan")
INF = float("inf")
NINF = float("-inf")

# 容差按**输入 dtype** 取档 (方案 §2.3；bf16 为 §2.4 标定值)。设计期定值，实施阶段不得放宽。
_TOL = {
    F32: (1e-4, 1e-5),
    F16: (1e-3, 1e-3),
    BF16: (2e-4, 1e-8),
}

# 「常规」标量参数：使 val 落在 clamp 区间中段 (D13 中段)。lr_min 取 1e-6 以保证 numel=1 等
# 极端 ratio 的用例同样不被 clamp 吃掉（否则精度矩阵对两个范数不敏感）。
_STD = dict(wd=0.1, trust=0.5, lr_max=10.0, offset=0.5, lr_min=1e-6)
# Schema 默认值 (需求 §2.2)，供省略实参的用例做 golden。
_DEF_OFFSET, _DEF_LR_MIN = 0.5, 0.02


# ═══════════════════════════════════════════════════════════════════════════
# Interface registration (L0-01)
# ═══════════════════════════════════════════════════════════════════════════
def test_c2_lars_l0_interface_exist():
    """L0-01: the c2_lars operator is registered in torch.ops.ops_multimodal_fusion."""
    assert hasattr(torch.ops.ops_multimodal_fusion, "c2_lars"), \
        "The 'c2_lars' operator is not registered in 'torch.ops.ops_multimodal_fusion'."


# ═══════════════════════════════════════════════════════════════════════════
# Input builders (方案 §3.3.1 数据族定义，设计期冻结)
# ═══════════════════════════════════════════════════════════════════════════
def _seed(cid):
    """Per-case seed frozen at design time (no runtime resampling, 方案 §6)."""
    return zlib.crc32(cid.encode()) & 0x7FFFFFFF


def _mag_floor(v):
    """|x| >= 1e-3：常规数据族远离 fp32 平方下溢区 (方案 §2.5-G1)。"""
    return v.abs().clamp(min=1e-3).copysign(v)


def _make_family(name, numel, seed, val=None):
    """Return a 1-D float64 tensor of the named family (方案 §3.3.1，设计期冻结)。"""
    g = torch.Generator().manual_seed(seed)
    f64 = torch.float64
    if name == "uniform":                                   # U(-1,1)
        return _mag_floor(torch.rand(numel, generator=g, dtype=f64) * 2 - 1)
    if name == "normal":                                    # N(0,1)
        return _mag_floor(torch.randn(numel, generator=g, dtype=f64))
    if name == "widepos":                                   # 10^U(-3,3)
        return torch.pow(10.0, torch.rand(numel, generator=g, dtype=f64) * 6 - 3)
    if name == "desc":                                      # widepos 降序排列
        w = torch.pow(10.0, torch.rand(numel, generator=g, dtype=f64) * 6 - 3)
        return torch.sort(w, descending=True).values
    if name == "tiny":                                      # U(-1e-3,1e-3)
        return (torch.rand(numel, generator=g, dtype=f64) * 2 - 1) * 1e-3
    if name == "big":                                       # U(-100,100)
        return _mag_floor((torch.rand(numel, generator=g, dtype=f64) * 2 - 1) * 100)
    if name == "u256":                                      # U(256,2000) —— L1-G fp16 平方溢出窗口
        return torch.rand(numel, generator=g, dtype=f64) * (2000.0 - 256.0) + 256.0
    if name == "ones":                                      # L1-F 哨兵载体，三种 dtype 均精确可表
        return torch.ones(numel, dtype=f64)
    if name == "zeros":
        return torch.zeros(numel, dtype=f64)
    if name == "onehot":                                    # 单点非零，其余全零
        t = torch.zeros(numel, dtype=f64)
        t[0] = 1.0
        return t
    if name == "const":                                     # 定值探针 (SV-10 / SV-11)
        return torch.full((numel,), float(val), dtype=f64)
    raise ValueError(f"unknown data family: {name}")


def _build(spec):
    """Return (X, dX) in the case dtype, shaped per spec（数据在 fp64 构造，末尾一次性量化）。"""
    shape = spec["shape"]
    numel = 1
    for d in shape:
        numel *= d
    x = _make_family(spec["xfam"], numel, spec["seed_x"], spec["xval"])
    d = _make_family(spec["dfam"], numel, spec["seed_d"], spec["dval"])

    clip = spec["clip"]
    if clip is not None:                                    # fp16 档 clip 到 ±200 以避开 G2 (方案 §3.3.1)
        x = x.clamp(-clip, clip)
        d = d.clamp(-clip, clip)

    if spec["sentinel"] is not None:                        # L1-F 尾块哨兵 (方案 §3.5)
        tgt, sval = spec["sentinel"]
        (x if tgt == "x" else d)[-1] = sval

    for tgt, idx, pval in spec["poison"]:                   # SV-09a/b/c 非常规浮点植入
        (x if tgt == "x" else d)[idx] = pval

    return x.to(spec["dtype"]).reshape(shape), d.to(spec["dtype"]).reshape(shape)


def _scalar(v, form):
    """wd / trust / lr_max：恒 float32，0-D 或 (1,) 均合法 (需求 C3 / 方案 D17)。"""
    t = torch.tensor(float(v), dtype=F32)
    return t.reshape(1) if form == "1d" else t


# ═══════════════════════════════════════════════════════════════════════════
# Golden —— 方案 §2.2 的五步直算，逐步对应需求 §1 公式
# ═══════════════════════════════════════════════════════════════════════════
def _host_attr_f64(v):
    """host 属性先经 fp32 再进 fp64 (方案 §2.2)：kernel 内 offset / lr_min 按 fp32 参与运算。"""
    return torch.tensor(float(v), dtype=F32).to(torch.float64)


def _golden_parts(X, dX, wd, trust, lr_max, offset, lr_min):
    """Return (X_norm, dX_norm, val, out) —— val 为 clamp 前的 fp64 中间量。"""
    f64 = torch.float64

    x64 = X.to(f64)                                          # ① X_norm = ||X||_2，fp64 累加
    sq = x64 * x64
    xn = torch.sqrt(torch.sum(sq))
    del x64, sq

    d64 = dX.to(f64)                                         # ② dX_norm = ||dX||_2
    sq = d64 * d64
    dn = torch.sqrt(torch.sum(sq))
    del d64, sq

    wd64 = wd.to(f64).reshape(())
    trust64 = trust.to(f64).reshape(())
    lrmax64 = lr_max.to(f64).reshape(())
    off64 = _host_attr_f64(offset)
    lmin64 = _host_attr_f64(lr_min)

    # ③ val = (X_norm > 0) ? trust / (dX_norm/X_norm + wd + offset) : 1.0
    #    X_norm 为 NaN 时谓词为假 -> 走 val = 1.0 分支 (方案 §3.6 SV-09a)。
    if bool(xn > 0):
        val = trust64 / (dn / xn + wd64 + off64)
    else:
        val = torch.tensor(1.0, dtype=f64)

    # ④ lr_rescaled = max(min(val, lr_max), lr_min)；用 minimum/maximum 保证 NaN 传播确定可复现
    out = torch.maximum(torch.minimum(val, lrmax64), lmin64)

    return xn, dn, val, out.to(F32).reshape(())              # ⑤ 输出 0-D fp32 (C7)


def _fp32_semantics_expected(mode, wd, trust, lr_max, offset, lr_min):
    """SV-10 / SV-11 的 fp32 语义期望值 (方案 §2.5 G1 / G2)，与 fp64 golden 被设计成分歧。"""
    f64 = torch.float64
    wd64 = wd.to(f64).reshape(())
    trust64 = trust.to(f64).reshape(())
    lrmax64 = lr_max.to(f64).reshape(())
    off64 = _host_attr_f64(offset)
    lmin64 = _host_attr_f64(lr_min)
    if mode == "G1":                     # x² 在 fp32 下溢为 0 -> X_norm == 0 -> val = 1.0
        val = torch.tensor(1.0, dtype=f64)
    else:                                # G2: x² 上溢 -> X_norm = +inf -> dX_norm/X_norm = 0
        val = trust64 / (wd64 + off64)
    return torch.maximum(torch.minimum(val, lrmax64), lmin64).to(F32).reshape(())


# ═══════════════════════════════════════════════════════════════════════════
# Call / assertions
# ═══════════════════════════════════════════════════════════════════════════
def _call(X, dX, wd, trust, lr_max, offset, lr_min, omit="none"):
    op = torch.ops.ops_multimodal_fusion.c2_lars
    args = (X.npu(), dX.npu(), wd.npu(), trust.npu(), lr_max.npu())
    if omit == "both":                       # 走 schema 默认 offset=0.5 / lr_min=0.02
        return op(*args)
    if omit == "lr_min":
        return op(*args, offset)
    return op(*args, offset, lr_min)


def _assert_contract(out, cid):
    """C7 结构契约：0-D、numel==1、dtype 恒 float32。结构断言，不受禁位精确比对条款约束。"""
    assert out.dtype == F32, f"{cid}: out.dtype {out.dtype} != torch.float32 (C7)"
    assert out.dim() == 0, f"{cid}: out.dim() {out.dim()} != 0 (C7)"
    assert out.numel() == 1, f"{cid}: out.numel() {out.numel()} != 1 (C7)"
    assert "npu" in str(out.device), f"{cid}: out.device {out.device} is not npu"


def _fail_msg(cid, spec, actual, expected, rtol, atol, xn, dn, val):
    a, e = actual.double().item(), expected.double().item()
    abs_d = abs(a - e)
    rel_d = abs_d / abs(e) if e != 0 else INF
    return (f"{cid}: precision FAIL dtype={_DN[spec['dtype']]} shape={spec['shape']} "
            f"X={spec['xfam']} dX={spec['dfam']}\n"
            f"  wd={spec['wd']} trust={spec['trust']} lr_max={spec['lr_max']} "
            f"offset={spec['offset']} lr_min={spec['lr_min']} omit={spec['omit']}\n"
            f"  X_norm={xn.item():.9e} dX_norm={dn.item():.9e} val={val.item():.9e}\n"
            f"  actual={a!r} expected={e!r}\n"
            f"  max_abs={abs_d:.6e} max_rel={rel_d:.6e} (rtol={rtol:.1e} atol={atol:.1e})")


def _run(spec):
    """统一用例执行：结构契约 -> golden -> allclose -> 失败诊断 (方案 §6 通用断言封装)。"""
    cid, dtype = spec["id"], spec["dtype"]
    X, dX = _build(spec)
    wd = _scalar(spec["wd"], spec["scalar_form"])
    trust = _scalar(spec["trust"], spec["scalar_form"])
    lr_max = _scalar(spec["lr_max"], spec["scalar_form"])
    offset, lr_min = spec["offset"], spec["lr_min"]

    xn, dn, val, expected = _golden_parts(X, dX, wd, trust, lr_max, offset, lr_min)

    # 前置条件（非验收判据、不放宽容差）：精度主矩阵 / 对抗组必须不被 clamp 吃掉，否则对两个范数不敏感。
    if spec["assert_unclamped"]:
        v = val.item()
        assert lr_min < v < spec["lr_max"], \
            f"{cid}: golden val={v:.6e} clamped into [{lr_min}, {spec['lr_max']}]; " \
            f"该用例失去对 X_norm/dX_norm 的敏感性，设计期参数需重选"

    out = _call(X, dX, wd, trust, lr_max, offset, lr_min, spec["omit"])
    _assert_contract(out, cid)
    actual = out.cpu()
    rtol, atol = _TOL[dtype]

    if spec["record"] is not None:
        # 行为记录型 (方案 §6)：仅 SV-10 / SV-11。golden 与 NPU 被设计成分歧 (§2.5 G1/G2)，
        # 不存在共同真值，故作弱（析取）断言并 logging 实际值供 CP3 登记。
        exp_fp32 = _fp32_semantics_expected(spec["record"], wd, trust, lr_max, offset, lr_min)
        hit_fp32 = torch.allclose(actual, exp_fp32, rtol=rtol, atol=atol)
        hit_f64 = torch.allclose(actual, expected, rtol=rtol, atol=atol)
        logging.info("%s [behaviour record %s] actual=%r fp32-semantics=%r fp64-golden=%r branch=%s",
                     cid, spec["record"], actual.item(), exp_fp32.item(), expected.item(),
                     "fp32" if hit_fp32 else ("fp64" if hit_f64 else "NEITHER"))
        assert hit_fp32 or hit_f64, \
            f"{cid}: {spec['record']} behaviour record —— actual={actual.item()!r} 既不匹配 fp32 语义 " \
            f"{exp_fp32.item()!r} 也不匹配 fp64 golden {expected.item()!r}"
    elif bool(torch.isnan(expected)):
        # 谓词断言（方案 §6 唯一例外）：golden 为 NaN 时 allclose 无定义，改用 isnan 强断言。
        assert bool(torch.isnan(actual)), \
            f"{cid}: golden is NaN but actual={actual.item()!r} (NaN 传播缺失)"
    else:
        assert torch.allclose(actual, expected, rtol=rtol, atol=atol), \
            _fail_msg(cid, spec, actual, expected, rtol, atol, xn, dn, val)

    if spec["expect_range"]:
        # SV-04 附加断言：分母为负时输出仍须落在 [min(lr_max,lr_min), max(lr_max,lr_min)] 且不报错。
        lo, hi = min(spec["lr_max"], lr_min), max(spec["lr_max"], lr_min)
        a = actual.item()
        assert lo - atol <= a <= hi + atol, f"{cid}: out={a!r} escaped clamp range [{lo}, {hi}]"

    if spec["large"]:
        del out, actual, X, dX
        torch.npu.empty_cache()


def _device_health_check():
    """异常拦截后确认设备上下文未被污染（host 侧 TORCH_CHECK 不应留下副作用）。"""
    g = torch.Generator().manual_seed(20260727)
    x = (torch.rand(256, generator=g, dtype=torch.float64) + 0.5).to(F32)
    d = (torch.rand(256, generator=g, dtype=torch.float64) + 0.5).to(F32)
    wd, trust, lr_max = _scalar(0.1, "0d"), _scalar(0.5, "0d"), _scalar(10.0, "0d")
    out = _call(x, d, wd, trust, lr_max, 0.5, 1e-6).cpu()
    _, _, _, expected = _golden_parts(x, d, wd, trust, lr_max, 0.5, 1e-6)
    assert torch.allclose(out, expected, rtol=1e-4, atol=1e-5), \
        f"device health check FAILED after a rejected call: " \
        f"actual={out.item()!r} expected={expected.item()!r}"


# ═══════════════════════════════════════════════════════════════════════════
# Case spec factory
# ═══════════════════════════════════════════════════════════════════════════
def _c(cid, dtype=F32, numel=None, shape=None, xfam="uniform", dfam="normal",
       wd=_STD["wd"], trust=_STD["trust"], lr_max=_STD["lr_max"],
       offset=_STD["offset"], lr_min=_STD["lr_min"],
       omit="none", scalar_form="0d", clip=None, sentinel=None, poison=(),
       xval=None, dval=None, assert_unclamped=False, expect_range=False, record=None, large=False):
    if shape is None:
        shape = (numel,)
    return dict(id=cid, dtype=dtype, shape=tuple(shape), xfam=xfam, dfam=dfam,
                wd=wd, trust=trust, lr_max=lr_max, offset=offset, lr_min=lr_min,
                omit=omit, scalar_form=scalar_form, clip=clip, sentinel=sentinel,
                poison=tuple(poison), xval=xval, dval=dval,
                seed_x=_seed(cid), seed_d=_seed(cid) ^ 0x5A5A5A5A,
                assert_unclamped=assert_unclamped, expect_range=expect_range,
                record=record, large=large)


def _ids(cases):
    return [c["id"] for c in cases]


# ═══════════════════════════════════════════════════════════════════════════
# L0 门槛用例 (12 条：L0-01 接口 + L0-02~12)              —— 方案 §4.1
# ═══════════════════════════════════════════════════════════════════════════
L0_CASES = [
    # L0-02 输出结构契约 (D20)；_run 对每个用例都执行 C7 结构断言，本例是其显式锚点。
    _c("L0-02", numel=1024, dtype=F32, assert_unclamped=True),
    _c("L0-03", numel=1024, dtype=F32, assert_unclamped=True),
    _c("L0-04", numel=1024, dtype=F16, assert_unclamped=True),
    _c("L0-05", numel=1024, dtype=BF16, assert_unclamped=True),
    _c("L0-06", shape=(32, 32), dtype=F32, assert_unclamped=True),
    _c("L0-07", shape=(8, 16, 32), dtype=F32, assert_unclamped=True),
    # L0-08 省略 offset / lr_min，走 schema 默认 0.5 / 0.02
    _c("L0-08", numel=1024, dtype=F32, omit="both", offset=_DEF_OFFSET, lr_min=_DEF_LR_MIN,
       assert_unclamped=True),
    _c("L0-09", numel=1, dtype=F32, assert_unclamped=True),
    _c("L0-10", numel=1000, dtype=F32, assert_unclamped=True),
    # L0-11 X 全零 -> val = 1.0 分支，lr_max=10 / lr_min=0.02 -> 输出 1.0
    _c("L0-11", numel=1024, dtype=F32, xfam="zeros", lr_max=10.0, lr_min=0.02),
    # L0-12 clamp 到 lr_max = 0.001
    _c("L0-12", numel=1024, dtype=F32, lr_max=0.001, lr_min=1e-6),
]


@requires_npu
@pytest.mark.parametrize("spec", L0_CASES, ids=_ids(L0_CASES))
def test_c2_lars_l0_basic(spec):
    _run(spec)


# ═══════════════════════════════════════════════════════════════════════════
# L1-A 量级 × 对齐 × dtype 精度主矩阵 (54 条)              —— 方案 §3.2
#   配对锁定 X=uniform / dX=normal（异族、独立 seed，|x| >= 1e-3 规避 G1）。
# ═══════════════════════════════════════════════════════════════════════════
_A_NUMELS = [
    1,                       # 任务强制：numel=1；起核数 = 1
    7,                       # 亚 32B，单块内非对齐尾
    63, 64, 65,              # fp32 lane(64) -1 / 对齐 / +1
    127, 128, 129,           # 低精度 lane(128) -1 / 对齐 / +1
    1000,                    # 任务强制 1e3，非 2 幂
    1023, 1024, 1025,        # tile 边界三点
    4096,                    # 多 tile 起点
    99991,                   # 任务强制 1e5 + 质数强非对齐
    100000,                  # 任务强制 1e5，非 2 幂
    262144,                  # 2^18 全对齐对照
    9999991,                 # 任务强制 1e7 + 质数强非对齐
    10000000,                # 任务强制 1e7；跨核归约压力最大
]
L1A_CASES = [
    _c(f"L1-A-{n}-{_DN[dt]}", numel=n, dtype=dt, xfam="uniform", dfam="normal",
       assert_unclamped=True, large=(n >= 1000000))
    for n in _A_NUMELS for dt in DTYPES
]


@requires_npu
@pytest.mark.parametrize("spec", L1A_CASES, ids=_ids(L1A_CASES))
def test_c2_lars_l1_a_precision_matrix(spec):
    _run(spec)


# ═══════════════════════════════════════════════════════════════════════════
# L1-B shape rank (9 条)                                   —— 方案 §3.3
# ═══════════════════════════════════════════════════════════════════════════
_B_SHAPES = [(), (4096,), (64, 64), (8, 16, 32), (2, 3, 4, 5), (2, 3, 4, 5, 6)]
L1B_CASES = [
    _c(f"L1-B-rank{len(s)}", shape=s, dtype=F32, assert_unclamped=True) for s in _B_SHAPES
] + [
    _c(f"L1-B-big4d-{_DN[dt]}", shape=(16, 16, 128, 128), dtype=dt,
       assert_unclamped=True, large=True) for dt in DTYPES
]


@requires_npu
@pytest.mark.parametrize("spec", L1B_CASES, ids=_ids(L1B_CASES))
def test_c2_lars_l1_b_shape_rank(spec):
    _run(spec)


# ═══════════════════════════════════════════════════════════════════════════
# L1-C 对抗数据族 × 长累加链 (24 条)                        —— 方案 §3.3.1
#   缺陷检出承重组：「核 partial 丢失」类缺陷在 L1-A 主矩阵上仅 9.10e-06（检不出），
#   只在 dX=desc 的 C-1 / C-5 配对上达到 7.5e-03~8.9e-03。C-1（uniform→desc）同时是
#   bf16 容差 rtol=2e-4 的标定 worst case（实测 8.85e-06），标定用例必须在测试集内。
#   不得以运行耗时为由裁剪本组。
# ═══════════════════════════════════════════════════════════════════════════
_C_PAIRS = [
    ("C-1", "uniform", "desc"),
    ("C-2", "desc", "uniform"),
    ("C-3", "uniform", "widepos"),
    ("C-4", "widepos", "tiny"),
    ("C-5", "big", "desc"),
    ("C-6", "normal", "big"),
]
L1C_CASES = [
    _c(f"L1-{tag}-1e7-{_DN[dt]}", numel=9999991, dtype=dt, xfam=xf, dfam=df,
       clip=(200.0 if dt is F16 else None), assert_unclamped=True, large=True)
    for (tag, xf, df) in _C_PAIRS for dt in DTYPES
] + [
    # C-7：同 6 组配对在 numel=1000003 上再铺一遍 bf16（对长链效应最敏感的一档）
    _c(f"L1-C-7-{tag}-bf16", numel=1000003, dtype=BF16, xfam=xf, dfam=df,
       assert_unclamped=True, large=True)
    for (tag, xf, df) in _C_PAIRS
]


@requires_npu
@pytest.mark.parametrize("spec", L1C_CASES, ids=_ids(L1C_CASES))
def test_c2_lars_l1_c_adversarial(spec):
    _run(spec)


# ═══════════════════════════════════════════════════════════════════════════
# L1-D clamp 落点 / 分支 (15 条)                            —— 方案 §3.3
# ═══════════════════════════════════════════════════════════════════════════
L1D_CASES = []
for _dt in DTYPES:                                   # clamp 三落点 (D13)
    # 落 lr_min：val ≈ 0.214 < lr_min = 0.5
    L1D_CASES.append(_c(f"L1-D-clamp-lo-{_DN[_dt]}", numel=1024, dtype=_dt,
                        lr_max=10.0, lr_min=0.5))
for _dt in DTYPES:
    # 落 val：lr_min <= val <= lr_max（中段）
    L1D_CASES.append(_c(f"L1-D-clamp-mid-{_DN[_dt]}", numel=1024, dtype=_dt,
                        assert_unclamped=True))
for _dt in DTYPES:
    # 落 lr_max：val ≈ 0.214 > lr_max = 0.001
    L1D_CASES.append(_c(f"L1-D-clamp-hi-{_DN[_dt]}", numel=1024, dtype=_dt,
                        lr_max=0.001, lr_min=1e-6))
for _dt in DTYPES:                                   # X 全零 -> val = 1.0 分支 (D14)
    L1D_CASES.append(_c(f"L1-D-xzero-{_DN[_dt]}", numel=1024, dtype=_dt, xfam="zeros",
                        lr_max=10.0, lr_min=0.02))
for _dt in DTYPES:                                   # X 单点非零，其余全零 (D06) @ numel=99991
    L1D_CASES.append(_c(f"L1-D-onehot-{_DN[_dt]}", numel=99991, dtype=_dt, xfam="onehot",
                        assert_unclamped=True))


@requires_npu
@pytest.mark.parametrize("spec", L1D_CASES, ids=_ids(L1D_CASES))
def test_c2_lars_l1_d_branch(spec):
    _run(spec)


# ═══════════════════════════════════════════════════════════════════════════
# L1-E 标量形态 / 默认参数 / offset-lr_min 取值 (15 条)      —— 方案 §3.3
# ═══════════════════════════════════════════════════════════════════════════
L1E_CASES = []
for _dt in DTYPES:                                   # D17：标量 Tensor 0-D 形态
    L1E_CASES.append(_c(f"L1-E-scalar0d-{_DN[_dt]}", numel=1024, dtype=_dt, scalar_form="0d",
                        assert_unclamped=True))
for _dt in DTYPES:                                   # D17：标量 Tensor (1,) 形态
    L1E_CASES.append(_c(f"L1-E-scalar1d-{_DN[_dt]}", numel=1024, dtype=_dt, scalar_form="1d",
                        assert_unclamped=True))
for _dt in DTYPES:                                   # D16：省略 offset + lr_min（走默认 0.5 / 0.02）
    L1E_CASES.append(_c(f"L1-E-default-both-{_DN[_dt]}", numel=1024, dtype=_dt, omit="both",
                        offset=_DEF_OFFSET, lr_min=_DEF_LR_MIN, assert_unclamped=True))
for _dt in DTYPES:                                   # D16：仅省略 lr_min（显式传 offset=0.25）
    L1E_CASES.append(_c(f"L1-E-default-lrmin-{_DN[_dt]}", numel=1024, dtype=_dt, omit="lr_min",
                        offset=0.25, lr_min=_DEF_LR_MIN, assert_unclamped=True))
L1E_CASES += [                                       # D11 / D12：offset / lr_min 取值
    _c("L1-E-offset0", numel=1024, dtype=F32, offset=0.0, assert_unclamped=True),
    _c("L1-E-offset1e3", numel=1024, dtype=F32, offset=1e3, assert_unclamped=True),
    _c("L1-E-lrmin0", numel=1024, dtype=F32, lr_min=0.0, assert_unclamped=True),
]


@requires_npu
@pytest.mark.parametrize("spec", L1E_CASES, ids=_ids(L1E_CASES))
def test_c2_lars_l1_e_param_form(spec):
    _run(spec)


# ═══════════════════════════════════════════════════════════════════════════
# L1-F 尾块哨兵 (21 条)                                     —— 方案 §3.5
#   「归约树接错 / 尾块漏算」类缺陷从 X 与 dX 中等比例剥掉同一份，在比值中精确抵消，
#   任何容差档位都检不出，只能靠哨兵。哨兵取 2^11 = 2048.0：含哨兵的 64 元素 chunk 求和
#   63 + 2^22 = 4194367 < 2^24，累加阶段全程零舍入；最弱一档（numel=1e7）信号仍达 16.1%。
#   哨兵值与各 numel 配置均为方案 §3.5 定值，不得调整。
# ═══════════════════════════════════════════════════════════════════════════
_F_NUMELS = [65, 129, 1023, 99991, 262143, 9999991]
_SENTINEL = 2048.0                                   # 2^11
L1F_CASES = [
    _c(f"L1-F-dx-{n}-{_DN[dt]}", numel=n, dtype=dt, xfam="ones", dfam="ones",
       sentinel=("dx", _SENTINEL), assert_unclamped=True, large=(n >= 1000000))
    for n in _F_NUMELS for dt in DTYPES
] + [
    # 镜像子组：哨兵放在 X 侧，确保 X 侧尾块路径同样受检
    _c(f"L1-F-x-99991-{_DN[dt]}", numel=99991, dtype=dt, xfam="ones", dfam="ones",
       sentinel=("x", _SENTINEL), assert_unclamped=True)
    for dt in DTYPES
]


@requires_npu
@pytest.mark.parametrize("spec", L1F_CASES, ids=_ids(L1F_CASES))
def test_c2_lars_l1_f_tail_sentinel(spec):
    _run(spec)


# ═══════════════════════════════════════════════════════════════════════════
# L1-G fp16 逐元素平方溢出窗口 (2 条)                        —— 方案 §3.3.2
#   |x| ∈ (256, 2000)：x² > 65504 必溢出 fp16，若在 fp16 中平方即得 inf -> 输出 NaN/inf；
#   正确实现（load 后先 cast fp32 再平方）得有限值。X / dX 用独立 seed（§2.2.1 条 4）。
# ═══════════════════════════════════════════════════════════════════════════
L1G_CASES = [
    _c("L1-G-1", numel=99991, dtype=F16, xfam="u256", dfam="u256", assert_unclamped=True),
    _c("L1-G-2", numel=9999991, dtype=F16, xfam="u256", dfam="u256",
       assert_unclamped=True, large=True),
]


@requires_npu
@pytest.mark.parametrize("spec", L1G_CASES, ids=_ids(L1G_CASES))
def test_c2_lars_l1_g_fp16_square_overflow(spec):
    # 数值判据由 _run 的 allclose 完成：错误实现（fp16 内平方）得 inf/NaN，必然 fail。
    _run(spec)


# ═══════════════════════════════════════════════════════════════════════════
# L1-S 特殊值 (41 条)                                       —— 方案 §3.6
#   除 SV-10 / SV-11（G1/G2 分歧，行为记录型）外一律强断言。
# ═══════════════════════════════════════════════════════════════════════════
_SV_N = 1024
L1S_CASES = []

# SV-01/02/03：分母为 0（dX 全零 + wd=0 + offset=0），trust 取 > 0 / < 0 / == 0
for _dt in DTYPES:                                   # val = +inf -> 输出 lr_max = 10
    L1S_CASES.append(_c(f"SV-01-{_DN[_dt]}", numel=_SV_N, dtype=_dt, dfam="zeros",
                        wd=0.0, offset=0.0, trust=1.0, lr_max=10.0, lr_min=0.02))
for _dt in DTYPES:                                   # val = -inf -> 输出 lr_min = 0.02
    L1S_CASES.append(_c(f"SV-02-{_DN[_dt]}", numel=_SV_N, dtype=_dt, dfam="zeros",
                        wd=0.0, offset=0.0, trust=-1.0, lr_max=10.0, lr_min=0.02))
for _dt in DTYPES:                                   # val = 0/0 = NaN -> 强断言 isnan
    L1S_CASES.append(_c(f"SV-03-{_DN[_dt]}", numel=_SV_N, dtype=_dt, dfam="zeros",
                        wd=0.0, offset=0.0, trust=0.0, lr_max=10.0, lr_min=0.02))
# SV-04：分母为负。X=big / dX=tiny 使 dX_norm/X_norm ≈ 1e-5，wd=-2.0 + offset=0.5 -> 分母 ≈ -1.5
for _dt in DTYPES:
    L1S_CASES.append(_c(f"SV-04-{_DN[_dt]}", numel=_SV_N, dtype=_dt, xfam="big", dfam="tiny",
                        wd=-2.0, offset=0.5, trust=1.0, lr_max=10.0, lr_min=0.02,
                        expect_range=True))
# SV-05：X 全零 -> val = 1.0 分支，两组 clamp 参数（输出 1.0 / 输出 0.5，验证 clamp 仍生效）
for _dt in DTYPES:
    L1S_CASES.append(_c(f"SV-05a-{_DN[_dt]}", numel=_SV_N, dtype=_dt, xfam="zeros",
                        lr_max=10.0, lr_min=0.02))
for _dt in DTYPES:
    L1S_CASES.append(_c(f"SV-05b-{_DN[_dt]}", numel=_SV_N, dtype=_dt, xfam="zeros",
                        lr_max=0.5, lr_min=0.02))
# SV-06：lr_max < lr_min -> max 后置，输出恒为 lr_min = 0.5
for _dt in (F32, BF16):
    L1S_CASES.append(_c(f"SV-06-{_DN[_dt]}", numel=_SV_N, dtype=_dt, lr_max=0.01, lr_min=0.5))
# SV-07：X 与 dX 均全零 -> X_norm==0 优先，val = 1.0（不进入 0/0）
L1S_CASES.append(_c("SV-07-fp32", numel=_SV_N, dtype=F32, xfam="zeros", dfam="zeros",
                    lr_max=10.0, lr_min=0.02))
# SV-08：dX 全零、wd > 0 -> 分母有限 = 0.6，val = trust/0.6
for _dt in (F32, BF16):
    L1S_CASES.append(_c(f"SV-08-{_DN[_dt]}", numel=_SV_N, dtype=_dt, dfam="zeros",
                        wd=0.1, offset=0.5, trust=0.5, lr_max=10.0, lr_min=0.02))
# SV-09a/b/c：X 含 NaN / +inf / -inf
for _dt in DTYPES:                                   # X_norm = NaN -> 谓词假 -> val = 1.0
    L1S_CASES.append(_c(f"SV-09a-{_DN[_dt]}", numel=_SV_N, dtype=_dt, poison=[("x", 0, NAN)],
                        wd=0.1, offset=0.5, trust=0.5, lr_max=10.0, lr_min=0.02))
for _dt in DTYPES:                                   # X_norm = +inf -> dX_norm/X_norm = 0
    L1S_CASES.append(_c(f"SV-09b-{_DN[_dt]}", numel=_SV_N, dtype=_dt, poison=[("x", 0, INF)],
                        wd=0.1, offset=0.5, trust=0.5, lr_max=10.0, lr_min=0.02))
for _dt in DTYPES:                                   # (-inf)² = +inf，同 SV-09b
    L1S_CASES.append(_c(f"SV-09c-{_DN[_dt]}", numel=_SV_N, dtype=_dt, poison=[("x", 0, NINF)],
                        wd=0.1, offset=0.5, trust=0.5, lr_max=10.0, lr_min=0.02))
# SV-10：fp32 平方下溢分歧 G1（|x|=1e-30 -> x²=1e-60 在 fp32 必舍为 0）；fp16 结构不可呈现，不设
for _dt in (F32, BF16):
    L1S_CASES.append(_c(f"SV-10-{_DN[_dt]}", numel=_SV_N, dtype=_dt, xfam="const", xval=1e-30,
                        dfam="uniform", wd=0.1, offset=0.5, trust=0.5, lr_max=10.0, lr_min=0.02,
                        record="G1"))
# SV-11：fp32 平方和上溢分歧 G2（|x|=1e20 -> x²=1e40 溢出 fp32）
for _dt in (F32, BF16):
    L1S_CASES.append(_c(f"SV-11-{_DN[_dt]}", numel=_SV_N, dtype=_dt, xfam="const", xval=1e20,
                        dfam="uniform", wd=0.1, offset=0.5, trust=0.5, lr_max=10.0, lr_min=0.02,
                        record="G2"))
# SV-12：trust 极小 / lr_max 极大 -> 输出 ~1e-6 量级，验证 atol 兜底路径
for _dt in (F32, BF16):
    L1S_CASES.append(_c(f"SV-12-{_DN[_dt]}", numel=_SV_N, dtype=_dt,
                        trust=1e-6, lr_max=1e30, lr_min=0.0, assert_unclamped=True))
# SV-13：lr_max == lr_min -> 输出恒为 0.3
L1S_CASES.append(_c("SV-13-fp32", numel=_SV_N, dtype=F32, lr_max=0.3, lr_min=0.3))
# SV-14：wd 为负但分母仍为正（1.73 - 0.2 + 0.5 ≈ 2.03）
for _dt in (F32, BF16):
    L1S_CASES.append(_c(f"SV-14-{_DN[_dt]}", numel=_SV_N, dtype=_dt, wd=-0.2, offset=0.5,
                        trust=0.5, lr_max=10.0, lr_min=0.02, assert_unclamped=True))


@requires_npu
@pytest.mark.parametrize("spec", L1S_CASES, ids=_ids(L1S_CASES))
def test_c2_lars_l1_s_special_values(spec):
    _run(spec)


# ═══════════════════════════════════════════════════════════════════════════
# L2 异常用例 (20 条：19 拦截 + 1 skip)                     —— 方案 §3.4
# ═══════════════════════════════════════════════════════════════════════════
def _op_raw(X, dX, wd, trust, lr_max, offset, lr_min):
    """L2 专用直调：入参已按用例意图放好 device，**不得**再走 _call 的 .npu()，
    否则 L2-18 / L2-19 这类 device 用例会被悄悄搬到 npu 上而失效。"""
    return torch.ops.ops_multimodal_fusion.c2_lars(X, dX, wd, trust, lr_max, offset, lr_min)


def _npu_x(shape, dtype=F32):
    g = torch.Generator().manual_seed(7)
    if dtype in (torch.int32, torch.int64):
        return torch.randint(1, 5, shape, generator=g, dtype=dtype).npu()
    if dtype == torch.complex64:
        n = 1
        for d in shape:
            n *= d
        return torch.randn(n, generator=g).to(torch.complex64).reshape(shape).npu()
    return (torch.rand(shape, generator=g) + 0.5).to(dtype).npu()


def _npu_scalars(wd_dtype=F32, trust_dtype=F32, lr_max_dtype=F32,
                 wd_shape=(), trust_shape=(), lr_max_shape=(), trust_on_cpu=False):
    wd = torch.full(wd_shape, 0.1).to(wd_dtype).npu()
    trust = torch.full(trust_shape, 0.5).to(trust_dtype)
    trust = trust if trust_on_cpu else trust.npu()
    lr_max = torch.full(lr_max_shape, 10.0).to(lr_max_dtype).npu()
    return wd, trust, lr_max


def _rej_shape(x_shape, d_shape, dtype=F32, d_dtype=None):
    def f():
        X = _npu_x(x_shape, dtype)
        dX = _npu_x(d_shape, d_dtype or dtype)
        wd, trust, lr_max = _npu_scalars()
        return _op_raw(X, dX, wd, trust, lr_max, 0.5, 0.02)
    return f


def _rej_x_dtype(bad):
    def f():
        X = _npu_x((100,), bad)
        dX = _npu_x((100,), bad)
        wd, trust, lr_max = _npu_scalars()
        return _op_raw(X, dX, wd, trust, lr_max, 0.5, 0.02)
    return f


def _rej_scalar(**kw):
    def f():
        X = _npu_x((100,))
        dX = _npu_x((100,))
        wd, trust, lr_max = _npu_scalars(**kw)
        return _op_raw(X, dX, wd, trust, lr_max, 0.5, 0.02)
    return f


def _rej_attr(offset, lr_min):
    def f():
        X = _npu_x((100,))
        dX = _npu_x((100,))
        wd, trust, lr_max = _npu_scalars()
        return _op_raw(X, dX, wd, trust, lr_max, offset, lr_min)
    return f


def _rej_x_on_cpu():
    g = torch.Generator().manual_seed(7)
    X = torch.rand((100,), generator=g) + 0.5
    dX = torch.rand((100,), generator=g) + 0.5
    wd, trust, lr_max = _npu_scalars()
    return _op_raw(X, dX, wd, trust, lr_max, 0.5, 0.02)   # X/dX 停留在 CPU


L2_REJECT = [
    ("L2-01", _rej_shape((0,), (0,))),                      # C1 空 Tensor
    ("L2-02", _rej_shape((0, 4), (0, 4))),                  # C1 空 Tensor（多维 0 轴）
    ("L2-03", _rej_shape((100,), (200,))),                  # C2 shape 不一致
    ("L2-04", _rej_shape((4, 25), (100,))),                 # C2 同 numel 异 shape（本仓收窄项）
    ("L2-05", _rej_shape((100,), (100,), F32, F16)),        # C2 dtype 不一致
    ("L2-06", _rej_x_dtype(torch.float64)),                 # C1 非法 dtype
    ("L2-07", _rej_x_dtype(torch.int32)),
    ("L2-08", _rej_x_dtype(torch.int64)),
    ("L2-09", _rej_x_dtype(torch.complex64)),
    ("L2-10", _rej_scalar(wd_dtype=F16)),                   # C3 标量 dtype 非 fp32
    ("L2-11", _rej_scalar(trust_dtype=torch.float64)),
    ("L2-12", _rej_scalar(lr_max_dtype=torch.int32)),
    ("L2-13", _rej_scalar(wd_shape=(2,))),                  # C3 标量 numel != 1
    ("L2-14", _rej_scalar(trust_shape=(2, 3))),
    ("L2-15", _rej_attr(-1e-6, 0.02)),                      # C4 offset < 0（最小负值）
    ("L2-16", _rej_attr(0.5, -0.01)),                       # C5 lr_min < 0
    ("L2-17", _rej_attr(-0.1, -0.1)),                       # C4 + C5 同时为负
    ("L2-18", _rej_x_on_cpu),                               # C1 X 非 npu device
    ("L2-19", _rej_scalar(trust_on_cpu=True)),              # C3 的 device 子句
]


@requires_npu
@pytest.mark.parametrize("cid,factory", L2_REJECT, ids=[c for c, _ in L2_REJECT])
def test_c2_lars_l2_reject(cid, factory):
    with pytest.raises(RuntimeError):
        factory()
    # 后置设备健康校验：host 侧 TORCH_CHECK 拦截不得污染设备上下文。
    _device_health_check()


# L2-20 非连续输入：当前 CANN 版本缺 D2D strided copy (error 561103)，调用方自行 contiguous()。
# 保留用例，CANN 版本升级后应复检并解除 skip（方案 §8 T3）。
@pytest.mark.skip(reason="CANN lacks D2D strided copy (error 561103); caller must .contiguous()")
@requires_npu
def test_c2_lars_l2_20_non_contiguous():
    X = _npu_x((64, 32)).t()
    dX = _npu_x((64, 32)).t()
    wd, trust, lr_max = _npu_scalars()
    out = _op_raw(X, dX, wd, trust, lr_max, 0.5, 0.02)
    _assert_contract(out, "L2-20")
