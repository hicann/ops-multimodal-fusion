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
# c2_lars white-box (source-derived) test supplement —— 阶段 3.3。
#
# 与 `test_c2_lars.py` 的分工：那一份实现 2.1 冻结的黑盒 L0/L1/L2 矩阵（213 例），用例由**接口契约**
# 反推；本文件的用例由 `applications/llm/c2_lars/arch35/c2_lars.asc` 的**源码执行分支**反推，shape /
# dtype / 标量参数都是为了打到某一条具体分支而解出来的。两份文件互不 import、各自自包含。
#
# 分支编号与覆盖论证见 `.cannbot/C2_LARS/3.3-白盒分支覆盖说明.md`。
#
# 五组：
#   W1  stage-2 归约 chunk tail mask（R4′，2.2 §6.2 的 M1 修复点）—— 本文件的承重组
#   W2  stage-1 tile 循环边界（myLen 相对 tileElems=8192 的 -1 / = / +1，以及多 tile）
#   W3  laneAlign 边界（fp32 8、b16 16 的 -1 / = / +1，DataCopyPad rightPadding）
#   W4  非末核的 chunk tail（blockLength % 64 != 0）
#   W5  epilogue 的 Select / Maxs 分支，且与 `numBlocks % 8 != 0` 叠加
#
# 执行（须在仓库目录之外，否则源码目录 `ops_multimodal_fusion/` 会遮蔽已安装 wheel）：
#     cd /tmp && pytest /workspace/asc-ops/ops-multimodal-fusion-tpc/tests/c2_lars/ -v
#     cd /tmp && pytest .../tests/c2_lars/test_c2_lars_whitebox.py -v -k "w1_"
#
# 判据（承接 2.1 §6，本文件不新立口径）：
#   * 数值比对一律 `torch.allclose`，禁止 `torch.equal` / `==`；容差按**输入 dtype** 取 2.1 §2.3 定稿值
#     fp32 (1e-4, 1e-5) / fp16 (1e-3, 1e-3) / bf16 (2e-4, 1e-8)，本文件不放宽。
#   * golden 为 2.1 §2.2 的五步 fp64 直算，消费的是送入 NPU 的同一个量化后张量；host 属性
#     offset / lr_min 先经 fp32 再进 fp64。
#   * 结构 / 谓词断言（dtype / dim / numel / isnan）不受禁位精确比对条款约束。
#
# 环境限制：本机 CANN 缺 npu 侧 ZerosLike（error 561103），device 标量一律先在 CPU 构造再 `.npu()`。

import math

import pytest
import torch
import torch_npu  # noqa: F401

import ops_multimodal_fusion  # noqa: F401

# ── module-level NPU_ARCH guard ────────────────────────────────────────────
if not hasattr(torch.ops.ops_multimodal_fusion, "c2_lars"):
    pytest.skip(
        "ops_multimodal_fusion.c2_lars not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )

requires_npu = pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")

F32, F16, BF16 = torch.float32, torch.float16, torch.bfloat16
F64 = torch.float64
DTYPES = [F32, F16, BF16]
_DN = {F32: "fp32", F16: "fp16", BF16: "bf16"}

# 容差 = 2.1 §2.3 定稿值，按输入 dtype 取档。设计期定值，本阶段不得放宽。
_TOL = {
    F32: (1e-4, 1e-5),
    F16: (1e-3, 1e-3),
    BF16: (2e-4, 1e-8),
}

# ── 源码常量镜像（applications/llm/c2_lars/arch35/c2_lars.asc）─────────────
# 用例的 shape 由这些常量解出，改动源码常量时本文件的 numel 需一并复核。
_MIN_ELEMS_PER_CORE = 2048      # c2_lars.asc: MIN_ELEMS_PER_CORE
_MAX_TILE_ELEMS = 8192          # c2_lars.asc: MAX_TILE_ELEMS，本 SKU 上 tileElems 恒取该值
_VL_F32 = 64                    # c2_lars.asc: VL_F32，vf 的 chunk 粒度
_WS_SLOT_FLOATS = 8             # c2_lars.asc: WS_SLOT_FLOATS，每核 32B workspace 槽

# GetCoreNumAiv()。numel -> numBlocks 的映射依赖它，故运行期取值而非硬编码（R2）。
_CORE_NUM = 0
if torch.npu.is_available():
    _CORE_NUM = int(getattr(torch.npu.get_device_properties(0), "vector_core_num", 0) or 0)


def _num_blocks(numel):
    """host tiling 的 numBlocks 求解式（c2_lars.asc calc_tiling_params）。"""
    return min(_CORE_NUM, max(1, -(-numel // _MIN_ELEMS_PER_CORE)))


def _numel_for_nb(k):
    """最小的、使 numBlocks 恰为 k 的整齐 numel（k <= coreNum 时成立）。"""
    return k * _MIN_ELEMS_PER_CORE


# ═══════════════════════════════════════════════════════════════════════════
# Golden —— 2.1 §2.2 的五步 fp64 直算
# ═══════════════════════════════════════════════════════════════════════════
def _host_attr_f64(v):
    """host 属性先经 fp32 再进 fp64：kernel 内 offset / lr_min 按 fp32 参与运算。"""
    return torch.tensor(float(v), dtype=F32).to(F64)


def _golden(X, dX, wd, trust, lr_max, offset, lr_min):
    """Return the 0-D fp32 golden for the same quantised tensors that go to the NPU."""
    x64 = X.to(F64)
    xn = torch.sqrt(torch.sum(x64 * x64))
    del x64
    d64 = dX.to(F64)
    dn = torch.sqrt(torch.sum(d64 * d64))
    del d64

    wd64 = wd.to(F64).reshape(())
    trust64 = trust.to(F64).reshape(())
    lrmax64 = lr_max.to(F64).reshape(())
    off64 = _host_attr_f64(offset)
    lmin64 = _host_attr_f64(lr_min)

    if bool(xn > 0):
        val = trust64 / (dn / xn + wd64 + off64)
    else:
        val = torch.tensor(1.0, dtype=F64)
    out = torch.maximum(torch.minimum(val, lrmax64), lmin64)
    return xn, dn, val, out.to(F32).reshape(())


def _scalar(v):
    """wd / trust / lr_max：恒 fp32。CPU 构造再 .npu()（CANN 缺 npu ZerosLike，561103）。"""
    return torch.tensor(float(v), dtype=F32)


def _assert_contract(out, cid):
    """C7 结构契约。结构断言，不受禁位精确比对条款约束。"""
    assert out.dtype == F32, f"{cid}: out.dtype {out.dtype} != torch.float32 (C7)"
    assert out.dim() == 0, f"{cid}: out.dim() {out.dim()} != 0 (C7)"
    assert out.numel() == 1, f"{cid}: out.numel() {out.numel()} != 1 (C7)"


def _check(cid, dtype, actual, expected, extra=""):
    rtol, atol = _TOL[dtype]
    if bool(torch.isnan(expected)):
        assert bool(torch.isnan(actual)), \
            f"{cid}: golden is NaN but actual={actual.item()!r} (NaN 传播缺失){extra}"
        return
    a, e = actual.double().item(), expected.double().item()
    rel = abs(a - e) / abs(e) if e != 0 else float("inf")
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"{cid}: precision FAIL dtype={_DN[dtype]}\n"
        f"  actual={a!r} expected={e!r}\n"
        f"  abs={abs(a - e):.6e} rel={rel:.6e} (rtol={rtol:.1e} atol={atol:.1e}){extra}")


# ═══════════════════════════════════════════════════════════════════════════
# 数据构造
# ═══════════════════════════════════════════════════════════════════════════
def _const(numel, value, dtype):
    """常量张量：白盒残留分析要求 UB 中每个 lane 的值已知，故 W1 一律用常量。"""
    return torch.full((numel,), float(value), dtype=F64).to(dtype)


def _ones_with_sentinel(numel, dtype, where):
    """L1-F 同款哨兵构造（2.1 §3.5）：ones + 末元素 2^11，打破 X / dX 的对称。

    三种 dtype 均精确可表；含哨兵的 64 元素 chunk 求和 63 + 2^22 = 4194367 < 2^24，
    累加阶段零舍入，残差只剩 epilogue 的 ulp。返回 (X, dX)。
    """
    x = torch.ones(numel, dtype=F64)
    d = torch.ones(numel, dtype=F64)
    (x if where == "x" else d)[-1] = 2048.0
    return x.to(dtype), d.to(dtype)


# ═══════════════════════════════════════════════════════════════════════════
# 调用
# ═══════════════════════════════════════════════════════════════════════════
_OP = torch.ops.ops_multimodal_fusion.c2_lars


def _call(X, dX, wd, trust, lr_max, offset, lr_min):
    return _OP(X.npu(), dX.npu(), wd.npu(), trust.npu(), lr_max.npu(), offset, lr_min)


# ── W1 专用：TBuf 污染前置调用 ──────────────────────────────────────────────
#
# stage 2 的 `wsXBuf_` / `wsDBuf_` 是 `TBuf`，`wsAlloc = ceil(wsLen/64)*64` 向上取整到整寄存器，
# 而 `CopyInRow` 只写入 `wsLen = numBlocks*8` 个 float。`numBlocks % 8 != 0` 时，末 chunk 中
# [wsLen, wsAlloc) 这段 lane 是 **UB 里的陈旧内容**，缺 tail mask 即进入求和。
#
# UB 陈旧内容有两个来源，本文件两条都用上（互为冗余，任一条成立即可让缺陷显形）：
#   (a) 自污染 —— 同一次调用的 stage-1 kernel 先于 stage-2 在同一块 UB 上跑，其 `inQueX_` 就落在
#       `wsXBuf_` / `wsDBuf_` 之上，残留即 X 的 tile 数据。令 X 为**小常量** c 时，残留贡献
#       ~ m·c 而正确和为 numel·c²，只要 c 足够小，残留就压倒正确值。
#       fp32 / bf16 有效（bf16 的两个相邻 lane 重解释为 fp32 后仍约等于 c）；fp16 重解释后落到
#       ~1e-27，自污染对 fp16 无效，故 fp16 走 (b)。
#   (b) 前置污染 —— 先跑一次 `numBlocks % 8 == 0`（自身不触发本分支）的大常量调用，让 UB 里留下
#       量级 ~1e7 的 stage-2 workspace 内容；随后目标调用取**极小 numel**，其 stage-1 只写 32B，
#       够不到残留 lane，于是前一次的大值原样进入末 chunk。
#
# 两次调用必须紧邻且中间无其它 kernel，故写在同一个用例函数内；所有 H2D 都在两次调用之前完成。
_POL_NUMEL = 65536      # numBlocks = 32（%8 == 0），污染调用自身不触发被测分支
_POL_CONST = 100.0      # 每核 partial = 2048 * 1e4 ≈ 2.05e7，残留量级远超目标调用的正确和


def _pollute_then(target_call):
    """先跑污染调用（不取回结果，避免插入额外 device 动作），再跑目标调用。"""
    px = _const(_POL_NUMEL, _POL_CONST, F32).npu()
    pd = _const(_POL_NUMEL, _POL_CONST, F32).npu()
    pwd, ptrust, plrmax = _scalar(0.0).npu(), _scalar(1.0).npu(), _scalar(1e30).npu()
    prepared = target_call()            # 目标张量的 H2D 全部先做完
    _pol = _OP(px, pd, pwd, ptrust, plrmax, 0.0, 0.0)
    out = prepared()
    float(_pol.cpu())                   # 污染调用的结果不参与判定，仅确保它真的被执行
    return out


# ═══════════════════════════════════════════════════════════════════════════
# W1 —— stage-2 归约 chunk tail mask（R4′）
#
#   wsLen  = numBlocks * 8            有效 float 数
#   wsAlloc= ceil(wsLen/64) * 64      TBuf 分配长度
#   末 chunk 的越界 lane 数 = wsAlloc - wsLen = 8 * (8 - numBlocks % 8)   （%8 != 0 时）
#
# 判定：`numBlocks % 8 != 0` 的用例在缺 mask 的实现下必然偏离 golden（实测见覆盖说明 §4）；
#       `numBlocks % 8 == 0` 的控制组两种实现都应通过，用于定位失败究竟来自本分支还是别处。
# ═══════════════════════════════════════════════════════════════════════════

# 目标调用的标量取 trust=1 / wd=0 / offset=0 / lr_max=1e30 / lr_min=0，
# 于是 out == X_norm / dX_norm，输出对两个平方和的**任何**污染都最大程度敏感（不被 clamp 吸收）。
_W1_WD, _W1_TRUST, _W1_LRMAX = 0.0, 1.0, 1e30
_W1_OFFSET, _W1_LRMIN = 0.0, 0.0
# 自污染档的常量：X 取 1e-4 使残留(~m·1e-4)压过正确和(numel·1e-8)；dX 取 1e-2 制造 100:1 的
# 不对称，避免两侧被同量污染后在比值中抵消。golden out == 1e-4/1e-2 == 1e-2，与 numel 无关。
_W1_CX, _W1_CD = 1e-4, 1e-2

_W1_TAIL_NB = [1, 2, 3, 7, 9, 49, 55]        # numBlocks % 8 != 0 —— 被测分支
_W1_CTRL_NB = [8, 16]                        # numBlocks % 8 == 0 —— 控制组


def _w1_run(cid, numel, dtype, cx, cd, wd, trust, lr_max, offset, lr_min):
    X, dX = _const(numel, cx, dtype), _const(numel, cd, dtype)
    wd_t, trust_t, lrmax_t = _scalar(wd), _scalar(trust), _scalar(lr_max)
    _, _, _, expected = _golden(X, dX, wd_t, trust_t, lrmax_t, offset, lr_min)

    def prepare():
        xn, dn = X.npu(), dX.npu()
        w, t, m = wd_t.npu(), trust_t.npu(), lrmax_t.npu()
        return lambda: _OP(xn, dn, w, t, m, offset, lr_min)

    out = _pollute_then(prepare)
    _assert_contract(out, cid)
    nb = _num_blocks(numel)
    extra = (f"\n  numel={numel} numBlocks={nb} nb%8={nb % 8} "
             f"wsLen={nb * _WS_SLOT_FLOATS} wsAlloc={-(-nb * _WS_SLOT_FLOATS // _VL_F32) * _VL_F32}")
    _check(cid, dtype, out.cpu(), expected, extra)
    del X, dX, out
    torch.npu.empty_cache()


_W1A = [(k, dt) for k in _W1_TAIL_NB for dt in (F32, BF16)]


@requires_npu
@pytest.mark.parametrize("nb,dtype", _W1A,
                         ids=[f"W1-A-selfpol-nb{k}-{_DN[d]}" for k, d in _W1A])
def test_c2_lars_wb_w1_a_stage2_tail_mask_selfpol(nb, dtype):
    """W1-A：自污染档。stage-1 把小常量 X 写进 UB，正是 stage-2 末 chunk 越界 lane 所在。"""
    if _CORE_NUM < nb:
        pytest.skip(f"needs GetCoreNumAiv() >= {nb}, got {_CORE_NUM}")
    numel = _numel_for_nb(nb)
    assert _num_blocks(numel) == nb
    _w1_run(f"W1-A-selfpol-nb{nb}-{_DN[dtype]}", numel, dtype, _W1_CX, _W1_CD,
            _W1_WD, _W1_TRUST, _W1_LRMAX, _W1_OFFSET, _W1_LRMIN)


_W1B = [(n, dt) for n in (1, 16) for dt in DTYPES]


@requires_npu
@pytest.mark.parametrize("numel,dtype", _W1B,
                         ids=[f"W1-B-prepol-n{n}-{_DN[d]}" for n, d in _W1B])
def test_c2_lars_wb_w1_b_stage2_tail_mask_prepol(numel, dtype):
    """W1-B：前置污染档。numel 极小 -> stage-1 只写 32B，末 chunk 的越界 lane 保留前一次调用的
    大值。这是 fp16 唯一可用的构造（fp16 的自污染重解释后量级 ~1e-27，不足以显形）。"""
    assert _num_blocks(numel) == 1
    _w1_run(f"W1-B-prepol-n{numel}-{_DN[dtype]}", numel, dtype, _W1_CX, _W1_CD,
            _W1_WD, _W1_TRUST, _W1_LRMAX, _W1_OFFSET, _W1_LRMIN)


@requires_npu
@pytest.mark.parametrize("dtype", DTYPES, ids=[f"W1-C-xzero-{_DN[d]}" for d in DTYPES])
def test_c2_lars_wb_w1_c_tail_mask_flips_xnorm_branch(dtype):
    """W1-C：X 全零 + numBlocks%8!=0。正确实现 X_norm == 0 走 val=1.0（epilogue B-S8）；
    缺 tail mask 时残留使 X_norm > 0，分支被翻到除法侧 —— 同一条用例同时锚住两个分支。"""
    _w1_run(f"W1-C-xzero-{_DN[dtype]}", 1, dtype, 0.0, 1.0,
            wd=0.1, trust=0.5, lr_max=10.0, offset=0.5, lr_min=0.02)


_W1D = [(k, dt) for k in _W1_CTRL_NB for dt in (F32, BF16)]


@requires_npu
@pytest.mark.parametrize("nb,dtype", _W1D,
                         ids=[f"W1-D-ctrl-nb{k}-{_DN[d]}" for k, d in _W1D])
def test_c2_lars_wb_w1_d_stage2_no_tail_control(nb, dtype):
    """W1-D：控制组，numBlocks % 8 == 0 -> wsAlloc == wsLen，末 chunk 无越界 lane。
    该组在「有 mask」「无 mask」两种实现下都应通过，用来把 W1-A/B/C 的失败定位到 tail mask。"""
    if _CORE_NUM < nb:
        pytest.skip(f"needs GetCoreNumAiv() >= {nb}, got {_CORE_NUM}")
    numel = _numel_for_nb(nb)
    assert _num_blocks(numel) == nb and nb % 8 == 0
    _w1_run(f"W1-D-ctrl-nb{nb}-{_DN[dtype]}", numel, dtype, _W1_CX, _W1_CD,
            _W1_WD, _W1_TRUST, _W1_LRMAX, _W1_OFFSET, _W1_LRMIN)


@requires_npu
@pytest.mark.parametrize("dtype", (F32, BF16), ids=[f"W1-E-fullgrid-{_DN[d]}" for d in (F32, BF16)])
def test_c2_lars_wb_w1_e_full_grid(dtype):
    """W1-E：满核（numBlocks == GetCoreNumAiv()）。本 SKU 为 56（%8 == 0，落控制侧），
    在 coreNum % 8 != 0 的 SKU 上会自动变成被测侧 —— 两种情形下判据相同。"""
    numel = 262144
    assert _num_blocks(numel) == _CORE_NUM
    _w1_run(f"W1-E-fullgrid-{_DN[dtype]}", numel, dtype, _W1_CX, _W1_CD,
            _W1_WD, _W1_TRUST, _W1_LRMAX, _W1_OFFSET, _W1_LRMIN)


# ═══════════════════════════════════════════════════════════════════════════
# W2/W3/W4 —— stage-1 切分路径
#
# 判据用 L1-F 同款哨兵（X=ones，dX=ones 且 dX[-1]=2^11）：X 与 dX 同形状同 tiling，任何
# 尾块 / tile / chunk 层面的漏算或重复算若在两侧等比例发生，会在 dX_norm/X_norm 中抵消；
# 哨兵刻意打破这一对称，使这类缺陷落在比值上而非被约掉（2.1 §3.5）。
# 标量取 trust=1 / wd=0 / offset=0，输出即 X_norm/dX_norm，无 clamp 吸收。
# ═══════════════════════════════════════════════════════════════════════════
def _wb_tiling_run(cid, numel, dtype, where="dx"):
    X, dX = _ones_with_sentinel(numel, dtype, where)
    wd, trust, lr_max = _scalar(0.0), _scalar(1.0), _scalar(1e30)
    _, _, _, expected = _golden(X, dX, wd, trust, lr_max, 0.0, 0.0)
    out = _call(X, dX, wd, trust, lr_max, 0.0, 0.0)
    _assert_contract(out, cid)
    nb = _num_blocks(numel)
    la = 32 // (4 if dtype is F32 else 2)
    units = -(-numel // la)
    bl = -(-units // nb) * la
    my_last = numel - bl * (nb - 1)
    extra = (f"\n  numel={numel} numBlocks={nb} laneAlign={la} blockLength={bl} "
             f"lastCoreLen={my_last} tiles/core={-(-bl // _MAX_TILE_ELEMS)} "
             f"blockLength%{_VL_F32}={bl % _VL_F32} sentinel_on={where}")
    _check(cid, dtype, out.cpu(), expected, extra)
    del X, dX, out
    torch.npu.empty_cache()


# ── W2：stage-1 tile 循环边界 ───────────────────────────────────────────────
# tileElems 恒为 MAX_TILE_ELEMS = 8192，每核 myLen = blockLength（末核可能更短）。
# blockLength = ceil(ceil(n/laneAlign)/coreNum)*laneAlign，故取 n = coreNum * B 即可把 blockLength
# 精确定在 B 上（B 为 laneAlign 的整数倍时对 fp32 / b16 同时成立）。B 相对 8192 的三个位置：
#   B = 8192   -> 单 tile，末 tile count == tileElems（循环恰好 1 次，且 count % 64 == 0）
#   B = 8208   -> 2 tile（8192 + 16），末 tile count << tileElems
#   B = 17872  -> 3 tile（8192 + 8192 + 1488）
# 三个 B 都取 16 的整数倍，使 fp32（laneAlign 8）与 b16（laneAlign 16）解出同一个 blockLength。
# 再加 coreNum*8192 - 1：blockLength 仍为 8192，但末核 myLen = 8191，制造 count % 64 == 63 的
# chunk tail 与 rightPadding = 1 的非对齐尾。
_W2_BLOCKLEN = [8192, 8208, 17872]
_W2_NUMELS = ([_CORE_NUM * b for b in _W2_BLOCKLEN] + [_CORE_NUM * 8192 - 1]) if _CORE_NUM else []
_W2 = [(n, dt) for n in _W2_NUMELS for dt in (F32, BF16)]


@requires_npu
@pytest.mark.parametrize("numel,dtype", _W2, ids=[f"W2-tile-{n}-{_DN[d]}" for n, d in _W2])
def test_c2_lars_wb_w2_stage1_tile_loop(numel, dtype):
    """W2：stage-1 `for (off = 0; off < myLen_; off += tileElems_)` 的 1 / 2 / 3 次迭代，
    以及末 tile `count < tileElems` 与 `count == tileElems` 两个出口。"""
    _wb_tiling_run(f"W2-tile-{numel}-{_DN[dtype]}", numel, dtype)


# ── W3：laneAlign 边界（DataCopyPad rightPadding）──────────────────────────
# CopyTile: padded = ceil(count/laneAlign)*laneAlign，rightPadding = padded - count。
# fp32 laneAlign = 8、fp16/bf16 laneAlign = 16，各取 -1 / = / +1 三点。
_W3_NUMELS = [7, 8, 9, 15, 16, 17]
_W3 = [(n, dt) for n in _W3_NUMELS for dt in DTYPES]


@requires_npu
@pytest.mark.parametrize("numel,dtype", _W3, ids=[f"W3-lane-{n}-{_DN[d]}" for n, d in _W3])
def test_c2_lars_wb_w3_lane_align_boundary(numel, dtype):
    """W3：rightPadding == 0（对齐）与 > 0（非对齐）两条出口，且 chunkCount == 1、
    末 chunk remaining < 64 —— stage-1 vf 的最短路径。"""
    _wb_tiling_run(f"W3-lane-{numel}-{_DN[dtype]}", numel, dtype)


@requires_npu
@pytest.mark.parametrize("numel,dtype", [(n, d) for n in (7, 16) for d in DTYPES],
                         ids=[f"W3-mirror-{n}-{_DN[d]}" for n in (7, 16) for d in DTYPES])
def test_c2_lars_wb_w3_lane_align_mirror(numel, dtype):
    """W3 镜像：哨兵放在 X 侧，使 X 路径的尾块同样受检（dX 侧为纯 ones）。"""
    _wb_tiling_run(f"W3-mirror-{numel}-{_DN[dtype]}", numel, dtype, where="x")


# ── W4：非末核的 chunk tail（blockLength % 64 != 0）────────────────────────
# W2/W3 里出现 remaining < 64 的都是**末核**的末 tile。blockLength 本身不是 64 的整数倍时，
# **每个核**的末 chunk 都带 tail mask —— 这是 stage-1 `LarsSqSumPairVf` 的另一条进入方式。
_W4_NUMELS = [
    6145,       # fp32: numBlocks=4, blockLength=1544 (%64 == 8)；bf16: blockLength=1552 (%64 == 16)
    98305,      # fp32: numBlocks=49, blockLength=2008 (%64 == 24)；bf16: blockLength=2016 (%64 == 32)
]
_W4 = [(n, dt) for n in _W4_NUMELS for dt in (F32, BF16)]


@requires_npu
@pytest.mark.parametrize("numel,dtype", _W4, ids=[f"W4-chunktail-{n}-{_DN[d]}" for n, d in _W4])
def test_c2_lars_wb_w4_non_last_core_chunk_tail(numel, dtype):
    """W4：blockLength % 64 != 0 -> 非末核也走 `remaining < 64` 的 chunk tail mask。"""
    nb = _num_blocks(numel)
    la = 32 // (4 if dtype is F32 else 2)
    bl = -(-(-(-numel // la)) // nb) * la
    assert bl % _VL_F32 != 0, f"numel={numel} {_DN[dtype]}: blockLength={bl} 未命中 %64 != 0"
    _wb_tiling_run(f"W4-chunktail-{numel}-{_DN[dtype]}", numel, dtype)


# ═══════════════════════════════════════════════════════════════════════════
# W5 —— epilogue 的 Select / Maxs 分支，与 numBlocks % 8 != 0 叠加
#
# 分支来自 c2_lars.asc LarsEpilogueVf：
#   Compares<GT>(xNorm, 0) + Select   -> val 侧 / 1.0 侧
#   Compare<GT>(val, lr_max) + Select -> lr_max 侧 / val 侧
#   Maxs(·, lr_min)                   -> lr_min 侧 / 保持
#   Compare<EQ>(val, val) + Select    -> NaN 回填侧 / 保持
# 这些分支 2.1 的 L1-D / L1-S 已在 numBlocks ∈ {1} 上覆盖；本组把它们挪到 numBlocks = 3
# （%8 != 0，末 chunk 带 40 个越界 lane）上重跑，检查两处分支的叠加没有互相掩盖。
# ═══════════════════════════════════════════════════════════════════════════
_W5_NUMEL = 6144        # numBlocks = 3 -> wsLen = 24, wsAlloc = 64, 40 个越界 lane

_W5_CASES = [
    # cid,          xfam,   dfam,   wd,   trust, lr_max, offset, lr_min
    ("W5-val", "ones", "const", 0.1, 0.5, 10.0, 0.5, 0.0),      # 落 val（无 clamp）
    ("W5-lrmax", "ones", "const", 0.1, 0.5, 1e-3, 0.5, 0.0),      # val > lr_max -> lr_max
    ("W5-lrmin", "ones", "const", 0.1, 0.5, 10.0, 0.5, 0.9),      # val < lr_min -> lr_min
    ("W5-xzero", "zeros", "const", 0.1, 0.5, 10.0, 0.5, 0.02),     # X_norm == 0 -> val = 1.0
    ("W5-inf", "ones", "zeros", 0.0, 1.0, 10.0, 0.0, 0.02),     # den == 0, trust>0 -> +inf -> lr_max
    ("W5-nan", "ones", "zeros", 0.0, 0.0, 10.0, 0.0, 0.02),     # 0/0 -> NaN -> 第 10 步回填
]


@requires_npu
@pytest.mark.parametrize("case", _W5_CASES, ids=[c[0] for c in _W5_CASES])
def test_c2_lars_wb_w5_epilogue_branch_with_stage2_tail(case):
    cid, xfam, dfam, wd_v, trust_v, lrmax_v, offset, lr_min = case
    numel = _W5_NUMEL
    assert _num_blocks(numel) % 8 != 0, "W5 需要 numBlocks % 8 != 0 才能叠加 stage-2 尾块分支"
    X = torch.ones(numel, dtype=F64) if xfam == "ones" else torch.zeros(numel, dtype=F64)
    dX = torch.zeros(numel, dtype=F64) if dfam == "zeros" else torch.full((numel,), 0.25, dtype=F64)
    X, dX = X.to(F32), dX.to(F32)
    wd, trust, lr_max = _scalar(wd_v), _scalar(trust_v), _scalar(lrmax_v)
    xn, dn, val, expected = _golden(X, dX, wd, trust, lr_max, offset, lr_min)
    out = _call(X, dX, wd, trust, lr_max, offset, lr_min)
    _assert_contract(out, cid)
    extra = (f"\n  numel={numel} numBlocks={_num_blocks(numel)} "
             f"X_norm={xn.item():.9e} dX_norm={dn.item():.9e} val={val.item():.9e}")
    _check(cid, F32, out.cpu(), expected, extra)
    del X, dX, out
    torch.npu.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
# W7 —— Meta 分派键（`TORCH_LIBRARY_IMPL(..., Meta, m)` -> `c2_lars_meta`）
#
# 白盒发现的覆盖空洞：`c2_lars_npu` **不调用** `c2_lars_meta`（它自己 `torch::empty({})` 造输出），
# 所以 Meta 实现只在 meta 设备 / `torch.compile` / autograd 形状推断时才被执行。2.1 的 213 例
# 全部走 PrivateUse1，Meta 分支与 `c2_lars_check(..., checkDevice=false)` 这条出口此前 0 覆盖。
# 本组不需要 NPU 设备。
# ═══════════════════════════════════════════════════════════════════════════
def _meta(shape, dtype=F32):
    return torch.empty(shape, dtype=dtype, device="meta")


@pytest.mark.parametrize("dtype", DTYPES, ids=[f"W7-meta-{_DN[d]}" for d in DTYPES])
def test_c2_lars_wb_w7_meta_shape_inference(dtype):
    """W7-A：Meta 键的形状 / dtype 推断 —— 输出恒 0-D fp32（C7），与输入 dtype 解耦。"""
    n = 100
    out = _OP(_meta((n,), dtype), _meta((n,), dtype), _meta(()), _meta(()), _meta(()), 0.5, 0.02)
    _assert_contract(out, f"W7-meta-{_DN[dtype]}")
    assert out.device.type == "meta", f"W7-meta: out.device {out.device} 不是 meta"


@pytest.mark.parametrize("shape", [(4, 25), (2, 3, 4, 5)], ids=["W7-meta-rank2", "W7-meta-rank4"])
def test_c2_lars_wb_w7_meta_rank(shape):
    """W7-B：多维输入下 Meta 仍归约为 0-D（`torch::empty({})` 而非 `empty_like`）。"""
    out = _OP(_meta(shape), _meta(shape), _meta(()), _meta(()), _meta(()), 0.5, 0.02)
    _assert_contract(out, f"W7-meta-rank{len(shape)}")


_W7_REJECT = [
    ("W7-meta-rej-empty", lambda: _OP(_meta((0,)), _meta((0,)), _meta(()), _meta(()),
                                      _meta(()), 0.5, 0.02)),                              # C1
    ("W7-meta-rej-dtype", lambda: _OP(_meta((8,), torch.float64), _meta((8,), torch.float64),
                                      _meta(()), _meta(()), _meta(()), 0.5, 0.02)),        # C1
    ("W7-meta-rej-shape", lambda: _OP(_meta((8,)), _meta((9,)), _meta(()), _meta(()),
                                      _meta(()), 0.5, 0.02)),                              # C2
    ("W7-meta-rej-scalar", lambda: _OP(_meta((8,)), _meta((8,)), _meta((2,)), _meta(()),
                                       _meta(()), 0.5, 0.02)),                             # C3
    ("W7-meta-rej-offset", lambda: _OP(_meta((8,)), _meta((8,)), _meta(()), _meta(()),
                                       _meta(()), -1e-6, 0.02)),                           # C4
    ("W7-meta-rej-lrmin", lambda: _OP(_meta((8,)), _meta((8,)), _meta(()), _meta(()),
                                      _meta(()), 0.5, -0.01)),                             # C5
]


@pytest.mark.parametrize("cid,factory", _W7_REJECT, ids=[c for c, _ in _W7_REJECT])
def test_c2_lars_wb_w7_meta_reject(cid, factory):
    """W7-C：`checkDevice == false` 这条出口仍必须执行 C1/C2/C3/C4/C5 —— 校验不能只挂在 NPU 侧，
    否则 torch.compile 的形状推断会先于设备执行悄悄放过非法入参。"""
    with pytest.raises(RuntimeError, match="c2_lars"):
        factory()


# ═══════════════════════════════════════════════════════════════════════════
# W6 —— tiling 求解式自洽（host 分支，无 kernel 判据）
#   本用例不比对数值，只锚住「本文件全部 numel 确实落在它们声称的 numBlocks 档位上」。
#   源码 MIN_ELEMS_PER_CORE / MAX_TILE_ELEMS 若被改动，本用例先失败，避免上面各组静默失去覆盖。
# ═══════════════════════════════════════════════════════════════════════════
@requires_npu
def test_c2_lars_wb_w6_tiling_assumptions():
    assert _CORE_NUM > 0, "GetCoreNumAiv() 未能从 device properties 取得，白盒 numel 推导失效"
    for k in _W1_TAIL_NB + _W1_CTRL_NB:
        if k > _CORE_NUM:
            continue
        n = _numel_for_nb(k)
        assert _num_blocks(n) == k, f"numel={n} 期望 numBlocks={k}，实得 {_num_blocks(n)}"
        assert (k % 8 != 0) == (k in _W1_TAIL_NB), f"nb={k} 的 %8 归组与用例分组不一致"
    assert _num_blocks(262144) == _CORE_NUM, "262144 应使 numBlocks 饱和到 coreNum"
    # W2 的 tile 边界依赖 blockLength 跨过 MAX_TILE_ELEMS
    for b, exp_tiles in zip(_W2_BLOCKLEN, (1, 2, 3)):
        numel = _CORE_NUM * b
        for la in (8, 16):
            nb = _num_blocks(numel)
            bl = -(-(-(-numel // la)) // nb) * la
            assert bl == b, f"numel={numel} laneAlign={la}: blockLength={bl}，期望 {b}"
            assert math.ceil(bl / _MAX_TILE_ELEMS) == exp_tiles, \
                f"numel={numel}: blockLength={bl} -> {math.ceil(bl / _MAX_TILE_ELEMS)} tile，期望 {exp_tiles}"
