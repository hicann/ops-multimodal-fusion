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
# c2_lars 上板性能采集框架 (直调算子，自包含)。
#
# 需求 §2.9：无硬性指标，要求归约实现合理、不成为明显瓶颈，CP4 以瓶颈分析结论判定。
# 需求 §3.2：计算量极小 (每元素 1 次乘加)，瓶颈在 MTE2 带宽 —— 需完整读入 X 与 dX 两个张量，
#            理论下界即 2 × numel × sizeof(dtype) 的搬入耗时。关键在多核切分是否充分利用带宽。
# 故本框架以 **kernel 平均时延 + 达成读带宽 (GB/s)** 为主指标，并给出「达成 / 峰值」占比供 CP4 判读。
#
# 单卡共享设备 (需求 §2.9 采集约束 / §3.3)：采集前请先确认无其他占用进程
#   timeout 15 npu-smi info    # 期望 "No running processes"（本机 npu-smi 可能挂起，非可靠健康信号）
# 并对同一配置多次采样、标注采样条件 (本框架已内置 warm-up + 多轮 + 中位数/最小值/标准差)。
#
# 运行 (必须在源码目录之外，避免 ops_multimodal_fusion/ 源目录遮蔽已安装 wheel，需求 §4)：
#   cd /tmp && python3 /workspace/asc-ops/ops-multimodal-fusion-tpc/tests/c2_lars/perf_c2_lars.py
# 可选参数：
#   --warm-up N     每个配置的预热次数 (规避 DVFS)，默认 5
#   --iters N       每个配置的计时轮数，默认 30
#   --out PATH      结果 JSON 输出路径，默认 <repo>/.cannbot/perf/c2_lars_perf.json
#   --dtypes ...    逗号分隔 dtype 子集 (float32,float16,bfloat16)，默认全部
#   --numels ...    逗号分隔 numel 子集，默认见 _NUMELS
#   --peak-gbps X   设备 HBM 峰值读带宽 (GB/s)，用于算达成率；不传则只报绝对带宽
#   --quick         只跑小规模冒烟 (调试用)
#
# 深度瓶颈分析 (CP4 按需，本框架不替代 msprof；见 ops-profiling skill)：
#   msprof --application="python3 .../perf_c2_lars.py --quick" --aic-metrics=... --output=./prof_out
#   python3 .claude/skills/ops-profiling/scripts/msprof_perf_summary.py ./prof_out/PROF_* <ops_dir>
# 归约类算子重点看：aiv_mte2_ratio (搬入占比，本算子应接近 1)、aiv_vec_ratio、多核 Task Duration
# 离散度 (反映切分是否均衡)、以及 numel 小档位下的固定启动成本占比。

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch_npu  # noqa: F401

import ops_multimodal_fusion  # noqa: F401

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_ELEM_BYTES = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}

# (numel, anchor 标签)。覆盖需求关注的 1 / 1e3 / 1e5 / 1e7 四量级 + lane/tile 对齐边界 +
# 多核切分边界，与功能测试 L1-A 的 numel 轴同源 (测试方案 §3.2)。
_NUMELS = [
    (1, "1 单元素 / 起核数=1"),
    (63, "1e1 fp32 lane-1"),
    (64, "1e1 fp32 lane 对齐"),
    (1000, "1e3 非 2 幂"),
    (1023, "1e3 tile-1"),
    (1024, "1e3 tile 对齐"),
    (4096, "1e3 多 tile 起点"),
    (99991, "1e5 质数强非对齐"),
    (100000, "1e5 非 2 幂"),
    (262144, "1e5 2^18 全对齐"),
    (1000003, "1e6 质数"),
    (4194304, "1e6 2^22 全对齐"),
    (9999991, "1e7 质数强非对齐"),
    (10000000, "1e7 跨核归约压力最大"),
]
_QUICK_NUMELS = [(1024, "1e3 tile 对齐"), (99991, "1e5 质数强非对齐"), (9999991, "1e7 质数")]


def _make_inputs(numel, dtype):
    """良态输入 (与功能测试 L1-A 同族：X=uniform / dX=normal)，避免 NaN/Inf 干扰计时。

    一律 CPU 构造后 .npu()：本机 CANN 缺 npu 侧 ZerosLike (error 561103)，
    不能直接在 npu 上造张量。
    """
    g = torch.Generator().manual_seed(0)
    x = (torch.rand(numel, generator=g, dtype=torch.float64) * 2 - 1)
    d = torch.randn(numel, generator=g, dtype=torch.float64)
    x = x.abs().clamp(min=1e-3).copysign(x).to(dtype)
    d = d.abs().clamp(min=1e-3).copysign(d).to(dtype)
    wd = torch.tensor(0.1, dtype=torch.float32)
    trust = torch.tensor(0.5, dtype=torch.float32)
    lr_max = torch.tensor(10.0, dtype=torch.float32)
    return x.npu(), d.npu(), wd.npu(), trust.npu(), lr_max.npu()


def _bench_one(numel, dtype, warmup, iters):
    x, d, wd, trust, lr_max = _make_inputs(numel, dtype)

    def call():
        return torch.ops.ops_multimodal_fusion.c2_lars(x, d, wd, trust, lr_max, 0.5, 1e-6)

    for _ in range(warmup):
        call()
    torch.npu.synchronize()

    # 逐次用 NPU Event 计时 (device 侧时延，剥离 host 派发抖动)。
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    samples_ms = []
    for _ in range(iters):
        start.record()
        call()
        end.record()
        torch.npu.synchronize()
        samples_ms.append(start.elapsed_time(end))

    med = statistics.median(samples_ms)
    mn = min(samples_ms)
    mean = statistics.fmean(samples_ms)
    std = statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0

    # 访存下界：X 与 dX 各读一遍（需求 §3.2），标量与 0-D 输出可忽略。
    read_bytes = 2 * numel * _ELEM_BYTES[dtype]
    bw_gbps = (read_bytes / (med * 1e-3)) / 1e9 if med > 0 else 0.0

    del x, d, wd, trust, lr_max
    torch.npu.empty_cache()
    return dict(median_ms=med, min_ms=mn, mean_ms=mean, std_ms=std,
                read_MiB=read_bytes / (1024 * 1024), bw_GBps=bw_gbps)


def main():
    ap = argparse.ArgumentParser(description="c2_lars NPU performance collection framework")
    ap.add_argument("--warm-up", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dtypes", default="float32,float16,bfloat16")
    ap.add_argument("--numels", default=None, help="comma separated numel subset")
    ap.add_argument("--peak-gbps", type=float, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if not torch.npu.is_available():
        print("ERROR: NPU not available; cannot collect performance data.", file=sys.stderr)
        return 2
    if not hasattr(torch.ops.ops_multimodal_fusion, "c2_lars"):
        print("ERROR: c2_lars not registered in the installed wheel "
              "(rebuild: bash build.sh --ops=c2_lars --soc=ascend950).", file=sys.stderr)
        return 2

    dtypes = [_DTYPE_MAP[s.strip()] for s in args.dtypes.split(",") if s.strip()]
    if args.numels:
        cases = [(int(s), "user") for s in args.numels.split(",") if s.strip()]
    else:
        cases = _QUICK_NUMELS if args.quick else _NUMELS

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_path = args.out or os.path.join(repo, ".cannbot", "perf", "c2_lars_perf.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    results = []
    header = (f"{'numel':>10} {'anchor':<26} {'dtype':<9} {'read(MiB)':>10} "
              f"{'median(ms)':>11} {'min(ms)':>9} {'std(ms)':>8} {'BW(GB/s)':>9}")
    if args.peak_gbps:
        header += f" {'%peak':>7}"
    print(header)
    print("-" * len(header))
    for (numel, anchor) in cases:
        for dtype in dtypes:
            name = str(dtype).replace("torch.", "")
            try:
                r = _bench_one(numel, dtype, args.warm_up, args.iters)
            except Exception as e:  # noqa: BLE001  keep collecting other configs
                print(f"{numel:>10} {anchor:<26} {name:<9} "
                      f"FAILED: {type(e).__name__}: {str(e)[:50]}")
                results.append(dict(numel=numel, anchor=anchor, dtype=name, error=str(e)[:200]))
                torch.npu.empty_cache()
                continue
            line = (f"{numel:>10} {anchor:<26} {name:<9} {r['read_MiB']:>10.2f} "
                    f"{r['median_ms']:>11.4f} {r['min_ms']:>9.4f} {r['std_ms']:>8.4f} "
                    f"{r['bw_GBps']:>9.1f}")
            if args.peak_gbps:
                pct = 100.0 * r["bw_GBps"] / args.peak_gbps
                r["pct_peak"] = pct
                line += f" {pct:>6.1f}%"
            print(line)
            results.append(dict(numel=numel, anchor=anchor, dtype=name, **r))

    meta = dict(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        torch=torch.__version__,
        device=torch.npu.get_device_name(0) if hasattr(torch.npu, "get_device_name") else "npu",
        warm_up=args.warm_up, iters=args.iters, quick=args.quick,
        peak_gbps=args.peak_gbps,
        note="single-card shared device; confirm no other running process before sampling "
             "(需求 §2.9 采集约束)。多次采样取稳定值，不可单次采样定论。",
        metric="median kernel latency + achieved read bandwidth; c2_lars reads X and dX in full "
               "and outputs a 0-D scalar, so the MTE2 lower bound is 2*numel*sizeof(dtype) "
               "(需求 §3.2)",
    )
    with open(out_path, "w") as f:
        json.dump(dict(meta=meta, results=results), f, indent=2)
    print(f"\nperf data written to {out_path}")
    print("CP4 深度瓶颈分析入口：msprof + ops-profiling skill 的 msprof_perf_summary.py"
          "（关注 aiv_mte2_ratio / 多核 Task Duration 离散度 / 小 numel 固定启动成本占比）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
