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
# addmv 上板性能采集框架 (直调算子，自包含)。
#
# 需求 2.8：功能优先、无硬性目标；采集各 shape/dtype 性能数据供 CP4 以瓶颈分析判定。gemv 属访存瓶颈型
# (需求 3.2)，故本框架以 kernel 平均时延 + mat 搬运达成带宽 (GB/s) 为主指标。
#
# 单卡共享设备 (需求 3.3 / 0-环境信息)：采集前请先确认无其他占用进程
#   timeout 15 npu-smi info    # 期望 "No running processes"
# 并对同一配置多次采样、标注采样条件 (本框架已内置 warm-up + 多轮 + 中位数/最小值)。
#
# 运行 (必须在源码目录之外，避免 ops_multimodal_fusion/ 源目录遮蔽已安装 wheel)：
#   cd /tmp && python3 /workspace/asc-ops/ops-multimodal-fusion-tpc/tests/addmv/perf_addmv.py
# 可选参数：
#   --warm-up N     每个配置的预热次数 (规避 DVFS)，默认 5
#   --iters N       每个配置的计时轮数，默认 30
#   --out PATH      结果 JSON 输出路径，默认 <repo>/.cannbot/perf/addmv_perf.json
#   --dtypes ...    逗号分隔 dtype 子集 (float32,float16,bfloat16,int32)，默认全部
#   --quick         只跑小 shape 冒烟 (调试用)
#
# 深度瓶颈分析 (CP4 按需，本框架不替代 msprof)：
#   msprof --application="python3 .../perf_addmv.py --quick" --aic-metrics=... --output=./prof_out
#   然后用 ops-profiling skill 的 msprof_perf_summary.py 解析主 Bound / RVEC* 冲突 / MTE2 占比。

import argparse
import json
import math
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
    "int32": torch.int32,
}
_ELEM_BYTES = {
    torch.float32: 4, torch.float16: 2, torch.bfloat16: 2, torch.int32: 4,
}

# (M, K, anchor 标签)。覆盖需求 2.9 的 S1~S6 shape 类别 + dtype 维度。
_SHAPES = [
    (2, 8, "small"),
    (64, 128, "small"),
    (256, 1024, "mid"),
    (100, 1023, "S1 tail-K"),
    (100, 4097, "S1 tail-K"),
    (333, 1023, "S1 tail-MK"),
    (56, 512, "S4 core=56"),
    (112, 512, "S4 core=2x56"),
    (55, 512, "S4 core-1"),
    (1, 65536, "S2 single-row"),
    (65536, 1, "S2 single-col"),
    (65536, 1024, "S3 large M-bound"),
    (4096, 16384, "S3 large"),
    (1024, 65536, "S3 large K-bound"),
    (8192, 8192, "S3 large square"),
]
_QUICK_SHAPES = [(64, 128, "small"), (256, 1024, "mid"), (4096, 16384, "S3 large")]


def _make_inputs(M, K, dtype):
    """良态正值输入 (与功能测试 D1/D1i 同族)，避免 NaN/Inf 干扰计时。"""
    g = torch.Generator().manual_seed(0)
    if dtype == torch.int32:
        self_ = torch.randint(-100, 101, (M,), generator=g, dtype=torch.int32)
        mat = torch.randint(-3, 4, (M, K), generator=g, dtype=torch.int32)
        vec = torch.randint(-3, 4, (K,), generator=g, dtype=torch.int32)
    else:
        s = 1.0 / math.sqrt(K) if K > 0 else 1.0
        self_ = (torch.rand(M, generator=g) + 0.5).to(dtype)
        mat = ((torch.rand(M, K, generator=g) + 0.5) * s).to(dtype)
        vec = ((torch.rand(K, generator=g) + 0.5) * s).to(dtype)
    return self_.npu(), mat.npu(), vec.npu()


def _bench_one(M, K, dtype, warmup, iters):
    self_npu, mat_npu, vec_npu = _make_inputs(M, K, dtype)

    def call():
        return torch.ops.ops_multimodal_fusion.addmv(self_npu, mat_npu, vec_npu, 1, 1)

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

    mat_bytes = M * K * _ELEM_BYTES[dtype]
    # gemv 访存下界 ≈ mat 全量 + vec(K) + self(M) + out(M)；mat 主导，用 mat 达成带宽作主指标。
    bw_gbps = (mat_bytes / (med * 1e-3)) / 1e9 if med > 0 else 0.0

    del self_npu, mat_npu, vec_npu
    torch.npu.empty_cache()
    return dict(median_ms=med, min_ms=mn, mean_ms=mean, std_ms=std,
                mat_MiB=mat_bytes / (1024 * 1024), bw_GBps=bw_gbps)


def main():
    ap = argparse.ArgumentParser(description="addmv NPU performance collection framework")
    ap.add_argument("--warm-up", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dtypes", default="float32,float16,bfloat16,int32")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if not torch.npu.is_available():
        print("ERROR: NPU not available; cannot collect performance data.", file=sys.stderr)
        return 2
    if not hasattr(torch.ops.ops_multimodal_fusion, "addmv"):
        print("ERROR: addmv not registered in the installed wheel.", file=sys.stderr)
        return 2

    dtypes = [_DTYPE_MAP[d.strip()] for d in args.dtypes.split(",") if d.strip()]
    shapes = _QUICK_SHAPES if args.quick else _SHAPES

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_path = args.out or os.path.join(repo, ".cannbot", "perf", "addmv_perf.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    results = []
    header = f"{'shape (M,K)':>16} {'anchor':<16} {'dtype':<9} {'mat(MiB)':>9} " \
             f"{'median(ms)':>11} {'min(ms)':>9} {'BW(GB/s)':>9}"
    print(header)
    print("-" * len(header))
    for (M, K, anchor) in shapes:
        for dtype in dtypes:
            try:
                r = _bench_one(M, K, dtype, args.warm_up, args.iters)
            except Exception as e:  # noqa: BLE001  keep collecting other configs
                print(f"{f'({M},{K})':>16} {anchor:<16} {str(dtype).replace('torch.',''):<9} "
                      f"FAILED: {type(e).__name__}: {str(e)[:50]}")
                results.append(dict(M=M, K=K, anchor=anchor, dtype=str(dtype), error=str(e)[:200]))
                torch.npu.empty_cache()
                continue
            print(f"{f'({M},{K})':>16} {anchor:<16} {str(dtype).replace('torch.',''):<9} "
                  f"{r['mat_MiB']:>9.1f} {r['median_ms']:>11.4f} {r['min_ms']:>9.4f} {r['bw_GBps']:>9.1f}")
            results.append(dict(M=M, K=K, anchor=anchor, dtype=str(dtype).replace("torch.", ""), **r))

    meta = dict(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        torch=torch.__version__,
        device=torch.npu.get_device_name(0) if hasattr(torch.npu, "get_device_name") else "npu",
        warm_up=args.warm_up, iters=args.iters, quick=args.quick,
        note="single-card shared device; confirm 'No running processes' before sampling (需求 3.3)",
        metric="median kernel latency + achieved mat-transfer bandwidth (gemv is MTE2/access bound)",
    )
    with open(out_path, "w") as f:
        json.dump(dict(meta=meta, results=results), f, indent=2)
    print(f"\nperf data written to {out_path}")
    print("CP4 深度瓶颈分析入口：msprof + ops-profiling skill 的 msprof_perf_summary.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
