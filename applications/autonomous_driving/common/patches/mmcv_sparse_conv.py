# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
mmcv 稀疏卷积纯 torch 替换（NPU 适配）。

背景：mmcv 的 SparseConv3d 依赖 CUDA 扩展（`get_indice_pairs` / `indice_conv_forward`，
见 mmcv/ops/csrc/.../sparse_indice.cu / spconv_ops_cuda.cu），在 NPU/CPU 上不可用
（报 "get_indice_pairs is not implemented on CPU"）。本模块用纯 torch 实现同语义的
前向，monkey-patch 进 `mmcv.ops.sparse_ops`，使 Part-A2 的 SparseUNet 能在 NPU 上运行。

算法语义逆向自 mmcv 2.2.0 csrc：
- get_indice_pairs 输出：outInds (M, coorDim+1)、indicePairs (K,2,N)、indiceNum (K,)
  - conv：    out = (in + padding - koff*dilation) / stride   （须整除、越界过滤）
  - deconv：  out = in*stride - padding + koff*dilation
  - subm：    out = in + (koff - center)*dilation，输出 coords = 输入 coords
  - 有效 pair 前移打包；indicePairs[k,0,:]=输入idx、indicePairs[k,1,:]=输出idx，
    indiceNum[k]=该 kernel offset 的有效 pair 数；无效位置填 -1
- indice_conv：output[indicePairs[k,1,i]] += features[indicePairs[k,0,i]] @ filters[k]
  - filters 形状 (kx,ky,kz,in,out)，view(-1,in,out) → filters[k]（行主序展平）
  - subm 用 center kernel（indiceNum 最大）对整个 features 做一次 mm
  - inverse（deconv）时 gather/scatter 的 src/dst 互换

按迁移原则归档为 patch：只 monkey-patch，不改 mmcv 源文件。
"""

import os
import sys

import torch
from mmcv.ops import sparse_ops as _mmcv_sparse_ops


def _kernel_offsets(ksize, ndim, device):
    """生成 kernel 偏移网格 (K, ndim)，行主序（首维最慢，末维最快），
    与 mmcv 权重 view(-1,in,out) 的展平顺序一致。"""
    ranges = [torch.arange(int(k), dtype=torch.int64, device=device) for k in ksize]
    grids = torch.meshgrid(*ranges, indexing='ij')  # 每张 (kx,ky,kz)
    offsets = torch.stack([g.reshape(-1) for g in grids], dim=1)  # (K, ndim)
    return offsets


def _out_shape(spatial_shape, ksize, stride, padding, dilation, out_padding,
               subm, transpose, ndim):
    if subm:
        return list(spatial_shape)
    if transpose:
        return [
            (spatial_shape[i] - 1) * stride[i] - 2 * padding[i]
            + dilation[i] * (ksize[i] - 1) + out_padding[i] + 1
            for i in range(ndim)
        ]
    return [
        (spatial_shape[i] + 2 * padding[i] - dilation[i] * (ksize[i] - 1) - 1)
        // stride[i] + 1 for i in range(ndim)
    ]


def get_indice_pairs_pure(indices, batch_size, spatial_shape, ksize, stride,
                          padding, dilation, out_padding, subm, transpose,
                          grid=None):
    """纯 torch 版 get_indice_pairs，签名与 mmcv.ops.sparse_ops.get_indice_pairs 一致。"""
    N = indices.shape[0]
    ndim = indices.shape[1] - 1
    device = indices.device

    ksize = [int(k) for k in ksize]
    stride = [int(s) for s in stride]
    padding = [int(p) for p in padding]
    dilation = [int(d) for d in dilation]
    out_padding = [int(op) for op in out_padding]
    spatial_shape = [int(s) for s in spatial_shape]

    K = 1
    for k in ksize:
        K *= int(k)

    indice_pairs = torch.full((K, 2, N), -1, dtype=torch.int32, device=device)
    indice_num = torch.zeros((K,), dtype=torch.int32, device=device)

    if N == 0:
        return indices, indice_pairs, indice_num

    coords = indices[:, 1:].to(torch.int64)  # (N, ndim)
    offsets = _kernel_offsets(ksize, ndim, device)  # (K, ndim)
    dil_t = torch.tensor(dilation, dtype=torch.int64, device=device)

    # ---------- subm ----------
    if subm:
        # subm 输出 voxel = 输入 voxel（索引不变）。对输出 i（coord c_i）、kernel 偏移 k：
        # 贡献输入在 coord c_i + (koff - center)*dilation，需在输入集合中存在。
        # indice_pairs[k,0]=邻居输入 idx，indice_pairs[k,1]=输出 idx(=i)
        center = [(k - 1) // 2 for k in ksize]
        center_t = torch.tensor(center, dtype=torch.int64, device=device)
        shape_t = torch.tensor(spatial_shape, dtype=torch.int64, device=device)
        # 线性编码 coord -> 索引，用于查邻居输入是否存在
        mult = torch.tensor([spatial_shape[1] * spatial_shape[2],
                             spatial_shape[2], 1],
                            dtype=torch.int64, device=device)
        enc = (coords * mult).sum(-1)  # (N,)
        sorted_enc, sort_idx = torch.sort(enc)  # 一次排序，K 维复用

        # 向量化整批（K 维广播）：cand (N,K,ndim)，一次 searchsorted + 掩码，
        # 消除逐 k 的 Python 循环与 per-k 的 .item() D2H 同步。
        cand = (coords[:, None, :]
                + (offsets[None, :, :] - center_t[None, None, :])
                * dil_t[None, None, :])                       # (N, K, ndim)
        in_range = ((cand >= 0) & (cand < shape_t[None, None, :])).all(-1)
        cand_enc = (cand * mult[None, None, :]).sum(-1)        # (N, K)
        pos = torch.searchsorted(sorted_enc, cand_enc)         # (N, K)
        pos = torch.clamp(pos, 0, N - 1)
        match = (sorted_enc[pos] == cand_enc) & in_range       # (N, K)

        # 每 k 有效数：一次列和写入 indice_num，无 per-k D2H 同步
        match_int = match.to(torch.int32)
        indice_num[:] = match_int.sum(0)

        # 填充 pairs：行主序取有效 (i,k)，rank 为 k 列内累计排名。
        # nonzero 行主序 ⇒ 固定 k 内 i 递增，与逐 k 循环顺序一致。
        # 用 1D 展平 scatter（k*2N + s*N + rank）写 pairs（语义等价、无 per-k 同步），
        # 避免「中间维标量索引」的写路径。
        i_idx, k_idx = match.nonzero(as_tuple=True)            # (P,), (P,)
        rank = (match_int.cumsum(0) - 1)[i_idx, k_idx]         # (P,) k 列内 0-based 排名
        in_i = sort_idx[pos][i_idx, k_idx].to(torch.int32)     # 邻居输入 idx
        pairs_flat = indice_pairs.view(-1)
        pairs_flat[k_idx * (2 * N) + rank] = in_i              # slot0 = 输入 idx
        pairs_flat[k_idx * (2 * N) + N + rank] = i_idx.to(torch.int32)  # slot1 = 输出 idx(=自身)
        return indices, indice_pairs, indice_num

    # ---------- conv / deconv ----------
    out_shape = _out_shape(spatial_shape, ksize, stride, padding, dilation,
                           out_padding, False, transpose, ndim)
    str_t = torch.tensor(stride, dtype=torch.int64, device=device)
    pad_t = torch.tensor(padding, dtype=torch.int64, device=device)
    out_shape_t = torch.tensor(out_shape, dtype=torch.int64, device=device)

    if transpose:
        out_cand = (coords[:, None, :] * str_t[None, None, :]
                    - pad_t[None, None, :]
                    + offsets[None, :, :] * dil_t[None, None, :])  # (N,K,ndim)
        valid_int = torch.ones_like(out_cand, dtype=torch.bool)
    else:
        numer = (coords[:, None, :] + pad_t[None, None, :]
                 - offsets[None, :, :] * dil_t[None, None, :])
        out_cand = torch.div(numer, str_t[None, None, :], rounding_mode='trunc')
        valid_int = (numer % str_t[None, None, :]) == 0

    valid_range = (out_cand >= 0) & (out_cand < out_shape_t[None, None, :])
    valid = (valid_int & valid_range).all(-1)  # (N, K)

    valid_flat = valid.reshape(-1)  # (N*K,)
    v_pos = valid_flat.nonzero(as_tuple=False).squeeze(1)  # (V,)
    if v_pos.numel() == 0:
        return indices, indice_pairs, indice_num

    V = v_pos.numel()
    out_cand_flat = out_cand.reshape(-1, ndim)[v_pos]  # (V, ndim)
    batch_ids = indices[:, 0].unsqueeze(1).expand(N, K).reshape(-1)[v_pos]  # (V,)
    uniq_in = torch.cat([batch_ids.unsqueeze(1), out_cand_flat], dim=1)  # (V, ndim+1)

    # (batch, x, y, z) -> 单个 int64 线性编码（乘 strides）。
    # strides 按降维单调设计，code 序 == (batch,x,y,z) 字典序，替代多维 torch.unique(dim=0)。
    b_stride = out_shape[0] * out_shape[1] * out_shape[2]
    x_stride = out_shape[1] * out_shape[2]
    y_stride = out_shape[2]
    codes = (uniq_in[:, 0] * b_stride
             + uniq_in[:, 1] * x_stride
             + uniq_in[:, 2] * y_stride
             + uniq_in[:, 3])  # (V,) int64

    # sort + 相邻去重：outids 按 code 序（=字典序）输出，与 torch.unique(dim=0) 一致；
    # inv 由 searchsorted 一次得到，顺序与 v_pos 一致。
    sorted_codes, sort_idx = torch.sort(codes)
    first_flag = torch.ones(V, dtype=torch.bool, device=device)
    first_flag[1:] = sorted_codes[1:] != sorted_codes[:-1]
    first_pos = first_flag.nonzero(as_tuple=False).squeeze(1)  # (M,)
    outids = uniq_in[sort_idx[first_pos]]  # (M, ndim+1) 字典序
    inv = torch.searchsorted(sorted_codes[first_pos], codes)  # (V,) 输出 voxel idx

    i_vals = v_pos // K  # 输入 idx
    k_vals = v_pos % K   # kernel idx

    # 每 k 有效数 + 按 k 列内累计排名填充（与逐 k 循环顺序一致，无 per-k 同步）
    # 用 1D 展平 scatter 写 pairs（语义等价、无 per-k 同步）。
    col_counts = valid.to(torch.int32).cumsum(0)  # (N, K)
    indice_num[:] = col_counts[N - 1]
    rank = col_counts[i_vals, k_vals] - 1  # (V,) k 内 0-based 排名
    pairs_flat = indice_pairs.view(-1)
    pairs_flat[k_vals * (2 * N) + rank] = i_vals.to(torch.int32)
    pairs_flat[k_vals * (2 * N) + N + rank] = inv.to(torch.int32)

    return outids, indice_pairs, indice_num


def indice_conv_pure(features, filters, indice_pairs, indice_num,
                     num_activate_out, inverse=False, subm=False):
    """纯 torch 版 indice_conv（含 subm / inverse）。"""
    K = indice_pairs.shape[0]
    C_in = features.shape[1]
    device = features.device
    dtype = features.dtype
    filters_k = filters.reshape(K, C_in, -1)  # (K, in, out)
    C_out = filters_k.shape[2]
    output = torch.zeros(num_activate_out, C_out, dtype=dtype, device=device)

    # 一次性拷回 host，避免逐 k 的 int(indice_num[k].item()) D2H 同步（K=27 次 → 1 次）
    indice_num_cpu = indice_num.detach().cpu()
    center = -1
    if subm:
        center = int(torch.argmax(indice_num_cpu).item())
        output = features @ filters_k[center]

    for k in range(K):
        n_hot = int(indice_num_cpu[k].item())
        if n_hot <= 0 or (subm and k == center):
            continue
        src = 1 if inverse else 0
        dst = 0 if inverse else 1
        idx_in = indice_pairs[k, src, :n_hot].to(torch.int64)
        idx_out = indice_pairs[k, dst, :n_hot].to(torch.int64)
        inp = features.index_select(0, idx_in)  # (n_hot, in)
        out_buf = inp @ filters_k[k]  # (n_hot, out)
        output.index_add_(0, idx_out, out_buf)

    return output


# ---------------------------------------------------------------------------
# Ascend C 稀疏卷积 kernel（ops_multimodal_fusion.indice_conv）接入
#
# 可选加速路径：host 侧预处理（build_csr）把 indice_pairs 拍平成按输出 voxel
# 排序的 CSR（out_ptr/k_idx/gathered），一次 kernel 启动完成全部稀疏 offset 的
# 累加（output[out_i] += features[in_i] @ filters[k_i]）。
#
# 语义与 indice_conv_pure 完全一致：
#   - subm：中心 kernel 在 host 用稠密 torch mm（快），CSR 剔除中心；
#   - conv/deconv（inverse）：全部 kernel offset 进 CSR，inverse 时 src/dst 互换。
#
# 可用性：kernel 依赖统一构建产物 libops_multimodal_fusion.so，非 NPU 设备、
# 未编译/加载失败、dtype 不支持或形状不适用时静默回退 indice_conv_pure。
# ---------------------------------------------------------------------------

# 项目标准接入：算子以 wheel 交付，`import ops_multimodal_fusion` 即自动
# load libops_multimodal_fusion.so（见其 __init__.py），indice_conv 随之注册。
# 不再自行搜 build/lib（避免同一 .so 两份导致重复注册）。
_KERNEL_AVAILABLE = False
_KERNEL_LOAD_ERROR = ""
_KERNEL_OP = None
_KERNEL_MATMUL_OP = None
# get_indice_pairs 第一步的 subm 邻居查重 Ascend C kernel（可选；缺失时回退纯 torch）。
_GIP_OP = None
_GIP_LOAD_ERROR = ""
try:
    import ops_multimodal_fusion  # noqa: F401  标准：装 wheel 后 import 自动 load .so
    _matmul_op = getattr(torch.ops.ops_multimodal_fusion, "indice_conv", None)
    if _matmul_op is None:
        raise RuntimeError("ops_multimodal_fusion.indice_conv not registered")
    _KERNEL_OP = _matmul_op
    _KERNEL_MATMUL_OP = _matmul_op
    _GIP_OP = getattr(torch.ops.ops_multimodal_fusion,
                      "get_indice_pairs_subm_lookup", None)
    _KERNEL_AVAILABLE = True
except Exception as _e:  # noqa: BLE001  kernel 不可用不影响纯 torch 回退
    _KERNEL_LOAD_ERROR = str(_e)


def _is_npu(t):
    return getattr(t.device, "type", "") == "npu"


def _coord_mult(spatial_shape, device):
    """坐标线性编码乘子：mult[d] = prod(spatial_shape[d+1:])（dim-0 最高位）。

    subm 与 conv/deconv 的 outids 字典序都用它；与 get_indice_pairs_pure 的
    [s1*s2, s2, 1]（ndim=3）完全一致，ndim 通用。
    """
    ndim = len(spatial_shape)
    mult = [1] * ndim
    acc = 1
    for d in range(ndim - 1, -1, -1):
        mult[d] = acc
        acc *= int(spatial_shape[d])
    return torch.tensor(mult, dtype=torch.int64, device=device)


def _gip_subm_kernel(indices, coords, spatial_shape, ksize, dilation,
                     indice_pairs, indice_num):
    """subm 路径的 kernel 版邻居查重 + host 打包（与 get_indice_pairs_pure 逐值一致）。

    kernel（get_indice_pairs_subm_lookup）对每个 (i,k) 在输入坐标集上做 lower_bound，
    替代纯 torch 里 (N,K) 次 torch.searchsorted（NPU 上 ~ms，占 subm 路径 90%+ 成本）。
    打包（per-k cumsum rank + scatter）与 pure 共用同一段向量步骤，保证顺序语义一致：
    pairs[k,0]=邻居输入行、pairs[k,1]=输出行(=i)，k 内按输入 i 升序。
    """
    N = indices.shape[0]
    K = indice_pairs.shape[0]
    device = indices.device
    mult = _coord_mult(spatial_shape, device)
    enc = (coords * mult).sum(-1)  # (N,)
    sorted_enc, sort_idx = torch.sort(enc)
    neighbor = _GIP_OP(coords, list(spatial_shape), list(ksize), list(dilation),
                       sorted_enc, sort_idx.to(torch.int32))  # (N, K) int32
    match = neighbor >= 0
    match_int = match.to(torch.int32)
    indice_num[:] = match_int.sum(0)
    i_idx, k_idx = match.nonzero(as_tuple=True)
    rank = (match_int.cumsum(0) - 1)[i_idx, k_idx]  # k 列内 0-based 排名
    in_i = neighbor[i_idx, k_idx]
    pairs_flat = indice_pairs.view(-1)
    pairs_flat[k_idx * (2 * N) + rank] = in_i              # slot0 = 邻居输入 idx
    pairs_flat[k_idx * (2 * N) + N + rank] = i_idx.to(torch.int32)  # slot1 = 输出 idx(=自身)
    return indices, indice_pairs, indice_num


def get_indice_pairs_fast(indices, batch_size, spatial_shape, ksize, stride,
                          padding, dilation, out_padding, subm, transpose,
                          grid=None):
    """加速版 get_indice_pairs，与 mmcv.ops.sparse_ops.get_indice_pairs 签名一致。

    - subm：可用 Ascend C kernel（get_indice_pairs_subm_lookup）做邻居查重，替代 NPU 上
      昂贵的 (N,K) searchsorted；打包与 pure 逐值一致。
    - conv/deconv：把 `inv = searchsorted(sorted_unique, codes)` 换成等价的
      「sort + 分组 cumsum + 反置 scatter」（queries 全部来自被排序集合，无需查找），
      也消除 NPU 上昂贵的 searchsorted。
    - kernel 不可用（未编译/加载失败/非 NPU/dtype/形状）时回退 get_indice_pairs_pure。
    返回与 get_indice_pairs_pure 完全相同的 (outids, indice_pairs, indice_num)。
    """
    N = indices.shape[0]
    ndim = indices.shape[1] - 1
    device = indices.device

    ksize = [int(k) for k in ksize]
    stride = [int(s) for s in stride]
    padding = [int(p) for p in padding]
    dilation = [int(d) for d in dilation]
    out_padding = [int(op) for op in out_padding]
    spatial_shape = [int(s) for s in spatial_shape]

    K = 1
    for k in ksize:
        K *= int(k)

    indice_pairs = torch.full((K, 2, N), -1, dtype=torch.int32, device=device)
    indice_num = torch.zeros((K,), dtype=torch.int32, device=device)

    if N == 0:
        return indices, indice_pairs, indice_num

    coords = indices[:, 1:].to(torch.int64).contiguous()  # (N, ndim)
    # 放宽 subm kernel 的 dtype 门槛到 int32/int64：真实模型里 voxelizer 的 int32 indices
    # 只喂第 1 个 subm，其后 3 个 subm 层的输入是上一 conv 分支产出的 int64 outids
    # （与 pure 完全一致），此前被 `dtype == torch.int32` 门槛拦下而回退 (N,K) searchsorted
    # （~48ms/帧，为当前最大剩余成本）。kernel 自身收的就是 int64 coords，对输入 indices 的
    # dtype 无内在依赖 → 放宽后 subm kernel 覆盖从 1/4 升到 4/4，零功能风险。
    # int32/int64 均先统一转为连续 int64：int64 输入时 `indices[:, 1:]` 是非连续视图、
    # `.to(int64)` 是恒等不复制，而 kernel 的 TORCH_CHECK 要求 coords.is_contiguous()，
    # 故必须 `.contiguous()`；int32 路径 `.to(int64)` 本就会新建连续张量，`.contiguous()` no-op。
    use_kernel = (
        _GIP_OP is not None
        and _is_npu(indices)
        and indices.dtype in (torch.int32, torch.int64)
        and indices.is_contiguous()
    )

    # ---------- subm ----------
    if subm:
        if use_kernel:
            return _gip_subm_kernel(indices, coords, spatial_shape, ksize,
                                    dilation, indice_pairs, indice_num)
        # 回退：与 get_indice_pairs_pure 的 subm 分支完全一致
        center = [(k - 1) // 2 for k in ksize]
        center_t = torch.tensor(center, dtype=torch.int64, device=device)
        shape_t = torch.tensor(spatial_shape, dtype=torch.int64, device=device)
        mult = _coord_mult(spatial_shape, device)
        enc = (coords * mult).sum(-1)
        sorted_enc, sort_idx = torch.sort(enc)
        offsets = _kernel_offsets(ksize, ndim, device)
        dil_t = torch.tensor(dilation, dtype=torch.int64, device=device)
        cand = (coords[:, None, :]
                + (offsets[None, :, :] - center_t[None, None, :])
                * dil_t[None, None, :])
        in_range = ((cand >= 0) & (cand < shape_t[None, None, :])).all(-1)
        cand_enc = (cand * mult[None, None, :]).sum(-1)
        pos = torch.searchsorted(sorted_enc, cand_enc)
        pos = torch.clamp(pos, 0, N - 1)
        match = (sorted_enc[pos] == cand_enc) & in_range
        match_int = match.to(torch.int32)
        indice_num[:] = match_int.sum(0)
        i_idx, k_idx = match.nonzero(as_tuple=True)
        rank = (match_int.cumsum(0) - 1)[i_idx, k_idx]
        in_i = sort_idx[pos][i_idx, k_idx].to(torch.int32)
        pairs_flat = indice_pairs.view(-1)
        pairs_flat[k_idx * (2 * N) + rank] = in_i
        pairs_flat[k_idx * (2 * N) + N + rank] = i_idx.to(torch.int32)
        return indices, indice_pairs, indice_num

    # ---------- conv / deconv ----------
    # 与 get_indice_pairs_pure 的 conv/deconv 分支逐行一致，唯一差别：
    # `inv = searchsorted(sorted_codes[first_pos], codes)` →
    #   grp = first_flag.cumsum(0)-1（已排序位置上每个候选所属唯一代码组序号）
    #   inv[sort_idx] = grp（反置回原候选序 v_pos）。
    # 两者逐值相等（每个候选代码必等于某个唯一代码，searchsorted 即返回该唯一组序号）。
    out_shape = _out_shape(spatial_shape, ksize, stride, padding, dilation,
                           out_padding, False, transpose, ndim)
    str_t = torch.tensor(stride, dtype=torch.int64, device=device)
    pad_t = torch.tensor(padding, dtype=torch.int64, device=device)
    out_shape_t = torch.tensor(out_shape, dtype=torch.int64, device=device)
    offsets = _kernel_offsets(ksize, ndim, device)
    dil_t = torch.tensor(dilation, dtype=torch.int64, device=device)

    if transpose:
        out_cand = (coords[:, None, :] * str_t[None, None, :]
                    - pad_t[None, None, :]
                    + offsets[None, :, :] * dil_t[None, None, :])
        valid_int = torch.ones_like(out_cand, dtype=torch.bool)
    else:
        numer = (coords[:, None, :] + pad_t[None, None, :]
                 - offsets[None, :, :] * dil_t[None, None, :])
        out_cand = torch.div(numer, str_t[None, None, :], rounding_mode='trunc')
        valid_int = (numer % str_t[None, None, :]) == 0

    valid_range = (out_cand >= 0) & (out_cand < out_shape_t[None, None, :])
    valid = (valid_int & valid_range).all(-1)  # (N, K)

    valid_flat = valid.reshape(-1)  # (N*K,)
    v_pos = valid_flat.nonzero(as_tuple=False).squeeze(1)  # (V,)
    if v_pos.numel() == 0:
        return indices, indice_pairs, indice_num

    V = v_pos.numel()
    out_cand_flat = out_cand.reshape(-1, ndim)[v_pos]  # (V, ndim)
    batch_ids = indices[:, 0].unsqueeze(1).expand(N, K).reshape(-1)[v_pos]  # (V,)
    uniq_in = torch.cat([batch_ids.unsqueeze(1), out_cand_flat], dim=1)  # (V, ndim+1)

    # (batch, x, y, z) -> 单个 int64 线性编码（乘 strides，code 序 == 字典序）
    b_stride = out_shape[0] * out_shape[1] * out_shape[2]
    x_stride = out_shape[1] * out_shape[2]
    y_stride = out_shape[2]
    codes = (uniq_in[:, 0] * b_stride
             + uniq_in[:, 1] * x_stride
             + uniq_in[:, 2] * y_stride
             + uniq_in[:, 3])  # (V,) int64

    # sort + 相邻去重：outids 按 code 序（=字典序）输出；inv 由分组 cumsum + 反置得到，
    # 顺序与 v_pos 一致（等价于 searchsorted，免去 NPU 昂贵的 V 次查找）。
    sorted_codes, sort_idx = torch.sort(codes)
    first_flag = torch.ones(V, dtype=torch.bool, device=device)
    first_flag[1:] = sorted_codes[1:] != sorted_codes[:-1]
    first_pos = first_flag.nonzero(as_tuple=False).squeeze(1)  # (M,)
    outids = uniq_in[sort_idx[first_pos]]  # (M, ndim+1) 字典序
    grp = first_flag.to(torch.int32).cumsum(0) - 1  # (V,) 每个已排序位置的唯一组序号
    inv = torch.empty(V, dtype=torch.int64, device=device)
    inv[sort_idx] = grp.to(torch.int64)  # 反置回原候选序（v_pos 序）

    i_vals = v_pos // K  # 输入 idx
    k_vals = v_pos % K   # kernel idx

    # 每 k 有效数 + 按 k 列内累计排名填充（与逐 k 循环顺序一致，无 per-k 同步）
    col_counts = valid.to(torch.int32).cumsum(0)  # (N, K)
    indice_num[:] = col_counts[N - 1]
    rank = col_counts[i_vals, k_vals] - 1  # (V,) k 内 0-based 排名
    pairs_flat = indice_pairs.view(-1)
    pairs_flat[k_vals * (2 * N) + rank] = i_vals.to(torch.int32)
    pairs_flat[k_vals * (2 * N) + N + rank] = inv.to(torch.int32)

    return outids, indice_pairs, indice_num


def build_csr(features, filters_k, indice_pairs, indice_num, M,
              subm=False, inverse=False, center=-1):
    """把有效 (input, output, kernel-offset) 三元组拍平、剔除 subm 中心、
    按输出 voxel 排序为 CSR。

    Returns (gathered (P, C_in), k_idx (P,), out_ptr (M+1,))，全部为 int32/原 dtype。
    """
    device = features.device
    K, _, N = indice_pairs.shape
    src = 1 if inverse else 0
    dst = 0 if inverse else 1

    # 一次性拷回 host，避免逐 k 的 int(indice_num[k].item()) D2H 同步
    num_cpu = indice_num.detach().cpu()
    P = int(num_cpu.sum().item())
    if subm and center >= 0:
        P -= int(num_cpu[center].item())
    if P <= 0:
        gathered = torch.zeros(0, features.shape[1], dtype=features.dtype, device=device)
        k_idx = torch.zeros(0, dtype=torch.int32, device=device)
        out_ptr = torch.zeros(M + 1, dtype=torch.int32, device=device)
        return gathered, k_idx, out_ptr

    # 向量化：valid(k, r) = r < indice_num[k]，indice_pairs 前 indice_num[k] 个有效
    ar = torch.arange(N, device=device)
    valid = ar[None, :] < indice_num[:, None]                    # (K, N)
    if subm and center >= 0:
        valid = valid.clone()
        valid[center, :] = False
    valid_flat = valid.reshape(-1)

    in_flat = indice_pairs[:, src, :].reshape(-1)                # (K*N,) k-major
    out_flat = indice_pairs[:, dst, :].reshape(-1)
    k_flat = torch.arange(K, device=device).repeat_interleave(N)  # (K*N,)
    in_idx = in_flat[valid_flat]                                 # (P,)
    out_idx = out_flat[valid_flat]
    k_idx = k_flat[valid_flat].to(torch.int32)

    order = torch.argsort(out_idx, stable=True)
    in_idx = in_idx[order]
    out_idx = out_idx[order]
    k_idx = k_idx[order]

    out_ptr = torch.zeros(M + 1, dtype=torch.int32, device=device)
    counts = torch.zeros(M, dtype=torch.int32, device=device)
    counts.index_add_(0, out_idx, torch.ones_like(out_idx))
    torch.cumsum(counts, 0, out=out_ptr[1:])

    gathered = features[in_idx.to(torch.int64)]
    return gathered, k_idx, out_ptr


def build_k(features, indice_pairs, indice_num, M, subm=False, inverse=False, center=-1):
    """k-major 拍平（无 argsort）：kernel 的 per-pair matmul 输入。

    indice_pairs 本身按 k 分组（(K,2,N)，k 为首维），有效 pair 按 k 拼接连续，
    kernel 中相邻线程读同一 filter 矩阵（L2 局部性）。返回：
      gathered (P, C_in)  = features[输入 voxel]   （k-major）
      k_idx    (P,)       = 每个 pair 的 kernel offset
      out_idx  (P,)       = 每个 pair 的输出 voxel（用于 host index_add scatter）
      s_idx    (P,)       = identity（kernel 写 partial 的索引，破坏 affine store 向量化）
    """
    device = features.device
    K, _, N = indice_pairs.shape
    src = 1 if inverse else 0
    dst = 0 if inverse else 1

    ar = torch.arange(N, device=device)
    valid = ar[None, :] < indice_num[:, None]          # (K, N)
    if subm and center >= 0:
        valid = valid.clone()
        valid[center, :] = False
    valid_flat = valid.reshape(-1)

    in_idx = indice_pairs[:, src, :].reshape(-1)[valid_flat].to(torch.int32)
    out_idx = indice_pairs[:, dst, :].reshape(-1)[valid_flat].to(torch.int32)
    # k_idx = row index of each valid pair (k-major nonzero). nonzero()[:,(0,1)] on a
    # (K, N) bool mask returns row-major coordinates -> row = k, col = position within k.
    v_rc = valid.nonzero(as_tuple=False)               # (P, 2)
    k_idx = v_rc[:, 0].to(torch.int32)
    P = in_idx.shape[0]
    s_idx = torch.arange(P, dtype=torch.int32, device=device)
    gathered = features[in_idx.to(torch.int64)]
    return gathered, k_idx, out_idx, s_idx


def _bmm_sparse(gathered, k_idx, out_idx, filters_k, M):
    """padded-batched-matmul：partial[p] = gathered[p] @ filters_k[k_idx[p]]，再 scatter。

    把 27 个 k-group 的稀疏 matmul 合并成一次 Cube bmm（每 k 一个 batch，pad 到
    maxn），避免 pure 路径 27 次 index_select+mm+index_add 的 op 开销。kernel 的
    per-pair SIMT matmul 在 C_out=32/64 时随 C_in*C_out 变慢，此路径在宽通道形态
    下快 5-13x。精度为 NPU Cube 的降精度（与 pure 的 per-k mm 同一路径，两者
    相对差 ~1e-7，与 fp64 相对差 ~3e-4）。
    """
    K = filters_k.shape[0]
    C_in = filters_k.shape[1]
    C_out = filters_k.shape[2]
    P = gathered.shape[0]
    device = gathered.device
    nk = torch.bincount(k_idx, minlength=K)               # (K,) 每 k pair 数
    maxn = int(nk.max().item())
    starts = torch.zeros(K, dtype=torch.int64, device=device)
    torch.cumsum(nk.to(torch.int64), 0, out=starts)
    starts = starts - nk.to(torch.int64)                  # 每 k 的起始 pair 位置
    pos = torch.arange(P, dtype=torch.int64, device=device)
    rank = pos - starts[k_idx.to(torch.int64)]            # k 内排名（k-major 下 = 列）
    flat_pos = (k_idx.to(torch.int64) * maxn + rank).to(torch.int32)

    Gpad = torch.zeros(K * maxn, C_in, dtype=gathered.dtype, device=device)
    Gpad[flat_pos.to(torch.int64)] = gathered             # 散入 padded 布局
    Z = torch.bmm(Gpad.view(K, maxn, C_in), filters_k)    # (K, maxn, C_out)
    Zf = Z.reshape(-1, C_out)
    out = torch.zeros(M, C_out, dtype=gathered.dtype, device=device)
    out.index_add_(0, out_idx.to(torch.int64), Zf[flat_pos.to(torch.int64)])
    return out


def _should_use_bmm(M, P, C_in, C_out):
    """padded-bmm 路由：实测标定（真实模型各 conv 层逐形状 A/B）。

    bmm 的成本 ~ O(K*maxn*C_in*C_out)（Cube）+ Gpad/Z 两次大张量搬运；实测
    （真实模型形态，见 ROOTFIX.md benchmark 表）：
      - C_out<=32 且 C_in<=64：bmm 相对 pure 快 ~1.2-2.2x，放行；
      - C_out>=64（尤其 C_out=128）：bmm 输出张量大、Cube 不占优，回退 pure；
      - M>200000（stride2 级宽浅）：bmm scatter 本身贵，回退 pure；
      - P 过小：bmm 的 pad 浪费 + 启动开销不划算，回退 pure。
    """
    if M <= 0 or P <= 0:
        return False
    if P < 256:
        return False
    if M > 200000:
        return False
    if C_out > 32:
        return False
    if C_in > 64:
        return False
    total_flops = P * C_in * C_out
    # kernel（per-pair matmul）已覆盖的形态不重复走 bmm
    if C_out <= 16 and total_flops <= 5e7:
        return False
    est_maxn = max(1, P // 26)
    est_mem = 2 * 27 * est_maxn * C_out * 4
    if est_mem > 1_000_000_000:
        return False
    return True


def _should_use_kernel(M, P, C_in, C_out):
    """新 per-pair matmul kernel 的路由（按真实模型形态重新标定）。

    kernel（per-pair matmul，k-major）的成本 ~ P*C_in*C_out*10.4ps（Ascend950PR
    实测），纯 torch 路径的成本 ~0.9ms 固定流水（27 次 gather+mm+index_add + 1 次
    D2H）。用 kernel 当且仅当 kernel 路径（pre + matmul + scatter）快于纯 torch。

    实测标定（真实模型形态 M=40000/P=115430）：
      - subm c16/c4x16（P*C_in*C_out <= ~2e7）：1.9~2.9x，放行；
      - subm c32（P=75k, C_in=C_out=32, P*C*C=7.7e7）：kernel 0.80ms > pure 0.92ms 的
        sparse 部分，总路径 0.78x，回退；
      - subm c64 / stride2 / deconv：kernel 随 C_out 的 y-grid 冗余与 C_in*C_out 变慢，
        回退（C_out>16 或 P*C_in*C_out 过大）。
    规则（保守：宁可回退纯 torch 也不退化）：
      - M 界与旧版一致（真实模型 M=40000）；
      - C_out>16 回退（C_out=32 的 y-grid=2 冗余使 kernel 不占优）；
      - P*C_in*C_out > 5e7 回退（Cube 稠密 matmul 占优）。
    """
    if M <= 0 or P <= 0:
        return False
    if M < 256:
        return False
    if M > 80000:
        return False
    if C_out > 16:
        return False
    total_flops = P * C_in * C_out
    if total_flops > 5e7:
        return False
    return True


def _should_use_unfold(M, P, C_in, C_out, K, subm, inverse):
    """dense-unfold（im2col）GEMM 路由：宽通道（C_out>=64）非 inverse 层的实测标定。

    unfold 把 conv/subm 稀疏卷积写成**一次稠密 Cube GEMM**：
      output[o] = im2col(o, k*C_in+i) @ filters.reshape(K*C_in, C_out)
    其中 im2col[o,k,:] = features[邻居(o,k)]（无邻居处补零）。它省掉 pure 的 27 次
    per-k gather+mm+index_add（固定流水 ~1ms），只付：neighbor 构造 + 稠密 im2col
    gather（M*K*C_in 个 float）+ 一次 GEMM。实测（真实模型 C_out>=64 形态，见
    COUT64.md）在多数宽通道层 1.2-2.4x；但在两类形态不占优，路由回退 pure：
      - im2col 过大（>80e6 个元素，如 C_in=128 & M=34676 → 480MB）：稠密展开的
        内存搬运吃掉 GEMM 收益（实测 0.81x）；
      - GEMM 过小（<1e9 FLOP，如 K=3 的 c64x128、P=18k）：pure 本身便宜，unfold
        的固定开销不划算（实测 0.51x）。
    仅适用于「每个 (输出 voxel, offset) 至多一个输入 pair」的 conv/subm；inverse
    （deconv）一个输出可被同 offset 多个输入贡献，稠密展开会丢项，回退 pure。
    """
    if M <= 0 or P <= 0:
        return False
    if inverse:
        return False
    if C_out < 64:
        return False
    im2col_elems = M * K * C_in
    gemm_flops = im2col_elems * C_out
    if im2col_elems > 80e6:
        return False
    if gemm_flops < 1e9:
        return False
    if P < 30000:
        return False
    return True


def indice_conv_unfold(features, filters, indice_pairs, indice_num,
                       num_activate_out, inverse=False, subm=False, center=-1):
    """宽通道 conv/subm 的 dense-unfold（im2col）单 GEMM 路径。

    语义与 indice_conv_pure 一致（仅 conv/subm，不支持 inverse）：
      output[o, :] = sum_k features[neighbor_k(o), :] @ filters[k, :, :]
    对 conv/subm，每个 (o, k) 至多一个输入，故可零填充为稠密 patch 后一次 GEMM：
      output = im2col.view(M, K*C_in) @ filters.reshape(K*C_in, C_out)
    subm 的 center 项（features @ filters[center]，每个输出 voxel 的自贡献）由
    neighbor[:, center] = self 显式补入。数值为 NPU Cube fp32（与 pure 的 per-k mm
    同源，两者相对差 ~1e-7）。
    """
    K = indice_pairs.shape[0]
    C_in = features.shape[1]
    dtype = features.dtype
    device = features.device
    filters_k = filters.reshape(K, C_in, -1)
    C_out = filters_k.shape[2]
    M = int(num_activate_out)
    N = indice_pairs.shape[2]

    if M <= 0 or C_in <= 0:
        return torch.zeros(M, C_out, dtype=dtype, device=device)

    src = 1 if inverse else 0
    dst = 0 if inverse else 1
    ar = torch.arange(N, device=device)
    valid = ar[None, :] < indice_num[:, None]
    if subm and center >= 0:
        valid = valid.clone()
        valid[center, :] = False
    valid_flat = valid.reshape(-1)
    in_idx = indice_pairs[:, src, :].reshape(-1)[valid_flat].to(torch.int64)
    out_idx = indice_pairs[:, dst, :].reshape(-1)[valid_flat].to(torch.int64)
    k_idx = torch.arange(K, device=device).repeat_interleave(N)[valid_flat].to(torch.int64)

    # neighbor[o, k] = input voxel contributing to output o at offset k；-1 = 无邻居
    neighbor = torch.full((M, K), -1, dtype=torch.int64, device=device)
    if in_idx.numel() > 0:
        neighbor.view(-1).index_put_((out_idx * K + k_idx,), in_idx)
    if subm:
        # subm center：输出 voxel o 的「自身邻居」即 o（features[o] @ filters[center]）
        neighbor[:, center] = torch.arange(M, dtype=torch.int64, device=device)

    N_feat = features.shape[0]
    # features 末尾补一行零：缺失邻居 index 指向它，得到零行，避免逐元素掩码乘
    feats_pad = torch.cat([features, features.new_zeros(1, C_in)], 0)
    neighbor = neighbor.masked_fill(neighbor < 0, N_feat)
    im2col = feats_pad[neighbor]  # (M, K, C_in)
    W_conv = filters_k.reshape(K * C_in, C_out)
    return im2col.view(M, K * C_in) @ W_conv


def indice_conv_fast(features, filters, indice_pairs, indice_num,
                     num_activate_out, inverse=False, subm=False):
    """加速版 indice_conv：subm 中心用稠密 mm + 稀疏 offset 走 per-pair matmul kernel。

    host-gather + kernel-compute 混合：
      - host 把有效 pair 拍平成 k-major（build_k），gather 输入行；
      - Ascend C kernel 对每个 pair 算 gathered[p] @ filters[k_idx[p]]（精确 fp32，
        per-pair matmul，16 通道寄存器累加 + float4 FMA）；
      - host 用 output.index_add_(out_idx, partial) 完成 scatter 累加。
    与 indice_conv_pure 同签名/同语义；kernel 不可用（未编译/加载失败/非 NPU/
    dtype 不支持/形状路由）时静默回退 indice_conv_pure。
    宽通道（C_out>=64）层：kernel 的 SIMT per-pair matmul 随 C_in*C_out 退化
    （P=469K c64 实测 13ms），padded-bmm 的 Gpad 分配 + 未排序大 index_add 也
    不占优（实测 0.4-1.0x）；对 conv/subm 改走 dense-unfold 单 GEMM（见
    indice_conv_unfold，实测 1.2-2.4x），inverse/小 GEMM/超大 im2col 回退 pure。
    """
    K = indice_pairs.shape[0]
    C_in = features.shape[1]
    device = features.device
    dtype = features.dtype
    filters_k = filters.reshape(K, C_in, -1)  # (K, in, out)
    C_out = filters_k.shape[2]
    M = num_activate_out

    use_kernel = (
        _KERNEL_AVAILABLE
        and _is_npu(features)
        and dtype in (torch.float32, torch.float16)
        and _KERNEL_MATMUL_OP is not None
    )
    num_cpu = None
    if use_kernel:
        # 路由决策基于「预处理 + matmul + scatter」 vs 纯 torch 的实测性价比
        num_cpu = indice_num.detach().cpu()
        center = -1
        if subm:
            center = int(torch.argmax(num_cpu).item())
        P = int(num_cpu.sum().item())
        if subm and center >= 0:
            P -= int(num_cpu[center].item())
        if not _should_use_kernel(M, P, C_in, C_out):
            use_kernel = False

    use_bmm = (
        _is_npu(features)
        and dtype in (torch.float32, torch.float16)
        and num_cpu is not None
    )
    if not use_kernel and use_bmm:
        center = -1
        if subm:
            center = int(torch.argmax(num_cpu).item())
        P = int(num_cpu.sum().item())
        if subm and center >= 0:
            P -= int(num_cpu[center].item())
        if not _should_use_bmm(M, P, C_in, C_out):
            use_bmm = False

    if not use_kernel and not use_bmm:
        # 宽通道（C_out>=64）非 inverse 层：dense-unfold 单 GEMM 加速；否则 pure。
        # 先做零开销预筛（避免非宽形状多付一次 D2H），再算 P 走 _should_use_unfold。
        if not (C_out >= 64 and not inverse):
            return indice_conv_pure(features, filters, indice_pairs, indice_num,
                                    num_activate_out, inverse=inverse, subm=subm)
        if num_cpu is None:
            num_cpu = indice_num.detach().cpu()
        center = -1
        if subm:
            center = int(torch.argmax(num_cpu).item())
        P = int(num_cpu.sum().item())
        if subm and center >= 0:
            P -= int(num_cpu[center].item())
        if (_is_npu(features)
                and dtype in (torch.float32, torch.float16)
                and _should_use_unfold(M, P, C_in, C_out, K, subm, inverse)):
            return indice_conv_unfold(features, filters, indice_pairs, indice_num,
                                      num_activate_out, inverse=inverse, subm=subm,
                                      center=center)
        return indice_conv_pure(features, filters, indice_pairs, indice_num,
                                num_activate_out, inverse=inverse, subm=subm)

    if num_cpu is None:
        num_cpu = indice_num.detach().cpu()
    center = -1
    if subm:
        center = int(torch.argmax(num_cpu).item())

    output = torch.zeros(M, C_out, dtype=dtype, device=device)
    if subm:
        output = features @ filters_k[center]
    # mmcv 权重布局 (K, C_in, C_out)，reshape 后通常即连续 view；.contiguous() 兜底
    # 非连续情况（kernel 的 meta/校验要求 filters 连续）
    filters_k = filters_k.contiguous()

    gathered, k_idx, out_idx, s_idx = build_k(features, indice_pairs, indice_num,
                                              M, subm=subm, inverse=inverse, center=center)
    if gathered.numel() > 0:
        if use_kernel:
            partial = _KERNEL_MATMUL_OP(gathered, filters_k, k_idx, s_idx, gathered.shape[0])
            output.index_add_(0, out_idx.to(torch.int64), partial)
        else:
            output = output + _bmm_sparse(gathered, k_idx, out_idx, filters_k, M)
    return output


def apply_mmcv_sparse_conv_patch():
    """monkey-patch mmcv.ops.sparse_ops 的 get_indice_pairs / indice_conv。"""
    _mmcv_sparse_ops.get_indice_pairs = get_indice_pairs_fast
    _mmcv_sparse_ops.indice_conv = indice_conv_fast
    return True


# 幂等：import 即应用
apply_mmcv_sparse_conv_patch()
