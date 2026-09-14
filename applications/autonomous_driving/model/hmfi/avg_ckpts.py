# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# -*- coding: utf-8 -*-
"""Average state_dicts of several fusion checkpoints (epoch-snapshots of the same run)."""
import sys, torch
out = sys.argv[1]
paths = sys.argv[2:]
print(f"averaging {len(paths)} ckpts -> {out}")
sds = []
meta = {}
for p in paths:
    ck = torch.load(p, map_location='cpu')
    sd = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    sds.append(sd)
    if isinstance(ck, dict):
        meta = ck
avg = {}
for k in sds[0]:
    stacked = torch.stack([sd[k].float() for sd in sds], dim=0)
    avg[k] = stacked.mean(dim=0)
# preserve non-tensor buffers like num_batches_tracked (int) - they should be equal across ckpts
meta = dict(meta)
meta['state_dict'] = avg
meta['epoch'] = 'avg'
torch.save(meta, out)
print("saved", out, "keys", len(avg))
