# AscendC 编码规则 R5-R10

## R5: 圈复杂度 ≤ 20

函数的 Cyclomatic Complexity 不超过 20，超过需拆分子函数。

**错误示例：圈复杂度超标的长函数**

```cpp
// CCN = 1（基础）+ 1（if）+ 1（for）+ 1（if）+ ... = 23+
void ProcessMatrix(int m, int n, int k,
                   const float* A, const float* B, float* C)
{
    if (m > 0) {
        for (int i = 0; i < m; i++) {
            if (i % 2 == 0) {
                ProcessEvenRow(i, A, B, C, n, k);
            } else {
                ProcessOddRow(i, A, B, C, n, k);
            }
            // ... 还有 20 行条件判断
        }
    } else if (m == 0) {
        ClearZero(C, n, k);
    } else {
        HandleError();
    }
}
```

**正确示例：拆分成子函数**

```cpp
// CCN = 3 (m > 0, i%2==0, m == 0)
void ProcessMatrix(int m, int n, int k,
                   const float* A, const float* B, float* C)
{
    if (m > 0) {
        ProcessRows(0, m, A, B, C, n, k);
    } else {
        HandleMatrixZeroOrError(m, C, n, k);
    }
}

// CCN = 2 (i%2==0 分支)
void ProcessRows(int start, int end, ...) {
    for (int i = start; i < end; i++) {
        (i % 2 == 0) ? ProcessEvenRow(...) : ProcessOddRow(...);
    }
}
```

**计算方式**（按分支数量）：
- `if` / `else if`：每出现一次 +1
- `for` / `while`：每出现一次 +1
- `case`（switch 分支）：每出现一次 +1
- `&&` / `||`（逻辑运算）：每出现一次 +1

---

## R6: 嵌套深度 ≤ 5

最大嵌套层级不超过 5，超过需提取内层循环/分支为独立函数。

**错误示例：嵌套深度 7 层**

```cpp
void CopyBlocks(...) {
    for (int batch = 0; batch < numBatches; batch++) {                 // 1
        for (int block = 0; block < numBlocks; block++) {                // 2
            for (int tile = 0; tile < numTiles; tile++) {                // 3
                for (int row = 0; row < tileHeight; row++) {             // 4
                    for (int col = 0; col < tileWidth; col++) {          // 5
                        if (mask[batch][block][tile][row][col]) {        // 6
                            if (validElement(batch, block, tile, row, col)) {  // 7 ← 超标
                                dst[...] = src[...];
                            }
                        }
                    }
                }
            }
        }
    }
}
```

**正确示例：拆分为单层处理函数**

```cpp
void CopyBlocks(...) {
    for (int batch = 0; batch < numBatches; batch++) {
        for (int block = 0; block < numBlocks; block++) {
            CopyBlockBatch(batch, block, ...);
        }
    }
}

// 嵌套深度：3（tile + row + col）
void CopyBlockBatch(int batch, int block, ...) {
    for (int tile = 0; tile < numTiles; tile++) {
        CopyTile(batch, block, tile, ...);
    }
}

// 嵌套深度：4（row + col + mask + valid）
void CopyTile(int batch, int block, int tile, ...) {
    for (int row = 0; row < tileHeight; row++) {
        CopyTileRow(batch, block, tile, row, ...);
    }
}

// 嵌套深度：4（col + mask + validCheck + valid）
void CopyTileRow(int batch, int block, int tile, int row, ...) {
    for (int col = 0; col < tileWidth; col++) {
        if (mask[...]) {
            CopyIfValid(batch, block, tile, row, col, src, dst);
        }
    }
}
```

---

## R7: 函数行数 ≤ 50 (NBNC)

NBNC（Non-Blank Non-Comment lines，非空非注释行）不超过 50，超过需拆分为多个函数。

**统计方式**：
- 空行不算
- 纯注释行（以 `//` 或 `/* */` 开头）不算
- 混合行（代码 + 注释）只计代码部分
- 函数签名 + 函数体开闭括号 `{` / `}` 各算 1 行
- 模板函数 `template <...>` 不计行

**错误示例：NBNC = 65 行**

```cpp
template <typename T>
void ComputeAdd(
    __gm__ T* x, __gm__ T* y, __gm__ T* z,
    int64_t totalLength, int64_t blockLength, uint32_t tileSize)
{
    __gm__ T* outPtr = z;
    __gm__ T* xPtr = x;
    __gm__ T* yPtr = y;
    int64_t elemPerTile = tileSize / sizeof(T);
    int64_t tileNum = blockLength / elemPerTile;
    int64_t tailNum = blockLength - tileNum * elemPerTile;
    int64_t blockIdx = AscendC::GetBlockIdx();
    // ... 60 行逻辑处理
    return;
}
```

**正确示例：按功能拆分**

```cpp
template <typename T>
void ComputeAdd(
    __gm__ T* x, __gm__ T* y, __gm__ T* z,
    int64_t totalLength, int64_t blockLength, uint32_t tileSize)
{
    // NBNC = 15
    PreparePointers(x, y, z);
    int64_t tileNum = blockLength / (tileSize / sizeof(T));
    for (int64_t i = 0; i < tileNum; i++) {
        ComputeOneTile(i);
    }
    WriteBackResults(z);
}
```

---

## R8: 除零防御

除法/取模运算的被除数必须校验非零，特别是来自外部输入的变量（如 `coreNum`、`blockCount`）。

**错误示例：`coreNum` 可能为 0**

```cpp
void TilingKernel(int totalElements, int coreNum)
{
    int perCore = totalElements / coreNum;   // ← 风险！coreNum 可能为 0
    int remainder = totalElements % coreNum; // ← 同样风险
    // ...
}
```

**正确示例：先校验再运算**

```cpp
void TilingKernel(int totalElements, int coreNum)
{
    TORCH_CHECK(coreNum > 0, "coreNum must be positive, got ", coreNum);
    int perCore = totalElements / coreNum;
    int remainder = totalElements % coreNum;
    // ...
}
```

**特别关注**：所有通过 `GetCoreNumAiv` 获取的数值都可能返回 0（获取失败），必须先判零再用于除法。

---

## R9: 许可证头

所有源码文件（`.asc`，含融合算子拆出的同目录 `.h`）必须包含标准许可证头。

**标准许可证头模板**：

```cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
```

**例外文件**：
- 测试脚本 `tests/{op}/test_{op}.py`（使用 shebang + coding + 自有版权头，而非 CANN 头）
- `.gitignore` / `requirements.txt` 等配置文件
- 自动生成的文件（在文件顶部注明自动生成来源）

---

## R10: 禁止 extern 引用

禁止在 Kernel/Host 代码中使用 `extern "C"` 声明外部函数接口，应通过头文件 include 引入。

**错误示例：直接 extern 声明外部函数**

```cpp
// ❌ 在 .asc 中直接 extern 声明其它文件里的函数
extern "C" void add_launch(uint8_t* x, uint8_t* y, uint8_t* z,
                           uint32_t numBlocks, int64_t blockLength, void* stream);
```

**正确示例：单文件内联 / 用 #include 引入**

```cpp
// 本仓算子为单文件形态：kernel 入口 __global__ __aicore__ 函数直接内联在 {op}.asc 中，
// 由 NPU dispatch 函数用 <<<numBlocks, nullptr, stream>>> 调用，无需任何 extern 声明。
// 复杂融合算子若把 kernel/tiling 拆到同目录头文件，也用 #include 引入而非 extern：
#include "add_kernel.h"   // 相关声明在该头文件中
```

**例外**：本仓 kernel 入口为模板函数 `template <typename T> __global__ __aicore__ __vector__ void add_kernel(...)`，直接内联在 `.asc` 中并以 `<<<>>>` 调用，本身无需 `extern "C"`。