# R4: 禁止用数组为每核分配独立地址

本仓算子不使用 TilingData 结构体：tiling 参数（`numBlocks` / `blockLength` / `tileSize`）由 host 侧 `calc_tiling_params` 计算后，以 scalar 形式随 `<<<numBlocks, nullptr, stream>>>` 直接传入 kernel。因此「禁止用数组为每个核分配独立地址」这一原则依然成立：无论 tiling 走 struct 还是 scalar，都不可为每核预先展开一个长度依赖核数的数组，而应记载核间间隔、由 kernel 用 `GetBlockIdx()` 自行计算本核偏移。

下面以一个假想的 TilingData 结构体说明该反模式（本仓实际以 scalar 传参，原理一致）。

## 错误示例

```cpp
// ❌ 错误：数组长度依赖核数，核数不确定
struct FooTilingData {
    uint32_t startOffset[MAX_CORE_NUM];
    uint32_t calNum[MAX_CORE_NUM];
};
```

问题：核数由运行时动态获取，编译期不可知，数组长度无法合法定义；同时 TilingData 体积膨胀。

## 正确示例

```cpp
// ✅ 正确：记载核间间隔，Kernel 自己算
struct FooTilingData {
    uint32_t totalN;     // 总元素数
    uint32_t perCoreN;   // 每核基础分配量
    uint32_t remainder;  // 余数（前 remainder 个核多分 1 个）
};

// Kernel 侧自行计算（本仓等价：以 scalar blockLength 传入，用 GetBlockIdx 算偏移）：
// int64_t blockIdx = AscendC::GetBlockIdx();
// uint32_t myOffset = blockIdx * perCoreN + min(blockIdx, remainder);
// uint32_t myCount  = perCoreN + (blockIdx < remainder ? 1 : 0);
```