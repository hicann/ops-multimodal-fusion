# R2: 动态获取 CoreNum

## 说明

禁止硬编码核数。不同芯片型号、不同形态（PCIE/EP）的 AI Core 数量不同，硬编码会导致跨平台兼容性问题。

## 错误示例

```cpp
constexpr int numBlocks = 28;  // 硬编码核数
add_kernel<float><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength, blockLength, tileSize);
```

## 正确示例

```cpp
// host 侧 tiling 中动态取核数
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
TORCH_CHECK(coreNum > 0, "coreNum must be positive.");
int64_t numBlocks = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
add_kernel<float><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength, blockLength, tileSize);
```