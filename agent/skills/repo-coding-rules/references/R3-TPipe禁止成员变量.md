# R3: TPipe 禁止作为成员变量

## 说明

`TPipe` 管理硬件流水线队列，其生命周期必须与单次 Kernel 调用绑定。作为类成员变量可能导致多次调用间状态残留或重复初始化。

## 错误示例

```cpp
template <typename T>
class AddKernel {
public:
    AscendC::TPipe pipe_;  // 禁止：作为类成员变量
};
```

## 正确示例

```cpp
// Kernel 入口函数中创建 TPipe（生命周期绑定单次 <<<>>> 调用），通过指针传入 Init
// tiling 以 scalar（blockLength/tileSize）随 <<<>>> 传入，无需解析 TilingData
template <typename T>
__global__ __aicore__ __vector__ void add_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                  int64_t totalLength, int64_t blockLength, uint32_t tileSize) {
    AscendC::TPipe pipe;
    AddKernel<T> op;
    op.Init(x, y, z, totalLength, blockLength, tileSize, &pipe);
    op.Process();
}
```