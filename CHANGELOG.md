# CHANGELOG

> 本文档记录各版本的重要变更，版本按时间倒序排列。

## v1.0.0
发布日期：2026-06-02

ops-multimodal-fusion 首个版本发布。
本仓库提供多模态融合类算子的实现，支持昇腾 NPU 平台。

### 🚀 关键特性

- 【算子实现】基于 AscendC 的 PyTorch 自定义算子库
- 【工程能力】使用 fast kernel launch（`<<<>>>` 直调）方式实现高性能 NPU 算子
- 【工程能力】编译为 Python wheel 包供直接安装使用
- 【文档】提供快速入门指南、贡献指南、安全声明等完整文档
- 【测试】包含算子测试框架和测试用例

### 📦 支持的算子

共 132 个算子。abs 同时支持 910B（arch22）与 950（arch35），其余算子支持 950（arch35）。完整接口签名详见[接口支持清单](docs/zh/op_support_list.md)。

| 算子名称 | 功能描述 | 支持数据类型 |
|---------|---------|-------------|
| abs | 张量绝对值运算 | FP32, FP16 |
| adaptive_avg_pool2d | 二维自适应平均池化 | FP16, FP32 |
| add | 张量逐元素加法 | FP16, FP32, INT32 |
| addmv | 矩阵-向量乘加运算 | BF16, FP16, FP32, INT32 |
| airy_ai | Airy Ai 函数 | FP32 |
| angle | 复数辐角 | FP16, FP32 |
| any | 沿维度判断是否存在真值 | BOOL, FP16, FP32, INT32 |
| avg_pool2d | 二维平均池化 | FP32 |
| bessel_j0 | 第一类 0 阶贝塞尔函数 | FP32 |
| bessel_j1 | 第一类 1 阶贝塞尔函数 | FP32 |
| bessel_y0 | 第二类 0 阶贝塞尔函数 | FP32 |
| bessel_y1 | 第二类 1 阶贝塞尔函数 | FP32 |
| binomial | 二项分布采样 | FP16, FP32 |
| bitwisenot | 按位取反 | INT16, INT32 |
| c2_accuracy | Top-K 分类准确率 | FP16, FP32 |
| c2_affine_channel | 仿射通道变换 | FP16, FP32 |
| c2_batch_moments | 批统计矩计算 | FP16, FP32 |
| c2_batch_permutation | 批排列 | FP16, FP32, INT32 |
| c2_boolean_mask | 布尔掩码提取 | FP16, FP32, INT32 |
| c2_boolean_unmask | 布尔掩码还原 | FP16, FP32, INT32 |
| c2_bucketize | 按边界分桶 | FP16, FP32 |
| c2_cbrt | 立方根运算 | FP16, FP32 |
| c2_lars | LARS 学习率重标定 | BF16, FP16, FP32 |
| cauchy | 柯西分布采样 | FP16, FP32 |
| cdist_backward | 成对距离反向梯度 | FP16, FP32 |
| chebyshev_polynomial_t | 切比雪夫多项式 Tn | FP32, INT32 |
| chebyshev_polynomial_u | 切比雪夫多项式 Un | FP32, INT32 |
| chebyshev_polynomial_v | 切比雪夫多项式 Vn | FP32, INT32 |
| chebyshev_polynomial_w | 切比雪夫多项式 Wn | FP32, INT32 |
| complex | 由实部与虚部构造复数张量 | FP16, FP32 |
| conjphysical | 复数共轭 | FP16, FP32 |
| copysign | 复制符号值 | FP16, FP32 |
| cummax | 沿维度累积最大值 | FP16, FP32 |
| cumprod | 沿维度累积乘积 | FP16, FP32 |
| depthwise_conv3d | 三维深度卷积 | FP16, FP32 |
| dequant_swiglu_quant | 反量化-SwiGLU-量化融合 | BF16, FP16, FP32, INT32, INT8 |
| digamma | 双伽马函数 | FP32 |
| dirichlet | 狄利克雷分布采样 | FP16, FP32 |
| dynamic_quant | 动态量化 | BF16, FP16 |
| entr | 熵函数 entr(x) = x·ln(x) | FP16, FP32 |
| erfcx | 缩放互补误差函数 | FP32 |
| exp | 自然指数运算 | FP16, FP32 |
| exp2 | 以 2 为底指数运算 | FP16, FP32 |
| exponential | 指数分布采样 | FP16, FP32 |
| fft_conj_symmetry | FFT 共轭对称性补全 | Complex32, Complex64 |
| foreach_ceil | 批量张量向上取整 | FP16, FP32 |
| foreach_floor | 批量张量向下取整 | FP16, FP32 |
| foreach_frac | 批量张量取小数部分 | FP16, FP32 |
| fractional_max_pool2d | 二维分数最大池化 | FP16, FP32 |
| fractional_max_pool3d | 三维分数最大池化 | FP16, FP32 |
| frexp | 浮点数分解为尾数与指数 | FP16, FP32 |
| gamma | 伽马分布采样 | FP16, FP32 |
| gelu | GeLU 激活函数 | FP32 |
| geometric | 几何分布采样 | FP16, FP32 |
| gru_cell | GRU 循环单元 | FP16, FP32 |
| hermite_polynomial_h | 埃尔米特多项式 Hn | FP32, INT32 |
| hermite_polynomial_he | 归一化埃尔米特多项式 Hen | FP32, INT32 |
| hypot | 直角三角形斜边计算 | FP16, FP32 |
| igamma | 正则化下不完全伽马函数 | FP32 |
| igammac | 正则化上不完全伽马函数 | FP32 |
| index_copy | 沿维度按索引复制 | FP16, FP32, INT32 |
| index_reduce | 沿维度按索引归约 | FP16, FP32, INT32 |
| int_repr | 量化张量整数表示 | INT8, UINT8, INT32 |
| kaiserwindow | Kaiser 窗函数 | FP32 |
| kthvalue | 沿维度取第 k 小值 | FP16, FP32, INT32 |
| laguerre_polynomial_l | 拉盖尔多项式 Ln | FP32, INT32 |
| layer_norm | 层归一化 | FP16, FP32 |
| lcm | 最小公倍数 | INT32 |
| leftshift | 按位左移 | INT32 |
| legendre_polynomial_p | 勒让德多项式 Pn | FP32, INT32 |
| log_add_exp2 | 以 2 为底的 log-add-exp | FP16, FP32 |
| log_normal | 对数正态分布采样 | FP16, FP32 |
| logcumsumexp | 沿维度累积 log-sum-exp | FP16, FP32 |
| logicalxor | 逻辑异或 | BOOL |
| logit | Logit 函数 | FP16, FP32 |
| logndtr | 高斯累积分布函数的对数 | FP32 |
| lstm_cell | LSTM 循环单元 | FP16, FP32 |
| make_per_tensor_quantized | 按张量量化 | INT8, UINT8, INT32 |
| matrix_exp_util | 矩阵指数辅助运算 | FP16, FP32 |
| max | 沿维度求最大值 | FP16, FP32, INT32 |
| max_unpool2d | 二维反最大池化 | FP16, FP32 |
| max_unpool3d | 三维反最大池化 | FP16, FP32 |
| mean | 沿维度求均值 | FP16, FP32 |
| mode | 沿维度求众数 | FP16, FP32, INT32 |
| modified_bessel_k0 | 第二类 0 阶修正贝塞尔函数 | FP32 |
| modified_bessel_k1 | 第二类 1 阶修正贝塞尔函数 | FP32 |
| mul | 张量逐元素乘法 | FP16, FP32, INT32 |
| multi_margin_loss | 多类间隔损失 | FP16, FP32 |
| multilabel_margin_loss | 多标签间隔损失 | FP16, FP32 |
| multinomial | 多项分布采样 | FP16, FP32 |
| nested_add_pad | 嵌套张量补齐填充 | FP16, FP32, INT32 |
| nested_binary_op | 嵌套张量二元运算 | FP16, FP32, INT32 |
| nested_bmm | 嵌套张量批量矩阵乘 | FP16, FP32 |
| nested_remove_pad | 嵌套张量去除填充 | FP16, FP32, INT32 |
| nextafter | 下一个可表示浮点值 | FP32 |
| norm | 沿维度求范数 | FP32 |
| pdist | 成对距离 | BF16, FP16, FP32 |
| pdist_backward | 成对距离反向梯度 | BF16, FP16, FP32 |
| poisson | 泊松分布采样 | FP16, FP32 |
| polar | 由模与辐角构造复数张量 | FP16, FP32 |
| polygamma | 多伽马函数 | FP32 |
| put | 按索引写入 | FP16, FP32, INT32 |
| quantized_relu | 量化 ReLU | INT8, UINT8, INT32 |
| rms_norm_gated | 门控 RMS 归一化 | BF16, FP16, FP32 |
| rsqrt | 平方根倒数 | FP16, FP32 |
| scaled_modified_bessel_k0 | 缩放第二类 0 阶修正贝塞尔函数 | FP32 |
| scaled_modified_bessel_k1 | 缩放第二类 1 阶修正贝塞尔函数 | FP32 |
| searchsorted | 有序序列二分查找 | FP16, FP32, INT32 |
| semi_structured_linear | 半结构化稀疏线性层 | FP32 |
| shifted_chebyshev_polynomial_t | 移位切比雪夫多项式 Tn | FP32, INT32 |
| shifted_chebyshev_polynomial_u | 移位切比雪夫多项式 Un | FP32, INT32 |
| shifted_chebyshev_polynomial_v | 移位切比雪夫多项式 Vn | FP32, INT32 |
| shifted_chebyshev_polynomial_w | 移位切比雪夫多项式 Wn | FP32, INT32 |
| sigmoid | Sigmoid 激活函数 | FP16, FP32 |
| sin | 正弦运算 | FP16, FP32 |
| sinc | Sinc 函数 | FP16, FP32 |
| sparse_binary_intersect | 稀疏二进制索引交集 | FP16, FP32, INT32 |
| sparse_mask_intersection | 稀疏掩码交集 | FP16, FP32, INT32 |
| sparse_mask_projection | 稀疏掩码投影 | FP16, FP32, INT32 |
| spherical_bessel_j0 | 0 阶球面贝塞尔函数 | FP32 |
| sqrt | 平方根运算 | FP16, FP32 |
| sum | 沿维度求和 | FP16, FP32, INT32 |
| swi_glu | SwiGLU 融合运算 | BF16, FP16, FP32 |
| take | 按索引取值 | FP16, FP32, INT32 |
| tril_indices | 下三角索引生成 | INT32, INT64 |
| triu_indices | 上三角索引生成 | INT32, INT64 |
| unpack_pivots | LU 主元解包 | INT32, INT64 |
| upsample_linear1d | 一维线性插值上采样 | FP16, FP32 |
| upsample_nearest1d | 一维最近邻上采样 | FP16, FP32 |
| upsample_trilinear3d | 三线性插值上采样 | FP16, FP32 |
| weight_norm | 权重归一化 | FP16, FP32 |
| zeta | Hurwitz Zeta 函数 | BF16, FP16, FP32 |

### 📌 版本配套

**ops-multimodal-fusion子包及相关组件与CANN版本配套关系**

| CANN子包版本 | 版本源码标签 | 配套CANN版本 |
|-------------|-------------|-------------|
| cann-ops-multimodal-fusion 1.0.0 | 1.0.0 | CANN 8.5.0 |