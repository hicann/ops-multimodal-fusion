# 接口支持清单

| 算子名 | Torch API | ascend910b | ascend910_93 | ascend950 |
| ------ | --------- | :--------: | :----------: | :-------: |
| ------ | --------- | :--------: | :----------: | :-------: |
| abs    | `torch.ops.ops_multimodal_fusion.abs(Tensor x) -> Tensor` | ✓ | ✓ | - |
| adaptive_avg_pool2d | `torch.ops.ops_multimodal_fusion.adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor` | - | - | ✓ |
| angle | `torch.ops.ops_multimodal_fusion.angle(Tensor input) -> Tensor` | - | - | ✓ |
| bessel_j0 | `torch.ops.ops_multimodal_fusion.bessel_j0(Tensor x) -> Tensor` | - | - | ✓ |
| bessel_j1 | `torch.ops.ops_multimodal_fusion.bessel_j1(Tensor x) -> Tensor` | - | - | ✓ |
| bessel_y0 | `torch.ops.ops_multimodal_fusion.bessel_y0(Tensor x) -> Tensor` | - | - | ✓ |
| bessel_y1 | `torch.ops.ops_multimodal_fusion.bessel_y1(Tensor x) -> Tensor` | - | - | ✓ |
| cauchy | `torch.ops.ops_multimodal_fusion.cauchy(Tensor x, float median=0.0, float sigma=1.0, int seed=0) -> Tensor` | - | - | ✓ |
| complex | `torch.ops.ops_multimodal_fusion.complex(Tensor real, Tensor imag) -> Tensor` | - | - | ✓ |
| conjphysical | `torch.ops.ops_multimodal_fusion.conjphysical(Tensor x) -> Tensor` | - | - | ✓ |
| cummax | `torch.ops.ops_multimodal_fusion.cummax(Tensor x, int dim) -> (Tensor values, Tensor indices)` | - | - | ✓ |
| cumprod | `torch.ops.ops_multimodal_fusion.cumprod(Tensor x, int dim) -> Tensor` | - | - | ✓ |
| depthwise_conv3d | `torch.ops.ops_multimodal_fusion.depthwise_conv3d(Tensor input, Tensor weight, Tensor? bias, int[3] stride, int[3] padding, int[3] dilation) -> Tensor` | - | - | ✓ |
| digamma | `torch.ops.ops_multimodal_fusion.digamma(Tensor x) -> Tensor` | - | - | ✓ |
| entr | `torch.ops.ops_multimodal_fusion.entr(Tensor x) -> Tensor` | - | - | ✓ |
| erfcx | `torch.ops.ops_multimodal_fusion.erfcx(Tensor x) -> Tensor` | - | - | ✓ |
| foreach_ceil | `torch.ops.ops_multimodal_fusion.foreach_ceil(Tensor[] tensors) -> Tensor[]` | - | - | ✓ |
| fractional_max_pool2d | `torch.ops.ops_multimodal_fusion.fractional_max_pool2d(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples) -> (Tensor, Tensor)` | - | - | ✓ |
| fractional_max_pool3d | `torch.ops.ops_multimodal_fusion.fractional_max_pool3d(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples) -> (Tensor, Tensor)` | - | - | ✓ |
| index_reduce | `torch.ops.ops_multimodal_fusion.index_reduce(Tensor self, int dim, Tensor index, Tensor source, str reduce, bool include_self=True) -> Tensor` | - | - | ✓ |
| int_repr | `torch.ops.ops_multimodal_fusion.int_repr(Tensor x) -> Tensor` | - | - | ✓ |
| kthvalue | `torch.ops.ops_multimodal_fusion.kthvalue(Tensor x, int k, int dim, bool keepdim) -> (Tensor values, Tensor indices)` | - | - | ✓ |
| log_normal | `torch.ops.ops_multimodal_fusion.log_normal(Tensor x, float mean=1.0, float std=2.0, int seed=0) -> Tensor` | - | - | ✓ |
| logndtr | `torch.ops.ops_multimodal_fusion.logndtr(Tensor x) -> Tensor` | - | - | ✓ |
| make_per_tensor_quantized | `torch.ops.ops_multimodal_fusion.make_per_tensor_quantized(Tensor x, float scale, int zero_point) -> Tensor` | - | - | ✓ |
| max_unpool2d | `torch.ops.ops_multimodal_fusion.max_unpool2d(Tensor input, Tensor indices, int[2] kernel_size, int[2] stride, int[2] padding, int[2] output_size) -> Tensor` | - | - | ✓ |
| max_unpool3d | `torch.ops.ops_multimodal_fusion.max_unpool3d(Tensor input, Tensor indices, int[3] kernel_size, int[] stride, int[] padding, int[] output_size) -> Tensor` | - | - | ✓ |
| mode | `torch.ops.ops_multimodal_fusion.mode(Tensor x, int dim, bool keepdim) -> (Tensor values, Tensor indices)` | - | - | ✓ |
| multi_margin_loss | `torch.ops.ops_multimodal_fusion.multi_margin_loss(Tensor input, Tensor target, Scalar p=1, Scalar margin=1.0, Tensor? weight=None, int reduction=1) -> Tensor` | - | - | ✓ |
| multilabel_margin_loss | `torch.ops.ops_multimodal_fusion.multilabel_margin_loss(Tensor input, Tensor target, int reduction=1) -> Tensor` | - | - | ✓ |
| polygamma | `torch.ops.ops_multimodal_fusion.polygamma(Tensor x, int n) -> Tensor` | - | - | ✓ |
| searchsorted | `torch.ops.ops_multimodal_fusion.searchsorted(Tensor sorted_sequence, Tensor values, bool out_int32=False, bool right=False) -> Tensor` | - | - | ✓ |
| shifted_chebyshev_polynomial_t | `torch.ops.ops_multimodal_fusion.shifted_chebyshev_polynomial_t(Tensor x, Tensor n) -> Tensor` | - | - | ✓ |
| shifted_chebyshev_polynomial_u | `torch.ops.ops_multimodal_fusion.shifted_chebyshev_polynomial_u(Tensor x, Tensor n) -> Tensor` | - | - | ✓ |
| shifted_chebyshev_polynomial_v | `torch.ops.ops_multimodal_fusion.shifted_chebyshev_polynomial_v(Tensor x, Tensor n) -> Tensor` | - | - | ✓ |
| shifted_chebyshev_polynomial_w | `torch.ops.ops_multimodal_fusion.shifted_chebyshev_polynomial_w(Tensor x, Tensor n) -> Tensor` | - | - | ✓ |
| tril_indices | `torch.ops.ops_multimodal_fusion.tril_indices(int row, int col, int offset, bool out_int32) -> Tensor` | - | - | ✓ |
| triu_indices | `torch.ops.ops_multimodal_fusion.triu_indices(int row, int col, int offset, bool out_int32) -> Tensor` | - | - | ✓ |
| upsample_linear1d | `torch.ops.ops_multimodal_fusion.upsample_linear1d(Tensor input, int output_size, bool align_corners=False, float scale=-1.) -> Tensor` | - | - | ✓ |
| upsample_trilinear3d | `torch.ops.ops_multimodal_fusion.upsample_trilinear3d(Tensor input, int[3] output_size, bool align_corners=False, float scales_d=-1., float scales_h=-1., float scales_w=-1.) -> Tensor` | - | - | ✓ |

> ✓ 表示已支持，- 表示暂未支持。详细的接口说明请参考[接口列表](api_list.md)。
