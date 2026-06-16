# 接口支持清单

| 算子名 | Torch API | ascend910b | ascend910_93 | ascend950 |
| ------ | --------- | :--------: | :----------: | :-------: |
| abs    | `torch.ops.ops_multimodal_fusion.abs(Tensor x) -> Tensor` | ✓ | ✓ | - |
| upsample_linear1d | `torch.ops.ops_multimodal_fusion.upsample_linear1d(Tensor input, int output_size, bool align_corners=False, float scale=-1.) -> Tensor` | - | - | ✓ |

> ✓ 表示已支持，- 表示暂未支持。详细的接口说明请参考[接口列表](api_list.md)。
