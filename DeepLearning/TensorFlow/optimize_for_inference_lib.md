### `optimize_for_inference_lib.py`解读

1. `optimize_for_inference_lib.py`的注释：
- `optimize_for_inference_lib.py`主要用于删除图中仅用于训练的部分：`GraphDef`中有一部分的常见变换是仅仅针对训练过程的，当网络仅用于预测时可以去除这些部分减少计算量。这些优化主要包括：
  
  -  删除仅用于训练的算子，比如`checkpoint saving`
  -  删除图中从未进行的操作(冗余节点)
  -  删除调试(debug)算子例如`CheckNumerics`
  -  将批标准化(batch norm)算子折叠到预计算(pre-calculate)中
     - 这个仅仅针对`BatchNormWithGlobalNormalization`与`FusedBatchNorm`两种BN
  -  将通用操作融合成统一的版本

- 这个脚本的输入是一个`pb`文件，输出一个优化过的`pb`文件
