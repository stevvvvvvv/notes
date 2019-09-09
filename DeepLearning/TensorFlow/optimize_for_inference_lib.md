##### 输出`pb`类型`graph`:

```
from tensorflow.python.framework import graph_io
graph_io.write_graph(self.sess.graph, '/save_path/', 'input_graph.pb')
```
或者(save model)：
```
import tensorflow as tf
import os
from tensorflow.python.framework import graph_util

pb_file_path = os.getcwd()

with tf.Session(graph=tf.Graph()) as sess:
    x = tf.placeholder(tf.int32, name='x')
    y = tf.placeholder(tf.int32, name='y')
    b = tf.Variable(1, name='b')
    xy = tf.multiply(x, y)
    # 这里的输出需要加上name属性
    op = tf.add(xy, b, name='op_to_store')

    sess.run(tf.global_variables_initializer())

    # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

    # 测试 OP
    feed_dict = {x: 10, y: 3}
    print(sess.run(op, feed_dict))

    # 写入序列化的 PB 文件
    with tf.gfile.FastGFile(pb_file_path+'model.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())

    # INFO:tensorflow:Froze 1 variables.
    # Converted 1 variables to const ops.
    # 31
    
    
    # 官网有误，写成了 saved_model_builder  
    builder = tf.saved_model.builder.SavedModelBuilder(pb_file_path+'savemodel')
    # 构造模型保存的内容，指定要保存的 session，特定的 tag, 
    # 输入输出信息字典，额外的信息
    builder.add_meta_graph_and_variables(sess,
                                       ['cpu_server_1'])


# 添加第二个 MetaGraphDef 
# with tf.Session(graph=tf.Graph()) as sess:
#  ...
#   builder.add_meta_graph([tag_constants.SERVING])
# ...

builder.save()  # 保存 PB 模型
```
保存好以后在`save_model_dir`目录下，会有一个`saved_model.pb`文件以及`variables`文件夹。这种方法对应的导入模型的方法：
```
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ['cpu_1'], pb_file_path+'savemodel')
    sess.run(tf.global_variables_initializer())

    input_x = sess.graph.get_tensor_by_name('x:0')
    input_y = sess.graph.get_tensor_by_name('y:0')

    op = sess.graph.get_tensor_by_name('op_to_store:0')

    ret = sess.run(op,  feed_dict={input_x: 5, input_y: 5})
    print(ret)
# 只需要指定要恢复模型的 session，模型的 tag，模型的保存路径即可,使用起来更加简单
```
或者：
```

constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output'])
pb_file_path = os.path.join(pb_file_path, 'mnist_test.pb')

with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
    f.write(constant_graph.SerializeToString())
sess.close()
```
- 直接保存的图偶尔会报错类型问题，使用以下方式转码再保存：

报错：
```
Invalid argument: Input 0 of node InceptionResnetV1/Block8/Branch_1/Conv2d_0c_3x1/BatchNorm/cond/AssignMovingAvg_1/Switch was passed float from InceptionResnetV1/Block8/Branch_1/Conv2d_0c_3x1/BatchNorm/moving_variance:0 incompatible with expected float_ref.
```
转码方法：(https://github.com/tensorflow/tensorflow/issues/3628)
```
for node in sess.graph_def.node:
    if node.op == 'RefSwitch':
        node.op = 'Switch'
        for index in range(len(node.input)):
            node.input[index] = node.input[index] + '/read'
    elif node.op == 'AssignSub':
        node.op = 'Sub'
        if 'use_locking' in node.attr: del node.attr['use_locking']

constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output'])
pb_file_path = os.path.join(pb_file_path, 'mnist_test.pb')

with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
    f.write(constant_graph.SerializeToString())
```
在不知道`tensor name`的情况下保存图需要给`add_meta_graph_and_variables`方法传入第三个参数：`signature_def_map`

##### 固化`freeze_graph`:

```
bazel build tensorflow/python/tools:freeze_graph&& \
bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=some_graph_def.pb \
__input_checkpoint=model.ckpt \
output_graph=/tmp/forzen_graph.pb \
--output_node_names=softmax
```
例如：
```
bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=./input_graph.pb \
--input_checkpoint=./VGGnet_fast_rcnn_iter_52700.ckpt \
--output_graph=./froze_graph1.pb \
--output_node_names=rois/Reshape,rois/PyFunc
```

##### 优化`optimize_for_inference`:
```
bazel build tensorflow/python/tools:optimize_for_inference&& \
bazel-bin/tensorflow/python/tools/optimize_for_inference \
--input=frozen_inception_graph.pb \
--output=optimized_inception_graph.pb \
--frozen_graph=True \
--input_names=Mul \
--output_names=softmax
```
例如：
```
bazel-bin/tensorflow/python/tools/optimize_for_inference
--input=froze_ctc.pb
--output=optimized_ctc.pb
--frozen_graph=True 
--input_names=Placeholder,seq_len
--output_names=CTCGreedyDecoder
--placeholder_type_enum=13
```
最后一个参数为`13`的原理如下：
```
from tensorflow.python.framework import dtypes
print(dtypes.float32.as_datatype_enum) # 1
print(dtypes.in32.as_datatype_enum) # 3
```
**我的应用:**
```
python -m tensorflow.python.tools.optimize_for_inference \
--input /path/frozen_inference_graph.pb \
--output /path/optimized_inference_graph.pb \
--input_names='x' \
--output_names='output'
```
- 确定输入输出节点的方法：
1. 在`tensorboard`中打印图，下面会讲
2. 在`python`中打印`op`分析：
```
# find op
for op in sess.graph.get_operations():
    print(op.name)
```
- `optimize_for_inference_lib.py`解读

  - `optimize_for_inference_lib.py`的注释：
    - `optimize_for_inference_lib.py`主要用于删除图中仅用于训练的部分：`GraphDef`中有一部分的常见变换是仅仅针对训练过程的，当网络仅用于预测时可以去除这些部分减少计算量。这些优化主要包括：
  
    -  删除仅用于训练的算子，比如`checkpoint saving`
    -  删除图中从未进行的操作(冗余节点)
    -  删除调试(debug)算子例如`CheckNumerics`
    -  将批标准化(batch norm)算子折叠到预计算(pre-calculate)中
        - 这个仅仅针对`BatchNormWithGlobalNormalization`与`FusedBatchNorm`两种BN
    -  将通用操作融合成统一的版本
  - 这个脚本的输入是一个`pb`文件，输出一个优化过的`pb`文件

##### 量化`quantize_graph`:
```
bazel build tensorflow/tools/quantization:quantize_graph \
&&bazel-bin/tensorflow/tools/quantization/quantize_graph \
--input=tensorflow_inception_graph.pb \
--output_node_names='softmax2' \
--print_nodes \
--output=/tmp/quantized_graph.pb \
--mode=eightbit \
--logtostderr
```
例如：
```
bazel-bin/tensorflow/tools/quantization/quantize_graph \
--input=./optimized_ctc.pb \
--output_node_names=CTCGreedyDecoder \
--print_nodes \
--output=./quantized_graph.pb \
--mode=eightbit \
--logtostderr
```

##### `Tensorboard`查看`pb`文件:
`tensorflow/python/tools/import_pb_to_tensorboard.py`
例如：
`python import_pb_to_tensorboard.py --model_dir=./input.pb --log_dir=./log`

**我的应用:**

控制台`$ pb-tensorboard resnet_50_v1.pb`

##### 查看节点计算时间`profile`:
```
bazel build -c opt tensorflow/tools/benchmark:benchmark_model&& \
bazel-bin/tensorflow/tools/benchmark/benchmark_model \
--graph=/tmp/tensorflow_inception_graph.pb \
--input_layer="Mul" \
--input_layer_shape="1,299,299,3" \
--input_layer_type="float" \
--output_layer="softmax:0" \
--show_run_order=false \
--show_time=false \
--show_memory=false \
--show_summary=true \
--show_flops=true \
--logtostderr
```
##### `ckpt`模型程序转为`pb`模型程序:
```
def load_graph(self, model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    
    with open(model_file, 'rb') as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    
    return graph
    
def __init__(self):
    ......
    # ......为自己的程序
    model_file = './input.pb'
    self.graph = self.load_graph(model_file)
    with self.graph.as_default():
        # import/input_1:0是自己网络输入节点的名字
        self.inputs = self.graph.get_tensor_by_name('import/input_1:0')
        # import/output_node:0是自己网络输出节点的名字
        self.output = self.graph.get_tensor_by_name('import/output_node:0')
        self.session = tf.Session(graph=self.graph)
        
    def proc(self, input_image):
        ......
        # ......为自己的程序
        test_feed = {self.inputs: input_image}
        predictions = self.session.run(self.output, test_feed)
```

##### `keras`模型转`pb`(`h5`转`pb`):
https://github.com/amir-abdi/keras_to_tensorflow

`python keras_to_tensorflow.py -input_model_file model.h5 --output_model_file model.pb`

- `keras_to_tensorflow.py`:
```

# coding: utf-8
# # Set parameters
 
"""
Copyright (c) 2017, by the Authors: Amir H. Abdi
This software is freely available under the MIT Public License. 
Please see the License file in the root for details.
The following code snippet will convert the keras model file,
which is saved using model.save('kerasmodel_weight_file'),
to the freezed .pb tensorflow weight file which holds both the 
network architecture and its associated weights.
""";
 
# setting input arguments
import argparse
parser = argparse.ArgumentParser(description='set input arguments')
parser.add_argument('-input_fld', action="store", 
                    dest='input_fld', type=str, default='.')
 
parser.add_argument('-output_fld', action="store", 
                    dest='output_fld', type=str, default='.')
 
parser.add_argument('-input_model_file', action="store", 
                    dest='input_model_file', type=str, default='model.h5')
 
parser.add_argument('-output_model_file', action="store", 
                    dest='output_model_file', type=str, default='model.pb')
 
parser.add_argument('-output_graphdef_file', action="store", 
                    dest='output_graphdef_file', type=str, default='model.ascii')
 
parser.add_argument('-num_outputs', action="store", 
                    dest='num_outputs', type=int, default=1)
 
parser.add_argument('-graph_def', action="store", 
                    dest='graph_def', type=bool, default=False)
 
parser.add_argument('-output_node_prefix', action="store", 
dest='output_node_prefix', type=str, default='output_node')
 
parser.add_argument('-f')
args = parser.parse_args()
print('input args: ', args)
 
# uncomment the following lines to alter the default values set above
# args.input_fld = '.'
# args.output_fld = '.'
# args.input_model_file = 'model.h5'
# args.output_model_file = 'model.pb'
 
# num_output: this value has nothing to do with the number of classes, batch_size, etc., 
# and it is mostly equal to 1. 
# If you have a multi-stream network (forked network with multiple outputs), 
# set the value to the number of outputs.
num_output = args.num_outputs
 
 
# # initialize
from keras.models import load_model
import tensorflow as tf
import os
import os.path as osp
from keras import backend as K
 
output_fld =  args.output_fld
if not os.path.isdir(output_fld):
    os.mkdir(output_fld)
weight_file_path = osp.join(args.input_fld, args.input_model_file)
 
# # Load keras model and rename output
K.set_learning_phase(0)
net_model = load_model(weight_file_path)
 
pred = [None]*num_output
pred_node_names = [None]*num_output
for i in range(num_output):
    pred_node_names[i] = args.output_node_prefix+str(i)
    pred[i] = tf.identity(net_model.outputs[i], name=pred_node_names[i])
print('output nodes names are: ', pred_node_names)
 
 
# #### [optional] write graph definition in ascii
sess = K.get_session()
 
if args.graph_def:
    f = args.output_graphdef_file 
    tf.train.write_graph(sess.graph.as_graph_def(), output_fld, f, as_text=True)
    print('saved the graph definition in ascii format at: ', osp.join(output_fld, f))
 
 
# #### convert variables to constants and save
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
graph_io.write_graph(constant_graph, output_fld, args.output_model_file, as_text=False)
print('saved the freezed graph (ready for inference) at: ', osp.join(output_fld, args.output_model_file))
```

##### 导入`pb`文件:
```
with tf.grile.FastGFile('./filename.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        # 这部我自己没有加入，存疑——为什么导入图之后还要初始化参数呢
        sess.run(init)
        input_x = sess.graph.get_tensor_by_name('input:0')
```

