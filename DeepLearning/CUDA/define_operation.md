# `TensorFlow`自定义`operation`

可能需要新定义`c++ operation`的几种情况：

- 现有的`operation`组合不出想要的`op`
- 现有的`operation`组合出的`operation`十分低效
- 想要手动融合一些操作

为了实现自定义操作，需要：

1. 在`c++`文件中注册一个新`op`：
`Op registration`定义了`op`的功能接口，它和`op`的实现是独立的。例如:`op registration`定义了`op`的名字和`op`的输入输出，同时也定义了`shape`方法，被用于`tensor`的`shape`接口。
2. 在`c++`中实现`op`：
`op`的实现称之为`kernel`，它是`op`的一个具体实现。
3. 创建一个`python wrapper`(optional)：
这个`wrapper`是一个公开的`API`，用来在`python`中创建`op`。`op registration`会生成一个默认的`wrapper`，我们可以直接使用或者自己添加一个。
4. 写一个计算`op`梯度的方法(optional)。
5. 测试`op`：
为了方便，我们通常在`python`中测试`op`。

## 定义`op`接口

在注册`op`的时候，需要指定：

- `op`的名字
- `op`的输入(名字、类型)，`op`的输出(名字、类型)
- `docstrings`
- `op`可能需要的一些attrs(<https://www.tensorflow.org/extend/adding_an_op#attrs>)

### 为了演示这是如何工作的，我们来看一个简单的例子

- 定义一个`op`：输入是一个`int32`的`tensor`，输出是输入的拷贝，除了第一个元素保留，其他元素全部置零

为了创建这个`op`的接口，我们需要：

- 创建一个文件，名字为`zero_out.cc`，然后调用`REGISTER_OP`宏，使用这个宏来定义`op`的接口：

```c++
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("ZeroOut")
      .Input("to_zero:int32")
      .Output("ezro_end"int32")
      .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* C){
          c->set_output(0, c->input(0));
          return Status::OK();
      });
```

这个`ZeroOut op`接收一个`int32`的`tensor`作为输入，输出同样是一个`int32`的`tensor`。这个`op`也使用了一个`shape`方法确保输入和输出的维度是一样的。

## 实现`op`对应的`kernel`

当我们定义了`op`的接口之后，可以提供一个或多个关于`op`的实现：

- 创建一个类，继承`OpKernel类`
- 重写`Opkernel`类的`Compute`方法
  - `Compute`方法提供一个类型为`OpKernelContext*`的`context`参数，从这里我们可以访问到一些有用的信息，比如输入`tensor`和输出`tensor`

将`kernel`代码也放到之前创建的`zero_out.cc`文件中：

```c++
#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // 获取输入 tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // 创建输出 tensor, context->allocate_output 用来分配输出内存？
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // 执行计算操作。
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output_flat(0) = input(0);
  }
};
```

在实现了`kernel`后，就可以将这个注册到`tensorflow`系统中去了。在注册时需要对`op`的运行环境指定一些限制，例如可能有一个`kernel`是给`CPU`用的，另一个是给`GPU`用的。通过把下列代码添加到`zero_out.cc`中来完成功能：

`REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);`

## 构建`op`库

### 使用系统编译器编译定义的`op`

我们可以使用系统上的`c++`编译器`g++`或者`clang`来编译`zero_out.cc`。二进制的`PIP`包已经将编译所需的头文件和库安装到了系统上。`Tensorflow`的`python library`提供了一个用来获取头文件目录的函数`get_include`，下面是这个函数在`ubuntu`上的输出：

```shell
$ python
>>> import tensorflow as tf
>>> tf.sysconfig.get_include()
'/usr/local/lib/python2.7/site-packages/tensorflow/include'
```

假设你已经装好了`g++`，可以使用下面的命令将`op`编译成一个动态库：

```shell
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC -I $TF_INC -O2
```

**如果你的`g++`版本大于`5.0`的话，加上这个参数`-D_GLIBCXX_USE_CXX!!_ABI=0`：**

```shell
#创建动态链接库的命令
g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I $TF_INC -O2
```

## 在`python`中使用`op`

`Tensorflow`的`python`接口提供了`tf.load_op_library`函数来加载动态`library`，同时将`op`注册到`tensorflow`框架上。`load_op_library`返回一个`python module`，它包含了`op`和`kernel`的`python wrapper`。因此，一点编译好了一个`op`，就可以使用下列代码通过`python`执行：

```python
import tensorflow as tf
zero_out_module = tf.load_op_library('./zero_out.so')
with tf.Session(''):
zero_out_module.zero_out([[1, 2], [3, 4]]).eval()

# prints
# array([[1, 0], [0, 0]], dtype=int32)
```

**记住：生成函数的名字是`snake_case`name，如果在`c++`文件中，`op`的名字是`ZeroOut`，那么在`python`中，名字是`zero_out`**

## 验证`op`

一个验证自定义`op`正确性的方法是写一个测试文件，创建一个`zero_out_op_test.py`文件然后运行：

```python
import tensorflow as tf

class ZeroOutTest(tf.test.TestCase):
    def testZeroOut(self):
        zero_out_module = tf.load_op_library('./zero_out.so')
        with self.test_session():
            result = zero_out_model.zero_out([5, 4, 3, 2, 1])
            self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])

if __name_- == '__main__':
    tf.test.main()
```

## 总结

`tensorflow`自定义`op`的方法可以总结为：

1. 写个`diy_op.cc`文件
2. 用`g++`把这个文件编译成动态链接库
3. 在`python`中使用`tf.load_op_library`将库导入
4. 使用

还有一种方法是用`bazel`编译。

参考：<https://www.tensorflow.org/extend/adding_an_op>

## `TensorFlow`添加自定义`operator`

### 基本原理

同一个`Operator`由于可能在CPU或GPU上执行，如果想要支持两个设备，就需要写两份代码

### 编写CPU版本程序

#### 基础代码

下面代码的功能是将输入的数组第一个元素保留，后面的元素都设置为0：

```c++
//  zero_op.cc
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

using namespace tensorflow;

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    // 将输入tensor从context中取出
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    // 创建一个output_tensor 使用context->allocate_output()给它分配空间
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output(0) = input(0);
  }
};

REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
```

#### 编译脚本

`ubuntu 16.04`需要添加`D_GLIBCXX_USE_CXX11_ABI=0`这个编译选项才可以加载成功

```shell
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++-std=c++11 -shared zero_op.cc -o zero_op.so -fPIC -I $TF_INC -02 -D_GLIBCXX_USE_CXX11_ABI=0
```

#### `python`调用

```python
import tensorflow as tf
zero_ot_module = tf.load_op_library('zero_os.so')

with tf.Session(''):
    x = zero_out_module.zero_out([[100, 2], [3, 4]]).eval()

print(x)
```

### 编写GPU版本程序

#### kernel部分代码

```c++
//  cuda_op_kernel.cu.cc
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupportedEigen/CXX11/Tensor"

__global__ vioid AddOneKernel(const int* in, const int N, int* out)
{
    fot (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        out[i] = in[i] + 1;
    }
}

void AddOneKernelLauncher(const int* in, const intN, int* out)
{
    AddOneKernel<<<32, 256>>>(in, N, out);
}

#endif
```

#### C++部分代码

```c++
//  cuda_op_kernel.cc
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("AddOne")
    .Input("input: int32")
    .Output("output: int32")
    .Doc(R"doc(
Adds 1 to all elements of the tensor.
output: A Tensor.
  output = input + 1
)doc");

void AddOneKernelLauncher(const int* in, const int N, int* out);

class AddOneOp : public OpKernel {
 public:
  explicit AddOneOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    // Call the cuda kernel launcher
    AddOneKernelLauncher(input.data(), N, output.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("AddOne").Device(DEVICE_GPU), AddOneOp);
```

#### 编译

```shell
nvcc -std=c++11 -c -o cuda_op_kernel.cu.o cuda_op_kernel.cu.cc \
-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o cuda_op_kernel.so cuda_op_kernel.cc \
cuda_op_kernel.cu.o -I $TF_INC -fPIC -lcudart
```

### 代码示例

#### ROI Pooling

`Faster R-CNN`中`ROI Pooling`需要重写`pooling`操作，系统没有自带的实现：

```c++
//i+=会让这个循环只执行一次,这样写代码的好处就是这个循环执行完成会释放掉括号内的内存? (疑问)
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

using std::max;
using std::min;

// namespace tensorflow {
using namespace tensorflow;

template <typename Dtype>
__global__ void ROIPoolForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int height, const int width,
    const int channels, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, int* argmax_data)
{
    //每次只处理一个location,这个循环只执行一次
  CUDA_1D_KERNEL_LOOP(index, nthreads)
  {

    // (n, ph, pw, c) is an element in the pooled output
    int n = index;
    int c = n % channels;
    n /= channels;
    int pw = n % pooled_width;
    n /= pooled_width;
    int ph = n % pooled_height;
    n /= pooled_height;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1] * spatial_scale);
    int roi_start_h = round(bottom_rois[2] * spatial_scale);
    int roi_end_w = round(bottom_rois[3] * spatial_scale);
    int roi_end_h = round(bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    bottom_data += roi_batch_ind * channels * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = (h * width + w) * channels + c;
        if (bottom_data[bottom_index] > maxval) {
          maxval = bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    if (argmax_data != nullptr)
      argmax_data[index] = maxidx;
  }
}
```
