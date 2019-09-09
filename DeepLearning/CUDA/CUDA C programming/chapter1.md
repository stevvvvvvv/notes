# CUDA C编程权威指南(chapter 1)
### 第一章 基于CUDA的异构并行计算
#### 1.1 并行计算
##### 1.1.1 串行编程和并行编程
- 任务的相关和独立主要看数据依赖关系。

##### 1.1.2 并行性
- 有两种基本的并行类型：**任务并行**和**数据并行**。
- CUDA编程非常适合解决**数据并行**计算的问题。
- 两种方法对数据进行划分：块划分和周期划分。块划分中一组连续的数据被分到一个块内，线程在同一时间只处理一个数据块；周期划分中每个线程处理多个数据块，每个线程作用于数据的多部分。
##### 1.1.3 计算机架构
- 单指令单数据(SISD)：只有一个核心。
- 单指令多数据(SIMD)：并行架构类型，计算机上有多个核心。
- 多指令单数据(MISD)：少见，每个核心通过多个指令流处理同一个数据流。
- 多指令多数据(MIMD)：并行架构，多个核心使用多个指令流来异步处理多个数据流，从而实现空间上的并行性。

* CPU核心比较重，用来处理非常复杂的控制逻辑，优化串行程序。
* GPU核心较轻，用于优化具有简单控制逻辑的数据并行任务，注重并行程序的吞吐量。

#### 1.2异构计算
##### 1.2.1 异构架构
- 一个典型的异构计算：GPU通过PCIe总线与基于CPU的主机相连来进行操作，主机代码在CPU上运行，设备代码在GPU上运行。

* 描述GPU容量的两个重要特征：**CUDA核心数量**和**内存大小**。
* 评估GPU性能的两个指标：**峰值计算性能**(GFlops或TFlops)和**内存带宽**(GB/s)。

##### 1.2.2 异构计算范例
- CPU和GPU的应用范围主要从**并行级**和**数据规模**两方面划分。

##### 1.2.3 CUDA:一种异构计算平台
- 一个CUDA程序包含在CPU上运行的主机代码和在GPU上运行的设备代码，通过nvidia的CUDA nvcc编译器将设备代码从主机代码中分离出来。
- 主机代码是标准的C代码，使用C编译器进行编译；设备代码即核函数(kernel)，是使用CUDA C编写的。

#### 1.3 用GPU输出Hello World
- 在linux中检查CUDA编译器是否正确安装：
```
$ which nvcc
```
- 检查机器上是否安装了GPU加速卡：
```
$ ls  -l /dev/nv*
```
- 写一个CUDA C程序，需要以下步骤：
1. 用专用扩展名.cu创建一个源文件。
2. 使用CUDA nvcc编译器编译程序。
3. 从命令行执行可执行文件。

- 首先写一个C语言版本的Hello World：

```
#include <stdio.h>
int main(void)
{
    printf("Hello World from CPU!\n");
}
```

将代码保存到hello.cu中，然后使用nvcc编译器进行编译。

- CUDA nvcc编译器和gcc编译器有相似的语义：
```
$ nvcc hello.cu -o hello
```

- 接下来，编写一个内核函数，命名为helloFromGPU，用来输出字符串"Hello World from GPU!"：
```
__global__ void helloFromGPU(void)
{
	printf("Hello World from GPU!\n");
}
```
修饰符__global__告诉编译器这个函数会从CPU中调用，然后在GPU上执行，用下面的代码启动内核函数：
```
helloFromGPU<<<1, 10>>>();
```
三重尖括号意味着从主线程到设备端代码的调用，括号内的执行配置也说明使用多少线程执行内核函数，完整的代码如下：
```
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void helloFromGPU(void)
{
	printf("Hello World from GPU!\n");
}
int main(void)
{
	// hello from cpu
	printf("Hello World from CPU!\n");

	helloFromGPU <<<1, 10 >>>();
	cudaDeviceReset();
	return 0;
}
```
函数cudaDeviceReset()用来显式释放和清空当前进程中与当前设备有关的所有资源。

- **CUDA编程结构**
- 一个典型的CUDA编程结构包括5个主要步骤：
1. 分配GPU内存;
2. 从CPU内存中拷贝数据到GPU内存;
3. 调用CUDA内核函数完成程序制定的运算;
4. 将数据从GPU拷回CPU内存;
5. 释放GPU内存空间。

#### 1.4 使用CUDA编程难吗
- CPU编程和GPU编程的主要区别是程序员对GPU架构的熟悉程度。用并行思维进行思考并对GPU架构有了基本的了解，会使你编写规模达到成百上千个核的并行程序， 如同写串行程序一样简单。
- **数据局部性**在并行编程中是一个非常重要的概念，数据局部性指的是数据重用， 以降低内存访问的延迟。
