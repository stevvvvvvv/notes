# CUDA C编程权威指南(chapter 2)

[toc]

## 第二章 CUDA编程模型

- CUDA程序实现向量加法和向量乘法。

### 2.1 CUDA编程模型概述

![程序和编程模型实现之间的抽象结构](https://note.youdao.com/yws/api/personal/file/FC8BE2A3C91C4D9A9C1F94984EB884F7?method=download&shareKey=372f5cd4aa8f6f9c3ccd51d54b645556)

- 除了与其他并行编程模型共有的抽象外，CUDA编程模型还利用GPU架构的计算能力提供了以下特用的功能：

1. 一种通过层次结构在GPU中组织线程的方法。
2. 一种通过层次结构在GPU中访问内存的方法。

#### 2.1.1 CUDA编程结构

- 一个异构环境中包含多个CPU和GPU，要注意区分以下内容：

1. 主机(host)：CPU及其内存(主机内存)
2. 设备(device)：GPU及其内存(设备内存)

- 在本书的代码示例中，主机内存中的变量名以h_为前缀，设备内存中的变量名以d_为前缀。
- 内核(kernel)是CUDA编程模型的一个重要组成部分。
- CUDA编程模型主要是异步的，因此在GPU上进行的运算可以与主机-设备通信重叠。
- 一个典型的CUDA程序包括由并行代码互补的串行代码，如下图所示：

![image](https://note.youdao.com/yws/api/personal/file/5A27C4574C624E6D87D77E676EE4B77C?method=download&shareKey=c70c7b328efac05c62837e254eb8397a)

- 一个典型的CUDA程序实现流程遵循以下模式：

1. 把数据从CPU内存拷贝到GPU内存。
2. 调用核函数对存储在GPU内存中的数据进行操作。
3. 将数据从GPU内存传送回到CPU内存。

#### 2.1.2 内存管理

- CUDA运行时负责分配与释放设备内存，并且在主机内存与设备内存间传输数据。标准的C函数以及相应地针对内存的CUDA C函数如下：

|标准的C函数|CUDA C函数|标准的C函数|CUDA C函数|
|---|---|---|---|
|`malloc`|`cudaMalloc`|`memset`|`cudaMemset`|
|`memcpy`|`cudaMemcpy`|`free`|`cudaFree`|

- 用于执行GPU内存分配的是`cudaMallo`c函数，其函数原型为：

```c++
cudaError_t cudaMalloc ( void** devPtr, size_t size)
```

该函数负责向设备分配一定字节的先行内存，并以devPtr的形式返回所分配内存的指针。

- 用于负责主机和设备之间数据传输的是`cudaMemcpy`函数，其函数原型为：

```c++
cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind)
```

该函数从src指向的源存储区复制一定数量的字节到dst指向的目标存储区，复制方向由kind制定，其中kind有以下几种：

`cudaMemcpyHostToHost`
`cudaMemcpyHostToDevice`
`cudaMemcpyDeviceToHost`
`cudaMemcpyDeviceToDevice`

这个函数以同步方式执行，因为在cudaMemcpy函数返回以及传输操作完成之前主机应用程序是阻塞的。除了内核启动之外的CUDA调用都会返回一个错误的枚举类型cudaError_t，如果GPU内存分配成功，函数返回

`cudaSuccess`

否则返回

`cudaErrorMemorAllocation`

也可以使用以下CUDA运行函数将错误代码转化为可读的错误信息：

`char* cudaGetErrorString(cudaError_t error)`

- GPU中抽象的内存层次结构如图：

![image](https://note.youdao.com/yws/api/personal/file/C8404E526C6547578655640F969028C3?method=download&shareKey=31d3393d97692ec52b2e59fd395be030)

在GPU内存层次结构中，最主要的两种内存是全局内存和共享内存。全局类似于CPU
的系统内存，而共享内存类似于CPU的缓存。然而GPU的共享内存可以由CUDA C的内核
直接控制。

- **CUDA C编程实现数组加法**`$a+b=c$`
- 数组`$a$`的第一个元素与数组`$b$`的第一个元素相加，得到的结果作为数组`$c$`的第一个元素，重复这个过程直到所有元素都进行了计算
- 首先，执行主机端代码使两个数组相加：

```c++
//  sumArraysOnHost.c
#include <stdlib.h>
#include <string.h>
#include <time.h>

void sumArrayOnHost(float* A, float* B, float* C, const int N)
{
    for (int idx = 0; idx < N; ++idx)
    {
        C[idx] = A[idx] + B[idx];
    }
}

void initialData(float *ip, int size)
{
    //  generate different seed for random number
    time_t t;
    srand((unsigned int)time(&t));
    for (int i = 0; i < size; ++i)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

int main(int argc, char **argv)
{
    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);
    float* h_A, * h_B, * h_C;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    h_C = (float*)malloc(nBytes);
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    sumArrayOnHost(h_A, h_B, h_C, nElem);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
```

- 现在我们在GPU上修改代码来进行数组加法运算，用cudaMalloc在GPU上申请内存：

```c++
float *d_A, *d_B, *d_C;
cudaMalloc((float**)&d_A, nBytes);
cudaMalloc((float**)&d_B, nBytes);
cudaMalloc((float**)&d_C, nBytes);
```

- 使用cudaMemcpy函数把数据从主机内存拷贝到GPU的全局内存中，参数cudaMemcpyHostToDevice指定数据拷贝方向：

```c++
cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
```

当数据被转移到GPU的全局内存后，主机端调用kernel函数在GPU上进行数组求和。**一旦内核被调用，控制权立即被传回主机，这样当GPU运行核函数时主机可以执行其他函数**，因此内核和主机是**异步**的。

- 当内核在GPU上完成数组元素处理后，其结果`d_C`存在GPU的全局内存中，使用cudaMemcpy函数将结果从GPU复制回主机的数组gpuRef中：

```c++
cudaMemcp(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
```

- cudaMemcpy的调用会造成主机运行阻塞，运行结束后调用cudaFree释放GPU的内存：

```c++
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
```

#### 2.1.3 线程管理

- 当核函数在主机端启动时，它的执行会移动到设备上，此时设备中会产生大量的线程并且每个线程都执行由核函数指定的语句。CUDA明确了线程层次，如下是一个两层的线程层次结构，由线程块(Block)和线程块网格组成(grid)：

![image](https://note.youdao.com/yws/api/personal/file/AE3BD1893D8F497AA74E99D0438C1A13?method=download&shareKey=7f0926dd199f1c1c31b8ffd0877e36a0)

- 由一个内核启动所产生的所有线程统称为一个网格，同一网格中的所有线程共享相同的全局内存空间；一个网格由多个线程块构成，一个线程块包含一组线程，**同一线程块**内的线程协作可以通过以下方式来实现：

1. 同步
2. 共享内存
**不同块内的线程不能协作**

- 线程依靠以下两个坐标变量来区分彼此：

1. blockIdx(线程块在线程格内的索引)
2. threadIdx(块内的线程索引)

- **这些变量是核函数中需要预初始化的内置变量，当执行一个核函数时，CUDA运行时为每个线程分配坐标变量blockIdx和threadIdx，基于这些坐标，我们将部分数据分配给不同的线程，这些坐标都是dim3类型的，最多可以有三个维度。**
- 网格和块的维度由下列两个dim3类型的内置变量制定：

1. blockDim(线程块的维度，用每个线程块中的线程数表示)
2. gridDim(线程格的维度，用每个线程格中的线程数来表示)

通常，一个线程格会被组织成线程块的二维数组形式，一个线程块会被组织成线程的三维数组格式。

- **CUDA C编程实现检查网格和块的索引和维度**
- 首先，定义程序所用的数据大小，我们先定义一个较小的数据：

```c++
int nElem = 6；
```

- 接下来，定义块的尺寸并基于块和数据的大小计算网格尺寸：

```c++
dim3 block(3);
dim3 grid((nElem+block.x-1)/block.x)
```

会发现网格大小是块大小的倍数(地板除)。

- 可以这样检查网格和块维度：

```c++
printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
```

- 在核函数中，每个线程都输出自己的线程索引、块索引、块维度和网格维度：

```c++
printf("threadIdx:(%d, %d, %d)  blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) gridDim:(%d, %d, %d)\n",
threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x,blockIdx.y, blockIdx.z, 
blockDim.x, blockDim.y, blockDim.z, gridDim.x,gridDim.y, gridDim.z);
```

- 将代码块聚合在一起，我们可以检查网格和块的索引和维度：

```c++
#include "cuda_runtime.h"
#include <stdio.h>

__global__ void checkIndex(void)
{
    printf("threadIdx:(%d, %d, %d)  blockIdx:(%d, %d, %d)  blockDim:(%d, %d, %d)  gridDim:(%d, %d, %d)\n",
        threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z,
        blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc, char **argv)
{
    int nElem = 6;
    dim3 block(3);
    dim3 grid((nElem + block.x - 1) / block.x);
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
    checkIndex <<<grid, block >>> ();
    cudaDeviceReset();
    return 0;
}
```

- 编译后的输出为：

```c++
grid.x 2 grid.y 1 grid.z 1
block.x 3 block.y 1 block.z 1
threadIdx:(0, 0, 0)  blockIdx:(0, 0, 0)  blockDim:(3, 1, 1)  gridDim:(2, 1, 1)
threadIdx:(1, 0, 0)  blockIdx:(0, 0, 0)  blockDim:(3, 1, 1)  gridDim:(2, 1, 1)
threadIdx:(2, 0, 0)  blockIdx:(0, 0, 0)  blockDim:(3, 1, 1)  gridDim:(2, 1, 1)
threadIdx:(0, 0, 0)  blockIdx:(1, 0, 0)  blockDim:(3, 1, 1)  gridDim:(2, 1, 1)
threadIdx:(1, 0, 0)  blockIdx:(1, 0, 0)  blockDim:(3, 1, 1)  gridDim:(2, 1, 1)
threadIdx:(2, 0, 0)  blockIdx:(1, 0, 0)  blockDim:(3, 1, 1)  gridDim:(2, 1, 1)
```

**可见，两个Dim的数据是不变的，只有索引在不断改变**

- 从主机端和设备端访问网格/块变量

1. 从主机端访问块变量：`block.x, block.y, block.z`
2. 从设备端访问块变量：`blockDim.x, blockDim.y, bolckDim.z`

- 对于一个给定的数据大小，确定网格和块尺寸的一般步骤为：

1. 确定块的大小
2. 在已知数据大小和块大小的基础上计算网格维度

- 要确定块尺寸，通常要考虑：

1. 内核的性能特性
2. GPU资源的限制

- **CUDA C编程实现在主机上定义网格和块的大小**

```c++
// defineGridBlock.cu
#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char **argv)
{
	//	define total data elements
	int nElem = 1024;

	//	define grid and bolck structure
	dim3 block (1024);
	dim3 grid((nElem + block.x - 1) / block.x);
	printf("grid.x %d block.x %d \n", grid.x, block.x);

	//	reset block
	block.x = 512;
	grid.x = (nElem + block.x - 1) / block.x;
	printf("grid.x %d block.x %d \n", grid.x, block.x);

	//	reset block
	block.x = 256;
	grid.x = (nElem + block.x - 1) / block.x;
	printf("grid.x %d block.x %d \n", grid.x, block.x);

	//	reset block
	block.x = 128;
	grid.x = (nElem + block.x - 1) / block.x;
	printf("grid.x %d block.x %d \n", grid.x, block.x);

	//	reset device before you leave
	cudaDeviceReset();
	return 0;
}
```

输出为：

```shell
grid.x 1 block.x 1024
grid.x 2 block.x 512
grid.x 4 block.x 256
grid.x 8 block.x 128
```

**可以视作`$grid = nElem // block$`**

#### 2.1.4 启动一个CUDA核函数

- C语言函数调用：

```c++
function_name (argument list);
```

- CUDA内核调用是对C语言调用语句的延伸，<<<>>>运算符内是核函数的配置参数：

```
kernel_name <<<grid, block>>>(argument list);
```

第一个值是网格维度，也就是启动块的数目；第二个值是块维度，也就是每个块中线程的数目。
通过指定网格和块的维度，可以配置**内核中线程的数目**和**内核中使用的线程布局**

- 同一个块中的线程之间可以相互协作，不同块内的线程不能协作。对于一个给定的问题，可以设计不同的网格和块布局来组织线程。
- 例如有32个数据元素用于计算，可以设置`$block=8, grid=32//8=4$`，即：

```c++
kernel_name<<<4, 8>>>(argument list);
```

如图所示：

![image](https://note.youdao.com/yws/api/personal/file/309B3162F40448DD94CD37256A0CA16A?method=download&shareKey=993684a43ef30d34a23440f8593a733b)

- 由于数据在全局内存中是线性存储的，因此可以用变量`blockIdx.x, threadId.x`进行以下操作：

1. 在网格中标识一个唯一的线程
2. 建立线程和数据元素之间的映射关系

- 核函数的调用与主机线程是异步的，核函数调用结束后控制权会立刻返回给主机端。可以调用以下函数强制主机端等待所有的核函数执行结束：

```c++
cudaError_t cudaDeviceSynchronize(viod);
```

- 一些CUDA运行时API在主机和设备之间是隐式同步的，当使用`cudaMemcpy`函数在主机和设备之间拷贝数据时，主机端隐式同步，即主机端程序必须等待数据拷贝完成后才能继续执行程序。

```c++
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
```

之前所有的核函数调用完成后开始拷贝数据，当拷贝完成后，控制权立即返回给主机。

- 异步行为：不同于C语言的函数调用，所有的CUDA核函数的启动都是异步的，CUDA内核调用完成后，控制权立即返回给CPU。

#### 2.1.5 编写核函数

- 核函数是在设备端执行的代码，在核函数中，需要为一个线程规定要进行的计算以及要进行的数据访问，以下是用`__global__`声明定义核函数：

```c++
__global__ void kernel_name(argument list);
```

- 下表总结了CUDA C程序中的函数类型限定符，函数类型限定符指定一个函数是在主机端还是设备端上执行，以及可被主机调用还是被设备调用：

|限定符|执行|调用|备注|
|---|---|---|---|
|`__global__`|在设备端执行|可从主机端调用，也可以从计算能力为3的设备中调用|**必须有一个`void`返回类型**|
|`__device__`|在设备端执行|仅能从设备端调用||
|`__host__`|在主机端执行|仅能从主机端调用|可以忽略|

- `__device__`和`__host__`限定符可以一齐使用，这样函数可以同时在主机和设备端进行编译。

- CUDA核函数的限制，以下限制适用于所有核函数：

1. 只能访问设备内存
2. 必须有`void`返回类型
3. 不支持静态变量
4. 显示异步行为

- 考虑一个简单的例子：将两个大小为`N`的向量`$A$`和`$B$`相加，主机端的向量加法C代码如下：

```c++
void sumArraOnHost(float *A, float *B, float *C, const int N)
{
    for (int i = 0; i < N; ++i)
    C[i] = A[i] + B[i];
}
```

这是一个迭代`N`次的串行程序，循环结束后将产生以下核函数：

```c++
__global__ void sumArrarysOnGPU(float *A, float *B, float *C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}
```

C函数和核函数间的区别主要在于循环体消失了，内置的线程坐标替换了数组索引。
假设有一个长度为32个元素的向量，可以用以下方法用32个线程调用核函数：

```c++
sumArrayOnGPU<<<1, 32>>>(float *A, float *B, float *C);
```

#### 2.1.6 验证核函数

- 需要用一个主机函数来验证核函数的结果：

```c++
void checkResult(float* hostRef, float* gpuRef, const int N)
{
	double epsilon = 1.0E-8;
	int match = 1;
	for (int i = 0; i < N; ++i)
	{
		if (abs(hostRef[i] - gpuRef[i] > epsilon))
		{
			match = 0;
			printf("Arrays do not mathc!\n");
			printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}
	if (match)
	{
		printf("Arrays match.\n");
	}
	return;
}
```

- 两个验证核函数代码的方法

1. 首先，可以在Fermi及更高版本的设备端的核函数中使用`printf`函数
2. 可以将执行参数设置为`<<<1, 1>>>`，模拟串行执行程序，**特别是遇到运算次序问题时**

#### 2.1.7 处理错误

- 定义一个错误处理宏封装所有的CUDA API调用，可以简化错误检查过程：

```c++
#define CHECK(call)															\
{																			\
	const cudaError_t error = call;											\
	if (error != cudaSuccess)												\
	{																		\
		printf("Error: %s:%d, ",__FILE__, __LINE__)							\
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));	\
		exit(1);															\
	}																		\
}                                                                           \
```

我们可以在以下代码中使用宏：

```c++
CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice);
```

也可以使用以下方法在核函数调用后检查核函数错误：

```c++
kernel_function<<<grid, block>>>(argument list);
CHECK(cudaDeviceSynchronize();
```

`CHECK(cudaDeviceSynchronize())`会阻塞主机端线程的运行直到设备端所有的请求任务都结束，并确保最后的核函数启动部分不会出错。

以上仅是以调试为目的的，因为在核函数启动后添加这个检查点会阻塞主机端线程，使该检查点成为全局屏障。

##### 2.1.8 编译和执行

- 现在把所有代码放在一个`sumArraysOnGPU-small-case.cu`文件中，如下所示：

```c++
\\  sumArraysOnGPU-small-case.cu
#include <cuda_runtime.h>
#include <stdio.h>
#define CHECK(call) \
{ \
　　const cudaError_t error = call; \
　　if (error != cudaSuccess) \
　　{ \
　　　　printf("Error: %s:%d, ", __FILE__, __LINE__); \
　　　　printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
　　　　exit(1); \
　　} \
}

void checkResult(float* hostRef, float* gpuRef, const int N) 
{
	double epsilon = 1.0E-8;
	bool match = 1;
	for (int i = 0; i < N; i++) {
		if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
			match = 0;
			printf("Arrays do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}
	if (match) printf("Arrays match.\n\n");
}

void initialData(float* ip, int size) 
{
	// generate different seed for random number
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < size; i++) {
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}
}

void sumArraysOnHost(float* A, float* B, float* C, const int N) 
{
	for (int idx = 0; idx < N; idx++)
		C[idx] = A[idx] + B[idx];
}

__global__ void sumArraysOnGPU(float* A, float* B, float* C) 
{
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}

int main(int argc, char** argv) 
{
	printf("%s Starting...\n", argv[0]);
	// set up device
	// cudaSetDevice函数主要是用来设置使用第几块GPU
	int dev = 0;
	cudaSetDevice(dev);

	// set up data size of vectors
	int nElem = 32;
	printf("Vector size %d\n", nElem);

	// malloc host memory
	size_t nBytes = nElem * sizeof(float);
	float* h_A, * h_B, * hostRef, * gpuRef;
	h_A = (float*)malloc(nBytes);
	h_B = (float*)malloc(nBytes);
	hostRef = (float*)malloc(nBytes);
	gpuRef = (float*)malloc(nBytes);

	// initialize data at host side
	initialData(h_A, nElem);
	initialData(h_B, nElem);
	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	// malloc device global memory
	float* d_A, * d_B, * d_C;
	cudaMalloc((float**)& d_A, nBytes);
	cudaMalloc((float**)& d_B, nBytes);
	cudaMalloc((float**)& d_C, nBytes);

	// transfer data from host to device
	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

	// invoke kernel at host side
	dim3 block(nElem);
	dim3 grid(nElem / block.x);
	sumArraysOnGPU <<< grid, block >>> (d_A, d_B, d_C);
	printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);

	// copy kernel result back to host side
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

	// add vector at host side for result checks
	sumArraysOnHost(h_A, h_B, hostRef, nElem);

	// check device results
	checkResult(hostRef, gpuRef, nElem);

	// free device global memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// free host memory
	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);
	return(0);
}
```
系统报告的结果如下：

```shell
./sumArraysOnGPU-small-case Starting...
Vector size 32
Execution configuration <<<1, 32>>>
Arrays match.
```

一般情况下，可以基于给定的一维网格和块的信息计算全局数据访问的唯一索引：

```c++
__global__ void sumArraysOnGPU(float *A, float *B, float *C)
{
    int i = bolckIdx.x * bolckDim.x + threadIdx.x;
    C[i] = A[i] + B[i];
}
```

### 2.2 给核函数计时

- 在主机段使用一个CPU或GPU计时器计算内核的执行时间

#### 2.2.1 用CPU计时器计时

- 使用`gettimeofday`系统调用来创建一个CPU计时器，返回1970.1.1以来的秒数，需要在程序中加入`sys/time.h`头文件：

```c++
double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
```

可以用`cpuSecond`测试核函数：

```c++
double iStart = cpuSecond();
kernel_name<<<grid, bolck>>>(argument list);
cudaDeviceSynchronize();
double iElaps = cpuSecond() - iStart;
```

现在通过设置数据集大小对一个有16M个元素的大向量进行测试：

```c++
\\  这个代码的意思是二进制将0000000001的1向左移位24个，即2^24
int nElem = 1<<24
```

由于GPU的可扩展性，需要借助块和线程的索引来计算一个按行有限的数组索引`i`，并对核函数进行修改，添加限定条件`i < N`来检验索引是否越界：

```c++
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {C[i} = A[i] + B[i];}
}
```

有了这些更改，可以使用不同的执行配置来衡量核函数，并且解决了创建线程总数大于向量元素总数的情况。

```c++
//  测试向量加法的核函数
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#define CHECK(call) \
{ \
	const cudaError_t error = call; \
	if (error != cudaSuccess) \
	{ \
		printf("Error: %s:%d, ", __FILE__, __LINE__); \
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
		exit(1); \
	} \
}

double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void initialData(float* ip, int size)
{
	// generate different seed for random number
	time_t t;
	srand((unsigned)time(&t));

	for (int i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}

	return;
}

void sumArraysOnHost(float* A, float* B, float* C, const int N)
{
	for (int idx = 0; idx < N; idx++)
	{
		C[idx] = A[idx] + B[idx];
	}
}

void checkResult(float* hostRef, float* gpuRef, const int N)
{
	double epsilon = 1.0E-8;
	bool match = 1;
	for (int i = 1; i < N; ++i)
	{
		if (abs(hostRef[i] - gpuRef[i]) > epsilon)
		{
			match = 0;
			printf("Array do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}
	if (match)
	{
		printf("Arrays match.\n\n");
		return;
	}
}

__global__ void sumArraysOnGPU(float* A, float* B, float* C, const int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) { C[i] = A[i] + B[i]; }
}
int main(int argc, char** argv)
{
	printf("%s Starting...\n", argv[0]);

	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	// set up data size of vectors
	int nElem = 1 << 24;
	printf("Vector size %d\n", nElem);

	// malloc host memory
	size_t nBytes = nElem * sizeof(float);

	float* h_A, * h_B, * hostRef, * gpuRef;
	h_A = (float*)malloc(nBytes);
	h_B = (float*)malloc(nBytes);
	hostRef = (float*)malloc(nBytes);
	gpuRef = (float*)malloc(nBytes);

	double iStart, iElaps;

	// initialize data at host side
	iStart = cpuSecond();
	initialData(h_A, nElem);
	initialData(h_B, nElem);
	iElaps = cpuSecond() - iStart;
	printf("initialData Time elapsed %f sec\n", iElaps);
	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	// add vector at host side for result checks
	iStart = cpuSecond();
	sumArraysOnHost(h_A, h_B, hostRef, nElem);
	iElaps = cpuSecond() - iStart;
	printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);

	// malloc device global memory
	float* d_A, * d_B, * d_C;
	CHECK(cudaMalloc((float**)& d_A, nBytes));
	CHECK(cudaMalloc((float**)& d_B, nBytes));
	CHECK(cudaMalloc((float**)& d_C, nBytes));

	// transfer data from host to device
	CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

	// invoke kernel at host side
	int iLen = 512;
	dim3 block(iLen);
	dim3 grid((nElem + block.x - 1) / block.x);

	iStart = cpuSecond();
	sumArraysOnGPU << <grid, block >> > (d_A, d_B, d_C, nElem);
	CHECK(cudaDeviceSynchronize());
	iElaps = cpuSecond() - iStart;
	printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
		block.x, iElaps);

	// check kernel error
	CHECK(cudaGetLastError());

	// copy kernel result back to host side
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

	// check device results
	checkResult(hostRef, gpuRef, nElem);

	// free device global memory
	CHECK(cudaFree(d_A));
	CHECK(cudaFree(d_B));
	CHECK(cudaFree(d_C));

	// free host memory
	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	return(0);
}
```
输出如下：

```shell
Using Device 0: GeForce RTX 2080 Ti
Vector size 16777216
initialData Time elapsed 0.504589 sec
sumArraysOnHost Time elapsed 0.074646 sec
sumArraysOnGPU <<<  32768, 512  >>>  Time elapsed 0.000445 sec
Arrays match.
```

可以通过调整块的大小验证内核的性能变化，但是块的维度过小会提示错误信息，表示块的总数超过了一维网格的限制。

#### 2.2.2 用nvprof工具计时

- nvidia提供了一个名为`nvprof`的命令行分析工具，可以帮助从应用程序的CPU和GPU活动情况中获取时间线信息，包括内核执行、内存传输以及CUDA API的调用：
`$ nvprof [nvprof_args] <application> [application_args]`
可以使用以下命令获取帮助：
`$ nvprof --help`
可以这样去测试内核：
`$ nvprof ./sumArraysOnGPU-timer`
会有如下输出：

```shell
Using Device 0: GeForce RTX 2080 Ti
Vector size 16777216
initialData Time elapsed 0.501685 sec
sumArraysOnHost Time elapsed 0.074545 sec
sumArraysOnGPU <<<  32768, 512  >>>  Time elapsed 0.000445 sec
Arrays match.

==128333== Profiling application: ./sumArraysOnGPU-timer
==128333== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   59.08%  18.980ms         3  6.3268ms  6.2576ms  6.4360ms  [CUDA memcpy HtoD]
                   39.80%  12.786ms         1  12.786ms  12.786ms  12.786ms  [CUDA memcpy DtoH]
                    1.11%  358.08us         1  358.08us  358.08us  358.08us  sumArraysOnGPU(float*, float*, float*, int)
      API calls:   80.71%  163.40ms         3  54.466ms  202.69us  162.98ms  cudaMalloc
                   15.88%  32.160ms         4  8.0401ms  6.3565ms  12.915ms  cudaMemcpy
                    1.78%  3.6091ms         3  1.2030ms  304.53us  2.1247ms  cudaFree
                    0.87%  1.7634ms       288  6.1230us     140ns  389.50us  cuDeviceGetAttribute
                    0.27%  542.19us         1  542.19us  542.19us  542.19us  cudaGetDeviceProperties
                    0.20%  396.10us         1  396.10us  396.10us  396.10us  cudaDeviceSynchronize
                    0.19%  382.68us         3  127.56us  112.13us  151.57us  cuDeviceTotalMem
                    0.07%  148.70us         3  49.566us  44.330us  58.950us  cuDeviceGetName
                    0.02%  42.440us         1  42.440us  42.440us  42.440us  cudaLaunchKernel
                    0.00%  8.5100us         1  8.5100us  8.5100us  8.5100us  cudaSetDevice
                    0.00%  5.6100us         3  1.8700us  1.0800us  2.5300us  cuDeviceGetPCIBusId
                    0.00%  1.4800us         6     246ns     140ns     630ns  cuDeviceGet
                    0.00%     890ns         3     296ns     130ns     560ns  cuDeviceGetCount
                    0.00%     570ns         3     190ns     180ns     210ns  cuDeviceGetUuid
                    0.00%     440ns         1     440ns     440ns     440ns  cudaGetLastError
```

以上结果前半部分来自程序输出，后半部分来自`nvprof`输出，实际上`nvprof`的结果更加精确。

- CPU上消耗的时间、数据传输所用时间和GPU计算所用时间消耗如下图所示：

![](https://note.youdao.com/yws/api/personal/file/3398CDA4D9C74FCEA755E11AA469A983?method=download&shareKey=13de7a52d4c40e92736bd084a035f13c)

- 比较应用程序的性能叫理论界限最大化

在进行程序优化时，如何将应用程序和理论界限进行比较是很重要的。由`nvprof`得到的计数器可以帮助我们获取应用程序的指令和内存吞吐量。
以Tesla K10为例，我们可以得到理论上的比率：

1. Tesla K10单精度峰值浮点运算次数

    745MHz核心频率*2 GPU/芯片 \* (8个多处理器/*192 个浮点单元/*32核心/多处理器)*2 FLOPS/周期=4.58 TFLOPS(FLOPS表示每秒浮点运算次数)
2. Tesla K10内存带宽峰值

    2 GPU/芯片/*256 位/*2500 MHz内存时钟/*2 DDR/8 位/字节=320 GB/s

3. 指令比：字节

    4.58 TFLOPS/320 GB/s，也就是13.6个指令:1个字节

对于Tesla K10而言，如果你的应用程序每访问一个字节所产生的指令数多于13.6，那么应用程序会受算法性能限制，大多数高性能计算工作负载受内存带宽的限制。

### 2.3 组织并行程序

- 矩阵加法

传统方法是在内核中使用二维网格与二维块组织线程，但是这样无法获得最佳性能。

在矩阵加法中使用以下布局将有助于了解更多网格和块的启发性的用法：

1. 由二维线程块构成的二维网格
2. 由一维线程块构成的一维网格
3. 由一维线程块构成的二维网格

#### 2.3.1 使用块和线程建立索引

- 通常情况下，一个矩阵用行优先的方式在全局内存中线性存储：

![image](https://note.youdao.com/yws/api/personal/file/38869C438DFE4050B6946E43C28D9327?method=download&shareKey=ca236cc1e8480383bf6bb8874cda78dd)

- 在一个矩阵加法函数中，一个线程通常被分配一个数据元素处理，首先使用块和线程索引从全局内存中访问指定数据，通常情况下对一个二维示例来说，需要管理**三种**索引：

1. 线程和块索引
2. 矩阵中给定点的坐标
3. 全局线性内存中的偏移量

对于一个给定的线程，首先通过把线程和块索引映射到矩阵坐标上来获取线程块和线程索引的全局内存偏移量，然后将这些矩阵坐标映射到全局内存的存储单元中。

---

1. 用以下公式把线程和块索引映射到矩阵坐标上：

    ```c++
    ix = threadEdx.x + blockIdx.x * blockDim.x
    iy = threadIdx.y + blockIdx.y * blockDim.y
    ```

2. 用以下公式把矩阵`(ix, iy)`坐标映射到全局内存的索引/存储单元上

    ```c++
    idx = iy * nx + ix
    //  这是因为内存是线性排序的
    ```

下图说明了块和线程索引、矩阵坐标以及线性全局内存索引之间的对应关系：

![image](https://note.youdao.com/yws/api/personal/file/CEC2E5AA1E924CEE8F6D6AADF87B9C2C?method=download&shareKey=447a8c609b2fd46edef49a797fa6c52c)

---

- 如下的`printThreadInfo`函数被用于输出关于每个线程的以下信息：

1. 线程索引
2. 块索引
3. 矩阵坐标
4. 线性全局内存偏移量
5. 相应元素的值

**注意，返回的`(idx, idy)`信息和矩阵表示正好是相反的**

代码示例如下：

```c++
//  checkThreadIndex.cu
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call) \
{ \
	const cudaError_t error = call; \
	if (error != cudaSuccess) \
	{ \
		printf("Error: %s:%d, ", __FILE__, __LINE__); \
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
		exit(-10*error); \
	} \
}

void initialInt(int* ip, int size)
{
	for (int i = 0; i < size; ++i)
	{
		ip[i] = i;
	}
}

void printMatrix(int* C, const int nx, const int ny)
{
	int* ic = C;
	printf("\nMatrix: (%d, %d)\n", nx, ny);
	for (int iy = 0; iy < ny; ++iy)
	{
		for (int ix = 0; ix < nx; ++ix)
		{
			printf("%3d", ic[ix]);
		}
		ic += nx;
		printf("\n");
	}
	printf("\n");
}

__global__ void printThreadIndex(int* A, const int nx, const int ny)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * nx + ix;
	printf("thread_d (%d, %d) block_id (%d, %d) coordinate (%d, %d) "
		"global index %2d ival %2d\n", threadIdx.x, threadIdx.y, blockIdx.x,
		blockIdx.y, ix, iy, idx, A[idx]);
}


int main(int argc, char** argv)
{
	printf("%s Starting...\n", argv[0]);

	// get device information
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	// set matrix dimension
	int nx = 8;
	int ny = 6;
	int nxy = nx * ny;
	int nBytes = nxy * sizeof(float);

	// malloc host memory
	int* h_A;
	h_A = (int*)malloc(nBytes);

	// iniitialize host matrix with integer
	for (int i = 0; i < nxy; i++)
	{
		h_A[i] = i;
	}
	printMatrix(h_A, nx, ny);

	// malloc device memory
	int* d_MatA;
	CHECK(cudaMalloc((void**)& d_MatA, nBytes));

	// transfer data from host to device
	CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));

	// set up execution configuration
	dim3 block(4, 2);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	// invoke the kernel
	printThreadIndex << <grid, block >> > (d_MatA, nx, ny);
	CHECK(cudaGetLastError());

	// free host and devide memory
	CHECK(cudaFree(d_MatA));
	free(h_A);

	// reset device
	CHECK(cudaDeviceReset());

	return (0);
}
```

输出如下：

```shell
./checkThreadIndex Starting...
Using Device 0: GeForce RTX 2080 Ti

Matrix: (8, 6)
  0  1  2  3  4  5  6  7
  8  9 10 11 12 13 14 15
 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31
 32 33 34 35 36 37 38 39
 40 41 42 43 44 45 46 47

thread_d (0, 0) block_id (0, 0) coordinate (0, 0) global index  0 ival  0
thread_d (1, 0) block_id (0, 0) coordinate (1, 0) global index  1 ival  1
thread_d (2, 0) block_id (0, 0) coordinate (2, 0) global index  2 ival  2
thread_d (3, 0) block_id (0, 0) coordinate (3, 0) global index  3 ival  3
thread_d (0, 1) block_id (0, 0) coordinate (0, 1) global index  8 ival  8
thread_d (1, 1) block_id (0, 0) coordinate (1, 1) global index  9 ival  9
thread_d (2, 1) block_id (0, 0) coordinate (2, 1) global index 10 ival 10
```

三种索引间的关系如下图：

![image](https://note.youdao.com/yws/api/personal/file/E6122A599A9247B9A3083BA0063AFDA7?method=download&shareKey=a46ada7d9cfedc914920d01f57fd9d45)

#### 2.3.2 使用二维网格和二维块对矩阵求和

- 编写一个CPU加法函数：

```c++
void sumMartixOnHost(float *A, float *B, float *C, const int nx, const int)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy=0; iy<ny; ++iy)
    {
        for (int ix=0; ix<nx; ++ix)
        {
            ic[ix] = ia[ix] + ib[ix]
        }
        ia += nx; ib += nx; ic += nx;
    }
}
```

- 然后，创建一个核函数，目的是采用一个二维线程块进行矩阵求和：

```c++
__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, intnx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = ix + iy * nx;

    if (ix<nx && iy<ny)
    {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}
```

这个核函数的关键步骤是将每个线程从它的线程索引映射到全局线性内存索引中，如图：

![image](https://note.youdao.com/yws/api/personal/file/E46A2ECDEBC144308915E78E06926844?method=download&shareKey=88b155804a08eb8715fb6808a207f77a)

- 整体代码如下：

```c++
//  sumMatrixOnGPU-2D-grid-2D-block
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void initialData(float *ip, const int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }

    return;
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx,
                     const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];

        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}


void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
}

// grid 2D block 2D
__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx,
                                 int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int nx = 1 << 14;
    int ny = 1 << 14;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side
    double iStart = seconds();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    double iElaps = seconds() - iStart;
    printf("Matrix initialization elapsed %f sec\n", iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    iStart = seconds();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = seconds() - iStart;
    printf("sumMatrixOnHost elapsed %f sec\n", iElaps);

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((void **)&d_MatA, nBytes));
    CHECK(cudaMalloc((void **)&d_MatB, nBytes));
    CHECK(cudaMalloc((void **)&d_MatC, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    // invoke kernel at host side
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    iStart = seconds();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x,
           grid.y,
           block.x, block.y, iElaps);
    // check kernel error
    CHECK(cudaGetLastError());

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());

    return (0);
}
```

不断调整块的尺寸可以比较效果。

#### 2.3.3 使用一维网格和一维块对矩阵求和

- 为了使用一维网格和一维块，我们需要一个新的核函数，其中每个线程处理`ny`个数据元素，如图：

![image](https://note.youdao.com/yws/api/personal/file/417A154FCB654B3A84F49538A5CD9642?method=download&shareKey=dd20e271e2f4ca0446b3e6af42200d1c)

由于在新的核函数中每个线程都要处理`ny`个元素，与二维网格二维块的矩阵求和的核函数相比从线程和块索引到全局线性内存索引的映射都将有很大不同，由于在这个核函数启动中使用了一个一维块布局，因此只有`threadIdx.x`是有用的，并且使用内核中的一个循环来处理线程中的`ny`个元素：

```c++
__global__ void sumMatrixOnGPU1D(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x'
    if (ix < nx)
    {
        for (int iy=0; iy<ny; ++iy)
        {
            int idx = iy * nx + ix;
            MatC[idx] = MatA[idx] + MatB[idx]
        }
    }
}
```

一维网格和块的配置如下：

```c++
dim3 block(32, 1);
dim3 grid((nx + block.x - 1) / block.x, 1);
```

在一定范围内增加块的大小可以加快核函数的运行速度。

#### 2.3.4 使用二维网格和一维块对矩阵求和

- 当使用一维块和二维网格时，每个线程都只关注一个数据元素且网格的第二个维度为`ny`，如图：

![image](https://note.youdao.com/yws/api/personal/file/BADA98CB8C414134BA2B152B5D721EA5?method=download&shareKey=740745aafbdb8c8262b677297f913a60)

这种情况可以视作二维块二维网格的特殊情况，其中块的第二个维数是`1`，因此从块和线程到矩阵坐标的映射变成：

```c++
ix = threadIdx.x + blockIdx.x * blockDim.x;
iy = blockIdx.y;
//  这种情况下threadIdx.y = 0, blociDim.y = 1
```

这种新内核的优势是省去了一次整数乘法和一次整数加法

- 从这些例子中可以看出：

1. 改变执行配置对内核性能有影响
2. 传统的核函数实现一般不能获得最佳性能
3. 对于一个给定的核函数，尝试使用不同的网格和线程块大小可以获得不同的性能

### 2.4 设备管理

- 两种对GPU进行管理的方法：

1. CUDA运行时API函数
2. NVIDIA系统管理界面命令行`nvidia-smi`

#### 2.4.1 使用运行时API查询GPU信息

在CUDA运行时AIP中有很多函数可以帮助管理设备，可以使用以下函数查询GPU信息：

`cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);`

#### 2.4.2 确定最优GPU

`props.multiProcessorCount`

#### 2.4.3 使用`nvidia-smi`查询GPU信息

`nvidia-smi`是一个命令行工具，用于管理和监控GPU设备，查询GPU数量和设备ID：

```shell
nvidia-smi -L
```

查询GPU 0的详细信息：

`$ nvidia-smi -q -i 0`
