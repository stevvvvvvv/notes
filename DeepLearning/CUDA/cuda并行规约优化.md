# CUDA并行规约优化

[toc]

## `cuda-gdb`断点调试

`gdb`解读：<https://linuxtools-rst.readthedocs.io/zh_CN/latest/tool/gdb.html>

在命令行中启动`cuda-gdb`调试：

```shell
nvcc -g -G xxx.cu -o xxx.o
```

然后使用`cuda-gdb xxx.o`即可对程序进行调试。

## 未优化并行规约

按照常规思路的加法运算：
![add](https://upload-images.jianshu.io/upload_images/5319256-439447a0d686dcb9.jpg?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

代码示意：

```c++
#include <stdio.h>

const int   threadsPerBlock = 512;
const int   N       = 2048;
const int   blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock; /* 4 */

__global__ void ReductionSum( float * d_a, float * d_partial_sum )
{
    /* 申请共享内存, 存在于每个block中 */
    __shared__ float partialSum[threadsPerBlock];

    /* 确定索引 */
    int i   = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    /* 传global memory数据到shared memory */
    partialSum[tid] = d_a[i];

    /* 传输同步 */
    __syncthreads();

    /* 在共享存储器中进行规约 */
    for ( int stride = 1; stride < blockDim.x; stride *= 2 )
    {
        if ( tid % (2 * stride) == 0 )
            partialSum[tid] += partialSum[tid + stride];
        __syncthreads();
    }

    /* 将当前block的计算结果写回输出数组 */
    if ( tid == 0 )
        d_partial_sum[blockIdx.x] = partialSum[0];
}


int main()
{
    int size = sizeof(float);

    /* 分配显存空间 */
    float   * d_a;
    float   * d_partial_sum;

    cudaMallocManaged( (void * *) &d_a, N * size );
    cudaMallocManaged( (void * *) &d_partial_sum, blocksPerGrid * size );

    for ( int i = 0; i < N; ++i )
        d_a[i] = i;

    /* 调用内核函数 */
    ReductionSum << < blocksPerGrid, threadsPerBlock >> > (d_a, d_partial_sum);

    cudaDeviceSynchronize();

    /* 将部分和求和 */
    int sum = 0;
    for ( int i = 0; i < blocksPerGrid; ++i )
        sum += d_partial_sum[i];

    printf( "sum = %d\n", sum );

    /* 释放显存空间 */
    cudaFree( d_a );
    cudaFree( d_partial_sum );

    return(0);
}
```

## 优化后并行规约

仅需要改变步长即可：
![add](https://upload-images.jianshu.io/upload_images/5319256-55242b13faac6493.jpg?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

`kernel`部分的代码改动：

```c++
__global__ void ReductionSum( float * d_a, float * d_partial_sum )
{
    // 相同, 略去
    /* 在共享存储器中进行规约 */
    for ( int stride = blockDim.x / 2; stride > 0; stride /= 2 )
    {
        if ( tid < stride )
            partialSum[tid] += partialSum[tid + stride];
        __syncthreads();
    }
    // 相同, 略去
}
```

## 结果分析

第二种方案可以更快地将更多`warp`闲置，交给`GPU`调度。如图：
![befort](https://upload-images.jianshu.io/upload_images/5319256-2f505e1fe1174734.jpg?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)
![after](https://upload-images.jianshu.io/upload_images/5319256-dfeadc72a409fe22.jpg?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

图一在运算一次后，没有`warp`被闲置，而图二在运算一次后就闲置了2个`warp`。
