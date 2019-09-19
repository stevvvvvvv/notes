# 利用shared memory——矩阵转置

[toc]

## CPU矩阵转置

```c++
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define LOG_
#define N 1024

/* 转置 */
void transposeCPU( float in[], float out[] )
{
    for ( int j = 0; j < N; j++ )
    {
        for ( int i = 0; i < N; i++ )
        {
            out[j * N + i] = in[i * N + j];
        }
    }
}


/* 打印矩阵 */
void logM( float m[] )
{
    for ( int i = 0; i < N; i++ )
    {
        for ( int j = 0; j < N; j++ )
        {
            printf( "%.1f ", m[i * N + j] );
        }
        printf( "\n" );
    }
}

int main()
{
    int size = N * N * sizeof(float);
    float *in = (float *) malloc( size );
    float *out = (float *) malloc( size );

    /* 矩阵赋值 */
    for ( int i = 0; i < N; ++i )
    {
        for ( int j = 0; j < N; ++j )
        {
            in[i * N + j] = i * N + j;
        }
    }

    struct timeval  start, end;
    double      timeuse;
    int     sum = 0;
    gettimeofday( &start, NULL );

    transposeCPU( in, out );

    gettimeofday( &end, NULL );
    timeuse = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
    printf( "Use Time: %fs\n", timeuse );

#ifdef LOG
    logM( in );
    printf( "\n" );
    logM( out );
#endif

    free( in );
    free( out );
    return(0);
}
```

## GPU实现

### 简单移植

```c++
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 1024
#define LOG_

/* 转置 */
__global__ void transposeSerial( float in[], float out[] )
{
    for ( int j = 0; j < N; j++ )
        for ( int i = 0; i < N; i++ )
            out[j * N + i] = in[i * N + j];
}

/* 打印矩阵 */
void logM( float m[] ){...}

int main()
{
    int size = N * N * sizeof(float);

    float *in, *out;

    cudaMallocManaged( &in, size );
    cudaMallocManaged( &out, size );

    for ( int i = 0; i < N; ++i )
        for ( int j = 0; j < N; ++j )
            in[i * N + j] = i * N + j;

    struct timeval  start, end;
    double      timeuse;
    gettimeofday( &start, NULL );

    transposeSerial << < 1, 1 >> > (in, out);

    cudaDeviceSynchronize();

    gettimeofday( &end, NULL );
    timeuse = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
    printf( "Use Time: %fs\n", timeuse );


#ifdef LOG
    logM( in );
    printf( "\n" );
    logM( out );
#endif

    cudaFree( in );
    cudaFree( out );
}
```

效率极低，比CPU还糟糕。

### 单`block`

单`block`可以开1024个线程

```c++
/* 转置 */
__global__ void transposeParallelPerRow( float in[], float out[] )
{
    int i = threadIdx.x;
    for ( int j = 0; j < N; j++ )
        out[j * N + i] = in[i * N + j];
}

int main()
{
    ...
    transposeParallelPerRow << < 1, N >> > (in, out);
    ...
}
```

效率有较大提升

### tile

如果利用多个`block`把矩阵切成更多的tile，效率会有更大的提升：

```c++
/* 转置 */
__global__ void transposeParallelPerElement( float in[], float out[] )
{
    int i = blockIdx.x * K + threadIdx.x;
    /* column */
    int j = blockIdx.y * K + threadIdx.y;
    /* row */
    out[j * N + i] = in[i * N + j];
}

int main()
{
    ...
    dim3 blocks( N / K, N / K );
    dim3 threads( K, K );

    ...

    transposeParallelPerElement << < blocks, threads >> > (in, out);
    ...
}
```

## GPU理论效率峰值

主要通过`Memory Clock Rate`和`Memory Bus Width`计算

## 进一步提升效率（使用shared memory）

读数据的时候是连着读的，一个`warp`读32个数据，可以同步操作，但是写的时候是散开写的，有一个较大的步长，导致效率下降。所以需要借助shared memory，用来转置数据。

```c++
/* 转置 */
__global__ void transposeParallelPerElementTiled( float in[], float out[] )
{
    int in_corner_i = blockIdx.x * K, in_corner_j = blockIdx.y * K;
    int out_corner_i    = blockIdx.y * K, out_corner_j = blockIdx.x * K;

    int x = threadIdx.x, y = threadIdx.y;

    __shared__ float tile[K][K];

    tile[y][x] = in[(in_corner_i + x) + (in_corner_j + y) * N];
    __syncthreads();
    out[(out_corner_i + x) + (out_corner_j + y) * N] = tile[x][y];
}

int main()
{

    ...
    dim3 blocks( N / K, N / K );
    dim3 threads( K, K );

    struct timeval  start, end;
    double      timeuse;
    gettimeofday( &start, NULL );

    transposeParallelPerElementTiled << < blocks, threads >> > (in, out);
    ...

}
```

效率会有进一步提升。

GPU存储架构：
![GPU](https://upload-images.jianshu.io/upload_images/5319256-d61472717c628bdc.png?imageMogr2/auto-orient/strip|imageView2/2/w/1000/format/webp)
