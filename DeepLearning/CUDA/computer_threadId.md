# 线程ID(`threadIdx`)计算

启动`kernel`时，需要指定`gridsize`和`blocksize`：

```c++
dim3 gridsize(2, 2);
dim3 blocksize(4, 4);
```

`gridsize`相当于是一个`2*2`的block，`gridDim.x, gridDim.y, gridDim.z`相当于这个`dim3`的`x, y, z`方向的维度，在这里是`2, 2, 1`序号从0到3，并且是从左到右的顺序，如下：

`grid`中的`blockIdx`序号标注情况为：

```c++
 0   1
 2   3
```

`blocksize`则是指`block`中`thread`的情况，`blockDim.x, blockDim.y, blockDim.z`相当于这个`dim3`的`x, y, z`方向的维度，在这里是`4, 4, 1`序号从0到15，并且是左到右的顺序，如下：

```c++
 0   1   2   3
 4   5   6   7
 8   9   10  11
 12  13  14  15
```

这样就一目了然，然后求实际的`threadId`的时候：

- 1D grid of 1D blocks

```c++
int threadId = blockIdx.x * blockDim.x + threadIdx.x
```

示意图：
|**表格中的blockDim.x=4**|threadIdx.x|threadIdx.x|threadIdx.x|threadIdx.x|
|---|---|---|---|---|---|
|**blockIdx.x**|线程块0|线程0|线程1|线程2|线程3|
|**blockIdx.x**|线程块1|线程0|线程1|线程2|线程3|
|**blockIdx.x**|线程块2|线程0|线程1|线程2|线程3|
|**blockIdx.x**|线程块3|线程0|线程1|线程2|线程3|
|**blockIdx.x**|线程块4|线程0|线程1|线程2|线程3|
`blockDim.x`表示`block`在`x`轴方向的`thread`数量

- 1D grid of 2D blocks

```c++
int threadId = blockIdx.x * blockDim.y * blockDim.x
             + threadIdx.y * blockDim.x
             + threadIdx.x
```

- 1D grid of 3D blocks

```c++
int threadId = blockIdx.x * blockDim.z * blockDim.y * blockDim.x
             + threadIdx.z * blockDim.y * blockDim.x
             + threadIdx.y * blockDim.x
             + threadIdx.x
```

- 2D grid of 1D blocks

```c++
int blockId = blockIdx.y * gridDim.x
            + blockIdx.x
int threadId = blockId * blockDim.x
             + threadIdx.x
```

- 2D grid of 2D blocks

```c++
int blockId = blockIdx.y * gridDim.x
            + blockIdx.x
int threadId = blockId * blockDim.y * blockDim.x
             + threadIdx.y * blockDim.x
             + threadIdx.x
```

- 2D grid of 3D blocks

```c++
int blockId = blockIdx.y * gridDim.x
            + blockIdx.x
int threadId = blockId * blockDim.z * blockDim.y * blockDim.x
             + threadIdx.z * blockDim.x * blockDim.y
             + threadIdx.y * blockDim.x
             + threadIdx.x
```

- 3D grid of 1D blocks

```c++
int blockId = blockIdx.z * gridDim.y * gridDim.x
            + blockIdx.y * gridDim.x
            + blockIdx.x
int threadId = blockId * blockDim.x
             + threadIdx.x
```

- 3D grid of 2D blocks

```c++
int blockId = blockIdx.z * gridDim.y * gridDim.x
            + blockIdx.y * gridDim.x
            + blockIdx.x
int threadId = blockId * blockDim.y * blockDim.x
             + threadIdx.y * blockDim.x
             + threadIdx.x
```

- 3D grid of 3D blocks

```c++
int blockId = blockIdx.z * gridDim.y * gridDim.x
            + blockIdx.y * gridDim.x
            + blockIdx.x
int threadId = blockId * blockDim.z * blockDim.y * blockDim.x
             + threadIdx.z * blockDim.y * blockDim.x
             + threadIdx.y * blockDim.x
             + threadIdx
```

## `2D*2D`表示`threadId`二维位置的示意图

```c++
x = blockIdx.x * blockDim.x + threadIdx.x
y = blockIdx.y * blockDim.y + threadIdx.y
```

![image](https://img-blog.csdn.net/20160809150525718)

## CUDA `grid block thread`和线程的绝对索引

`grid, block`都看成两个不同的坐标系，`grid`轴上单位刻度表示`block`索引，`block`轴上单位刻度表示`thread`索引。

**`gridDim.x`表示`grid`的宽度，`gridDim.y`表示`grid`高度，`grod`包含的是`block`**

**`blockIdx.x`相当于在`grid`中的`x`坐标，`blockIdx.y`相当于在`grid`中的`y`坐标**

**`blockDim.x`表示`block`的宽度，`blockDim.y`表示`block`高度，`block`包含的是`thread`**

**`threadIdx.x`相当于在`block`中的`x`坐标，`threadIdx.y`相当于在`block`中的`y`坐标**

如：(注:`x`和`y`都从`0`开始)

- (1)

```c++
idx = blockIdx.x * blockDim.x + threadIdx.x
=> grid坐标系x轴单位刻度block
-> block坐标系x轴单位刻度thread

idy = blockIdx.y * blockDim.y + threadIdx.y
=> grid坐标系y轴单位刻度block
-> block坐标系y轴单位刻度thread

===>(idx, idy)就是grid(thread)坐标系的绝对坐标

那么一维数组索引：
= idy * grid的(x轴)宽度(单位刻度是线程) + idx

thread_idx = ((gridDim.x * blockDim.x) * idy) + idx;
```

- (2)

先求已经`thread`满了的`block`数量，算线程总数，再算剩下`thread`不满的`block`中线程(一维)`idx`，两次计算结果求和
