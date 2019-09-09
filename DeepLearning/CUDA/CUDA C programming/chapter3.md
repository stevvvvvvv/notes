# CUDA C编程权威指南(chapter 3)
### 第三章 CUDA执行模型
- 通过配置文件驱动的方法优化内核
- 理解线程束执行的本质
- 增大GPU的并行性
- 掌握网格和线程块的启发式配置
- 学习多种CUDA的性能指标和事件
- 了解动态并行与嵌套执行

#### 3.1 CUDA执行模型概述
##### 3.1.1 GPU架构概述
GPU是围绕一个流式多处理器(SM)可扩展阵列搭建的：
![](https://note.youdao.com/yws/api/personal/file/971361E58561486B9867D989B511476F?method=download&shareKey=4fc4632f49acab6a81f7319404f57c11)

GPU中的每一个SM都能支持数百个线程并发执行，多个线程块可能被分配到一个SM上，这是通过SM资源的可用性进行调度的。

CUDA采用单指令多线程(SMIT)架构管理和执行线程，每32个线程为一组，被称为线程束(wrap)，每个SM都将分配给它的线程块划分到包含32个线程的线程束中进行调度、执行。

##### 3.1.2 Fermi架构
Fermi有512个加速器核心，被称为CUDA核心。每个CUDA核心都有一个全流水线的整数算术逻辑单元(ALU)和一个浮点运算单元(FPU)。CUDA核心被组织到16个SM中，每个SM含有32个CUDA核心。
##### 3.1.3 Kepler架构
Kepler K20X芯片包含15个SM，每个SM单元包含192个单精度CUDA核心，64个双精度单元，32个特殊功能单元(SFU)和32个加载/存储单元(LD/ST)。Kepler架构有三个重要创新：

- 强化的SM
- 动态并行
- Hyper-Q技术

#### 3.2 理解线程束执行的本质
##### 3.2.1 线程束和线程块
线程束是SM中基本的执行单元。当一个线程块的网格被启动后，网格中的线程块分布在SM中。一旦线程块被调度到一个SM上，线程块中的线程会被进一步划分为线程束，一个线程束由32个连续的线程组成。

二维线程块中每个线程的独特标识符计算方法：
`threadIdx/y * blockDim.x + threadIdx.x`

三维线程块中每个线程的独特标识符计算方法：
`threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x`
