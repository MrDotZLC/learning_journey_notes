[代码链接](https://github.com/MrDotZLC/cuda_practice)

# 零、目录
1. CUDA流
2. 




# 十一、CUDA流
### 核函数内部的并行
- 主要指的是在核函数内部的并行计算，利用线程、线程块以及多级线程层次结构进行并行处理。
### 核函数外部的并行
核函数外部的并行主要关注的是 **核函数计算和其他计算任务或数据传输** 之间的并行性，具体包括以下五种情况：
1. **核函数计算与数据传输之间的并行**
2. **主机计算与数据传输之间的并行**
3. **不同的数据传输之间的并行**
    - 在多个数据传输任务之间进行并行操作，尤其是多个 `cudaMemcpy` 操作可以并行执行，避免数据传输时的空闲时间。
4. **核函数计算与主机计算之间的并行**
5. **不同核函数之间的并行**
    - 如果核函数的并行规模足够大，可以在同一个设备上同时运行多个核函数（但这通常在单一核函数已经达到最大并行度时效果不明显）。
### 提高性能的关键
- **减少主机与设备之间的数据传输**
- **设备端完成所有计算**：
- **合理使用CUDA流（CUDA streams）**：通过流的并行机制，可以实现在核函数计算、数据传输和主机计算之间的并行，从而优化整体性能。
## 11.1 CUDA流
主机和设备都能发出，CUDA流式并行的，若非指定，一般是默认流（default stream），又称空流（null stream）。
```
// 见cuda_stream.cu
cudaStream_t stream_1; // 流类型别名
cudaStreamCreate(&stream_1); // 创建流，注意要传流的地址
cudaStreamDestroy(stream_1); // 销毁流
cudaError_t cudaStreamSynchronize(cudaStream_t stream); // 阻塞主机，等待流中所有操作
cudaError_t cudaStreamQuery(cudaStream_t stream); // 不阻塞主机，仅检查
```
## 11.2 主机操作与设备操作
### 默认流：
主机操作：正常C++代码，同步/阻塞
	设备操作：核函数代码    ，主机中异步/非阻塞，同CUDA流中同步/阻塞
优化方向：
		1个核函数：
			若与C++代码无依赖关系，单个核函数主导先执行，C++代码后执行
		多个核函数：
			创建非默认流执行核函数
### 非默认流：
```
// 核函数的3中调用方式
my_kernel<<<N_grid, N_block>>>(函数参数);
my_kernel<<<N_grid, N_block, N_shared>>>(函数参数);
my_kernel<<<N_grid, N_block, N_shared, stream_id>>>(函数参数); // 不用N_shared，设为0
```
- N_grid ：网格大小，类型为dim3或整数
- N_block ：线程块大小，类型为dim3或整数
- N_shared ：动态共享内存字节数
- stream_id ：CUDA流编号
### 异步数据传输
异步传输函数cudaMemcpyAsync由GPU 中的DMA（direct memory access）[[GPU 中的 DMA]]直接实现。
```
// 见cuda_stream.cu

// 内存异步复制函数
cudaError_t cudaMemcpyAsync
(
void *dst,
const void *src,
size_t count,
enum cudaMemcpyKind kind,
cudaStream_t stream
);

// 示例
// 1. 分配不可分页主机内存
cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaHostAlloc(void** ptr, size_t size, size_t flags);

// 2. 异步传输内存
cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);

// 3. 释放主机内存
cudaError_t cudaFreeHost(void* ptr);
```
