[代码链接](https://github.com/MrDotZLC/cuda_practice)

# 零、目录
11. CUDA流
12. 统一内存编程




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
// 见cuda_stream.cu
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
// 见cs_transfer.cu

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
假设每个流都有三个步骤：主机数据复制到设备（H2D）、核函数运行（KER）、设备内存复制到主机（D2H）。
**流并发核心：** PCIe copy engine 是“硬件固定数量”的
- **H2D copy engine：1 个**
- **D2H copy engine：1 个**
- 不能：
    - 2 个 H2D 同时跑
    - 2 个 D2H 同时跑
```
// 每个CUDA流的数据处理量是单个流的1/4
// 将12个操作减少至6个，效率提升12/6=2倍
// 提升比：3n/(3+n-1)，理论上最高提升3倍
Stream 1：H2D -> KER -> D2H
Stream 2：       H2D -> KER -> D2H
Stream 3：              H2D -> KER -> D2H
Stream 4：                     H2D -> KER -> D2H
```




# 十二、统一内存编程
## 12.1 基本概念
- **统一内存（UM/Managed Memory）**：CPU 与 GPU 共享的一块虚拟内存空间。
- **核心特点**：
    1. 单一指针访问：CPU 和 GPU 使用同一个指针。
    2. 自动数据迁移：系统根据访问模式自动在 CPU/GPU 之间迁移数据。
    3. 简化编程：无需显式调用 `cudaMemcpy`。
## 12.2 **工作原理**
- 基于 **虚拟内存机制** 和 **页表（Page Table）**。
- **页粒度管理**：
    - 默认 4 KB 页大小。
    - GPU 或 CPU 访问不在本地的页会触发 **页错误（Page Fault）**，CUDA 驱动负责迁移。
- **单一视图一致性（Single View Consistency）**：CPU 和 GPU 看到的都是同一块逻辑内存。
## 12.3 **声明与使用**
### 12.3.1 动态统一内存
#### 1 内存分配
```cpp
float *data;
cudaMallocManaged(&data, N * sizeof(float)); // N为元素数量
```
#### 2 CPU/GPU 访问
```cpp
// CPU访问
for (int i = 0; i < N; i++) data[i] = i;

// GPU访问
kernel<<<blocks, threads>>>(data);
cudaDeviceSynchronize();

// CPU再次访问
float sum = 0;
for (int i = 0; i < N; i++) sum += data[i];
```
#### 3 释放内存
```cpp
cudaFree(data);
```
#### 4 可选标志
- `cudaMemAttachGlobal`：全局可访问（默认）
- `cudaMemAttachHost`：仅CPU可见，设备访问需显式迁移
- `cudaMemAttachSingle`：仅分配它的GPU可访问
### 12.3.2 静态统一内存
在所有函数外，使用\_\_device\_\_和\_\_managed\_\_修饰符。
```
__device__ __managed__ int ret[1000];
__global__ void AplusB(int a, int b)
{
    ret[threadIdx.x] = a + b + threadIdx.x;
}
  
int main(int argc, char *argv[])
{
    AplusB<<<1, 1000>>>(10, 100);
    cudaDeviceSynchronize();
    for (int i = 0; i < 1000; i++)
    {
        printf("%d: A+B = %d\n", i, ret[i]);
    }
}
```
## 12.4 **页错误（Page Fault）机制**
### 4.1 定义
- 页错误是 **访问不在本地内存的统一内存页时触发的事件**。
- CUDA 驱动捕获页错误，并自动迁移数据到访问方内存。
### 4.2 触发条件
1. GPU访问CPU内存中的数据。
2. CPU访问GPU内存中的数据。
3. 随机访问不同页，跨页访问。
### 4.3 内部机制
1. GPU访问统一内存：
    - 检查页是否在本GPU显存。
    - 否 → 触发 Page Fault → CUDA 驱动迁移页 → 更新页表 → 重试访问。
2. CPU访问统一内存：
    - 检查页是否在主机内存。
    - 否 → 页错误 → 迁移页回CPU → 访问继续。
## 12.5 **性能影响**
- **Page Fault 延迟**：迁移数据涉及 PCIe/NVLink，开销大。
- **频繁页错误（Thrashing）**：
    - 随机访问不同页导致频繁迁移。
    - 性能可能下降十倍以上。
- **顺序访问优化**：
    - 避免随机跨页访问，减少页错误。
## 12.6 **性能优化策略**
1. **预取（Prefetch）**：
```cpp
cudaMemPrefetchAsync(data, N*sizeof(float), device_id);
```
2. **批量处理**：
    - 将大数组分块，每次迁移部分数据。
3. **流（Stream）与异步迁移结合**：
    - 隐藏迁移延迟。
4. **减少 CPU/GPU 交替访问**：
    - 在 GPU 内核连续完成操作，再回 CPU。
## 12.7 **优缺点总结**

| 特性    | 统一内存           | 显式 cudaMemcpy |
| ----- | -------------- | ------------- |
| 编程复杂度 | 低              | 高             |
| 内存迁移  | 自动按需           | 手动            |
| 性能控制  | 受访问模式影响        | 高，可优化批量拷贝     |
| 大数据支持 | 支持（依赖系统内存）     | 受限于GPU内存      |
| 使用场景  | 快速原型、CPU/GPU协同 | 高性能核心计算       |
|       |                |               |
## 12.8 **页错误与迁移示意**
```
CPU/GPU访问统一内存
       │
       ▼
检查页是否在本地
       │
       ├── 是：直接访问
       │
       └── 否：触发 Page Fault
             │
             ▼
        CUDA 驱动迁移页
             │
             ▼
       更新页表，重试访问
```
- 顺序访问：连续访问同页，页错误少，性能高。
- 随机访问：跨页访问，频繁触发 Page Fault，性能低。


# 