# 1. 操作系统视角的内存模型：Virtual Memory

现代操作系统（Linux / Windows）采用 **虚拟内存 (Virtual Memory)** 机制。

CPU 访问的地址并不是物理地址，而是 **虚拟地址 (Virtual Address)**。  
操作系统通过 **页表 (Page Table)** 维护如下映射关系：

[  
VA \xrightarrow{\text{Page Table}} PA  
]

其中：

- (VA)：Virtual Address（虚拟地址）
    
- (PA)：Physical Address（物理地址）
    

虚拟地址空间被划分为固定大小的 **页 (Page)**：

[  
\text{Page Size} \approx 4,KB \quad (\text{x86 Linux 默认})  
]

系统通过 **MMU (Memory Management Unit)** 完成地址转换。

---

# 2. 可分页内存（Pageable Memory）

## 2.1 定义

**Pageable Memory** 是默认的用户态内存分配方式，例如：

```cpp
malloc()
new
std::vector
stack variable
```

示例：

```cpp
std::vector<float> data(N);
float* ptr = new float[N];
```

底层实际调用：

```
malloc → glibc allocator → mmap/brk
```

## 2.2 操作系统行为

当物理内存紧张时，OS 会执行 **页面置换 (Paging)**：

### Swap Out

将不活跃页面写入磁盘：

[  
\text{RAM} \rightarrow \text{Swap Disk}  
]

### Swap In

当再次访问该页时：

1. 触发 **Page Fault**
    
2. OS 从磁盘加载数据
    
3. 更新页表
    

流程：

```
CPU access VA
     ↓
Page table miss
     ↓
Page Fault interrupt
     ↓
OS loads page from disk
     ↓
Update page table
```

## 2.3 核心问题

Pageable Memory 具有两个关键特性：

|特性|说明|
|---|---|
|物理地址不稳定|OS 可以随时重新映射|
|可能不在 RAM|可能被 swap 到磁盘|

因此：

```
Virtual Address ≠ 固定 Physical Address
```

这对 GPU DMA 传输是致命问题。

---

# 3. 页锁定内存（Pinned Memory / Page-Locked Memory）

## 3.1 定义

Pinned Memory 是 **锁定在物理 RAM 的内存**。

CUDA API：

```cpp
cudaMallocHost()
cudaHostAlloc()
```

示例：

```cpp
float* data;

cudaMallocHost((void**)&data, N * sizeof(float));
```

## 3.2 操作系统约束

Pinned Memory 具有以下性质：

|属性|说明|
|---|---|
|不可 Swap|永远不会被写入磁盘|
|物理地址固定|Page Table 不允许重新映射|
|RAM 常驻|始终存在于物理内存|

形式化约束：

[  
PA(t) = \text{constant}  
]

即：

```
Physical Address 在整个生命周期保持不变
```

---

# 4. GPU 传输机制：DMA

CPU 与 GPU 的数据传输通过 **PCIe DMA 控制器**完成。

DMA（Direct Memory Access）特点：

|特性|说明|
|---|---|
|不经过 CPU|硬件直接读写|
|使用物理地址|不理解虚拟地址|
|传输期间地址必须稳定|否则数据损坏|

DMA 访问流程：

```
GPU DMA Engine
     ↓
PCIe Bus
     ↓
System RAM (Physical Address)
```

关键约束：

[  
DMA: \quad \text{requires fixed physical memory}  
]

因此：

```
DMA 只能访问 Pinned Memory
```

---

# 5. 为什么 cudaMemcpyAsync 在 Pageable Memory 上退化为同步

## 5.1 示例代码

```cpp
cudaMemcpyAsync(
    d_dst,
    h_src_pageable,
    size,
    cudaMemcpyHostToDevice,
    stream
);
```

假设：

```
h_src_pageable = malloc/new/std::vector
```

即 **Pageable Memory**。

---

## 5.2 CUDA Driver 的内部处理

CUDA 驱动不能直接让 DMA 访问 Pageable Memory。

否则如果 OS 重新映射页面：

```
DMA → invalid physical address
```

可能导致：

- 数据损坏
    
- Kernel panic
    
- 系统 crash
    

因此 CUDA Driver 使用 **Staging Buffer 机制**。

---

## 5.3 Staging Buffer 机制

内部流程：

```
Step1  分配隐藏 Pinned Buffer
Step2  CPU memcpy (Pageable → Pinned)
Step3  GPU DMA (Pinned → Device)
Step4  API 返回
```

详细过程：

```
User Memory (Pageable)
        │
        │ CPU memcpy  ← 同步
        ▼
Staging Buffer (Pinned)
        │
        │ PCIe DMA  ← 异步
        ▼
GPU Global Memory
```

关键点：

```
Step2 是 CPU 同步操作
```

因此：

```
cudaMemcpyAsync 被阻塞
```

从应用视角：

```
Async API → 实际同步行为
```

结果：

```
CPU-GPU Overlap 失效
```

---

# 6. 常见认知误区：vector vs float*

## 6.1 错误理解

错误结论：

```
vector<float> = pageable
float* = pinned
```

该说法是错误的。

---

## 6.2 本质：分配方式决定内存类型

指针类型 **不决定内存属性**。

```
float* 只是地址变量
```

内存类型取决于 **分配 API**。

|分配方式|内存类型|
|---|---|
|malloc/new|Pageable|
|cudaMallocHost|Pinned|

---

## 6.3 std::vector 的行为

标准 vector 使用：

```cpp
std::allocator
```

底层实现：

```
std::allocator
    ↓
operator new
    ↓
malloc
```

因此：

```
std::vector → Pageable Memory
```

---

## 6.4 float* 成员变量的真实情况

示例：

```cpp
class MyClass {

public:

    float* data;

    MyClass(size_t n) {
        cudaMallocHost((void**)&data, n*sizeof(float));
    }

};
```

这里：

```
float* → 指向 Pinned Memory
```

如果写成：

```cpp
data = new float[n];
```

则：

```
Pageable Memory
```

结论：

```
变量类型 ≠ 内存类型
分配 API 才决定内存属性
```

---

# 7. 为什么不能全部使用 Pinned Memory

Pinned Memory 并非免费资源。

## 7.1 分配成本高

Pinned Memory 分配涉及：

```
OS page lock
TLB 更新
kernel syscall
```

因此：

```
cudaMallocHost >> malloc
```

分配延迟明显更高。

---

## 7.2 消耗系统 RAM

Pinned Memory **强制占用物理内存**：

[  
Pinned + System + Applications \le RAM  
]

如果锁定大量内存：

```
系统可用 RAM ↓
Swap 使用 ↑
```

最终结果：

```
系统卡顿 / OOM / 进程被 kill
```

经验建议：

```
Pinned Memory < 50% RAM
```

---

# 8. 工程实践：Pinned std::vector

为了同时获得：

- vector 自动管理
    
- Pinned Memory
    

需要 **Custom Allocator**。

---

# 9. CUDA Pinned Allocator 实现

```cpp
#include <vector>
#include <cuda_runtime.h>

template <typename T>
struct CudaPinnedAllocator {

    using value_type = T;

    CudaPinnedAllocator() = default;

    template <class U>
    CudaPinnedAllocator(const CudaPinnedAllocator<U>&) {}

    T* allocate(std::size_t n) {

        T* ptr = nullptr;

        cudaError_t status =
            cudaHostAlloc(
                (void**)&ptr,
                n * sizeof(T),
                cudaHostAllocDefault
            );

        if (status != cudaSuccess)
            throw std::bad_alloc();

        return ptr;
    }

    void deallocate(T* p, std::size_t) {

        cudaFreeHost(p);

    }
};
```

定义别名：

```cpp
template <typename T>
using pinned_vector = std::vector<T, CudaPinnedAllocator<T>>;
```

使用方式：

```cpp
pinned_vector<float> data(N);
```

---

# 10. 为什么 pinned_vector 可以实现真正异步

调用：

```cpp
cudaMemcpyAsync(
    d_ptr,
    data.data(),
    size,
    cudaMemcpyHostToDevice,
    stream
);
```

由于：

```
data 已经是 Pinned Memory
```

Driver 行为：

```
直接启动 DMA
```

流程：

```
Pinned Host Memory
       │
       │ PCIe DMA
       ▼
GPU Global Memory
```

因此：

```
cudaMemcpyAsync 立即返回
```

实现：

```
CPU-GPU Overlap
```

---

# 11. 带宽提升原因

Pageable Memory 传输：

```
Pageable
   ↓ CPU memcpy
Pinned Staging
   ↓ DMA
GPU
```

Pinned Memory 传输：

```
Pinned
   ↓ DMA
GPU
```

减少一次：

```
CPU memcpy
```

PCIe 有效带宽通常提升：

```
20% ~ 50%
```

---

# 12. 生命周期管理

Pinned Memory 在异步传输中必须保持有效。

错误示例：

```cpp
{
    pinned_vector<float> data(N);

    cudaMemcpyAsync(...);

} // data 析构
```

GPU 仍可能在读取：

```
invalid memory access
```

正确方式：

```cpp
cudaMemcpyAsync(...);

cudaStreamSynchronize(stream);
```

或：

```cpp
cudaEventSynchronize(event);
```

确保：

```
DMA 完成 → 再释放内存
```

---

# 13. CUDA Host Memory 使用策略

工程建议：

|场景|建议|
|---|---|
|GPU 高频 IO Buffer|使用 Pinned Memory|
|临时 CPU 数据|使用 Pageable|
|大规模 Host Buffer|控制比例|
|高频分配|使用 Memory Pool|

推荐模式：

```
Pinned Memory Pool
        ↓
复用
        ↓
GPU Transfer
```

避免：

```
cudaMallocHost / cudaFreeHost 高频调用
```

---

# 14. 总结

核心规律：

```
DMA 只能访问固定物理地址
```

因此：

```
GPU Async Copy → 必须使用 Pinned Memory
```

关键结论：

|机制|Pageable|Pinned|
|---|---|---|
|Physical Address|不稳定|固定|
|Swap|允许|禁止|
|DMA|不安全|安全|
|cudaMemcpyAsync|退化同步|真异步|

工程原则：

```
Pinned 用于 GPU IO
Pageable 用于普通数据
```