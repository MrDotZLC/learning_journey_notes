## 1. 核心定义与设计初衷

### 1.1 什么是 GroupGEMM？

GroupGEMM（在 NVIDIA CUTLASS 库中称为 `GemmGrouped`）允许在**单次 CUDA Kernel Launch** 中，计算一组维度完全不同（$M_i, N_i, K_i$）、数据指针各异的矩阵乘法任务。

$$C_i = \alpha A_i B_i + \beta C_i \quad \text{for} \quad i \in \{0, 1, ..., N_{groups}-1\}$$

### 1.2 解决的痛点

- **抹除 CPU 启动开销 (Launch Overhead)**：传统方法中，处理 100 个不同尺寸的矩阵需要 100 次 `cudaLaunchKernel`，CPU 端的调度延迟往往远超 GPU 的计算时间。GroupGEMM 将其合并为 1 次启动。
- **拒绝无效填充 (Zero Padding)**：相比传统的 Batched GEMM（要求所有矩阵尺寸一致），GroupGEMM 不需要为了对齐最大尺寸而填充零，极大节省了算力和带宽。
- **极致的负载均衡**：通过 GPU 内部的“抢单”机制，确保所有流多处理器（SM）都能处于饱和工作状态。

---

## 2. 底层切分与调度机制（核心原理）

GroupGEMM 的高效归功于其 **“逻辑上一维平铺，物理上二维计算”** 的映射逻辑。

### 2.1 几何切分 (Tiling)

CUTLASS 会为整个 Group 设定一个固定的 **Threadblock Tile Size**（例如 $128 \times 128$）。

1. **分块计算**：每个问题 $i$ 根据其 $(M_i, N_i)$ 维度被切分为若干个 Tile。
2. **总量统计**：Host 端提前计算出所有问题产生的 Tile 总数（Total Tiles）。

### 2.2 任务“清单化”与线性映射

为了实现高效寻址，系统会在 Host 端构建一个**前缀和 (Prefix Sum) 数组**：

- **问题 0**: 范围 $[0, 9]$ (共 10 个 Tile)
- **问题 1**: 范围 $[10, 29]$ (共 20 个 Tile)
- ...
    
    该数组被拷贝到 Device 端的全局内存中，作为 Threadblock 的“任务地图”。

### 2.3 “抢单式”动态调度 (Problem Visitor)

这是实现负载均衡的关键步骤：

1. **原子抢号**：每个 Threadblock 在启动后，通过 `atomicAdd` 对全局计数器进行“抢单”，获得一个全局唯一的序号 `idx`。
2. **二分寻址**：Threadblock 拿着 `idx` 在前缀和数组中检索。如果 `idx = 15`，检索结果显示其落在问题 1 的范围内。
3. **坐标转换**：
    
    - 计算在该问题内的偏移：`local_idx = 15 - 10 = 5`。
    - 根据问题 1 的几何尺寸，将 `local_idx` 映射回二维坐标 $(m, n)$。
    - 计算对应的 A、B 矩阵起始指针。
        
4. **循环往复**：计算完当前 Tile 后，Threadblock 不会退出，而是再次“抢号”，直到清单上所有任务完成。

---

## 3. 典型应用场景

- **混合专家模型 (MoE)**：不同专家处理的 Token 数量（$M$ 维度）动态变化，GroupGEMM 是 MoE 算子的事实标准实现方式。
- **LLM 连续批处理 (Continuous Batching)**：在处理不同长度的 Prompt 拼接请求时，利用 GroupGEMM 避免 Padding，提升吞吐量。
- **多模型服务 (Multi-Model Serving)**：在单一显卡上并行跑多个不同规格的推理请求。

---

## 4. 工程落地与性能调优建议

### 4.1 内存与 C++ 规范

- **RAII 封装**：在 C++ 层面推荐使用智能指针或自定义 Buffer 类管理 Device 指针数组，避免在复杂的指针传递过程中发生内存泄漏。
- **异步元数据传输**：`problem_sizes` 等元数据的拷贝应尽量与计算 Stream 异步化，或使用 Pinned Memory 降低延迟。

### 4.2 针对架构的专项优化 (以 Turing/1660 Ti 为例)

- **架构适配**：确保实例化模板时 `ArchTag` 设置为 `cutlass::arch::Sm75`，以充分利用 $m16n8k8$ Tensor Core 指令。
- **对齐要求**：为了保持 MBU（算力利用率），$K$ 维度应尽量保持 8 的倍数（针对 FP16/BF16），以触发最佳的向量化访存。
- **Swizzling 策略**：启用 CUTLASS 的 **Threadblock Swizzling**（如 `IdentitySwizzle`），通过改变物理执行顺序来提升 L2 缓存命中率。

---

## 5. 总结

> **核心思路总结**：
> 
> GroupGEMM 将复杂的 **Host-Side 控制流**（多次 Kernel 启动）转化为了简单的 **Device-Side 数据流**（通过原子操作和指针映射领取任务）。这种设计让 GPU 能够像一个高效的流水线工厂，不再等待 CPU 的指令，而是自主地根据任务清单吞噬所有算力。
