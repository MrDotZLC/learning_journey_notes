## PagedAttention 详细介绍
### 1. 问题背景：LLM 推理的内存瓶颈
LLM 推理的核心数据结构为 KV Cache（Key-Value Cache）。
Transformer 的 Attention 计算依赖于当前 Token 与所有历史 Token 的交互：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
自回归生成阶段，为避免 $O(N^2)$ 的重复计算，需将历史 Token 的 Key 和 Value 张量缓存在 GPU 显存中。
单请求 KV Cache 显存占用公式：
$$\text{Size} = 2 \times L \times H \times d_h \times S \times \text{dtype\_bytes}$$
其中：
- $L$：网络层数 (num_layers)
- $H$：注意力头数 (num_heads)
- $d_h$：每头维度 (head_dim)
- $S$：序列长度 (seq_len)
- $2$：包含 K 和 V 两个张量
以 LLaMA-2 7B 为例（采用 FP16 数据类型，即 2 bytes）：
$$2 \times 32 \times 32 \times 128 \times 4096 \times 2\ \text{bytes} \approx 2\ \text{GB/请求}$$

---

### 2. 原始方案的缺陷与内存碎片
传统推理引擎（如早期的 FasterTransformer）采用静态预分配策略：请求到达时，按 `max_seq_len` 预分配连续的显存块。
**碎片化分类：**
1. **内部碎片 (Internal Fragmentation):** 预分配按最大长度设定，实际生成长度未知。若实际序列短于预留长度，未使用的显存被长期独占。
2. **外部碎片 (External Fragmentation):** 请求生命周期不一，释放的显存块大小不均且不连续，无法被新请求的长序列复用。
3. **缺乏共享机制:** 相同 Prompt 的并发请求无法跨请求复用 KV Cache。
此策略导致显存实际利用率通常仅 20%–40%，成为制约大并发吞吐量 (Throughput) 的核心瓶颈（Memory-bound）。

---

#### 3. Paged Attention 核心思想与内存映射
源自 vLLM 系统的 Paged Attention，核心在于将操作系统的**虚拟内存分页 (Virtual Memory Paging)** 机制引入 KV Cache 显存管理。
##### 3.1 核心概念映射

|**OS 概念**|**Paged Attention 概念**|**物理/逻辑定义**|
|---|---|---|
|Page (页)|Block (块)|最小分配单元，固定大小，存储 $B$ 个 token 的 KV 张量|
|Page Table (页表)|Block Table (块表)|维护逻辑块号 (Logical Block) 到物理块号 (Physical Block) 的映射关系|
|Memory Manager|Block Manager|全局组件，管理 GPU HBM 中物理块的分配、回收与引用计数|
|Virtual Page|Logical Block|请求视角的连续块索引 ($0, 1, 2, \dots$)|
|Physical Frame|Physical Block|GPU 显存中实际划分的固定大小内存块|

##### 3.2 内存布局拓扑
通过 Block Table 实现逻辑连续、物理离散的映射，彻底消除外部碎片。
物理显存 (Physical GPU Memory) 划分为等长物理块：
`[ P0 | P1 | P2 | P3 | P4 | P5 | P6 | P7 ]`
请求的 Block Table 映射关系：
- `Request A`: `[逻辑块 0 -> P2]`, `[逻辑块 1 -> P5]`, `[逻辑块 2 -> P1]`
- `Request B`: `[逻辑块 0 -> P0]`, `[逻辑块 1 -> P3]`

---

#### 4. 执行流程与 CUDA 级实现细节
##### 4.1 阶段状态机
1. **Prefill 阶段:** 按需分配物理块。若 Prompt 长度需跨越 $N$ 个 Block，分配 $N$ 个物理块并构建 Block Table。最后一块按实际 Token 数记录填充水位线。
2. **Decode 阶段:** 查表获取最后一个逻辑块对应的物理块。若存在空余 Slot，直接写入新 Token 的 KV；若满，向 Block Manager 申请新物理块并追加至 Block Table。
##### 4.2 数学推导：分块注意力与 Online Softmax
标准 Attention 需全局数据结构连续。在不连续的 Paged 结构中计算 Attention，必须引入 **Online Softmax**，这也是 FlashAttention 能够分块计算的基础。
定义第 $i$ 个 Block 的输入为 $K_i, V_i$。在遍历 Block Table 时，维护三个全局标量/向量：
1. **局部最大值** $m_i$:
    $$m_i = \max(m_{i-1}, \max(Q K_i^T))$$
2. **局部归一化因子** $l_i$:
    $$l_i = e^{m_{i-1} - m_i} l_{i-1} + \sum e^{Q K_i^T - m_i}$$
3. **局部输出** $O_i$:
    $$O_i = \frac{e^{m_{i-1} - m_i} l_{i-1} O_{i-1} + e^{Q K_i^T - m_i} V_i}{l_i}$$
上述推导证明了可以在非连续显存块上进行局部 Attention 计算，最终合并为严格等价的全局 Attention 结果，且无需分配大型中间张量 (如 $N \times N$ 的 Attention Score 矩阵)。
##### 4.3 C++ / CUDA 侧调度结构
推理引擎 C++ 后端的数据结构与调度逻辑设计：
```C++
// 物理层抽象
struct PhysicalBlock {
    int block_id;
    int ref_count;
    // 显存指针，指向 [2, block_size, num_heads, head_dim] 的半精度张量
    half* kv_data_ptr; 
};
// 逻辑层映射
struct BlockTable {
    std::vector<int> logical_to_physical;
    int current_fill_offset; // 记录最后一个 block 的写入位置
};
// CUDA Kernel 伪接口：Paged Attention 计算
__global__ void paged_attention_kernel(
    const half* Q,                   // [num_heads, head_dim]
    const int* block_table,          // 当前 sequence 的映射表
    const int num_logical_blocks,
    const half* kv_cache_pool,       // 全局 KV 显存池
    float* out                       // 输出张量
) {
    // 1. 获取当前 thread/block 负责的 head 和 logical_block_idx
    // 2. 根据 block_table[logical_block_idx] 定位到物理块内存指针
    // 3. 执行上述推导的 Online Softmax (计算 QK^T, 维护 m_i, l_i)
    // 4. 计算 softmax * V
    // 5. 将结果写回 out
}
```

---

#### 5. Copy-on-Write 机制与显存复用
针对 Parallel Sampling、Beam Search 及 System Prompt 复用场景，Paged Attention 引入引用计数 (Reference Counting) 与写时复制 (Copy-on-Write, CoW)。
初始状态下，多个请求共享同一段 Prompt 的物理块：
- `P0` 的引用计数 `ref_count[P0] = 2`
- 请求 A 与请求 B 的 Logical Block 0 均指向物理块 `P0`
当请求 A 进入 Decode 阶段需向 `P0` 追加新 Token 发现其剩余空间，但检测到 `ref_count[P0] > 1` 时触发 CoW：
1. 分配新的物理块 `P0'`。
2. 将 `P0` 的已有数据 `cudaMemcpy` (Device to Device) 至 `P0'`。
3. 将请求 A 的 Block Table 映射更新为 `[逻辑块 0 -> P0']`。
4. 执行 `ref_count[P0] -= 1`，`ref_count[P0'] = 1`。
5. 在 `P0'` 中写入新 Token 的 KV 数据。
此机制使等长 Prompt 占用的显存空间从 $O(N)$ 优化至 $O(1)$。

---

#### 6. 内存碎片定量分析

|**碎片类型**|**传统静态分配**|**Paged Attention**|
|---|---|---|
|**内部碎片**|严重，取决于 `max_seq_len` - `actual_seq_len`|极小，仅发生在序列末尾的最后一个 Block|
|**外部碎片**|高度存在，小块无法被大请求利用|彻底消除（所有 Block 定长，可离散分配）|
|**复用支持**|隔离不支持|原生支持 CoW|

**内部碎片显存浪费上界：**
$$\text{Worst Case Waste} = (B - 1) \times \text{单 Token KV Size}$$
在典型配置中，Block Size $B$ 取 16 或 32，相对于动辄数千的 Context Length，碎片率从传统方案的 60%-80% 降至 4% 以下。

---

#### 7. 显存超载处理：抢占与交换机制 (Preemption & Swapping)
当系统面临突发高并发请求，`BlockManager` 中的全局空闲物理块 (Free Physical Blocks) 耗尽时，推理引擎必须执行 OOM (Out of Memory) 应对策略，即抢占 (Preemption)。
核心策略分为两种：
1. **Swapping (换出)**
    - **机制**：将低优先级请求的 GPU 物理块通过 PCIe 总线换出 (Swap out) 至 CPU 宿主机内存 (Host RAM)。当 GPU 显存恢复空闲时，再将其换入 (Swap in)。
    - **系统底层逻辑**：依赖 CUDA 的 `cudaMemcpyAsync` 与 Pinned Memory (锁页内存) 实现计算与传输重叠。
    - **代价**：PCIe 4.0/5.0 带宽（通常为 32GB/s - 64GB/s）远低于 GPU HBM 带宽（如 H100 的 3TB/s），导致较高延迟。
2. **Recomputation (重计算)**
    - **机制**：直接释放 (Free) 被抢占请求的 KV Cache 物理块引用。当请求重新获得调度时，将其作为全新的请求，重新执行 Prefill 阶段以恢复 KV Cache。
    - **代价**：消耗额外的浮点算力 (FLOPs)，但节省了 CPU 内存和 PCIe 带宽。
**工程实现选型**：在长上下文 (Long Context) 场景下，重计算的算力开销呈二次方增长，Swapping 更具优势；而在短序列场景中，GPU 计算极快，Recomputation 策略的整体系统吞吐量更高。

---

#### 8. 架构解耦：PD 分离 (Prefill-Decode Disaggregation)
2025 年后的主流推理架构已从单一实例混合调度演进为 **Prefill-Decode (PD) Disaggregation (预填充与解码解耦)**。
##### 8.1 物理瓶颈分析
- **Prefill 阶段**：Compute-bound (算力瓶颈)。大量输入 Token 并行计算 Attention，容易打满 GPU 的 Tensor Core，但对显存容量和访存带宽的要求相对稳定。
- **Decode 阶段**：Memory-bound (访存瓶颈)。逐个 Token 生成，计算量极小，但每次生成都需要将整个历史 KV Cache 从 HBM 搬运到 SRAM，极度消耗显存带宽。
##### 8.2 解耦架构设计
将系统的计算图从单 GPU 或单集群实例拆解：
1. **Prefill Cluster**：专用 GPU 节点仅处理 Prompt，批处理大小 (Batch Size) 设置极高，最大化利用计算密集型算力。
2. **KV Transfer Engine**：底层依赖高性能通信库。通过 RDMA (如 RoCEv2/InfiniBand) 网络与 UCX/NCCL 协议，将 Prefill 节点生成的 KV Cache 张量以零拷贝 (Zero-copy) 方式跨节点直传至 Decode 节点的显存池。现代开源组件如 Mooncake Transfer Engine 或 NIXL 已成为此环节的标准中间件。
3. **Decode Cluster**：专用 GPU 节点接收 KV Cache，执行自回归生成。
**底层进阶方向**：该方案将传统的单体 `CacheEngine` 演变为分布式的生产者-消费者模型。推理优化开发需深度关注底层网络套接字 (Sockets)、共享内存 (Shared Memory) 与 GPU-Direct RDMA 技术，将 Linux 系统级网络编程与 GPU 显存管理深度融合。

---

#### 9. 缓存结构化复用：Radix Tree 与 APC
为应对 System Prompt 冗长或 Multi-turn Conversation (多轮对话) 中重复的历史上下文，KV Cache 的跨请求复用成为核心优化点。
##### 9.1 自动前缀缓存 (Automatic Prefix Caching, APC)
以早期 vLLM 为代表的方案基于哈希表 (Hash Table)。
- **计算方式**：以 Block 为粒度，计算当前 Block 内 Tokens 及所有前置 Tokens 的哈希值。
- **复用逻辑**：若多个并发请求的 Block Hash 相同，则 Block Table 均指向同一物理块，并增加该物理块的引用计数 (`ref_count`)，后续分支修改通过 CoW (Copy-on-Write) 机制物理隔离。
##### 9.2 RadixAttention (基数树注意力机制)
以 SGLang 及 2025 版 vLLM 为代表的前沿架构，在内存管理器之上构建了 Radix Tree (基数树) 数据结构。
- **数据结构**：将请求的 Token 序列视为路径，物理块映射作为节点。树的边对应 Token 子序列。
- **动态匹配**：新请求到达时，系统在 Radix Tree 中执行最长前缀匹配 (Longest Prefix Match)，命中缓存后仅需对增量 Token `(input_tokens - hit_tokens)` 触发前向计算。
- **驱逐策略**：显存告急时，采用算法优先驱逐 Radix Tree 中引用计数为 $0$ 且访问时间最靠前的叶子节点。
**底层进阶方向**：实现从 `std::unordered_map` 到自定义并发树状结构的底层改造。需在 C++ 层面处理极高频的树节点读写锁 (Read-Write Locks) 与无锁 (Lock-free) 数据结构设计，确保 CPU 调度开销不会反向吞噬 GPU 优化收益。

---

#### 10. 极致压缩：KV Cache 硬件级量化
伴随 1M+ Token 上下文窗口的普及，单纯的 Paged 结构优化已不足以对抗物理显存容量上限，数据类型量化 (Quantization) 成为推理引擎必选项。
##### 10.1 FP8 量化 (E4M3 / E5M2)
- **格式定义**：FP8 (8-bit Floating Point) 包含 E4M3 (精度更高，动态范围较小) 与 E5M2 (动态范围大，精度较低)。
- **显存收益**：将标准 FP16/BF16 KV Cache 的显存占用硬性减半（例如 LLaMA-2 7B 每请求降至 1GB 左右）。
- **硬件协同**：在 NVIDIA Hopper 架构上，配合 FlashAttention 3 及 TMA (Tensor Memory Accelerator)，CUDA Kernel 可直接以 FP8 格式加载 KV 数据，并调用 Tensor Core 执行 WGMMA 矩阵乘法指令。彻底消除了反量化为 FP16 的内核开销，同步提升计算与访存效率。
##### 10.2 NVFP4 亚比特演进 (2025/2026 前沿)
- 伴随 NVIDIA Blackwell 架构的部署，NVFP4 (4-bit 浮点数) 将 KV Cache 的内存占用在 FP8 的基础上再降 50%。
- **校准 (Calibration)**：需在推理侧引入复杂的逐通道 (Per-Channel) 或逐 Token (Per-Token) 动态缩放因子 (Scaling Factor) 内存管理机制，以补偿极端低位宽导致的精度损失。
**底层进阶方向**：深入剖析 PTX 汇编与 Cutlass 模板库。需要掌握如何编写自定义 CUDA Kernel，精准控制线程块级别 (Thread Block) 的内存对齐 (Memory Alignment)、向量化访存 (Vectorized Access，如使用 `float4` 类型) 以及软硬件协同的量化算术逻辑。

---

#### 11. 块级选择与主动驱逐策略 (Block-Level Eviction)
面对极端长度的 Prompt，系统无法仅依赖被动的全局 OOM Swap，而应在 Paged Attention 的 Block 维度进行主动的数据流过滤。
- **注意力稀疏预言机 (H2O / SnapKV 变体)**：LLM 内部的 Attention 权重呈现极强稀疏性，少数 Token (Heavy Hitters) 汇聚了超 90% 的注意力。
- **动态剪枝实现**：在 Paged Attention 前向传播的循环中，实时统计每个 Physical Block 的累计 Attention Score 均值。算法主动识别并释放 (`free()`) 低阈值物理块（通常为停止词、标点等低价值信息），永久驻留前置指令块 (Sink Tokens) 与关键上下文。此机制完全建立在离散的 Block Table 寻址之上，利用 Paged Attention 的物理不连续特性实现了 $O(1)$ 的显存碎片回收。
