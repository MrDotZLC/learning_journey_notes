## 1. 核心定义：张量并行 (TP) 与 专家并行 (EP)

在 DeepSeek-V3（671B）这类超大规模模型推理中，单一 GPU 无法容纳其 $1.2\text{ TB}$ (BF16) 的参数。因此，必须将模型权重“切碎”并分布到多个节点（如 4 个 H100 节点，共 32 颗 GPU）中。

### 1.1 TP (Tensor Parallelism) - 节点内算子切分

- **核心逻辑**：将单个线性层（Linear Layer）的权重矩阵切分。例如 $Y = XW$，$W$ 被水平或垂直切分为 8 份。
- **物理映射**：**节点内 (Intra-node)**。通常 $TP=8$，对应 H100 节点内的 8 颗 GPU。
- **通信特性**：使用 `ncclAllReduce`。由于每层都要进行多次同步，极度依赖 **NVLink**（$450\text{--}900\text{ GB/s}$）的低延迟高带宽。

### 1.2 EP (Expert Parallelism) - 跨节点专家切分

- **核心逻辑**：针对 MoE（Mixture of Experts）层，将不同的专家（Experts）分配到不同的 GPU 或节点上。
- **物理映射**：**跨节点 (Inter-node)**。DeepSeek-V3 常采用 $EP=32$（每颗 GPU 一个专家组）或 $EP=4$（每个节点一个专家组）。
- **通信特性**：使用 `ncclAllToAll`。数据仅在 MoE 层路由时交换，频率低于 TP，通常运行在 **InfiniBand/RoCE** 网络上。

---

## 2. DeepSeek-V3 的混合并行方案：TP8 + EP32

在 32 GPU 集群中，DeepSeek-V3 实现了层级化的并行嵌套，确保计算与通信的最佳平衡。

### 2.1 显存分配推导

- **非专家层 (Attention/Embedding)**：采用 TP8。全集群维护 4 份副本（每节点一份），每份副本切分为 8 份存入 GPU。
- **专家层 (Routed Experts)**：采用 EP32。全集群仅维护 1 份完整副本，每颗 GPU 仅存放 $1/32$ 的专家参数。
    - **计算公式**：每 GPU 专家显存 $\approx 1.2\text{ TB} \times 80\% (\text{MoE比率}) \div 32 \approx 30\text{--}40\text{ GB}$。

### 2.2 执行流阵型变换

1. **Attention 阶段 (TP8)**：32 颗 GPU 分为 4 组并行。组内 8 颗 GPU 通过 NVLink 强同步计算本地 Batch。
2. **路由分发 (Dispatch)**：Token 经过 Router 计算，通过 `All-to-All` 跨节点发送到目标专家所在的 GPU。
3. **专家计算 (EP32)**：每颗 GPU 独立计算发往其本地专家的 Token。
4. **结果回收 (Combine)**：通过 `All-to-All` 将计算结果传回初始 TP 组。

---

## 3. 高级调度：计算通信掩盖 (Overlap)

为了消除跨节点通信带来的延迟（Latency Bubble），DeepSeek-V3 采用了“信令与数据解耦”策略。

### 3.1 异步信令机制 (Control Plane)

- **索引先行**：GPU 10 算完 Router 索引后，即便数据还没搬运，先发一个轻量级“信令包”给 GPU 31。
- **零信号解除**：若 GPU 10 没有数据发给 GPU 31，发出“零信号”。GPU 31 收到 32 个 Rank 的确认后，立即启动专家 Kernel。

### 3.2 细粒度数据搬运 (Data Plane)

- **流式发送**：使用 `ncclSend/ncclRecv` (P2P) 替代阻塞式的 `ncclAllToAll`。
- **RDMA Write**：利用 GPUDirect RDMA，发送端 GPU 算出部分 Token 后立即写入接收端显存，实现计算与传输的并行。

---

## 4. 架构变体：TP8 + EP4 的层级优势

在某些网络带宽受限的场景下，使用 EP4（节点级专家并行）比 EP32 更稳健。

|**特性**|**TP8 + EP32**|**TP8 + EP4**|
|---|---|---|
|**通信 Rank 数**|32 (全集群 $32 \times 31$ 连接)|4 (节点间 $4 \times 3$ 连接)|
|**专家归属**|单个 GPU 独占 8 个专家|**8 颗 GPU 共有** 64 个专家|
|**节点内协作**|各自算各自的专家|8 颗 GPU 利用 **TP8** 合力算一个专家|
|**核心优势**|最大化模型参数容量|减少长尾延迟，提升小 Batch 吞吐|

---

## 5. 总结：工程实现要点 (C++ 视角)

作为推理引擎工程师，在 Ubuntu/C++ 环境下实现 TP+EP 需关注：

1. **通信域划分**：创建两套 `ncclComm_t`，一套关联节点内 NVLink（8 Rank），一套关联全局网卡（32 Rank）。
2. **异步流编排**：使用 `cudaStream_t` 严格区分计算流、信令流和数据搬运流。
3. **状态机同步**：在 C++ 中维护一个 `Ready_Matrix[32][32]`，通过原子操作（Atomic）跟踪全集群 Token 到港情况，动态触发专家 Kernel 发射。

> **结论**：DeepSeek-V3 的卓越不仅在于算法，更在于其对 **TP (NVLink 延迟敏感)** 与 **EP (网卡吞吐敏感)** 的物理特性进行了精准的软硬协同调度。
