## 1. 核心定义与背景

**Pipeline Parallelism (PP)** 是模型并行（Model Parallelism）的一个子类，沿模型**深度维度**（Vertical Partitioning）将 $L$ 层切分为 $N$ 个 **Stage**，每个 Stage 驻留于独立加速器（GPU）上。

### 1.1 并行策略体系定位

| 并行维度 | 技术名称                      | 切分对象     | 通信类型                    | 典型库              |
| ---- | ------------------------- | -------- | ----------------------- | ---------------- |
| 数据维度 | Data Parallelism (DP)     | Batch    | All-Reduce              | DDP, FSDP        |
| 模型宽度 | Tensor Parallelism (TP)   | 权重矩阵     | All-Reduce / All-Gather | Megatron-LM      |
| 模型深度 | Pipeline Parallelism (PP) | 层（Layer） | P2P Send/Recv           | GPipe, PipeDream |
| 专家路由 | Expert Parallelism (EP)   | MoE 专家   | All-to-All              | DeepSpeed-MoE    |

PP 的核心挑战是**气泡（Bubble）**：流水线填充和排空阶段，部分 GPU 不可避免地处于空闲状态。所有调度策略的本质，均是在**气泡率**、**显存占用**与**通信开销**三者之间寻求帕累托最优。

### 1.2 关键术语定义

|符号|含义|
|---|---|
|$N$|流水线 Stage 数量（= GPU 数量，假设一卡一 Stage）|
|$M$|Micro-batch 数量|
|$v$|Virtual Stage 数量（Interleaved 场景）|
|$t_f$|单个 Micro-batch 在单个 Stage 的前向计算时间|
|$t_b$|单个 Micro-batch 在单个 Stage 的后向计算时间（含 Activation Gradient）|
|$t_w$|单个 Micro-batch 在单个 Stage 的权重梯度计算时间|
|$t_{\text{comm}}$|相邻 Stage 间的 P2P 通信时间（假设计算可与通信重叠时为 0）|
|$\eta$|气泡率（Bubble Fraction）|

**注**：标准假设 $t_b \approx 2 t_f$（反向传播需要计算两条梯度路径），且 $t_b = t_w + t_{\text{act\_grad}}$，ZBPP 正是利用 $t_w$ 的可推迟性设计调度。

---

## 2. 调度策略深度解析

### 2.1 朴素模型并行（Naive Model Parallelism）

GPipe 诞生前的原始方案。模型层顺序分配到多卡，一次处理完整 Batch，设备串行执行。

**执行时序**（$N=4$，$M=1$）：

```
GPU 0: [F][ ][ ][ ][B][ ][ ][ ]
GPU 1: [ ][F][ ][ ][ ][B][ ][ ]
GPU 2: [ ][ ][F][ ][ ][ ][B][ ]
GPU 3: [ ][ ][ ][F][ ][ ][ ][B]
       |<--- 气泡 --->|<--- 气泡 --->|
```

**气泡率推导**：

设总执行时间中，GPU $i$ 的有效计算时间为 $t_f + t_b$，等待时间为 $(N-1-i)(t_f) + i(t_b)$（前向等待上游，后向等待下游）。

对 Stage 0 而言，气泡占总时间比：

$$\eta_{\text{naive}} = \frac{(N-1)(t_f + t_b)}{N(t_f + t_b)} = \frac{N-1}{N}$$

$N=8$ 时，$\eta = 87.5\%$，**无法用于生产**。

---

### 2.2 GPipe（同步流水线）

2018 年 Google 提出，引入 **Micro-batch** 将 Batch $B$ 划分为 $M$ 份，是现代 PP 的基石。

**执行逻辑（$N=4$，$M=4$）**：

```
        |<--  Fill  -->|<--     Barrier Wait      -->|<--  Drain  -->|
GPU 0:  [F1][F2][F3][F4][  Wait  ][  Wait  ][  Wait  ][B1][B2][B3][B4]
GPU 1:      [F1][F2][F3][F4][ Wait ][ Wait ][B1][B2][B3][B4]
GPU 2:          [F1][F2][F3][F4][ Wait ][B1][B2][B3][B4]
GPU 3:              [F1][F2][F3][F4][B1][B2][B3][B4]
```

#### 2.2.1 气泡率推导（精确版）

**有效计算时间**（所有 GPU 均需完成 $M$ 次前向 + $M$ 次后向）：

$$T_{\text{useful}} = M(t_f + t_b)$$

**总耗时**（最慢 Stage 视角，通常为 GPU $N-1$）：

前向阶段：等待 $N-1$ 个 micro-batch 通过前面所有 Stage，加上自身 $M$ 次前向：

$$T_{\text{total}} = (M + N - 1)(t_f + t_b)$$

**气泡时间**（单 GPU 视角）：

$$T_{\text{bubble, per GPU}} = T_{\text{total}} - M(t_f + t_b) = (N-1)(t_f + t_b)$$

**气泡率**：

$$\eta_{\text{GPipe}} = \frac{T_{\text{bubble, per GPU}}}{T_{\text{total}}} = \frac{(N-1)(t_f + t_b)}{(M + N-1)(t_f + t_b)} = \frac{N-1}{M+N-1}$$

当 $M \gg N$ 时，$\eta \to 0$。

#### 2.2.2 显存代价分析

GPipe 采用**全前向后才开始后向**的策略，所有 Stage 必须缓存当前流水线中**正在运行**的全部 micro-batch 的中间激活值。

- **Stage $i$ 的激活值缓存量**：$M$ 个 micro-batch 的该 Stage 激活（因为后向阶段来临时，前向已全部完成）。
- **系统总激活显存**：$O(N \cdot M \cdot A)$，其中 $A$ 为单个 micro-batch 在单个 Stage 的激活大小。
- **单 Stage 显存**：$O(M \cdot A)$，与流水线深度 $N$ 解耦，仅受 $M$ 控制。

GPipe 还提出 **Gradient Checkpointing（重计算）** 将显存降为 $O(\sqrt{M \cdot A})$，代价是额外约 $33\%$ 的计算开销。

---

### 2.3 1F1B（One-Forward-One-Backward）

PipeDream（2019，Microsoft）提出，Megatron-LM 完善为同步版本，解决 GPipe 显存瓶颈。

#### 2.3.1 三阶段执行逻辑

**热身期（Warm-up Phase）**：Stage $i$ 预先执行 $N-1-i$ 个前向，为后续稳态提供流水线中的"库存"。

**稳态期（Steady State）**：严格执行"1次前向 → 1次后向"循环（称为 1F1B 调度单元）。

**收尾期（Cool-down Phase）**：处理热身期积累的未完成后向。

```
        |<-- Warmup -->|<------ Steady State ------>|<Cooldown>|
GPU 0:  [F1][F2][F3][F4]------------[B1]----[B2]----[B3]----[B4]
GPU 1:      [F1][F2][F3][F4]----[B1]----[B2]----[B3]----[B4]----
GPU 2:          [F1][F2][F3][B1][F4][B2]----[B3]----[B4]----
GPU 3:              [F1][B1][F2][B2][F3][B3][F4][B4]
```

#### 2.3.2 显存分析

1F1B 稳态时，Stage $i$ 在内存中同时持有的 **in-flight** micro-batch 激活数量为 $N - i$（Stage 0 最多，Stage $N-1$ 最少为 1）。

- **Stage $i$ 峰值激活数**：$N - i$ 个 micro-batch 的激活。
- **系统总激活显存**：$O(N \cdot A)$（与 $M$ 无关），其中 $A$ 为单个 micro-batch 在单个 Stage 的激活大小。

相比 GPipe 的 $O(M \cdot A)$（每 Stage），当 $M > N$ 时，1F1B 显著节省显存。

**气泡率与 GPipe 相同**（采用同步梯度更新时）：

$$\eta_{\text{1F1B}} = \frac{N-1}{M+N-1}$$

1F1B 的核心收益是**显存效率**，而非气泡率。

---

### 2.4 Interleaved 1F1B（Virtual Pipeline）

Megatron-LM v2（2021）提出，每块 GPU 承担 $v$ 个不连续的 **Virtual Stage（Model Chunk）**。

#### 2.4.1 层分配逻辑

假设模型共 $L$ 层，$N$ 块 GPU，每块 GPU 持有 $v$ 个 Virtual Stage：

- **物理层分配**：GPU $k$ 负责第 ${k, k+N, k+2N, \ldots, k+(v-1)N}$ 组共 $v \times (L/(Nv))$ 层。
- **实例**：$L=16$，$N=4$，$v=2$，则 GPU 0 负责层 ${1,2,9,10}$，GPU 1 负责 ${3,4,11,12}$，以此类推。

#### 2.4.2 气泡率推导

引入 Virtual Stage 后，流水线深度从 $N$ 变为 $N \cdot v$（逻辑上），但每个 micro-batch 需要经历 $N \cdot v$ 个 Stage 边界。

**总耗时**：

$$T_{\text{total}} = \left(\frac{M}{v} + N - 1\right) v \cdot (t_f + t_b) = (M + v(N-1))(t_f + t_b)$$

但通信次数增加 $v$ 倍，若通信无法与计算完全重叠，需加入通信开销：

$$T_{\text{total}} = (M + v(N-1))(t_f + t_b) + 2v(N-1)t_{\text{comm}}$$

**气泡率**（忽略通信且 M >> P 时）：

$$\eta_{\text{interleaved}} = \frac{v(N-1)}{M + v(N-1)} \approx \frac{N-1}{M/v}$$

对比 $\eta_{\text{1F1B}} = \frac{N-1}{M}$：气泡率降低 $v$ 倍，但**通信量增加 $v$ 倍**。

#### 2.4.3 实际工程权衡

```
v=1 (标准 1F1B):  气泡率高，通信少
v=2 (Interleaved): 气泡率减半，通信翻倍
v→M:              气泡率趋近0，通信量不可接受
```

Megatron-LM 推荐 $v \leq 4$，具体取决于 `t_comm / (t_f + t_b)` 的比值。

---

### 2.5 Zero Bubble Pipeline（ZB-H1 / ZB-H2）

2024 年 NUS 团队（论文《Zero Bubble Pipeline Parallelism》）提出，华为、NVIDIA 等跟进实现。

#### 2.5.1 核心思想：B/W 计算解耦

传统 1F1B 的"后向"将以下两部分合并：

|计算类型|符号|数学含义|依赖关系|
|---|---|---|---|
|Activation Gradient|$B$|$\frac{\partial \mathcal{L}}{\partial x}$，需向前级传递|依赖上游 $B$ 的到来|
|Weight Gradient|$W$|$\frac{\partial \mathcal{L}}{\partial \theta}$，仅用于本 Stage 权重更新|**不依赖**上游 $B$，可推迟|

**关键洞察**：$W$ 的计算不需要将梯度传递给其他 Stage，因此可以**填入原本是气泡的时间槽**，而不影响流水线的依赖拓扑。

#### 2.5.2 三类计算单元定义

设 $F_i^k$、$B_i^k$、$W_i^k$ 分别表示 micro-batch $i$ 在 Stage $k$ 的前向、激活梯度后向、权重梯度计算。

```
依赖关系图：
F_i^k → F_i^{k+1}   (前向：结果传给下一 Stage)
B_i^{k+1} → B_i^k   (后向激活梯度：从下一 Stage 流回)
F_i^k + B_i^k → W_i^k   (权重梯度：依赖同 Stage 的 F 和 B)
```

**$W$ 唯一的约束**：必须在同 Stage 的 $F_i^k$ 和 $B_i^k$ 均完成后才能执行，且必须在优化器步骤（Optimizer Step）前完成。

#### 2.5.3 ZB-H1 调度：填充热身气泡

ZB-H1（Zero Bubble Schedule H1）在热身期（Warm-up）的气泡槽中插入 $W$ 计算。

**时序示意（$N=4$，$M=8$，仅展示 Stage 0）**：

```
时间轴 →
Stage 0: [F1][F2][F3][F4] [B1][W1][F5][B2][W2][F6][B3][W3][F7][B4][W4][F8][B5][W5][B6][W6][B7][W7][B8][W8]
                           ↑
                      原气泡位置
```

ZB-H1 的气泡**不为零**，但将热身期和收尾期的气泡大幅缩小，利用率提升约 $30\%$~$50\%$。

#### 2.5.4 ZB-H2 调度：达到真零气泡

ZB-H2 通过**重新排布热身期的前向次数**，使得每个时间槽均被 $F$、$B$ 或 $W$ 占满。

**设计约束方程**：

设热身期 Stage $k$ 执行 $p_k$ 个前向，则整个调度需满足：

$$\forall k: \sum_{i} \mathbb{1}[t_{B,i}^k < t_{F,j}^k] \leq \text{Buffer Size}$$

具体而言，ZB-H2 要求调度器求解一个整数规划，使得：

1. 每个时间步至少有一个 $F/B/W$ 任务可执行；
2. Stage 间 P2P 依赖不产生死锁；
3. $W_i^k$ 的完成时刻早于 Optimizer Step。

当 $M$ 足够大时（经验上 $M \geq 2N$），ZB-H2 实现**理论零气泡**：

$$\eta_{\text{ZB-H2}} \to 0 \quad (M \to \infty)$$

**显存代价**：$W$ 的推迟意味着激活值需要保留更长时间，ZB-H2 的峰值激活显存略高于 1F1B，大约增加 $\approx 1/N$ 的激活开销。

#### 2.5.5 调度复杂度对比

|调度策略|气泡率 $\eta$|每 Stage 峰值激活数|通信量|权重更新延迟|
|---|---|---|---|---|
|GPipe|$\frac{N-1}{M+N-1}$|$M \cdot A$|$1\times$|同步|
|1F1B|$\frac{N-1}{M+N-1}$|$(N-i) \cdot A$|$1\times$|同步|
|Interleaved 1F1B|$\frac{v(N-1)}{M+v(N-1)}$|$(N-i) \cdot A$|$v\times$|同步|
|ZB-H1|$\approx \frac{(N-1)^2}{M \cdot N}$|$(N-i+\epsilon) \cdot A$|$1\times$|同步|
|ZB-H2|$\to 0$|$(N-i+\delta) \cdot A$|$1\times$|同步|

注：$\epsilon, \delta$ 为 $W$ 延迟引入的微量额外显存，通常 $< 5\%$。

---

## 3. 数学推导：气泡率统一框架

### 3.1 通用气泡率公式推导

**以 1F1B / GPipe 为例，单 GPU 视角**：

总有效计算时间（单 GPU）：

$$T_{\text{useful}} = M(t_f + t_b)$$

总耗时（单 GPU，从第一个 micro-batch 入 Stage 到最后一个梯度传出）：

$$T_{\text{total}} = (M + N - 1)(t_f + t_b)$$

单 GPU 气泡时间：

$$T_{\text{bubble}} = T_{\text{total}} - T_{\text{useful}} = (N-1)(t_f + t_b)$$

气泡率（单 GPU 视角，等价于全局硬件利用率损失）：

$$\boxed{\eta = \frac{T_{\text{bubble}}}{T_{\text{total}}} = \frac{N-1}{M+N-1}}$$

**若要计算所有 GPU 的气泡总量**（如用于分析总算力浪费）：

$$T_{\text{bubble, total}} = N \cdot T_{\text{bubble}} = N(N-1)(t_f + t_b)$$

此时气泡率定义为总气泡占总计算资源：

$$\eta_{\text{global}} = \frac{N(N-1)(t_f + t_b)}{N \cdot (M+N-1)(t_f + t_b)} = \frac{N-1}{M+N-1}$$

两种视角结果相同，但中间量含义不同，需注意区分。

### 3.2 非对称 $t_f / t_b$ 的影响

当 $t_b \neq 2t_f$ 时（实测值可能因激活函数、混合精度等而偏移），气泡率变为：

$$\eta = \frac{(N-1)(t_f + t_b)}{M t_f + (M+N-1)t_b + (N-1)t_f}$$

化简后等价于（保持 $T_{\text{total}} = (N-1)t_f + (M+N-1)t_b + \ldots$ 视具体调度而定），实践中通常直接 profile 而非理论推导。

---

## 4. 显存管理：Activation 生命周期工程实现

### 4.1 Activation Stashing 机制

前向阶段，每个 Stage 必须将中间激活张量保存至显存缓冲区，直至对应的后向梯度到来后才能释放。

```cpp
// C++17 推理引擎激活值管理（完整版）
#include <unordered_map>
#include <memory>
#include <mutex>
#include <condition_variable>

class ActivationBuffer {
public:
    // 注册前向激活（前向通路调用）
    void stash(uint64_t micro_batch_id, std::shared_ptr<Tensor> act) {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_[micro_batch_id] = std::move(act);
    }

    // 取出激活用于后向计算（后向通路调用，取出即释放所有权）
    std::shared_ptr<Tensor> retrieve(uint64_t micro_batch_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = buffer_.find(micro_batch_id);
        if (it == buffer_.end()) {
            throw std::runtime_error("Activation not found for mb " 
                                     + std::to_string(micro_batch_id));
        }
        auto act = std::move(it->second);
        buffer_.erase(it);  // 立即释放 map 中的持有，显存由 Tensor 析构释放
        return act;
    }

    size_t in_flight_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.size();
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, std::shared_ptr<Tensor>> buffer_;
};
```

### 4.2 Object Pool 优化（消除 cudaMalloc 热路径）

流水线稳态中频繁分配/释放显存是性能杀手。使用预分配的 Object Pool：

```cpp
// 显存池：固定大小 Tensor 桶，用于 Activation 复用
class TensorPool {
public:
    explicit TensorPool(size_t pool_size, size_t tensor_bytes)
        : tensor_bytes_(tensor_bytes) {
        for (size_t i = 0; i < pool_size; ++i) {
            void* ptr = nullptr;
            cudaMalloc(&ptr, tensor_bytes);  // 仅在初始化时分配
            free_list_.push(ptr);
        }
    }

    // 从池中获取（O(1)，无 cudaMalloc）
    void* acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_list_.empty()) {
            throw std::runtime_error("Tensor pool exhausted");
        }
        void* ptr = free_list_.front();
        free_list_.pop();
        return ptr;
    }

    // 归还到池（O(1)，无 cudaFree）
    void release(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        free_list_.push(ptr);
    }

    ~TensorPool() {
        while (!free_list_.empty()) {
            cudaFree(free_list_.front());
            free_list_.pop();
        }
    }

private:
    std::mutex mutex_;
    std::queue<void*> free_list_;
    size_t tensor_bytes_;
};
```

**1F1B 稳态下所需池大小**：Stage $i$ 最多同时持有 $N-i$ 个激活张量，因此池大小应初始化为 $N - i + \epsilon$（$\epsilon$ 为通信 Buffer 预留）。

### 4.3 双流（Dual-Stream）计算通信重叠

```cpp
// 计算流与通信流分离，实现 Compute/Comm Overlap
cudaStream_t compute_stream, comm_stream;
cudaStreamCreate(&compute_stream);
cudaStreamCreate(&comm_stream);

cudaEvent_t fwd_done_event;
cudaEventCreate(&fwd_done_event);

// 前向计算（compute_stream）
launch_forward_kernel(input, output, compute_stream);

// 记录前向完成事件
cudaEventRecord(fwd_done_event, compute_stream);

// 通信流等待前向完成后再发送
cudaStreamWaitEvent(comm_stream, fwd_done_event, 0);
ncclSend(output.data_ptr(), output.numel(), ncclFloat16,
         next_rank, nccl_comm, comm_stream);

// compute_stream 可立即开始处理下一 micro-batch（如果有）
launch_forward_kernel(next_input, next_output, compute_stream);
```

---

## 5. 推理场景下的 PP 特殊行为

> **注意**：上文所有调度分析均以**训练**为背景（含前向+后向+权重更新）。推理（Inference）场景的 PP 行为存在本质差异。

### 5.1 推理 PP 的核心差异

|维度|训练 PP|推理 PP|
|---|---|---|
|计算方向|前向 + 后向|**仅前向**|
|气泡来源|填充/排空延迟|填充/排空 + **自回归串行依赖**|
|显存管理|激活值暂存（训练）|KV Cache 管理|
|Micro-batch 概念|真实 mini-batch|Request batch（多个独立请求）|
|权重更新|有|无|

### 5.2 推理 PP 气泡：自回归的特殊挑战

LLM 推理分两阶段：

- **Prefill（预填充）**：处理完整 Prompt，计算量大，单次通过；
- **Decode（解码）**：自回归逐 Token 生成，每次生成 1 Token，依赖上一步输出。

Decode 阶段的特性导致 PP 气泡率极高：

$$\eta_{\text{decode}} = \frac{N-1}{1 + N - 1} = \frac{N-1}{N}$$

此时 $M=1$（每步只生成1个 Token），等同于朴素模型并行。

**缓解策略**：

1. **Request Batching**：将多个请求的 Decode Token 批量处理，等效于增大 $M$；
2. **Prefill/Decode 分离**（PD 分离）：Prefill 节点与 Decode 节点分开部署，各自针对性优化；
3. **Micro-batch 交叉调度**：在 Decode 等待期间，插入其他请求的 Prefill，实现流水线填充。

### 5.3 KV Cache 在 PP 场景的分布

PP 将模型层分布在多 GPU，**每个 Stage 的 KV Cache 仅存储其负责层的 KV 对**：

$$\text{KV Cache}_{\text{Stage } k} = \sum_{l \in \text{Stage } k} 2 \cdot B \cdot S \cdot H \cdot \text{sizeof}(\text{dtype})$$

其中 $B$ = Batch Size，$S$ = Sequence Length，$H$ = Head Dimension。

Stage 间 KV Cache **不共享**，仅通过 P2P 传递 Activation（Hidden States）。

---

## 6. 各调度策略全量对比

|维度|Naive MP|GPipe|1F1B|Interleaved 1F1B|ZB-H1|ZB-H2|
|---|---|---|---|---|---|---|
|**气泡率**|$\frac{N-1}{N}$|$\frac{N-1}{M+N-1}$|$\frac{N-1}{M+N-1}$|$\frac{v(N-1)}{M+v(N-1)}$|$\approx\frac{(N-1)^2}{MN}$|$\to 0$|
|**峰值激活（单 Stage）**|$O(A)$|$O(MA)$|$O(NA)$|$O(NA)$|$O(NA)$+|$O(NA)$++|
|**通信量（相对）**|$1\times$|$1\times$|$1\times$|$v\times$|$1\times$|$1\times$|
|**实现复杂度**|低|中|中|高|高|极高|
|**适用场景**|无|训练，$M \gg N$|训练，推荐默认|训练，带宽充足|训练 2024+|训练 2024+|
|**参考实现**|—|JAX/Lingvo|Megatron-LM|Megatron-LM v2|veScale|—|

---

## 7. P2P 通信：死锁避免与 NCCL 调度

### 7.1 死锁场景分析

在双向流水线（前向发、后向收）中，若 Stage $k$ 和 Stage $k+1$ 同时阻塞等待对方的消息，将产生死锁：

```
Stage k:   ncclSend(fwd) → 阻塞等待 ncclRecv(bwd)
Stage k+1: ncclSend(bwd) → 阻塞等待 ncclRecv(fwd)
           → 双向阻塞，死锁
```

### 7.2 死锁避免策略

**策略一：奇偶交替（Megatron-LM 实现）**

- 奇数 Stage：先 `Send` 后 `Recv`；
- 偶数 Stage：先 `Recv` 后 `Send`。

```cpp
void pipeline_communicate(int stage_id, bool is_forward,
                           Tensor& send_buf, Tensor& recv_buf,
                           ncclComm_t comm, cudaStream_t stream) {
    int peer = is_forward ? (stage_id + 1) : (stage_id - 1);
    
    if (stage_id % 2 == 0) {
        // 偶数 Stage：先接收，后发送
        ncclRecv(recv_buf.data_ptr(), recv_buf.numel(), 
                 ncclFloat16, peer, comm, stream);
        ncclSend(send_buf.data_ptr(), send_buf.numel(), 
                 ncclFloat16, peer, comm, stream);
    } else {
        // 奇数 Stage：先发送，后接收
        ncclSend(send_buf.data_ptr(), send_buf.numel(), 
                 ncclFloat16, peer, comm, stream);
        ncclRecv(recv_buf.data_ptr(), recv_buf.numel(), 
                 ncclFloat16, peer, comm, stream);
    }
}
```

**策略二：非阻塞 isend/irecv + barrier（适用于更复杂调度）**

```cpp
// 使用 ncclGroupStart/End 批量提交，由 NCCL 内部处理依赖
ncclGroupStart();
if (has_send) ncclSend(send_ptr, count, dtype, dst, comm, stream);
if (has_recv) ncclRecv(recv_ptr, count, dtype, src, comm, stream);
ncclGroupEnd();  // NCCL 内部按最优顺序执行，无死锁
```

---

## 8. 工程实践检查清单

### 8.1 关键性能指标

| 指标            | 测量方法                                | 健康范围          |
| ------------- | ----------------------------------- | ------------- |
| 气泡率 $\eta$    | `(T_total - M*(t_f+t_b)) / T_total` | $< 5\%$（生产环境） |
| P2P 带宽利用率     | `nvlink_bandwidth_util`             | $> 80\%$      |
| 激活显存峰值        | `torch.cuda.memory_stats()`         | $< 80\%$ 显存容量 |
| 稳态 Throughput | `tokens/sec`                        | 与理论峰值之比       |

### 8.2 调参指南

$$M_{\text{recommended}} \geq 2N \quad \text{（保证气泡率} < 33\%\text{）}$$

$$M_{\text{optimal}} \approx 4N \sim 8N \quad \text{（经验值，平衡显存与气泡）}$$

若显存不足无法增大 $M$，考虑：

1. 对 Activation 启用 Gradient Checkpointing（代价：$+33\%$ 计算）；
2. 降低 Micro-batch 的 Sequence Length；
3. 启用 ZB-H1 替换 1F1B 以在不增加显存的前提下降低气泡率。

---

## 9. 扩展阅读与参考文献

|论文/资源|贡献|
|---|---|
|GPipe (Huang et al., 2018)|Micro-batch + 同步 PP 奠基|
|PipeDream (Narayanan et al., 2019)|1F1B 异步 PP + 权重版本管理|
|Megatron-LM (Narayanan et al., 2021)|同步 1F1B + Interleaved 调度|
|Zero Bubble PP (Qi et al., 2023)|B/W 解耦，ZB-H1/H2 调度|
|vAttention / PD 分离 (2024)|推理场景 PP + KV Cache 管理|
