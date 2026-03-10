## 1. 背景与问题根源

### 1.1 LLM 推理的计算特征

大语言模型（LLM）的推理分为两个阶段：

- **Prefill 阶段**：一次性并行处理所有输入 token，计算并缓存 KV Cache，属于 **compute-bound**（算力瓶颈）。
- **Decode 阶段**：每次 forward pass 只生成一个新 token，反复迭代直到 EOS（End-of-Sequence），属于 **memory bandwidth-bound**（显存带宽瓶颈）。

这种 **multi-iteration** 特性是所有 batch 调度问题的根源：不同请求在 decode 阶段生成的 token 数量差异极大，导致 batch 内各序列的生命周期（lifetime）不一致。

### 1.2 Static Batching 的根本缺陷

```
设 batch 内有 N 个序列，第 i 个序列需生成 L_i 个 token。
整个 batch 的完成时间由最长序列决定：

T_batch = max(L_1, L_2, ..., L_N)
```

若 $L_1 = 500$，$L_2 = L_3 = \ldots = L_{N} = 50$，则序列 $2 \sim N$ 在完成后必须等待序列 1，期间对应的 GPU slot 持续空转。

```
┌────────────────────────────────────────────────────────┐
│  Seq1: [prefill]█████████████████████████████[decode]  │
│  Seq2: [prefill]██████[decode]░░░░░░░░░░░░░░░░░░░░░░░  │
│  Seq3: [prefill]████[decode]░░░░░░░░░░░░░░░░░░░░░░░░░  │
│  Seq4: [prefill]███[decode]░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│                                             ↑ 空转浪费  │
└────────────────────────────────────────────────────────┘
```

`[图示 1：Static Batching 中的 GPU 空转。灰色区域表示已完成但被迫等待的 slot，直到最长序列结束后 batch 才整体释放。]`

---

---

## 2. Dynamic Batching 的三个发展层次

术语在工程实践中存在混用，需按层次严格区分：

|层次|名称|调度粒度|核心机制|
|:-:|:--|:--|:--|
|L1|Static Batching|请求（request）|等凑满固定 batch size 再发射|
|L2|Dynamic Batching|请求（request）|时间窗口 + size 双阈值触发|
|L3|Continuous Batching|迭代步（iteration）|每步重新调度，即时插入新请求|

Dynamic Batching 不强制要求固定 batch size，而是设定一个时间窗口（time window），在窗口内积累到达的请求；若 batch 提前达到 size 上限则立即发射。这类似于公交车"按时发车或满员即发"的策略，平衡了吞吐与尾延迟。

---

---

## 3. Dynamic Batching（L2）详解

### 3.1 触发条件与形式化描述

设请求到达时间为 $t_i$，调度器维护一个队列 $\mathcal{Q}$。触发条件为以下任意一个：

$$ \text{发射条件} = \left( |\mathcal{Q}| \geq B_{\max} \right) \lor \left( t_{\text{now}} - t_{\text{oldest}} \geq \Delta T \right) $$

其中：

- $B_{\max}$：batch size 上限
- $\Delta T$：最大等待时间窗口
- $t_{\text{oldest}}$：队列中最早到达请求的时间戳

### 3.2 Padding 问题

由于 batch 内序列长度不一，必须将短序列 padding 到最长序列的长度 $L_{\max}$，才能构成矩形 tensor 送入 GPU。

设 batch 内序列长度为 ${L_1, L_2, \ldots, L_{B}}$，实际计算量与有效计算量之比（Padding Ratio）为：

$$ \eta_{\text{pad}} = \frac{B \cdot L_{\max}}{\sum_{i=1}^{B} L_i} $$

$\eta_{\text{pad}} > 1$ 时表示存在无效计算开销。当序列长度分布方差大时，$\eta_{\text{pad}}$ 可达 $2 \sim 4 \times$，严重浪费算力。

### 3.3 仍存在的根本缺陷

Dynamic Batching（L2）只解决了**等待策略**问题，并未解决 **batch 生命周期不一致**的问题：一旦 batch 发射，仍需等待最长序列完成才能释放整个 batch，回到了 Static Batching 的困境。

---

---

## 4. Continuous Batching / In-flight Batching（L3）

### 4.1 学术起源：ORCA（OSDI '22）

ORCA 提出了 **iteration-level scheduling**：调度粒度从"请求"降至"单次 forward pass iteration"，调度器在每次迭代后重新决定下一次迭代的 batch 组成。

在此基础上，ORCA 提出 **selective batching**：对 Attention 以外的算子（Linear、LayerNorm、GeLU 等）统一做 token-wise batching，Attention 操作因序列间相互独立而分开执行。

ORCA 在 GPT-3 175B 上相比 FasterTransformer 实现了 **36.9× 的吞吐提升**，同等延迟水平下测得。

### 4.2 核心机制

Continuous Batching 的调度逻辑可用以下伪代码描述：

```cpp
// 每个 decode iteration 的调度器逻辑（简化）
void scheduler_step(RequestPool& pool, ActiveBatch& batch) {
    // Step 1: 移除已完成的序列（生成了 EOS token）
    for (auto it = batch.begin(); it != batch.end(); ) {
        if (it->is_finished()) {
            release_kv_cache(it->seq_id);   // 立即释放显存
            it = batch.erase(it);
        } else {
            ++it;
        }
    }

    // Step 2: 从等待队列补充新请求（只要显存允许）
    while (!pool.empty() && batch.size() < max_batch_size) {
        auto req = pool.pop_front();
        if (can_allocate_kv_cache(req)) {
            batch.push_back(req);  // 即时插入
        } else {
            pool.push_front(req);  // 显存不足，放回队列
            break;
        }
    }

    // Step 3: 执行一次 forward pass（仅 decode step）
    engine.forward_one_iteration(batch);
}
```

### 4.3 Ragged Batching（非规则 Batch）

Continuous Batching 引入了一个新问题：batch 内的序列处于不同的 decode 步骤，序列长度各异，无法再构成规则矩形 tensor。

解决方案是 **Ragged Batching（也称 Packing / Flattening）**：将 batch 内所有 token 展平为一维序列，使用 `cu_seqlens`（累积序列长度）数组标记各序列边界，在 Attention 算子内部还原正确的序列归属关系。

设 batch 内有 $B$ 条序列，第 $i$ 条序列在当前迭代中的有效 token 数为 $n_i$（decode 阶段通常 $n_i = 1$），则展平后的总 token 数为：

$$ N_{\text{total}} = \sum_{i=1}^{B} n_i $$

`cu_seqlens` 数组定义为：

$$ \text{cu_seqlens}[0] = 0, \quad \text{cu_seqlens}[i] = \sum_{j=1}^{i} n_j $$

FlashAttention-2 原生支持此格式（`varlen` 接口），直接以 `cu_seqlens` 作为输入参数，避免 padding 浪费。

`[图示 2：Ragged Batching 的 token 展平示意。左侧为 batch 内 4 条不等长序列，右侧为展平后的一维 token 流，cu_seqlens 标记每条序列的起始偏移。]`

---

---

## 5. Chunked Prefill

Continuous Batching 的一个衍生问题是 **Prefill Stall**：当一条新请求的 prefill 阶段 token 数很多时（如 2048 tokens），该请求独占整个 iteration 做 prefill，导致正在 decode 的其他请求被阻塞，Time-to-First-Token（TTFT）劣化。

Chunked Prefill 将长 prefill 拆分为多个固定大小的 chunk，每个 iteration 只处理一个 chunk，利用 KV Cache 存储中间状态跨 iteration 延续，从而让 prefill 和 decode 在时间上交错执行。

设 prefill 总长度为 $n$，chunk 大小为 $m$，需要的 chunk 数为：

$$ K = \left\lceil \frac{n}{m} \right\rceil $$

第 $k$ 个 chunk（$k = 1, \ldots, K$）处理 token 范围 $[(k-1)m,\ \min(km, n))$。第 $k$ 次 forward pass 时，将前 $k-1$ 个 chunk 的 KV Cache 从显存读取并 concat，送入当前 chunk 的 Attention 计算。

---

---

## 6. 显存约束与 KV Cache 调度

### 6.1 显存是 batch size 的根本瓶颈

Continuous Batching 的 batch size 上限不再由计算量决定，而由 **KV Cache 显存占用**决定。

单条序列的 KV Cache 大小（以 FP16 为例）：

$$ M_{\text{kv}} = 2 \times L_{\text{seq}} \times N_{\text{layers}} \times N_{\text{heads}} \times d_{\text{head}} \times 2\ \text{bytes} $$

其中：

- 系数 $2$：K 和 V 各一份
- $L_{\text{seq}}$：当前序列已生成的 token 数（动态增长）
- $N_{\text{layers}}$：Transformer 层数
- $N_{\text{heads}}$：注意力头数
- $d_{\text{head}}$：每头维度
- $2\ \text{bytes}$：FP16 精度

整个 batch 的 KV Cache 总量：

$$ M_{\text{kv,total}} = \sum_{i=1}^{B} M_{\text{kv}}^{(i)} $$

必须满足 $M_{\text{kv,total}} \leq M_{\text{GPU}} - M_{\text{weights}} - M_{\text{activations}}$。

### 6.2 PagedAttention（vLLM）

传统方案为每条序列预分配 $L_{\max}$ 大小的 KV Cache 连续显存，内部碎片严重。vLLM 的 **PagedAttention** 借鉴 OS 虚拟内存思想，将 KV Cache 划分为固定大小的 Block（Page），按需分配，彻底消除内部碎片。

`[图示 3：PagedAttention 的 Block 分配示意。每条序列的 KV Cache 由若干非连续 Block 组成，Block Table 维护逻辑 Block 到物理 Block 的映射，类比操作系统页表。]`

---

---

## 7. Memory-aware Dynamic Batching（2025 前沿）

arXiv 2503.05248（2025）提出一种实时动态 batch size 调整方法：持续监控 GPU 显存利用率，同时纳入 SLA 延迟约束，将 batch size 配置从静态超参转变为实时控制问题。

核心优化目标可形式化为：

$$ \max_{B(t)}\ \text{Throughput}(B(t)) \quad \text{s.t.} \quad M_{\text{kv,total}}(B(t)) \leq M_{\text{avail}}(t),\ \text{Latency}(B(t)) \leq \text{SLA} $$

其中 $B(t)$ 是 $t$ 时刻的 batch size，$M_{\text{avail}}(t)$ 是当前可用显存。

实验结果显示，该方法相比传统静态 batch 方法获得 8%~28% 的吞吐提升以及 22% 的容量改善。

---

---

## 8. 工程对照：主流框架实现

|框架|Continuous Batching 实现名称|备注|
|:--|:--|:--|
|vLLM|Continuous Batching + PagedAttention|FCFS，iteration-level，非混合 prefill/decode batch|
|TensorRT-LLM|In-flight Batching|NVIDIA 官方实现，支持多精度|
|HuggingFace TGI|Continuous Batching|基于 ORCA 思路|
|LMDeploy|Persistent Batching|针对 turbomind 引擎优化|
|SGLang|RadixAttention + Continuous Batching|KV Cache 前缀共享优化|

vLLM、SGLang、TensorRT-LLM（in-flight batching）、LMDeploy（persistent batching）以及 HuggingFace TGI 均支持 Continuous Batching 或等效机制。

---

---

## 9. 关键指标与调优参数

```
┌─────────────────────────────────────────────────────────────────┐
│                    Batch 调度的核心 Trade-off                    │
│                                                                 │
│   Throughput ←──── batch size ────→ Latency (TTFT / TPOT)      │
│                         ↕                                       │
│                    KV Cache Memory                              │
└─────────────────────────────────────────────────────────────────┘
```

主要调优参数（以 vLLM 为例）：

|参数|含义|典型值|
|:--|:--|:--|
|`max_num_seqs`|同时活跃的最大序列数|64~256|
|`max_num_batched_tokens`|单次 iteration 最大 token 数|2048~8192|
|`gpu_memory_utilization`|KV Cache 可用显存比例|0.85~0.95|
|`max_model_len`|最大序列长度（影响 Block 数量）|模型配置|
|`chunked_prefill_size`|Chunked Prefill 每块大小|512~2048|

---

---

## 10. 总结：演进路径一览

```
Static Batching
    ↓ 问题：等待凑满 batch，尾延迟高
Dynamic Batching（时间窗口 + size 双阈值）
    ↓ 问题：batch 内最长序列拖慢整体，GPU 空转
Continuous Batching（Iteration-level Scheduling）
    ↓ 解决：每步重调度，即时插入/移除序列
    + Ragged Batching：消除 padding 浪费
    + Chunked Prefill：消除 Prefill Stall
    + PagedAttention：消除 KV Cache 显存碎片
    ↓ 前沿（2025）
Memory-aware + SLA-constrained Dynamic Batch Size
    → 将 batch size 作为实时控制变量，最大化受约束吞吐
```
