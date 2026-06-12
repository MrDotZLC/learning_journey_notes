## 1. 两阶段的本质差异

LLM 自回归推理由两个在计算特性上根本不同的阶段构成。

### 1.1 Prefill 阶段

输入为用户 prompt，包含 $L_p$ 个 token（$L_p \gg 1$）。所有 token 的 hidden states 在同一次 forward pass 中并行计算，Attention 需要完整的因果矩阵乘法：

$$ \text{Attn}_{\text{prefill}} = \text{softmax}!\left(\frac{Q_{[L_p,, d]} \cdot K_{[L_p,, d]}^\top}{\sqrt{d}}\right) \cdot V_{[L_p,, d]} $$

中间矩阵形状为 $[L_p \times L_p]$，运算量为 $O(L_p^2 \cdot d)$。

**计算瓶颈**：Compute-bound。大量 FLOP 集中在矩阵乘，SM 利用率高，适合 FlashAttention（tile-based tiling，数据驻留 SRAM）。

### 1.2 Decode 阶段

每步仅生成 1 个新 token，Query 退化为单行向量 $q \in \mathbb{R}^{1 \times d}$，需与 KV Cache 中已缓存的 $L_{kv}$ 条历史记录做注意力：

$$ \text{Attn}_{\text{decode}} = \text{softmax}!\left(\frac{q_{[1,, d]} \cdot K_{\text{cache}[L_{kv},, d]}^\top}{\sqrt{d}}\right) \cdot V_{\text{cache}[L_{kv},, d]} $$

本质为一次 **GeMV（向量-矩阵乘法）**，运算量为 $O(L_{kv} \cdot d)$，远低于 Prefill。

**计算瓶颈**：Memory-bandwidth-bound。FLOP 极少，大量时间消耗在从 HBM 加载 KV Cache 数据上。

### 1.3 差异汇总

|维度|Prefill|Decode|
|---|---|---|
|输入长度|$L_p \gg 1$|$L_d = 1$|
|Attention 计算量|$O(L_p^2 \cdot d)$，矩阵乘|$O(L_{kv} \cdot d)$，GeMV|
|硬件瓶颈|Compute-bound|Memory-bandwidth-bound|
|GPU MFU|高（Tensor Core 饱和）|低（访存受限）|
|适配 Kernel|FlashAttention|Flash-Decoding / PagedAttention|

---

## 2. 朴素方案及其问题

### 2.1 朴素方案：完全分离调度

最直观的实现是：所有 Prefill 请求先跑完，再开始 Decode；或 Prefill 和 Decode 请求分批、串行处理。

### 2.2 核心问题：TTFT 与 TPOT 的天然矛盾

定义两个关键延迟指标：

- **TTFT**（Time To First Token）：从请求到达到生成第一个 token 的时延。主要由 Prefill 耗时决定。
- **TPOT**（Time Per Output Token）：每生成一个 token 的平均时延。主要由 Decode 吞吐决定。

朴素分离调度导致两者之间存在直接矛盾：

**场景 A：Prefill 优先**

系统持续接收新请求，优先执行 Prefill。等待队列中的 Decode 请求被持续延后，TPOT 劣化，已在生成中的请求出现"卡顿"。

**场景 B：Decode 优先**

Decode 请求占用 GPU，新到来的 Prefill 请求在队列中堆积，TTFT 恶化，首字时延变高。

**根本原因**：两个阶段对 GPU 资源的竞争无法通过"先后顺序"调节，必须在同一个 batch 内混合执行。

### 2.3 朴素 Continuous Batching 的局限

vLLM 早期的 Continuous Batching 允许将不同请求的 Decode token 合并为一个 batch，解决了吞吐问题，但对 Prefill 的处理是整体执行（不切分）。当一个长 prompt 请求（$L_p = 8192$）进入 batch，其单次 Prefill 会独占 GPU 数十至数百毫秒，同 batch 内的 Decode 请求必须等待，导致 TPOT 抖动。

---

## 3. 混合调度中的计算分离原则

混合调度（将 Prefill token 与 Decode token 放入同一 batch）并非所有算子都需要分离执行。正确做法是：**线性层合并，Attention 分离**。

### 3.1 可以合并的算子

所有 token（无论来自 Prefill 还是 Decode 请求）的以下操作可以沿 batch 维度合并为单次计算：

**条件**：算子对序列位置无感知，token 间无依赖。

$$ \text{Output}_{[N_{\text{total}},, d']} = \text{Input}_{[N_{\text{total}},, d]} \cdot W_{[d,, d']} $$

其中 $N_{\text{total}} = \sum_i L_p^{(i)} + \sum_j 1$，即全部 Prefill token 数与全部 Decode token 数之和。

可合并的算子包括：

- QKV Linear Projection
- Output Projection
- FFN（两次矩阵乘 + 激活函数 SiLU/GeLU）
- RMSNorm / LayerNorm（每 token 独立归一化）
- Residual Add

合并后 GEMM 矩阵更大，Tensor Core 利用率提升，整体效率更高。

### 3.2 必须分开执行的算子：Attention

Attention 是混合 batch 中**唯一必须分离**的计算，原因如下：

**Prefill Attention**：$Q \in \mathbb{R}^{L_p \times d}$，需要因果掩码（下三角 mask），适合 FlashAttention。

**Decode Attention**：$q \in \mathbb{R}^{1 \times d}$，访问独立分页的 KV Cache，适合 Flash-Decoding 或 PagedAttention kernel。

二者无法用同一个 kernel 高效覆盖：

- Prefill 的 $[L_p \times L_p]$ 中间矩阵需要 tile-based 计算以规避 HBM 读写
- Decode 的 GeMV 不产生大中间矩阵，瓶颈完全在 KV Cache 的 HBM 加载
- KV Cache 存储方式对两者是不同的访问模式（连续 vs 分页）

### 3.3 执行时序

两路 Attention kernel 并非并行执行，而是在同一 CUDA stream 上串行 dispatch：

```
stream: [QKV Linear (merged)] → [Prefill Attn kernel] → [Decode Attn kernel] → [Output Proj (merged)] → ...
```

**不并行的原因**：

- Prefill Attention（FlashAttention）本身 compute-bound，已将绝大多数 SM 占满
- 即便向第二个 stream dispatch Decode Attention kernel，无空闲 SM 可用，退化为串行等待
- 并发访问 HBM 可能引发带宽争用，反而劣化 Decode 性能

### 3.4 计算图结构

```
所有 token 拼接 → [N_total, d]
        │
        ▼
  ┌─────────────────────────────┐
  │  合并计算（共享）             │  ← RMSNorm · QKV Linear
  └─────────────────────────────┘
        │
        ├──────────────────────────┐
        ▼                          ▼
  Prefill Attention          Decode Attention
  FlashAttention             Flash-Decoding /
  (causal mask)              PagedAttention
        │                          │
        └──────────┬───────────────┘
                   ▼
        ┌─────────────────────────────┐
        │  合并计算（共享）             │  ← Output Proj · FFN · Residual
        └─────────────────────────────┘
```

---

## 4. 解决方案：Chunked Prefill

### 4.1 核心思想

将长 Prefill 序列切分为固定大小的 chunk（如 $C = 512$ tokens），每次调度时将当前 chunk 的 token 与所有 Decode token 合并入同一 batch：

$$ \text{Batch} = \underbrace{\text{Prefill chunk tokens}}_{\leq C} ;\cup; \underbrace{\text{Decode tokens}}_{\text{all active requests}} $$

### 4.2 调度收益

**TTFT 与 TPOT 解耦**：

- 单次 Prefill chunk 执行时间从"整个 prompt"缩短为"一个 chunk"，Decode 等待时间上限从 $O(L_p)$ 降为 $O(C)$
- Decode token 始终与 Prefill chunk 同 batch 执行，不再被整段 Prefill 阻塞

**GPU 利用率提升**：

- Prefill chunk（compute-bound）与 Decode（memory-bound）在 Linear 层共享 GEMM 计算，矩阵更大，打满 Tensor Core
- 两种计算模式的资源需求在时间上交叠，缓解单纯 Decode batch 过小导致的 MFU 低下

### 4.3 Attention 路径处理

Chunked Prefill 下，Attention 仍需分两路：

- Prefill chunk 的 Attention：FlashAttention，处理 chunk 内部因果关系，并访问该请求已完成 chunk 的 KV Cache
- Decode 的 Attention：PagedAttention / Flash-Decoding，访问各自独立的 KV Cache page 表

两路 kernel 串行执行，顺序为：Prefill Attn → Decode Attn（或实现决定）。

### 4.4 工程实现

SGLang、vLLM（v0.4+）均已实现 Chunked Prefill。关键参数为 `chunked_prefill_size`（即 $C$），典型值为 512 或 1024。

---

## 5. 延伸：P/D 分离部署

若需在硬件层面真正并行 Prefill 与 Decode，需将两阶段部署在**不同物理 GPU 或节点**：

- **Prefill 节点**：专用于执行 prompt 的 Attention 计算，compute-bound，配置大量 Tensor Core
- **Decode 节点**：专用于逐 token 生成，memory-bandwidth-bound，优先配置高 HBM 带宽

KV Cache 通过高速互联（NVLink / RDMA）从 Prefill 节点传输至 Decode 节点。代表工作包括 Mooncake（月之暗面）、Splitwise（Microsoft Research）。

这是单卡 Chunked Prefill 调度策略在系统架构层面的进一步延伸，两者解决的是同一根本矛盾：Prefill 与 Decode 的计算特性不匹配导致的资源竞争。
