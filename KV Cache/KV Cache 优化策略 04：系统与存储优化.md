## 1. 背景：内存管理的核心挑战

KV Cache 的内存管理面临三类问题：

**外部碎片（External Fragmentation）**：不同请求序列长度不同，若按最大长度预分配连续内存块，短请求浪费大量空间；传统系统浪费 $60\%\text{--}80\%$ 的 KV Cache 内存。

**内部碎片（Internal Fragmentation）**：自回归生成不知道最终序列长度，预留超出实际的空间无法被他人使用。

**请求间冗余（Cross-Request Redundancy）**：多个请求共用相同系统 Prompt 时，各自独立存储重复的 KV Cache，浪费显存。

---

## 2. PagedAttention（vLLM，SOSP 2023）

### 2.1 设计原理

借鉴操作系统的**虚拟内存与分页**思想（OS Virtual Memory & Paging）：

- **逻辑块（Logical Block）**：从用户视角，KV Cache 是连续的 token 序列；
- **物理块（Physical Block / Page）**：实际分配的固定大小内存单元，默认 $B = 16$ tokens/page；
- **块表（Block Table）**：维护每个请求的逻辑页→物理页映射，类比 OS 页表。

物理块**按需分配**，不预留连续空间；不同请求的物理块无需邻接。

### 2.2 物理内存空间计算

设 GPU 显存 $M$（bytes），模型权重占用 $M_{\text{model}}$，每个 token 的 KV Cache 占用 $m_{\text{token}} = 2 \cdot L \cdot n_{kv} \cdot d_h \cdot b$，每页 $B$ tokens：

$$ N_{\text{pages}} = \left\lfloor \frac{M - M_{\text{model}} - M_{\text{overhead}}}{B \cdot m_{\text{token}}} \right\rfloor $$

vLLM 在启动时以 dummy 前向传播测量 $M_{\text{overhead}}$，动态确定 $N_{\text{pages}}$。

### 2.3 写时复制（Copy-on-Write, CoW）

当多个请求共享同一物理页（前缀共享）且其中一个请求需要写入时，触发 CoW：复制该物理页，写入者获得新副本，其他共享者保持原页不变。

支持 Beam Search 的多路径分支：

```
Request
├── Beam 0: [page_0, page_1, page_2_beam0]   ← CoW
├── Beam 1: [page_0, page_1, page_2_beam1]   ← CoW（共享 page_0, page_1）
└── Beam 2: [page_0, page_1, page_2_beam2]   ← CoW
```

### 2.4 实测

对比 FasterTransformer 和 Orca：同等延迟下吞吐量提升 $2\text{--}4\times$；对 Beam Search 等复杂解码场景提升更显著；GPU 内存浪费率从 $60\%\text{--}80\%$ 降至 $< 4\%$。

> 【图示占位】：左图：传统连续内存分配，展示 3 个请求各占一大块，中间有大量灰色空洞；右图：PagedAttention 分页管理，展示相同请求的物理页分散排布，空洞极少。

---

## 3. Prefix Caching（前缀缓存）

### 3.1 原理

对于多次请求中重复出现的 Prompt 前缀（如系统 Prompt、Few-shot 示例、RAG 文档），避免每次重新计算 KV，直接复用已缓存的 KV Cache 块。

**命中条件**：请求前缀的 Token ID 序列与缓存条目完全一致（Hash 匹配）。

**vLLM Automatic Prefix Caching（APC）**：

以页为单位计算 token 序列的哈希值，建立哈希→物理页的映射表。新请求到达时，逐页检查哈希匹配，匹配的页直接复用，未匹配的页正常 Prefill。

```python
from vllm import LLM
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_prefix_caching=True,
)
```

**实测**：有一致系统 Prompt 的应用（如 RAG、Few-shot 推理）缓存命中率可达 $87\%+$，TTFT（Time To First Token）大幅降低。

### 3.2 RAGCache / CacheBlend

**RAGCache**（arXiv 2024）：针对 RAG 场景，将检索文档的 KV Cache 与模型权重一同存储在 GPU/CPU 分级存储体系中，实现文档级的 KV 预热（Pre-warming），命中时直接注入 KV 而无需重新 Prefill。

**CacheBlend**（arXiv 2024）：在共享前缀与动态上下文混合场景中，通过重新计算部分 KV（选择注意力影响最大的 token 重算）补偿 Prefix 注意力遗漏的跨序列交互，在缓存复用与精度之间平衡。实测吞吐提升 $3.9\times$。

---

## 4. RadixAttention（SGLang，OSDI 2024）

### 4.1 Radix Tree 管理

vLLM APC 只支持完全一致的前缀匹配，对多轮对话中不断增长的前缀支持有限。SGLang 引入 **Radix Tree**（基数树）管理 KV Cache，支持**任意共享前缀模式**：

```
Root
├── [system_prompt_tokens] → KV_block_1
│   ├── [user_turn_1] → KV_block_2
│   │   ├── [assistant_turn_1 + user_turn_2] → KV_block_3 (Thread A)
│   │   └── [assistant_turn_1 + user_turn_2'] → KV_block_4 (Thread B)
│   └── [user_turn_1'] → KV_block_5
```

树节点按 token 序列哈希标识，LRU（Least Recently Used）驱逐策略管理物理页分配。

### 4.2 Cache-Aware Scheduling

调度器优先将共享前缀更多的请求批量处理，最大化物理页复用率，减少不必要的 Prefill 计算。

**实测**：对比 vLLM，SGLang 在多轮对话与 agent 工作流上吞吐提升 $5\times$（Llama-7B，A10G）；在共享 $1000$-token 系统 Prompt 的场景下，TTFT 降低约 $75\%$。

### 4.3 MambaRadixCache（Hybrid 模型扩展）

对 Mamba/SSM 层，需额外缓存递推状态（Mamba State）而非 KV Tensor。SGLang 实现 `MambaRadixCache`，分离 Mamba State 与 KV Cache 的驱逐策略：

- KV Cache 必须从叶节点向根驱逐（保证前缀完整性）；
- Mamba State 可从任意节点驱逐（因 SSM 状态无结构依赖）。

---

## 5. KV Cache 分级卸载（Offloading）

### 5.1 内存层次结构

```
HBM (GPU VRAM)
    ↕  PCIe (50–100 GB/s)
DRAM (CPU RAM)
    ↕  NVMe SSD (3–7 GB/s seq read)
NVMe / eMMC
```

KV Cache 优先存放于 HBM；当 HBM 不足时，按重要性卸载至 DRAM；极端场景卸载至 NVMe SSD。

### 5.2 FlexGen（ICML 2023）

**设计**：将模型权重与 KV Cache 均卸载至磁盘，逐层按需加载，支持单 GPU 推理超大模型（如 30B+ 模型在单张 A10 上推理）。

**IO 策略**：以层为粒度流水线预取（Prefetch），重叠 I/O 与计算：当第 $l$ 层在 GPU 计算时，预取第 $l+1$ 层的权重和 KV Cache。

**核心限制**：完整加载 KV Cache 至 GPU 执行全量注意力，Decode 带宽严重受限于 NVMe 速度（$\leq 7$ GB/s），吞吐量远低于 GPU 原生推理。

### 5.3 ShadowKV（NeurIPS 2024）

**改进 FlexGen 的关键思路**：不卸载完整 KV，而是只在 GPU 上存储 **SVD 压缩的低秩 Key**（用于注意力分数预测），Value Cache 卸载至 CPU：

1. **Prefill** 阶段：对 Key 做 SVD，GPU 保留低秩表示 $\mathbf{K}_r$（$r \ll d_k$），原始 Value 卸载至 CPU；
2. **Decode** 阶段：用 $\mathbf{K}_r$ 计算粗粒度注意力分数，确定 Top-$k$ 重要 token；
3. 仅将这 $k$ 个 token 的 Value 从 CPU 传回 GPU，执行精确注意力计算。

带宽节省：原本需要传输 $T \cdot d_v$ 的 Value，现只需传输 $k \cdot d_v$（$k \ll T$）。

**实测**：在 A100 80GB 上支持 LLaMA-3-8B 的 $1\text{M}$ token 上下文推理，吞吐比完整卸载高约 $3.04\times$。

### 5.4 KVSwap（arXiv 2025）

**场景**：资源受限的边端设备（如 eMMC/NVMe 存储的嵌入式系统）。

**技术**：在 FlexGen 与 PagedAttention 基础上，维护 GPU 内的紧凑 Key 表示（非完整 KV），以预测各磁盘 KV 页的重要性，只预取重要页：

- 完整 KV 存于磁盘；
- 内存中只维护轻量级 Key 摘要（用于选页）；
- 计算与磁盘预取完全重叠（Fully Overlapped Prefetch）。

**实测（LLaMA3-8B，32K 上下文）**：NVMe 上 KVSwap 吞吐优于所有对比卸载方法，eMMC 上对比 InfiniGen 提升约 $10\times$。

### 5.5 InfiniGen（OSDI 2024）

**异步预取 + 部分权重**：不在 GPU 存储 KV 摘要，而是使用部分 Attention 权重（子集投影）在 CPU 上快速估算下一步关键 token，GPU-CPU 传输与 GPU 计算完全流水线并行。

适用于服务端 CPU 内存充裕（$\geq 100$ GB）的场景。

---

## 6. 分布式 KV Cache 与跨节点共享

### 6.1 DistAttention（arXiv 2024）

**问题**：单 GPU 无法存储超长上下文（$> 256\text{K}$）的完整 KV Cache。

**方案**：将 KV Cache 分布存储于多节点，注意力计算也分布执行（分段计算后合并 softmax）：

$$ \text{Attn}(\mathbf{q},\ \mathbf{K},\ \mathbf{V}) = \text{Merge}!\left(\text{Attn}(\mathbf{q},\ \mathbf{K}_{\text{shard1}},\ \mathbf{V}_{\text{shard1}}),\ \ldots\right) $$

利用 FlashAttention 的分块 softmax 合并公式，无需节点间传输完整 KV，只传输各分片的 $(\text{sum\_exp},\ \text{weighted\_sum})$，带宽远低于传输完整 KV。实测吞吐提升 $3.61\times$。

### 6.2 LMCache（2024）

作为 vLLM 的外部 KV 连接层（KVConnector），将 KV Cache 存储于 CPU 内存（或 S3 对象存储），多 vLLM 实例共享：

```python
from lmcache import LMCacheEngine
cache_engine = LMCacheEngine(
    backend="cpu",
    max_gpu_cache_size="20GB",
    cpu_cache_size="100GB",
)
```

块标识符为 token 序列哈希，无需中央协调节点，多实例直接通过哈希查找共享缓存。GPU-CPU 传输延迟约 $10\text{--}50$ ms/次缓存检索。

### 6.3 KVFlow（arXiv 2025）

针对 **Agent 工作流**（多 agent 协作推理）的 KV Cache 管理：

- 建模 agent 执行计划为有向图（Agent Step Graph）；
- 为每个 agent 节点计算"距下次激活的步数"（steps-to-execution）；
- 基于该值执行细粒度驱逐（步数多的 agent 的 KV 优先驱逐）；
- 提前将下一步激活 agent 的 KV 从 CPU 异步预取至 GPU（完全重叠计算与传输）。

对比 SGLang+HiCache：在 10-agent 顺序工作流上提速 $1.83\times$。

---

## 7. 系统级最佳实践

### 7.1 生产环境配置建议

|场景|推荐配置|
|---|---|
|单 GPU，固定 Prompt 工作负载|vLLM + APC + FP8 KV|
|多轮对话 / Agent 工作流|SGLang + RadixAttention|
|超长上下文（$> 256\text{K}$），多 GPU|DistAttention + Tensor Parallelism|
|GPU 内存不足，CPU 充裕|ShadowKV / InfiniGen|
|边缘设备，磁盘卸载|KVSwap|
|多实例共享 KV，RAG 场景|LMCache + SGLang|

### 7.2 关键 vLLM 参数

```bash
# 启动 vLLM Server 的关键 KV Cache 相关参数

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --kv-cache-dtype fp8 \               # 量化 KV Cache
    --enable-prefix-caching \            # 启用 APC
    --gpu-memory-utilization 0.90 \      # GPU 内存利用率上限
    --max-model-len 131072 \             # 最大上下文长度
    --tensor-parallel-size 4             # 张量并行度（多 GPU）
```

### 7.3 监控指标

```
# Prometheus 指标（vLLM 暴露于 /metrics）
kv_cache_usage_percent      # KV Cache 使用率（>90% 时需扩容或减小 batch）
kv_cache_total_blocks       # 总物理页数
kv_cache_used_blocks        # 已使用物理页数
prefix_cache_hit_rate       # APC 命中率（<50% 说明前缀缓存配置不合理）
```

---

## 8. 存储层次性能参数参考

|存储层|带宽|延迟|典型容量|
|---|---|---|---|
|HBM（H100 80GB）|~3.35 TB/s|~ns|80 GB|
|LPDDR5 DRAM|~100 GB/s|~50 ns|512 GB|
|PCIe 5.0 x16|~64 GB/s|~μs|—|
|NVMe Gen4 SSD|~7 GB/s|~100 μs|4–8 TB|
|NVMe Gen5 SSD|~14 GB/s|~50 μs|4–8 TB|
|eMMC 5.1|~300 MB/s|~ms|64–256 GB|

卸载策略的实际吞吐受**带宽**与**延迟**双重约束：低延迟的 CPU DRAM 适合频繁小块访问，高带宽 NVMe 适合大块顺序传输（FlexGen 按层加载正是利用此特点）。

---

## 9. 文献索引

|方法|论文|会议/期刊|年份|
|---|---|---|---|
|PagedAttention / vLLM|Efficient Memory Management for LLM Serving with PagedAttention|SOSP|2023|
|SGLang / RadixAttention|SGLang: Efficient Execution of Structured Language Model Programs|OSDI|2024|
|FlexGen|High-Throughput Generative Inference with a Single GPU|ICML|2023|
|ShadowKV|ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference|NeurIPS|2024|
|CacheBlend|CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion|EuroSys|2025|
|RAGCache|RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation|arXiv|2024|
|DistAttention|DistAttention: Distributed Memory-Efficient Attention for Long-Context LLMs|arXiv|2024|
|InfiniGen|InfiniGen: Efficient Generative Inference of LLMs with Dynamic KV Cache Management|OSDI|2024|
|KVSwap|KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference|arXiv|2025|
|LMCache|LMCache: A KV Cache Compression System for Large Language Model Serving|arXiv|2024|
|KVFlow|KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows|arXiv|2025|
