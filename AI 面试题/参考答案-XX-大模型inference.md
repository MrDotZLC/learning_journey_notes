## 1. KV Cache

### 1.1 KV Cache 是什么

**背景**：Transformer Decoder 在自回归生成时，每步仅生成一个 token，但 Attention 计算需要对序列中所有历史 token 的 Key/Value 做点积。若每步重新计算，复杂度为 $O(n^2)$，随序列增长代价极高。

**核心思想**：将每一层 Attention 已经计算过的 $K$、$V$ 矩阵缓存在显存中，下一步直接复用，仅对新增 token 计算增量 $K$、$V$ 并追加到缓存末尾。

**计算路径对比**：

|阶段|无 KV Cache|有 KV Cache|
|---|---|---|
|Prefill（第一次）|计算全序列 $K$、$V$|计算全序列 $K$、$V$，同时写入 Cache|
|Decode（第 $t$ 步）|重新计算所有 $t$ 个 token 的 $K$、$V$|只计算新 token 的 $\Delta K$、$\Delta V$，追加入 Cache|

---

### 1.2 KV Cache 大小计算

**符号定义**：

|符号|含义|
|---|---|
|$L$|Transformer 层数|
|$H$|Attention Head 数（MHA 场景）|
|$d_h$|每个 Head 的维度，$d_h = d_{model} / H$|
|$s$|序列长度（含 Prefill + 已生成 token）|
|$B$|Batch size|
|$b$|每元素字节数（FP16 = 2，FP8 = 1）|

**单请求 KV Cache 大小**（MHA）：

$$ \text{KV Cache} = 2 \times L \times H \times d_h \times s \times b \quad \text{(bytes)} $$

因子 2 来自 $K$ 与 $V$ 各一份。

**示例**：LLaMA-3-8B，$L=32$，$H=32$，$d_h=128$，$s=4096$，FP16：

$$ 2 \times 32 \times 32 \times 128 \times 4096 \times 2 = 2 \times 32 \times 32 \times 128 \times 4096 \times 2 $$

$$ = 2 \times 1{,}073{,}741{,}824 \approx 2 \text{ GB} $$

（实际 $2 \times 32 \times 32 \times 128 \times 4096 \times 2 = 2{,}147{,}483{,}648$ bytes $= 2$ GiB）

---

## 2. Attention 机制变体

### 2.1 MHA 与 GQA 的区别

**MHA（Multi-Head Attention）**：

$$ \text{head}_i = \text{Attention}(Q_i, K_i, V_i), \quad i = 1, \ldots, H $$

每个 Query Head 拥有独立的 $K_i$、$V_i$ 投影，共 $H$ 组 KV Head。

**GQA（Grouped Query Attention）**：

将 $H$ 个 Query Head 分为 $G$ 组（$G < H$），每组共享同一对 $K$、$V$ Head。令每组 Query 数量 $N_q = H / G$：

$$ \text{head}_i = \text{Attention}(Q_i, K_{\lfloor i/N_q \rfloor}, V_{\lfloor i/N_q \rfloor}) $$

**MQA（Multi-Query Attention）** 是 GQA 的特例，$G = 1$，所有 Query 共享同一 KV。

|机制|KV Head 数|参数量|KV Cache|
|---|---|---|---|
|MHA|$H$|最大|最大|
|GQA|$G$（$1 < G < H$）|中等|按比例缩减|
|MQA|$1$|最小|最小|

**权衡**：GQA/MQA 牺牲少量精度换取显著的 KV Cache 压缩和 Decode 带宽节省；LLaMA-2 70B、LLaMA-3、Mistral 均采用 GQA。

---

### 2.2 GQA 下 KV Cache 的变化

GQA 下 KV Cache 公式修正为：

$$ \text{KV Cache}_{\text{GQA}} = 2 \times L \times G \times d_h \times s \times b $$

与 MHA 相比，压缩比为：

$$ r = \frac{G}{H} $$

**示例**：LLaMA-3-8B 采用 $H=32$，$G=8$（KV Head 数）：

$$ r = \frac{8}{32} = 25\% $$

即 KV Cache 缩减至 MHA 的 $\mathbf{1/4}$。

---

### 2.3 DeepSeek V3 的注意力机制——MLA

**MLA（Multi-head Latent Attention）** 是 DeepSeek 系列提出的低秩 KV 压缩方案，核心是将 KV 投影分解为低秩 Latent 向量的线性变换。

**投影结构**：

对输入隐状态 $h_t \in \mathbb{R}^{d}$，下压投影（Down-projection）：

$$ c_t^{KV} = W^{DKV} h_t, \quad c_t^{KV} \in \mathbb{R}^{d_c} $$

其中 $d_c \ll d$（$d_c$ 为 Latent 维度，远小于原始 $d_{model}$）。

再经上压投影（Up-projection）还原出 $K$、$V$：

$$ K_t = W^{UK} c_t^{KV}, \quad V_t = W^{UV} c_t^{KV} $$

Query 侧也有类似的低秩压缩：

$$ c_t^Q = W^{DQ} h_t, \quad Q_t = W^{UQ} c_t^Q $$

**Decoupled RoPE**：为保留位置编码能力，对 $K_t$ 补充单独的 RoPE 分量 $K_t^R$，通过拼接方式得到最终 Key：

$$ K_t^{\text{final}} = [K_t;\ K_t^R] $$

---

### 2.4 MLA 的好处与缓存内容

**好处**：

- **KV Cache 压缩**：Cache 中仅存低秩向量 $c_t^{KV} \in \mathbb{R}^{d_c}$ 而非展开后的全量 $K$、$V$，显存占用大幅降低。
- **质量保持**：相比 MQA/GQA 直接减少 Head，低秩分解保留更多表征能力。
- **兼容解码加速**：Decode 时从 $c_t^{KV}$ 重建 $K$、$V$，计算量可与矩阵融合优化。

**MLA 存储的内容**：

|缓存项|维度|说明|
|---|---|---|
|$c_t^{KV}$|$d_c$|KV 低秩 Latent（核心缓存）|
|$K_t^R$|$d_R$|Decoupled RoPE 的 Key 分量|

不缓存展开后的 $K_t$、$V_t$，Decode 时动态上压还原。

**压缩效果**（DeepSeek-V2 数据）：KV Cache 相比 MHA 缩减约 $\mathbf{93.3%}$。

---

## 3. Paged Attention

### 3.1 Paged Attention 原理

**问题背景**：传统 KV Cache 需预分配连续显存块（最大序列长度），导致：

- **内部碎片**：预分配但未使用的空间浪费（平均约 $60%$）。
- **外部碎片**：不同大小请求导致空闲块无法合并复用。

**核心思想**：借鉴操作系统虚拟内存的分页机制。将 KV Cache 切分为固定大小的 **Page（Block）**，通过 Block Table（逻辑页→物理页的映射表）管理，使不同请求的 KV Cache 在物理显存中无需连续存储。

**逻辑结构**：

```
Request A: [Block 0] → [Block 3] → [Block 7]   (非连续物理地址)
Request B: [Block 1] → [Block 5]
Shared Prefix: [Block 2] (可被 A、B 共享)
```

**Copy-on-Write**：Prompt 共享的 Block 只有在需要写入时才复制，天然支持 Prefix Caching（RadixAttention）。

---

### 3.2 Page Size 的选择

**Page Size 定义**：每个 Block 存储的 token 数量（即 Block 中 KV 向量的数量）。

**vLLM 默认值**：`block_size = 16`（token/block）。

**选择因素**：

|因素|偏大 Page Size|偏小 Page Size|
|---|---|---|
|内部碎片|较多（末尾 Block 空置多）|较少|
|Block Table 开销|小（映射项少）|大（映射项多）|
|GPU 访存效率|较好（连续内存读写）|较差（碎片化访存）|
|Prefix 共享粒度|粗（共享命中率低）|细（共享命中率高）|

---

### 3.3 Page Size = 1 vs Page Size = 16/32

**Page Size = 1 的优势**：

- 零内部碎片：每个 token 独占一个 Block，不存在浪费。
- Prefix 共享粒度最细，命中率最高。

**Page Size = 1 的劣势**：

- **Block Table 膨胀**：$s$ 个 token 需 $s$ 条映射项，内存开销 $O(s)$，远高于 $O(s / 16)$。
- **访存不连续**：GPU 读取 KV 时需频繁通过间接寻址跳转，破坏 Cache Line 局部性，带宽利用率下降。
- **内核调度开销**：CUDA Kernel 处理大量小块的 gather/scatter 操作开销上升。
- **FlashAttention 等内核对块对齐有假设**，Page Size = 1 会破坏其 tiling 效率。

**结论**：工程上 Page Size 通常选 16 或 32，是碎片率与访存效率的权衡点。

---

### 3.4 一个 Page 只用了 15 个 Token，剩余空间能否继续使用

**不能**（在当前请求的生命周期内）。

Paged Attention 的 Block 以请求为单位分配，末尾 Block 的剩余槽位保留给该请求后续生成的 token（最多再写入 $\text{block_size} - 15$ 个 token）。

该 Block 不会被其他请求复用，直到当前请求结束并释放所有 Block 后，这些 Block 才回归空闲池。这是 Paged Attention 中唯一不可避免的 **内部碎片来源**，平均浪费约 $(\text{block_size} - 1) / 2$ 个槽位。

---

## 4. PD 分离（Prefill-Decode Disaggregation）

### 4.1 PD 分离概念

**Prefill 阶段**：处理输入 prompt，一次性并行计算所有输入 token 的 KV Cache，生成第一个输出 token。

**Decode 阶段**：基于已有 KV Cache 逐步自回归生成后续 token，每步仅处理 1 个新 token。

**PD 分离**：将两个阶段部署在不同的物理机器（或实例）上：

- **Prefill 实例**（P 节点）：专门处理 Prefill，生成 KV Cache 后通过网络传输给 D 节点。
- **Decode 实例**（D 节点）：专门执行 Decode，接收 KV Cache 后进行自回归生成。

---

### 4.2 Prefill 与 Decode 的瓶颈

|阶段|计算特征|主要瓶颈|
|---|---|---|
|Prefill|大批量矩阵乘（GEMM），Arithmetic Intensity 高|**Compute Bound**（计算密集）|
|Decode|每步 Batch = 1（或极小），需读取全量权重和 KV Cache|**Memory Bandwidth Bound**（带宽密集）|

- Prefill 的 FLOPs 主导，GPU 算力利用率高。
- Decode 的 MBU（Memory Bandwidth Utilization）主导，权重加载成为瓶颈。

---

### 4.3 PD 分离后的优化策略

**P 节点专项优化**：

- **Chunked Prefill**：将长 Prefill 切块，与 Decode 请求混合调度，降低 TTFT 方差。
- **大 Batch 聚合**：多个请求的 Prefill 合并成大 GEMM，提升 GPU 利用率。
- **FlashAttention**：减少 HBM 读写，加速 Prefill 的 Attention 计算。

**D 节点专项优化**：

- **Continuous Batching**：动态插入/移除请求，最大化 Batch 内有效 token 数。
- **投机采样（Speculative Decoding）**：小模型草稿 + 大模型验证，提升 Decode 吞吐。
- **KV Cache 量化**：FP8/INT4 压缩 KV Cache，降低 HBM 带宽压力。
- **权重量化（W4A16/W8A8）**：减少权重读取量。
- **MTP（Multi-Token Prediction）**：单步预测多 token，提升 Decode 效率。

**系统层优化**：

- **KV 迁移带宽优化**：P→D 的 KV Cache 传输使用 RDMA/NVLink，减少传输延迟。
- **分离式调度**：P/D 节点可独立扩缩容，按实际负载比例配置资源。

---

## 5. Chunked Prefill

### 5.1 Chunked Prefill 原理

**问题**：长 Prompt 的 Prefill 会独占 GPU 数百毫秒，期间所有 Decode 请求被阻塞，导致 TTFT（Time to First Token）波动极大。

**核心思想**：将 Prefill 序列切割为固定大小的 Chunk（如 512 token），与 Decode token 一起混合打包成一个 Micro-batch，送入同一次 Forward Pass。

**一次 Forward 的组成**：

$$ \text{Micro-batch} = \underbrace{[\text{Prefill Chunk}_1, \ldots, \text{Prefill Chunk}_k]}_{\text{Prefill tokens}} + \underbrace{[\text{Decode token}_1, \ldots, \text{Decode token}_m]}_{\text{Decode tokens}} $$

---

### 5.2 Chunked Prefill vs 一次性 Prefill

|维度|一次性 Prefill|Chunked Prefill|
|---|---|---|
|TTFT|长 Prompt 下极高|受控，接近 Chunk 粒度|
|TPOT|Decode 被 Prefill 阻塞，较高|Decode 与 Prefill 并行，TPOT 更稳定|
|GPU 利用率|Prefill 期间高，Decode 等待期间低|混合调度，持续较高|
|KV Cache 内存压力|一次性分配全量|按 Chunk 增量分配|
|实现复杂度|简单|需要 Attention Mask 拆分处理|

---

### 5.3 单机能否使用 Chunked Prefill

**可以**。Chunked Prefill 是调度策略，不依赖 PD 分离。vLLM（v0.4+）、SGLang 均在单机模式下支持 Chunked Prefill。

单机场景下，Chunked Prefill 的收益：

- 减少长 Prompt 请求对 Decode 请求的延迟干扰。
- 使 Prefill 和 Decode 的 token 混合调度，提升整体 GPU 利用率（避免 Compute Bound 和 Memory Bound 交替空转）。

---

### 5.4 固定 Chunk Size 的长序列 Prefill 计算过程

设序列长度 $S$，Chunk Size $C$，共 $\lceil S/C \rceil$ 个 Chunk。

**第 $i$ 步处理 Chunk $i$（token $[(i-1)C, iC)$）**：

1. 对 Chunk $i$ 中的 token 计算 $Q$、$K$、$V$。
2. Attention 计算：Chunk $i$ 的 $Q$ 与 $[0, iC)$ 范围内全部 $K$、$V$ 做 Attention（包含之前 Chunk 已存入 Cache 的部分）。
3. 将 Chunk $i$ 的 $K$、$V$ 写入 KV Cache。
4. 输出 Chunk $i$ 对应位置的隐状态。

**Attention Mask**：Chunk 内部保持 Causal Mask（下三角），对之前的 Chunk 全部可见（无 Mask）。

$$ M_{ij} = \begin{cases} 0 & \text{if } j \leq i \text{（当前 token 可见历史）} \ -\infty & \text{if } j > i \text{（屏蔽未来）} \end{cases} $$

---

### 5.5 多个 Chunk 能否并行计算

**不能**（对单个请求的 Prefill）。

原因：自回归的 Causal Attention 要求第 $i$ 个 Chunk 的 Attention 必须看到前 $i-1$ 个 Chunk 的 KV，存在数据依赖：

$$ \text{Chunk}_i \text{ 的 Attention} \leftarrow \text{KV Cache of Chunk}_{1\ldots i-1} $$

因此同一请求的 Chunk 必须顺序处理。但**不同请求**的 Chunk 可以在同一 Micro-batch 内并行（通过 Batch 维度并行）。

---

## 6. MTP（Multi-Token Prediction）

### 6.1 MTP 大致流程

**核心思想**：在标准 LLM 单步预测 1 个 token 的基础上，附加若干轻量级预测头（Draft Head），单次 Forward 同时预测接下来的 $D$ 个 token。

**结构**：

```
主模型 Transformer（L 层）
    ↓ 隐状态 h_t
  [Draft Head 1] → token_{t+1}
  [Draft Head 2] → token_{t+2}（基于 h_t 和 Draft Head 1 的结果）
  ...
  [Draft Head D] → token_{t+D}
```

每个 Draft Head 通常是共享主模型权重 + 单独的轻量 MLP/Transformer 层。

**Verify 阶段**（若与投机采样配合）：Draft 输出经主模型并行验证，接受前缀最长的连续正确 token。

---

### 6.2 MTP 接受率

接受率（Acceptance Rate）定义为每次 Draft 中被接受的平均 token 数 / Draft 数量 $D$。

典型值（依赖任务类型）：

|场景|接受率参考范围|
|---|---|
|代码生成（重复性高）|$60%$–$85%$|
|通用对话|$50%$–$75%$|
|随机/创意文本|$30%$–$55%$|

DeepSeek-V3 官方报告 MTP 接受率约 $85%$（代码/数学场景）。

_注：实际测试结果与模型、任务、temperature 设置密切相关，无确切通用数值。_

---

### 6.3 TTFT / TPOT 定义，MTP 优化哪部分

**TTFT（Time to First Token）**：从请求到达到输出第一个 token 的时延，主要由 Prefill 阶段决定。

$$ \text{TTFT} = t_{\text{first token}} - t_{\text{request arrive}} $$

**TPOT（Time Per Output Token）**：生成阶段每个 token 的平均时延：

$$ \text{TPOT} = \frac{t_{\text{last token}} - t_{\text{first token}}}{N_{\text{output tokens}} - 1} $$

**MTP 优化的指标**：主要优化 **TPOT** 和 **吞吐（Throughput）**。

MTP 不改变 Prefill 流程，对 TTFT 无直接影响；通过每步生成多个 token，降低单 token 的平均解码时延（TPOT），提升整体生成速度。

---

### 6.4 MTP 引入额外计算后为何仍能提升吞吐

**关键洞察**：Decode 阶段是 Memory Bandwidth Bound，GPU 的算力（FLOP/s）远未饱和。

**分析**：

- 标准 Decode：每步生成 1 个 token，需加载全量权重 $W$（$\approx 2 \times \text{param_count}$ bytes for FP16），算力利用率极低。
- MTP Decode：同样加载一次权重，同时预测 $D$ 个 token，额外的 Draft Head 计算量相对主模型极小（$< 5%$），不构成瓶颈。

**吞吐提升来源**：

$$ \text{有效吞吐} = \frac{\text{接受的 token 数}}{1 \text{ 步时间}} = \frac{1 + \alpha \cdot (D-1)}{T_{\text{step}}} $$

其中 $\alpha$ 为接受率，$T_{\text{step}}$ 几乎不变（带宽仍主导），因此分子增大而分母不变，吞吐线性提升。

---

## 7. Flash Attention

### 7.1 Flash Attention 核心思想

**问题**：标准 Attention 的中间矩阵 $S = QK^\top \in \mathbb{R}^{N \times N}$ 需要写回 HBM，$N$ 为序列长度，I/O 复杂度为 $O(N^2)$，成为瓶颈。

**核心思想（IO-Aware）**：将 $Q$、$K$、$V$ 分块（Tiling）载入 SRAM，在片内完成 Softmax 和矩阵乘，避免 $N \times N$ 中间结果写回 HBM。

**IO 复杂度对比**：

|方法|HBM 读写次数|
|---|---|
|标准 Attention|$O(N^2)$|
|Flash Attention|$O(N^2 / M)$（$M$ 为 SRAM 大小）|

---

### 7.2 分块计算与在线 Softmax

**分块设置**：$Q$ 按行分块为 $T_r$ 块，$K$、$V$ 按列分块为 $T_c$ 块，块大小 $B_r$、$B_c$ 满足 $B_r \cdot B_c \leq M$（SRAM 容量）。

**Softmax 分母（归一化因子）**：

标准 Softmax 对行 $i$ 的归一化因子：

$$ \ell_i = \sum_{j=1}^{N} e^{s_{ij} - m_i}, \quad m_i = \max_j s_{ij} $$

其中 $s_{ij} = Q_i K_j^\top / \sqrt{d_h}$。

**在线（Online）Softmax**：逐块更新最大值和累计和，无需预先计算全行：

初始化：$m_i^{(0)} = -\infty$，$\ell_i^{(0)} = 0$，$O_i^{(0)} = 0$。

处理第 $j$ 块时：

$$ m_i^{(j)} = \max(m_i^{(j-1)},\ \max_k s_{ik}^{(j)}) $$

$$ \ell_i^{(j)} = e^{m_i^{(j-1)} - m_i^{(j)}} \cdot \ell_i^{(j-1)} + \sum_k e^{s_{ik}^{(j)} - m_i^{(j)}} $$

$$ O_i^{(j)} = e^{m_i^{(j-1)} - m_i^{(j)}} \cdot O_i^{(j-1)} + e^{S_i^{(j)} - m_i^{(j)}} V^{(j)} $$

最终输出：

$$ O_i = O_i^{(T_c)} / \ell_i^{(T_c)} $$

这保证了分块计算与全局 Softmax 数值等价，且全程不写 $N \times N$ 矩阵到 HBM。

---

## 8. DBO（Disaggregated/Distributed Block-wise Overlap）

### 8.1 DBO 是什么

**DBO（Disaggregated Block-wise Overlapping）** 是一种将 Transformer 层间的 **计算（Compute）与通信（Communication）重叠（Overlap）** 的技术，常见于 Pipeline Parallelism 或 Tensor Parallelism 场景，用于隐藏跨节点通信延迟。

**背景**：在 Tensor Parallelism（TP）下，每个 Transformer 层后需要 AllReduce 通信（对 FFN 输出求和）；在 Pipeline Parallelism（PP）下，层间激活需要 P2P 传输。这些通信直接阻塞下一层的计算。

**DBO 目标**：将第 $l$ 层的通信与第 $l+1$ 层（或同层内其他部分）的计算并行执行，实现 Overlap。

---

### 8.2 DBO 中通信的主要来源

|并行维度|通信操作|通信量|
|---|---|---|
|Tensor Parallelism|AllReduce（FFN 输出）|$O(B \times s \times d_{model})$|
|Tensor Parallelism|AllGather（MHA 输入）|同上|
|Pipeline Parallelism|P2P Send/Recv（层间激活）|$O(B \times s \times d_{model})$|
|Expert Parallelism|AlltoAll（token 路由）|$O(B \times s \times d_{model})$|

TP 中 AllReduce 是主要通信来源，带宽需求高，延迟直接暴露在关键路径上。

---

### 8.3 DBO 是否通过 Microbatch 实现计算通信 Overlap

**是**。DBO 的典型实现使用 Microbatch 切分：

将一个 Batch 切为多个 Microbatch $\mu_1, \mu_2, \ldots$。当 $\mu_1$ 在执行第 $l$ 层后发起 AllReduce 通信时，$\mu_2$ 开始执行第 $l$ 层的 GEMM 计算，实现二者的并行：

```
时间线:
  μ1: [Layer l GEMM] → [AllReduce] → [Layer l+1 GEMM] → ...
  μ2:                    [Layer l GEMM] → [AllReduce] → ...
```

理想情况下通信完全被计算掩盖，有效带宽利用率接近 100%。

---

## 9. AM 分离（Attention-MLP Disaggregation）

### 9.1 AM 分离适用场景

**AM 分离**（Attention-MoE 分离，或广义 Attention-FFN 分离）将 Transformer 中的 **Attention 模块** 与 **FFN/MoE 模块** 部署在不同类型或数量的节点上。

**适用场景**：

- **MoE 模型**（如 DeepSeek-V3、Mixtral）：FFN 为稀疏 MoE，Expert 数量多（如 256），Expert 并行度（EP）极高，但 Attention 参数量小，无需相同并行度。
- **长上下文推理**：Attention 的 KV Cache 随序列增长，显存压力大；FFN 权重固定，无此问题。
- **异构集群**：Attention 和 FFN 的算力/带宽需求不同，可分配到不同型号 GPU。

---

### 9.2 Attention 节点与 FFN 节点的瓶颈

|节点类型|计算特征|主要瓶颈|
|---|---|---|
|Attention 节点|KV Cache 读写，序列长度敏感|**HBM 带宽 Bound**（KV Cache 读取）|
|FFN/Expert 节点|大规模权重矩阵乘（GEMM）|**Compute Bound**（Dense GEMM）或 **Expert 路由延迟**|

Decode 阶段 Attention 节点的 KV Cache 访问量为：

$$ \text{KV read per step} = 2 \times L \times H \times d_h \times s \times b \ \text{bytes} $$

随 $s$ 线性增长，带宽压力主导。

---

### 9.3 AM 分离的收益

- **独立扩缩容**：Attention 节点按序列长度/并发量扩容；FFN/MoE 节点按 Expert 数量扩容，避免捆绑浪费。
- **资源匹配优化**：Attention 节点使用高 HBM 带宽 GPU（如 H100 SXM），FFN 节点可使用高算力或更多数量的 GPU。
- **降低 MoE 的 Expert Parallelism 通信开销**：Attention 部分不参与 AlltoAll 路由通信，通信拓扑更简洁。
- **KV Cache 隔离**：KV Cache 集中在 Attention 节点，便于统一管理和量化压缩。

---

## 10. 单机显存不足的解决方案

### 10.1 显存不足的处理策略

**10.1.1 模型权重侧**

- **量化（Quantization）**：FP16→FP8/INT8/INT4/FP4，权重显存直接减半或更多。
- **张量并行（Tensor Parallelism）**：多 GPU 切分权重矩阵，每卡只持有 $1/N$ 权重。
- **CPU Offload**：将部分层权重卸载到 CPU 内存，按需加载（以延迟换显存）。

**10.1.2 KV Cache 侧**

- **KV Cache 量化**：FP8/INT4 压缩 KV，显存减半。
- **KV Cache 压缩（MQA/GQA/MLA）**：减少 KV Head 数量。
- **Token Eviction（如 StreamingLLM、H2O）**：丢弃历史 token 的 KV，保留 Sink + 近期 token。
- **KV Offload 到 CPU/SSD**：将不活跃的 KV Page 换出。

**10.1.3 系统层**

- **Reduce max batch size**：牺牲吞吐换显存。
- **缩短 max sequence length**：限制 KV Cache 上限。

---

### 10.2 推理时显存的组成

$$ \text{显存总量} = \text{模型权重} + \text{KV Cache} + \text{激活（Activation）} + \text{框架开销} $$

|组成部分|典型占比（Decode）|说明|
|---|---|---|
|模型权重|60–80%|FP16：约 $2 \times \text{参数量}$ bytes|
|KV Cache|15–35%|随 Batch × 序列长度线性增长|
|Activation|$<5%$|仅当前层中间结果，较小|
|框架/CUDA 上下文|$\sim1$ GB|固定开销|

---

### 10.3 针对权重与 KV Cache 的优化

|目标|优化方法|典型收益|
|---|---|---|
|权重|FP8 量化|$2\times$ 压缩|
|权重|INT4/W4A16|$4\times$ 压缩|
|权重|TP 多卡切分|按卡数线性分摊|
|KV Cache|FP8 KV 量化|$2\times$ 压缩|
|KV Cache|GQA（$H \to G$）|$H/G$ 倍压缩|
|KV Cache|MLA 低秩压缩|$>10\times$ 压缩|
|KV Cache|Token Eviction|按保留比例压缩|
|KV Cache|Offload 到 CPU|几乎无上限（受带宽限制）|

---

### 10.4 Offload 卸载的内容

**通常卸载的对象**：

- **KV Cache**：最常见，将非活跃请求的 KV Page 换出到 CPU DRAM（FlexGen、vLLM 的 `cpu_offload_gb` 选项）。
- **模型权重（部分层）**：将不在当前执行层的 Transformer Block 卸载到 CPU，按层轮换加载（FlexGen、HuggingFace Accelerate `device_map="auto"`）。

**极少卸载的对象**：激活（Activation），因为激活仅在当前层存在，生命周期极短，卸载收益低。

---

### 10.5 Offload 是在 CPU 计算还是再加载回 GPU

**再加载回 GPU 计算**（主流方案）。

- 权重/KV Cache 存在 CPU DRAM，需要时通过 PCIe 传输到 GPU 显存，在 GPU 上执行 GEMM/Attention。
- CPU 上执行 GEMM 吞吐极低（无 Tensor Core），对大模型不可行。
- 少数方案（如 llama.cpp CPU 推理）直接在 CPU 上计算，但仅适用于对延迟不敏感的小规模场景。

---

## 11. 高效 Attention 方法

### 11.1 Linear Attention

**核心思想**：将 Softmax Attention 的 $O(N^2)$ 复杂度降至 $O(N)$，通过 Kernel 近似或特征映射替代 Softmax：

$$ \text{Attention}(Q, K, V) \approx \frac{\phi(Q)(\phi(K)^\top V)}{\phi(Q) \sum_j \phi(K_j)} $$

其中 $\phi(\cdot)$ 为特征映射函数（如 ELU+1、Random Fourier Features）。

**优势**：推理时递归形式，$O(1)$ 内存，适合超长序列。  
**劣势**：表达能力弱于 Softmax Attention，精度损失明显；实际推理速度优势需序列极长才显现。

**代表**：Linear Transformer、RetNet（Recurrent Formulation）、Mamba（SSM 近似）。

### 11.2 窗口 Attention（Sliding Window Attention）

仅对每个 token 计算其前 $w$ 个 token 的 Attention（局部窗口），复杂度降为 $O(N \cdot w)$。

**代表**：Longformer、Mistral（Sliding Window + Rolling KV Buffer）。

**局限**：无法建模超过窗口范围的长程依赖，通常与全局 Attention 混合使用。

---

## 12. 模型量化

### 12.1 量化概念

**量化（Quantization）** 将浮点权重/激活映射为低比特整数或低精度浮点，以减少存储和计算开销。

**基本公式**：均匀量化（对称）：

$$ x_q = \text{round}\left(\frac{x}{s}\right), \quad s = \frac{\max(|x|)}{2^{b-1} - 1} $$

反量化：

$$ \hat{x} = x_q \cdot s $$

其中 $b$ 为量化位宽，$s$ 为 Scale Factor。

---

### 12.2 FP16 / FP8 / FP4 格式

|格式|指数位|尾数位|符号位|动态范围|典型用途|
|---|---|---|---|---|---|
|FP32|8|23|1|$\sim 10^{\pm 38}$|训练参考精度|
|FP16|5|10|1|$\sim 10^{\pm 4.8}$|推理主流精度|
|BF16|8|7|1|$\sim 10^{\pm 38}$|训练/推理|
|FP8 E4M3|4|3|1|$\pm 448$|权重/激活量化|
|FP8 E5M2|5|2|1|$\pm 57344$|梯度量化|
|FP4|2|1|1|极小|实验性权重量化|

FP8 E4M3 用于权重（更大精度），FP8 E5M2 用于梯度（更大范围）。

---

### 12.3 NVFP4 量化对象

**NVFP4**（NVIDIA FP4）是 Blackwell（B200/B100）架构引入的 4-bit 浮点格式（E2M1）。

量化对象：**模型权重（Weights）**，通常配合 FP8 或 FP16 的 Activation 进行 W4A8/W4A16 混合精度推理。

具体形式：权重以 FP4 存储，计算时反量化到 FP8/FP16 再进行 GEMM（或使用 Blackwell 原生 MXFP4 Tensor Core 直接支持 FP4×FP4 计算）。

---

### 12.4 量化的时机：提前还是加载后

**两种模式**：

|模式|时机|代表方法|
|---|---|---|
|**Post-Training Quantization（PTQ）**|训练后离线校准，存储低精度权重|GPTQ、AWQ、SmoothQuant|
|**On-the-fly 量化**|加载原始 FP16 权重后在线量化|bitsandbytes（load_in_4bit）|

工业推理部署主流采用 **PTQ**：提前量化好权重文件，直接加载低精度格式，启动快、无运行时开销。

---

### 12.5 权重低精度、Activation FP16 时的计算方式

以 W4A16（INT4 权重 + FP16 激活）为例：

**计算步骤**：

1. 从显存读取 INT4 权重（$\frac{1}{4}$ 存储量）。
2. **反量化（Dequantize）**：在 GPU Compute Kernel 内将 INT4 权重转换为 FP16：

$$ W_{\text{fp16}} = W_{\text{int4}} \times s + z \quad (s \in \mathbb{R}^{N \times 1}, z \text{ 为零点}) $$

3. 执行 **FP16 GEMM**：$Y = X_{\text{fp16}} \cdot W_{\text{fp16}}^\top$。

**关键点**：

- 反量化在 CUDA Kernel 内 fused 完成，不额外写回显存。
- 实际 FLOP 为 FP16，但 HBM 读取量降至 $1/4$（INT4 vs FP16），对 Memory Bound 的 Decode 阶段收益显著。
- NVIDIA 的 Marlin 内核和 vLLM 均实现了高效的 W4A16 fused dequant+GEMM。

---

## 13. Decode-only 大模型

### 13.1 熟悉的 Decode-only 模型

|模型系列|机构|特点|
|---|---|---|
|LLaMA 3|Meta|GQA，RoPE，RMSNorm，开源主流|
|DeepSeek-V3|DeepSeek|MoE + MLA，FP8 训练|
|Mistral / Mixtral|Mistral AI|Sliding Window + GQA / MoE|
|Qwen 2.5|Alibaba|GQA，长上下文|
|GPT 系列|OpenAI|闭源，MHA → GQA（GPT-4）|

---

### 13.2 LLaMA 简介

LLaMA（Large Language Model Meta AI）是 Meta 开源的 Decoder-only Transformer 系列。

**关键架构设计**：

|组件|设计选择|说明|
|---|---|---|
|位置编码|RoPE|相对位置，支持长度外推|
|归一化|RMSNorm（Pre-Norm）|稳定训练，去除 LayerNorm 均值项|
|激活函数|SwiGLU|$\text{SwiGLU}(x) = \text{Swish}(W_1 x) \odot W_2 x$|
|Attention|GQA（LLaMA-2 70B+，LLaMA-3 全系）|减少 KV Head 数|
|词表|BPE（SentencePiece）||
|无偏置项|线性层不加 bias|简化结构|

---

### 13.3 LLaMA-3-8B 推理过程的 Shape 变化

**模型参数**：$L=32$，$d_{model}=4096$，$H_Q=32$，$H_{KV}=8$（GQA），$d_h=128$，$d_{FFN}=14336$，词表大小 $V=128256$。

**输入**：Prompt 长度 $s=512$，Batch size $B=1$。

---

**Embedding 层**：

$$ \text{Input IDs}: (B, s) = (1, 512) \xrightarrow{\text{Embed}} X: (1, 512, 4096) $$

---

**每个 Transformer 层（重复 $L=32$ 次）**：

**RMSNorm**：

$$ (1, 512, 4096) \to (1, 512, 4096) $$

**Q/K/V 投影**：

$$ W_Q: (4096, 4096),\ W_K: (4096, 1024),\ W_V: (4096, 1024) $$

$$ Q = X W_Q^\top: (1, 512, 4096) \to Q: (1, 32, 512, 128) $$

$$ K = X W_K^\top: (1, 512, 1024) \to K: (1, 8, 512, 128) $$

$$ V = X W_V^\top: (1, 512, 1024) \to V: (1, 8, 512, 128) $$

（GQA：KV Head 数 8 vs Q Head 数 32，每组 4 个 Q Head 共享一组 KV）

**GQA Attention**（$K$、$V$ 广播到 32 个 Head）：

$$ \text{Scores}: (1, 32, 512, 512) \xrightarrow{\text{Softmax}} \text{Attn}: (1, 32, 512, 128) $$

$$ \xrightarrow{\text{reshape}} (1, 512, 4096) $$

**Output 投影**：$W_O: (4096, 4096)$

$$ (1, 512, 4096) \to (1, 512, 4096) $$

**FFN（SwiGLU）**：

$$ W_1, W_3: (4096, 14336),\ W_2: (14336, 4096) $$

$$ (1, 512, 4096) \xrightarrow{W_1, W_3} (1, 512, 14336) \xrightarrow{\text{SwiGLU}} (1, 512, 14336) \xrightarrow{W_2} (1, 512, 4096) $$

---

**最终输出层**：

$$ \text{RMSNorm}: (1, 512, 4096) \to (1, 512, 4096) $$

$$ \text{LM Head } (W_{LM}: (4096, 128256)): (1, 512, 4096) \to \text{Logits}: (1, 512, 128256) $$

$$ \xrightarrow{\text{取最后一个 token}} (1, 128256) \xrightarrow{\text{argmax/sampling}} \text{next token id}: (1,) $$

---

**Decode 阶段**（每步输入 1 个新 token，$s=1$，KV Cache 长度已达历史长度）：

$$ X_{\text{new}}: (1, 1, 4096) $$

$$ Q: (1, 32, 1, 128),\quad K_{\text{new}}: (1, 8, 1, 128),\quad V_{\text{new}}: (1, 8, 1, 128) $$

$$ K_{\text{cache}} \leftarrow \text{concat}(K_{\text{cache}},\ K_{\text{new}}),\ \text{shape}: (1, 8, s+1, 128) $$

$$ \text{Attn scores}: (1, 32, 1, s+1) \to \text{Output}: (1, 1, 4096) $$

$$ \to \text{Logits}: (1, 1, 128256) \to \text{next token} $$
