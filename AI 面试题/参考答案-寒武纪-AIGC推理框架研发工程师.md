## 1. 推理框架研发工程师岗位理解

推理框架研发工程师的核心职责是构建和优化将训练好的大语言模型部署到生产环境的软件栈。该岗位处于算法与硬件之间的系统层，工作内容横跨以下维度：

**系统层面**

- 设计高吞吐、低延迟的 inference serving 系统（如 vLLM、TensorRT-LLM、SGLang）
- 实现 continuous batching、chunked prefill、P/D disaggregation 等调度策略
- KV Cache 内存管理（PagedAttention、prefix caching、offloading）

**算子/内核层面**

- 编写或优化 CUDA kernel（FlashAttention、GEMM、量化算子）
- 算子融合（kernel fusion）减少 kernel launch overhead 与 memory round-trip
- 使用 CUTLASS、Triton、cuBLAS 等库

**系统集成层面**

- Tensor Parallelism / Pipeline Parallelism / Expert Parallelism 分布式推理
- 模型量化（INT8/FP8/INT4）的工程实现与精度保障
- 与上层 API 服务（OpenAI-compatible endpoint）的对接

**差异化要求**（区别于训练框架研发）

|维度|推理框架|训练框架|
|---|---|---|
|延迟敏感度|极高（TTFT/TPOT SLA）|较低（吞吐优先）|
|内存模式|KV Cache 动态管理|梯度/优化器状态静态分配|
|批处理模式|Dynamic batching|Static batch|
|主要瓶颈|Memory bandwidth bound|Compute bound|

---

## 2. 降低大模型推理成本手段

### 2.1 计算效率优化

**算子融合（Kernel Fusion）**：将多个小算子合并为一个 CUDA kernel，消除中间 tensor 的 HBM 读写。典型案例：FlashAttention 将 $QK^T$、Softmax、$\cdot V$ 融合为单次 kernel。

**量化（Quantization）**：将权重/激活从 FP16 降低到 INT8/FP8/INT4，降低计算量和内存带宽需求。

**推测解码（Speculative Decoding）**：用小 draft model 生成候选 token，大 target model 并行验证，在维持输出分布不变的前提下提升吞吐。

### 2.2 内存效率优化

**KV Cache 量化**：将 KV Cache 从 FP16 压缩为 FP8/INT8，降低 KV Cache 占用，提升可服务的并发请求数。

**PagedAttention**：非连续 KV Cache 内存分配，消除 internal fragmentation，KV Cache 利用率从约 60% 提升至接近 100%。

**Prefix Caching / RadixAttention**：对共享前缀的请求复用 KV Cache，减少重复计算。

	**MQA / GQA / MLA**：减少 KV head 数量，直接压缩 KV Cache 体积。

### 2.3 调度与服务优化

**Continuous Batching**：请求完成即释放 slot，新请求填入，GPU 利用率相比 static batching 大幅提升。

**Chunked Prefill**：将长 prefill 拆分为小 chunk 与 decode 请求混合调度，降低 TTFT 同时维持 TPOT。

**P/D Disaggregation**：Prefill 与 Decode 阶段分离到不同机器，分别针对 compute-bound 和 memory-bandwidth-bound 进行硬件选型。

### 2.4 模型架构层优化

**MoE（Mixture of Experts）**：激活参数量仅为总参数量的一小部分，在保持模型容量的同时降低推理计算量。

**蒸馏（Distillation）**：训练小模型模仿大模型输出，部署时使用小模型。

**剪枝（Pruning）**：移除不重要的权重或注意力头。

---

## 3. 为什么要做算子融合

### 3.1 问题根源：内存墙（Memory Wall）

现代 GPU 的计算峰值（Compute Throughput）远高于内存带宽（Memory Bandwidth）。以 A100 SXM 为例：

|指标|数值|
|---|---|
|FP16 Tensor Core FLOPS|312 TFLOPS|
|HBM2e 带宽|2 TB/s|
|Arithmetic Intensity 临界点（Ridge Point）|约 156 FLOP/Byte|

对于 memory-bound 算子（如 LayerNorm、GeLU、逐元素加法），Arithmetic Intensity 远低于 Ridge Point，GPU 的计算单元大量空闲，瓶颈在 HBM 带宽。

### 3.2 未融合 vs. 融合的 Memory Round-trip

以 Transformer 中 `Linear → GeLU → Linear` 为例，未融合时：

$$ \text{HBM I/O} = \underbrace{W_1 + X}_{\text{read}} + \underbrace{H}_{\text{write}} + \underbrace{H}_{\text{read}} + \underbrace{H'}_{\text{write}} + \underbrace{W_2 + H'}_{\text{read}} + \underbrace{O}_{\text{write}} $$

融合后，中间张量 $H$、$H'$ 常驻 SRAM（L2 Cache 或 Shared Memory），HBM I/O 减少为：

$$ \text{HBM I/O}_{\text{fused}} = \underbrace{W_1 + W_2 + X}_{\text{read}} + \underbrace{O}_{\text{write}} $$

### 3.3 Kernel Launch Overhead

每次 CUDA kernel 启动在 CPU 端有约 5–20 μs 的调度延迟。在 decode 阶段批量较小时，kernel launch overhead 在总时延中占比显著。融合后多个逻辑操作对应单次 kernel 启动。

### 3.4 典型案例

**FlashAttention**：将 $S = QK^T / \sqrt{d}$、$P = \text{Softmax}(S)$、$O = PV$ 三步融合，通过 tiling 使中间矩阵 $S$、$P$ 常驻 Shared Memory，避免 $O(N^2)$ 的 HBM 写入，将 Attention 的内存复杂度从 $O(N^2)$ 降至 $O(N)$。

**RMSNorm + Linear**：归一化与线性变换融合，消除归一化结果的 HBM 写回。

---

## 4. CUDA Graph 原理、作用、使用场景

### 4.1 原理

标准 CUDA 执行模型中，CPU 端逐个向 CUDA Stream 提交 kernel（`cudaLaunchKernel`）、内存拷贝（`cudaMemcpyAsync`）等操作，每次提交均有 CPU-GPU 交互开销。

CUDA Graph 将一段计算图**录制（capture）**为 `cudaGraph_t` 对象，实例化为 `cudaGraphExec_t` 后，单次 `cudaGraphLaunch` 即可触发整张图的执行，CPU 开销从 $O(N_{\text{ops}})$ 降至 $O(1)$。

```
Capture:  cudaStreamBeginCapture(stream)
          ... kernel launches, memcpy ...
          cudaStreamEndCapture(stream, &graph)

Instantiate: cudaGraphInstantiate(&graphExec, graph, ...)

Execute: cudaGraphLaunch(graphExec, stream)  // 单次调用
```

### 4.2 内部机制

录制阶段不实际执行 kernel，而是将每个 CUDA API 调用记录为图节点（`cudaGraphNode_t`），依据 Stream 依赖关系自动推导节点间的边（happens-before 关系），形成 DAG。

实例化阶段在 GPU 端生成执行计划（execution plan），预分配所需资源，消除运行时动态决策。

### 4.3 作用

|优化项|说明|
|---|---|
|消除 CPU kernel launch overhead|每次 launch 约 5–20 μs，decode 阶段数十个 kernel 累积可达数百 μs|
|消除 CPU-GPU synchronization|图执行期间 CPU 无需介入|
|允许驱动级调度优化|驱动可对图内无数据依赖的 kernel 进行并发调度|

### 4.4 使用场景

**适用**：

- LLM decode 阶段（固定 batch size、固定 sequence length）：计算图拓扑稳定，可一次录制、反复执行
- 循环体内重复执行相同计算序列
- kernel launch overhead 占比显著（小 batch、多 kernel 场景）

**不适用**：

- 动态控制流（if-else、while 依赖运行时数据）
- Prefill 阶段（sequence length 变化，图拓扑随之改变）
- 动态 batch size 变化（需重新实例化或使用 `cudaGraphExecUpdate`）

### 4.5 PyTorch 中的使用

```python
# 预热（warm-up）：填充 CUDA cache allocator
for _ in range(3):
    output = model(input)

# 录制
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(static_input)

# 执行（复用已录制图）
static_input.copy_(new_input)
g.replay()
result = output.clone()
```

vLLM 的 decode 阶段正是通过此机制将 CPU overhead 降至最低。

---

## 5. 3D 并行、EP、CP、SP 原理与作用

### 5.1 Tensor Parallelism（TP）

将单层的权重矩阵沿某一维度切分到多卡。以 $Y = XW$ 为例：

**列切分（Column Parallel Linear）**：$W$ 按列切分为 $[W_1 | W_2]$，各卡计算 $Y_i = X W_i$，结果沿列维度 All-Gather 或直接传给下一层。

**行切分（Row Parallel Linear）**：$W$ 按行切分为 $\begin{bmatrix}W_1 \ W_2\end{bmatrix}$，各卡持有 $X$ 的对应列分片，计算局部结果后 All-Reduce 求和。

**TP 中的通信**：每个 Transformer block 需要 2 次 All-Reduce（MLP 的两层线性各一次），通信量与激活大小成正比，要求高带宽 NVLink。

### 5.2 Pipeline Parallelism（PP）

将模型按层切分到多台机器，每台机器持有若干连续层（stage）。数据以 micro-batch 形式流水线执行。

**GPipe**：先全部前向，再全部后向（推理场景下只有前向），气泡率（bubble ratio）为：

$$ \text{bubble ratio} = \frac{p-1}{m + p - 1} $$

其中 $p$ 为 pipeline stage 数，$m$ 为 micro-batch 数。

**1F1B**：前向完成一个 micro-batch 立即开始后向，降低 activation 内存占用。

### 5.3 Data Parallelism（DP）

每卡持有完整模型副本，不同 micro-batch 并行，梯度 All-Reduce 后更新权重。推理场景下退化为多副本独立服务。

### 5.4 3D 并行

同时使用 TP + PP + DP，将 GPU 集群分为三个维度：

$$ N_{\text{GPU}} = N_{TP} \times N_{PP} \times N_{DP} $$

典型配置：TP 在节点内（NVLink 高带宽），PP/DP 跨节点（InfiniBand）。

### 5.5 Expert Parallelism（EP）

针对 MoE 模型。每个 expert 被分配到不同 GPU，Token 经 Router 决策后通过 All-to-All 通信路由到持有对应 expert 的 GPU 计算，结果再 All-to-All 返回。

**EP 的核心挑战**：

- All-to-All 通信量大，延迟高
- Token 在 expert 间分布不均（load imbalance）导致部分 GPU 空闲

### 5.6 Context Parallelism（CP）

将超长 sequence 的 attention 计算沿 sequence 维度切分到多卡，用于 long-context 推理（如 128K+ token）。每卡持有 sequence 的一个分片，通过 Ring-Attention 等方式实现跨卡 attention，各卡 $Q$ 与本地 $K$、$V$ 计算，旋转 $K$、$V$ 块遍历所有卡。

### 5.7 Sequence Parallelism（SP）

与 TP 配合使用，将 Dropout、LayerNorm 等不可 TP 切分的算子的激活值沿 sequence 维度切分，避免 TP 组内这些算子持有完整激活复制。Megatron-LM 在 TP 的 All-Reduce 基础上改为 Reduce-Scatter + All-Gather，激活内存降低 $N_{TP}$ 倍。

---

## 6. 模型量化类型、原理、作用

### 6.1 量化基础

将浮点数 $x \in \mathbb{R}$ 映射到整数 $x_q \in \mathbb{Z}$：

$$ x_q = \text{clamp}\!\left(\left\lfloor \frac{x}{s} \right\rceil + z,\ q_{\min},\ q_{\max}\right) $$

反量化：

$$ \hat{x} = s \cdot (x_q - z) $$

其中 $s$（scale）和 $z$（zero-point）为量化参数，$\lfloor \cdot \rceil$ 为四舍五入。

对称量化（$z = 0$）：

$$ s = \frac{\max(|x|)}{2^{b-1} - 1} $$

### 6.2 量化粒度

|粒度|描述|精度|开销|
|---|---|---|---|
|Per-tensor|整个 tensor 共享一组 $(s, z)$|最低|最小|
|Per-channel|每个输出通道独立 $(s, z)$|中|小|
|Per-group|每 $g$（如 128）个元素共享 $(s, z)$|高|中|
|Per-token|激活量化每个 token 独立|高|中|

### 6.3 PTQ（训练后量化）

训练完成后直接量化，无需重训。

**GPTQ**：基于二阶信息（Hessian）的逐层权重量化。对每一列权重 $w_q$，通过 OBQ（Optimal Brain Quantization）最小化量化误差，并将误差传播到剩余列：

$$ \delta W_F = -\frac{w_q - \text{quant}(w_q)}{[H_F^{-1}]_{qq}} \cdot (H_F^{-1})_{:,q} $$

使用 Cholesky 分解高效求解 $H^{-1}$，Lazy Batch-Updates 技术批量处理列以提升 GPU 利用率。

**AWQ（Activation-Aware Weight Quantization）**：观察到权重中存在 salient channel（对应激活值大的通道），对这些通道进行缩放以保护其精度：

$$ \hat{W} = W \cdot \text{diag}(s)^{-1}, \quad \hat{X} = X \cdot \text{diag}(s) $$

通过搜索最优 $s$ 使量化误差最小，且无需 Hessian 计算。

**SmoothQuant**：将激活的量化难度迁移到权重侧：

$$ Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) = \hat{X} \hat{W} $$

### 6.4 QAT（量化感知训练）

在训练过程中引入 fake quantization 节点，模拟量化误差，使模型适应量化噪声。精度高于 PTQ，但需要重训成本。

### 6.5 KV Cache 量化

将 KV Cache 从 FP16 量化为 FP8 或 INT8。KV 中 K 和 V 的数值分布差异显著：K 分布较均匀，V 存在少量大值（outlier），需分别处理。

### 6.6 作用

|量化精度|内存压缩比|计算加速|精度损失|
|---|---|---|---|
|FP16 → INT8|2×|~1.5–2×|小|
|FP16 → FP8|2×|~1.5–2×|极小（H100 原生支持）|
|FP16 → INT4|4×|~3–4×|中等，需 GPTQ/AWQ|
|FP16 → INT2|8×|理论高|较大|

---

## 7. MTP（Multi-Token Prediction）介绍

### 7.1 背景

标准自回归解码每步预测 1 个 token，GPU 计算单元在 decode 阶段因 batch 小、memory-bound 而大量空闲。MTP 尝试在单次前向中预测多个未来 token，以更充分地利用计算资源。

### 7.2 核心思想

在主模型最后几层之上，添加 $D$ 个额外的 MTP head，第 $k$ 个 head 预测位置 $t+k$ 处的 token：

$$ \hat{p}(x_{t+k} \mid x_{<t}) = \text{MTPHead}_k(h_t) $$

其中 $h_t$ 为主干网络在位置 $t$ 的隐状态。

DeepSeek-V3 采用的 MTP 方案中，每个额外 head 由独立的 Transformer block 构成，共享 embedding 层，各 head 预测深度递增的 token：

$$ \text{Head}_k: \quad \tilde{h}^{(k)}_t = \text{TransformerBlock}_k\!\left(\text{concat}(h_t,\ \text{Emb}(x_{t+k-1}))\right) $$

### 7.3 训练目标

多任务交叉熵损失，主预测头权重高，MTP 辅助头权重较低：

$$ \mathcal{L} = \mathcal{L}_{\text{main}} + \lambda \sum_{k=1}^{D} \mathcal{L}_k $$

DeepSeek-V3 中 $\lambda = 0.3$，$D = 1$（仅 1 个额外 MTP head）。

### 7.4 推理时的使用方式

**作为推测解码的 draft model**：MTP head 在每步 decode 时同时产出 $D$ 个候选 token，主模型并行验证，接受率高时吞吐提升接近 $D$ 倍，且无需额外的独立 draft model。

**接受条件**（与 Speculative Decoding 一致）：

$$ \text{accept } \hat{x}_{t+k} \iff u \sim U[0,1] \leq \frac{p_{\text{target}}(\hat{x}_{t+k})}{p_{\text{draft}}(\hat{x}_{t+k})} $$

### 7.5 MTP vs. 标准 Speculative Decoding

|维度|MTP|独立 Draft Model|
|---|---|---|
|额外模型|无（head 内置）|需要单独小模型|
|Draft 质量|受限于 head 容量|取决于 draft 模型质量|
|部署复杂度|低|高|
|训练成本|需联合训练|可独立训练|

---

## 8. vLLM 整体架构与请求流转

### 8.1 整体架构

```
┌─────────────────────────────────────────────┐
│              API Server (FastAPI)            │
│          /v1/completions, /v1/chat           │
└────────────────────┬────────────────────────┘
                     │  AsyncLLMEngine
┌────────────────────▼────────────────────────┐
│              LLMEngine                       │
│  ┌──────────────┐   ┌──────────────────────┐│
│  │  Scheduler   │   │  BlockSpaceManager   ││
│  │ (Continuous  │   │  (PagedAttention)    ││
│  │  Batching)   │   └──────────────────────┘│
│  └──────┬───────┘                            │
└─────────┼───────────────────────────────────┘
          │  SchedulerOutputs
┌─────────▼───────────────────────────────────┐
│           Worker (per GPU)                   │
│  ┌──────────────────────────────────────────┐│
│  │         ModelRunner                      ││
│  │  prepare_input → execute_model           ││
│  │  (CUDAGraph replay in decode)            ││
│  └──────────────────────────────────────────┘│
│  ┌──────────────────────────────────────────┐│
│  │         CacheEngine                      ││
│  │  KV Cache blocks (GPU/CPU)               ││
│  └──────────────────────────────────────────┘│
└─────────────────────────────────────────────┘
```

### 8.2 请求流转流程

**Step 1：请求接入**

HTTP 请求到达 `AsyncLLMEngine`，转换为 `SequenceGroup`（含 1 个或多个 `Sequence`，sampling params，arrival time）。

**Step 2：Scheduler 调度**

`Scheduler.schedule()` 在每个 step 执行：

1. **Prefill 队列**（`waiting`）：按 FCFS 顺序，为请求分配 KV Cache block（通过 `BlockSpaceManager`），加入 `running` 队列
2. **Preemption**：若 GPU 内存不足，将低优先级请求的 KV Cache swap 到 CPU（swap out）或直接 recompute
3. **Decode 队列**（`running`）：已完成 prefill 的请求继续 decode，分配新 block（按需）

输出为 `SchedulerOutputs`，包含本次 step 要执行的 `seq_group_metadata_list`。

**Step 3：Worker 执行**

`Worker.execute_model()`：

- `ModelRunner.prepare_inputs()`：将 token IDs、position IDs、KV Cache block table 打包为 GPU tensor
- Prefill 请求：执行完整前向（variable-length batching，使用 `flash_attn_varlen`）
- Decode 请求：执行单 token 前向（若 batch shape 固定则 CUDAGraph replay）
- 输出 logits，采样得到 next token

**Step 4：采样与输出**

`Sampler` 根据 `SamplingParams`（temperature、top-p、top-k）执行采样，产出 `SamplerOutput`，更新各 `Sequence` 的 token 列表。

**Step 5：结束判断**

若 `Sequence` 生成 EOS token 或达到 `max_tokens`，从 `running` 移至 `finished`，释放其 KV Cache block。

**Step 6：结果返回**

`AsyncLLMEngine` 将生成的 token 流式或批量返回给上层 API。

---

## 9. KV Cache 空间计算

### 9.1 单 token 的 KV Cache 大小

Transformer 每层存储 K 和 V 各一个向量，对于 GQA 模型：

$$ \text{KV per token per layer} = 2 \times N_{kv\_heads} \times d_{head} \times \text{dtype\_bytes} $$

其中：

- $N_{kv_heads}$：KV head 数量（GQA 下 $< N_{heads}$，MQA 下 $= 1$）
- $d_{head} = d_{model} / N_{heads}$：每个 head 的维度
- $\text{dtype\_bytes}$：FP16 = 2 bytes，FP8 = 1 byte，INT8 = 1 byte

整个模型（$L$ 层）单 token 的 KV Cache：

$$ \text{KV per token} = 2 \times N_{kv\_heads} \times d_{head} \times L \times \text{dtype\_bytes} $$

### 9.2 具体示例：LLaMA-3 8B

|参数|数值|
|---|---|
|$L$（层数）|32|
|$N_{heads}$|32|
|$N_{kv_heads}$（GQA）|8|
|$d_{model}$|4096|
|$d_{head}$|128|
|dtype|FP16（2 bytes）|

$$ \text{KV per token} = 2 \times 8 \times 128 \times 32 \times 2 = 131{,}072 \text{ bytes}$$

若同时服务 $B$ 个请求，每个请求最大序列长度 $S$：

$$ \text{Total KV Cache} = B \times S \times 131{,}072 \text{ bytes} $$

例：$B=100$，$S=4096$：$\frac{100 \times 4096 \times 131{,}072 \text{ bytes}} {10^9}  \approx 53.69 \text{ GB}$

### 9.3 vLLM 的 Block 管理

vLLM 将 KV Cache 切分为大小固定的 block（默认 `block_size = 16` tokens）：

$$ \text{block size (bytes)} = 2 \times N_{kv\_heads} \times d_{head} \times L \times \text{dtype\_bytes} \times \text{block\_size\_tokens} $$

GPU KV Cache 总 block 数由启动时根据剩余 GPU 显存自动计算：

$$ N_{\text{blocks}} = \left\lfloor \frac{\text{GPU Memory} \times \text{gpu\_memory\_utilization} - \text{Model Weights} - \text{Activations}}{\text{block size (bytes)}} \right\rfloor $$

`gpu_memory_utilization` 默认 0.90，即预留 10% 显存供 activation 等使用。

---

## 10. Prefix Cache 介绍

### 10.1 动机

多个请求往往共享相同前缀（system prompt、few-shot 示例、长文档），对这些共享前缀重复计算 KV Cache 浪费计算与内存。

### 10.2 核心机制

Prefix Cache（在 SGLang 中称为 RadixAttention）维护一棵以 token sequence 为 key 的前缀树（Radix Tree / Trie），每个节点对应一段 token 序列及其已计算的 KV Cache block：

```
Root
 ├── [sys_prompt_1] → KV block A
 │     ├── [user_turn_1a] → KV block B
 │     └── [user_turn_1b] → KV block C
 └── [sys_prompt_2] → KV block D
```

新请求到来时：

1. 在 Radix Tree 中查找最长匹配前缀
2. 命中部分的 KV block 直接复用，无需重新计算
3. 未命中的 suffix 进行正常 prefill
4. 新生成的 KV block 插入树中以备后续复用

### 10.3 Block 引用计数与驱逐

KV block 使用引用计数管理生命周期。无活跃请求引用的 block 成为驱逐候选，采用 LRU（Least Recently Used）或类 LRU 策略在内存压力时释放。

### 10.4 Hash Key 设计

block 的 cache key 为其覆盖的 token ID 序列的 hash（通常为 SHA256 或 xxHash），保证不同请求的相同 token 序列命中同一 block。

### 10.5 效果

- **TTFT 降低**：共享前缀无需重新 prefill，对 system prompt 较长的场景（如 RAG、agent）效果显著
- **吞吐提升**：节省的 prefill 计算可服务更多请求
- **内存复用**：多请求共享同一物理 KV block，节省 GPU 内存

典型场景下（system prompt 占 80% token），TTFT 可降低 40–80%。

---

## 11. vLLM V0 / V1 比较

### 11.1 架构差异概览

|维度|V0|V1|
|---|---|---|
|调度架构|单进程，CPU 端调度 + GPU 执行耦合|解耦：Scheduler 独立进程，异步|
|KV Cache 管理|BlockSpaceManager V1（链表）|BlockSpaceManager V2（更精细的 prefix caching）|
|Prefix Caching|可选，基础实现|默认开启，Radix Tree + hash-based|
|Chunked Prefill|支持|增强，与 decode 混合调度更优|
|多模态支持|基础|原生设计支持|
|Worker 通信|ZMQ|改进的 IPC 机制|
|CUDA Graph|decode 阶段|扩展覆盖范围|
|Tokenization|在线|支持异步离线 tokenization|

### 11.2 V1 核心改进

**异步调度（Async Scheduling）**：V1 将 Scheduler 从 GPU 执行关键路径解耦，Scheduler 在 GPU 执行第 $t$ 步时预先调度第 $t+1$ 步，消除 CPU 调度延迟对 GPU 的阻塞。

**BlockSpaceManager V2**：支持 prefix caching 的 block 引用计数与 LRU 驱逐，block 可被多请求共享，memory 利用率更高。

**统一 Memory Pool**：V1 引入统一的 KV Cache 内存池，支持跨请求 block 复用而不受序列边界限制。

### 11.3 性能对比

在 prefix cache hit rate 高的场景（如 chatbot 的 system prompt），V1 相较 V0 TTFT 可降低 30–50%；在 decode-heavy 场景吞吐提升约 10–20%。

---

## 12. TP 下不同模块如何汇总结果

### 12.1 MLP 模块

以两层 FFN（$Y = \text{GeLU}(XW_1)W_2$）为例，TP 的标准切分方式：

**第一层（Column Parallel）**：$W_1$ 按列切分为 $[W_1^{(1)} | W_1^{(2)} | \cdots | W_1^{(N)}]$

- 各卡各自完整输入 $X$（无需通信）
- 各卡计算 $H^{(i)} = \text{GeLU}(X W_1^{(i)})$
- 各卡持有 $H$ 的列分片，无需通信

**第二层（Row Parallel）**：$W_2$ 按行切分为 $\begin{bmatrix}W_2^{(1)} \ W_2^{(2)} \ \vdots\end{bmatrix}$

- 各卡计算局部结果 $Y^{(i)} = H^{(i)} W_2^{(i)}$（各卡持有 $H$ 的对应列分片）
- **All-Reduce**（sum）：$Y = \sum_i Y^{(i)}$

### 12.2 Attention 模块

$W_Q$、$W_K$、$W_V$ 按 head 维度列切分，各卡持有若干完整 head 的 Q、K、V 投影，独立计算注意力；$W_O$ 按行切分，attention 输出经 **All-Reduce** 汇总。

### 12.3 Embedding 与 LM Head

**Vocab Parallel Embedding**：词表按 vocab 维度切分，各卡持有部分词的 embedding；lookup 后经 **All-Reduce** 汇总（仅对应 token 有值，其余为 0）。

**LM Head**：对应列切分，各卡计算部分 vocab 的 logit，收集全量 logit 时需 **All-Gather**。

### 12.4 通信汇总

|模块|通信原语|通信量（per token）|
|---|---|---|
|MLP 第二层|All-Reduce|$d_{model} \times 2 \times (N-1)/N$ bytes|
|Attention $W_O$|All-Reduce|$d_{model} \times 2 \times (N-1)/N$ bytes|
|LM Head|All-Gather|$V \times 2$ bytes（$V$ 为 vocab size）|

每个 Transformer block 共 2 次 All-Reduce，TP 要求节点内 NVLink 带宽（A100 600 GB/s，H100 900 GB/s）以隐藏通信延迟。

---

## 13. Multi-Head Attention 口述

Multi-Head Attention 的核心思想是让模型在多个"表示子空间"中并行捕捉序列不同位置之间的依赖关系。

**输入投影**：给定输入序列 $X \in \mathbb{R}^{N \times d_{model}}$，通过三组线性投影得到 Query、Key、Value：

$$ Q = X W_Q, \quad K = X W_K, \quad V = X W_V $$

其中 $W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d_{model}}$。

**拆分到多头**：将 $Q$、$K$、$V$ 按 head 维度切分为 $h$ 份，第 $i$ 个 head 对应：

$$ Q_i = Q[\cdot, i \cdot d_h : (i+1) \cdot d_h], \quad d_h = d_{model} / h $$

**单头注意力**：

$$ \text{Attention}(Q_i, K_i, V_i) = \text{Softmax}!\left(\frac{Q_i K_i^T}{\sqrt{d_h}}\right) V_i $$

除以 $\sqrt{d_h}$ 是为了防止点积值随维度增大而方差过大，导致 Softmax 进入饱和区梯度消失。

**拼接与输出投影**：

$$ \text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O $$

**计算复杂度**：$O(N^2 d)$，其中 $N^2$ 来自 $QK^T$ 的 token-by-token 内积矩阵。这也是 long-context 场景的主要计算瓶颈，FlashAttention、GQA、MQA、MLA 均针对此展开优化。

**因果掩码（Causal Mask）**：decoder-only 模型中，在 $QK^T$ 上叠加下三角掩码，使位置 $t$ 的 token 仅能 attend 到 $\leq t$ 的位置。

---

## 14. Git、PyTorch 相关问题

### 14.1 Git 常见操作

**撤销操作**：

```bash
git reset --soft HEAD~1   # 撤销 commit，保留 staged 变更
git reset --mixed HEAD~1  # 撤销 commit，变更回到 unstaged
git reset --hard HEAD~1   # 撤销 commit，丢弃所有变更
git revert <commit>       # 生成新 commit 以撤销指定 commit（安全，保留历史）
```

**分支管理**：

```bash
git checkout -b feature/xxx   # 创建并切换分支
git merge --no-ff feature/xxx # 非快进合并，保留 merge commit
git rebase main               # 将当前分支变基到 main 顶端
git cherry-pick <commit>      # 摘取单个 commit 到当前分支
```

**stash**：

```bash
git stash push -m "WIP: xxx"  # 保存未提交变更
git stash pop                  # 恢复并删除最近 stash
git stash apply stash@{0}     # 恢复但保留 stash
```

**查看历史**：

```bash
git log --oneline --graph --all  # 图形化查看所有分支历史
git diff HEAD~3 HEAD -- file.cpp  # 查看指定文件在指定范围的变更
git blame file.cpp                # 逐行查看最后修改者
```

### 14.2 PyTorch 常见问题

**Tensor 操作**：

```python
# view vs reshape
x = torch.randn(4, 6)
y = x.view(2, 12)       # 要求 contiguous，共享内存
z = x.reshape(2, 12)    # 自动处理非 contiguous，可能拷贝

# contiguous 检查
x.is_contiguous()
x.contiguous()          # 强制 contiguous

# 原地操作（避免不必要内存分配）
x.add_(1)   # in-place
x += 1      # 等价
```

**梯度相关**：

```python
# 禁用梯度（推理阶段必须）
with torch.no_grad():
    output = model(input)

# detach（从计算图中分离，但保留 tensor 值）
loss = criterion(output.detach(), target)

# register_hook（调试梯度）
x.register_hook(lambda grad: print(grad.norm()))
```

**设备管理**：

```python
device = torch.device("cuda:0")
model = model.to(device)
input = input.to(device, non_blocking=True)  # 异步传输（需 pin_memory）
```

**数据类型转换**：

```python
x = x.half()     # FP32 → FP16
x = x.bfloat16() # FP32 → BF16
x = x.float()    # → FP32
```

**DataLoader 常见参数**：

```python
DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,       # 多进程预加载
    pin_memory=True,     # 锁页内存，加速 CPU→GPU 传输
    prefetch_factor=2,   # 每个 worker 预取 batch 数
    persistent_workers=True  # 避免 epoch 间重新创建进程
)
```

---

## 15. C++ 函数模板与内存泄露规避

### 15.1 函数模板

```cpp
// 基本函数模板
template <typename T>
T max_val(T a, T b) {
    return (a > b) ? a : b;
}

// 多类型参数
template <typename T, typename U>
auto add(T a, U b) -> decltype(a + b) {
    return a + b;
}

// C++14 返回类型推导
template <typename T, typename U>
auto multiply(T a, U b) {
    return a * b;
}
```

**模板特化**：

```cpp
// 全特化
template <>
const char* max_val<const char*>(const char* a, const char* b) {
    return strcmp(a, b) > 0 ? a : b;
}

// 偏特化（仅适用于类模板）
template <typename T>
class Container<T*> { /* 指针特化 */ };
```

**SFINAE（C++17 if constexpr 更简洁）**：

```cpp
// C++17
template <typename T>
void process(T val) {
    if constexpr (std::is_integral_v<T>) {
        // 整数分支
    } else {
        // 其他分支
    }
}

// Concepts（C++20）
template <std::integral T>
T gcd(T a, T b) {
    return b == 0 ? a : gcd(b, a % b);
}
```

**可变参数模板**：

```cpp
template <typename... Args>
void log(Args&&... args) {
    (std::cout << ... << args) << '\n';  // C++17 fold expression
}
```

### 15.2 内存泄露规避

**原则：不使用裸指针管理所有权**

```cpp
// 错误示范
void bad() {
    int* p = new int[100];
    if (some_condition) return;  // 泄漏
    delete[] p;
}

// 正确：unique_ptr（独占所有权）
void good_unique() {
    auto p = std::make_unique<int[]>(100);
    if (some_condition) return;  // 自动释放
}

// 正确：shared_ptr（共享所有权）
std::shared_ptr<Resource> get_resource() {
    return std::make_shared<Resource>();
}

// 注意：避免循环引用（用 weak_ptr 打破）
struct Node {
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev;  // weak_ptr 不增加引用计数
};
```

**RAII（Resource Acquisition Is Initialization）**：

```cpp
class CudaMemory {
    void* ptr_ = nullptr;
public:
    explicit CudaMemory(size_t bytes) {
        cudaMalloc(&ptr_, bytes);
    }
    ~CudaMemory() {
        if (ptr_) cudaFree(ptr_);
    }
    CudaMemory(const CudaMemory&) = delete;             // 禁止拷贝
    CudaMemory& operator=(const CudaMemory&) = delete;
    CudaMemory(CudaMemory&& o) noexcept : ptr_(std::exchange(o.ptr_, nullptr)) {}
    void* get() const { return ptr_; }
};
```

**常见泄漏场景**：

|场景|规避方式|
|---|---|
|异常路径跳过 delete|RAII / `unique_ptr`|
|循环引用 shared_ptr|`weak_ptr` 打破环|
|C 接口资源（FILE*、CUDA handle）|自定义 Deleter 的 `unique_ptr`|
|容器持有裸指针|容器持有 `unique_ptr`|
|全局/静态对象持有资源|程序退出时显式清理或使用 RAII 封装|

**工具辅助**：

```bash
valgrind --leak-check=full ./program         # 检测内存泄漏
AddressSanitizer:  -fsanitize=address        # 编译期注入，运行时检测
```

---

## 16. Torch Compiler 与 CUDA Graph

### 16.1 torch.compile

`torch.compile`（PyTorch 2.0+）是 PyTorch 的图编译入口，底层由三个组件构成：

**TorchDynamo**：Python 字节码级别的图捕获器。通过 `eval_frame` hook 拦截 Python 执行，将可追踪的计算片段提取为 FX Graph（`torch.fx.GraphModule`），保留动态控制流对应的 "guard"（守卫条件）。

**AOT Autograd（Ahead-of-Time Autograd）**：将前向与后向计算展开为联合图（joint graph），允许后端对整个前反向进行整体优化。

**后端编译器（Backend）**：

|后端|说明|
|---|---|
|`inductor`（默认）|生成 OpenAI Triton kernel，支持 kernel fusion|
|`cudagraphs`|自动录制 CUDA Graph|
|`eager`|不优化，用于调试|
|`onnxrt`|导出 ONNX 并用 ONNX Runtime 执行|

```python
import torch

model = MyModel().cuda()

# 编译（首次调用时触发追踪与编译，有启动延迟）
compiled_model = torch.compile(model, backend="inductor", mode="max-autotune")

# 之后的调用使用编译后的 kernel
output = compiled_model(input)
```

**mode 选项**：

- `"default"`：平衡编译时间与运行时性能
- `"reduce-overhead"`：优先减少 Python/框架 overhead（等效于 CUDA Graph）
- `"max-autotune"`：穷举最优 kernel 配置，编译时间最长，运行时最快

### 16.2 CUDA Graph 与 torch.compile 的关系

`torch.compile(mode="reduce-overhead")` 内部自动启用 CUDA Graph 捕获，无需手动管理 `torch.cuda.CUDAGraph`。其工作原理与手动 CUDA Graph 相同：首次执行时录制，后续执行直接 `replay`。

差异：

|维度|手动 CUDA Graph|torch.compile|
|---|---|---|
|控制粒度|精细，可手动控制录制范围|自动，由编译器决定|
|与 kernel fusion 配合|需手动融合|Inductor 自动 fusion + graph|
|动态 shape 支持|需手动处理多个图实例|部分支持（dynamic=True）|
|调试难度|中|较高（多层抽象）|

---

## 17. PTX（Parallel Thread Execution）

### 17.1 定义

PTX 是 NVIDIA 定义的虚拟 ISA（Instruction Set Architecture），位于 CUDA C++ 与最终硬件机器码（SASS, Streaming ASSembler）之间。

```
CUDA C++ (.cu)
     │  nvcc
     ▼
   PTX (.ptx)       ← 虚拟 ISA，可移植
     │  ptxas
     ▼
   SASS (.cubin)     ← 硬件相关机器码
```

### 17.2 PTX 的特点

- **可移植性**：PTX 不绑定具体 GPU 架构，NVIDIA driver 在运行时（JIT）将 PTX 编译为目标架构的 SASS
- **虚拟寄存器**：PTX 使用无限虚拟寄存器（`.reg .f32 %f<N>`），由 ptxas 分配实际物理寄存器
- **抽象线程模型**：与 CUDA 线程模型（`%tid.x`、`%ctaid.x`、`%ntid.x`）直接对应

### 17.3 PTX 代码结构示例

```ptx
.version 7.0
.target sm_80
.address_size 64

.visible .entry vector_add(
    .param .u64 param0,  // A
    .param .u64 param1,  // B
    .param .u64 param2,  // C
    .param .u32 param3   // N
)
{
    .reg .u64 %rd<4>;
    .reg .f32 %f<3>;
    .reg .u32 %r<4>;
    .reg .pred %p<1>;

    ld.param.u64   %rd0, [param0];  // A 指针
    ld.param.u64   %rd1, [param1];  // B 指针
    ld.param.u64   %rd2, [param2];  // C 指针
    ld.param.u32   %r0,  [param3];  // N

    mov.u32        %r1, %tid.x;
    mov.u32        %r2, %ctaid.x;
    mov.u32        %r3, %ntid.x;
    mad.lo.u32     %r1, %r2, %r3, %r1;  // global index

    setp.ge.u32    %p0, %r1, %r0;   // index >= N?
    @%p0 bra       END;

    cvt.u64.u32    %rd3, %r1;
    shl.b64        %rd3, %rd3, 2;   // * 4 bytes

    add.u64        %rd0, %rd0, %rd3;
    add.u64        %rd1, %rd1, %rd3;
    add.u64        %rd2, %rd2, %rd3;

    ld.global.f32  %f0, [%rd0];
    ld.global.f32  %f1, [%rd1];
    add.f32        %f2, %f0, %f1;
    st.global.f32  [%rd2], %f2;

END:
    ret;
}
```

### 17.4 内联 PTX（Inline PTX）

在 CUDA C++ 中通过 `asm volatile` 插入 PTX 指令，用于访问高性能原语：

```cpp
// warp-level shuffle（PTX 原语）
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float tmp;
        asm volatile(
            "shfl.sync.down.b32 %0, %1, %2, 0x1f, 0xffffffff;"
            : "=f"(tmp) : "f"(val), "r"(offset)
        );
        val += tmp;
    }
    return val;
}

// 128-bit 向量化内存访问
__device__ void load128(float4& dst, const float* src) {
    asm volatile(
        "ld.global.v4.f32 {%0,%1,%2,%3}, [%4];"
        : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w)
        : "l"(src)
    );
}
```

### 17.5 PTX 在推理优化中的应用场景

|场景|PTX 指令|
|---|---|
|Warp reduce / scan|`shfl.sync.*`|
|向量化 load/store|`ld.global.v4.f32`、`st.global.v4.f32`|
|原子操作|`atom.global.add.f32`|
|纹理/共享内存控制|`cp.async.ca.shared.global`（A100 异步拷贝）|
|特殊数学函数|`sin.approx.f32`、`sqrt.approx.f32`|
|访问特殊寄存器|`mov.u64 %rd0, %clock64`（性能计数器）|

### 17.6 PTX vs SASS

|维度|PTX|SASS|
|---|---|---|
|可读性|高（人可读）|低（近机器码）|
|可移植性|跨架构（sm_80 PTX 可在 sm_90 上运行）|架构绑定|
|性能|JIT 编译可能欠优化|最终执行性能|
|获取方式|`nvcc -ptx`、`cuobjdump --dump-ptx`|`cuobjdump --dump-sass`|
