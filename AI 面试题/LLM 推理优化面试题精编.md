
> 综合来源：DeepLearing-Interview-Awesome-2024 / LLM-Interview-Questions-and-Answers-Hub / llms-interview-questions / LLM-interview（手撕题方向） 覆盖方向：Transformer 架构 · 注意力机制 · KV Cache · Flash Attention · 量化 · Batching · 解码策略 · MoE · 并行化 · 推理系统

---

## 一、Transformer 架构基础

---

### 1. Transformer 的整体架构是什么？现代 LLM 为何普遍采用 Decoder-Only 结构？

**答：**

原始 Transformer（Vaswani et al., 2017）由 Encoder 和 Decoder 两部分构成，分别用于理解输入序列和生成输出序列。

现代 LLM（GPT 系列、LLaMA、Qwen 等）普遍采用 **Causal Decoder-Only** 结构，原因如下：

|对比维度|Encoder-Decoder|Decoder-Only|
|---|---|---|
|预训练目标|Masked LM + Causal LM|Causal LM（Next Token Prediction）|
|推理阶段|需两次前向传播|单次自回归前向传播|
|上下文利用|双向（编码器）+ 单向（解码器）|单向（因果掩码）|
|长文本扩展|复杂|直接扩展 Context Window|
|工程实现复杂度|高|低|

Decoder-Only 的核心优势在于：**预训练目标（Next Token Prediction）与推理目标完全对齐**，训练数据利用率高，且推理时 KV Cache 缓存策略天然适配自回归生成。

---

### 2. Self-Attention 的计算流程及时间复杂度是什么？

**答：**

给定输入序列 $X \in \mathbb{R}^{n \times d_{\text{model}}}$，Self-Attention 的计算步骤如下：

**Step 1：线性投影**

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

其中 $W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$。

**Step 2：Scaled Dot-Product Attention**

$$\text{Attention}(Q, K, V) = \text{softmax}!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

缩放因子 $\frac{1}{\sqrt{d_k}}$ 的作用是：防止 $d_k$ 较大时内积值过大，导致 softmax 进入梯度极小的饱和区。

**时间 / 空间复杂度：**

|操作|时间复杂度|空间复杂度|
|---|---|---|
|$QK^\top$ 矩阵乘法|$O(n^2 d_k)$|$O(n^2)$|
|softmax|$O(n^2)$|$O(n^2)$|
|$\text{Attn} \cdot V$|$O(n^2 d_k)$|$O(nd_k)$|
|**总计**|$O(n^2 d)$|$O(n^2)$|

注意力矩阵 $A \in \mathbb{R}^{n \times n}$ 的 $O(n^2)$ 空间是长上下文推理中显存瓶颈的根本来源，也是 Flash Attention 要解决的核心问题。

---

### 3. Multi-Head Attention（MHA）为何比单头注意力更优？其计算方式是什么？

**答：**

单头注意力仅在一个表示子空间中计算关系，表达能力受限。多头注意力并行在 $h$ 个子空间中捕捉不同类型的依赖关系（局部 / 全局、句法 / 语义等）。

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q,\ KW_i^K,\ VW_i^V)$$

其中 $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$，$d_k = d_{\text{model}} / h$，$W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$。

参数量与单头相比**不变**（因 $d_k$ 等比缩小），但表达能力显著增强。

---

### 4. Causal Masking（因果掩码）的作用与实现方式？

**答：**

在 Decoder 的自回归生成中，位置 $i$ 的 token 只能 attend 到位置 $\leq i$ 的 token，以防止信息泄露（看到未来 token）。

实现方式：在 softmax 之前，对注意力得分矩阵 $S = QK^\top / \sqrt{d_k}$ 施加上三角掩码：

$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \ -\infty & \text{if } i < j \end{cases}$$

$$\text{Output} = \text{softmax}(S + M) \cdot V$$

$-\infty$ 经过 softmax 后趋近于 0，等价于对未来 token 的注意力权重清零。

训练时这一操作使整个序列可以**并行计算**（所有位置同时前向传播），而非逐 token 自回归，这是 Transformer 相对 RNN 的核心训练效率优势。

---

### 5. Positional Encoding 的作用？RoPE 与绝对位置编码的区别？

**答：**

Self-Attention 本身对序列顺序**不敏感**（置换等变），需引入位置编码赋予模型位置感知能力。

**绝对位置编码（Sinusoidal / Learned）：**

$$PE_{(pos, 2i)} = \sin!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos!\left(\frac{pos}{10000^{2i/d}}\right)$$

直接叠加到 Embedding 上，不具备长度外推能力。

**RoPE（Rotary Position Embedding，现代 LLM 主流）：**

核心思想：将位置信息编码为**旋转矩阵**，使得内积 $q_m^\top k_n$ 只依赖于相对位置 $m - n$：

$$\langle f_q(x_m, m),\ f_k(x_n, n) \rangle = g(x_m, x_n, m - n)$$

优势：天然支持相对位置建模，通过频率缩放（如 YaRN）可外推到训练时更长的上下文。

---

### 6. LayerNorm 在 Transformer 中的作用？Pre-Norm 与 Post-Norm 的区别？

**答：**

LayerNorm 对每个样本的特征维度归一化，稳定训练过程中的激活分布：

$$\text{LN}(x) = \frac{x - \mu}{\sigma + \epsilon} \cdot \gamma + \beta$$

其中 $\mu$、$\sigma$ 为当前 token 特征维度的均值和标准差，$\gamma$、$\beta$ 为可学习仿射参数。

|方案|位置|训练稳定性|性能|
|---|---|---|---|
|Post-Norm（原始论文）|残差加法之后|较差，需 warmup|略高|
|Pre-Norm（现代 LLM 主流）|残差加法之前|好，梯度更稳定|略低但可接受|
|RMSNorm（LLaMA 系列）|Pre-Norm 变体，去掉均值项|好，计算更快|接近 LN|

现代 LLM 几乎全部使用 **Pre-RMSNorm**，去掉均值项后计算量减少约 $1/3$，且实验表明精度损失可忽略。

---

## 二、注意力机制优化：MQA / GQA / MLA

---

### 7. MHA / MQA / GQA 三者的区别与推理阶段的显存影响？

**答：**

三者的核心差异在于 KV head 的数量：

|机制|Q heads|KV heads|KV Cache 大小|代表模型|
|---|---|---|---|---|
|MHA|$h$|$h$|$2 \cdot h \cdot d_k \cdot n$|GPT-2、BERT|
|MQA|$h$|1|$2 \cdot 1 \cdot d_k \cdot n$|PaLM、早期 Falcon|
|GQA|$h$|$g$（$1 < g < h$）|$2 \cdot g \cdot d_k \cdot n$|LLaMA-3、Qwen2、Mistral|

**GQA（Grouped-Query Attention）** 将 $h$ 个 Query head 分为 $g$ 组，每组共享一对 KV head：

$$\text{KV Cache 节省比} = \frac{h}{g}$$

以 LLaMA-3-70B（$h=64$，$g=8$）为例，KV Cache 降低至 MHA 的 $1/8$。

**推理影响：** KV Cache 的加载是 Decode 阶段内存带宽瓶颈的主因，GQA 直接降低每步 Decode 的数据搬运量，显著降低延迟。

---

### 8. DeepSeek-V2/V3 引入的 MLA（Multi-head Latent Attention）是什么？

**答：**

MLA 通过**低秩联合压缩** KV 矩阵，将 KV Cache 的存储量从 $O(h \cdot d_k)$ 压缩至 $O(d_c)$（$d_c \ll h \cdot d_k$）。

核心思想：不直接缓存 $K$ 和 $V$，而是缓存一个低维潜变量 $c_t^{KV}$，在需要时通过投影矩阵恢复：

$$c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}$$

$$K_t = W^{UK} c_t^{KV}, \quad V_t = W^{UV} c_t^{KV}$$

相比 MHA，MLA 的 KV Cache 显存减少 **5~13 倍**（取决于配置），同时模型能力不下降。

---

## 三、KV Cache

---

### 9. KV Cache 的原理是什么？解决了什么问题？

**答：**

**问题根源：** 自回归解码中，生成第 $t$ 个 token 时，需要对前 $t-1$ 个 token 重新计算 $K$、$V$，导致计算量随序列长度线性增长：

$$\text{总计算量} = \sum_{t=1}^{T} O(t \cdot d) = O(T^2 d)$$

**KV Cache 方案：** 将历史 token 的 $K$、$V$ 矩阵缓存到 GPU HBM（显存），每步 Decode 只计算当前新 token 的 $Q$、$K$、$V$，并与缓存拼接后计算注意力。

$$\text{Step } t: K_{\text{cache}} \leftarrow [K_{\text{cache}};\ k_t], \quad V_{\text{cache}} \leftarrow [V_{\text{cache}};\ v_t]$$

**显存消耗公式（MHA 模型）：**

$$\text{KV Cache} = 2 \times p_a \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times L_{\text{seq}} \times B$$

以 Llama-2-7B（32 层，32 头，$d_{\text{head}}=128$，FP16）、序列长 10000、Batch=1 为例：

$$= 2 \times 2 \times 32 \times 32 \times 128 \times 10000 \approx 5\text{ GB}$$

约为模型权重（13 GB）的 38%，是长上下文推理的主要显存压力来源。

---

### 10. PagedAttention 是什么？它如何解决 KV Cache 的显存碎片问题？

**答：**

**问题：** 传统 KV Cache 为每个请求预分配一块**连续显存**（按最大序列长度），导致：

- **内部碎片**：实际序列未用满预分配空间
- **外部碎片**：不同请求的内存块无法被其他请求利用，GPU 利用率低

**PagedAttention（vLLM 提出）：** 借鉴操作系统虚拟内存的分页思想：

- 将 KV Cache 切分为固定大小的 **Block**（如每块 16 个 token）
- 用**逻辑 Block 表**到**物理 Block**的映射管理内存
- 允许同一请求的 KV Cache 存储在**不连续的物理内存块**中
- 多个请求可**共享**相同前缀的 KV Cache 块（Prefix Sharing）

效果：显存利用率从传统方案的 20%~40% 提升至 96% 以上（vLLM 原始论文数据）。

---

### 11. KV Cache 量化的意义与主流方案？

**答：**

**意义：** Decode 阶段是**内存带宽受限**（Memory-Bound）的操作，非计算受限（Compute-Bound）。KV Cache 量化减少每步 Decode 时从 HBM 搬运到 SRAM 的数据量，直接降低延迟。

**主流量化方案对比：**

|方案|位宽|精度损失|特点|
|---|---|---|---|
|FP16（基准）|16-bit|无|TensorRT-LLM 默认|
|INT8|8-bit|极小|vLLM 0.3.0+ 支持 FP8 KV|
|INT4（KIVI）|4-bit|可接受|Key 按 channel，Value 按 token 量化|
|INT2|2-bit|明显下降|实验性，不建议生产使用|

**KIVI 关键发现：**

- Key 矩阵存在固定异常通道 → 采用**按通道（Per-Channel）量化**
- Value 矩阵异常值按 token 分布 → 采用**按 token（Per-Token）量化**

---

## 四、Flash Attention

---

### 12. Flash Attention 解决了什么问题？核心思路是什么？

**答：**

**问题：** 标准 Self-Attention 的计算步骤中，注意力矩阵 $A = \text{softmax}(QK^\top / \sqrt{d_k}) \in \mathbb{R}^{n \times n}$ 需要写入 GPU HBM（慢），再读回 SRAM（快）做 softmax，再写入 HBM，再读回做 $A \cdot V$。

反复的 HBM 读写产生大量 **IO 开销**，在长序列下成为瓶颈。时间复杂度为 $O(n^2)$ 的是 HBM 访问次数，而非计算量。

**Flash Attention 核心思路（Dao et al., 2022）：**

**Tiling（分块计算）+ Online Softmax（流式归一化）**，避免将完整注意力矩阵写入 HBM：

1. 将 $Q$、$K$、$V$ 切分为若干块（Block）
2. 在 SRAM 中逐块计算部分注意力得分
3. 利用 online softmax 技巧维护全局的归一化分母 $l$ 和最大值 $m$，合并各块的结果：

$$m^{\text{new}} = \max(m^{\text{old}},\ \max_j s_{ij})$$

$$l^{\text{new}} = e^{m^{\text{old}} - m^{\text{new}}} \cdot l^{\text{old}} + \sum_j e^{s_{ij} - m^{\text{new}}}$$

4. 输出 $O$ 在 SRAM 内累积，最终仅写入 HBM 一次

**结果：**

- HBM 读写次数从 $O(n^2)$ 降至 $O(n)$（对 $A$ 矩阵不落盘）
- 显存从 $O(n^2)$ 降至 $O(n)$（无需存储完整 $A$ 矩阵）
- 速度提升 2~4 倍（A100 上，序列长度 1k~16k）
- Flash Attention 2/3 进一步优化 Warp 调度与 FP8 支持

---

## 五、推理系统与 Batching

---

### 13. 推理延迟（Latency）与吞吐量（Throughput）的定义及其 trade-off？

**答：**

|指标|定义|典型单位|
|---|---|---|
|Latency（延迟）|从请求提交到完整响应生成的时间|ms / s|
|TTFT|Time To First Token，Prefill 阶段完成时间|ms|
|TPOT|Time Per Output Token，每个 token 的生成时间|ms/token|
|Throughput（吞吐量）|单位时间内处理的 token 或请求数|tokens/s 或 req/s|

**Trade-off：**

- **增大 Batch Size** → 提高吞吐（GPU 利用率高）→ 延迟增大（请求需等待凑批）
- **减小 Batch Size** → 降低延迟 → 吞吐降低，GPU 利用率不足
- **Decode 阶段**为 Memory-Bound，增大 Batch 可提升 FLOPS 利用率而几乎不增加延迟
- **Prefill 阶段**为 Compute-Bound，Batch 增大会显著影响 TTFT

---

### 14. Static Batching vs. Continuous Batching（动态批处理），二者的区别与优劣？

**答：**

**Static Batching：**

- 等待一批请求全部到达后一起处理
- 所有请求必须等批次中**最长序列**生成完毕才能释放
- GPU 长时间处于空闲（等待短序列生成完成后等长序列）
- 问题：**尾部序列**导致 GPU 利用率低

**Continuous Batching（Orca / vLLM）：**

- **Iteration-level scheduling**：在每个 Decode 步（每生成一个 token）之后，将已完成的序列移出批次，将等待中的新请求插入
- GPU 始终处于满负荷状态
- 吞吐提升可达 **23× 以上**（Orca 论文数据）

```
Static:  [Req1────────┤ Req2────┤ Req3─────────┤]  (GPU idle on short reqs)
                       ↑ all wait for Req3

Continuous: [Req1, Req2, Req3] → step → [done: Req2] → [insert Req4] → ...
```

---

### 15. Prefill 与 Decode 两个阶段的计算特性有何本质区别？

**答：**

|阶段|输入|并行性|瓶颈类型|
|---|---|---|---|
|Prefill|完整 prompt（$n$ 个 token）|所有 token 并行计算|Compute-Bound（算力密集）|
|Decode|单个新 token（或少量）|自回归，串行|Memory-Bound（带宽密集）|

**Decode 阶段为 Memory-Bound 的原因：**

每生成一个 token，需要从 HBM 中加载：

- 全部模型权重：$\approx 2 \times \text{params}$ Bytes（FP16）
- 全部 KV Cache：$\approx 2 \times 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times L$

但实际的矩阵乘法规模极小（batch=1 时，GEMM 退化为 GEMV），GPU 算力远未打满。

这一特性决定了：**量化（减少数据搬运）和大 Batch（均摊权重加载开销）是 Decode 阶段最有效的优化手段。**

---

## 六、解码策略

---

### 16. Greedy Search、Beam Search、Temperature Sampling 的区别与适用场景？

**答：**

|策略|核心机制|优点|缺点|适用场景|
|---|---|---|---|---|
|Greedy Search|每步选 argmax token|快速、确定性|容易陷入局部最优，输出单调|快速摘要、结构化输出|
|Beam Search|维护 $k$ 条候选序列|输出质量高|计算量 $k$ 倍，多样性差|机器翻译、语音识别|
|Temperature Sampling|对 logits 缩放后采样|多样性可控|低温趋向 Greedy，高温随机|创意写作、对话|
|Top-K Sampling|只从概率最高的 K 个 token 中采样|避免极低概率 token|K 难以自适应|开放域生成|
|Top-P（Nucleus）Sampling|从累计概率 $\geq p$ 的最小集合中采样|自适应候选集大小|超参数敏感|通用生成场景|

**Temperature** $T$ 的作用：

$$P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- $T \to 0$：等价于 Greedy（输出确定性最强）
- $T = 1$：原始分布
- $T > 1$：分布趋于均匀（多样性增加，相关性下降）

---

### 17. Speculative Decoding（投机解码）的原理与加速效果？

**答：**

**核心动机：** Decode 阶段是 Memory-Bound 的，生成 1 个 token 与生成少量 token 所用时间几乎相同（显存读写是瓶颈，而非算力）。

**Speculative Decoding 流程：**

1. 用小模型（Draft Model，如 7B）快速连续生成 $\gamma$ 个候选 token（草稿）
2. 用大模型（Target Model，如 70B）对这 $\gamma$ 个 token **并行验证**（一次前向传播）
3. 根据接受概率判断是否采纳草稿 token，最终输出与大模型单独生成**分布等价**

**接受率 $\alpha$ 对加速比的影响：**

$$\text{加速比} \approx \frac{1 + \alpha + \alpha^2 + \cdots + \alpha^\gamma}{1 + \text{cost_ratio}(D, T)}$$

当 Draft 与 Target 分布接近时（$\alpha \to 1$），加速比可达 2~4×。

变体：

- **Medusa**：在目标模型上附加多个解码头，无需额外 Draft Model
- **EAGLE**：Draft 模型以 Target 的中间特征为输入，接受率更高

---

## 七、模型量化

---

### 18. 量化的基本原理？PTQ 与 QAT 的区别？

**答：**

量化将高精度浮点数（FP32/FP16）映射到低比特整数，基本公式（对称量化）：

$$X_{\text{int}} = \text{round}!\left(\frac{X}{S}\right), \quad S = \frac{\max(|X|)}{2^{b-1} - 1}$$

非对称量化（含零点 $Z$）：

$$X_{\text{int}} = \text{round}!\left(\frac{X}{S}\right) - Z, \quad S = \frac{\max(X) - \min(X)}{2^b - 1}$$

|方案|训练需求|精度|工程成本|
|---|---|---|---|
|PTQ（训练后量化）|无需重训，仅需校准数据|INT8 损失极小；INT4 需方法优化|低|
|QAT（量化感知训练）|需在训练中插入 fake quantization|最高|高|

---

### 19. 主流权重量化方案（GPTQ、AWQ、SmoothQuant）的核心思想？

**答：**

**GPTQ（OBD 二阶优化，INT4 W4A16）：**

基于 Optimal Brain Surgeon，逐层最小化量化误差：

$$\min_{\hat{W}} |WX - \hat{W}X|_2^2$$

通过 Hessian 矩阵的逆矩阵对量化误差进行补偿，每量化一个权重后更新其他权重，实现低精度下的高精度恢复。

**AWQ（激活感知权重量化，W4A16）：**

发现权重中只有 1% 的通道对激活影响巨大（显著权重），通过**缩放变换**保护这些通道：

$$\hat{W} = W \cdot \text{diag}(s)^{-1}, \quad \hat{x} = \text{diag}(s) \cdot x$$

不修改激活，量化精度优于 GPTQ，且无需 Hessian 计算。

**SmoothQuant（W8A8）：**

激活量化难点在于激活异常值（Outliers），通过通道级缩放将量化难度从激活迁移到权重：

$$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W^\top) = \hat{X} \hat{W}^\top$$

使激活和权重的量化难度均衡，在 INT8 精度下接近 FP16。

**量化精度对比（LLaMA-3-8B，参考数据）：**

|方案|显存（相对 FP16）|Perplexity 损失|
|---|---|---|
|W8A16|~50%|< 0.1|
|W4A16（AWQ）|~25%|< 0.5|
|W4A8（QoQ）|~25%，算力提升|~1.0|
|W2A16|~12.5%|明显下降|

---

### 20. INT8 和 FP8 的适用场景及硬件支持？

**答：**

|数据类型|硬件支持|适用场景|特点|
|---|---|---|---|
|INT8|Ampere（A100）+|W8A8，SmoothQuant|需要 scale 校准|
|FP8（E4M3/E5M2）|Hopper（H100）+|训练 + 推理均可|动态范围优于 INT8|
|INT4|Ampere+（via CUTLASS）|Weight-Only，W4A16|需 Dequant 计算|

V100 不支持 INT8/INT4 硬件加速；H100 FP8 Tensor Core 峰值算力为 FP16 的 2 倍。

工程实践：W4A16（AWQ/GPTQ）是当前**性价比最高**的推理部署方案，INT4 存储精度、FP16 计算精度，吞吐提升 2~3×，精度损失通常 < 1%。

---

## 八、Mixture of Experts（MoE）

---

### 21. MoE 架构的核心思想？推理阶段如何降低计算量？

**答：**

**核心思想：** 用多个专家网络（Expert FFN）替代单个 FFN，每个 token 仅激活其中 Top-$k$ 个专家，实现**参数量大而实际计算量小**的效果。

$$\text{MoE}(x) = \sum_{i=1}^{K} G(x)_i \cdot E_i(x), \quad G(x) = \text{TopK}(\text{softmax}(W_g x))$$

**典型配置（DeepSeek-V3）：**

|参数|值|
|---|---|
|总参数量|671B|
|每 token 激活参数|37B（Top-2 专家）|
|专家数量|256|
|路由方式|Auxiliary-loss-free 负载均衡|

**推理优势：** 671B 参数的模型只需 37B 的计算量，接近同算力规模的 Dense 模型速度，但知识容量大幅提升。

**工程挑战：**

- **专家并行（Expert Parallelism）：** 专家分布在多 GPU，需全互联通信（All-to-All）
- **负载不均衡：** 部分专家过热，需辅助损失或 Token Dropping 策略
- **显存占用：** 全部专家权重必须加载

---

## 九、并行化策略

---

### 22. Tensor Parallelism、Pipeline Parallelism、Data Parallelism 的区别？

**答：**

|并行策略|切分对象|通信需求|适用场景|
|---|---|---|---|
|Data Parallelism（DP）|数据（Batch）按 GPU 切分|All-Reduce 梯度|训练，模型放得下单 GPU|
|Tensor Parallelism（TP）|权重矩阵按行 / 列切分|All-Reduce 激活|单层过大，需多 GPU|
|Pipeline Parallelism（PP）|模型层按 GPU 切分|P2P（层间）|模型层数过多|
|Sequence Parallelism（SP）|序列维度切分|All-Gather|长上下文，Attention 过大|

**Tensor Parallelism 细节（Megatron-LM 方案）：**

MLP 中的两层线性：

$$Y = \text{GeLU}(XA),\quad Z = YB$$

将 $A$ 按列切分、$B$ 按行切分，各 GPU 并行计算后通过 All-Reduce 汇合，每个 Transformer 层引入 2 次 All-Reduce。

---

### 23. Kernel Fusion 的意义与实现方式？

**答：**

**问题：** GPU Kernel 的启动有固定开销（Launch Overhead）。多个逐元素操作（LayerNorm、GeLU、残差加法）若独立执行，每次都需要从 HBM 读数据 → 计算 → 写回 HBM，显存带宽被浪费。

**Kernel Fusion：** 将多个小 Kernel 合并为一个大 Kernel，数据在 SRAM（Shared Memory / Register）中流转，仅在最终步骤写回 HBM。

实现方式：

- **手写 CUDA Kernel**：最高性能，复杂度高
- **Triton**：Python DSL，自动生成 GPU 代码，Flash Attention 基于此实现
- **CUDA Graph**：将多次 Kernel 调用固化为一张执行图，消除启动开销

典型融合场景：`LayerNorm + Linear`、`Linear + GeLU + Linear`（FFN 融合）、`Softmax + Dropout`。

---

## 十、LoRA 与参数高效微调

---

### 24. LoRA 的核心原理？为何在推理优化中也相关？

**答：**

LoRA（Low-Rank Adaptation）假设微调时权重的更新矩阵 $\Delta W$ 本质上是低秩的：

$$W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d, k)$$

训练时冻结 $W$，仅训练低秩矩阵 $B$、$A$，参数量从 $d \times k$ 降至 $r(d+k)$。

**与推理的关系：**

- LoRA Adapter 在推理时可**合并回原权重**（$W' = W + BA$），零额外推理开销
- **多 LoRA 场景（如服务多租户）：** 可保持基础模型权重共享，仅切换低秩 Adapter，减少显存占用（相比为每个用户加载一个完整 fine-tuned 模型）

---

### 25. QLoRA 相比 LoRA 的改进？

**答：**

QLoRA = 4-bit 量化 + LoRA：

- 将基础模型权重量化为 NF4（4-bit NormalFloat）格式存储
- 引入 Double Quantization（对量化常数再量化）和 Paged Optimizer（CPU 显存换入换出）
- 显著降低微调所需显存：65B 模型可在单张 48GB A100 上微调

QLoRA 用于推理部署时，需将量化权重与 LoRA 分离存储，运行时先对权重 Dequant，再加上 LoRA，计算开销略高于 LoRA merge。

---

## 十一、LLM 推理系统与框架

---

### 26. vLLM、TensorRT-LLM、SGLang 的定位与核心差异？

**答：**

|框架|核心特性|适用场景|
|---|---|---|
|**vLLM**|PagedAttention、Continuous Batching、OpenAI 兼容接口|通用 LLM serving，入门首选|
|**TensorRT-LLM**|NVIDIA 官方，深度 Kernel 优化，FP8 / INT4 支持，Inflight Batching|生产环境高性能部署|
|**SGLang**|RadixAttention（前缀 KV Cache 共享）、结构化生成、高并发|Agent / RAG / Multi-turn 场景|
|**MLC-LLM**|跨平台（手机 / 浏览器），编译期优化|端侧部署|

---

### 27. 评估推理系统性能的核心指标有哪些？如何系统性地定位瓶颈？

**答：**

**核心指标：**

|指标|含义|
|---|---|
|TTFT|Time To First Token，反映 Prefill 性能|
|TPOT|Time Per Output Token，反映 Decode 性能|
|Throughput|tokens/s 或 req/s|
|P99 Latency|99 分位延迟，反映尾部抖动|
|GPU MFU|Model FLOPs Utilization，算力利用率|

**瓶颈定位流程：**

```
1. 监控 GPU 显存占用 → 是否 OOM？KV Cache 是否溢出？
2. 监控 GPU 利用率（nvidia-smi / nsys）
   - 利用率 < 50%？→ Memory-Bound，考虑量化 / 增大 Batch
   - 利用率接近 100% 但延迟高？→ Compute-Bound，考虑 TP / Kernel Fusion
3. 分析 Prefill vs. Decode 耗时比
   - TTFT 高 → Prefill 瓶颈，考虑 Chunked Prefill
   - TPOT 高 → Decode 瓶颈，考虑 GQA / 量化 / Speculative Decoding
4. 检查通信开销（多卡场景）→ All-Reduce / All-to-All 是否是主要耗时
```

---

## 十二、综合系统设计题

---

### 28. 如何为一个 70B 参数 LLM 设计高吞吐生产推理系统？

**答：**

**硬件选型：**

- 4× A100 80GB（Tensor Parallelism，单节点）
- 或 2× H100 80GB（NVLink，支持 FP8）

**部署方案：**

```
推理配置：
- 精度：W4A16（AWQ）→ 显存降至 ~35GB，4× A100 可容纳
- 注意力：GQA + Flash Attention 2
- KV Cache 管理：PagedAttention（vLLM）
- 并行：TP=4（单节点），不需 PP（层数够）
- Batching：Continuous Batching
- 解码加速：Speculative Decoding（7B Draft Model）
```

**延迟 / 吞吐权衡：**

|场景|配置|
|---|---|
|低延迟（实时对话）|Batch=1~4，FP16，Greedy/Low-T Sampling|
|高吞吐（离线批处理）|Batch=32~64，INT4，Continuous Batching|
|长上下文（128K）|KV Cache 量化为 INT8，GQA，PagedAttention|

---

### 29. 大模型推理中"显存墙"如何突破？

**答：**

显存压力来自三部分：**模型权重 + KV Cache + 激活值**。

|优化方向|技术手段|显存降幅|
|---|---|---|
|权重压缩|INT4 量化（AWQ/GPTQ）|~75%|
|KV Cache 压缩|GQA（$h/g$ 倍）+ INT8 量化|50%~87.5%|
|KV Cache 管理|PagedAttention，减少碎片|逻辑容量 ↑|
|模型分片|Tensor / Pipeline Parallelism|线性扩展|
|Offloading|KV Cache 卸载至 CPU / NVMe|实际容量 ↑（带宽代价）|
|架构优化|MLA（DeepSeek）|KV Cache 降低 5~13×|

---

## 十三、高频手撕 / 推导题

---

### 30. 手推：Self-Attention 的时间与空间复杂度分析

设序列长度 $n$，模型维度 $d$，头数 $h$，每头维度 $d_k = d/h$。

**前向传播各步骤复杂度（单头）：**

|操作|计算量|显存|
|---|---|---|
|$Q, K, V$ 投影|$3 \times O(nd^2)$|$O(nd)$|
|$QK^\top$|$O(n^2 d_k)$|$O(n^2)$|
|Softmax|$O(n^2)$|$O(n^2)$|
|$\text{Attn} \times V$|$O(n^2 d_k)$|$O(nd_k)$|
|输出投影|$O(nd^2)$|$O(nd)$|

总计：$O(nd^2 + n^2 d)$，当 $n < d$ 时计算瓶颈为投影；当 $n > d$ 时为注意力矩阵。

---

### 31. 手推：KV Cache 显存计算（通用公式）

$$M_{\text{KV}} = 2 \times p_a \times n_{\text{layers}} \times n_{\text{kv_heads}} \times d_{\text{head}} \times L \times B$$

各符号含义：

|符号|含义|
|---|---|
|$2$|K 和 V 各一份|
|$p_a$|精度字节数（FP16=2, INT8=1, INT4=0.5）|
|$n_{\text{layers}}$|Transformer 层数|
|$n_{\text{kv_heads}}$|KV 头数（GQA 时小于 $h$）|
|$d_{\text{head}}$|每头维度|
|$L$|序列长度（prompt + completion）|
|$B$|Batch Size|

---

> **文档说明**
> 
> - 本文档聚焦推理优化岗位的共通高频考点，已去除 CV / AIGC 专项题
> - 推导遵循"不跳步"原则，所有公式均在 Obsidian LaTeX 格式下可正常渲染
> - 建议配合 vLLM / Flash Attention 源码阅读作为延伸
