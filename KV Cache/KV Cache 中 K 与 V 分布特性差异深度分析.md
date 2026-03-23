
> **摘要**：本文从 Attention 机制的数学结构出发，系统分析 $K$ 矩阵产生通道级离群值（channel-wise outliers）而 $V$ 矩阵分布平滑的根本原因，并延伸至量化工程中的应对策略。

---

## 1. 问题的工程背景

### 1.1 KV Cache 量化的动机

在自回归推理（autoregressive decoding）中，每生成一个 token，模型需对所有历史 token 重新计算 Attention。KV Cache 将历史 token 的 $K$、$V$ 矩阵缓存在显存中以避免重复计算：

$$ \text{KV Cache size} = 2 \times B \times L \times H \times d_k \times \text{bytes\_per\_element} $$

其中 $B$ 为 batch size，$L$ 为序列长度，$H$ 为注意力头数，$d_k$ 为每头维度。

以 LLaMA-3 70B（GQA，8 个 KV 头，$d_k = 128$）为例，序列长度 $L = 4096$、batch size $B = 32$、FP16 存储时：

$$ \text{KV Cache} = 2 \times 32 \times 4096 \times 8 \times 128 \times 2\ \text{bytes} \approx 17.2\ \text{GB} $$

这一体积使 KV Cache 量化（通常目标 INT4/INT8）成为长序列推理的关键优化手段。

### 1.2 观测到的分布差异

对主流 LLM（LLaMA-2/3、Mistral、Qwen）的 KV Cache 激活值进行统计分析，系统性地发现以下现象：

|统计指标|$K$ 矩阵|$V$ 矩阵|
|---|---|---|
|通道间方差变异系数（CV）|高（$> 2.0$）|低（$< 0.5$）|
|离群值通道比例（$\|x\| > 6\sigma$）|1%–5%|$\approx 0$|
|最大值 / 均值比|10–50×|2–5×|
|离群值通道的位置稳定性|跨 token/batch 固定|不适用|
|INT4 量化后 PPL 劣化|显著（+0.5–2.0）|轻微（$< 0.2$）|

**离群值通道的位置稳定性**是关键观测：$K$ 的离群值不是随机出现的数值噪声，而是固定在特定通道索引上。这说明离群值是 $W_K$ 权重结构的体现，而非输入数据的随机性。

---

## 2. Attention 机制的不对称数学结构

### 2.1 标准 Multi-Head Attention 的计算链

$$ Q = X W_Q,\quad K = X W_K,\quad V = X W_V $$

$$ A = \text{softmax}!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{L \times L} $$

$$ \text{Output} = A V $$

$K$ 与 $V$ 在计算链中承担不同角色：

- **$K$** 的功能：参与 $\frac{QK^\top}{\sqrt{d_k}}$ 计算注意力得分（logits），随后进入 softmax 的指数运算，最终决定注意力权重 $\alpha_{ij}$ 的分布形态。
- **$V$** 的功能：作为信息载体，被注意力权重线性聚合，输出上下文表示。

这一功能差异是所有分布特性差异的根源。

### 2.2 softmax 对得分幅度的极度敏感性

设查询 $q \in \mathbb{R}^{d_k}$，键集合 ${k_1, \ldots, k_L} \subset \mathbb{R}^{d_k}$，注意力权重：

$$ \alpha_i = \frac{\exp(s_i)}{\sum_{j=1}^{L} \exp(s_j)},\quad s_i = \frac{q \cdot k_i}{\sqrt{d_k}} $$

分析 softmax 的"尖锐程度"（peakedness）。设 $s_1 = s + \delta$，$s_j = s$（$j \neq 1$），则：

$$ \alpha_1 = \frac{e^{s+\delta}}{e^{s+\delta} + (L-1)e^s} = \frac{e^\delta}{e^\delta + (L-1)} $$

当 $L = 1000$，需使 $\alpha_1 \geq 0.99$ 时：

$$ e^\delta \geq 99 \times 999 \approx 98901 \implies \delta \geq \ln(98901) \approx 11.5 $$

即得分差 $\delta = s_1 - s_j$ 需达到约 $11.5$。由于 $\delta = \frac{(q - 0) \cdot (k_1 - k_j)}{\sqrt{d_k}}$，在 $q$ 与 $k_j$ 幅度有限的情况下，**模型必须在 $k_1$ 的某些维度上编码极大幅值**，以保证内积差异足够大。

---

## 3. $K$ 离群值的形成机制

### 3.1 反向传播的通道放大效应

设损失函数 $\mathcal{L}$ 关于 $K$ 矩阵第 $i$ 行、第 $c$ 通道元素 $k_{i,c}$ 的梯度：

$$ \frac{\partial \mathcal{L}}{\partial k_{i,c}} = \sum_{j} \frac{\partial \mathcal{L}}{\partial \alpha_{ji}} \cdot \frac{\partial \alpha_{ji}}{\partial s_{ji}} \cdot \frac{\partial s_{ji}}{\partial k_{i,c}} $$

其中：

$$ \frac{\partial \alpha_{ji}}{\partial s_{ji}} = \alpha_{ji}(1 - \alpha_{ji}),\quad \frac{\partial s_{ji}}{\partial k_{i,c}} = \frac{q_{j,c}}{\sqrt{d_k}} $$

当模型需要对 token $i$ 分配极高注意力时，损失梯度会持续沿使 $\alpha_{ji}$ 增大的方向更新 $W_K$。关键在于，梯度对不同通道 $c$ 的放大效果不均匀——那些与 $q_{j,c}$ 相关性高的通道会被优先强化，形成**通道级特征专化（channel specialization）**。

训练收敛后，$W_K$ 在少数通道上积累了极大的权重范数，使得这些通道的激活值对任何输入 $X$ 都产生显著放大。

### 3.2 键向量的方向编码需求

从信息论视角，$K$ 承担的是**检索索引**功能：给定查询 $q$，键 $k_i$ 需要精确编码"token $i$ 应被关注"的方向信息。

在 $d_k$ 维空间中，要使 $k_i$ 在内积意义下与 $q$ 高度对齐，而与其他 $k_j$ 低度对齐（即高检索选择性），一种高效策略是**使用稀疏强特征通道**：让少数通道携带极大幅值，而多数通道接近零。这与稀疏码（sparse coding）的原理类似，是高维空间中实现方向区分的有效编码方式。

量化离群值通道数量 $n_{\text{out}}$ 与 attention 熵 $H(A)$ 之间存在负相关关系（注意力越尖锐，离群值通道越多），这在实验中得到印证。

### 3.3 RoPE 的频率诱导幅值波动

主流 LLM（LLaMA-3、Qwen、Mistral）使用 RoPE 编码相对位置信息。RoPE 对 $Q$、$K$ 施加旋转变换：

$$ \tilde{q}_m = \mathbf{R}_{\Theta,m} q_m,\quad \tilde{k}_n = \mathbf{R}_{\Theta,n} k_n $$

旋转矩阵 $\mathbf{R}_{\Theta,m}$ 对维度对 $(2i, 2i+1)$ 施加旋转角 $m\theta_i$，其中 $\theta_i = 10000^{-2i/d_k}$。

内积的相对位置依赖性体现为：

$$ \tilde{q}_m^\top \tilde{k}_n = \sum_{i=0}^{d_k/2 - 1} \left[ q_{2i} k_{2i} \cos((m-n)\theta_i) + q_{2i+1} k_{2i+1} \cos((m-n)\theta_i) + \ldots \right] $$

当旋转角 $(m-n)\theta_i \approx 0$ 时（低频维度、近距离位置），该维度对内积的贡献最大，模型会优先在这些维度上编码强特征。低频维度（小 $i$，大 $\theta_i^{-1}$）因此成为离群值的高发区——这与实验观察吻合：$K$ 的离群值通道集中在前几十个维度。

**$V$ 不参与 RoPE**，无此效应。

### 3.4 Sink Token 现象的关联

实验发现（StreamingLLM），Attention 矩阵中普遍存在"注意力汇聚"（attention sink）现象：首个 token（通常是 BOS token）被几乎所有查询分配极高注意力权重，与语义无关。

这一现象的键向量 $k_{\text{BOS}}$ 具有极端幅值——模型在训练中学会将 BOS token 的键编码到离群值通道，作为"安全归宿"（safe default）以保证 softmax 的数值稳定性。这进一步证实了 $K$ 离群值是**训练诱导的结构性特征**，而非随机现象。

---

## 4. $V$ 分布平滑的机制

### 4.1 凸组合的平滑性保证

$V$ 的输出是所有值向量的凸组合：

$$ \text{out} = \sum_{i=1}^{L} \alpha_i v_i,\quad \alpha_i \geq 0,\quad \sum_{i=1}^{L} \alpha_i = 1 $$

设 $v_i \in [a, b]^{d_v}$，则 $\text{out} \in [a, b]^{d_v}$（凸包内）。无论 $v_i$ 本身分布如何，凸组合操作天然地将输出约束在值向量的凸包内，不产生幅值放大。

更重要的是，$V$ 的梯度路径：

$$ \frac{\partial \mathcal{L}}{\partial v_{i,c}} = \alpha_i \cdot \frac{\partial \mathcal{L}}{\partial \text{out}_c} $$

由于 $\alpha_i \leq 1$，梯度被天然衰减，不存在使特定通道幅值持续增大的梯度信号。

### 4.2 无竞争性约束

$K$ 需要在内积空间中实现**竞争性区分**：$k_i$ 必须与 $q$ 的内积大于所有 $k_j$（$j \neq i$）。这是一个对抗性约束，推动 $K$ 的特征向"极端但有辨别力"的方向演化。

$V$ 无需竞争。它只需携带语义信息，供注意力权重线性聚合。训练中没有任何损失项要求 $V$ 的某个通道超过其他通道——梯度均匀分布在所有语义相关通道上，不产生通道专化。

### 4.3 $W_V$ 权重谱的统计证据

对 LLaMA-2 7B 的权重矩阵奇异值分布进行分析（截止知识范围内的研究结果）：

- $W_K$ 的奇异值分布：重尾（heavy-tailed），最大与最小奇异值之比（条件数）达 $10^3$–$10^4$，少数奇异方向主导能量。
- $W_V$ 的奇异值分布：相对均匀，条件数通常在 $10^1$–$10^2$ 量级。

$W_K$ 的高条件数意味着输入向量在特定方向上会被强烈放大，直接导致输出激活值的通道极端不均匀。

---

## 5. 量化误差的数学分析

### 5.1 均匀量化的误差模型

对张量 $X \in \mathbb{R}^{d}$ 进行 $b$ 位均匀量化，量化步长：

$$ \Delta = \frac{\max(X) - \min(X)}{2^b - 1} $$

量化误差的均方根（RMSE）近似为：

$$ \text{RMSE} \approx \frac{\Delta}{\sqrt{12}} = \frac{\max(X) - \min(X)}{\sqrt{12} \cdot (2^b - 1)} $$

当 $X$ 存在离群值时，$\max(X) - \min(X)$ 被极大值主导，导致正常值的量化粒度（$\Delta$）远大于其实际范围，造成严重的量化误差。

**示例**：假设 99% 的值在 $[-1, 1]$ 之间，1% 的离群值达到 $\pm 20$：

- 无离群值时，INT8 量化步长 $\Delta = \frac{2}{255} \approx 0.0078$
- 有离群值时，$\Delta = \frac{40}{255} \approx 0.157$，误差放大约 **20 倍**

### 5.2 $K$ 与 $V$ 的量化误差对比

设 $K$ 的通道 $c$ 有离群值 $k_{\max} \gg k_{\text{normal}}$，对整个 $K$ 矩阵做 per-tensor INT4 量化：

$$ \text{RMSE}_K \propto \frac{k_{\max}}{2^4 - 1} = \frac{k_{\max}}{15} $$

而 $V$ 的各通道幅值接近，$v_{\max} \approx v_{\text{normal}}$：

$$ \text{RMSE}_V \propto \frac{v_{\max}}{15} \ll \frac{k_{\max}}{15} $$

这直接解释了为何 $K$ 的 per-tensor 量化效果远差于 $V$。

### 5.3 量化误差对 Attention 输出的影响

$K$ 的量化误差 $\varepsilon_K$ 通过 softmax 传播：

$$ \tilde{s}_i = \frac{q \cdot (k_i + \varepsilon_{K,i})}{\sqrt{d_k}} = s_i + \frac{q \cdot \varepsilon_{K,i}}{\sqrt{d_k}} $$

量化噪声对得分的扰动为 $\delta s_i = \frac{q \cdot \varepsilon_{K,i}}{\sqrt{d_k}}$。由于 softmax 对得分差异高度敏感（指数放大），即使 $|\varepsilon_{K,i}|$ 较小，也可能显著改变注意力权重分布，引发下游输出质量劣化。

$V$ 的量化误差 $\varepsilon_V$ 对输出的影响是线性的：

$$ \widetilde{\text{out}} = \sum_i \alpha_i (v_i + \varepsilon_{V,i}) = \text{out} + \sum_i \alpha_i \varepsilon_{V,i} $$

由于 $\alpha_i \leq 1$ 且 $\sum_i \alpha_i = 1$，误差项 $\sum_i \alpha_i \varepsilon_{V,i}$ 是 $\varepsilon_{V,i}$ 的加权平均，天然具有**误差抵消效应**，输出误差小于单个 $\varepsilon_{V,i}$ 的最大值。

---

## 6. 工程应对策略

### 6.1 针对 $K$ 离群值的量化方案

#### 6.1.1 Per-Channel / Per-Group 量化

对 $K$ 矩阵的每个通道 $c$ 独立计算量化范围：

$$ \Delta_c = \frac{\max_i(k_{i,c}) - \min_i(k_{i,c})}{2^b - 1} $$

每个通道有独立的缩放因子（scale）和零点（zero point）。Per-channel 量化可完全消除离群值通道对正常通道量化粒度的影响，但引入额外存储开销（每通道一个 FP16 缩放因子）。

Per-group 量化是折中方案：将 $d_k$ 个通道分为 $g$ 组，每组共享一对缩放因子，组大小通常为 $64$ 或 $128$。

#### 6.1.2 混合精度存储（Outlier Channel Isolation）

识别离群值通道索引集合 $\mathcal{C}_{\text{out}}$（通常 $|\mathcal{C}_{\text{out}}| / d_k < 5\%$），对这些通道保留 FP16，其余通道用 INT4：

$$ K = \underbrace{K_{:,\mathcal{C}_{\text{out}}}}_{\text{FP16}} \oplus \underbrace{K_{:,\overline{\mathcal{C}}_{\text{out}}}}_{\text{INT4}} $$

代表工作：KVQuant（Hooper et al., 2024）。实际有效比特率约为 $4.2$–$4.5$ bits，接近纯 INT4 的存储效率，但量化精度接近 FP16。

#### 6.1.3 旋转变换平滑（Rotation-based Smoothing）

使用随机正交矩阵 $R \in \mathbb{R}^{d_k \times d_k}$ 对 $K$、$Q$ 施加旋转：

$$ \tilde{K} = K R,\quad \tilde{Q} = Q R $$

旋转保持内积不变（$RR^\top = I$）：

$$ \tilde{Q}\tilde{K}^\top = QRR^\top K^\top = QK^\top $$

旋转将 $K$ 的能量从少数离群值通道均匀分散到所有通道，各通道幅值趋于一致，大幅降低量化难度。代表工作：**QuaRot**（Ashkboos et al., 2024）、**SpinQuant**（Liu et al., 2024）。

旋转矩阵可在离线（offline）阶段预计算并合并进权重矩阵 $W_K$、$W_Q$，推理时无额外计算开销：

$$ \tilde{W}_K = W_K R,\quad \tilde{W}_Q = W_Q R $$

#### 6.1.4 激活感知缩放（Activation-Aware Scaling）

借鉴 SmoothQuant 的思路，将 $K$ 激活值的通道级缩放因子 $s_c$ 转移到权重矩阵：

$$ K W = (K \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) $$

通过选取合适的 $s_c$（如 $s_c = \sqrt{\max|k_c|}$），平衡激活值与权重的量化难度。这需要代表性校准数据（calibration data）来估计 $s_c$。

### 6.2 $V$ 矩阵的量化方案

由于 $V$ 分布平滑，可直接使用更简单的方案：

- **Per-tensor INT4/INT8**：单一缩放因子，开销最小，精度损失可接受。
- **Per-token 量化**：对每个 token 的值向量独立量化，适用于动态范围随序列位置变化的场景。

实测表明，$V$ 的 INT4 per-tensor 量化通常可将 PPL 劣化控制在 $0.1$–$0.2$ 以内（相对 FP16 baseline）。

### 6.3 主流框架中的实现

|框架|$K$ 量化方案|$V$ 量化方案|备注|
|---|---|---|---|
|vLLM（FP8 KV）|Per-tensor FP8|Per-tensor FP8|硬件原生支持，H100/H20|
|TensorRT-LLM|Per-channel INT8|Per-tensor INT8|工程最优，广泛部署|
|KVQuant|Outlier isolation + INT4|INT4|学术方案，精度最优|
|KIVI|Per-channel INT2/INT4|Per-tensor INT2/INT4|极低比特探索|
|QuaRot|Rotation + INT4|INT4|无精度损失目标|

---

## 7. 超越量化：离群值对其他优化的影响

### 7.1 KV Cache 压缩（Eviction）

基于重要性的 KV Cache 淘汰（eviction）策略（如 H2O、SnapKV）通常以 Attention 权重 $\alpha_{ij}$ 衡量 token 重要性。

由于 $K$ 的离群值通道集中编码了注意力汇聚信息（如 BOS token 的极高权重），直接按 $\alpha$ 淘汰可能错误保留或丢弃 token。一些工作（如 MagicPIG）在相似度度量中显式处理 $K$ 的离群值通道，以提高淘汰精度。

### 7.2 投机解码（Speculative Decoding）

在 MedusaHead 或 EAGLE 等框架中，草稿模型需要复用 KV Cache。$K$ 离群值导致的量化误差会通过注意力权重影响草稿 token 的接受率（acceptance rate），需要在草稿验证阶段额外补偿。

### 7.3 稀疏 Attention 的 Mask 精度

FlashAttention 的块稀疏变体（如 BigBird、Longformer）通过提前 mask 低分块来跳过计算。$K$ 的离群值使块得分估计（block score estimation）不准确——少数极大值通道可能使整个块的得分估计偏高，导致错误保留低相关性块，降低稀疏化收益。

---

## 8. 总结

$K$ 与 $V$ 的分布差异不是偶然的统计现象，而是 Attention 机制功能不对称性在权重空间的必然体现：

1. **$K$ 承担竞争性方向编码**：softmax 的指数放大机制迫使 $K$ 在少数通道上编码极大幅值，以精确控制注意力分布的峰值位置。训练通过反向传播持续强化这一特征专化，形成固定位置的离群值通道。
2. **RoPE 加剧 $K$ 的通道幅值不均**：频率依赖的旋转变换在低频维度上产生周期性幅值放大，使 $K$ 的离群值进一步集中在特定维度。
3. **$V$ 的凸组合本质保证平滑性**：线性聚合操作对幅值无放大要求，梯度均匀分散，不产生通道专化，分布天然均匀。
4. **量化影响的不对称传播**：$K$ 的量化误差经 softmax 指数放大后严重影响注意力分布；$V$ 的量化误差经加权平均后自然衰减。

上述机制共同决定了 $K$ 需要专门的量化处理策略（per-channel、outlier isolation、旋转变换），而 $V$ 可用更简单的 per-tensor 方案高效压缩。

---

_参考工作（知识截止范围内）：_

- _KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization（Hooper et al., 2024）_
- _KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache（Liu et al., 2024）_
- _QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs（Ashkboos et al., 2024）_
- _SpinQuant: LLM Quantization with Learned Rotations（Liu et al., 2024）_
- _StreamingLLM: Efficient Streaming Language Models with Attention Sinks（Xiao et al., 2024）_
- _SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs（Xiao et al., 2023）_
