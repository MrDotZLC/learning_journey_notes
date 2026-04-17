## 1. 核心问题：KV Cache 的内存瓶颈

在自回归推理中，每个 token 生成需要访问所有历史 token 的 K、V 矩阵。

**KV Cache 内存公式：**

$$M_{KV} = 2 \times L \times H_{kv} \times S \times d_h \times \text{dtype\_bytes}$$

- $L$：层数，$H_{kv}$：KV 头数，$S$：序列长度，$d_h$：每头维度

**瓶颈本质**：推理是 Memory-Bound，不是 Compute-Bound。减少 KV 表示的体积是核心优化方向。

---

## 2. Prefill vs Decode 阶段的行为差异

理解后续所有优化的前提。

|阶段|输入|Q shape|KV 来源|计算特征|
|---|---|---|---|---|
|Prefill|完整 prompt（$S$ tokens）|$[B, S, d]$|当前输入，全部计算|Compute-Bound，矩阵×矩阵|
|Decode|单个新 token|$[B, 1, d]$|读 Cache + 追加当前|Memory-Bound，矩阵×向量|

**关键推论：**

- KV Cache 优化收益集中在 **Decode 阶段**（内存带宽是瓶颈）
- Prefill 阶段收益主要来自减少 KV 的**计算量**（$H_{kv}$ 越小，Prefill FLOPS 越少）
- GQA/MQA 对 Decode 加速比 Prefill 更显著

---

## 3. MHA（Multi-Head Attention）

**论文**：Attention Is All You Need — Vaswani et al., Google Brain, 2017

**背景**：设计目标是替代 RNN，多头并行捕获不同语义子空间的依赖关系。KV Cache 瓶颈在当时（序列长度 ≤512，模型 <1B）尚未成为工程矛盾，MHA 无推理优化意图。

### 3.1 结构

每个注意力头独立拥有 Q、K、V 投影矩阵：

$$\text{head}_i = \text{Attention}(XW_i^Q,\ XW_i^K,\ XW_i^V)$$

$$\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H), W^O$$

$$\text{Attention}(Q,K,V) = \text{softmax}!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

### 3.2 参数量

|矩阵|形状|参数量|
|---|---|---|
|$W^Q$|$d_{\text{model}} \times d_{\text{model}}$|$d_{\text{model}}^2$|
|$W^K$|$d_{\text{model}} \times d_{\text{model}}$|$d_{\text{model}}^2$|
|$W^V$|$d_{\text{model}} \times d_{\text{model}}$|$d_{\text{model}}^2$|
|$W^O$|$d_{\text{model}} \times d_{\text{model}}$|$d_{\text{model}}^2$|

KV Cache 内存系数 = $H$

---

## 4. MQA（Multi-Query Attention）

**论文**：Fast Transformer Decoding — Shazeer, Google, 2019

**背景**：Decode 阶段每步仅生成一个 token，算力消耗极小，但需从 HBM 读取所有历史 K/V，arithmetic intensity 远低于 GPU 峰值 ops:byte 比，推理严重受限于内存带宽。多头 K/V 是带宽浪费的主要来源，而 Q 多头对质量有正向贡献——两者边际价值不对等。MQA 将 K/V 压缩至单头以最大化带宽节省。

### 4.1 结构

所有 Q 头共享同一组 K、V：

$$\text{head}_i = \text{Attention}(XW_i^Q,\ XW^K,\ XW^V)$$

- $W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$（单组）
- $W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$（单组）

### 4.2 KV Cache 对比

|机制|KV 头数|KV Cache 大小|内存节省|
|---|---|---|---|
|MHA|$H$|$2Hd_h$ per token|基准|
|MQA|$1$|$2d_h$ per token|$H\times$|

### 4.3 MQA 可以共享 KV 的理论依据

注意力的本质是用 $Q$ 对序列上下文加权聚合 $V$，权重由 $Q \cdot K^\top$ 决定。共享 KV 并不意味着各头权重相同——不同 head 的 $Q_i$ 不同，因此：

$$A_i = \text{softmax}!\left(\frac{Q_i K^\top}{\sqrt{d_k}}\right)$$

各 head 的 $A_i$ **仍然不同**，差异被保留在 $Q$ 的多样性中，而非 $K/V$ 的多样性中。

### 4.4 质量下降的根本原因

#### 4.4.1 Key 子空间的信息瓶颈

MHA 中第 $i$ 个 head 的 Key 投影 $K_i = XW_i^K$ 将输入映射至独立子空间 $\mathcal{S}_i = \text{col}(W_i^K)$，各 head 可关注不同的语义特征。

$$\text{MHA：}\bigoplus_{i=1}^{H} \mathcal{S}_i \subseteq \mathbb{R}^{H \cdot d_k} \quad \text{（各子空间独立）}$$

$$\text{MQA：}\mathcal{S}_{\text{shared}} \subseteq \mathbb{R}^{d_k} \quad \text{（单一子空间）}$$

表达容量从 $H \cdot d_k$ 维压缩至 $d_k$ 维，形成严重的**信息瓶颈**。

#### 4.4.2 Value 多样性的丧失

MHA 中每个 head 从独立 $W_i^V$ 提取不同语义片段；MQA 中所有 head 从同一 $V = XW^V$ 聚合，即便 $A_i$ 不同，输出也只是**对同一特征集合的不同加权**，head 多样性（head diversity）实质性下降。

#### 4.4.3 梯度干扰

所有 head 的梯度信号叠加至同一个 $W^K$：

$$\frac{\partial \mathcal{L}}{\partial W^K} = \sum_{i=1}^{H} \frac{\partial \mathcal{L}}{\partial A_i} \cdot \frac{\partial A_i}{\partial K} \cdot \frac{\partial K}{\partial W^K}$$

各 head 对 Key 空间的优化方向不一致时，$W^K$ 陷入折中，无法同时满足所有 head 的偏好。MHA 中 $W_i^K$ 独立，各 head 可自由特化（specialization）。

#### 4.4.4 实验现象汇总

|观测维度|MHA|MQA|
|---|---|---|
|Perplexity（同等参数）|更低|更高（约 +0.3~1.5，依模型规模）|
|长序列任务（RAG, summarization）|更强|明显退化|
|Head 间 attention 分布多样性|高|低（head collapse 现象）|
|KV Cache 显存占用|$O(H)$|$O(1)$|
|推理吞吐量|受 memory bandwidth 瓶颈|显著提升（2~5×）|

> **Head Collapse**：MQA 中多个 head 的注意力矩阵 $A_i$ 趋于相似，模型退化为伪单头注意力，可通过计算 ${A_i}$ 之间的 JS 散度验证。

**代价**：K/V 表达能力严重弱化，大模型质量损失明显。

**使用模型**：PaLM、Falcon、Gemini（早期）

---

## 5. GQA（Grouped-Query Attention）

**论文**：GQA: Training Generalized Multi-Query Transformer Models — Ainslie et al., Google, 2023

**背景**：LLaMA-1 发布后开源社区大规模部署推理，MHA 内存开销过大，MQA 质量损失不可接受。GQA 的目标是在两者之间提供可工程化的折中，并给出将已有 MHA 模型低成本转换为 GQA 的路径（mean pooling 初始化 + 少量微调），使存量模型无需从头训练即可获得 KV Cache 收益。

### 5.1 核心思想

将 $H$ 个 Q 头分为 $G$ 组，每组共享一对 K/V 头：

$$G \in [1,\ H], \quad G = 1 \Rightarrow \text{MQA}, \quad G = H \Rightarrow \text{MHA}$$

### 5.2 结构示意

```
Q heads:  [ Q1  Q2 | Q3  Q4 | Q5  Q6 | Q7  Q8 ]   H=8
               ↓         ↓         ↓         ↓
KV heads: [  KV1   |  KV2   |  KV3   |  KV4  ]    G=4
```

### 5.3 数学表达

第 $i$ 个 Q 头所属组 $g(i) = \lfloor i \cdot G / H \rfloor$：

$$\text{head}_i = \text{softmax}!\left(\frac{Q_i W_i^Q \left(K_{g(i)} W_{g(i)}^K\right)^\top}{\sqrt{d_k}}\right) V_{g(i)} W_{g(i)}^V$$

### 5.4 三机制 KV Cache 对比

|机制|KV 头数|KV Cache（相对）|质量|
|---|---|---|---|
|MHA|$H$|$1.0\times$|最高|
|GQA|$G$|$G/H$|中间|
|MQA|$1$|$1/H$|最低|

### 5.5 典型配置

|模型|$H$（Q头）|$G$（KV头）|
|---|---|---|
|LLaMA-2 7B|32|32（MHA）|
|LLaMA-2 70B|64|8（GQA）|
|LLaMA-3 8B|32|8（GQA）|
|LLaMA-3 70B|64|8（GQA）|
|Mistral 7B|32|8（GQA）|
|Qwen2 7B|28|4（GQA）|

### 5.6 RoPE 与 GQA 的交互

RoPE 仅施加在 Q 和 K 上，不施加在 V 上：

$$Q_i' = \text{RoPE}(Q_i W_i^Q,\ \text{pos}), \quad K_g' = \text{RoPE}(K_g W_g^K,\ \text{pos})$$

每个 KV 头 $g$ 的 $K_g$ 只需做一次 RoPE，被组内所有 Q 头共享，不重复计算。KV Cache 存储的是**施加 RoPE 之后**的 K，否则每次 decode 步需对历史 K 重新编码。

### 5.7 GQA 分组数 $G$ 的选取依据

$G$ 的选取不是任意的经验值，而是由三个正交维度共同约束。

#### 5.7.1 工程约束：硬件决定边界

**KV Cache 显存预算**

单层 KV Cache 占用：

$$M_{\text{KV}} = 2 \times G \times d_k \times n \times b$$

总显存需满足：

$$L \times M_{\text{KV}} + M_{\text{weights}} \leq M_{\text{GPU}}$$

反解出 $G$ 的上界：

$$G \leq \frac{(M_{\text{GPU}} - M_{\text{weights}}) / L - \text{其他激活}}{2 \times d_k \times n \times b}$$

**Tensor Parallelism（TP）整除约束**

在多 GPU 并行推理中，KV head 需按 GPU 数量 $P$ 整除分配：

$$G \bmod P = 0, \quad G / P \geq 1$$

这是**最硬性的工程约束**，违反则某些 rank 无 KV head 可分配。

|模型|$H$|$G$|TP 度 $P$|$G/P$|
|---|---|---|---|---|
|LLaMA-3 8B|32|8|1/2/4|8/4/2|
|LLaMA-3 70B|64|8|8|1|
|Mistral 7B|32|8|1/2/4/8|8/4/2/1|
|Qwen2 72B|64|8|8|1|

**Memory Bandwidth 对齐**

KV Cache 读取量正比于 $G$，当 $G$ 过大时成为带宽瓶颈，延迟线性增长。经验规则：

$$G \times d_k \leq \frac{\text{BW} \times t_{\text{target}}}{\text{batch_size} \times n \times b}$$

#### 5.7.2 训练目标：质量决定下界

**原论文实验结论（Ainslie et al., 2023）**

在 T5 系列上：

- $G = 1$（MQA）：perplexity 差距约 0.5~2.0
- $G \geq H/8$：与 MHA 几乎无差异（差距 $< 0.1$ perplexity）
- 质量对 $G$ 的敏感性呈**边际递减**：从 $G=1$ 增大到 $G=4$ 提升显著，从 $G=4$ 增大到 $G=H$ 提升微弱

> 【图示占位】折线图：横轴为 $G$（1, 2, 4, 8, 16, 32=$H$），纵轴为 validation perplexity，曲线在 $G \geq H/8$ 后趋于平坦。

**Head 多样性的信息论支持**

定义 head 间多样性为平均 JS 散度：

$$D_{\text{div}} = \frac{2}{H(H-1)} \sum_{i < j} D_{\text{JS}}(A_i | A_j)$$

实验观察（Voita et al., 2019）：MHA 中真正功能性 head 约占 $H$ 的 20%~30%。若 $H=32$ 中仅有约 8 个功能性 head，则 $G=8$ 已能覆盖主要的语义多样性，这从信息论角度支持了 $G \ll H$ 的合理性。

#### 5.7.3 结构初始化：从 MHA Checkpoint 蒸馏时的分组策略

从已训练 MHA 模型转换为 GQA 时（uptrain 方案），将 $H$ 个 KV head 合并为 $G$ 个，采用**顺序等分 + Mean Pooling**：

$$W_j^K = \frac{1}{H/G} \sum_{i \in \mathcal{G}_j} W_i^K, \quad j = 1, \ldots, G$$

$$\mathcal{G}_j = \left{(j-1) \cdot \frac{H}{G} + 1,\ \ldots,\ j \cdot \frac{H}{G}\right}$$

顺序等分优于随机分组，原因是相邻 head 在训练中倾向于学习相近的特征（head space 中的位置局部性）。Uptrain 仅需约 5%~10% 原始 token 量即可恢复接近 MHA 的质量。

#### 5.7.4 综合约束与工程收敛点

|约束来源|对 $G$ 的限制|
|---|---|
|Tensor Parallelism|$G \bmod P = 0$，且 $G \geq P$|
|KV Cache 显存预算|$G \leq M_{\text{budget}} / (2 d_k n b L)$|
|质量下界|$G \geq H/8$（经验值，任务相关）|
|带宽效率|$G$ 越小越好，边际收益递减|

**典型收敛点**：

$$G = \frac{H}{8}, \quad G \bmod P = 0$$

这解释了为何 $H=32$ 时 $G=8$、$H=64$ 时 $G=8$ 几乎是标准配置——同时满足 TP 整除性、显存预算和质量下界三重约束。

**从头训练 vs. Uptrain 的差异**

|场景|分组策略|$G$ 的决定因素|
|---|---|---|
|从头训练|随机初始化 $G$ 组 KV 投影|工程约束 + 质量实验|
|从 MHA Uptrain|Mean Pooling 合并相邻 head|目标 $G$ 由工程约束决定，合并方式由相邻性决定|

---

## 6. MLA（Multi-head Latent Attention）

> **论文**：DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model — DeepSeek, 2024

**背景**：GQA 通过减少头数压缩 Cache，但头数减少直接削弱模型的多头表达能力，在超大规模模型（$H=128$）上质量代价显著。MLA 换一个压缩维度：不减少头数，而是对 K/V 做低秩压缩，缓存低维隐向量，推理时按需还原，在极端压缩比下保留完整头数的表达能力。

---

### 6.1 压缩与还原流程

**Down-projection（压缩）**

$$ c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}, \quad d_c \ll d_{\text{model}} $$

- $h_t$：第 $t$ 个 token 的隐状态
- $W^{DKV} \in \mathbb{R}^{d_c \times d_{\text{model}}}$：down-projection 矩阵
- $d_c$：压缩维度（KV latent dimension）

Cache 中仅存储压缩向量 $c_t^{KV}$，而非完整的 $K_t$、$V_t$，这是 MLA 节省内存的核心。

**Up-projection（还原）**

$$ K_t = W^{UK}c_t^{KV} \in \mathbb{R}^{H \cdot d_h} , \quad V_t = W^{UV}c_t^{KV} \in \mathbb{R}^{H \cdot d_h} $$

- $W^{UK},\ W^{UV} \in \mathbb{R}^{(H \cdot d_h) \times d_c}$：up-projection 矩阵
- $H$：注意力头数，$d_h$：每头维度

Up-projection 在推理时按需执行，不占用持久化 Cache。

**完整注意力计算**

$$ \text{head}_i = \text{Softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_h}}\right) V_i $$

$$ \text{MLA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H), W^O $$

形式上与 MHA 完全一致，差异仅在于 $K_i$、$V_i$ 来自低秩还原而非直接投影。

---

### 6.2 Q 侧低秩压缩

类比 KV 侧，Q 的投影同样可做低秩分解以减少参数量：

$$ c_t^Q = W^{DQ}, h_t \in \mathbb{R}^{d_c'}, \quad Q_t = W^{UQ}, c_t^Q \in \mathbb{R}^{H \cdot d_h} $$

- $d_c'$：Q 侧压缩维度（可与 $d_c$ 不同）
- $W^{DQ} \in \mathbb{R}^{d_c' \times d_{\text{model}}}$，$W^{UQ} \in \mathbb{R}^{(H \cdot d_h) \times d_c'}$

**关键区别**：$c_t^Q$ **不需要缓存**，每个 decode step 重新计算，仅用于减少 Q 投影的参数量（将 $H \cdot d_h \times d_{\text{model}}$ 的矩阵分解为两个小矩阵之积），对 Cache 大小无影响。

---

### 6.3 Decoupled RoPE

#### 6.3.1 冲突根源

RoPE（Rotary Position Embedding）将位置信息注入 Q、K：

$$ \text{RoPE}(x,\ t) = R_t \cdot x $$

其中 $R_t$ 是依赖位置 $t$ 的旋转矩阵。

若将 RoPE 应用于还原后的 $K_t$，则：

$$ K_t^{\text{RoPE}} = R_t \cdot W^{UK} c_t^{KV} $$

Cache 中存的是 $c_t^{KV}$（与位置无关），历史位置 $t' < t$ 对应的 $R_{t'}$ 无法在当前 step 复现，导致位置编码错误。

**结论**：直接对还原后的 K 注入 RoPE 与低秩 Cache 方案不相容。

#### 6.3.2 解耦方案

引入独立的 RoPE Key 分量 $k_t^R$，与压缩 Key $k_t^C$（还原自 $c_t^{KV}$）拼接：

$$ k_t^C = W^{UK}, c_t^{KV} \in \mathbb{R}^{H \cdot d_h^C} $$

$$ k_t^R = \text{RoPE}(W^{KR}, h_t,\ t) \in \mathbb{R}^{H \cdot d_h^R} $$

$$ k_t = \left[k_t^C;\ k_t^R\right] \in \mathbb{R}^{H \cdot (d_h^C + d_h^R)} $$

对应地，Q 侧也拆分为两部分：

$$ q_t = \left[q_t^C;\ q_t^R\right], \quad q_t^R = \text{RoPE}(W^{QR}, c_t^Q,\ t) $$

注意力计算时：

$$ \text{score}_{t,t'} = \frac{(q_t^C)^\top k_{t'}^C + (q_t^R)^\top k_{t'}^R}{\sqrt{d_h^C + d_h^R}} $$

**实际 Cache 存储**（每个 token）：

$$ \text{Cache per token} = c_t^{KV} \oplus k_t^R $$

即低秩压缩向量 $c_t^{KV} \in \mathbb{R}^{d_c}$ 与 RoPE Key $k_t^R \in \mathbb{R}^{H \cdot d_h^R}$ 的拼接。$k_t^C$ 在每步推理时从 $c_t^{KV}$ 即时还原，无需缓存。

---

### 6.4 Absorption Trick（推理加速）

#### 6.4.1 动机

朴素实现中，每个 decode step 需要对所有历史 token 执行 up-projection：

$$ K_{\text{hist}} = W^{UK}, C_{\text{hist}} \in \mathbb{R}^{T_{\text{hist}} \times H \cdot d_h} $$

当历史长度 $T_{\text{hist}}$ 很大时，这一步计算量和内存访问量显著。

#### 6.4.2 矩阵吸收

注意力得分的计算路径如下（以第 $i$ 个头为例，忽略 RoPE 部分）：

$$ \text{score} = Q_i K_i^\top = (W_i^{UQ}, c_t^Q)(W_i^{UK}, C)^\top $$

调整结合顺序：

$$ = \underbrace{(c_t^Q)^\top (W_i^{UQ})^\top W_i^{UK}}_{\tilde{q}_i^\top \in \mathbb{R}^{1 \times d_c}} C^\top $$

令 $\tilde{W}_i^{QK} = (W_i^{UQ})^\top W_i^{UK} \in \mathbb{R}^{d_c' \times d_c}$，则：

$$ \tilde{q}_i = \tilde{W}_i^{QK}, c_t^Q \in \mathbb{R}^{d_c} $$

$$ \text{score}_i = \tilde{q}_i^\top C^\top $$

注意力直接在压缩空间 $\mathbb{R}^{d_c}$ 中计算，**无需显式还原完整 $K$ 矩阵**。

类似地，输出端：

$$ O_i = \text{Softmax}(\text{score}_i), V_i = \text{Softmax}(\text{score}_i), W_i^{UV}, C $$

令 $\tilde{o}_i = \text{Softmax}(\text{score}_i), C$，再左乘 $W_i^{UV}$，V 的还原同样被吸收入矩阵乘法。

**合并后的矩阵**：

$$ \tilde{W}_i^{QK} = (W_i^{UQ})^\top W_i^{UK}, \quad \tilde{W}_i^{OV} = W_i^{UV} $$

这些矩阵可在**部署前预计算**，推理时不产生额外开销。

#### 6.4.3 加速效果

|实现方式|KV Cache 读取量|注意力计算维度|
|---|---|---|
|朴素（先还原）|$T \times H \cdot d_h$|$H \times d_h$|
|Absorption（直接压缩空间）|$T \times d_c$|$d_c$|

$d_c \ll H \cdot d_h$（DeepSeek-V2 中 $d_c = 512$，$H \cdot d_h = 128 \times 128 = 16384$），内存带宽消耗减少约 $32\times$。

---

### 6.5 Cache 大小对比

以 DeepSeek-V2 参数为例：$H=128$，$d_h=128$，$d_c=512$，$d_h^R=64$，FP16（2 bytes/element）。

#### 6.5.1 各机制 Cache per token per layer

**MHA**

$$ \text{Cache}_{\text{MHA}} = 2 \times H \times d_h \times 2\ \text{bytes} = 2 \times 128 \times 128 \times 2 = 65{,}536\ \text{bytes} $$

（K 和 V 各 $H \times d_h$，FP16）

**GQA（$G=8$ KV heads）**

$$ \text{Cache}_{\text{GQA}} = 2 \times G \times d_h \times 2\ \text{bytes} = 2 \times 8 \times 128 \times 2 = 4{,}096\ \text{bytes} $$

**MLA**

$$ \text{Cache}_{\text{MLA}} = \underbrace{d_c \times 2}_{\text{压缩向量}\ c_t^{KV}} + \underbrace{H \times d_h^R \times 2}_{\text{RoPE Key}\ k_t^R} $$

$$ = 512 \times 2 + 128 \times 64 \times 2 = 1{,}024 + 16{,}384 = 17{,}408\ \text{bytes} $$

#### 6.5.2 对比表

|机制|Cache per token per layer（bytes）|相对 MHA|
|---|---|---|
|MHA|65,536|$1.0\times$|
|GQA（$G=8$）|4,096|$\approx 1/16$|
|MLA|17,408|$\approx 1/3.8$|

**注**：MLA 的 Cache 压缩比不及 GQA（$1/3.8$ vs $1/16$），但 MLA 保留完整 $H=128$ 头的表达能力，GQA 实际 KV head 数仅为 8；在模型质量与 Cache 效率的 Pareto 前沿上，MLA 处于不同的位置。

#### 6.5.3 DeepSeek-V2 整模型 Cache 估算

DeepSeek-V2 共 60 层（$L=60$），以 MLA Cache per token 估算：

$$ \text{KV Cache per token} = 17{,}408 \times 60 = 1{,}044{,}480\ \text{bytes} \approx 1\ \text{MB/token} $$

与等规模 MHA 模型（$\approx 3.9\ \text{MB/token}$）相比，单 token Cache 降低约 $3.8\times$，可在相同显存下支持约 $3.8\times$ 更长的上下文或更大的并发批量。

---

### 6.6 MLA 与 GQA/MQA 的本质差异

|维度|MHA|GQA|MQA|MLA|
|---|---|---|---|---|
|KV head 数|$H$|$G$（$1 < G < H$）|1|$H$（逻辑上）|
|Cache 压缩方式|无|减少头数|极端减少头数|低秩压缩 KV|
|Cache 维度|$2 H d_h$|$2 G d_h$|$2 d_h$|$d_c + H d_h^R$|
|多头表达能力|完整|降低|最低|完整（还原后）|
|位置编码方式|标准 RoPE|标准 RoPE|标准 RoPE|Decoupled RoPE|
|推理端需 Up-projection|否|否|否|是（或 Absorption）|
|训练额外开销|无|无|无|少量（额外投影矩阵）|

---

### 6.7 实现要点（推理引擎视角）

**Cache 布局**

与 MHA/GQA 不同，MLA 的 Cache 存储两类异构数据：压缩向量 $c_t^{KV}$（维度 $d_c$）与 RoPE Key $k_t^R$（维度 $H d_h^R$）。推理框架需为二者分别分配 PagedAttention block，或拼接为统一布局后按偏移索引。

**Absorption 的前提**

Absorption Trick 要求 $W^{UQ}$、$W^{UK}$、$W^{UV}$ 在部署前已合并，且不能与 LoRA 等运行时权重修改方案直接组合（合并后矩阵会失效）。

**FlashAttention 兼容性**

压缩空间注意力（$\tilde{q}_i^\top C^\top$）的 head 维度为 $d_c$（512），大于标准 $d_h$（128），FlashAttention kernel 的 tile 策略需相应调整；部分实现回退到标准 GEMM 路径。

**Quantization**

$c_t^{KV}$ 的数值分布不同于原始 K/V，FP8/INT8 量化需独立校准；$k_t^R$ 的分布与 RoPE 旋转后的 K 一致，可复用 GQA 量化方案。## 7. 推理实现细节

### 7.1 GQA 的 KV Expand：naive vs zero-copy

**Naive 实现**（训练/调试，产生实际内存复制）：

```cpp
// k: [B, G, S, d_h]  →  [B, H, S, d_h]
int repeat = num_q_heads / num_kv_heads;
k = k.repeat_interleave(repeat, /*dim=*/1);
v = v.repeat_interleave(repeat, /*dim=*/1);
```

**Zero-copy 实现**（推理引擎，仅修改 stride）：

```cpp
// 原始 KV: [B, G, S, d_h]，strides: [G*S*d_h, S*d_h, d_h, 1]
// 目标逻辑 shape: [B, H, S, d_h]
// head 维 stride 置 0，多个 Q 头映射同一 KV 内存块
std::vector<int64_t> new_shape  = {B, H, S, d_h};
std::vector<int64_t> new_stride = {G*S*d_h, 0, d_h, 1};  // head dim stride=0
```

|实现方式|内存行为|适用场景|
|---|---|---|
|repeat_interleave|实际复制，$H/G \times$ 内存|训练、调试|
|stride trick|零拷贝，原始大小|推理引擎（vLLM/TRT-LLM）|
|FlashAttention-2|原生 GQA 支持，内部按 group 分块加载 KV|生产推理|

### 7.2 实际内存计算示例（LLaMA-3 8B，$L=32$，$G=8$，$d_h=128$，fp16）

$$M_{KV} = 2 \times 32 \times 8 \times S \times 128 \times 2\ \text{bytes}$$

|序列长度 $S$|KV Cache 大小|
|---|---|
|4,096|512 MB|
|32,768|4 GB|
|131,072|16 GB|

---

## 8. 面试题库

### 8.1 概念辨析类

**Q1：MHA/MQA/GQA 的本质区别？**

KV 头数：$H$ / $1$ / $G$。本质是 KV Cache 内存与模型表达能力的 trade-off。

**Q2：GQA 为什么比 MQA 质量更好？**

多组 KV 保留了头间多样性。不同 Q 组可以关注不同语义子空间，单组 KV 构成表达瓶颈。

**Q3：MLA 与 GQA 的压缩思路有何本质不同？**

GQA：减少头数，保留每头维度，牺牲多样性。MLA：保留全部头，压缩 KV 特征维度到低秩空间，缓存压缩隐向量而非 K/V 本身，质量损失更小。

**Q4：MLA 为什么不能直接对压缩向量施加 RoPE？**

RoPE 是位置相关的旋转变换，需在还原后的 K 上施加特定位置的旋转矩阵。若缓存压缩向量，历史 token 的位置信息无法在还原时正确注入。解法是 Decoupled RoPE，单独维护一组携带位置信息的 $k^R$。

**Q5：MQA 共享 KV 为何在理论上可行，质量又为何下降？**

可行性：各 head 的 $Q_i$ 不同，$A_i = \text{softmax}(Q_i K^\top / \sqrt{d_k})$ 仍各异，差异由 Q 的多样性承载。质量下降：Key 子空间从 $H \cdot d_k$ 维压缩至 $d_k$ 维，head-specific 语义特征检测能力丢失；Value 多样性消失；共享参数导致梯度干扰，$W^K/W^V$ 无法对各 head 分别特化。

### 8.2 推理优化类

**Q6：推理引擎中 GQA 的 KV expand 为什么用 stride trick？**

repeat_interleave 产生 $H/G$ 倍内存复制，长序列下不可接受。stride trick 仅修改张量元数据，多个 Q 头逻辑上映射同一块 KV 内存，零拷贝。

**Q7：FlashAttention-2 如何支持 GQA？**

FA2 在 CUDA kernel 内以 group 为单位分块加载 KV，Q 头在 group 内展开计算，不做显式 expand，HBM 访问量按 $G$ 而非 $H$ 计算。

**Q8：Tensor Parallelism 下 GQA 的切分约束？**

$G$ 必须整除 TP degree。若 $G < \text{TP}$，KV 头无法均匀切分，需在每个 rank 上复制完整 KV。LLaMA-3 70B（$G=8$，TP=8）恰好每 rank 一个 KV 头。

**Q9：PagedAttention 在 GQA 下的内存变化？**

KV Block 按 `num_kv_heads` 分配，Block 大小正比于 $G/H$，内存池总量等比缩减，碎片率不变但绝对浪费量减少。

### 8.3 训练/转换类

**Q10：如何将已训练的 MHA 模型转换为 GQA？**

GQA 论文方法：对同组内多个 KV 头做 mean pooling 作为初始化，再以约 5% tokens 数据量微调恢复质量。局限：大模型（70B+）质量损失不可忽视，不如从头训练 GQA。

**Q11：GQA 中 $G$ 值如何选择？**

|约束|要求|
|---|---|
|TP 并行|$G$ 整除 TP degree，且 $G \geq P$|
|内存预算|$G$ 越小越省|
|质量下限|$G \geq H/8$ 通常接近 MHA|
|工程惯例|$G = H/8$ 或 $G = H/4$|

### 8.4 手写代码类

**Q12：手写标准 Scaled Dot-Product Attention（含 mask）**

```cpp
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

// Q: [seq_q, d_k], K: [seq_k, d_k], V: [seq_k, d_v]
// mask: [seq_q, seq_k]，true 表示该位置被屏蔽（置 -inf）
std::vector<float> scaled_dot_product_attention(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    const std::vector<bool>&  mask,
    int seq_q, int seq_k, int d_k, int d_v)
{
    const float scale   = 1.0f / std::sqrt(static_cast<float>(d_k));
    const float neg_inf = -std::numeric_limits<float>::infinity();

    // Step 1: S = Q @ K^T * scale  →  [seq_q, seq_k]
    std::vector<float> S(seq_q * seq_k, 0.0f);
    for (int i = 0; i < seq_q; ++i)
        for (int j = 0; j < seq_k; ++j) {
            float dot = 0.0f;
            for (int k = 0; k < d_k; ++k)
                dot += Q[i * d_k + k] * K[j * d_k + k];
            S[i * seq_k + j] = mask[i * seq_k + j] ? neg_inf : dot * scale;
        }

    // Step 2: softmax（数值稳定，减去行最大值）
    std::vector<float> P(seq_q * seq_k);
    for (int i = 0; i < seq_q; ++i) {
        float max_val = *std::max_element(
            S.begin() + i * seq_k, S.begin() + (i + 1) * seq_k);
        float sum = 0.0f;
        for (int j = 0; j < seq_k; ++j) {
            P[i * seq_k + j] = std::exp(S[i * seq_k + j] - max_val);
            sum += P[i * seq_k + j];
        }
        for (int j = 0; j < seq_k; ++j)
            P[i * seq_k + j] /= sum;
    }

    // Step 3: O = P @ V  →  [seq_q, d_v]
    std::vector<float> O(seq_q * d_v, 0.0f);
    for (int i = 0; i < seq_q; ++i)
        for (int v = 0; v < d_v; ++v)
            for (int j = 0; j < seq_k; ++j)
                O[i * d_v + v] += P[i * seq_k + j] * V[j * d_v + v];

    return O;
}
```

### 8.5 手推计算类

**Q13：LLaMA-3 70B，$H=64$，$G=8$，fp16，batch=1，$S=8192$，KV Cache 多大？**

$$M = 2 \times L \times G \times S \times d_h \times 2\ \text{bytes}$$

$$= 2 \times 80 \times 8 \times 8192 \times 128 \times 2$$

$$= 2{,}684{,}354{,}560\ \text{bytes} \approx 2.5\ \text{GB}$$

若 MHA（$G=64$）：$\approx 20\ \text{GB}$，节省 **8×**

---

## 9. 全局对比总览

|维度|MHA|MQA|GQA|MLA|
|---|---|---|---|---|
|KV 头数|$H$|$1$|$G$|逻辑 $H$，物理压缩|
|压缩方式|无|减少头数|减少头数|低秩降维|
|Cache 内容|$K, V$|$K, V$（单头）|$K, V$（$G$ 头）|$c^{KV}$ + $k^R$|
|RoPE 施加位置|Q、K|Q、K|Q、K|Decoupled（$k^R$ 单独）|
|质量损失|基准|大|中|小|
|代表模型|BERT, GPT-2|Falcon, PaLM|LLaMA-3, Mistral|DeepSeek-V2/V3|
