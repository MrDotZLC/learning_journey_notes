## 1. 问题形式化

### 1.1 Eviction Policy 定义

设时刻 $i$ 时已保留的 KV 索引集合为 $S_{i-1}$，预算上限为 $k$（$k \ll T$），Eviction Policy 为映射：

$$ g : S_{i-1} \cup {i} \to S_i, \quad |S_i| = k $$

### 1.2 注意力稀疏性——驱逐的理论依据

设第 $l$ 层第 $h$ 个 head 的注意力权重为：

$$ \mathbf{A}^{(l,h)}_{i,j} = \frac{\exp!\left(\mathbf{q}^{(l,h)}_i \cdot \mathbf{k}^{(l,h)\top}_j / \sqrt{d_h}\right)}{\sum_{j'} \exp!\left(\mathbf{q}^{(l,h)}_i \cdot \mathbf{k}^{(l,h)\top}_{j'} / \sqrt{d_h}\right)} $$

**幂律分布观测**：$\sum_i \mathbf{A}_{i,j}$ 服从幂律分布——少量 token（**Heavy Hitter, H²**）获得绝大部分注意力权重，其余 token 贡献趋近于零。设置 $1\%$ 最大值作为阈值，实测稀疏率超过 $95\%$（H2O, NeurIPS 2023）。

**重要性持久性假设（Importance Persistence Hypothesis）**：H² token 在跨层与跨 step 间具有一定稳定性，为 Eviction 的可行性提供依据。

---

## 2. 两阶段打分框架

所有方法均遵循：

**评分阶段**：利用 $m$ 个历史 Query $\mathbf{q}_1, \ldots, \mathbf{q}_m$，对每个 KV 位置 $j$ 计算重要性得分 $s_{t,j}$。

**聚合阶段**：将多次观测聚合为单一分值 $\hat{s}_j$，默认均值聚合：

$$ \hat{s}_j = \frac{1}{m} \sum_{t=1}^{m} s_{t,j} $$

保留 $\hat{s}$ 最高的 $k$ 个位置，驱逐其余。

---

## 3. 策略一：StreamingLLM（滑动窗口 + Attention Sink）

### 3.1 Attention Sink 现象

大量注意力权重聚集在 `<BOS>` 等首部 token，即使其语义内容有限。原因：Softmax 须分配非零权重于所有位置，初始 token 成为"默认倾向目标"。该现象在 Llama、GPT-NeoX、Falcon 等主流模型中普遍存在。

### 3.2 保留集合

$$ S_i = \underbrace{{0, 1, \ldots, n_{\text{sink}}-1}}_{\text{Sink Tokens}} \cup \underbrace{{i - w + 1, \ldots, i}}_{\text{Recent Window}} $$

默认 $n_{\text{sink}} = 4$，$w$ 为滑动窗口大小，总预算 $k = n_{\text{sink}} + w$。

### 3.3 优缺点

**优点**：实现极简，内存完全受控，支持无限长序列推理。

**缺点**：无法保留中间语义信息，长距离依赖（中段事实、代码逻辑）丢失严重，PPL（困惑度）在长文档任务上显著劣于 H2O。

> 【图示占位】：StreamingLLM 保留区域示意，横轴为序列位置（0 → T），深色矩形标注 Sink 区与 Recent 窗口，中间灰色为被驱逐区域。

---

## 4. 策略二：H2O（Heavy-Hitter Oracle）

### 4.1 问题建模——动态次模优化

设函数 $F(S) = \sum_i \max_{j \in S} \mathbf{A}_{i,j}$，可证 $F$ 为单调次模函数（Monotone Submodular Function）：

$$ F(S \cup {e}) - F(S) \geq F(T \cup {e}) - F(T), \quad \forall S \subseteq T $$

贪婪算法可获 $(1 - 1/e) \approx 63.2\%$ 最优比的近似解。

### 4.2 重要性度量

对 token $j$，以所有 Query step 的累积注意力分数为重要性代理：

$$ \text{score}(j) = \sum_{i=1}^{T} \mathbf{A}_{i,j} $$

### 4.3 双集合 Greedy 算法

维护 Heavy Hitter 集 $\mathcal{H}$（大小 $k_{\text{HH}}$）与 Recent 集 $\mathcal{R}$（大小 $k_{\text{recent}}$），总预算 $k = k_{\text{HH}} + k_{\text{recent}}$：

1. 新 token $i$ 到来，加入 $\mathcal{R}$；
2. 若 $|\mathcal{R}| > k_{\text{recent}}$，将 $\mathcal{R}$ 中最旧者 $j^*$ 弹出；
3. 更新 $j^*$ 的累积分数并与 $\mathcal{H}$ 中最低分者比较，若 $\text{score}(j^*) > \min_{j \in \mathcal{H}} \text{score}(j)$，替换之；
4. 否则驱逐 $j^*$。

### 4.4 实测性能

OPT-6.7B，保留 $20\%$ Heavy Hitter 时：对比 Full Cache 性能损失极小，吞吐量提升约 $29\times$（对比 DeepSpeed-ZeRO Inference）。

---

## 5. 策略三：SnapKV（Prefill 结束后一次性驱逐）

### 5.1 核心洞察

Prefill 阶段末尾的若干 Query（"观察窗口"）已足以预测哪些历史 token 在后续 Decode 中重要。无需在 Decode 阶段逐步驱逐，而是**在 Prefill 结束后一次性确定保留集**，此后 KV Cache 大小固定。

### 5.2 评分公式

设观察窗口大小为 $w_{\text{obs}}$，对位置 $j$：

$$ s_j^{\text{obs}} = \frac{1}{w_{\text{obs}}} \sum_{i=T-w_{\text{obs}}+1}^{T} \mathbf{A}_{i,j} $$

### 5.3 Max Pooling 平滑

防止分值在相邻位置间剧烈抖动（受 Rotary Positional Embedding 影响），引入核大小为 $2r+1$ 的 Max Pooling：

$$ \tilde{s}_j = \max_{|j'-j| \leq r} s_{j'}^{\text{obs}}, \quad r = 3\ (\text{默认 kernel size}=7) $$

取 $\tilde{s}$ Top-$k$ 位置保留，拼接观察窗口自身：

$$ S_{\text{final}} = \text{TopK}_{k-w_{\text{obs}}}(\tilde{s}_{0:T-w_{\text{obs}}}) \cup {T-w_{\text{obs}}, \ldots, T-1} $$

### 5.4 优缺点

**优点**：Decode 阶段内存固定，Prefill 后驱逐一次完成，延迟稳定。

**缺点**：对观察窗口有强依赖；若关键信息集中于序列头部（如长文档前置摘要），窗口尾部评分对其权重系统性偏低。

> 【图示占位】：SnapKV 流程图，横轴为序列位置，纵轴为评分值；左侧展示均值分数 $s_j^{\text{obs}}$，右侧展示 Max Pooling 平滑后的 $\tilde{s}_j$，底部标注 Top-k 保留位置（深色）与驱逐位置（浅色）。

---

## 6. 策略四：PyramidKV（层间金字塔预算分配）

### 6.1 核心发现

底层注意力分布均匀（低稀疏度），高层注意力高度集中（高稀疏度），因此各层所需保留的 KV 条目数应随层深增加而减少，形如金字塔。

### 6.2 预算分配公式

设底层预算 $k_{\max}$，顶层预算 $k_{\min}$，第 $l$ 层（$l = 1, \ldots, L$）的预算：

$$ k^{(l)} = k_{\max} - \frac{(k_{\max} - k_{\min})}{L - 1} \cdot (l - 1) $$

总预算约束：

$$ \sum_{l=1}^{L} k^{(l)} = K_{\text{total}} $$

各层内部使用 SnapKV 风格（观察窗口 + Max Pooling + Top-$k^{(l)}$）执行驱逐，差异仅在预算 $k^{(l)}$。

### 6.3 实测（LLaMA-3-8B，LongBench）

保留 $12\%$ KV Cache（预算 2048 tokens）时性能接近 Full Cache；极端 $0.7\%$ 下仍优于同等预算的 H2O 和 SnapKV。

---

## 7. 策略五：AdaKV（Head-wise 自适应预算）

### 7.1 问题

PyramidKV 层内预算在所有 head 间均匀分配，而不同 head 注意力集中度差异显著。高集中度 head 只需少量 token 即可恢复，低集中度 head 则需更大预算。

### 7.2 误差上界推导

第 $h$ 个 head 的 Eviction 后注意力输出误差（$L_1$ 范数形式）上界：

$$ \mathcal{L}^h \leq \sum_{j \notin S^h} \bar{a}^h_j \cdot |\mathbf{v}_j|_1 $$

其中 $\bar{a}^h_j$ 为 token $j$ 的平均注意力权重，$S^h$ 为 head $h$ 保留的集合。最小化误差上界等价于将预算分配给使 $\bar{a}^h_j$ 最大的位置，即优先保留高注意力权重 token。

### 7.3 Head-wise 预算分配

以各 head 的注意力熵（分布集中度的逆指标）为权重：

$$ \text{Entropy}^h = -\sum_j \bar{a}^h_j \log \bar{a}^h_j $$

集中度越高（熵越低）的 head 分得更少预算，在总预算约束下按熵反比分配：

$$ k^h \propto \text{Entropy}^h, \quad \sum_h k^h = K_{\text{layer}} $$

AdaKV 以 plug-and-play 方式接入 SnapKV 或 PyramidKV，仅修改预算分配步骤。

---

## 8. 策略六：CAKE（时空联合动态预算分配）

### 8.1 动机

PyramidKV 的层间分配依赖离线先验（固定线性金字塔），无法适应不同输入的实时注意力模式。CAKE 引入**空间信息**（注意力分布均匀程度）与**时间信息**（注意力分布随 step 的变化程度）动态决定各层预算。

### 8.2 空间信息（熵）

$$ \mathcal{H}^{(l)} = -\sum_j \bar{a}^{(l)}_j \log \bar{a}^{(l)}_j $$

$\mathcal{H}^{(l)}$ 高（分布均匀）→ 该层 token 无明显优先级 → 需更大预算。

### 8.3 时间信息（方差）

$$ \mathcal{V}^{(l)} = \text{Var}!\left(\bar{a}^{(l)}_j\right) $$

$\mathcal{V}^{(l)}$ 大（分布偏移明显）→ 该层 token 重要性不稳定 → 也需更大预算。

### 8.4 级联架构

CAKE 采用两级级联：先确定层间分配（基于 $\mathcal{H}^{(l)}$ 与 $\mathcal{V}^{(l)}$ 的联合指标），再在层内用 SnapKV 风格做 head-level 选择。

---

## 9. 策略七：NACL（Proxy Token + 随机混合策略）

### 9.1 注意力分数偏差问题

若仅用序列尾部 Query 计算重要性，首部 token 因距离远、注意力分值系统性偏低，被错误驱逐——即**注意力分数偏差（Attention Score Bias）**。

### 9.2 Proxy Token 评分

选取序列中若干分散的 Proxy Token（非连续，索引集 $\mathcal{P}$）作为"代理 Query"：

$$ s_j^{\text{proxy}} = \frac{1}{|\mathcal{P}|} \sum_{p \in \mathcal{P}} \mathbf{A}_{p,j} $$

覆盖序列全局，消除尾部偏差。

### 9.3 随机驱逐组件

按均匀分布随机抽取部分 token 保留，防止评分指标的系统性偏差导致某类 token 被永久忽视：

$$ \Pr[\text{retain token } j] \propto \text{uniform} $$

### 9.4 混合预算

总预算 $k = k_{\text{proxy}} + k_{\text{random}}$，两部分分别由上述策略产生后合并。

**实测**（ACL 2024）：在 $20\%$ 预算下，短文本任务准确率提升约 $80\%$，长文本提升约 $76\%$（对比仅用累积注意力分数的 H2O 基线）。

---

## 10. 策略八：DefensiveKV（鲁棒聚合）

### 10.1 重要性不稳定性

"重要性持久性假设"并不总成立：token 的重要性在不同 Query step 下可能大幅波动。简单均值聚合对异常观测值高度敏感。

### 10.2 截断均值聚合

丢弃最高和最低各 $\alpha$ 比例的观测后再取均值：

$$ \hat{s}_j^{\text{def}} = \text{TrimmedMean}!\left({s_{t,j}}_{t=1}^{m},\ \alpha\right) $$

默认 $\alpha = 0.1$（各端去掉 $10\%$），抵抗异常扰动，提升 Eviction 决策的鲁棒性。

---

## 11. 评分改进：LAVa Score

标准累积注意力分数忽略 Value 向量对输出的实际贡献幅度。LAVa（Layer-wise Attention and Value Score）引入联合度量：

$$ \text{LAVa}_{j}^{(l,h)} = \bar{a}^{(l,h)}_j \cdot |\mathbf{v}^{(l,h)}_j|_2 $$

兼顾"被注意的程度"与"被注意后的输出贡献幅度"，理论上是注意力输出 $L_1$ 误差更紧的上界，在数学推理等对精度敏感的任务上优于纯注意力分数方法。

---

## 12. 策略横向对比

|策略|驱逐时机|重要性指标|预算分配|主要缺陷|
|---|---|---|---|---|
|StreamingLLM|实时每 step|位置（Recency+Sink）|均匀/固定|丢失中间语义|
|H2O|实时每 step|累积注意力分数|均匀/固定|评分代价随 $T$ 增长|
|SnapKV|Prefill 后一次|窗口注意力均值+MaxPool|均匀/固定|依赖尾部窗口|
|PyramidKV|Prefill 后一次|同 SnapKV|层间金字塔（固定先验）|先验不适配所有输入|
|AdaKV|Prefill 后一次|同 SnapKV/PyramidKV|**Head-wise 熵驱动自适应**|需计算各 head 熵|
|CAKE|Prefill 后一次|窗口注意力+时空联合|**层间动态自适应**|计算 $\mathcal{H}/\mathcal{V}$ 额外开销|
|NACL|Prefill 后一次|Proxy Token+随机混合|均匀/固定|Proxy 选取影响效果|
|DefensiveKV|Prefill 后一次|截断均值聚合|任意|参数 $\alpha$ 需调优|
|LAVa|Prefill 后一次|注意力×Value 范数|层间动态|额外计算 $\|\mathbf{v}\|_2$|

---

## 13. FlashAttention 兼容性

标准 FlashAttention（Tiling 计算）不保存完整 $\mathbf{A}$ 矩阵，仅存 $\text{logsumexp}$，与需要读取完整注意力分数的驱逐方法存在冲突。

**适配方式**（NACL 论文 Algorithm 2）：在 FlashAttention 的 Tiling 循环中，额外累积对每个 Key 列的注意力求和：

$$ \text{AttnSum}_j \mathrel{+}= \sum_{i \in \text{tile}} \mathbf{A}_{i,j} $$

在 SRAM 内完成局部累积，避免 HBM 回写完整矩阵，保持内存访问效率。此适配已被 vLLM FlashInfer 后端部分支持。

---

## 14. 工程实现伪代码（SnapKV 风格，Python）

```python
import torch
import torch.nn.functional as F

def snapkv_evict(
    keys: torch.Tensor,      # [T, n_heads, d_head]
    values: torch.Tensor,    # [T, n_heads, d_head]
    queries: torch.Tensor,   # [T, n_heads, d_head]
    obs_window: int = 32,
    budget: int = 512,
    kernel_size: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    T, n_heads, d_head = keys.shape
    assert budget > obs_window, "budget must exceed obs_window"

    # 1. 观察窗口内 Query
    obs_q = queries[-obs_window:]  # [w, n_heads, d_head]

    # 2. 计算注意力分数 [w, n_heads, T]
    scale = d_head ** -0.5
    scores = torch.einsum('wnd,tnd->wnt', obs_q, keys) * scale
    scores = F.softmax(scores, dim=-1)

    # 3. 均值聚合 -> [n_heads, T]
    mean_scores = scores.mean(dim=0)

    # 4. Max Pooling 平滑 (沿 token 维度)
    # mean_scores: [n_heads, T] -> reshape for 1d pool
    pad = kernel_size // 2
    pooled = F.max_pool1d(
        mean_scores.unsqueeze(0),     # [1, n_heads, T]
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
    ).squeeze(0)                      # [n_heads, T]

    # 5. 排除观察窗口自身，选 Top-(budget-obs_window) 历史位置
    hist_scores = pooled[:, :-obs_window]    # [n_heads, T-w]
    topk_k = budget - obs_window
    # 跨 head 取平均后选 topk（SnapKV 原版策略）
    mean_pooled = hist_scores.mean(dim=0)    # [T-w]
    top_idx = mean_pooled.topk(topk_k).indices  # [topk_k]
    top_idx, _ = top_idx.sort()

    # 6. 拼接历史保留 + 观察窗口
    obs_idx = torch.arange(T - obs_window, T, device=keys.device)
    retained = torch.cat([top_idx, obs_idx])

    return keys[retained], values[retained]
```

---

## 15. 文献索引

|方法|论文|会议/期刊|年份|
|---|---|---|---|
|StreamingLLM|Efficient Streaming Language Models with Attention Sinks|ICLR|2024|
|H2O|Heavy-Hitter Oracle for Efficient Generative Inference|NeurIPS|2023|
|Scissorhands|Exploiting the Persistence of Importance Hypothesis|NeurIPS|2023|
|SnapKV|LLM Knows What You Are Looking for Before Generation|NeurIPS|2024|
|PyramidKV|Dynamic KV Cache Compression based on Pyramidal Info Funneling|arXiv|2024|
|AdaKV|Optimizing KV Cache Eviction by Adaptive Budget Allocation|NeurIPS|2025|
|CAKE|Cascading and Adaptive KV Cache Eviction with Layer Preferences|arXiv|2025|
|NACL|A General and Effective KV Cache Eviction Framework|ACL|2024|
|DefensiveKV|Taming the Fragility of KV Cache Eviction in LLM Inference|arXiv|2025|
|LAVa|Layer-wise KV Cache Eviction with Dynamic Budget Allocation|EMNLP Findings|2025|
