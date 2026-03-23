## 1. 低秩分解的理论基础

### 1.1 KV 矩阵的低秩结构

设第 $l$ 层 Key 矩阵 $\mathbf{K}^{(l)} \in \mathbb{R}^{T \times d_k}$，其奇异值分解（SVD）为：

$$ \mathbf{K}^{(l)} = \mathbf{U}^{(l)} \boldsymbol{\Sigma}^{(l)} \mathbf{V}^{(l)\top} $$

其中 $\mathbf{U}^{(l)} \in \mathbb{R}^{T \times d_k}$，$\boldsymbol{\Sigma}^{(l)} = \text{diag}(\sigma_1, \sigma_2, \ldots, \sigma_{d_k})$，$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{d_k}$。

**低秩假设**：若奇异值谱的能量集中于前 $r$ 个奇异值（$\sigma_{r+1}, \ldots \approx 0$），则可以低秩近似：

$$ \mathbf{K}^{(l)} \approx \mathbf{U}_r^{(l)} \boldsymbol{\Sigma}_r^{(l)} \mathbf{V}_r^{(l)\top}, \quad r \ll d_k $$

近似误差（Frobenius 范数）由 Eckart-Young 定理保证为最优：

$$ \left| \mathbf{K}^{(l)} - \mathbf{U}_r \boldsymbol{\Sigma}_r \mathbf{V}_r^\top \right|_F = \left(\sum_{i=r+1}^{d_k} \sigma_i^2\right)^{1/2} $$

### 1.2 层间奇异向量对齐（xKV 的核心发现）

相邻层的 KV Cache 矩阵之间，虽然 per-token 余弦相似度可能较低（MiniCache 假设不成立），但**主奇异向量（Dominant Singular Vectors）高度对齐**：

$$ \text{align}(\mathbf{U}_r^{(l)},\ \mathbf{U}_r^{(l')}) \gg \text{sim}(\mathbf{K}^{(l)}_t,\ \mathbf{K}^{(l')}_t), \quad l' = l+1,\ldots,l+G-1 $$

这意味着多层的 KV Cache 共享同一低秩子空间，可以跨层联合 SVD 进一步压缩。

> 【图示占位】：左图：某两相邻层 per-token 余弦相似度热力图（颜色较浅，表明整体相似度低）；右图：这两层的前 $r=8$ 个左奇异向量的内积矩阵（接近单位阵，表明奇异向量高度对齐）。

---

## 2. 方法一：Palu（层内低秩分解，ICLR 2025）

### 2.1 核心思路

对 Key/Value 投影权重矩阵做 SVD 分解，将高维 KV 投影映射到低维空间后缓存，推理时从低维恢复：

$$ \mathbf{W}_K \approx \mathbf{A}_K \mathbf{B}_K, \quad \mathbf{A}_K \in \mathbb{R}^{d \times r},\ \mathbf{B}_K \in \mathbb{R}^{r \times d_k} $$

则缓存的 Key 为低维表示 $\mathbf{c}_t = \mathbf{h}_t \mathbf{A}_K$（$\in \mathbb{R}^{r}$），推理时恢复：$\hat{\mathbf{k}}_t = \mathbf{c}_t \mathbf{B}_K$。

### 2.2 Group Low-Rank Decomposition（G-LRD）

M-LRD（每 head 独立 SVD）会导致非可忽略的精度下降，原因是 SVD 无法捕获跨 head 的共同信息。G-LRD 对多 head 联合分解：

设一组 $g$ 个 head 的 Key 投影权重竖向拼接为 $\mathbf{W}_{K,g} \in \mathbb{R}^{d \times (g \cdot d_h)}$，执行统一 SVD，共享左奇异矩阵 $\mathbf{A}_{K,g} \in \mathbb{R}^{d \times r}$，各 head 独立右奇异矩阵。

默认 group size $= 4$（GQA-based 模型如 LLaMA-3，Mistral）。

### 2.3 矩阵融合消除 Decode 延迟

RoPE 施加于 Key 前可与 $\mathbf{B}_K$ 融合（对非 RoPE 注意力），避免推理时的解压投影：

$$ \mathbf{W}_Q \cdot \mathbf{W}_K^\top \approx \mathbf{W}_Q \cdot \mathbf{B}_K^\top \cdot \mathbf{B}_K \cdot \mathbf{A}_K^\top \to \text{融合后直接用} \mathbf{c}_t \text{计算} QK $$

**Walsh-Hadamard Transform（WHT）**：G-LRD 后低秩坐标的分布出现系统性异常（前几维幅度异常大），影响后续量化。Palu 施加 WHT 消除异常：

$$ \hat{\mathbf{c}}_t = \text{WHT}(\mathbf{c}_t), \quad \text{WHT}^{-1}(\hat{\mathbf{c}}_t) = \mathbf{c}_t $$

### 2.4 自动秩分配（Automatic Rank Allocation）

根据各层 SVD 谱的能量集中度（Singular Value Ratio）分配压缩率：谱集中度高的层分配更低的秩（更激进压缩），保留谱分布均匀的层更高的秩。

### 2.5 实测

LLaMA-3-8B，LongBench：$50\%$ 压缩率（平均秩 $r = 0.5 d_k$）时，困惑度与 Full Cache 基本持平；Decode 延迟与全精度持平（矩阵融合消除开销）；相比 KVQuant，PALU 在相同内存预算下 LongBench 综合分更高。

---

## 3. 方法二：xKV（跨层 SVD，arXiv 2025）

### 3.1 核心思路

在 Palu 层内低秩分解的基础上，将相邻 $G$ 层的 KV Cache 联合 SVD，提取跨层共享的低秩子空间：

设第 $k$ 组（$G_k = {kG, kG+1, \ldots, kG+G-1}$）的 Key 矩阵横向拼接：

$$ \mathbf{X}_G = \left[ \mathbf{K}^{(kG)} \mid \mathbf{K}^{(kG+1)} \mid \cdots \mid \mathbf{K}^{(kG+G-1)} \right] \in \mathbb{R}^{T \times (G \cdot d_k)} $$

执行 SVD：

$$ \mathbf{X}_G \approx \mathbf{U}_r \boldsymbol{\Sigma}_r \mathbf{V}_r^\top $$

$\mathbf{U}_r \in \mathbb{R}^{T \times r}$ 为跨层**共享的低秩表示**（存储一份），$\mathbf{V}_r$ 的对应列块为各层的重建矩阵 $\mathbf{B}_{\ell_i}$：

$$ \hat{\mathbf{K}}^{(l)} \approx \mathbf{U}_r \boldsymbol{\Sigma}_r \mathbf{B}_{\ell}^\top $$

### 3.2 内存分析

不分组时，$G$ 层各存一份 $\mathbf{K}^{(l)} \in \mathbb{R}^{T \times d_k}$，共 $G \cdot T \cdot d_k$ 个元素。使用 xKV 后，只存共享的 $\mathbf{U}_r \in \mathbb{R}^{T \times r}$ 加各层重建矩阵 $\mathbf{B}_l \in \mathbb{R}^{r \times d_k}$（离线存储，小量），主存储从 $G \cdot T \cdot d_k$ 降至 $T \cdot r + G \cdot r \cdot d_k$：

$$ \text{压缩比} \approx \frac{G \cdot T \cdot d_k}{T \cdot r} = \frac{G \cdot d_k}{r} $$

Group size $G = 4$，$r = d_k / 4$ 时，压缩比约 $16\times$（保守实现约 $6\text{--}8\times$）。

### 3.3 工程权衡

xKV 需在 Prefill 结束后对每组层执行在线 SVD，计算开销不可忽视：实测（8K 上下文，RTX A6000）SVD 构建时间约为整个 Prefill 延迟的 $6\times$。CommonKV 通过离线参数共享消除此开销。

### 3.4 与 MLA 的兼容性

xKV 与 DeepSeek-Coder-V2 的 MLA 架构兼容，在编程任务上实现 $3\times$ 压缩而无性能下降。

---

## 4. 方法三：CommonKV（跨层参数共享，arXiv 2025）

### 4.1 对 xKV 的改进

xKV 的 Online SVD 开销过大。CommonKV 将 SVD 从在线激活空间转移到**离线权重空间**：

利用相邻层 $W_K^{(l)}$ 与 $W_K^{(l+1)}$ 的高相似性（CKA 度量），对权重矩阵做 SVD 后共享低秩子空间，生成一个更易合并的潜在 KV 表示：

$$ \mathbf{W}_K^{(l)} \approx \mathbf{W}_K^{(l+1)} \approx \mathbf{U}_{\text{shared}} \boldsymbol{\Sigma} \mathbf{V}^\top $$

推理时所有层使用共享的 $\mathbf{U}_{\text{shared}}$ 下投影，KV Cache 只存低维潜在向量。

### 4.2 自适应预算分配

基于余弦相似度动态分配各层组的压缩预算：相似度高的层组激进压缩，相似度低的层组保留更大秩，避免过压缩。

**实测**：以 $6\times$ 压缩率下，CommonKV 对比 xKV 在 LongBench 表现持平或略优，且 Decode 延迟与常规自回归推理无显著差别（因无 Online SVD）。

---

## 5. 方法四：CaM（Cache Merging，ICML 2024）

### 5.1 核心思路

Eviction 驱逐会丢失被驱逐 token 的信息。CaM 在驱逐时不直接丢弃，而是将被驱逐 token 的信息**合并到相邻保留 token 中**：

$$ \mathbf{k}_j^{\text{merged}} = \mathbf{k}_j + \sum_{i \in \text{evict}(j)} \alpha_i \cdot \mathbf{k}_i $$

其中 $\alpha_i$ 为合并权重（由注意力分数归一化），$\text{evict}(j)$ 为分配给 $j$ 的被驱逐 token 集合（通常为最近邻）。

### 5.2 信息保留的理论依据

注意力输出可以近似分解：

$$ \text{Attn}(\mathbf{q},\ \mathbf{K},\ \mathbf{V}) \approx \text{Attn}(\mathbf{q},\ \mathbf{K}_S,\ \mathbf{V}_S) + \text{correction term} $$

当被驱逐 token 的贡献可以由其邻近保留 token"代理"时，合并等价于对注意力输出做一阶 Taylor 近似补偿。

### 5.3 优缺点

**优点**：比纯驱逐保留更多信息，LongBench 长文本任务上明显优于 H2O；**缺点**：合并后 Key/Value 偏离原始分布，可能引入误差累积；合并操作本身有 $O(k)$ 计算开销。

---

## 6. 方法五：KVMerger（自适应 Gaussian 加权合并，arXiv 2024）

### 6.1 核心观测

Key 状态在**同一序列内相邻 token 间**具有高余弦相似度。这与 xKV 的跨层观测互补。

具体：在 LLaMA2-7B 的多层多任务测试中，相邻 Key 的平均余弦相似度超过 $0.95$，且此稀疏性与数据集无关（模型级属性），提供了设计合并集合的依据。

### 6.2 合并集合识别算法

基于贪婪聚类：

1. 以余弦相似度阈值 $\tau$（如 $0.9$）扫描序列；
2. 高相似度的连续 token 组成合并集 $\mathcal{M}_i$；
3. 合并集中注意力分数最高者为"枢轴（Pivot）"，其余为待合并成员；
4. Heavy Hitter 与 Attention Sink 排除在合并集之外（直接保留）。

### 6.3 Gaussian 核加权合并

对合并集 $\mathcal{M}_i = {j_0, j_1, \ldots, j_m}$（$j_0$ 为枢轴），合并后的 Key 为：

$$ \hat{\mathbf{k}}_{j_0} = \frac{\sum_{j \in \mathcal{M}_i} w(|j - j_0|) \cdot \mathbf{k}_j}{\sum_{j \in \mathcal{M}_i} w(|j - j_0|)} $$

$$ w(\Delta) = \exp!\left(-\frac{\Delta^2}{2\sigma^2}\right) $$

距离枢轴越近的 token 贡献越大，符合注意力的局部性先验。

### 6.4 实测

LLaMA2-7B-chat，LongBench：$50\%$ budget 下 KVMerger 优于 H2O 和 CaM；$35\%$ budget 下差距进一步扩大，体现合并相比纯驱逐的信息保留优势。

---

## 7. 方法六：D2O（动态判别操作，ICLR 2025）

### 7.1 统一驱逐 + 合并框架

D2O 将 Eviction 与 Merging 统一为可微分的动态操作，通过学习得分函数（Learned Scoring Function）动态判断每个 token 应执行"保留"、"驱逐"还是"合并"：

$$ \text{action}(j) = \arg\max_{{keep, evict, merge}} f_\theta(\mathbf{k}_j, \mathbf{v}_j, \text{context}) $$

其中 $f_\theta$ 为轻量化参数网络（线性层 + 激活）。三类操作中"合并"会将 token $j$ 的信息融入其最近邻保留 token，实现信息无损压缩。

### 7.2 与静态方法的对比

D2O 无需手动设定相似度阈值或分组规则，通过端到端训练学习最优策略。但需要 Fine-tuning 支持，增加了部署复杂度。

---

## 8. 策略总览与对比

|方法|技术路线|压缩维度|是否需要训练|典型压缩比|
|---|---|---|---|---|
|Palu|层内权重 SVD，G-LRD|头维度 $d_k$|无（Post-training）|2×|
|xKV|跨层激活 SVD|层数 $L$ × 头维度|无（Online SVD）|6–8×|
|CommonKV|跨层权重共享 SVD|层数 × 头维度|无（离线参数）|3–6×|
|CaM|驱逐时合并到邻居|Token 数 $T$|无|2–5×（与驱逐联合）|
|KVMerger|聚类后 Gaussian 加权合并|Token 数 $T$|无|2–3×|
|D2O|学习判别（保留/驱逐/合并）|Token 数 $T$|需 fine-tuning|自适应|
|LORC|渐进式低秩压缩|头维度 $d_k$|无|2–4×|
|FourierKV|Fourier 域频率压缩|头维度 $d_k$|无|2×|

---

## 9. 低秩分解与量化的叠加

Palu 指出，G-LRD 产生的低维潜在坐标因 SVD 的排序特性（前几维幅度大）会导致分布扭曲，直接量化精度差。解决方案：

1. WHT（Walsh-Hadamard Transform）消除异常后再量化；
2. 效果：G-LRD + INT4 量化对比 Full Cache FP16 的困惑度差距 $< 0.3$（Wikitext-2）。

xKV 在 MLA 场景下：MLA 本身已是低秩潜在向量，xKV 跨层 SVD 进一步压缩潜在向量维度，二者完全兼容（DeepSeek-Coder-V2 实测 $3\times$ 额外压缩）。

---

## 10. 工程实现要点：在线 SVD 的数值稳定性

### 10.1 Prefill 后一次性 SVD

```python
import torch

def xkv_compress(kv_group: list[torch.Tensor], rank: int) -> tuple:
    """
    kv_group: List of Key tensors for G layers, each [T, d_k]
    rank: target rank r
    Returns: shared U_r [T, r], layer-specific B_l list [r, d_k]
    """
    # 横向拼接 [T, G*d_k]
    X_G = torch.cat(kv_group, dim=-1)
    # SVD（使用 randomized SVD 降低计算开销）
    U, S, Vh = torch.linalg.svd(X_G, full_matrices=False)
    U_r = U[:, :rank]                           # [T, r] 共享表示
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]                         # [r, G*d_k]
    # 拆分各层重建矩阵
    d_k = kv_group[0].shape[-1]
    B_list = [Vh_r[:, i*d_k:(i+1)*d_k] for i in range(len(kv_group))]
    return U_r, S_r, B_list

def xkv_reconstruct(U_r: torch.Tensor, S_r: torch.Tensor, B_l: torch.Tensor) -> torch.Tensor:
    """重建第 l 层的 Key: [T, d_k]"""
    return (U_r * S_r) @ B_l
```

### 10.2 数值稳定性注意事项

- 使用 Randomized SVD（`torch.linalg.svd` with `driver='gesvda'` 或 sklearn 的 `randomized_svd`）在大 $T$ 下更高效；
- 拼接前对各层 KV 做层归一化（Layer-wise Scale），防止不同层幅度差异导致 SVD 偏向某些层；
- 秩 $r$ 的选择：按能量占比 $\sum_{i \leq r} \sigma_i^2 / \sum_i \sigma_i^2 \geq 0.95$ 自动确定。

---

## 11. 文献索引

|方法|论文|会议/期刊|年份|
|---|---|---|---|
|Palu|Compressing KV-Cache with Low-Rank Projection|ICLR|2025|
|xKV|Cross-Layer SVD for KV-Cache Compression|arXiv|2025|
|CommonKV|Compressing KV Cache with Cross-layer Parameter Sharing|arXiv|2025|
|CaM|Cache Merging for Memory-efficient LLMs Inference|ICML|2024|
|KVMerger|Model Tells You Where to Merge: Adaptive KV Cache Merging|arXiv|2024|
|D2O|Dynamic Discriminative Operations for Efficient Generative Inference|ICLR|2025|
|LORC|Low-rank Compression for LLMs KV Cache with Progressive Strategy|arXiv|2024|
|MiniCache|KV Cache Compression in Depth Dimension|NeurIPS|2024|
|KVSharer|Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing|arXiv|2024|
