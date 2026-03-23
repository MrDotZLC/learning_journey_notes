## 1. 量化的基本原理

### 1.1 量化定义

量化将浮点激活值 $x \in \mathbb{R}$ 映射至有限离散集合 $\mathcal{Q} \subset \mathbb{Z}$：

$$ \hat{x} = \text{clamp}!\left(\text{round}!\left(\frac{x - z}{s}\right),\ q_{\min},\ q_{\max}\right) $$

其中 $s$ 为缩放因子（scale），$z$ 为零点（zero-point）。反量化（Dequantization）：

$$ \tilde{x} = s \cdot \hat{x} + z $$

**量化误差**定义为 $\epsilon = x - \tilde{x}$，其大小受 $s$ 与 $z$ 选取质量的直接影响。

### 1.2 内存收益

BF16 → INT8：节省 $2\times$；BF16 → INT4：节省 $4\times$；BF16 → INT2：节省 $8\times$；BF16 → NVFP4：节省 $4\times$（相对于 BF16）、$2\times$（相对于 FP8）。

### 1.3 K 与 V 的分布差异（关键观测）

**Key Cache 的通道级异常值（Channel Outlier）**：

- Key 矩阵在特定通道（channel）上存在持续性的大幅度异常值（outliers），与 token 位置无关，与通道索引强相关；
- 量化 Key 应优先使用**Per-channel 量化**（每通道独立 scale）；
- Per-token 量化对 Key 精度损失显著更大（KIVI, ICML 2024 实测）。

**Value Cache 无固定通道模式**：

- Value 矩阵无显著通道级异常，异常值随 token 位置变化；
- 量化 Value 应使用**Per-token 量化**（每 token 独立 scale）；
- Per-channel 量化对 Value 精度损失严重（OB 2, KIVI）。

> 【图示占位】：Key Cache 与 Value Cache 各层分布可视化，展示 Key 在特定通道的固定大幅度异常（竖条纹）与 Value 无固定模式的分布差异。

---

## 2. 量化粒度分类

### 2.1 Per-tensor 量化

整个 KV Tensor 共享一个 $(s, z)$：

$$ s = \frac{\max(x) - \min(x)}{2^b - 1}, \quad z = -\text{round}!\left(\frac{\min(x)}{s}\right) $$

**优点**：实现最简单，硬件加速最友好（单一 scale 无需额外存储）；**缺点**：精度最差，无法应对局部异常值。

### 2.2 Per-token 量化

每个 token（即 $\mathbf{k}_t \in \mathbb{R}^{n_{\text{heads}} \cdot d_h}$ 或 $\mathbf{v}_t$）独立一组 $(s_t, z_t)$：

$$ s_t = \frac{\max(\mathbf{k}_t) - \min(\mathbf{k}_t)}{2^b - 1} $$

适合 **Value Cache**，存储开销增加 $O(T)$ 个 scale（通常可忽略）。

### 2.3 Per-channel 量化

每个通道 $c$（即 $k_{\cdot, c}$ 跨所有 token）共享一组 $(s_c, z_c)$：

$$ s_c = \frac{\max_t(k_{t,c}) - \min_t(k_{t,c})}{2^b - 1} $$

适合 **Key Cache**，需离线或在线估计通道统计量。

### 2.4 Group 量化

将通道分组（group size $g$），组内共享 $(s, z)$，在 per-tensor 与 per-channel 之间折中：

$$ s_{c / g} = \frac{\max_{c \in \text{group}} - \min_{c \in \text{group}}}{2^b - 1} $$

KVQuant 采用 $g = 64$ 的 Group-NUQ（非均匀量化）。

---

## 3. 代表性方法详解

### 3.1 KIVI（ICML 2024）

**核心设计**：非对称 2-bit 量化（Key per-channel，Value per-token），保留近端 Local Window（默认 32 tokens）在 FP16，避免 Recent Token 精度损失：

$$ \mathbf{K}_{\text{cache}} = \underbrace{\text{INT}b_{\text{per-ch}}(\mathbf{K}_{\text{hist}})}_{\text{历史 token}} \oplus \underbrace{\mathbf{K}_{\text{local}}}_{\text{FP16}} $$

$$ \mathbf{V}_{\text{cache}} = \underbrace{\text{INT}b_{\text{per-tok}}(\mathbf{V}_{\text{hist}})}_{\text{历史 token}} \oplus \underbrace{\mathbf{V}_{\text{local}}}_{\text{FP16}} $$

**实测（KIVI-2bit，Llama-2-7B）**：显存节省约 $2.6\times$，LongBench 性能与 FP16 持平。主要限制：纯 2-bit 时特定任务出现 token flipping（关键 token 被错误替换，影响推理链）。

**HuggingFace 集成**：KIVI 的 per-channel Key 量化已被 HuggingFace Transformers KV Cache 量化采用（2024 年 6 月）。

---

### 3.2 KVQuant（NeurIPS 2024）

**核心贡献**：处理 Key 激活中存在的**重尾异常值（Heavy-tailed Outliers）**。

**非均匀量化（NUQ, Non-Uniform Quantization）**：

基于敏感度加权而非等间隔划分量化区间：

$$ \ell_{q_i} = \arg\min_{\ell} \sum_{x \in \text{bin}_i} w(x) \cdot (x - \ell)^2 $$

其中 $w(x)$ 为敏感度权重（由校准集上的 Hessian 对角估计），使异常值附近的量化区间更密集。

**Pre-RoPE Key 量化**：

RoPE（旋转位置编码）施加于 Key 后会引入额外分布旋转，增大量化误差。KVQuant 对 Pre-RoPE 的 Key 激活量化，在 RoPE 施加前完成：

$$ \hat{\mathbf{k}}^{\text{pre}} = \text{Quantize}(\mathbf{k}^{\text{pre}}), \quad \tilde{\mathbf{k}} = R_{\theta}(p) \cdot \hat{\mathbf{k}}^{\text{pre}} $$

**稀疏异常值处理**：将极端异常值（如 $|x| > 3\sigma$）单独以 FP16 存储，主体以低比特存储。

**实测**：3-bit KVQuant 在 Llama-7B 上实现 $4.8\times$ KV Cache 压缩，困惑度降低 $< 0.1$（对比 FP16 基线）；NUQ-2bit 可在单 A100 80GB 上支持 Llama-7B 的 $10\text{M}$ token 上下文。

---

### 3.3 KITTY（2025）

**问题**：KIVI 和 KVQuant 在 2-bit 下精度损失仍较大，尤其在推理（Reasoning）任务。

**核心策略：通道精度提升（Channel Boost）**

将 Key Cache 中特定比例（Boost Rate）的高重要性通道从 INT2 提升至 INT4：

$$ \mathbf{K}_{c} = \begin{cases} \text{INT4}(\mathbf{K}_c) & c \in \mathcal{C}_{\text{boost}} \ \text{INT2}(\mathbf{K}_c) & c \notin \mathcal{C}_{\text{boost}} \end{cases} $$

其中 $\mathcal{C}_{\text{boost}}$ 由 Hessian 对角估计（通道敏感度）确定，默认 Boost Rate = $12.5%$（KITTY）或 $25%$（KITTY-Pro）。

**实测（Qwen3-8B，A100 80GB）**：相同内存预算下，KITTY-Pro 对比 FP16 基线启用 $8\times$ 更大 batch size，吞吐提升 $2.1\times \to 4.1\times$；在 32K 上下文 GSM8K 任务上精度接近 FP16。

---

### 3.4 KVTuner（ICML 2025）

**问题**：各层、各模型对 Key/Value 量化精度的敏感度不一致，统一精度策略（如全层 INT4 K + INT8 V）存在过压缩或欠压缩。

**层间混合精度自动搜索**：

KVTuner 通过代理指标（Proxy Metric）——量化后注意力分数误差估计——为每层独立搜索最优 $(b_K^{(l)},\ b_V^{(l)})$ 精度对：

$$ \mathcal{E}^{(l)} = \left| \mathbf{A}^{(l)} - \hat{\mathbf{A}}^{(l)} \right|_F $$

其中 $\hat{\mathbf{A}}^{(l)}$ 为使用量化 KV 计算得到的注意力矩阵，$\mathcal{E}^{(l)}$ 越小说明该精度配置对该层损失越小。

**关键发现**：

- 绝大多数模型（Qwen、Llama、Mistral）仅在 INT2 Key 时出现显著困惑度上升；
- 但 Qwen2.5-{7B, Math-7B}-Instruct 即使在 INT4 Key 下也出现精度下降，需特殊处理；
- Key Cache 精度对模型质量的影响显著大于 Value Cache（Key 承担了更多的注意力分布决定权）；
- K8V2（5-bit 平均）性能等于甚至优于 K4V8（同样 5-bit），且额外节省 $12.5\%$ 内存。

---

### 3.5 FP8 量化（生产实践）

**格式**：

|格式|位宽|符号|指数|尾数|动态范围|
|---|---|---|---|---|---|
|BF16|16|1|8|7|极大|
|FP8 E4M3|8|1|4|3|中等（推荐，精度优先）|
|FP8 E5M2|8|1|5|2|大（范围优先）|

**vLLM 集成**：

```python
from vllm import LLM
# Per-tensor FP8，无需校准
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", kv_cache_dtype="fp8")

# Per-attention-head FP8（仅 Flash Attention 后端，需校准）
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    kv_cache_dtype="fp8",
    calculate_kv_scales=True,          # 从 warmup batch 估计 scale
)
```

FP8 E4M3 在 Hopper（H100）及以上 GPU 上有原生硬件支持，CUDA 11.8+ 可用，内存节省 $2\times$（相对 BF16）。

---

### 3.6 NVFP4（NVIDIA Blackwell，2025）

**格式规格**：1 符号位 + 2 指数位 + 1 尾数位，组级缩放因子（group size = 16），缩放因子本身以 FP8 E4M3 存储。

**内存收益**：相对 FP8 节省 $2\times$，相对 BF16 节省 $4\times$。

**工作流**：NVFP4 存储 → 注意力计算前反量化至 FP8 → 注意力计算在 FP8 域执行。

**精度评估**（Qwen3-480B-A35B）：LiveCodeBench、MMLU-PRO、MBPP、Ruler 64K 相对 FP16 基线准确率损失均 $< 1\%$。

**适用场景**：Multi-agent、MoE 大规模部署，对内存容量和带宽压力极高的场景（Blackwell GPU 原生支持）。

---

### 3.7 GEAR（量化 + 低秩 + 稀疏联合）

**核心思路**：对量化误差本身再进行压缩，而非只处理原始激活。

三阶段分解：

1. **均匀量化（主体）**：对绝大多数条目执行 INT4 均匀量化；
2. **低秩残差（近似误差）**：对量化残差矩阵 $\mathbf{E} = \mathbf{K} - \hat{\mathbf{K}}_{\text{INT4}}$ 做 SVD 低秩近似：

$$ \mathbf{E} \approx \mathbf{U}_r \boldsymbol{\Sigma}_r \mathbf{V}_r^\top $$

3. **稀疏矩阵（处理异常值误差）**：用稀疏矩阵 $\mathbf{S}$ 补充低秩近似无法覆盖的极端异常值误差。

最终重建：$\tilde{\mathbf{K}} = \hat{\mathbf{K}}_{\text{INT4}} + \mathbf{U}_r \boldsymbol{\Sigma}_r \mathbf{V}_r^\top + \mathbf{S}$

---

## 4. 混合精度策略：重要 Token 保留高精度

**MiKV（Mixed-precision KV-Cache）**：将被 Eviction 判定为"次重要"但不立即驱逐的 token 降精度（INT2）保留，重要 token 保持 FP16。驱逐与量化的联合优化。

**IntactKV / KIVI-HQQ**：静态保留 Prefill 的前 $n$ 个 token（Prefix token，常见 $n=32$）在 FP16，原因是 Attention Sink 等重要 token 多位于序列首部，对量化误差敏感度更高。

**QAQ（Quality Adaptive Quantization）**：对 Key Cache 使用自适应量化，用注意力历史窗口预测未来注意力分数，对将来可能重要的 token 提前保持高精度。

---

## 5. 量化误差的传播分析

层间误差累积（Layer-wise Error Propagation）：设第 $l$ 层量化后注意力输出误差为 $\delta^{(l)}$，由于 Transformer 层之间是串联关系：

$$ \delta^{(l+1)} \approx \delta^{(l)} + \nabla_{\mathbf{h}^{(l)}} \mathcal{L} \cdot \delta^{(l)} $$

即误差在前向传播中可能累积放大。KVTuner 的实验图示（Figure 3）显示，Key Cache 从 8-bit 降至 4-bit 再降至 2-bit 时，平均注意力分数误差分别增大约 $4.6\times$，且误差分布在不同层间差异显著，验证了层间混合精度的必要性。

---

## 6. 量化与 Eviction 的正交叠加

量化（每条目字节数↓）与 Eviction（条目数↓）正交，二者联合可获得更大压缩比：

$$ \text{实际内存} = \frac{k}{T} \cdot \frac{b_{\text{quant}}}{b_{\text{fp16}}} \cdot M_{\text{KV,FP16}} $$

其中 $k/T$ 为 Eviction 保留比例，$b_{\text{quant}}/b_{\text{fp16}}$ 为量化压缩比。

示例：$40\%$ token 保留（H2O）× INT4（$4\times$）× 原始 42 GB = $42 \times 0.4 / 4 \approx 4.2$ GB。

---

## 7. 各方法对比

|方法|精度|Key 粒度|Value 粒度|异常值处理|硬件支持|
|---|---|---|---|---|---|
|KIVI-2bit|INT2|Per-channel|Per-token|Local Window FP16|自定义 CUDA kernel|
|KVQuant-3bit|INT3|Per-channel + NUQ|Per-token|稀疏 FP16 残差|自定义 kernel|
|KITTY-Pro|~3.5bit|Per-channel 混合|Per-token|通道 Boost 到 INT4|自定义 kernel|
|KVTuner|层间混合|层自适应|层自适应|层间精度搜索|自定义 kernel|
|vLLM FP8|FP8|Per-tensor/head|Per-tensor/head|—|原生 H100/A100|
|NVFP4|FP4|Group-wise FP8 scale|Group-wise FP8 scale|—|原生 Blackwell|
|GEAR|INT4+低秩+稀疏|Per-token|Per-token|低秩+稀疏误差补偿|部分 CUDA|

---

## 8. 文献索引

|方法|论文|会议/期刊|年份|
|---|---|---|---|
|KIVI|A Tuning-Free Asymmetric 2bit Quantization for KV Cache|ICML|2024|
|KVQuant|Towards 10 Million Context Length LLM Inference with KV Cache Quantization|NeurIPS|2024|
|KITTY|Accurate and Efficient 2-Bit KV Cache Quantization with Channel Boost|arXiv|2025|
|KVTuner|Sensitivity-Aware Layer-wise Mixed Precision KV Cache Quantization|ICML|2025|
|GEAR|Generative Inference with Approximation Error Reduction|arXiv|2024|
|QAQ|Quality Adaptive Quantization for KV Cache|arXiv|2024|
|MiKV|Mixed-precision KV-Cache Compression|arXiv|2024|
|NVFP4 KV Cache|Optimizing Inference for Long Context with NVFP4 KV Cache|NVIDIA Tech Blog|2025|
|ZipCache|Accurate and Efficient KV Cache Quantization with Salient Token Identification|arXiv|2024|
