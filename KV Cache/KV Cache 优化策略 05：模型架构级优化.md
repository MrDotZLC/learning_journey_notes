> 本文档描述的优化内嵌于模型结构，**需要从头预训练或大规模 Fine-tuning**，不可 Post-training 注入。了解这些架构是理解现代 LLM（LLaMA-3、Qwen、DeepSeek 等）KV Cache 行为的基础。

---

## 1. 基准：Multi-Head Attention（MHA）的 KV Cache 开销

标准 MHA 中，每个 token $t$ 需存储 $n_h$ 个 head 各自独立的 $(\mathbf{k}^{(h)}_t,\ \mathbf{v}^{(h)}_t)$，每 head 维度 $d_h$，KV Cache 总大小：

$$ M_{\text{MHA}} = 2 \cdot L \cdot n_h \cdot d_h \cdot T \cdot b \quad \text{(bytes)} $$

其中 $n_h \cdot d_h = d_{\text{model}}$（模型宽度）。$n_h$ 与 $d_{\text{model}}$ 成比例增长，KV Cache 压力随模型规模激增。

---

## 2. Multi-Query Attention（MQA，Shazeer 2019）

### 2.1 架构设计

所有 Query head 共享**唯一一组** Key/Value head（$n_{kv} = 1$）：

$$ \mathbf{q}^{(h)}_t = \mathbf{h}_t \mathbf{W}^{(h)}_Q,\quad h = 1,\ldots, n_h $$

$$ \mathbf{k}_t = \mathbf{h}_t \mathbf{W}_K,\quad \mathbf{v}_t = \mathbf{h}_t \mathbf{W}_V \quad (\text{全局共享一对 } W_K, W_V) $$

### 2.2 KV Cache 压缩比

$$ M_{\text{MQA}} = 2 \cdot L \cdot 1 \cdot d_h \cdot T \cdot b $$

$$ \frac{M_{\text{MQA}}}{M_{\text{MHA}}} = \frac{1}{n_h} $$

对 $n_h = 32$（如 LLaMA-7B）：KV Cache 压缩 $32\times$。

### 2.3 代价

单一 KV head 削减了注意力机制的表征容量，在下游任务（特别是推理、代码、长文档 QA）上性能低于 MHA。DeepSeek-V2 的消融实验直接对比 MHA vs MQA，MQA 在所有任务上均劣于 MHA。

### 2.4 应用模型

PaLM-540B（2022），Falcon，早期 LLM serving 优化场景。

---

## 3. Grouped-Query Attention（GQA，Ainslie et al. 2023）

### 3.1 架构设计

将 $n_h$ 个 Query head 分为 $g$ 个组，每组 $n_h / g$ 个 Query head 共享一对 KV head：

$$ n_{kv} = g, \quad \text{每个 KV head 服务 } n_h / g \text{ 个 Query head} $$

### 3.2 KV Cache 大小

$$ M_{\text{GQA}} = 2 \cdot L \cdot g \cdot d_h \cdot T \cdot b $$

$$ \frac{M_{\text{GQA}}}{M_{\text{MHA}}} = \frac{g}{n_h} $$

典型配置：LLaMA-3-8B（$n_h=32,\ g=8$），LLaMA-3-70B（$n_h=64,\ g=8$）。

### 3.3 MQA 到 MHA 的平滑插值

$g = 1$：退化为 MQA；$g = n_h$：退化为 MHA。GQA 可视为在 KV Cache 大小与模型质量之间的连续权衡旋钮。

### 3.4 从 MHA Checkpoint 转换

**Uptrain 策略**：从已训练的 MHA 模型出发，将 $n_h / g$ 个 Query head 的 KV head 做**均值池化（Mean Pooling）**初始化为 GQA 的共享 KV head，在少量数据上继续训练恢复质量（Ainslie et al. 2023 实验，GPT-3 风格模型，$5%$ 原始训练数据即可恢复 MHA 性能）。

### 3.5 与 Tensor Parallelism 的兼容性

GQA 的 $g$ 通常设为 Tensor Parallelism 度 $P$ 的整数倍，确保每个 GPU 分配到完整的 KV head 组：

$$ g = k \cdot P, \quad k \in \mathbb{Z}^+ $$

LLaMA-3-70B（$g=8,\ P=8$）：每 GPU 精确分配 1 个 KV head，无跨 GPU KV 同步开销。

### 3.6 应用模型

LLaMA-2/3、Mistral、Qwen2/2.5、Gemma2、Phi-3 等主流开源模型均采用 GQA。

---

## 4. Multi-head Latent Attention（MLA，DeepSeek-V2 2024）

### 4.1 设计动机

GQA 通过减少 KV head 数降低 Cache，但直接减少参数数量影响表征能力。MLA 的核心问题是：**是否可以在不减少参数量的情况下，压缩需要缓存的数据量？**

答案：通过低秩联合压缩（Low-Rank Joint Compression），只缓存低维潜在向量。

### 4.2 Down-Projection：生成潜在向量

对输入 hidden state $\mathbf{h}_t \in \mathbb{R}^{d}$，通过 Down-projection 生成 KV 的低维潜在表示：

$$ \mathbf{c}_t^{KV} = \mathbf{h}_t \mathbf{W}^{DKV}, \quad \mathbf{W}^{DKV} \in \mathbb{R}^{d \times d_c} $$

其中 $d_c \ll n_h \cdot d_h$（$d_c$ 为压缩维度，DeepSeek-V2 中 $d_c = 512$，而 $n_h \cdot d_h = 2048$）。

**缓存的是 $\mathbf{c}_t^{KV}$，而非展开后的 $\mathbf{K}, \mathbf{V}$。**

### 4.3 Up-Projection：从潜在向量恢复 KV

推理时，对每个 head $h$ 通过 Up-projection 恢复 Key 和 Value：

$$ \mathbf{k}^{(h)}_t = \mathbf{c}_t^{KV} \mathbf{W}^{UK,h}, \quad \mathbf{W}^{UK,h} \in \mathbb{R}^{d_c \times d_h} $$

$$ \mathbf{v}^{(h)}_t = \mathbf{c}_t^{KV} \mathbf{W}^{UV,h}, \quad \mathbf{W}^{UV,h} \in \mathbb{R}^{d_c \times d_h} $$

### 4.4 解耦 RoPE（Decoupled RoPE）

**问题**：RoPE 位置编码是位置敏感的，需要施加于 Key 上。若将 RoPE 直接施加于已压缩的 $\mathbf{c}_t^{KV}$，则无法从 Up-projection 中恢复正确的带位置编码的 Key（因为 Up-projection 矩阵可以被融合进 Query 侧，但融合后 RoPE 变得不可分离）。

**解决方案**：为每个 token 额外缓存一小段"RoPE 专用 Key"$\mathbf{k}_t^R$（$n_h$ 个 head 共用，MQA 风格），维度为 $d_h^R$（通常 $d_h^R = 64$）：

$$ \mathbf{k}_t^{(h)} = [\underbrace{\mathbf{c}_t^{KV} \mathbf{W}^{UK,h}}_{\text{非 RoPE 部分}} \mid \underbrace{\mathbf{k}_t^R}_{\text{RoPE 部分（共享）}}] $$

**实际缓存内容**：$\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ + $\mathbf{k}_t^R \in \mathbb{R}^{d_h^R}$，总维度远小于完整 KV。

### 4.5 KV Cache 大小对比

$$ M_{\text{MLA}} = 2 \cdot L \cdot (d_c + d_h^R) \cdot T \cdot b $$

对 DeepSeek-V2（$d_c=512,\ d_h^R=64,\ n_h=128,\ d_h=128$）：

$$ \frac{M_{\text{MLA}}}{M_{\text{MHA}}} = \frac{512 + 64}{128 \times 128} \approx \frac{576}{16384} \approx 3.5% $$

即 MLA 相对完整 MHA 压缩约 **$28\times$**（$1/0.035$），相对 GQA-8 约压缩 **$\frac{8 \times 128}{576} \approx 1.8\times$**。

> 【图示占位】：MHA / GQA / MQA / MLA 的 KV Cache 大小对比柱状图，横轴为方法名，纵轴为每 token 每层的 KV Cache 字节数（归一化为 MHA=1），展示 MLA 最小。

### 4.6 矩阵吸收（Matrix Absorption）：推理加速

Up-projection $\mathbf{W}^{UK,h}$ 与 Query 侧的 Down-projection 可在数学上合并（Absorb），消除推理时的 Up-projection 矩阵乘法：

$$ \mathbf{q}^{(h)} \cdot \mathbf{k}^{(h)} = (\mathbf{h}_t \tilde{\mathbf{W}}^Q) \cdot (\mathbf{c}_t^{KV} \mathbf{W}^{UK,h}) = \mathbf{c}_t^{KV} \cdot (\mathbf{W}^{UK,h\top} \tilde{\mathbf{W}}^{Q\top} \mathbf{h}_t) $$

将 $\mathbf{W}^{UK,h\top} \tilde{\mathbf{W}}^{Q\top}$ 预计算为融合矩阵，每步 Decode 无需解压 Key，直接以 $\mathbf{c}_t^{KV}$ 参与运算。但此技巧不适用于 RoPE 部分（RoPE 的旋转矩阵依赖位置，无法预计算融合），因此缓存 $\mathbf{k}_t^R$ 是必要的。

### 4.7 模型质量对比（DeepSeek-V2 消融）

|注意力机制|相对 MHA 的 KV Cache|MMLU|代码|推理|
|---|---|---|---|---|
|MQA|$1/n_h \approx 3%$|--|--|--|
|GQA-4|$4/n_h \approx 12.5%$|略低|略低|略低|
|MHA|$100%$|基准|基准|基准|
|**MLA**|**$\approx 3.5%$**|**优于 MHA**|**优于 MHA**|**优于 MHA**|

MLA 在 KV Cache 占用与 MQA 相当的前提下，性能超越 MHA，原因是 Up-projection 矩阵为每个 head 提供了独立丰富的参数，而不像 MQA 直接共用相同 Key/Value。

### 4.8 应用模型

DeepSeek-V2、DeepSeek-V3、DeepSeek-R1（全系列），以及基于 DeepSeek 架构的衍生模型。

### 4.9 FlashMLA

针对 MLA 特殊结构（缓存 $\mathbf{c}_t^{KV}$，推理时需执行 Up-projection）的定制化 CUDA Kernel，在 Hopper GPU（H100）上利用 Tensor Core 加速：将 Up-projection 与 QK 乘法融合为单一 Kernel，消除中间激活的 HBM 读写。

---

## 5. Cross-Layer Attention（CLA，NeurIPS 2024）

### 5.1 思路

在预训练阶段，某些层**完全复用相邻层的 KV head**（不创建自己的 KV），只生成独立的 Query：

$$ \text{Layer } l:\ \mathbf{K}^{(l)} = \mathbf{K}^{(l-1)},\quad \mathbf{V}^{(l)} = \mathbf{V}^{(l-1)} $$

交替设定"计算 KV 的层"与"只生成 Query 的层"。设复用率 $\rho$（$0 < \rho < 1$），则：

$$ M_{\text{CLA}} \approx (1 - \rho) \cdot M_{\text{MHA}} $$

$\rho = 0.5$（每两层共享一次 KV）时，KV Cache 减半。

### 5.2 与 GQA/MQA 的组合

CLA 可与 GQA 正交叠加：先用 GQA 减少 head 数，再用 CLA 减少需独立 KV 的层数，两者相乘可获得极高压缩比。

### 5.3 限制

CLA 层的 Query 无法感知本层自己的 KV 信息（因为直接使用上一层的 KV），在某些任务上会带来质量损失。Brandon et al.（NeurIPS 2024）报告在语言模型困惑度上 CLA 损失较小，但在复杂推理任务上有待验证。

---

## 6. MHA2MLA 迁移（arXiv 2025）

**问题**：大量已部署的 MHA/GQA 模型无法直接享受 MLA 的 KV Cache 收益。

**MHA2MLA**：无需从头预训练，通过数据高效的 Fine-tuning 将 MHA 迁移至 MLA：

1. **Partial RoPE 移除**：识别对注意力分数贡献最小的 Query/Key 维度，移除其 RoPE（为低秩压缩让路）；
2. **联合 SVD 近似**：对预训练的 $\mathbf{W}_K$、$\mathbf{W}_V$ 做联合 SVD，初始化 Down/Up-projection 矩阵；
3. **继续训练**：以约 $0.3%\text{--}0.6%$ 原始预训练 token 数量的数据 Fine-tuning 恢复质量。

实测（Llama2-7B，压缩至 $18.75%$ KV 大小）：性能下降 $< 0.61%$（7B 模型），且优于同等内存的 INT2 量化基线。

---

## 7. 架构级对比总结

|机制|KV Cache 相对 MHA|模型质量|是否需预训练|代表模型|
|---|---|---|---|---|
|MHA|$1\times$（基准）|最高（基准）|—|GPT-2/3, OPT|
|MQA|$1/n_h \approx 3\text{--}5%$|明显低于 MHA|是|PaLM, Falcon|
|GQA-$g$|$g/n_h \approx 12\text{--}25%$|接近 MHA|是（或 Uptrain）|LLaMA-3, Qwen2.5|
|CLA（$\rho=0.5$）|$50%$|接近 MHA|是|实验性|
|MLA|$\approx 3.5%$（vs MHA）|**优于 MHA**|是|DeepSeek-V2/V3/R1|
|MHA2MLA（迁移）|$\approx 18.75%$|接近 MHA|少量 Fine-tuning|实验性|

---

## 8. 与其他压缩技术的交互

### 8.1 GQA + Eviction

GQA 减少 KV head 数，Eviction 减少 token 数，两者正交可叠加：

$$ M_{\text{GQA+Evict}} = 2 \cdot L \cdot g \cdot d_h \cdot k \cdot b \quad (k < T) $$

LLaMA-3-70B GQA-8（$8\times$ 压缩）+ SnapKV 20% Budget（$5\times$ 压缩）= 约 $40\times$ 总压缩（相对于 MHA Full Cache）。

### 8.2 MLA + xKV

MLA 缓存 $\mathbf{c}_t^{KV}$（维度 $d_c = 512$），xKV 进一步对跨层的 $\mathbf{c}^{KV}$ 执行 SVD 压缩。实测（DeepSeek-Coder-V2 编程任务）：$3\times$ 额外压缩，无显著性能损失。

### 8.3 MLA + Quantization（SnapMLA）

MLA 的潜在向量维度更小（$d_c < n_h \cdot d_h$），量化误差在低维空间中积累较慢，FP8 量化对 MLA 的精度损失显著低于对 MHA KV 的损失。SnapMLA（SGLang-FluentLLM，2025）将 MLA KV 量化至 FP8，Throughput 提升 $1.91\times$，长上下文任务（数学推理、代码生成）无显著性能下降。

---

## 9. 文献索引

|方法|论文|会议/期刊|年份|
|---|---|---|---|
|MQA|Fast Transformer Decoding: One Write-Head is All You Need|arXiv|2019|
|GQA|GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints|EMNLP|2023|
|MLA|DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model|arXiv|2024|
|CLA|Reducing Transformer Key-Value Cache Size with Cross-Layer Attention|NeurIPS|2024|
|MHA2MLA|Towards Economical Inference: Enabling DeepSeek's MLA in Any Transformer-based LLMs|arXiv|2025|
|SnapMLA|SnapMLA: Quantizing MLA KV Cache with FP8 for Efficient Long-Context Inference|arXiv|2025|
|FlashMLA|FlashMLA: Efficient MLA Decoding Kernel for Hopper GPUs|GitHub|2025|
