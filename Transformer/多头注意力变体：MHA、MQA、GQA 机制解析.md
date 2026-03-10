## 1. 核心问题：KV Cache 的内存瓶颈
在自回归推理中，每个 token 生成需要访问所有历史 token 的 K、V 矩阵。
**KV Cache 内存公式：**
$$M_{KV} = 2 \times L \times H\_{kv} \times S \times d\_h \times \text{dtype\_bytes}$$
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
$$\text{head}_i = \text{Attention}(QW_i^Q,\ KW_i^K,\ VW_i^V)$$
$$\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W^O$$
$$\text{Attention}(Q,K,V) = \text{softmax}!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
### 3.2 参数量

|矩阵|形状|参数量|
|---|---|---|
|$W^Q$|$d_{model} \times d_{model}$|$d_{model}^2$|
|$W^K$|$d_{model} \times d_{model}$|$d_{model}^2$|
|$W^V$|$d_{model} \times d_{model}$|$d_{model}^2$|
|$W^O$|$d_{model} \times d_{model}$|$d_{model}^2$|

KV Cache 内存系数 = $H$

---

## 4. MQA（Multi-Query Attention）
**论文**：Fast Transformer Decoding — Shazeer, Google, 2019
**背景**：Decode 阶段每步仅生成一个 token，算力消耗极小，但需从 HBM 读取所有历史 K/V，arithmetic intensity 远低于 GPU 峰值 ops:byte 比，推理严重受限于内存带宽。多头 K/V 是带宽浪费的主要来源，而 Q 多头对质量有正向贡献——两者边际价值不对等。MQA 将 K/V 压缩至单头以最大化带宽节省。
### 4.1 结构
所有 Q 头共享同一组 K、V：
$$\text{head}_i = \text{Attention}(QW_i^Q,\ K W^K,\ V W^V)$$
- $W^K \in \mathbb{R}^{d_{model} \times d_k}$（单组）
- $W^V \in \mathbb{R}^{d_{model} \times d_v}$（单组）
### 4.2 KV Cache 对比
|MHA|MQA|
|---|---|---|
|KV 头数|$H$|$1$|
|KV Cache 大小|$2 H d_h$ per token|$2 d_h$ per token|
|内存节省|基准|$H \times$|
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
$$\text{head}_i = \text{softmax}!\left(\frac{Q_i W_i^Q (K_{g(i)} W_{g(i)}^K)^T}{\sqrt{d_k}}\right) V_{g(i)} W_{g(i)}^V$$
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

---

## 6. MLA（Multi-head Latent Attention）
**论文**：DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model — DeepSeek, 2024
**背景**：GQA 通过减少头数压缩 Cache，但头数减少直接削弱模型的多头表达能力，在超大规模模型（$H=128$）上质量代价显著。MLA 换一个压缩维度：不减少头数，而是对 K/V 做低秩压缩，缓存低维隐向量，推理时按需还原，在极端压缩比下保留完整头数的表达能力。
### 6.1 压缩与还原流程
**Down-projection（压缩）：**
$$c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}, \quad d_c \ll d_{model}$$
$h_t$：第 $t$ 个 token 的隐状态，$W^{DKV}$：down-projection 矩阵，$d_c$：压缩维度
**Up-projection（还原）：**
$$K_t = W^{UK} c_t^{KV} \in \mathbb{R}^{H \cdot d_h}, \quad V_t = W^{UV} c_t^{KV} \in \mathbb{R}^{H \cdot d_h}$$
完整注意力计算：
$$\text{head}_i = \text{softmax}!\left(\frac{Q_i K_i^T}{\sqrt{d_h}}\right) V_i$$
### 6.2 Q 侧低秩压缩
$$c_t^Q = W^{DQ} h_t \in \mathbb{R}^{d_c'}, \quad Q_t = W^{UQ} c_t^Q \in \mathbb{R}^{H \cdot d_h}$$
$c_t^Q$ **不需要缓存**，每步重新计算，仅用于减少 Q 投影参数量。
### 6.3 Decoupled RoPE
低秩压缩与 RoPE 存在根本冲突：Cache 存压缩向量 $c^{KV}$，还原出的 K 无法正确注入历史位置的 RoPE。解法是引入独立的带位置编码的 K：
$$k_t = [k_t^C;\ k_t^R], \quad k_t^R = \text{RoPE}(W^{KR} h_t,\ t)$$
实际 Cache 存储：
$$\text{Cache per token} = c_t^{KV} \oplus k_t^R$$
### 6.4 Absorption Trick（推理加速）
Up-projection 矩阵 $W^{UK}$、$W^{UV}$ 可与 Q 投影提前合并，注意力在压缩空间中直接计算，无需显式还原完整 K/V：
$$Q_i (W^{UK} C)^T = (Q_i W_i^{UK,T}) C^T = \tilde{Q}_i C^T$$
### 6.5 Cache 大小对比（DeepSeek-V2，$H=128$，$d_h=128$，$d_c=512$，$d_h^R=64$）

|机制|Cache per token per layer|相对 MHA|
|---|---|---|
|MHA|$2 \times 128 \times 128 = 32768$|$1.0\times$|
|GQA（G=8）|$2 \times 8 \times 128 = 2048$|$1/16$|
|MLA|$512 + 128 \times 64 = 8704$|$\approx 1/3.8$|

MLA 压缩比不及 GQA，但保留完整 $H=128$ 头表达能力。

---

## 7. 推理实现细节
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

|repeat_interleave|stride trick|
|---|---|---|
|内存行为|实际复制，$H/G \times$ 内存|零拷贝，原始大小|
|适用场景|训练、调试|推理引擎（vLLM/TRT-LLM）|
|FlashAttention|需显式展开|FA2 原生支持，内部处理|
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
> KV 头数：$H$ / $1$ / $G$。本质是 KV Cache 内存与模型表达能力的 trade-off。
**Q2：GQA 为什么比 MQA 质量更好？**
> 多组 KV 保留了头间多样性。不同 Q 组可以关注不同语义子空间，单组 KV 构成表达瓶颈。
**Q3：MLA 与 GQA 的压缩思路有何本质不同？**
> GQA：减少头数，保留每头维度，牺牲多样性。MLA：保留全部头，压缩 KV 特征维度到低秩空间，缓存压缩隐向量而非 K/V 本身，质量损失更小。
**Q4：MLA 为什么不能直接对压缩向量施加 RoPE？**
> RoPE 是位置相关的旋转变换，需在还原后的 K 上施加特定位置的旋转矩阵。若缓存压缩向量，历史 token 的位置信息无法在还原时正确注入。解法是 Decoupled RoPE，单独维护一组携带位置信息的 $k^R$。

---

### 8.2 推理优化类
**Q5：推理引擎中 GQA 的 KV expand 为什么用 stride trick？**
> repeat_interleave 产生 $H/G$ 倍内存复制，长序列下不可接受。stride trick 仅修改张量元数据，多个 Q 头逻辑上映射同一块 KV 内存，零拷贝。
**Q6：FlashAttention-2 如何支持 GQA？**
> FA2 在 CUDA kernel 内以 group 为单位分块加载 KV，Q 头在 group 内展开计算，不做显式 expand，HBM 访问量按 $G$ 而非 $H$ 计算。
**Q7：Tensor Parallelism 下 GQA 的切分约束？**
> $G$ 必须整除 TP degree。若 $G < \text{TP}$，KV 头无法均匀切分，需在每个 rank 上复制完整 KV。LLaMA-3 70B（$G=8$，TP=8）恰好每 rank 一个 KV 头。
**Q8：PagedAttention 在 GQA 下的内存变化？**
> KV Block 按 `num_kv_heads` 分配，Block 大小正比于 $G/H$，内存池总量等比缩减，碎片率不变但绝对浪费量减少。

---

### 8.3 训练/转换类
**Q9：如何将已训练的 MHA 模型转换为 GQA？**
> GQA 论文方法：对同组内多个 KV 头做 mean pooling 作为初始化，再以约 5% tokens 数据量微调恢复质量。**局限**：大模型（70B+）质量损失不可忽视，不如从头训练 GQA。
**Q10：GQA 中 $G$ 值如何选择？**

|约束|要求|
|---|---|
|TP 并行|$G$ 整除 TP degree|
|内存预算|$G$ 越小越省|
|质量下限|$G \geq 4$ 通常接近 MHA|
|工程惯例|$G = H/8$ 或 $G = H/4$|

---

### 8.4 手写代码类
**Q11：手写标准 Scaled Dot-Product Attention（含 mask）**
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

---

### 8.5 手推计算类
**Q12：LLaMA-3 70B，$H=64$，$G=8$，fp16，batch=1，$S=8192$，KV Cache 多大？**
$$M = 2 \times L \times G \times S \times d_h \times 2\ \text{bytes}$$ $$= 2 \times 80 \times 8 \times 8192 \times 128 \times 2$$ $$= 2{,}684{,}354{,}560\ \text{bytes} \approx 2.5\ \text{GB}$$
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
