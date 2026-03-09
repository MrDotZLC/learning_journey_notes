---
tags:
- LLM
- 推理优化
- NLP
- 深度学习
- 工程实践 aliases:
- Decode策略
- 解码策略
- LLM推理 created: 2025-03-09 updated: 2025-03-09 status: 完整
---
> [!abstract] 概述 本文系统梳理自回归语言模型的解码策略，涵盖核心原理、主流算法、工程优化与前沿研究，适合 LLM 推理工程师与研究者参考。
## 目录
- [[#1. 核心概念与理论背景]]
- [[#2. 主流解码策略详解]]
- [[#3. 工程优化技术]]
- [[#4. 参数调优实战指南]]
- [[#5. 前沿研究方向]]
- [[#6. 实际部署与框架对比]]
- [[#7. 深度数学推导]]

---

## 1. 核心概念与理论背景
### 1.1 自回归语言模型推理基础
**Decode（解码）** 是自回归语言模型（Autoregressive LM）在推理阶段的核心过程。模型根据已生成的 token 序列，逐步预测下一个 token，本质是从词表概率分布中进行采样或选择的策略问题。
每一步的条件概率：
$$P(x_t \mid x_1, x_2, \ldots, x_{t-1}) = \text{softmax}(W \cdot h_t + b)$$
其中 $h_t$ 是 Transformer 最后一层在位置 $t$ 处的隐状态，$W$ 是 unembedding 矩阵（通常与 embedding 矩阵共享权重，即 tied weights）。
### 1.2 Logit 到概率的转换
Transformer 输出的原始向量称为 **logit**，经 softmax 归一化后即为概率分布：
$$P(x_t = v) = \frac{\exp(z_v)}{\sum_i \exp(z_i)}, \quad v \in \mathcal{V}$$
词表大小 $|\mathcal{V}|$ 对主流大模型通常为：

|模型|词表大小|
|---|---|
|LLaMA-2|32K|
|LLaMA-3|128K|
|Gemma|256K|

### 1.3 解码策略的本质权衡
所有解码策略均在以下三个目标间寻求平衡：
```
质量 (Quality)    ←────────→    多样性 (Diversity)
        ↑
        │
        ↓
  效率 (Efficiency)
```
> [!tip] 任务导向
> 
> - **事实问答**：优先质量
> - **创意写作**：侧重多样性
> - **实时推理服务**：关注效率

---

## 2. 主流解码策略详解
### 2.1 Greedy Decoding（贪心解码）
每步选取概率最高的 token，是最简单的确定性解码方式。
```python
# 核心实现
next_token = torch.argmax(logits, dim=-1)
# 等价数学形式
# x_t = argmax_v P(v | x_{<t})
```
**优势**：速度快（$O(V)$ 复杂度）、输出完全确定、易于调试。
**局限**：容易陷入局部最优，极易产生**重复循环文本**（repetition trap）。
#### 重复陷阱的数学解释
由于语言模型在条件上具有马尔可夫性，若某短语在第 $k$ 次出现时概率最高，则第 $k+1$ 次出现时同样如此，形成正反馈循环：
$$P(\text{"the"} \mid \text{"...the the the"}) > P(\text{任意其他 token} \mid \text{"...the the the"})$$
**应对方案**：引入 [[#4.1 Repetition Penalty（重复惩罚）|Repetition Penalty]] 或改用采样策略。

---

### 2.2 Beam Search（束搜索）
同时维护 $k$ 条候选序列（beam），每步展开后保留联合对数概率最高的 $k$ 条路径。
$$\text{score}(y_{1:t}) = \sum_{i=1}^{t} \log P(y_i \mid y_{<i}, x)$$
```python
def beam_search(model, beam_width=4, max_len=100):
    beams = [([], 0.0)]  # (序列, 累积log概率)
    for step in range(max_len):
        candidates = []
        for seq, score in beams:
            logits = model(seq)
            top_k_probs, top_k_ids = torch.topk(logits, beam_width)
            for prob, token_id in zip(top_k_probs, top_k_ids):
                new_seq = seq + [token_id]
                new_score = score + torch.log(prob)
                candidates.append((new_seq, new_score))
        # 保留 top-k 序列
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    return beams[0][0]
```
#### 2.2.1 长度归一化（Length Normalization）
##### 问题根源
未归一化时，更长序列因累加更多负对数概率而得分偏低，算法偏好短输出。
**数学本质**：设每步概率均为 $p$，则：
$$\log P(Y_{\text{short}}) = L_s \cdot \log p \quad > \quad \log P(Y_{\text{long}}) = L_l \cdot \log p \quad (L_s < L_l,\ \log p < 0)$$
短序列必然胜出，与语言质量无关。
##### 解决方案
$$S_\text{norm}(y_{1:t}) = \frac{1}{t^\alpha} \sum_{i=1}^{t} \log P(y_i \mid y_{<i})$$
```python
score = log_prob / (sequence_length ** length_penalty)
# length_penalty 典型值 0.6（Google NMT 论文推荐）
```
##### Google NMT 完整公式（Wu et al., 2016）
$$\text{score}(Y, X) = \frac{\log P(Y|X)}{lp(Y)} + cp(X, Y)$$
**长度惩罚项（Length Penalty）**：
$$lp(Y) = \frac{(5 + |Y|)^{\alpha}}{(5 + 1)^{\alpha}}$$
**覆盖惩罚项（Coverage Penalty，可选）**：
$$cp(X, Y) = \beta \sum_{i=1}^{|X|} \log\left(\min\left(\sum_{j=1}^{|Y|} p_{i,j},\ 1\right)\right)$$
其中 $p_{i,j}$ 为第 $j$ 步对源端第 $i$ 个 token 的 attention weight，惩罚未被充分 attend 的源端 token。
##### 参数 $\alpha \in [0, 1]$ 的影响

|$\alpha$|行为|
|---|---|
|$0$|无归一化，严重偏短|
|$0.6$|Google NMT 推荐值（NMT 任务）|
|$1.0$|完全按长度平均，可能偏长|
|$> 1.0$|反向惩罚，偏向超长序列|

**选取原则**：在验证集上以 BLEU/ROUGE 为指标调参；LLM decoding 常用 $0.7 \sim 1.0$。
#### 2.2.2 Diverse Beam Search（多样性束搜索）
##### 问题：标准 Beam Search 的同质性
标准 Beam Search 的 $k$ 条候选序列往往高度相似——$k$ 条 beam 聚集在概率质量最高的局部区域，来自相同前缀。
##### 核心思想
将 $B$ 个 beam 分为 $G$ 组，每组 $b = B/G$ 个 beam，**组间引入差异惩罚**，使不同组产生不同输出。
##### 目标函数
**标准 Beam Search**：
$$\max_{Y^1, \dots, Y^B} \sum_{i=1}^{B} \log P(Y^i | X)$$
**DBS（加入多样性项）**：
$$\max_{Y^1, \dots, Y^B} \sum_{i=1}^{B} \log P(Y^i | X) + \lambda \cdot \Delta(Y^i, {Y^1, \dots, Y^{i-1}})$$
- $\lambda$：多样性强度系数
- $\Delta$：差异奖励函数（Diversity Reward），衡量 $Y^i$ 与已生成序列的差异
#### 2.2.3 Diverse Sibling Penalty（每步 token 级惩罚）
在每个时间步 $t$ 对已被前序组选过的 token 施加惩罚：
$$\hat{s}(y_t^g \mid \cdot) = s(y_t^g \mid \cdot) - \lambda \cdot \mathbb{1}[y_t^g \in \mathcal{A}^{<g}_t]$$
$$\text{score}_g(y_t) = \log P(y_t \mid y_{<t}) - \lambda \cdot \text{dissimilarity}(y_t, {\text{beams}_{1\ldots g-1}})$$
- $\mathcal{A}^{<g}_t$：第 $1, \dots, g-1$ 组在时间步 $t$ 已选择的 token 集合
- $\mathbb{1}[\cdot]$：指示函数，若该 token 已被前序组选过则惩罚 $\lambda$
##### 解码流程伪代码
```python
# 总 beam 数 B = G × b
# 按组顺序解码（关键：顺序依赖，第 g 组依赖前 g-1 组的选择）
for t in range(max_len):
    group_selected_tokens = {}          # 记录各组已选 token
    for g in range(G):                  # 组间顺序解码
        candidates = expand(groups[g])  # 展开当前组候选
        for candidate in candidates:
            token = candidate.last_token
            # 施加多样性惩罚：若前序组已选该 token
            for prev_g in range(g):
                if token in group_selected_tokens[prev_g]:
                    candidate.score -= lambda_diversity
        # 惩罚后取 top-b
        groups[g] = top_b(candidates)
        group_selected_tokens[g] = {b.last_token for b in groups[g]}
```
##### DBS 变体对比

|变体|差异施加位置|差异度量|计算复杂度|
|---|---|---|---|
|**Diverse Sibling Penalty**|每步 token 级|Token 重叠|$O(G \cdot b \cdot V)$|
|**Hamming Diversity**|每步 token 级|Hamming 距离|$O(G^2 \cdot T)$|
|**DPP（行列式点过程）**|序列级|核矩阵行列式|$O(G^3)$，昂贵|

##### 复杂度与适用场景
复杂度 $O(k \cdot V \cdot T)$。当 $k=4$、$V=128\text{K}$、$T=2048$ 时，每次推理约需 $10^{12}$ 次浮点运算，适合**离线批处理**而非实时推理。
##### 超参调节建议

| 参数             | 推荐范围         | 行为                 |
| -------------- | ------------ | ------------------ |
| $\lambda$      | $[0.5, 2.0]$ | 过大牺牲流利度；过小退化为标准 BS |
| $G$（组数）        | $2 \sim 4$   | 越多多样性越强，计算线性增长     |
| $b$（每组 beam 数） | $1 \sim 4$   | $b=1$ 时每组单条路径      |

#### 2.2.4两种机制对比总结

| 维度       | 长度归一化                   | Diverse Beam Search       |
| -------- | ----------------------- | ------------------------- |
| **解决问题** | 短序列偏差（length bias）      | 输出同质性（diversity collapse） |
| **作用时机** | 选 beam 时重新打分            | 每步生成时修改候选得分               |
| **关键超参** | $\alpha \in [0.6, 1.0]$ | $\lambda > 0$，$G$（组数）     |
| **计算开销** | $O(1)$ 额外开销             | $O(G)$ 额外（顺序组解码）          |
| **常用场景** | NMT、摘要、所有生成任务           | 对话多样性、数据增强、reranking      |
| **是否正交** | ✅ 可同时使用                 | ✅ 可同时使用                   |

---

### 2.3 采样策略（Sampling Methods）
与确定性解码不同，采样从概率分布中随机抽取 token，天然具备多样性。
#### 2.3.1 Temperature Sampling（温度采样）
通过超参数 $\tau$ 对 logit 进行缩放，控制概率分布的「尖锐度」：
$$P_\tau(x_t \mid x_{<t}) = \frac{\exp(z_t / \tau)}{\sum_i \exp(z_i / \tau)}$$
```python
logits = logits / temperature    # τ 直接缩放 logit
probs = F.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

|τ 值|分布形态|效果描述|典型应用|
|---|---|---|---|
|τ → 0|近似 one-hot|趋近 Greedy，极度确定|精确代码生成|
|τ = 0.3|较尖锐|倾向高概率 token|事实性问答|
|τ = 1.0|原始分布|忠实模型预测|通用对话|
|τ = 1.5|较平坦|增加随机性与创意|头脑风暴、创作|
|τ > 2.0|近均匀分布|高噪声，语义混乱|通常不推荐|

#### 2.3.2 Top-k Sampling
仅从概率最高的 $k$ 个 token 中采样，完全屏蔽长尾低概率词汇：
```python
top_k_probs, top_k_ids = torch.topk(probs, k=50)
top_k_probs = top_k_probs / top_k_probs.sum()  # 重归一化
next_token = top_k_ids[torch.multinomial(top_k_probs, 1)]
```
> [!warning] 核心缺陷 固定 $k$ 无法自适应分布形态：
> 
> - **宽峰分布时**：概率集中于少数 token，$k=50$ 可能引入大量噪声
> - **尖峰分布时**：模型高度确定，$k=50$ 却仍强制引入随机性
> 
> 实验表明 Top-k 在开放生成任务中的表现不稳定，建议优先使用 Top-p 或 Min-p。
#### 2.3.3 Top-p Sampling（Nucleus Sampling，核采样）
动态构建最小候选集 $\mathcal{V}^{(p)}$，使其累积概率不低于阈值 $p$：
$$\mathcal{V}^{(p)} = \underset{\mathcal{V}' \subseteq \mathcal{V}}{\arg\min} |\mathcal{V}'| \quad \text{s.t.} \quad \sum_{x \in \mathcal{V}'} P(x \mid x_{<t}) \geq p$$
```python
sorted_probs, sorted_ids = torch.sort(probs, descending=True)
cumulative = torch.cumsum(sorted_probs, dim=-1)
# 移除使累积概率超过 p 的 token
sorted_probs[cumulative - sorted_probs > p] = 0.0
sorted_probs = sorted_probs / sorted_probs.sum()  # 重归一化
next_token = sorted_ids[torch.multinomial(sorted_probs, 1)]
```
**自适应截断**优势（以 $p=0.9$ 为例）：
- **确定性上下文**（如代码补全）：可能只保留 3-5 个 token，输出精准
- **开放性上下文**（如故事续写）：可能保留 50-200 个 token，输出多样
> [!tip] 组合使用 Top-p 与 Temperature 可叠加使用：
> 
> - 创意生成：`τ=0.8, p=0.95`
> - 代码生成：`τ=0.3, p=0.9`
#### 2.3.4 Min-p Sampling（2024 年新方法）
相对概率截断：过滤掉概率低于「最高概率 × min_p」的 token：
$$\text{threshold} = \text{min\_p} \times \max_{v} P(v \mid x_{<t})$$
```python
max_prob = probs.max()
threshold = min_p * max_prob    # 相对阈值
probs[probs < threshold] = 0.0
probs = probs / probs.sum()     # 重归一化
next_token = torch.multinomial(probs, 1)
```
**与 Top-p 的本质区别**：

|维度|Top-p|Min-p|
|---|---|---|
|截断基准|绝对累积概率|相对最高概率比值|
|高确定性时|仍可能保留较多噪声|自适应收缩，更「专注」|
|低确定性时|动态扩展|同样宽松|

#### 2.3.5 Mirostat 采样（动态困惑度控制）
通过实时控制输出困惑度（Perplexity）来稳定生成质量：
$$\text{target\_perplexity} = \exp(\tau) \quad \Rightarrow \quad \text{动态调整截断阈值 } \mu$$
```bash
# llama.cpp 参数
--mirostat 2           # Mirostat v2（推荐）
--mirostat-lr 0.1     # 学习率 η（控制适应速度）
--mirostat-ent 5.0    # 目标熵 τ（越高越多样）
```
**适用场景**：长文本生成（小说、报告），可防止文本随时间退化或变得单调。

---

### 2.4 Contrastive Decoding（对比解码）
利用「专家模型」与「业余模型」的概率差异，放大专家模型的独特优势：
$$\text{score}(x_t) = \log P_\text{expert}(x_t \mid x_{<t}) - \alpha \cdot \log P_\text{amateur}(x_t \mid x_{<t})$$
```python
# 概念实现（alpha ≈ 0.1）
logits_expert  = large_model(input_ids)   # 70B
logits_amateur = small_model(input_ids)   # 7B
logits_final   = logits_expert - alpha * logits_amateur
next_token = torch.argmax(logits_final)
```
#### 变体：VCD（Visual Contrastive Decoding）
在多模态模型中，通过对比「有图像输入」与「无图像输入」的分布，减少视觉幻觉：
$$\text{score}_\text{VCD}(x_t) = \log P(x_t \mid \text{image, text}) - \beta \cdot \log P(x_t \mid \text{text only})$$
---
### 2.5 Speculative Decoding（推测解码）
用小模型（Draft Model）快速生成多个候选 token，再由大模型（Target Model）并行验证，通过「拒绝采样」大幅提升吞吐量。
#### 算法流程
```python
def speculative_decode(target, draft, prompt, K=5):
    # Step 1: Draft 小模型自回归生成 K 个候选
    draft_tokens, draft_probs = [], []
    x = prompt
    for _ in range(K):
        q = draft(x)           # draft 的概率分布
        t = sample(q)          # 从 draft 采样
        draft_tokens.append(t)
        draft_probs.append(q[t])
        x = x + [t]
    # Step 2: Target 大模型一次前向，并行计算 K+1 个位置的概率
    target_probs = target(prompt + draft_tokens)  # 一次并行前向！
    # Step 3: 逐 token 验证（拒绝采样）
    accepted = 0
    for i in range(K):
        accept_prob = min(1, target_probs[i] / draft_probs[i])
        if random() < accept_prob:
            accepted += 1
        else:
            break  # 拒绝后从修正分布重采样
    return draft_tokens[:accepted]
```
#### 理论加速比分析
$$\mathbb{E}[\text{tokens per step}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$
其中 $\alpha$ 为平均 token 接受率：

|接受率 α|Lookahead K|期望加速比|实际场景|
|---|---|---|---|
|0.9|5|~4.6×|高度同分布（同系列小大模型）|
|0.8|5|~3.4×|任务相关微调 draft|
|0.7|4|~2.8×|通用 draft 模型|
|0.5|3|~1.9×|分布差异较大|

#### 工程实现变体

|变体|核心思路|优势|
|---|---|---|
|**Medusa**|在大模型顶层添加多个预测头|无需独立 draft 模型|
|**Lookahead Decoding**|维护 N-gram 缓存池复用历史片段|降低 draft 计算量|
|**Self-Speculative**|跳过若干中间层构建 draft|无额外参数|
|**Eagle / Eagle-2**|特征级 draft + 动态候选树|目前 SOTA，~5× 加速|

---

## 3. 工程优化技术
### 3.1 KV Cache
将已计算的 Key/Value 矩阵缓存，避免每步重复计算历史部分：
```python
# ❌ 无 KV Cache：每步重新计算全部 K, V
K = x @ W_k  # [full_seq_len, d_k] ← 重复计算历史部分
# ✅ 有 KV Cache：仅计算新 token
K_new   = x_new @ W_k                        # [1, d_k]
K_cache = torch.cat([K_cache, K_new], dim=0) # [seq_len+1, d_k]
attn    = (q_new @ K_cache.T) / sqrt(d_k)    # 利用缓存
```
#### 显存占用估算
对于 LLaMA-3-70B（FP16）：
$$\text{KV\_size} = 2 \times 2,\text{bytes} \times 80,\text{layers} \times 8,\text{heads} \times 128,\text{dims} \approx 0.32,\text{MB/token}$$
> [!warning] 显存限制 最大上下文 128K token 时，KV Cache 约占 **40 GB**，接近单卡 A100 80GB 的显存上限。

---

### 3.2 PagedAttention
vLLM 提出的核心创新，借鉴操作系统虚拟内存分页思想：
```
KV Cache 被分割为固定大小的 Block（通常 16 tokens/block）
逻辑地址                    物理地址
┌──────────┐              ┌──────────┐
│ seq_0    │  Block Table │ Block #3 │
│ block[0] │──────────────│          │
│ block[1] │──────────────│ Block #7 │
│ block[2] │──────────────│ Block #1 │
└──────────┘              └──────────┘
```
```python
# 逻辑 Block -> 物理 Block 映射
block_table = {
    'seq_0': [physical_block_3, physical_block_7, physical_block_1],
    'seq_1': [physical_block_5, physical_block_2],
}
# Copy-on-Write: beam search / parallel sampling 共享 prefix block
```
**关键优势**：
- **内存利用率**：相比传统静态分配提升至 90%+
- **Copy-on-Write**：多序列共享相同前缀（如 system prompt）的 KV Cache
- **Prefix Caching**：跨请求复用相同前缀，显著降低 TTFT

---

### 3.3 Continuous Batching（连续批处理）

|特性|Static Batching|Continuous Batching|
|---|---|---|
|序列管理|固定批大小，整批进出|动态插入/移除单条序列|
|GPU 利用率|~60-70%（等待）|~90-95%（连续）|
|延迟公平性|长序列影响整批延迟|各序列独立完成|
|实现框架|早期 Triton 推理|vLLM, TensorRT-LLM, SGLang|

---

### 3.4 Chunked Prefill
将长 Prompt 的 Prefill 阶段分块执行，与 Decode 阶段**交错进行**，避免 Prefill 独占 GPU 导致正在 Decode 的请求延迟飙升（stall）：
```python
# 每个调度周期：
# ① 执行 chunk_size=512 tokens 的 prefill
# ② 执行当前所有 in-flight 序列的 decode
# 两者在同一个 forward pass 中完成（batching prefill + decode）
```
> [!note] Chunked Prefill 是解决「长 Prompt TTFT 与在线 Decode 延迟矛盾」的关键技术，vLLM v0.4+ 默认开启。

---

### 3.5 量化对解码的影响

|量化方法|精度损失|吞吐提升|推荐温度补偿|适用解码策略|
|---|---|---|---|---|
|FP16（基准）|0%|1×|无需|全部|
|W8A8 (INT8 SmoothQuant)|~0.5-1%|1.5-2×|+0.05|Greedy / Beam|
|W4A16 (GPTQ / AWQ)|~2-3%|2-3×|+0.1|Top-p / Top-k|
|W4A8 (QuaRot)|~3-4%|2.5-3.5×|+0.1-0.15|Top-p 建议 0.92+|
|W2 (QuIP#)|~5-8%|3-4×|+0.2|需调高 min_p/top_p|

**机制解析**：量化误差将原始尖锐的 logit 分布「平滑化」，使高概率 token 的优势降低。贪心解码对此最敏感，采样策略可通过适当降低 top_p 或提高 temperature 进行补偿。

---

## 4. 参数调优实战指南
### 4.1 Repetition Penalty（重复惩罚）
对已生成 token 的 logit 施加惩罚，防止陷入重复循环：
$$\text{score}(x_t) = \begin{cases} z_t / \text{penalty} & \text{if } x_t \in x_{<t} \ z_t & \text{otherwise} \end{cases}$$
```python
# transformers 实现（简化版）
for token_id in set(input_ids[0]):  # 对已出现 token
    if scores[token_id] < 0:
        scores[token_id] *= repetition_penalty   # 负数乘惩罚 = 更负
    else:
        scores[token_id] /= repetition_penalty   # 正数除惩罚 = 更小
```

|penalty 值|效果|适用场景|
|---|---|---|
|1.0|无惩罚（默认）|需要重复的任务（如格式化输出）|
|1.05-1.1|轻微惩罚|通用对话，防止短语循环|
|1.1-1.2|中等惩罚|长文本生成，避免段落重复|
|1.2-1.3|强惩罚|创意写作，但可能影响连贯性|
|>1.5|过强，慎用|可能导致语法错误|

---

### 4.2 Presence Penalty 与 Frequency Penalty
OpenAI API 区分两种惩罚（范围均为 $[-2, 2]$）：
- **Presence Penalty**：对出现过的 token（不论频率）施加**固定惩罚**，鼓励引入新概念
- **Frequency Penalty**：惩罚与**出现次数成正比**，重复越多惩罚越重
```python
# 二者同时作用
logit[t] -= frequency_penalty * count[t]                       # 频率惩罚
logit[t] -= presence_penalty * (1 if count[t] > 0 else 0)     # 存在惩罚
```

---

### 4.3 任务导向的参数配置
```python
# 事实性任务（翻译 / 摘要 / RAG）
config_factual = {
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.05,
}
# 代码生成
config_code = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.1,
}
# 通用对话
config_chat = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 0,      # 仅使用 top_p
    "repetition_penalty": 1.05,
}
# 创意写作
config_creative = {
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 100,
    "repetition_penalty": 1.0,
}
# 头脑风暴
config_brainstorm = {
    "temperature": 1.1,
    "top_p": 0.98,
    "top_k": 0,
    "repetition_penalty": 1.1,
}
```
> [!tip] 调参顺序 建议：先调 `temperature` → 再微调 `top_p` → 最后考虑 `repetition_penalty`

---

### 4.4 格式化输出中的解码控制
结构化输出（JSON / XML 等）需要特殊处理，防止模型生成非法格式：
- **Logit Masking**：将不符合当前 JSON 状态机约束的 token logit 设为 $-\infty$，强制输出合法结构
- **Grammar Sampling（llama.cpp GBNF）**：使用 BNF 文法约束解码，确保输出 100% 符合指定文法
- **Outlines / Guidance**：通过正则表达式或 Pydantic Schema 约束 LLM 输出
```python
# llama.cpp 文法约束示例
llm = Llama(model_path="model.gguf")
grammar = LlamaGrammar.from_string(r"""
  root ::= "{" ws string ws ":" ws number ws "}"
""")
output = llm("生成JSON", grammar=grammar)
```

---

## 5. 前沿研究方向
### 5.1 Medusa（多头推测解码）
在 LLM 顶层附加 $K$ 个轻量 MLP 预测头，每个头负责预测未来第 $i$ 个 token：
```python
class MedusaModel(LlamaForCausalLM):
    def __init__(self, config, medusa_num_heads=5):
        super().__init__(config)
        self.medusa_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.vocab_size)
            for _ in range(medusa_num_heads)
        ])
    def forward(self, x):
        base_logits    = self.lm_head(x)                         # 预测 step t
        medusa_logits  = [h(x) for h in self.medusa_heads]      # 预测 t+1..t+K
        return base_logits, medusa_logits
```
**实测加速**：Medusa-2（5 个预测头）在 Vicuna-7B 上实现约 **2.2-2.8×** 的吞吐提升，无需额外 draft 模型，部署成本低。

---

### 5.2 Eagle / Eagle-2（2024）
通过预测 LLM **最后一层的特征向量**（而非直接预测 token），大幅提高 draft 质量：
$$\text{draft\_feature}_{t+1} = \text{EAGLE\_model}(\text{feature}_t, \text{token}_t)$$
Eagle-2 进一步引入**动态候选树（Dynamic Draft Tree）**，根据上下文实时调整树的形状，在 Llama-3-70B 上实现接近 **5×** 的加速，是目前 Speculative Decoding 的 SOTA。

---

### 5.3 Diffusion-Based Decoding（扩散解码）
与传统自回归逐 token 生成不同，扩散语言模型以掩码扩散过程**一次性生成整段文本**：
$$x_T \text{（噪声）} \to x_{T-1} \to \cdots \to x_0 \text{（文本）}$$
代表工作：MDLM、Plaid
- **优点**：天然支持双向上下文，延迟与序列长度弱相关
- **缺点**：目前质量仍弱于顶级自回归模型，处于快速发展阶段

---

### 5.4 Tree Attention 与 Parallel Decoding
构建候选序列树，并行验证多条路径：
```
# Tree Attention: 多条路径共享 prefix KV Cache
# path_A: [token_a1, token_a2, ...]  ─┐
# path_B: [token_b1, token_b2, ...]  ─┤── 共享 prefix KV
# path_C: [token_c1, token_c2, ...]  ─┘
# 通过 attention mask 控制各路径的可见范围
```

---

### 5.5 Reward-Guided Decoding（奖励引导解码）
在解码时引入外部奖励模型对每步候选进行打分：
$$\text{score}(x_t) = \log P_\text{LM}(x_t \mid x_{<t}) + \beta \cdot R(x_t, x_{<t})$$

|方法|说明|
|---|---|
|**Best-of-N Sampling**|生成 N 条序列，选择奖励最高的一条，推理时对齐的简单基线|
|**ARGS（2024）**|Autoregressive Reward Guided Search，在 beam search 中用 reward 重排候选|
|**RLHF 解码推断**|直接将 RLHF 中的 reward signal 用于 token 级引导，无需额外训练|

---

## 6. 实际部署与框架对比
### 6.1 延迟-吞吐-显存三角

|策略|TTFT|吞吐 (tok/s)|显存占用|输出质量|推荐场景|
|---|---|---|---|---|---|
|Greedy|最低|高|1×（基准）|确定性强|翻译、摘要|
|Beam (k=4)|中等|低（4× 计算）|4×|质量最高|离线批处理|
|Top-p Sampling|最低|高|1×|多样性好|对话、生成|
|Speculative (K=5)|中等|最高（3-5×）|1.3-1.5×|与 target 一致|实时推理服务|
|Medusa|低|高（2-3×）|+5% 参数量|接近原模型|单卡部署加速|

---

### 6.2 主流推理框架支持矩阵（2025 年 3 月）

|框架|Greedy/Sampling|Beam|Speculative|PagedAttn|Cont. Batch|量化支持|
|---|---|---|---|---|---|---|
|**vLLM**|✅ 全支持|✅|✅ Eagle/Medusa|✅ 核心特性|✅|AWQ/GPTQ/INT8|
|**TensorRT-LLM**|✅ 全支持|✅|✅ Draft Plugin|✅|✅|INT8/INT4/FP8|
|**llama.cpp**|✅ 含 Mirostat|⚠️ 有限|✅ --draft|❌|⚠️ 部分|GGUF 全系列|
|**SGLang**|✅ 全支持|⚠️ 有限|✅|✅|✅|AWQ/GPTQ|
|**HF TGI**|✅ 全支持|✅|⚠️ 实验性|✅|✅|GPTQ/AWQ|

---

### 6.3 生产环境调优 Checklist
- [ ] 确定任务类型（事实性 / 对话 / 创意），选择基础解码策略
- [ ] 根据 GPU 型号评估量化方案（A100/H100 优先 FP8，RTX 4090 优先 W4A16）
- [ ] 吞吐优先：启用 Speculative Decoding + Continuous Batching + PagedAttention
- [ ] 延迟优先：Greedy + KV Cache + Chunked Prefill，禁用 Beam Search
- [ ] 量化后实测 MMLU/MT-Bench 基准，下降 >3% 则适当调高 temperature 或换更高精度量化
- [ ] 长文本生成（>8K tokens）启用 Mirostat 或动态 repetition_penalty 防止退化
- [ ] 结构化输出场景使用 Grammar Sampling 或 Outlines，禁用随机性

---

## 7. 深度数学推导
### 7.1 Beam Search 最优性分析
Beam Search 并非全局最优解。以 $|\mathcal{V}|=32000$，$T=100$ 为例：
- **穷举最优**：需要 $32000^{100}$ 次评估（不可行）
- **Beam Search (k=4)**：仅评估 $4 \times 32000 \times 100 = 12.8\text{M}$ 次
**命题**：Beam Search 找到的序列 $y^_$ 满足 $P(y^_ | x) \geq P(y_\text{greedy} | x)$，但不保证达到全局最优 $\arg\max_y P(y|x)$。
在 BLEU 等指标上，解码质量随 $k$ 增加单调提升，但**边际收益递减**。

---

### 7.2 采样分布熵分析
给定温度 $\tau$，采样分布的熵为：
$$H(P_\tau) = -\sum_v P_\tau(v) \log P_\tau(v) \approx \frac{H(P_1)}{\tau} \quad \text{（低熵情形近似）}$$
- $\tau \to 0$：$H \to 0$（确定性）
- $\tau \to \infty$：$H \to \log |\mathcal{V}|$（均匀分布）
**有效词表大小** $\approx \exp(H)$，可直觉判断当前参数下模型的实际选择空间。

---

### 7.3 Speculative Decoding 正确性证明
**关键性质**：输出分布与 target 模型完全一致。
设 $p = P_\text{target}(x)$，$q = P_\text{draft}(x)$。接受后的有效分布为：
$$P_\text{accept}(x) = q(x) \cdot \min!\left(1, \frac{p(x)}{q(x)}\right) + \delta \cdot \text{norm}(p - q)_+$$
其中 $\delta$ 为拒绝后从修正分布重采样的贡献项。可以证明：
$$P_\text{accept} = P_\text{target}$$
即推测解码在统计意义上与**直接从大模型采样**完全等价，同时获得显著加速。

---

### 相关笔记
- [[Transformer 架构详解]]
- [[KV Cache 实现与优化]]
- [[vLLM 部署指南]]
- [[量化方法对比：GPTQ vs AWQ vs GGUF]]
- [[RLHF 与对齐技术]]

---

> [!info] 文档说明 本文档覆盖截止 2025 年 3 月的主流 Decode 技术。Eagle-2、MDLM 等前沿方向仍在快速演进中，建议结合最新论文动态持续更新。
