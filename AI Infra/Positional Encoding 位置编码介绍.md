# 一、Sinusoidal Position Encoding 总结

## 1. 为什么需要位置编码？
Transformer 的 Self-Attention **没有顺序概念**。  
Sinusoidal PE 为每个 token 提供可区分的“位置信息”。
## 2. 核心思想
transformer中维度之间无关联，只和前后同维度有关联，同时，为表达token的整体和局部信息，使用多种频率的 **sin/cos** 函数，将位置 `pos` 转换为一个 **唯一的高维向量**。
## 3. 公式
对于位置 pos、维度 i：
$$PE(pos, 2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE(pos, 2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right)$$
- 偶数维度：sin
- 奇数维度：cos
- 维度越大 → 频率越低（周期更长）
## 4. 为什么要多种频率？
- 高频：反映局部相邻 token 的位置信息
- 低频：反映长距离全局位置  
    → 多尺度位置特征同时存在。
## 5. 几何直觉
- 每个 token 的位置编码可以看作在 **多维空间中的螺旋轨迹**：
    - 每一维对应一个频率的 sin/cos 振动
    - 低频维度 → 长周期变化 → 捕捉远距离关系
    - 高频维度 → 短周期变化 → 捕捉局部位置差异
- 组合后形成一个**独特的向量**表示位置，距离近的向量相似，距离远的向量差异大

**直觉图像**：可以想象一个多频振动的指纹，每个位置有唯一指纹，Attention 可以通过内积区分距离。
## 6. 最大优势：支持长度外推
利用三角恒等式：
`sin(a+b) = sin(a)*cos(b) + cos(a)*sin(b)`
模型可以通过两段 PE 推断 **相对距离 (pos2 - pos1)**，  
因此 **支持比训练时更长的序列**。
## 7. 为什么基数是 10000？
让不同维度的周期合理覆盖：
- 从 2π（高频）
- 到 1e4（低频）
适配 NLP 典型长度（几十到几千）。
## 8. 如何与模型结合？
`InputEmbedding = TokenEmbedding + PositionalEncoding`
直接相加即可。
## 9. 特点
### ✔ 优点
- 无需训练（固定 PE）
- 可外推到更长序列
- 描述相对位置信息
- 不随深度衰减
### ✘ 缺点
- 特别长序列容易出现周期折返
- 现代大模型多改用 RoPE、ALiBi 等更强的方案
## 10. 一句话总结
> **Sinusoidal PE 用一组不同周期的正弦/余弦函数，为 Transformer 提供可区分、可外推、具有相对距离结构的绝对位置编码，是最经典也最具数学优雅的 PE 方案。**

# 二、Learnable Positional Embedding（可训练位置编码）

在 Transformer 中，为了让模型感知序列顺序，需要给每个 token 添加位置编码（Positional Embedding, PE）。  
Learnable PE 是最简单直观的一种方法：将每个位置的编码作为**可训练向量**，模型在训练中自动学习最优表示。
## 1️⃣ 原理
Transformer 输入 token embedding：
$$E = [e_1, e_2, \dots, e_n], \quad e_i \in \mathbb{R}^{d_{model}}$$
Learnable PE 为每个位置定义向量：
$$P = [p_1, p_2, \dots, p_{max\_len}], \quad p_i \in \mathbb{R}^{d_{model}}$$
最终输入 Transformer 的向量为：
$$X_i = e_i + p_i
$$
- **$e_i$**：第 i 个 token 的 embedding  
- **$p_i$**：第 i 个位置的可训练向量  
- **$X_i$**：最终输入 Transformer 的向量  
## 2️⃣ 参数表示
- 序列最大长度 `max_len`  
- 隐藏维度 `d_model`  
Learnable PE 参数量：
$$\text{params} = max\_len \times d_{model}$$
> 例如：`max_len=512`，`d_model=768` → 参数约 39 万个
- 参数随机初始化（均匀分布或正态分布）  
- 在训练中通过梯度下降学习  （权重共享）

## 3️⃣ 与 Sinusoidal PE 的区别

| 特性 | Sinusoidal PE | Learnable PE |
|------|---------------|--------------|
| 是否可训练 | ❌（固定公式） | ✅（随机初始化，训练中优化） |
| 外推能力 | 高 | 低（超出训练长度效果差） |
| 灵活性 | 低 | 高，可适应任务 |
| 参数量 | 0 | max_len × d_model |
| 实现复杂度 | 低 | 低 |
## 4️⃣ 实现方式（PyTorch 示例）

```python
import torch
import torch.nn as nn

max_len = 512
d_model = 768

# 可训练的位置编码
pos_embedding = nn.Embedding(max_len, d_model)

seq_len = 128
position_ids = torch.arange(seq_len).unsqueeze(0)  # [1, seq_len]

# 获取位置向量
pos_vec = pos_embedding(position_ids)  # [1, seq_len, d_model]

# 假设已有 token embedding
token_embedding = torch.randn(1, seq_len, d_model)

# 将 token embedding 和位置编码相加
x = token_embedding + pos_vec
```
## 5️⃣ 举例说明
假设序列长度为 3，隐藏维度为 4：
**Token Embedding**：
	
	| Token | Embedding            |
	| ----- | -------------------- |
	| I     | [0.1, 0.3, 0.5, 0.7] |
	| love  | [0.2, 0.4, 0.6, 0.8] |
	| AI    | [0.3, 0.5, 0.7, 0.9] |
**Learnable PE（随机初始化示例）**：
	
	| Position | PE                       |     |
	| -------- | ------------------------ | --- |
	| 0        | [0.01, 0.02, 0.03, 0.04] |     |
	| 1        | [0.05, 0.06, 0.07, 0.08] |     |
	| 2        | [0.09, 0.10, 0.11, 0.12] |     |
**相加后输入 Transformer 的向量**：
		
	| Token | Final Input              |
	| ----- | ------------------------ |
	| I     | [0.11, 0.32, 0.53, 0.74] |
	| love  | [0.25, 0.46, 0.67, 0.88] |
	| AI    | [0.39, 0.60, 0.81, 1.02] |
## 6️⃣ 优缺点
**优点**：
1. 灵活，可根据任务数据学习最优位置表示
2. 对短序列或固定长度序列收敛快
3. 实现简单，参数量可控

**缺点**：
1. 外推能力差，超出训练长度的序列可能无法编码正确位置信息
2. 对长序列任务可能不如相对位置编码和 RoPE

## 7️⃣ 应用场景
- **BERT** 系列（Base、Large）
- **GPT-1/2**
- 中小型 Transformer 模型，尤其是短文本任务

> 大型长序列模型（如 LLaMA、GPT-4）多改用 RoPE 或相对位置编码以提升长序列能力。

# 三、Relative Positional Embedding（相对位置编码）
在 Transformer 中，绝对位置编码（Absolute PE）只告诉模型每个 token 的具体位置，而很多任务中更重要的是 **token 之间的相对距离**。  
Relative PE 的核心思想是：**注意力不仅依赖 token 内容，还依赖它们之间的相对位置**。  
## 1️⃣ 背景
- Transformer 原生的绝对位置编码：
$$X_i = e_i + p_i$$
  - **$e_i$**：token embedding  
  - **$p_i$**：绝对位置向量  
- 问题：
  1. 只表示 token 的绝对位置，忽略相对关系  
  2. 对长序列外推能力有限  
- Relative PE 提出：
  - 每个 token pair (i, j) 的注意力与 **i 到 j 的相对位置** 相关  
  - 更自然地捕捉局部依赖和长距离依赖  
## 2️⃣ 核心思想
在标准 self-attention 中：
$$Attention(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
- $Q, K, V$ 分别是 query、key、value  
- 对每对 token，加入相对位置编码：
$$Attention(Q, K, V) = \text{softmax}\left(\frac{Q K^T + Q R^T}{\sqrt{d_k}}\right) V$$
- **$R$**：表示 token i 相对 token j 的位置信息向量（可训练或固定）  
- 注意力值不仅依赖内容，还依赖相对位置  
> 核心优势：捕捉 token 间的相对关系，而非绝对位置。
## 3️⃣ 公式（简化版）
假设序列长度为 L，隐藏维度为 d：
1. 定义相对位置偏移向量：
$$r_{i-j} \in \mathbb{R}^{d}$$
2. 对 attention logits 进行修正：
$$\text{score}(i,j) = Q_i \cdot K_j + Q_i \cdot r_{i-j}$$
- $Q_i·K_j$：内容相关性  
- $Q_i·r_{i-j}$：位置相关性  
1. 注意力计算：
$$\alpha_{i,j} = \frac{\exp(\text{score}(i,j))}{\sum_{k=1}^{L} \exp(\text{score}(i,k))}
$$
2. 输出：
$$O_i = \sum_{j=1}^{L} \alpha_{i,j} V_j$$
> 相比绝对位置编码，每个 token 的注意力会根据 **与其他 token 的相对距离** 自动调整。
## 4️⃣ 实现方式（简化示例）

```python
import torch
import torch.nn as nn

max_rel = 4  # 相对位置范围 [-4,4]
d_model = 4

# 可训练相对位置向量
rel_embedding = nn.Embedding(2*max_rel+1, d_model)

seq_len = 5
# 构建相对位置索引矩阵
pos_ids = torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)
# clip到[-max_rel,max_rel]并偏移到[0,2*max_rel]
pos_ids = pos_ids.clamp(-max_rel,max_rel) + max_rel

# 获取相对位置向量
R = rel_embedding(pos_ids)  # [seq_len, seq_len, d_model]

# 假设 Q 和 K
# Q: [seq_len, d_model], K: [seq_len, d_model]
score = torch.matmul(Q, K.T) + torch.einsum('id,ijd->ij', Q, R)
```
`einsum` 用于计算 Q_i·r_{i-j}
## 5️⃣ Absolute PE 与 Relative PE 注意力加权对比

| 特性    | Absolute PE（绝对位置编码）        | Relative PE（相对位置编码）                 |
| ----- | -------------------------- | ----------------------------------- |
| 表示方式  | 每个 token 对应固定位置向量 pip_ipi​ | 每对 token 之间有相对位置向量 ri−jr_{i-j}ri−j​ |
| 关注点   | token 的绝对位置                | token 之间的距离/相对关系                    |
| 外推能力  | 高（Sinusoidal PE 可计算任意位置）   | 高（适合长序列）                            |
| 注意力加权 | Q·K，位置编码间接影响注意力            | Q·K + Q·r_{i-j}，位置直接作用于注意力分数        |
## 6️⃣ 优缺点
**优点**：
1. 更自然地捕捉 **token 间的相对关系**
2. 对长序列或外推更友好
3. 能更好地建模局部依赖和长距离依赖
**缺点**：
4. 计算复杂度略高
5. 实现复杂，需要处理相对位置索引和偏移
## 7️⃣ 应用场景
- **Transformer-XL**：长文本建模
- **T5**：相对位置编码提升性能
- **XLNet**：结合 permutation LM 和相对位置
- 长序列任务，如代码、音乐、DNA 序列建模
## 8️⃣ 总结理解
- 绝对位置编码告诉模型“我在第 i 个位置”
- 相对位置编码告诉模型“我和其他 token 的距离是多少”
- 对很多任务，相对位置比绝对位置更有效，尤其是**序列长度不固定或需要长距离依赖**

# 四、RoPE（Rotary Positional Encoding）
## 1. 背景
Transformer 模型需要对序列中每个 token 的位置进行编码，以利用顺序信息。常见位置编码方法包括：
- **绝对位置编码 (Absolute PE)**：例如 Sinusoidal PE 或 Learnable PE
- **相对位置编码 (Relative PE)**：通过表示 token 之间的相对位置来计算注意力
RoPE 是一种 **在注意力计算中直接融入位置信息的旋转编码方法**，能够自然支持 **相对位置关系**，并兼容标准的自注意力机制。
## 2. 核心思想
RoPE 的核心是对 **query/key 向量进行旋转**，旋转角度与 token 的位置相关，使注意力计算天然编码了相对位置信息。
与传统加法式位置编码不同，RoPE 通过 **二维旋转矩阵或复数旋转**作用在向量上，实现位置编码。
### 2.1 说明
1. 每个token的单个维度不具有可解释的“语义含义”
2. **一对维度**：构成一个 **可以旋转的向量**，即一个带频率 $\omega_i$ 的相对位置通道
3. 一对维度根据“位置信息”进行旋转，长度不变，内积结构不变，即原本信息不变
4. **单个维度对**：只能感知某一尺度的距离模式
5. **所有维度对叠加**：构成连续、多尺度、可学习的相对位置注意力
6. **Attention 的本质**：内容匹配 × 距离匹配 的加权和
## 3. 数学公式
假设 token embedding 向量 $x \in \mathbb{R}^d$，将其拆成每两个维度一组$(x_{2i}, x_{2i+1})$，对每组应用旋转矩阵：
$$\begin{bmatrix} x'_{2i} \\ x'_{2i+1} \end{bmatrix} =
\begin{bmatrix} \cos \theta_i & -\sin \theta_i \\ \sin \theta_i & \cos \theta_i \end{bmatrix}
\begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix} =
\begin{bmatrix} x_{2i} \cos \theta_i - x_{2i+1}\sin \theta_i \\ x_{2i} \sin \theta_i + x_{2i+1} \cos \theta_i
\end{bmatrix}$$
其中：
$$
\theta_i = \text{position} \times \omega_i
$$
- $\text{position}$：token 在序列中的位置（0, 1, 2, …）  
- $\omega_i = 10000^{-2i/d}$：每组维度的频率，低维度频率高，高维度频率低。随着维度 $i$ 增大，频率 $\omega_i$ **指数级减小**，类比“d/2根角度相差指数倍的向量”。在注意力计算中，每个维度对在不同区间对注意力分数的贡献有正有负，一对维度只体现周期性的距离模式，多对=连续傅里叶基，分数累加=在相对距离轴上，用多对正弦基对 token-token 的距离关系做投影。总结，所有**维度对**体现了两个token的**相对位置**。
x=[1,16,5,128]
![](Learning/AI%20Infra/Pasted%20image%2020260114054506.png)
扩展到向量计算：
$\omega(dim/2,1) \cdot \text{position}(1,num_{seq}) = \theta_t(dim/2,num_{seq}) \xrightarrow{transpose} \theta_{half}(num_{seq},dim/2)$
$\theta_{half}(num_{seq},dim/2) \xrightarrow{concatenate} \theta(num_{seq},dim)$
$x(num_{seq},dim) \cdot cos\theta() - x \cdot sin\theta$


## 4. 注意力计算中的作用
标准自注意力：
$$
\text{Attention}(Q, K, V) = \text{softmax}\Big(\frac{QK^T}{\sqrt{d}}\Big)V
$$
使用 RoPE 后：
$$
Q' = \text{RoPE}(Q, \text{pos}), \quad K' = \text{RoPE}(K, \text{pos})
$$
$$
\text{Attention}(Q', K', V)
$$
- $Q'_i {K'_j}^T$ 自动包含 **相对位置 $i-j$** 信息  
- 不依赖额外的相对位置矩阵  
- 支持任意长度序列  
## 5. 几何直觉
### RoPE（旋转）
- 将 embedding 看作高维空间中的箭头  
- 每两个维度一组旋转，方向随 token 位置变化  
- 内积大小 = cos(方向差) → 自然表示相对位置  

**二维示意：**
```
y  
↑  
| ↑ (pos=3)  
| ↖ (pos=2)  
| ↗ (pos=1)  
| → (pos=0)  
+----------------> x
```
### Sinusoidal PE（平移）
- 在 embedding 上直接加上 `[sin(pos*ω), cos(pos*ω)]`  
- 向量方向不变，只是平移  
- 相对位置需要额外计算  
```
位置 0: →  
位置 1: →  
位置 2: →  
位置 3: →
```
## 6. 优点总结
1. **自然支持相对位置**：注意力分数直接编码 \(i-j\) 信息  
2. **长度可扩展**：不受最大位置限制  
3. **与多头注意力兼容**：直接作用在 Q/K 上  
4. **无需额外参数**：频率可固定，可选可训练频率  
## 7. 对比总结

| 方法 | 几何变化 | 相对位置 |
|------|---------|---------|
| Sinusoidal PE | 向量平移 | 内积不直接包含 |
| RoPE | 向量旋转 | 内积天然包含 |