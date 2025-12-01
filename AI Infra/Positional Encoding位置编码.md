# Sinusoidal Position Encoding 总结

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

# Learnable Positional Embedding（可训练位置编码）

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

# 