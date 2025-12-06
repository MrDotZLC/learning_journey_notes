# 一、Attention（注意力）机制
### 1. 核心概念
- **本质**：让模型在处理信息时，不是平均对待每个输入，而是**动态选择性关注重要部分**。
- **灵感来源**：人类视觉和认知：看一幅图或读一句话时，我们会不均匀地关注不同部分。
### 2. 组成
注意力机制通常有三个核心向量：
1. **Query (Q)**：当前需要关注的信息。
2. **Key (K)**：输入中每个位置的“标签”或“特征”。
3. **Value (V)**：实际要提取的信息。
- 公式（Scaled Dot-Product Attention）：
$$\text{Attention}(Q,K,V) = \text{softmax}\Big(\frac{QK^T}{\sqrt{d_k}}\Big) V$$
解释：
- $QK^T$：计算 Query 对 Key 的匹配程度（相似度）。
- $\frac{1}{\sqrt{d_k}}$​：缩放，防止维度大导致梯度消失。
- softmax：转成权重（概率分布）。
- 乘 V：加权求和得到输出。
### 3. 注意力类型
1. **普通注意力（vanilla attention）**
	- Q、K、V 可以来自不同来源。
	- 例子：
		- **Encoder-Decoder Attention**（机器翻译）：
			- Q：Decoder 上下文
			- K/V：Encoder 输出
			- 作用：让 Decoder 生成下一个词时关注 Encoder 的相关输入。
2. **自注意力（Self-Attention）**
    - Q、K、V 都来自同一个输入序列。
    - 作用：建模序列内部各位置的依赖关系。
# 二、Self-Attention（自注意力）机制

## 1. 核心概念
- **目标**：捕捉序列中每个位置与所有位置的依赖关系（全局依赖）。
- **优点**：
	- 高效捕捉全局依赖，不依赖递归传播。
	- 并行化友好，序列计算不依赖前一步结果，适合 GPU/TPU 等硬件加速。
	- 处理长距离依赖能力强，长距离元素间的依赖可以直接建模，不会被梯度衰减影响。
	- 权重由内容相似性驱动，实现动态加权信息聚合。
	- 适合多头并行，可以捕捉不同语义子空间的特征。
## 2. 计算流程（Scaled Dot-Product Attention）
给输入矩阵$X \in \mathbb{R}^{n \times d}$，n 为token序列大小，d 为每个token的维度数：
1. 线性映射：
$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V​$$
2. 计算相似度（点积）并缩放：
$$\text{scores} = \frac{QK^\top}{\sqrt{d_k}}$$
3. softmax 得到注意力权重：
$$A = \text{softmax}(\text{scores})$$
4. 加权求和得到输出：
$$\text{Attention}(Q,K,V) = AV$$
**一句话总结**：
		输出 = `softmax(QK^T / sqrt(d_k)) * V`，每个位置融合序列中所有信息。
## 3. 关键细节
- **缩放系数 $\sqrt{d_k}$​​**：控制点积范围，防止 softmax 梯度过小。[[Transformer 中单头维度 d_k ​与缩放因子总结]] 
- **Mask**：
  在softmax 前加上 mask ，将无效注意力权重置为负无穷。
$$\text{Attention}(Q,K,V) = \text{softmax}\Big(\frac{QK^\top}{\sqrt{d_k}} + \text{mask}\Big) V$$
    - Causal mask（生成任务，防止看到未来）
$$\text{mask}_{i,j} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$
    - Padding mask（处理变长序列，即补齐的token）
$$\text{mask}_{i,j} = \begin{cases} 0 & j \text{ 是有效 token} \\ -\infty & j \text{ 是 padding token} \end{cases}$$
- **位置编码**：补充序列顺序信息[[Positional Encoding位置编码]]
    - 绝对位置编码（sin/cos 或 learnable）
    - 相对位置编码（T5、Transformer-XL 等）
## 4. 多头注意力（Multi-Head Attention）[[Transformer 中单头维度 d_k ​与缩放因子总结]]
- 将 d 维分为 h 个头，每个头独立做 attention；
- 输出拼接后线性变换回原维度：
$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1,...,\text{head}_h)W_O$$
- 不同头可以学习不同类型的依赖关系。
## 5. 特点与复杂度
- **特点**：
    - 捕捉全局依赖
    - 并行化友好
    - 可解释性较好（attention 权重）
- **复杂度**：时间和空间为 $O(n^2 d)$，长序列时代价大
- **优化方向**：
    - 稀疏/局部注意力（Longformer、BigBird）
    - 低秩或核化近似（Performer、Linformer）
    - 滑动/层次化 attention
## 6. 工程实践
- 常见结构：`Attention → Dropout → Residual → LayerNorm → FFN → Residual → LayerNorm`
- 初始化：线性变换 $W_Q,W_K,W_V$ 使用标准初始化（例如 xavier）
- 数值稳定性：在 softmax 前做缩放，并在实现时使用稳定的 softmax（常见库都处理得好）
- 注意 dropout：在 attention 权重（A）或输出上做 dropout，有助 regularize
- 可视化注意力权重矩阵辅助理解模型行为
- 对长序列或显存紧张，可用低精度训练或稀疏注意力。

# 三、注意力 vs 自注意力对比

| 特性       | 注意力机制（Attention）                | 自注意力（Self-Attention）           |
| -------- | ------------------------------- | ------------------------------ |
| Q/K/V 来源 | 可以不同                            | 相同（来自同一序列）                     |
| 作用       | 可以跨序列或跨模态关注                     | 序列内部各元素互相关注                    |
| 典型应用     | Encoder-Decoder Attention、图像注意力 | Transformer Encoder/Decoder 内部 |
- **注意力机制**是一个**泛化概念**，自注意力是其中的一种特殊形式。
- 自注意力是 **Transformer 的核心**，让序列内部每个元素能够互相“关注”，捕捉全局依赖。