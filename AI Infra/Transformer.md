# 一、背景
为什么需要 Transformer？
在 2017 年之前，主流模型是：[[CNN、RNN、LSTM、GRU介绍与对比]]
- **RNN / LSTM**：顺序处理 → 无法并行 → 训练慢
- **CNN**：擅长局部模式，但难以捕捉长距离依赖
**Transformer = 完全抛弃循环结构 + 完全并行化**  
关键创新就是一句话：
> **Attention is all you need：让模型在任意位置之间直接建立依赖关系。**

# 二、一个最小可运行 Transformer 的构成（AI Infra 视角）
可以把 Transformer 看成一个 **堆叠的计算模块 System**，每层包含：
```
输入 Embedding
   ↓
Self-Attention
   ↓
Feed Forward Network (MLP)
   ↓
残差连接 + LayerNorm
```
这种结构可以重复 N 层，全是矩阵乘法 → **非常适合 GPU/TPU、张量并行、流水并行、数据并行等 Infra 技术**。
1. Token Embedding（把词变成向量）[[Token Embedding 全面解析]]
文本 → 数字（tokenize） → 向量（lookup table）
2. Positional Encoding（让模型知道顺序）[[Positional Encoding位置编码]]
		因为 transformer 没有循序结构，所以必须告诉它：
	- 句子顺序
	- 哪个 token 在哪个位置
		可以用：
	- 经典的 Sinusoidal PE
	- 或者可训练位置编码（大模型主流）
3. Self-Attention（自注意力机制，核心：全局依赖）
		特点：
	- Q·Kᵀ 得到 token 与 token 之间的相关度
	- softmax 归一化成权重
	- 权重对 V 求和，实现“聚焦”
		用一句工程化的话总结：
		`Self-Attention = 一个可学习的“相关性矩阵”，告诉模型哪些位置互相有关系。`
		对每个 token 计算：
```
Q = XWq
K = XWk
V = XWv
Attention(Q, K, V) = softmax(Q*Kᵀ / sqrt(d)) * V
```
4. 多头 Attention[[Transformer 中单头维度 d_k ​与缩放因子总结]]
		不是一个 Q/K/V，而是多个头并行：
	- 不同 head 学不同模式（语法、语义、指代等）
	- 拼接 concat 后再映射回 d_model
5. Feed Forward Network (FFN)[[FFN与MLP]]
		特点：
	- 完全位置独立 → 容易张量并行
	- 能提升模型表达能力
		对每个 token 位置独立做两层 MLP：
```
FFN(x) = max(0, xW1 + b1)W2 + b2
```
6. 残差连接 + LayerNorm（保持稳定训练）[[梯度消失、梯度爆炸、残差、LayerNorm、BatchNorm]]
   作用：
	- 残差：避免梯度消失
	- LN：使训练更稳定、收敛更快
		每个子模块后面：
	```
	x = x + dropout(sub_layer_output)
	x = LayerNorm(x)
	```
# 三、Transformer 的整体结构（最简图）
```
Input → Embedding → Positional Encoding
      ↓
   ┌───────────────┐
   │  Self-Attn     │
   ├───────────────┤
   │    FFN         │  ← 这两部分重复 N 次 (堆叠)
   └───────────────┘
      ↓
   Output Layer → Softmax (语言模型)
```

大型语言模型（LLM）就只是把：
- 层数 N 增加
- 隐藏维度增大
- 多头数量增多
- 训练语料增加
- 训练规模用到数千 GPU
你看到的 GPT-3、GPT-4、Llama3 本质仍是 **这个结构堆得更高、数据更多、训练更久**。

# 四、为什么 Transformer 是 AI Infra 的核心？
因为：
## ✔ 1. 完全矩阵化 → 完全可以硬件加速（GPU/TPU）
所有计算都是：
- GEMM（矩阵乘法）
- softmax
- LayerNorm

这些都是 GPU 的强项 → **scale 很容易**。
## 2. 模型可以拆成不同并行方式
比如：
- **Data Parallel**（样本并行）
- **Tensor Parallel**（切 W 矩阵）
- **Pipeline Parallel**（层间分布到不同 GPU）
- **MoE 并行**（专家选择）
    
所有分布式训练框架（Megatron、DeepSpeed、Colossal-AI）都是基于这个结构。

---
## ✔ 3. 模式统一 → 同一架构能做 NLP、CV、Audio
Transformer 已经成为：
- 文本（GPT、Llama）
- 图像（ViT）
- 多模态（CLIP）
- 语音（Whisper）
- Agents
的底座。

从 infra 角度就是：
> **你只需要一次工程优化就能服务所有 AI 模型。**


