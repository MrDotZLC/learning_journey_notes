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
3. Self-Attention（自注意力机制，核心：全局依赖）[[注意力和自注意力（Attention vs Self-Attention）]]
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

# 四、Transformer 的三种常见架构
**Encoder-only、Decoder-only、Encoder-Decoder**，它们在任务、结构和训练目标上有明显差异。
![[transformer.png]]
注：positional encoding输出的三个箭头分别为Q、K、V。
## 1. Encoder-only（只用 Encoder）
### ✔ 结构特点
- 只有 **Encoder 堆叠层**
- Self-Attention **无 mask**（可以看到全局）
- 只负责“**理解输入**”，不用于自回归生成
### ✔ 数据流
输入序列 → Encoder → 句子/Token 表示 → 下游任务
### ✔ 适用任务
需要理解输入、输出固定长度或不需要生成：
- 文本分类（情感分析）
- 语义匹配
- 命名实体识别（Token 分类）
- 检索 / Embedding 生成
### ✔ 代表模型
- BERT
- RoBERTa
- ALBERT

## 2. Decoder-only（只用 Decoder）
### ✔ 结构特点
- 只有 **Decoder 层**
- 使用 **Masked Self-Attention**  
    → 只能看到左侧 token（自回归生成）
- “理解输入 + 生成输出”在同一序列中完成
### ✔ 数据流
完整上下文序列（含历史 + 当前输入） → Masked Self-Attention → 预测下一个 token
### ✔ **对当前序列自回归**
$$P(x_t | x_1, ..., x_{t-1})$$
- “前文”已包含在序列本身，不需要 Encoder 提供额外上下文
- 理解依靠自注意力内部完成
### ✔ 适用任务
- 对话生成
- 文本续写
- 写作助手、代码生成
- 大型通用语言模型（LLM）
### ✔ 代表模型
- GPT 系列
- LLaMA 系列
- Falcon, Mistral 等
## 3. Encoder-Decoder（Seq2Seq）
### ✔ 结构特点
- Encoder：编码输入序列（前文）
- Decoder：基于输入 + 已生成 token 自回归生成输出
- Decoder 具有两种注意力：
    - Masked Self-Attention：对输出序列自回归
    - Encoder–Decoder Attention：引用前文表示
### ✔ 数据流
Encoder 输入序列 X → Encoder 输出  
Decoder 输入（已生成的 Y） →  
Masked Self-Attention + Encoder-Decoder Attention → 下一个 token
### ✔ 只对当前的“输出序列”自回归
$$P(y_t | y_1, ..., y_{t-1}, X)$$
- 输入序列 X 不参与自回归，只是条件 context
### ✔ 适用任务
需要“**输入序列 → 输出序列**”的场景：
- **机器翻译**（最典型）
- 摘要生成
- 语音 → 文本
- 文本 → SQL / 文本 → 结构化信息
### ✔ 代表模型
- T5
- BART / mBART
- Whisper（ASR）
## 🔍 三种结构的核心区别（一句话版）

| 架构类型            | 自回归在哪里？ | 前文如何使用？        | 理解与生成关系                  |
| --------------- | ------- | -------------- | ------------------------ |
| Encoder-only    | ❌ 不自回归  | 不需要            | 只理解                      |
| Decoder-only    | ✔ 当前序列  | 前文已包含在序列中      | 理解与生成合一                  |
| Encoder-Decoder | ✔ 输出序列  | Encoder 输出作为条件 | 分工：Encoder 理解、Decoder 生成 |
## 🧩 最直观总结
- **Encoder-only = 理解机器**
- **Decoder-only = 生成机器（LLM 默认架构）**
- **Encoder-Decoder = 输入→输出的转换机器（翻译等）**

