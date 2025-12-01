# Sinusoidal Position Encoding 总结（Markdown 版）

## 1. 为什么需要位置编码？
Transformer 的 Self-Attention **没有顺序概念**。  
Sinusoidal PE 为每个 token 提供可区分的“位置信息”。
## 2. 核心思想
使用多种频率的 **sin/cos** 函数，将位置 `pos` 转换为一个 **唯一的高维向量**。

## 3. 公式
`PE(pos, 2i)   = sin(pos / 10000^(2i/d)) PE(pos, 2i+1) = cos(pos / 10000^(2i/d))`
- 偶数维度：sin
- 奇数维度：cos
- 维度越大 → 频率越低（周期更长）

## 4. 为什么要多种频率？
- 高频：反映局部相邻 token 的位置信息
- 低频：反映长距离全局位置  
    → 多尺度位置特征同时存在。

## 5. 最大优势：支持长度外推
利用三角恒等式：
`sin(a+b) = sin(a)*cos(b) + cos(a)*sin(b)`
模型可以通过两段 PE 推断 **相对距离 (pos2 - pos1)**，  
因此 **支持比训练时更长的序列**。

## 6. 为什么基数是 10000？
让不同维度的周期合理覆盖：
- 从 2π（高频）
- 到 1e4（低频）
适配 NLP 典型长度（几十到几千）。

## 7. 如何与模型结合？
`InputEmbedding = TokenEmbedding + PositionalEncoding`
直接相加即可。

## 8. 特点
### ✔ 优点
- 无需训练（固定 PE）
- 可外推到更长序列
- 描述相对位置信息
### ✘ 缺点
- 特别长序列容易出现周期折返
- 现代大模型多改用 RoPE、ALiBi 等更强的方案

## 9. 一句话总结

> **Sinusoidal PE 用一组不同周期的正弦/余弦函数，为 Transformer 提供可区分、可外推、具有相对距离结构的绝对位置编码，是最经典也最具数学优雅的 PE 方案。**