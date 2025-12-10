本文总结了 Token Embedding 的原理、训练、语义结构、BPE 影响、大模型 embedding 规模及优化策略。适合用作 AI 知识库参考。

---
## 1. Embedding 从随机到有语义的过程
- **初始化**：Embedding Matrix 随机生成
- **训练中**：参与前向传播计算 loss，再通过梯度下降更新
- **相似上下文词**：梯度方向相似 → 向量接近 → 形成语义簇
示意：
```
训练前：位置随机
cats → [0.12, -0.87, 0.33]
dogs → [-0.55, 0.20, -0.11]
sleep → [0.77, 0.10, 0.89]

训练后：语义簇明显
动物类：cat, dog, rabbit, horse
动作类：run, walk, jump
情绪类：love, hate, like
食物类：apple, orange, banana
```
## 2. Embedding 维度选择
- **Transformer hidden size 决定 embedding dim**
- 示例：GPT-2 small: 768, GPT-2 medium: 1024, GPT-3 175B: 12,288
- **维度越大** → 能表达语义越丰富 → 可支持复杂语法和多任务语义
## 3. BPE 对 embedding 的影响
- 传统 word embedding 需要每个单词一个向量 → 词表大
- LLM 使用 BPE，将单词拆成 subword:
```
unbelievable → un + believe + able
cats → cat + s
playing → play + ing
```
- **优势**：
    - 能处理新词、拼写错误、生词、混合语言
    - embedding 可组合形成新词向量
## 4. 大模型 embedding 规模及优化
- 规模：
    - 词表大小：100k
    - embedding dim：4096
    - 数据类型：fp16 或 fp8
    - 内存需求：100k × 4096 × 2 bytes ≈ 800 MB
- **优化技巧**：
    1. Shared weights（输入 embedding = 输出 embedding）
    2. 低精度存储（fp16 / fp8 / int8）
    3. 词表裁剪（pruning unused token）
## 5. Embedding 空间结构
- **词类簇（cluster）**
    - 上下文相似 → embedding 接近
- **语义方向（semantic directions）**
    - king - queen ≈ man - woman
    - happy - sad ≈ positive - negative
- **句法方向（syntactic）**
    - 动词时态、复数、形容词程度等
## 6. PyTorch 最小示例

```python
import torch
import torch.nn as nn

# 简单词表和 embedding
vocab = {"I":0, "love":1, "cats":2}
embedding_dim = 3
embedding = nn.Embedding(num_embeddings=3, embedding_dim=embedding_dim)
print("初始 embedding：")
print(embedding.weight.data)

optimizer = torch.optim.SGD(embedding.parameters(), lr=0.1)
inputs  = torch.tensor([0, 1])   # "I love"
target  = torch.tensor([2])      # 预测 "cats"
loss_fn = nn.CrossEntropyLoss()
linear = nn.Linear(embedding_dim, 3)

for epoch in range(10):
    optimizer.zero_grad()
    E = embedding(inputs)
    out = linear(E[-1])
    loss = loss_fn(out.unsqueeze(0), target)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, loss = {loss.item():.4f}")

print("训练后 embedding：")
print(embedding.weight.data)
```
## 7. 总结
- Embedding 随机初始化 → 梯度下降训练 → 自动学习语义
- 相似上下文 → 相似向量 → 语义簇与方向
- BPE 提供子词组合能力 → 可处理新词、生词、多语言
- 大模型 embedding 与 Transformer hidden size 一致 → 高维表达丰富语义
- 优化技巧包括共享权重、低精度存储、词表裁剪
- embedding 空间自发形成语义/句法几何结构
---
