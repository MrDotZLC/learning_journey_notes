**动机：** 标准 Attention 计算复杂度 $O(N^2)$，对超长序列（$N > 32k$）代价极高。Sparse Attention 通过**限制每个 Token 只关注部分 Token**，将复杂度降至 $O(N \cdot k)$（$k$ 为关注窗口大小）。

**主要模式：**

| 模式                         | 原理                                             | 代表方案               |
| -------------------------- | ---------------------------------------------- | ------------------ |
| **Sliding Window（局部窗口）**   | 每个 Token 只关注前后 $w/2$ 个 Token，形成带状 Attention 矩阵 | Longformer、Mistral |
| **Global Token（全局 Token）** | 特定 Token（如 `[CLS]`）关注全序列，其余只关注局部               | BigBird、Longformer |
| **随机稀疏**                   | 每个 Token 随机关注 $r$ 个 Token                      | BigBird            |
| **Strided（跨步）**            | 以固定步长采样 Token                                  | Sparse Transformer |

**Sliding Window Attention（Mistral/Mixtral）：**

窗口大小 $w$，每个 Token 只 Attend 最近 $w$ 个 Token，多层叠加后感受野为 $w \times L$（$L$ 为层数），可覆盖较远依赖。

- 优点：计算复杂度 $O(N \cdot w)$，KV Cache 大小从 $O(N)$ 降为 $O(w)$（每层只需保存 $w$ 个 KV）。
- 缺点：超出窗口的长距离依赖完全丢失（见 Q103 Attention Sink 问题）。

**BigBird：** 结合局部窗口 + 全局 Token + 随机稀疏三种模式，理论上可近似任意全注意力，适合文档级别的长文本理解任务。

**适用场景总结：**

- 超长文档理解、代码分析（$N > 32k$）：Sliding Window 足够，局部依赖为主。
- 需要全局语义整合（如问答、摘要）：BigBird 的全局 Token 机制更优。
- 生成任务（Decoder-only）：Sliding Window 结合 Attention Sink（StreamingLLM）实现无限流式生成（见 Q47）。
