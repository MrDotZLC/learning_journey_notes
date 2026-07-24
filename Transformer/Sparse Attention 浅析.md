## 一、背景 

标准 Attention 计算复杂度 $O(N^2)$，对超长序列（$N > 32k$）代价极高。Sparse Attention 通过**限制每个 Token 只关注部分 Token**，将复杂度降至 $O(N \cdot k)$（$k$ 为关注窗口大小）。

---

## 二、主要模式

| 模式                         | 原理                                                      | 代表方案               |
| -------------------------- | ------------------------------------------------------- | ------------------ |
| **Sliding Window（局部窗口）**   | 每个 Token 只关注前后 $w/2$ 个 Token，形成带状 Attention 矩阵          | Longformer、Mistral |
| **StreamingLLM**           | 在 Sliding Window 的基础上，始终保留前 $k$ 个 Sink Token 的 KV Cache |                    |
| **Global Token（全局 Token）** | 特定 Token（如 `[CLS]`）关注全序列，其余只关注局部                        | BigBird、Longformer |
| **随机稀疏**                   | 每个 Token 随机关注 $r$ 个 Token                               | BigBird            |
| **Strided（跨步）**            | 以固定步长采样 Token                                           | Sparse Transformer |

---

## 三、Sliding Window Attention（Mistral/Mixtral）

窗口大小 $w$，每个 Token 只 Attend 最近 $w$ 个 Token，多层叠加后感受野为 $w \times L$（$L$ 为层数），可覆盖较远依赖。

- 优点：计算复杂度 $O(N \cdot w)$，KV Cache 大小从 $O(N)$ 降为 $O(w)$（每层只需保存 $w$ 个 KV）。
- 缺点：超出窗口的长距离依赖完全丢失（见 Q103 Attention Sink 问题）。

---

## 四、StreamingLLM

 **Sliding Window Attention 的问题**： 
 
 直接丢弃窗口外的早期 Token 会导致困惑度（Perplexity）骤增，模型输出崩溃。

**Attention Sink 现象：**

分析 Attention 分布发现，**序列最开始的几个 Token（通常是前 4 个）持续获得异常高的 Attention 权重**，无论输入内容如何。这些 Token 被称为 **Attention Sink**（注意力汇聚点）。

**原因：** Softmax 要求所有 Attention 权重之和为 1。当模型不需要关注任何特定 Token 时，多余的权重"涌入"最初的 Token 作为"垃圾桶"（Sink Token）。这是 Softmax 归一化的数学特性导致的，与 Token 的语义内容无关。

**StreamingLLM 解决方案：**

在 Sliding Window 的基础上，**始终保留前 $k$（默认 4）个 Sink Token 的 KV Cache**，不受窗口限制：

$$\text{KV Cache} = \text{Sink Tokens}(k) \cup \text{Recent Tokens}(w)$$

总 KV Cache 大小固定为 $k + w$，可实现**无限长序列流式生成**，且 Perplexity 与全 KV Cache 方案几乎相同（相差 < 0.1）。

**局限性：** 仅适合不依赖远距离历史的生成任务（如对话），对需要长程依赖的任务（如超长文档问答）无法使用。

---

## 五、BigBird

结合局部窗口 + 全局 Token + 随机稀疏三种模式，理论上可近似任意全注意力，适合文档级别的长文本理解任务。

---

## 六、适用场景总结：

- 超长文档理解、代码分析（$N > 32k$）：Sliding Window 足够，局部依赖为主。
- 需要全局语义整合（如问答、摘要）：BigBird 的全局 Token 机制更优。
- 生成任务（Decoder-only）：Sliding Window 结合 Attention Sink（StreamingLLM）实现无限流式生成（见 Q47）。
