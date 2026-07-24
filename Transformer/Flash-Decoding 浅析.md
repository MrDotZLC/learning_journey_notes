## 一、Decode 阶段 FA 的并行度瓶颈

FA（FA-1/2）的并行维度为 Batch Size × Head 数。Decode 阶段的典型参数：

- $B_{\text{seq}} = 1$（单请求）或 $\leq 64$（在线服务）
- $H = 32$（LLaMA-2 7B）

总并行度 $= B_{\text{seq}} \times H \leq 2048$，而 H100 有 **132 个 SM**，每 SM 可运行多个 Block。

当 $B_{\text{seq}} = 1$，$H = 32$ 时，仅 32 个 CUDA Block 参与计算，大量 SM 空闲。即使每个 Block 处理完整的序列长度 $S$（如 $S = 32768$），也无法填满硬件。

---

## 二、Flash-Decoding 的核心思想：沿序列维度并行

Flash-Decoding 在 FA 的 Batch/Head 并行基础上，增加**第三个并行维度：KV 序列的分块**。

设按 KV 序列 $S$ 切分为 $C$ 块，每块长度 $S/C$，不同 SM 并行处理不同 KV 块。

**三步计算流程：**

**Step 1：并行局部 Attention（各 SM 独立）**

每个 SM 负责 Q（固定，仅 1 Token）与其分配的 KV 块做局部 Attention：

$$o_c,\ \ell_c,\ m_c = \text{LocalAttention}(Q,\ K[c:c+S/C],\ V[c:c+S/C])$$

输出：局部未归一化输出 $o_c \in \mathbb{R}^d$，局部 softmax 统计量 $(\ell_c, m_c)$。

**Step 2：写出中间结果**

所有块将 $(o_c, \ell_c, m_c)$ 写入显存中间缓冲区，大小 $O(C \cdot d)$（而非 $O(S \cdot d)$，通常 $C \ll S$）。

**Step 3：归约（Reduction）**

单独启动一个轻量 Kernel，将 $C$ 个局部结果合并为最终输出，利用 Online Softmax 的可结合性：

$$m_{\text{final}} = \max_c(m_c)$$

$$\ell_{\text{final}} = \sum_c e^{m_c - m_{\text{final}}} \cdot \ell_c$$

$$o_{\text{final}} = \frac{1}{\ell_{\text{final}}} \sum_c e^{m_c - m_{\text{final}}} \cdot o_c$$

---

## 三、并行度与延迟分析

| 方案             | 并行度                                        | 序列 $S=32768$，$H=32$，$B=1$ 的 SM 利用率         |
| -------------- | ------------------------------------------ | ------------------------------------------ |
| FA-2           | $B \times H = 32$                          | $32/132 \approx 24\%$                      |
| Flash-Decoding | $B \times H \times C$（$C$ 可取 $128\sim512$） | $32 \times 128 / 132 \approx 3100\%$（充足过载） |

Flash-Decoding 在长序列 Decode 场景下，延迟可降低 $8\times$（实测 $S=8192$，$d=64$，$B=1$）。

****

## 四、代价：额外显存与归约开销

中间缓冲区大小：$C \times H \times d \times 3$（存 $o, \ell, m$），取 $C=256$，$H=32$，$d=128$，FP32：

$$256 \times 32 \times 128 \times 3 \times 4 \approx 12 \text{ MB}$$

归约 Kernel 的计算量：$O(C \times H \times d)$，远小于主计算量，可忽略。
