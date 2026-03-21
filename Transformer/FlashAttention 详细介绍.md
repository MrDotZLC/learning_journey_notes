## 1. 背景：Attention 的数学定义与工程现实之间的矛盾

Transformer 中的 Attention 在数学上定义为：

$$ O = \sum_j \alpha_j V_j, \quad \alpha_j = \frac{e^{x_j - m}}{\sum_k e^{x_k - m}}, \quad x_j = Q \cdot K_j $$

其本质是对所有 Key 的贡献做一次**全局归一化加权和**。

数学层面无任何问题，但工程上隐含了代价极高的中间量：

- $x_j$ 来自 $QK^\top$，尺寸为 $T \times T$
- Softmax 需保存每个 $e^{x_j}$ 或 $\alpha_j$
- Backward 还需再次使用这些量

结论：**Attention 的显存复杂度为 $O(T^2)$**，在长序列场景下不可接受。

---

## 2. 问题：为什么"直接算 $O$ 再丢弃中间量"行不通

一个自然的想法是：既然最终只需要 $O = \sum_j \alpha_j V_j$，直接累加后丢弃中间结果是否可行？

问题出在 **Softmax 的反向传播**。Softmax 的 Jacobian 为：

$$ \frac{\partial \alpha_i}{\partial x_j} = \alpha_i (\delta_{ij} - \alpha_j), \quad \delta_{ij} = \begin{cases} 1 & i = j \ 0 & i \neq j \end{cases} $$

每个输入 $x_i$ 的梯度显式依赖对应的 $\alpha_i$。而 $O = \sum_j \alpha_j V_j$ 已将所有 $\alpha_j$ 混合压缩，无法反推出每个位置的权重。

**结论：只存 $O$，不可能正确执行 Softmax Backward。**

这条约束直接决定了后续所有方案的设计空间。

---

## 3. 解决方案总体思路：分块 + 保持全局 Softmax

在显存受限的前提下，目标是：

1. Forward 不存 $QK^\top$，不存 $\alpha$
2. Backward 仍能恢复**全局 Softmax 的 $\alpha_j$**

唯一可行的方向：

> **把 Attention 的"求和结构"拆开，但不改变 Softmax 的数学定义。**

拆的是**计算顺序**，不是 Softmax 空间。这正是 Flash Attention 的出发点。

---

## 4. 方案细节：Flash Attention 的实现
![](assets/Pasted%20image%2020260129072911.png)

### 4.1 从数学上重写 Attention

令 $m = \max_j x_j$ 为**全局最大值**（所有位置上的最大得分）。定义：

$$ P = \sum_j e^{x_j - m} V_j \in \mathbb{R}^d, \qquad Z = \sum_k e^{x_k - m} \in \mathbb{R} $$

则 Attention 输出为：

$$ O = \frac{P}{Z} $$

减去 $m$ 不改变结果（$P$ 和 $Z$ 同乘 $e^{-m}$，相除后与原式完全等价），仅用于数值稳定。

关键观察：$P$ 是对 $j$ 的**向量逐项累加**，$Z$ 是对 $k$ 的**标量逐项累加**，二者相互独立，均可分块增量维护，因此：

> 只要能把所有位置的贡献逐步累加到 $P$ 和 $Z$，**不需要一次性看到所有位置**。

> **注意：** 此处 $m$ 是针对全序列的全局最大值，实际分块处理时事先未知。4.2 节的 $m_b$ 是 Block $b$ 的局部最大值，二者不同；如何在遍历 Block 的过程中动态维护全局 $m$，正是 4.3/4.5 节在线算法的核心。

### 4.2 按 Key/Value 维度分块

将所有 Key/Value 划分为若干 Block，每次只处理一小块。对某个 Block $b$，可独立计算的**局部**统计量为：

$$ m_b = \max_{j \in b} x_j, \quad l_b = \sum_{j \in b} e^{x_j - m_b} $$

其中 $m_b$ 是 Block $b$ 的**局部最大值**，仅用于 Block 内部的数值稳定计算。

Block $b$ 内部的局部归一化输出（FA-1 的 Block 级中间量）：

$$ O_b = \frac{\sum_{j \in b} e^{x_j - m_b} V_j}{l_b} $$

Block $b$ 内部的局部未归一化加权和（FA-2 的 Block 级中间量）：

$$ \tilde{O}_b = \sum_{j \in b} e^{x_j - m_b} V_j $$

> **注意：** $m_b$ 因 Block 而异，导致不同 Block 的 $l_b$ 和 $\tilde{O}_b$ 基准各不相同，**不能跨 Block 直接相加**。全局可加的累积形式需要以滚动全局最大值 $m_t$ 统一基准，见 4.3/4.5 节。

### 4.3 在线（Online）Softmax 合并所有 Block（Flash Attention 1）

FA-1 在遍历 Block 时维护三个全局状态：全局最大值 $m_{t-1}$、全局归一化因子 $l_{t-1}$、当前累积输出 $O_{t-1}$。

**更新最大值：**

$$ m_t = \max(m_{t-1},\ m_t^{(\text{block})}) $$

**更新归一化因子：**

$$ l_t = \sum_{j \leq t} e^{x_j - m_t} = \sum_{j < t} e^{x_j - m_t} + \sum_i e^{x_i^t - m_t} $$

由于 $l_{t-1} = \sum_{j < t} e^{x_j - m_{t-1}}$，历史部分换基准：

$$ \sum_{j < t} e^{x_j - m_t} = \sum_{j < t} e^{x_j - m_{t-1}} \cdot e^{m_{t-1} - m_t} = l_{t-1} e^{m_{t-1} - m_t} $$

则：

$$ l_t = l_{t-1} e^{m_{t-1} - m_t} + \sum_i e^{x_i^t - m_t} $$

**更新输出：**

展开定义：

$$ O_t = \frac{\sum_{(x,V) \in S_{\leq t}} e^{x - m_t} V}{l_t} = \frac{\sum_{(x,V) \in S_{< t}} e^{x - m_t} V + \sum_{(x,V) \in S^{(t)}} e^{x - m_t} V}{l_t} $$

由 $O_{t-1}$ 的定义得：

$$ \sum_{(x,V) \in S_{< t}} e^{x - m_t} V = \sum_{(x,V) \in S_{< t}} e^{x - m_{t-1}} e^{m_{t-1} - m_t} V = l_{t-1} e^{m_{t-1} - m_t} O_{t-1} $$

代入得 FA-1 的递推公式：

$$ O_t = \frac{l_{t-1} e^{m_{t-1} - m_t} O_{t-1} + \sum_i e^{x_i^t - m_t} V}{l_t} $$

**最终保证：**

$$ O = \frac{\sum_{\text{all } j} e^{x_j} V_j}{\sum_{\text{all } j} e^{x_j}} $$

> Flash Attention 始终在做"一次全局 Softmax"，只是分多步完成。

### 4.4 为什么只需要存 $m$ 和 $l$

Softmax 权重可写为：

$$ \alpha_j = \frac{e^{x_j - m}}{l} $$

因此：

- Forward 时存下 $m,\ l$
- Backward 时重新计算 $x_j = QK^\top$
- 即可精确恢复每一个 $\alpha_j$

满足 Softmax Backward 的要求：不存 $\alpha$，但能重建 $\alpha$。

### 4.5 FA-2：推迟归一化到最后一步

**FA-1 的低效点：** $O_t$ 在每步都是归一化后的结果。下一步需要历史未归一化加权和时，必须反乘 $l_t$：

$$ \sum_{(x,V) \in S_{< t}} e^{x - m_t} V = l_t \cdot O_t $$

每块都经历一次多余的"除后又乘"。

**FA-2 改动：** 全程维护未归一化的累积输出 $\tilde{O}$，$m_{t-1}$ 和 $l_{t-1}$ 的更新与 FA-1 完全相同。

**$\tilde{O}$ 的递推推导：**

按定义展开：

$$ \tilde{O}_t = \sum_{(x,V) \in S_{\leq t}} e^{x - m_t} V $$

拆分为历史部分与当前块：

$$ \tilde{O}_t = \sum_{(x,V) \in S_{< t}} e^{x - m_t} V + \sum_{(x,V) \in S^{(t)}} e^{x - m_t} V $$

对历史部分换基准：

$$ \sum_{(x,V) \in S_{< t}} e^{x - m_t} V = \sum_{(x,V) \in S_{< t}} e^{x - m_{t-1}} \cdot e^{m_{t-1} - m_t} \cdot V = e^{m_{t-1} - m_t} \cdot \tilde{O}_{t-1} $$

代入得 FA-2 的递推公式：

$$ \boxed{\tilde{O}_t = e^{m_{t-1} - m_t} \cdot \tilde{O}_{t-1} + \sum_{(x,V) \in S^{(t)}} e^{x - m_t} V} $$

**FA-1 vs FA-2 对比：**

|版本|递推公式|每步是否做除法|
|---|---|---|
|FA-1|$O_t = \dfrac{l_{t-1} e^{m_{t-1}-m_t} O_{t-1} + \sum_i e^{x_i^t - m_t} V}{l_t}$|✅ 每块除一次 $l_t$|
|FA-2|$\tilde{O}_t = e^{m_{t-1}-m_t} \tilde{O}_{t-1} + \sum_{(x,V) \in S^{(t)}} e^{x-m_t} V$|❌ 全程无除法|

**最终归一化：** 所有 Block 处理完毕后，执行唯一一次除法：

$$ O = \frac{\tilde{O}_T}{l_T} $$

展开验证：

$$ O = \frac{\sum_{(x,V) \in S_{\leq T}} e^{x - m_T} V}{\sum_{j \leq T} e^{x_j - m_T}} = \frac{\sum_{\text{all } j} e^{x_j} V_j}{\sum_{\text{all } j} e^{x_j}} $$

与标准 Softmax Attention 完全等价。$\square$

---

## 5. 方案对比：Flash Attention vs MemEff Attention

Forward 结果完全一致，但数学结构不同。

### 5.1 MemEff 的核心思路

MemEff Attention 的做法：每个 Block **独立完成一次 Softmax**，得到 Block 内的输出 $O_b$，再用 Block 权重加权平均：

$$ O = \frac{\sum_b w_b O_b}{\sum_b w_b} $$

从代数上看，这是成立的。

### 5.2 本质差异

|维度|MemEff|Flash|
|---|---|---|
|Softmax 空间|Block 局部|全局|
|归一化因子|每 Block 一个|全局一个|
|数学形式|"先 Softmax，再平均"|"一次 Softmax，分块算"|
|Backward|需存 Block 权重|可重算 $\alpha$|

### 5.3 为什么 Backward 迫使选择 Flash
[Attention Softmax Backward反向传播公式推导](Attention%20Softmax%20Backward反向传播公式推导.md)
Softmax Backward 的梯度形式为：

$$ \frac{\partial L}{\partial x_i} = \alpha_i \left(\frac{\partial L}{\partial O}\right)^\top (V_i - O) $$

此处 $\alpha_i$ **必须是全局 Softmax 权重**。

- Flash Attention：可通过 $m,\ l$ 恢复全局 $\alpha_i$
- MemEff Attention：Block 内的 $\alpha_i$ 不等于全局 $\alpha_i$

> **Flash Attention 是被 Softmax Backward "逼出来的最优解"。**

---

## 6. FA-1 vs FA-2 综合对比

|版本|特点|优点|缺点|
|---|---|---|---|
|**FA-1**|每块结束立即归一化，存储 $m,\ l,\ O$|显存显著降低，可直接重算 Softmax|每块一次除法，SRAM 写回次数多|
|**FA-2**|存储 $\tilde{O}$，归一化延迟到最后；$Q$ 分块置于外循环|消除中间除法，支持 Warp 级并行，吞吐提升约 2×|Kernel 实现更复杂，需严格调度|

**FA-2 性能提升的真实来源**（省去每块除法本身算术收益有限）：

**减少 SRAM 写回。** FA-1 每块须将归一化后的 $O_t$ 写回，下一块读回再乘 $l_t$。FA-2 的 $\tilde{O}$ 在整个序列处理期间是同一累积量，只在最后写出一次。

**支持 Warp 级并行。** FA-2 将 $Q$ 分块置于外循环、KV 置于内循环，不同 Warp 处理不同 $Q$ 块时彼此完全独立，无需同步 $\tilde{O}$，直接支持 Thread Block 内的 Warp 级并行。

---

## 7. 整体因果链

$$ \text{全局 Softmax 定义} \Rightarrow \text{Backward 要求知道每个 } \alpha_i \Rightarrow \text{显存限制不允许保存 } \alpha \text{ 或 } QK^\top $$

$$ \Rightarrow \text{Forward 存 } m, l \text{，Backward 重算} \Rightarrow \text{在线维护 } m, l \text{ 要求全局在线 Softmax} \Rightarrow \text{Flash Attention} $$

> Flash Attention 不是一个"更快的 trick"，而是在数学正确性、反向传播和显存约束三者之间，**唯一自洽的工程实现方式**。
