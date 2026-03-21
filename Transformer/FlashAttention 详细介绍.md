## 一、背景：attention 的数学定义与工程现实之间的矛盾
Transformer 中的 attention 在数学上非常简单。  
对一个固定的 query，其输出定义为：
$$O = \sum_j \alpha_j V_j,\quad  
\alpha_j = \frac{e^{x_j-m}}{\sum_k e^{x_k-m}},\quad  
x_j = Q \cdot K_j$$
这是一个**标准的全局safe softmax 加权和**，其中：
- attention 的本质是：  
    **对所有 key 的贡献做一次全局归一化**

在数学层面，这个定义没有任何问题。
但在工程上，这个定义隐含了一个代价极高的中间量：
- $x_j$ 来自 $QK^\top$，尺寸是 $T \times T$
- softmax 需要保存每个 $e^{x_j}$ 或 $\alpha_j$
- backward 还要再次使用这些量
结果是：  
**attention 的显存复杂度是 (O(T^2))**，在长序列场景下不可接受。
# 二、问题：为什么“简单省内存”行不通
一个自然的想法是：
既然最终只需要  $\displaystyle O = \sum_j \alpha_j V_j$，那为什么不直接算 O，把中间结果丢掉？
问题出在 **softmax 的反向传播**。
softmax 的 Jacobian 是：
$$\frac{\partial \alpha_i}{\partial x_j}=\alpha_i(\delta_{ij} - \alpha_j),\quad \delta_{ij} = \begin{cases} 1 & i=j \\ 0 & i\neq j \end{cases}$$
这意味着：
- 每一个输入 $x_i$ 的梯度
- 都**显式依赖对应的 $\alpha_i$**
而 $O$ 只是一个加权和：
$$O = \sum_j \alpha_j V_j$$  
它**已经把所有 $\alpha_j$ 混合压缩**，无法反推出每个位置的权重。因此结论是：**只存 O，不可能正确做 softmax backward**
这条约束，直接决定了后面所有方案的设计空间。
# 三、解决方案的总体思路：分块 + 保持全局 softmax
在显存受限的前提下，想要做到：
1. forward 不存 $QK^\top$、不存 $\alpha$
2. backward 仍然能恢复 **全局 softmax 的 $\alpha_j$**

唯一可行的方向是：
> **把 attention 的“求和结构”拆开，但不改变 softmax 的数学定义**

注意这里的关键词是：
- 拆的是 **计算顺序**
- 不是拆 **softmax 空间**
这正是 Flash Attention 的出发点。
# 四、方案细节：Flash Attention 是如何做到的
![](assets/Pasted%20image%2020260129072911.png)
## 4.1 从数学上重写 attention
attention 输出可以写成：
$$O=\frac{\sum_j e^{x_j-m} V_j}{\sum_j e^{x_j-m}}$$
这是一个非常关键的形式，
二者都是 **对 j 的可加求和项**。
这意味着：

> 只要能把所有 j 的贡献累加起来，  
> **不需要一次性看到所有 j**
## 4.2 按 key/value 维度分块
将所有 key/value 划分为若干 blocks，每次只处理一小块。
对某个 block (b)，定义：
$$m_b = \max_{j\in b} x_j$$
$$l_b = \sum_{j\in b} e^{x_j - m_b}$$
FA-1 中维护 $O_b$：
$$O_b = \frac{\sum_{j \in b} e^{x_j - m_b} V_j} {l_b}$$
FA-2 中维护 $\tilde{O}_b$：
$$\tilde{O}_b = \sum_{j\in b} e^{x_j - m_b} V_j$$
注意一个非常重要的点：

> $\tilde{O}_b$ 不是 attention 输出，它只是 softmax 分子的“一部分”。
## 4.3 在线（online）softmax 合并所有 block（Flash Attention 1）
Flash Attention 在遍历 blocks 时，维护三个全局状态：
- 全局最大值 $m_{t-1}$
- 全局归一化因子 $l_{t-1}$​
- 当前累积输出 $O_{t-1}​$

当处理 block t 时：
- 更新最大值
$$m_t = \max(m_{t-1}, m_t^{(block)})$$

- 更新归一化因子
$$l_t = \sum_{j \le t} e^{x_j - m_t} \\ = \sum_{j \lt t} e^{x_j - m_t} + \sum_i e^{x_i^t-m_t}$$

$$l_{t-1} = \sum_{j < t} e^{x_j - m_{t-1}}$$
$$\sum_{j < t} e^{x_j - m_t} = \sum_{j < t} e^{x_j - m_{t-1}} \cdot e^{m_{t-1} - m_t}$$
则：
$$l_t = l_{t-1} e^{m_{t-1} - m_t} + \sum_i e^{x_i^{t}-m_t}$$
- 更新输出
  
  于是：
  $$O_t = \frac{\sum_{(x,V)\in S_{ \le t}} e^{x - m_t} V} {l_t} =  \frac{\sum_{(x,V)\in S_{ \lt t}} e^{x - m_t} V + \sum_{(x,V)\in S^{(t)}} e^{x - m_t} V} {l_t}$$
  $$O_{t-1} = \frac{\sum_{(x,V)\in S_{ \lt t}} e^{x - m_t} V} {l_{t-1}}$$
$$\sum_{(x,V)\in S_{ \lt t}} e^{x - m_t} V = \sum_{(x,V)\in S_{ \lt t}} e^{x - m_{t-1}} e^{m_{t-1} - m} V $$

$$\begin{aligned}
O_t = \frac{\sum_{(x,V)\in S_{ \lt t}} e^{x - m_{t-1}} e^{m_{t-1} - m} V + \sum_{(x,V)\in S^{(t)}} e^{x - m_t} V} {l_t} \\
= \frac{ l_{t-1} e^{m_{t-1}-m_t} O_{t-1} + \sum_i e^{x_i^t - m_t} V }{l_t}   
\end{aligned}$$
- 最终保证：
$$O=\frac{\sum_{\text{all } j} e^{x_j} V_j}{\sum_{\text{all } j} e^{x_j}}$$  
也就是说：
> **Flash Attention 始终在做“一次全局 softmax”，只是分多步完成**
## 4.4 为什么只需要存 m 和 l
softmax 权重可以写成：
$$\alpha_j = \frac{e^{x_j - m}}{l}$$
因此：
- forward 时存下 $m,l$
- backward 时重新计算 $x_j = QK^\top$
- 就能精确恢复每一个 $\alpha_j$

这正好满足 softmax backward 的要求：
- 不存 $\alpha$
- 但能重建 $\alpha$
## 4.5 FA-2 推迟归一化到最后一步
全局状态：
- 全局最大值 $m_{t-1}$
- 全局归一化因子 $l_{t-1}$​
- 全局未归一化的累计输出 $\tilde{O}_{t-1}$
  
当处理 block t 时：
- $m_{t-1}$ 和 $l_{t-1}$ 的更新同 FA-1。
- 更新 $\tilde{O}$（FA-2 的核心差异）
  按定义展开：
  $$ \tilde{O}_t = \sum_{(x,V) \in S_{\leq t}} e^{x - m_t} V $$
  拆分为历史部分与当前块：
  $$ \tilde{O}_t = \sum_{(x,V) \in S_{< t}} e^{x - m_t} V + \sum_{(x,V) \in S^{(t)}} e^{x - m_t} V $$
  对历史部分，将基准从 $m_t$ 换回 $m_{t-1}$ 再还原：
  $$ \sum_{(x,V) \in S_{< t}} e^{x - m_t} V = \sum_{(x,V) \in S_{< t}} e^{x - m_{t-1}} \cdot e^{m_{t-1} - m_t} \cdot V = e^{m_{t-1} - m_t} \cdot \tilde{O}_{t-1} $$
  代入得：
  $$ \boxed{\tilde{O}_t = e^{m_{t-1} - m_t} \cdot \tilde{O}_{t-1} + \sum_{(x,V) \in S^{(t)}} e^{x - m_t} V} $$
- 与 FA-1 对比：

|递推公式|每步是否做除法|
|---|---|---|
|FA-1|$O_t = \dfrac{l_{t-1} e^{m_{t-1}-m_t} O_{t-1} + \sum_i e^{x_i^t - m_t} V}{l_t}$|✅ 每块除一次 $l_t$|
|FA-2|$\tilde{O}_t = e^{m_{t-1}-m_t} \tilde{O}_{t-1} + \sum_{(x,V) \in S^{(t)}} e^{x-m_t} V$|❌ 全程无除法|
- 最终归一化
  所有 block 处理完毕后，执行唯一一次除法：
  $$ O = \frac{\tilde{O}_T}{l_T} $$
  展开验证：
  $$ O = \frac{\sum_{(x,V) \in S_{\leq T}} e^{x - m_T} V}{\sum_{j \leq T} e^{x_j - m_T}} = \frac{\sum_{\text{all } j} e^{x_j} V_j}{\sum_{\text{all } j} e^{x_j}} $$
  与标准 softmax attention 完全等价。

# 五、方案对比：Flash Attention vs MemEff Attention
在工程实践中，常见的对比对象是 MemEff Attention。  
两者 **forward 结果完全一致**，但数学结构不同。
## 5.1 MemEff 的核心思路
MemEff Attention 的做法是：
- 每个 block **独立完成一次 softmax**
- 得到 block 内的输出 $O_b$
- 再用 block 权重加权平均：
$$O = \frac{\sum_b w_b O_b}{\sum_b w_b}$$  
从代数上看，这是成立的。
## 5.2 本质差异
关键区别在于：
- **Flash Attention：**
    - 只有一个全局 softmax
    - block 只是计算顺序的拆分
- **MemEff Attention：**
    - 每个 block 都有自己的 softmax
    - 全局结果是 block softmax 的组合

这在 forward 阶段没有问题，但在 backward 阶段是致命差异。

| 维度         | MemEff          | Flash            |
| ---------- | --------------- | ---------------- |
| softmax 空间 | block 局部        | 全局               |
| 归一化因子      | 每 block 一个      | 全局一个             |
| 数学形式       | “先 softmax，再平均” | “一次 softmax，分块算” |
| backward   | 需存 block 权重     | 可重算 α            |
## 5.3 为什么 backward 迫使选择 Flash
softmax backward 的梯度形式：[Attention Softmax Backward反向传播公式推导](Attention%20Softmax%20Backward反向传播公式推导.md)
$$\frac{\partial L}{\partial x_i}=
\alpha_i  
\left(\frac{\partial L}{\partial O}\right)^\top  
(V_i - O)  $$

这里的 $\alpha_i$ **必须是全局 softmax 权重**。
- Flash Attention：可以通过 $m,l$ 恢复
- MemEff Attention：block 内的 $\alpha_i$ 不等于全局 $\alpha_i$
因此：
> **Flash Attention 是被 softmax backward“逼出来的最优解”**
## 5.4 Flash Attention v1 vs v2
### 5.4.1 Flash Attention 的发展主要分为 v1 和 v2 两个阶段：

| 版本     | 特点                                                                                                                  | 优点                                               | 缺点 / 限制                          |
| ------ | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | -------------------------------- |
| **v1** | - 基于 **单次 forward pass block-wise 计算 softmax** <br>- 存储中间 $m,l,O$ 用于 backward <br>- 支持 **半精度（fp16）/ bf16**          | - 内存使用显著降低 <br>- 可直接重算 softmax，减少 backward 的显存压力 | - 对某些序列长度极长时仍可能 hit memory limit |
| **v2** | - 引入 **更优化的流水线和 tile 计算** ，存储 $\tilde{O}$，将归一化延迟到最后<br>- 支持 **更长序列 & 更灵活的 mixed precision** <br>- 减少重复计算，提高 GPU 利用率 | - 最大化显存效率 - forward/backward 更统一 <br>- 对大模型训练更友好 | - 实现更复杂，需要更严格的 kernel 调度         |
### 5.4.2 Flash Attention v1 → v2 优化对比
| 优化方向                     | v1 实现                                                   | v2 改进                                               | 优势 / 意义                             |
| ------------------------ | ------------------------------------------------------- | --------------------------------------------------- | ----------------------------------- |
| **计算顺序**                 | 按 block 计算 softmax，每个 block 独立 forward，backward 时需要重算 α | 引入 tile/pipeline 机制，将 block 内计算分段、流水线执行             | 提升 GPU 并行度，减少 idle 时间，提高 throughput |
| **显存占用**                 | 需要存储每个 block 的中间 $m,l$ 值用于 backward                     | 存储 $\tilde{O}$，将归一化延迟到最后，减少中间缓存，通过更精细的 tile 控制和重算策略 | 降低显存峰值，支持更长序列                       |
| **序列长度支持**               | 支持到一定长度，超长序列可能 OOM                                      | 支持超长序列（可动态切分 tile）                                  | 适合大模型训练，Transformer 超长上下文           |
| **精度 / mixed precision** | 支持 fp16/bf16，但部分重算可能重复累积                                | 更灵活 mixed precision，减少重复 cast，优化数值稳定性               | 提升训练稳定性和性能                          |
| **backward 机制**          | 重算全局 softmax，存储 block 权重                                | 更智能的重算策略，避免多次重复计算                                   | backward 显存消耗更低，速度更快                |
| **GPU 利用率**              | 部分 kernel 受 block size 限制，利用率不均                         | 优化 kernel 调度，流水线 tile，使 GPU 负载均匀                    | 更高吞吐量，训练大模型更高效                      |
| **实现复杂度**                | 相对简单，v1 kernel 容易理解                                     | kernel 设计更复杂，需要严格的 block/tile 调度                    | 复杂度增加，但收益明显：显存、速度、序列长度支持            |
#### **核心总结：**
- **v2 核心在于“tile + pipeline”**，把原本 v1 的 block 拆得更细，同时优化 forward/backward 的调度。
- **显存优化 + 超长序列支持** 是 v2 最大亮点。
- **GPU throughput 提升**：减少 idle kernel，充分利用计算资源。
- **数值精度控制更好**：混合精度更稳健，减少重算误差累积。
- 省去每块的除法本身算术收益有限，真正的性能提升来自**减少 SRAM 写回**和**支持 Warp 级并行**。
# 六、整体收束
把所有线索连起来，可以得到一条非常清晰的因果链：
- attention 的定义要求全局 softmax
- softmax backward 要求知道每个 (\alpha_i)
- 显存限制不允许保存 (\alpha) 或 (QK^\top)
- 只能 forward 存 (m,l)，backward 重算
- 为了能在线维护 (m,l)，必须做全局在线 softmax
- 这直接导向 Flash Attention 的设计

**最终结论是：**
> Flash Attention 不是一个“更快的 trick”，  
> 而是在数学正确性、反向传播和显存约束三者之间，  
> **唯一自洽的工程实现方式**。
