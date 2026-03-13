## 1. 背景与动机

### 1.1 序列数据的特殊性

图像、表格等数据的样本之间相互独立，MLP / CNN 的前向传播在每个样本上独立执行。序列数据（文本、语音、时间序列）的核心特征是**样本内部存在时序依赖**——"我爱北京天安门"中"天安门"的语义依赖于前面的"北京"，不能独立处理每个词。

标准的前馈网络无法处理此类依赖关系：

- 输入维度固定，无法处理可变长度序列
- 不同时刻的输入之间没有信息传递机制
- 对序列位置没有概念，无法区分"猫吃鱼"和"鱼吃猫"

### 1.2 RNN 的核心思路

引入**隐藏状态** $h_t$，作为网络的"记忆"，在时间步之间传递信息。每个时间步的计算同时依赖当前输入 $x_t$ 和上一时刻的隐藏状态 $h_{t-1}$，形成循环连接（Recurrence）。

权重在所有时间步**共享**——这与 CNN 的权重共享思路类似，使得网络可以处理任意长度的序列，而参数量固定。

### 1.3 历史地位

RNN 由 Rumelhart et al.（1986）提出基本框架，Elman（1990）提出经典结构（Elman Network）。1990–2012 年间，RNN 及其变体 LSTM（详见 [[LSTM]]）、GRU（详见 [[GRU]]）是序列建模的绝对主流，驱动了机器翻译、语音识别、语言模型的早期突破。Transformer（2017）出现后，RNN 在 NLP 主流任务中被取代，但在低延迟流式推理、边缘设备等场景仍有应用。

---

## 2. 基本结构

### 2.1 前向传播公式

> 【图示占位】：RNN 时间步展开图，展示 $x_1, x_2, \ldots, x_T$ 依次输入，$h_0 \to h_1 \to \cdots \to h_T$ 的状态传递，以及每个时间步的输出 $y_t$

隐藏状态更新：

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b_h)$$

输出（若每步都需要输出）：

$$y_t = W_y h_t + b_y$$

符号定义：

|符号|含义|维度|
|---|---|---|
|$x_t$|$t$ 时刻的输入|$\mathbb{R}^{d_{\text{in}}}$|
|$h_t$|$t$ 时刻的隐藏状态|$\mathbb{R}^{d_h}$|
|$h_0$|初始隐藏状态|$\mathbb{R}^{d_h}$（通常初始化为零向量）|
|$W_x$|输入到隐藏的权重矩阵|$\mathbb{R}^{d_h \times d_{\text{in}}}$|
|$W_h$|隐藏到隐藏的权重矩阵|$\mathbb{R}^{d_h \times d_h}$|
|$W_y$|隐藏到输出的权重矩阵|$\mathbb{R}^{d_{\text{out}} \times d_h}$|
|$b_h, b_y$|偏置|$\mathbb{R}^{d_h}$，$\mathbb{R}^{d_{\text{out}}}$|

### 2.2 参数量

$$\text{参数量} = d_h \times d_{\text{in}} + d_h \times d_h + d_h + d_{\text{out}} \times d_h + d_{\text{out}}$$

关键特性：参数量与序列长度 $T$ **无关**，所有时间步共享同一组 $W_x$、$W_h$、$W_y$。

### 2.3 三种常见的输入输出结构

|类型|描述|典型应用|
|---|---|---|
|Many-to-One|序列 $\to$ 单个输出（取最后时刻 $h_T$）|文本分类、情感分析|
|One-to-Many|单个输入 $\to$ 序列输出|图像描述（Image Captioning）|
|Many-to-Many|序列 $\to$ 等长序列|序列标注（NER、POS Tagging）|
|Encoder-Decoder|序列 $\to$ 压缩向量 $\to$ 序列|机器翻译|

---

## 3. 反向传播：BPTT（Backpropagation Through Time）

### 3.1 原理

RNN 按时间步展开后等价于一个极深的前馈网络（深度为序列长度 $T$）。反向传播需要沿时间步回传，称为 Backpropagation Through Time（BPTT）。

损失 $\mathcal{L} = \sum_{t=1}^T \mathcal{L}_t$，对 $W_h$ 的梯度需对每个时间步求和：

$$\frac{\partial \mathcal{L}}{\partial W_h} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial W_h}$$

其中每个 $\partial \mathcal{L}_t / \partial W_h$ 需要将梯度从时刻 $t$ 回传至时刻 $1$：

$$\frac{\partial \mathcal{L}_t}{\partial W_h} = \sum_{k=1}^{t} \frac{\partial \mathcal{L}_t}{\partial h_t} \cdot \left(\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}\right) \cdot \frac{\partial h_k}{\partial W_h}$$

### 3.2 梯度消失/爆炸的推导

每个时间步的局部梯度：

$$\frac{\partial h_t}{\partial h_{t-1}} = W_h \cdot \text{diag}(\tanh'(h_{t-1}))$$

$T$ 步回传的梯度链：

$$\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}} = \prod_{j=k+1}^{t} W_h \cdot \text{diag}(\tanh'(h_{j-1}))$$

令 $\rho = \rho(W_h)$（谱范数），$\gamma = \max |\tanh'(\cdot)| \leq 1$：

- 若 $\rho \cdot \gamma < 1$：梯度指数消失，$(\rho \gamma)^{t-k} \to 0$
- 若 $\rho \cdot \gamma > 1$：梯度指数爆炸，$(\rho \gamma)^{t-k} \to \infty$

实践中 $\tanh' \leq 1$，因此只有当 $\rho(W_h)$ 显著大于 1 时才爆炸；消失则几乎是必然的（$\rho(W_h) < 1$ 或 Tanh 饱和均可触发）。结果是 Vanilla RNN **有效记忆长度通常不超过 10–20 步**。

### 3.3 截断 BPTT（Truncated BPTT）

完整 BPTT 对长序列计算量和内存消耗极大（需存储所有时间步的激活值）。实践中通常将序列切分为长度为 $k$ 的片段，在每个片段内做完整 BPTT，片段间隐藏状态向前传递但梯度截断（不回传跨片段边界）。

代价：跨片段的长程依赖无法学习，但大幅降低计算和内存开销。

---

## 4. 优缺点

**优点**：

- 结构简单，参数量少（仅三组权重矩阵）
- 理论上能处理任意长度序列，参数量与序列长度无关
- 实现和调试容易，框架支持成熟
- 推理时内存占用低（只需保存当前隐藏状态 $h_t$，不需要存储全序列）
- 天然支持流式（Online）推理：每到来一个新 token 即可更新 $h_t$，无需等待完整序列

**缺点**：

- 梯度消失使有效记忆长度严重受限（通常 $\leq 20$ 步）
- **不可并行**：时间步之间有严格的串行依赖（$h_t$ 依赖 $h_{t-1}$），训练速度远慢于 Transformer
- 完整 BPTT 对长序列内存消耗 $O(T)$（需存储所有时间步激活值）
- 在 NLP 主流任务上效果已被 Transformer 全面超越

---

## 5. 与 LSTM / GRU 的关系

Vanilla RNN 的梯度消失问题使其无法学习长程依赖，LSTM 和 GRU 通过引入门控机制系统性地解决了这一问题：

|模型|解决机制|参数量（相对）|有效记忆长度|
|---|---|---|---|
|Vanilla RNN|无|$1\times$|$\leq 20$ 步|
|LSTM|细胞状态 $c_t$ + 三个门|$\approx 4\times$|$100$ 步以上|
|GRU|两个门（无独立细胞状态）|$\approx 3\times$|与 LSTM 相当|

详细内容见 [[LSTM]] 和 [[GRU]]。

---

## 6. 相关笔记

- [[梯度稳定性]]
- [[LSTM]]
- [[GRU]]
- [[CNN]]
