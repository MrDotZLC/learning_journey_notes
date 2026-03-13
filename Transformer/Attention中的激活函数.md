## 1. 背景

激活函数是神经网络引入非线性的核心机制，同时直接决定反向传播中梯度的传播质量。早期 Sigmoid / Tanh 的饱和特性是梯度消失的主要来源（详见 [[梯度稳定性]]）；ReLU 的提出将深层网络训练变为可能，但引入了死亡神经元问题；GELU 和 SiLU 在平滑性与计算效率之间取得更好的平衡，成为现代 LLM 的主流选择。

Transformer / LLM 中激活函数的使用位置：FFN（Feed-Forward Network）层，即每个 Transformer block 中的两层 MLP（或 SwiGLU 三路结构）。Self-Attention 本身不含激活函数（Softmax 用于归一化 Attention Score，不属于激活函数语义）。

---

## 2. Sigmoid 与 Tanh

### 2.1 Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}, \quad \sigma'(x) = \sigma(x)(1-\sigma(x)) \in (0,\ 0.25]$$
**缺陷**：

- 饱和区（$|x|$ 较大时）导数趋近于 $0$，链式乘积导致梯度消失
- 输出范围 $(0, 1)$，非零均值，导致下游权重梯度方向受限（**zigzag 问题**：Sigmoid 输出恒正，下一层权重的梯度正负方向只能统一变化，优化路径呈锯齿形，收敛慢）
- 计算含指数运算，开销中等

**历史地位**：早期 MLP 和 Logistic 回归的标配，现主要用于二分类输出层和 LSTM / GRU 的门控（门控语义要求输出范围 $(0,1)$）。

### 2.2 Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}, \quad \tanh'(x) = 1 - \tanh^2(x) \in (0,\ 1]$$

相比 Sigmoid 的改进：输出零均值（范围 $(-1, 1)$），消除 zigzag 问题。但饱和区导数趋零问题依然存在，深层网络中仍会梯度消失。

**历史地位**：历史 RNN 的标准激活函数，现在仍用于 LSTM 细胞状态的输出变换（$\tanh(c_t)$，将细胞状态压缩至 $(-1, 1)$）。

---

## 3. ReLU

$$\text{ReLU}(x) = \max(0,\ x), \quad \text{ReLU}'(x) = \begin{cases} 1 & x > 0 \ 0 & x \leq 0 \end{cases}$$

### 3.1 优点

- 正区间梯度恒为 $1$，彻底解决链式乘积缩小导致的梯度消失
- 计算极简（仅一次比较操作），GPU/CPU 上均高效
- 引入稀疏激活（约一半神经元输出为零），有一定正则化效果
- 推动了 AlexNet、VGG、ResNet 等深层 CNN 的成功

### 3.2 缺点 — 死亡 ReLU（Dying ReLU）

负区间导数为 $0$。若某神经元输入在训练中持续为负，则该神经元输出恒为 $0$，梯度恒为 $0$，**永久失活**，无法被任何后续梯度复活。

判定条件：对所有训练样本 $x$，若 $Wx + b < 0$ 恒成立，则该神经元死亡。

触发原因：初始化权重过大导致负偏，或学习率过大使权重在一次更新后跳至极端负值。大模型中死亡 ReLU 导致的神经元稀疏性可达 $30%$–$50%$，这是 LLM 转向 SiLU / GELU 的主要原因之一。

**历史地位**：历史 CNN（AlexNet、VGG、ResNet 早期版本）的标准激活函数，现代 LLM 已基本不用。

---

## 4. GELU（Gaussian Error Linear Unit）

### 4.1 背景

Hendrycks & Gimpel（2016）提出，设计动机是将 Dropout 的随机门控思想确定化：输入越大，通过的概率越高。BERT 和 GPT 系列的激活函数选择。

### 4.2 公式推导

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}!\left(\frac{x}{\sqrt{2}}\right)\right]$$

其中 $\Phi(x)$ 为标准正态分布 CDF，$\text{erf}$ 为误差函数。直觉：$x$ 被保留的概率为 $\Phi(x)$，即越大的输入越有可能完整通过，负区间有小幅抑制而非硬截断。

实用近似（避免 $\text{erf}$ 计算开销）：

$$\text{GELU}(x) \approx 0.5x!\left(1 + \tanh!\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)$$

导数（精确形式）：

$$\text{GELU}'(x) = \Phi(x) + x \cdot \phi(x), \quad \phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$$

### 4.3 优缺点

**优点**：

- 负区间梯度不完全为零，平滑过渡，避免死亡神经元
- 输出均值接近零，统计特性好
- 在 BERT、GPT 等模型上验证效果优于 ReLU

**缺点**：

- 计算开销高于 ReLU 和 SiLU（含 $\text{erf}$ 或 $\tanh$ 近似）
- 不天然支持门控结构，与 GLU 结合不如 SiLU 简洁

---

## 5. SiLU（Sigmoid Linear Unit / Swish）

### 5.1 背景

Ramachandran et al.（2017）通过神经架构搜索发现，又称 Swish。LLaMA、Mistral、Qwen 等主流 LLM 均采用 SiLU 配合 SwiGLU 结构，是当前 LLM FFN 层的事实标准激活函数。

### 5.2 公式推导

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

导数：

$$\text{SiLU}'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) = \sigma(x)\left(1 + x(1 - \sigma(x))\right)$$
![](assets/Pasted%20image%2020260120043348.png)
### 5.3 与 ReLU / GELU 的对比

**相比 ReLU**：

- 负半轴仍有小幅负值梯度（非单调），保留部分负区间信息，完全避免死亡神经元
- 连续可导，无硬截断，对大规模参数训练更稳定
- 计算量略高（需一次 Sigmoid）

**相比 GELU**：

- 计算量更低（无需 $\text{erf}$，仅需一次 Sigmoid）
- 本身结构为 `x * sigmoid(x)`，与 GLU 的门控机制**天然契合**，直接组合成 SwiGLU

### 5.4 与 RMSNorm 的协同作用

LLaMA 使用 RMSNorm（不减均值，保留幅值信息），SiLU 输出同样保留尺度信息（不像 Tanh 压缩至 $\pm 1$）。二者在数值分布上是协同设计的，避免激活值经归一化后信息损失。

### 5.5 优缺点

**优点**：

- 非单调，负区间保留信息，完全避免死亡神经元
- 全域梯度连续且非零，训练稳定
- 计算量低于 GELU，仅需一次 Sigmoid
- 与 GLU 门控结构天然契合，组合成 SwiGLU 后参数效率高
- 实现简单，易于 Kernel Fusion，GPU / CPU 上均高效

**缺点**：

- 相比 ReLU 计算量略高（需一次 Sigmoid）
- 非单调性在极少数场景下可能引入轻微训练不稳定

---

## 6. SwiGLU 门控结构

### 6.1 背景

Noam Shazeer（2020）提出，将 SiLU 与 GLU（Gated Linear Unit）结合。LLaMA 将其作为 FFN 的标准结构，相比传统 ReLU FFN 表达能力更强、参数效率更高。

### 6.2 公式

经典 Transformer FFN（ReLU 版本）：

$$\text{FFN}(x) = W_2\left(\text{ReLU}(W_1 x)\right)$$

LLaMA 使用的 SwiGLU 版本：

$$\text{FFN}(x) = W_2\left(\text{SiLU}(W_1 x) \odot (W_3 x)\right)$$

### 6.3 代码示意

```cpp
auto gate = silu(linear_w1(x));   // W1·x → SiLU → 控制"开多少门"
auto up   = linear_w3(x);         // W3·x → 提供"要通过的信号"
auto out  = linear_w2(gate * up); // 逐元素相乘后投影
```

### 6.4 门控语义

- $W_1 x$：决定"开多少门"（gate），经 SiLU 后值域连续，而非 ReLU 的硬开/关
- $W_3 x$：提供"要通过的信号"（value）
- 逐元素相乘：让模型动态决定每个通道的信息通量，比对所有通道一视同仁表达能力更强

### 6.5 参数效率

SwiGLU 通常设置 `hidden_dim ≈ 8/3 * d_model`（LLaMA 的实际配置），比传统 FFN 的 `4 * d_model` 参数量少约 $1/3$，但表达能力更强，是 LLaMA 在参数规模受限下追求高质量的关键设计。

### 6.6 推理实现优势

SiLU 的实现极为简单：`silu(x) = x / (1 + exp(-x))`，易于 Fused Kernel（与前后线性层合并），易于 SIMD 向量化，GPU / CPU 上均高效。

---

## 7. 激活函数全面对比

|特性|Sigmoid|Tanh|ReLU|GELU|SiLU|
|---|---|---|---|---|---|
|输出范围|$(0, 1)$|$(-1, 1)$|$[0, +\infty)$|$\approx(-0.17, +\infty)$|$\approx(-0.28, +\infty)$|
|正区间导数|$\leq 0.25$|$\leq 1$|恒为 $1$|$> 0$|$> 0$|
|负区间导数|$\approx 0$（饱和）|$\approx 0$（饱和）|恒为 $0$|小正值|小正/负值|
|死亡神经元|否（但饱和）|否（但饱和）|**是**|否|否|
|零均值输出|否|是|否|近似是|近似是|
|计算开销|中|中|低|高（含 erf）|中|
|门控兼容性|一般|一般|差（硬截断）|一般|**优秀**（天然契合）|
|主要应用|LSTM 门控、输出层|LSTM 输出变换|历史 CNN|BERT、GPT|LLaMA、Qwen、Mistral|
