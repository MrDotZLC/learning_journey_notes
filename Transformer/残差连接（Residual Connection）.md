## 1. 背景与动机

### 1.1 深度退化问题（Degradation Problem）

2015 年 He et al. 发现：单纯增加网络层数，在 CIFAR-10 / ImageNet 上**训练误差**反而升高，并非过拟合导致（过拟合的特征是训练误差低、测试误差高）。

|网络|训练误差|测试误差|
|---|---|---|
|20 层 Plain Network|较低|较低|
|56 层 Plain Network|**更高**|更高|
|56 层 ResNet|更低|更低|

根本原因：若增加的层能学成恒等映射，深层网络至少不会比浅层差。但实验表明，**直接优化恒等映射极为困难**——网络倾向于在新增层中引入噪声，而非干净地学成 $F(x) = x$。

### 1.2 残差学习的核心思路

残差连接将优化目标从"学恒等映射 $H(x) = x$"改为"学零映射 $F(x) = H(x) - x = 0$"。当 $F(x) \to 0$ 时，网络退化为恒等传播，是安全的下界。零映射在优化上比恒等映射容易得多：权重趋于零即可，而恒等映射需要权重精确配合激活函数构造出 $y = x$。

---

## 2. 基本公式

$$y = x + F(x,\ {W_i})$$

- $x$：输入，经由 shortcut path 直接加到输出
- $F(x, {W_i})$：残差函数，网络只需学习对输入的**增量修正**
- 当 $F \to 0$ 时，$y \to x$（退化为恒等映射，训练安全）

### 2.1 维度不匹配时的处理

当 $x$ 与 $F(x)$ 维度不一致时，引入线性投影：

$$y = W_s x + F(x,\ {W_i})$$

$W_s$ 为 $1 \times 1$ 卷积或线性变换，仅对齐维度，不引入非线性。

### 2.2 Bottleneck 结构

ResNet 中实际大量使用的三层 Bottleneck 结构：

$$\text{Block}: \text{Conv}_{1\times1}^{\text{reduce}} \to \text{BN+ReLU} \to \text{Conv}_{3\times3} \to \text{BN+ReLU} \to \text{Conv}_{1\times1}^{\text{expand}} \to \text{BN}$$

设计逻辑：第一个 $1\times1$ 卷积降维减少 $3\times3$ 卷积的计算量，$3\times3$ 卷积在低维空间执行空间特征提取，第二个 $1\times1$ 卷积升维恢复通道数。

计算量对比（输入输出均为 $256$ 通道）：

|结构|FLOPs|
|---|---|
|两层 $3\times3$ 卷积|$\approx 1.18\text{M}$|
|Bottleneck（64 维瓶颈）|$\approx 0.07\text{M}$（约 $1/17$）|

**与 LLM FFN 的联系**：LLaMA SwiGLU FFN 中先升维（$d \to 8d/3$）再降维（$8d/3 \to d$），与 Bottleneck 的降维-操作-升维逻辑相同，只是维度方向相反。

---

## 3. 梯度流动分析

### 3.1 单个残差块

损失 $\mathcal{L}$ 对输入 $x$ 的梯度：

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left(1 + \frac{\partial F}{\partial x}\right)$$

括号内的 $1$ 保证了一条**不经过任何权重矩阵**的梯度直接路径。即使 $\partial F / \partial x \approx 0$，梯度仍能直达 $x$，彻底阻断梯度消失。

### 3.2 多个残差块的累积效应

设网络由 $n$ 个残差块堆叠，$x_n = x_1 + \sum_{k=1}^{n-1} F_k(x_k)$，则：

$$\frac{\partial \mathcal{L}}{\partial x_1} = \frac{\partial \mathcal{L}}{\partial x_n} \cdot \left(1 + \sum_{k=1}^{n-1} \frac{\partial F_k}{\partial x_1}\right)$$

括号内的 $1$ 在任意深度下始终存在，这是 1000 层 ResNet 仍能训练的根本原因。

### 3.3 残差网络的集成解释（Veit et al., 2016）

具有 $n$ 个残差块的网络等价于 $2^n$ 条不同深度路径的集成。展开 $x_n$：

$$x_n = x_1 + \sum_{k} F_k + \sum_{j<k} F_j \circ F_k + \cdots$$

每一项对应一条从 $x_1$ 到 $x_n$ 的路径，路径长度从 $0$ 到 $n$。推论：

- 删除单个残差块，仅影响 $2^{n-1}$ 条路径，网络性能平滑退化
- 大多数梯度信号由**短路径**贡献，网络实际上更像浅层集成
- 这解释了为什么 ResNet 比同深度 Plain Network 更容易优化

---

## 4. 与 Highway Networks 的关系

Highway Networks（2015，早于 ResNet）是残差连接的前身：

$$y = H(x) \cdot T(x) + x \cdot C(x), \quad C(x) = 1 - T(x)$$

|特性|Highway Networks|ResNet|
|---|---|---|
|Transform Gate|可学习（Sigmoid 输出）|恒为 $1$|
|Carry Gate|$1 - T(x)$，可学习|恒为 $1$|
|额外参数|需要 $T(x)$ 的参数|无|
|优化难度|较高（门控引入额外非线性）|较低|
|实用性|较少使用|成为主流标准|

ResNet 可视为 Highway Networks 将门控固定为全开的特例。固定全开反而更稳定，原因：可学习门控初始化时若 $T(x) \approx 0$，则信息无法流通，训练初期陷入局部最优。

---

## 5. Pre-LN vs Post-LN

残差连接在 Transformer 中与归一化层结合，存在两种变体。

### 5.1 Post-LN（原始 Transformer / BERT）

$$y = \text{LN}(x + F(x))$$

归一化层横跨在 shortcut 路径上，LN 的缩放操作削弱恒等路径的梯度量级。LN 对梯度的 Jacobian 矩阵在训练初期（$\sigma$ 较大时）显著缩小梯度幅值，导致训练初期不稳定，通常需要 Warmup 才能收敛。

### 5.2 Pre-LN（GPT-2 / LLaMA / Mistral）

$$y = x + F(\text{LN}(x))$$

恒等路径 $x$ 完全绕过 LN，梯度直接流过。$F(\text{LN}(x))$ 支路的梯度经过 LN，但 shortcut 梯度不受影响，训练更稳定，可大幅缩短或完全省略 Warmup 阶段。

> 【图示占位】：Post-LN vs Pre-LN 信号流对比图，展示两种结构中 shortcut 路径是否穿过 LN 层

### 5.3 Pre-LN 的潜在问题与 DeepNorm

Liu et al.（2020）发现 Pre-LN 在极深网络（$> 100$ 层）中，后层梯度量级远大于前层，导致前层更新过慢。

**DeepNorm**（Microsoft, 2022）通过对残差缩放解决此问题：

$$y = \text{LN}(\alpha \cdot x + F(x))$$

其中 $\alpha = (2N)^{1/4}$（$N$ 为层数），同时对初始化缩放（权重乘以 $\beta = (8N)^{-1/4}$），使各层梯度量级趋于一致，在 $1000$ 层以上网络中实现稳定训练。

### 5.4 主流模型选型

|架构|归一化位置|归一化方法|
|---|---|---|
|Transformer（原始）|Post-LN|LayerNorm|
|BERT|Post-LN|LayerNorm|
|GPT-2|Pre-LN|LayerNorm|
|LLaMA / Mistral|Pre-LN|RMSNorm|
|Qwen|Pre-LN|RMSNorm|
|DeepNet（1000 层）|DeepNorm|LayerNorm|

---

## 6. 优缺点

**优点**：

- 彻底解决深度退化问题，使极深网络（1000 层以上）可训练
- 梯度有恒等捷径路径，系统性缓解梯度消失
- 网络只需学习增量修正，优化难度大幅降低
- 推理时 element-wise add 可与前序算子 Kernel Fusion，减少 HBM 读写
- 无额外参数开销（shortcut 路径无可学习参数）

**缺点**：

- shortcut 路径与主路径数值范围不同，Per-tensor 量化时需独立校准量化参数，否则精度损失显著
- 残差结构要求输入输出维度匹配，维度不一致时需引入 $W_s$，增加少量参数
- KV Cache 中各层隐状态通过残差累积，内存占用与层数成正比

---

## 7. 在推理优化中的意义

**Kernel Fusion**：残差加法（element-wise add）是 memory-bound 算子，通常与前一个算子（RMSNorm、激活函数）融合为单个 CUDA Kernel，减少 HBM 读写次数。

**量化**：shortcut 路径 $x$ 与主路径 $F(x)$ 的数值范围通常不同。Per-tensor 量化时需分别校准量化参数，加法前在 FP16 或 INT32 域执行；Per-channel 量化可细化粒度缓解此问题。

**KV Cache**：Transformer decoder 中每层隐状态通过残差累积 $h^{(l)} = h^{(l-1)} + F^{(l)}(h^{(l-1)})$，KV Cache 存储的 $K$、$V$ 矩阵均由当前层隐状态线性投影得到，残差结构保证历史 token 信息在各层的稳定传递。
