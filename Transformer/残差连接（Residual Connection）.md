## 1. 背景与动机

### 1.1 深度退化问题（Degradation Problem）

2015 年 He et al. 发现：单纯增加网络层数，在 CIFAR-10 / ImageNet 上**训练误差**反而升高，并非过拟合导致（过拟合的特征是训练误差低、测试误差高）。

|网络深度|训练误差|测试误差|
|---|---|---|
|20 层（Plain Network）|较低|较低|
|56 层（Plain Network）|**更高**|更高|
|56 层（ResNet）|更低|更低|

根本原因：若增加的层能学成恒等映射，深层网络至少不会比浅层差。但实验表明，**直接优化恒等映射极为困难**——网络倾向于在新增层中引入噪声，而非干净地学成 $F(x) = x$。

### 1.2 残差学习的核心思路

残差连接将优化目标从"学恒等映射 $H(x) = x$"改为"学零映射 $F(x) = H(x) - x = 0$"。

当 $F(x) \to 0$ 时，网络退化为恒等传播，是安全的下界。零映射在优化上比恒等映射容易得多：权重趋于零即可，而恒等映射需要权重精确配合激活函数构造出 $y = x$。

---

## 2. 基本公式

$$y = x + F(x,\ {W_i})$$

- $x$：输入，经由 shortcut path 直接加到输出
- $F(x, {W_i})$：残差函数，网络只需学习对输入的**增量修正**
- 当 $F \to 0$ 时，$y \to x$（退化为恒等映射，训练安全）

### 2.1 维度不匹配时的处理

当 $x$ 与 $F(x)$ 维度不一致（通道数变化或下采样）时，引入线性投影：

$$y = W_s x + F(x,\ {W_i})$$

$W_s$ 为 $1 \times 1$ 卷积或线性变换，仅对齐维度，不引入非线性。实验表明，$W_s$ 使用恒等填零（zero-padding）与线性投影效果相近，线性投影略优但参数量更多。

### 2.2 Bottleneck 结构

ResNet 中实际大量使用的不是简单两层残差块，而是三层 **Bottleneck** 结构：

$$\text{Block}: \text{Conv}_{1\times1} \to \text{BN+ReLU} \to \text{Conv}_{3\times3} \to \text{BN+ReLU} \to \text{Conv}_{1\times1} \to \text{BN}$$

$$y = x + \text{Conv}_{1\times1}^{\text{expand}}!\left(\text{Conv}_{3\times3}!\left(\text{Conv}_{1\times1}^{\text{reduce}}(x)\right)\right)$$

**设计逻辑**：

- 第一个 $1\times1$ 卷积：降维（如 $256 \to 64$），减少 $3\times3$ 卷积的计算量
- $3\times3$ 卷积：在低维空间执行空间特征提取
- 第二个 $1\times1$ 卷积：升维（如 $64 \to 256$），恢复通道数

**计算量对比**（输入 $256$ 通道，输出 $256$ 通道）：

|结构|计算量（FLOPs）|
|---|---|
|两层 $3\times3$ 卷积|$2 \times 3^2 \times 256^2 \approx 1.18\text{M}$|
|Bottleneck（64 维瓶颈）|$256{\times}64 + 3^2{\times}64^2 + 64{\times}256 \approx 0.07\text{M}$|

计算量降至约 $1/17$，使得 ResNet-50/101/152 在可接受计算量下实现极深网络。

**与 LLM FFN 的联系**：LLaMA FFN 中 SwiGLU 结构同样是先升维（$d \to 4d$）再降维（$4d \to d$），与 Bottleneck 的降维-操作-升维逻辑完全类似，只是维度方向相反。

---

## 3. 梯度流动分析

### 3.1 单个残差块

损失 $\mathcal{L}$ 对输入 $x$ 的梯度：

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left(1 + \frac{\partial F}{\partial x}\right)$$

括号内的 $1$ 是恒等项，保证了一条**不经过任何权重矩阵**的梯度直接路径。即使 $\partial F / \partial x \approx 0$，梯度仍能直达 $x$，彻底阻断梯度消失。

### 3.2 多个残差块的累积效应

设网络由 $n$ 个残差块堆叠，任意块 $k$ 的输出 $x_k$，最终输出 $x_n$ 满足：

$$x_n = x_1 + \sum_{k=1}^{n-1} F_k(x_k)$$

损失对 $x_1$ 的梯度：

$$\frac{\partial \mathcal{L}}{\partial x_1} = \frac{\partial \mathcal{L}}{\partial x_n} \cdot \left(1 + \sum_{k=1}^{n-1} \frac{\partial F_k}{\partial x_1}\right)$$

括号内的 $1$ 在任意深度下始终存在，这是 1000 层 ResNet 仍能训练的根本原因。

### 3.3 残差网络的集成解释（Veit et al., 2016）

Veit et al. 证明：具有 $n$ 个残差块的网络等价于 $2^n$ 条不同深度路径的**集成**。

展开 $x_n$：

$$x_n = x_1 + \sum_{k} F_k + \sum_{j<k} F_j \circ F_k + \cdots$$

每一项对应一条从 $x_1$ 到 $x_n$ 的路径，路径长度从 $0$（直接跳过所有块）到 $n$（经过所有块）不等。

**推论**：

- 删除单个残差块，仅影响 $2^{n-1}$ 条路径，网络性能平滑退化（而非断崖式崩溃）
- 大多数梯度信号由**短路径**贡献（长路径梯度在连乘后很小），网络实际上更像浅层集成而非真正的深层网络
- 这解释了为什么 ResNet 比同深度 Plain Network 更容易优化：有效路径深度的期望值远低于标称深度

---

## 4. 与 Highway Networks 的关系

Highway Networks（2015，早于 ResNet）是残差连接的前身，引入可学习门控：

$$y = H(x) \cdot T(x) + x \cdot C(x), \quad C(x) = 1 - T(x)$$

- $H(x)$：非线性变换
- $T(x)$：Transform Gate，Sigmoid 输出，控制使用多少 $H(x)$
- $C(x)$：Carry Gate，控制直接保留多少 $x$

|特性|Highway Networks|ResNet|
|---|---|---|
|Transform Gate|可学习（Sigmoid 输出）|恒为 $1$|
|Carry Gate|$1 - T(x)$，可学习|恒为 $1$|
|额外参数|需要 $T(x)$ 的参数|无|
|优化难度|较高（门控引入额外非线性）|较低|
|实用性|较少使用|成为主流标准|

ResNet 可视为 Highway Networks 将门控固定为全开的特例。固定全开反而更稳定的原因：可学习门控本身引入了额外的优化困难，初始化时若 $T(x) \approx 0$，则信息无法流通，训练初期陷入局部最优。

---

## 5. Pre-LN vs Post-LN

残差连接在 Transformer 中与归一化层结合，存在两种主流变体。

### 5.1 Post-LN（原始 Transformer / BERT）

$$y = \text{LN}(x + F(x))$$

梯度路径：$\mathcal{L} \to y \xrightarrow{\text{LN}} (x + F(x)) \to x$

归一化层横跨在 shortcut 路径上。LN 的缩放操作会**改变恒等路径的梯度量级**，具体地，LN 对梯度的雅可比矩阵为：

$$\frac{\partial \text{LN}(z)}{\partial z} = \frac{1}{\sigma}!\left(I - \frac{1}{d}\mathbf{1}\mathbf{1}^\top - \frac{(z-\mu)(z-\mu)^\top}{d\sigma^2}\right) \cdot \text{diag}(\gamma)$$

该矩阵在训练初期（$\gamma \approx 1$，$\sigma$ 较大）会显著缩小梯度幅值，导致训练初期不稳定，通常需要 Warmup 才能收敛。

> 【图示占位】：Post-LN vs Pre-LN 信号流对比图，展示两种结构中 shortcut 路径是否穿过 LN 层，以及梯度回传路径的差异

### 5.2 Pre-LN（GPT-2 / LLaMA / Mistral）

$$y = x + F(\text{LN}(x))$$

梯度路径：$\mathcal{L} \to y \to x$（shortcut 完全绕过 LN）

恒等路径 $x$ 不经过任何归一化层，梯度直接流过。$F(\text{LN}(x))$ 支路的梯度经过 LN，但 shortcut 的梯度不受影响，因此训练更稳定，可大幅缩短或完全省略 Warmup 阶段。

### 5.3 Pre-LN 的潜在问题与 DeepNorm

Pre-LN 并非完美。Liu et al.（2020）发现 Pre-LN 在**极深网络**（$> 100$ 层）中，后层的梯度量级远大于前层，导致前层更新过慢。

**DeepNorm**（Microsoft, 2022）通过对残差进行缩放解决此问题：

$$y = \text{LN}(\alpha \cdot x + F(x))$$

其中 $\alpha > 1$ 是超参数（如 $\alpha = (2N)^{1/4}$，$N$ 为层数），同时对初始化进行缩放（权重乘以 $\beta = (8N)^{-1/4}$）。

**效果**：使各层梯度量级趋于一致，在 $1000$ 层以上的网络中实现稳定训练，是目前已知在极深 Transformer 上表现最好的归一化方案之一。

### 5.4 主流模型选型

|架构|归一化位置|归一化方法|
|---|---|---|
|Transformer（原始）|Post-LN|LayerNorm|
|BERT|Post-LN|LayerNorm|
|GPT-2|Pre-LN|LayerNorm|
|LLaMA / Mistral|Pre-LN|RMSNorm|
|Qwen|Pre-LN|RMSNorm|
|DeepNet（1000层）|DeepNorm|LayerNorm|

现代 LLM 几乎全部采用 Pre-LN + RMSNorm 的组合。

---

## 6. 在推理优化中的意义

### 6.1 Kernel Fusion

残差加法（element-wise add）是 memory-bound 算子（逐元素操作，计算强度极低）。通常与前一个算子融合为单个 CUDA Kernel：

```
[RMSNorm + Linear + SiLU + element-wise mul] + [残差 add]
↓  Kernel Fusion
单个 Kernel：读一次 x，写一次 output，中间结果留在寄存器/shared memory
```

减少 HBM 读写次数，对 memory-bound 算子收益显著。

### 6.2 量化中的精度问题

shortcut 路径的 $x$ 与主路径 $F(x)$ 的数值范围通常不同。Per-tensor 量化时，若对两者使用同一个量化参数（scale + zero-point），会因量化粒度过粗导致精度损失。

处理方案：

- 对 $x$ 和 $F(x)$ 分别校准量化参数，加法前在 FP16 或 INT32 域执行
- 或使用 Per-channel 量化，细化量化粒度

### 6.3 KV Cache 与残差

Transformer decoder 中，每层的隐状态 $h^{(l)}$ 通过残差累积：

$$h^{(l)} = h^{(l-1)} + F^{(l)}(h^{(l-1)})$$

KV Cache 存储的是各层 Attention 计算中的 $K$、$V$ 矩阵，均由当前层的隐状态线性投影得到。残差结构保证了历史 token 的信息在各层稳定传递，是 KV Cache 机制能够正确复用历史计算的基础。

---

## 7. 相关笔记

- [[梯度稳定性问题]]
- [[归一化方法]]
