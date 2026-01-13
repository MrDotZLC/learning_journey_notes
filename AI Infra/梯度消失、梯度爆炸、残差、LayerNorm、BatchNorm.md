# 一、梯度消失Gradient Vanishing
## 📌 定义
在反向传播中，梯度在层与层之间不断相乘，如果每一步的导数都 < 1，则梯度会越来越小，最终趋近于 0，使得前面的层 **几乎无法更新**。
## 📌 发生在哪？
最典型：
- **深层网络（深度 MLP、深 CNN、RNN）**
- **sigmoid / tanh 激活函数**
- **RNN 的时间步展开非常深**（几十甚至几百步）
## 📌 为什么会发生？
看 sigmoid 的导数：
$\sigma'(x) = \sigma(x)(1-\sigma(x)) \in (0, 0.25)$
如果你有 50 层：
$(0.25)^{50} \approx 8 \times 10^{-31}$
基本等于 0 → 训练前面完全学不到。
## 📌 后果
- 网络前层几乎不更新
- 学习变慢甚至停滞
- RNN 记不了长期依赖（10 步以上就忘光）
## 📌 解决方案
✔ ReLU / GELU / SiLU 
✔ 残差连接（ResNet）  
✔ 归一化（BatchNorm、LayerNorm）  
✔ LSTM / GRU（通过门控减少梯度衰减）  
✔ 用 xavier / kaiming 初始化

# 二、梯度爆炸Gradient Exploding
## 📌 定义
反向传播中，如果每一步梯度都 > 1，梯度会指数级增长，变得极大，导致参数更新量失控，被推到非常大的值。
## 📌 发生在哪？
- **深 RNN** 是重灾区（长序列、递归乘 Jacobian）
- Deep MLP 初始化不好
- 使用激活如 ReLU 在坏初始化时也可能爆
## 📌 为什么会发生？
当链式求导中多次出现导数 > 1，例如：
$(1.5)^{50} \approx 7 \times 10^{8}$
梯度瞬间爆炸。
## 📌 后果
- Loss 直接变成 NaN
- 参数在一次更新后爆炸
- 模型完全无法收敛
## 📌 解决方案
✔ **梯度裁剪（Gradient Clipping）** ← RNN 标配
`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
✔ 更好的初始化 
✔ 使用 LSTM / GRU 替代 vanilla RNN （门控）
✔ 使用 LayerNorm

# 三、残差（Residual）
## 残差连接（Residual Connection） = 跳跃连接（Skip Connection）
**网络只需要学习“对输入的修正”**，而不是重新从零生成输出。

---
## **ResNet/Transformer block 的核心是：**
$$y = x + F(x)$$
反向传播：
$$\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}​$$
---
## ✔ **梯度至少有一条捷径路径是 1（恒不消失）**
因为反向传播时至少有：
$$\frac{\partial L}{\partial x} \supset \frac{\partial L}{\partial y} \cdot I$$
所以：
- 梯度 _不会全部依赖_ 深层的链式求导
- 梯度不可能完全消失（因为有 identity shortcut）
---
## 残差结构让网络更容易学习到：
- 如果 F(x) 很难学，那至少可以学得**接近 0**
- 那输出就接近 x
- 这样就变成了一条“没有改变输入”的安全路径
- 梯度可以轻松从后面流到前面
📌 **本质：给神经网络加了一条“捷径”。**

这就是为什么有 1000 层的 ResNet 仍然能训练。

---
## 残差连接与 Highway Networks 的关系
### **Highway Networks（高速公路网络）**
- 早于 ResNet 的深层网络结构
- 公式：
$$y = H(x) \cdot T(x) + x \cdot C(x)$$
其中：
- H(x)：非线性变换
- T(x)：Transform Gate（学习多少使用 H(x)）
- C(x) = 1 - T(x)：Carry Gate（学习多少直接保留输入 x）

**特点：**
- 通过门控机制让信息在深层网络中可以直接传递
- 门控是可学习的
### **ResNet 残差连接**
- ResNet 公式：
$$y = x + F(x)$$
- 可以看作是 Highway Networks 的简化版本：
    - 恒等映射直接加到输出
    - **没有门控**（T(x) 恒为 1, C(x) 恒为 1）
    - 参数更少，计算更快

**直观理解：**
- Highway Networks = 带门控的残差连接
- ResNet = 不带门控的残差连接（“默认全通路”）

✅ 重点：残差连接是 Highway Networks 的简化版本，但足够让深度网络稳定训练。
---
## 残差连接如何改善梯度流动？（数学解释）
假设一层残差块：
$$y = x + F(x)$$
损失函数 L 对输入 x 的梯度：
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial L}{\partial y} \cdot (1 + \frac{\partial F}{\partial x})$$
### **关键点**
1. **梯度中有一条恒等项**
$$\frac{\partial L}{\partial y} \cdot 1$$
- 即使 F(x) 很深或很复杂，梯度仍然有一条直接通路
- 避免梯度消失

2. **梯度可以直接累积到前面层**
- 深层网络中，每个残差块都像搭了一个“高速通道”
- 梯度不会被多层非线性完全阻断

1. **增量学习更稳定**
- F(x) 只需要学习对输入的微调（residual）
- 网络整体训练更容易收敛
### 🔹 直观比喻
- 普通深层网络：梯度像水，要经过每层“滤网”，容易流失
- 残差网络：每层有一条平滑的高速通道，梯度直接流回去，不被阻断
- Highway Networks：可以调节通道大小（门控），梯度流动更灵活
---
## 残差连接优点：
### ✓ 1. 让深度网络更容易训练
梯度回传路径更短，不易消失。
### ✓ 2. 允许模型学“增量”
F(x) 只需要学到 **“对 x 的修正”**，而不是从零开始学。
### ✓ 3. 防止深层退化
层数增加不会让精度变差。

### 简短总结（面试可直接说）
残差连接：把输入加到输出上，形成“x + F(x)”，让神经网络训练更稳定、更深。

# 四、Batch Normalization（BN）
## 1.1 定义
BatchNorm 是 **按 batch 对每个特征维度** 做归一化，目的是 **缓解梯度消失/爆炸，保持方差在合理范围，提高训练稳定性和收敛速度**。

---
## 1.2 公式
给定 mini-batch $\mathcal{B} = \{ x_1, x_2, ..., x_m \}$，每个样本$x_i \in \mathbb{R}^d$：
1. **计算 batch 均值和方差**：
$$\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^m x_i$$​$$\sigma_\mathcal{B}^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_\mathcal{B})^2$$
2. **归一化**：
$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}$$
3. **线性变换（可学习）**：
$$y_i = \gamma \hat{x}_i + \beta$$
- $\gamma, \beta$ 是可训练参数，保证网络有能力恢复原来的分布。
- $\epsilon$ 是数值稳定项。
---
## 1.3 特性和优势
1. 缓解 **梯度消失 / 梯度爆炸**
2. 加快 **收敛速度**
3. 有一定 **正则化效果**，减少过拟合
4. 与卷积和全连接网络都兼容
---
## 1.4 局限
1. 对 **RNN/Transformer** 不友好
    - 长序列 batch 太小 → 均值方差不稳定
2. 对 **batch size 很小** 时不稳定
3. 对训练和推理处理不同：
    - 推理阶段使用 **moving average 的均值方差**
    - 训练阶段用 batch 的即时均值方差
---
## 1.5 应用
- ResNet / CNN 系列几乎必备
- 在 Transformer 中常用 **LayerNorm 替代 BN**


# 五、Layer Normalization（LN）

## 2.1 定义
LayerNorm 是 **对每个样本自身的所有特征维度做归一化**，保持方差在合理范围，不依赖 batch。  
公式和 BN 类似，但归一化的对象不同。

---
## 2.2 公式
给定样本 $x \in \mathbb{R}^d$：
1. **计算均值和方差**（沿特征维度）：
$$\mu = \frac{1}{d} \sum_{j=1}^{d} x_j$$ 
$$\sigma^2 = \frac{1}{d} \sum_{j=1}^{d} (x_j - \mu)^2$$
2. **归一化**：

$$\hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}}​$$

3. **可学习线性变换**：

$$y_j = \gamma \hat{x}_j + \beta$$
- $\gamma, \beta \in \mathbb{R}^d$

---
## 2.3 特性和优势
1. **不依赖 batch** → 小 batch 或单样本也稳定
2. 非常适合 **RNN / Transformer**
3. 梯度更稳定，训练长序列或深层模型不会消失
4. 与序列模型结合，通常放在每个子层前（Pre-LN）或者后（Post-LN）

---
## 2.4 局限
1. 对 CNN 有局限，因为特征之间空间关系被 LN 混合
2. 不产生 BN 的 mini-batch 正则化效果

