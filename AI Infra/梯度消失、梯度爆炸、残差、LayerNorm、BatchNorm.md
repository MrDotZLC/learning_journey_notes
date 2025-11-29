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
相对于多层非线性地从“0”学习一个函数，从"y=x"学起可以防止梯度消失或爆炸，残差连接就是保证至少存在“y=x”
### **ResNet/Transformer block 的核心是：**
y = x + F(x)
反向传播：
$\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}​$
### ✔ **梯度至少有一条捷径路径是 1（恒不消失）**
因为反向传播时至少有：
$\frac{\partial L}{\partial x} \supset \frac{\partial L}{\partial y} \cdot I$
所以：
- 梯度 _不会全部依赖_ 深层的链式求导
- 梯度不可能完全消失（因为有 identity shortcut）

### 残差结构让网络更容易学习到：
- 如果 F(x) 很难学，那至少可以学得**接近 0**
- 那输出就接近 x
- 这样就变成了一条“没有改变输入”的安全路径
- 梯度可以轻松从后面流到前面
📌 **本质：给神经网络加了一条“捷径”。**

这就是为什么有 1000 层的 ResNet 仍然能训练。

### 优点：
#### ✓ 1. 让深度网络更容易训练
梯度回传路径更短，不易消失。
#### ✓ 2. 允许模型学“增量”
F(x) 只需要学到 **“对 x 的修正”**，而不是从零开始学。
#### ✓ 3. 防止深层退化
层数增加不会让精度变差。

### 简短总结（面试可直接说）
残差连接：把输入加到输出上，形成“x + F(x)”，让神经网络训练更稳定、更深。

