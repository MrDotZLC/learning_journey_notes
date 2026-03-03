## 一、RMSNorm 是什么（核心定义）
RMSNorm（均方根归一化）通过将向量按其均方根进行缩放，使其整体幅值保持稳定，同时不改变向量的相对方向。
它的核心思想是：
> **只对向量的“幅值大小”做归一化，不对均值做中心化（zero-mean）。**

## 二、数学定义（严格版）
给定一个向量：
$$  
\mathbf{x} = (x_1, x_2, \dots, x_d)  
$$
### 1️⃣ 计算 RMS（均方根）

$$  
\mathrm{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}  
$$
- d：hidden size / embedding dim
- $\epsilon$：数值稳定项（如 (1e{-6})）
### 2️⃣ 归一化 + 可学习缩放

$$ 
\mathrm{RMSNorm}(\mathbf{x}) = \mathbf{g} \odot \frac{\mathbf{x}}{\mathrm{RMS}(\mathbf{x})}  
$$
- $\mathbf{g}$：**可学习的 scale 参数**（shape = `[d]`）
- 没有 bias（通常）
## 三、和 LayerNorm 的本质区别（非常关键）

|对比项|LayerNorm|RMSNorm|
|---|---|---|
|是否减均值|✅ 是|❌ 否|
|是否除标准差|✅ 是|❌ 否|
|是否除 RMS|❌|✅|
|参数|weight + bias|通常只有 weight|
|计算量|高|更低|
|数值稳定性|好|对 LLM 足够好|

### 数学对比
**LayerNorm：**
$$\frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

**RMSNorm：**
$$\frac{x_i}{\sqrt{\frac{1}{d} \sum x_i^2 + \epsilon}} \cdot g_i$$ 
## 四、为什么 RMSNorm 适合大语言模型
这是 RMSNorm 在 LLaMA / InternLM / Qwen / Mistral 中被大量采用的原因。
### 1️⃣ Transformer 中“均值”并不重要
在自注意力和 FFN 中：
- 线性层 + residual
- attention score 主要依赖相对值
- embedding 本身已经近似零均值

👉 **减均值的收益有限**
### 2️⃣ RMSNorm 更便宜（推理关键）
相比 LayerNorm：
- 少一次 reduce（不算 mean）
- 少一次减法
- 内存访问更少
- 更适合 SIMD / 向量化

在 CPU 推理中，这是**实打实的性能收益**。
### 3️⃣ 数值行为更“温和”
RMSNorm 的作用更像是：
> **“限制向量长度”，而不是“重新分布向量形状”**

这对深层 Transformer 非常重要。
## 五、RMSNorm 在 Transformer 中的位置
以 LLaMA / InternLM 为例（Pre-Norm 架构）：
```text
x → RMSNorm → Attention → +x
x → RMSNorm → FFN       → +x
```
注意：
- RMSNorm 在 **子层之前**
- residual 保持原始尺度

这和早期 Post-Norm Transformer 不同。
## 六、一个具体数值例子（直观理解）
假设：
```text
x = [1, 2, 3, 4]
g = [1, 1, 1, 1]
epsilon = 0
```
### 1️⃣ 计算 RMS
$$\sqrt{(1^2 + 2^2 + 3^2 + 4^2)/4}  
= \sqrt{7.5}  
≈ 2.7386  $$
### 2️⃣ 归一化
```text
x_norm = [0.365, 0.730, 1.095, 1.460]
```
向量方向不变，只是“缩放”。
## 七、RMSNorm 的工程实现（ggml / llama.cpp 风格）
伪代码如下：
```c
float sumsq = 0;
for (i = 0; i < d; i++) {
    sumsq += x[i] * x[i];
}

float rms = sqrt(sumsq / d + eps);

for (i = 0; i < d; i++) {
    y[i] = x[i] / rms * g[i];
}
```
### 工程要点
- 可用 SIMD 做平方和
- scale (`g`) 是连续内存
- 不需要保存 mean / variance
- **非常适合 CPU kernel**
## 八、RMSNorm 在量化推理中的优势
在 Q4 / Q8 推理中：
- RMSNorm 通常在 **反量化之后**
- scale 参数保持 FP16 / FP32
- 运算稳定，不放大量化误差

相比 LayerNorm：
- 更少的算子
- 更少的误差传播路径
## 九、常见误解澄清
### ❌ RMSNorm = 简化版 LayerNorm
不完全对。它是 **不同归一化假设下的设计**。
### ❌ RMSNorm 会导致训练不稳定
在大模型实践中已被证伪（LLaMA 系列）。
### ❌ 没有 bias 就一定不好
在 Transformer 中 bias 的作用极弱，scale 才是关键。
## 十、一句话总结（请记住）
> **RMSNorm = 只控制“向量长度”，不关心“向量中心”。**

或者工程化一点：

> **用更低的计算成本，换取在 LLM 中“足够好”的归一化效果。**