## 1. 背景与动机

深层网络训练中，前层参数更新导致后层输入分布持续变化（Internal Covariate Shift，ICS），使后层需要不断适应新分布，降低训练效率，并加剧梯度消失/爆炸（详见 [[梯度稳定性]]）。

归一化层的统一目标：**控制各层输入的统计分布，使训练更稳定、收敛更快**。

> 注：ICS 的理论解释后续受到质疑（Santurkar et al., 2018 指出 BN 的真实作用可能是平滑损失曲面），但实际效果被广泛验证。

### 1.1 归一化维度直觉

> 【图示占位】：`[B, T, d]` 三维张量示意图，分别高亮 BN（沿 Batch 维）、LN（沿 Feature 维）、RMSNorm（沿 Feature 维，无中心化）的归一化方向

设输入张量形状为 `[B, T, d]`（Batch × Sequence Length × Hidden dim）：

|方法|归一化维度|每次统计范围|
|---|---|---|
|BatchNorm|Batch 维（per feature）|同一特征的 $B \times T$ 个值|
|LayerNorm|Feature 维（per sample）|同一 token 的 $d$ 个特征|
|RMSNorm|Feature 维（per sample，无中心化）|同一 token 的 $d$ 个特征|
|GroupNorm|分组 Channel 维|同一样本内分组的 $C/G$ 个通道|

---

## 2. Batch Normalization（BN）

### 2.1 背景

Ioffe & Szegedy（2015）提出，核心动机是解决深层 CNN 训练时因输入分布变化导致的训练不稳定问题。BN 使 ResNet、VGG 等深层 CNN 可以使用更大学习率、更快收敛，同时附带一定正则化效果，是 2015–2020 年 CV 领域几乎必用的模块。

### 2.2 完整公式推导

给定 mini-batch $\mathcal{B} = {x_1, \ldots, x_m}$，$x_i \in \mathbb{R}^d$。

**Step 1：计算 batch 统计量（逐特征维度）**

$$\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i \in \mathbb{R}^d, \quad \sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2 \in \mathbb{R}^d$$

**Step 2：归一化**

$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}, \quad \epsilon \approx 10^{-5}$$

**Step 3：仿射变换（可学习）**

$$y_i = \gamma \odot \hat{x}_i + \beta, \quad \gamma, \beta \in \mathbb{R}^d$$

### 2.3 反向传播梯度推导

定义 $\bar{\sigma}_{\mathcal{B}} = \sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}$。

**Step 1**：

$$\frac{\partial \mathcal{L}}{\partial \hat{x}_i} = \frac{\partial \mathcal{L}}{\partial y_i} \cdot \gamma$$

**Step 2**：

$$\frac{\partial \mathcal{L}}{\partial \sigma_{\mathcal{B}}^2} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot (x_i - \mu_{\mathcal{B}}) \cdot \left(-\frac{1}{2}\right) \bar{\sigma}_{\mathcal{B}}^{-3}$$

**Step 3**：

$$\frac{\partial \mathcal{L}}{\partial \mu_{\mathcal{B}}} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \left(-\frac{1}{\bar{\sigma}_{\mathcal{B}}}\right) + \frac{\partial \mathcal{L}}{\partial \sigma_{\mathcal{B}}^2} \cdot \frac{-2}{m} \sum_{i=1}^{m}(x_i - \mu_{\mathcal{B}})$$

**Step 4（最终梯度）**：

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \frac{1}{\bar{\sigma}_{\mathcal{B}}} + \frac{\partial \mathcal{L}}{\partial \sigma_{\mathcal{B}}^2} \cdot \frac{2(x_i - \mu_{\mathcal{B}})}{m} + \frac{\partial \mathcal{L}}{\partial \mu_{\mathcal{B}}} \cdot \frac{1}{m}$$

**关键结论**：$\partial \mathcal{L} / \partial x_i$ 包含 batch 内所有样本的耦合项（通过 $\mu_{\mathcal{B}}$ 和 $\sigma_{\mathcal{B}}^2$），这是 BN **不适用于单样本推理**的根本原因。

### 2.4 训练 vs 推理的差异

|阶段|$\mu$ 来源|$\sigma^2$ 来源|
|---|---|---|
|训练|当前 mini-batch 即时计算|当前 mini-batch 即时计算|
|推理|训练期 EMA（指数移动平均）|训练期 EMA|

EMA 更新规则（$\alpha$ 通常取 $0.1$）：

$$\mu_{\text{run}} \leftarrow (1-\alpha)\mu_{\text{run}} + \alpha\mu_{\mathcal{B}}, \quad \sigma^2_{\text{run}} \leftarrow (1-\alpha)\sigma^2_{\text{run}} + \alpha\sigma^2_{\mathcal{B}}$$

### 2.5 小 batch 下的退化

BN 统计量估计误差与 $1/m$ 成正比。ImageNet 实验数据（He et al.）：

|batch size|Top-1 误差（ResNet-8×）|
|---|---|
|32|23.0%|
|16|23.6%|
|8|24.7%|
|4|26.6%|
|2|**29.7%**（$+6.7%$）|

经验阈值 $m < 8$ 来源于此：误差开始以非线性速率恶化的临界点。GroupNorm 在所有 batch size 下误差稳定在 $\approx 24%$，小 batch 场景可完全替代 BN。

### 2.6 BN Folding（推理优化）

推理时 BN 使用固定统计量，可完全融合进前一层卷积权重：

$$W' = \text{diag}!\left(\frac{\gamma}{\sqrt{\sigma^2_{\text{run}} + \epsilon}}\right) \cdot W, \quad b' = \frac{\gamma \odot (b - \mu_{\text{run}})}{\sqrt{\sigma^2_{\text{run}} + \epsilon}} + \beta$$

推理等价于 $y = W'x + b'$，**BN 被完全吸收，推理时零额外计算开销**。TensorRT、TVM、OpenVINO 均自动执行此优化。

### 2.7 优缺点

**优点**：

- 显著加快训练收敛速度，允许使用更大学习率
- 有一定正则化效果（mini-batch noise），减少对 Dropout 的依赖
- 推理时可完全 Fold，零额外计算开销
- 对权重初始化不敏感，降低调参难度

**缺点**：

- 强依赖 batch size，$m < 8$ 时性能急剧退化
- 不适用于 RNN / Transformer（序列长度可变，batch 统计无意义）
- 训练与推理行为不一致（EMA 估计存在误差）
- 分布式训练需要 SyncBN（跨设备 All-Reduce），增加通信开销
- 单样本 Online 推理场景下行为退化

---

## 3. Layer Normalization（LN）

### 3.1 背景

Ba et al.（2016）提出，专门针对 BN 不适用于 RNN 的问题而设计。归一化维度从 Batch 维切换到 Feature 维，统计量只依赖当前样本自身，彻底摆脱对 batch size 的依赖。LN 成为 Transformer（Vaswani et al., 2017）的标准归一化层，BERT、GPT 系列均采用。

### 3.2 完整公式推导

给定单样本 $x \in \mathbb{R}^d$。

**Step 1：计算特征维度统计量**

$$\mu = \frac{1}{d} \sum_{j=1}^{d} x_j, \quad \sigma^2 = \frac{1}{d} \sum_{j=1}^{d} (x_j - \mu)^2$$

**Step 2：归一化**

$$\hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

**Step 3：仿射变换**

$$y_j = \gamma_j \hat{x}_j + \beta_j, \quad \gamma, \beta \in \mathbb{R}^d$$

### 3.3 反向传播梯度推导

定义 $\bar{\sigma} = \sqrt{\sigma^2 + \epsilon}$。

**Step 1**：

$$\frac{\partial \mathcal{L}}{\partial \hat{x}_j} = \frac{\partial \mathcal{L}}{\partial y_j} \cdot \gamma_j$$

**Step 2**：

$$\frac{\partial \mathcal{L}}{\partial \sigma^2} = \sum_{j=1}^{d} \frac{\partial \mathcal{L}}{\partial \hat{x}_j} \cdot (x_j - \mu) \cdot \left(-\frac{1}{2}\right) \bar{\sigma}^{-3}$$

**Step 3**（利用 $\sum_j(x_j - \mu) = 0$，第二项消去）：

$$\frac{\partial \mathcal{L}}{\partial \mu} = -\frac{1}{\bar{\sigma}} \sum_{j=1}^{d} \frac{\partial \mathcal{L}}{\partial \hat{x}_j}$$

**Step 4（最终梯度）**：

$$\frac{\partial \mathcal{L}}{\partial x_j} = \frac{\partial \mathcal{L}}{\partial \hat{x}_j} \cdot \frac{1}{\bar{\sigma}} + \frac{\partial \mathcal{L}}{\partial \sigma^2} \cdot \frac{2(x_j - \mu)}{d} + \frac{\partial \mathcal{L}}{\partial \mu} \cdot \frac{1}{d}$$

**与 BN 梯度的本质区别**：LN 的梯度完全由单样本自身决定，**无 batch 间耦合**，任意 batch size 下行为完全一致。

### 3.4 优缺点

**优点**：

- 完全不依赖 batch size，单样本推理稳定
- 训练与推理行为完全一致，无需维护 EMA
- 天然适合序列模型，分布式训练无需额外同步

**缺点**：

- 推理时统计量依赖运行时输入，无法做 BN Folding，每次推理需在线计算
- 本质是 memory-bound 算子，性能瓶颈在显存带宽
- 对 CNN 有局限，跨通道混合破坏空间特征关系
- 不产生 BN 的 mini-batch 正则化效果

---

## 4. RMSNorm

### 4.1 背景

Zhang & Sennrich（2019）提出，核心论点是 LN 中的均值中心化步骤对模型效果贡献极为有限，去掉后计算量降低约 40%，实验指标几乎不变。设计假设：**只对向量的幅值做归一化，不对均值做中心化**。均值中心化作用有限的原因：线性层 + Kaiming 初始化已保证激活均值近似为零；Self-Attention 的 Softmax 对全局均值偏移不敏感。LLaMA、InternLM、Qwen、Mistral 等主流 LLM 均采用 RMSNorm。

### 4.2 完整公式推导

给定向量 $\mathbf{x} = (x_1, \ldots, x_d) \in \mathbb{R}^d$。

**Step 1：计算 RMS（均方根）**

$$\mathrm{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}, \quad \epsilon \approx 10^{-6}$$

**Step 2：归一化 + 可学习缩放**

$$y_i = g_i \cdot \frac{x_i}{\mathrm{RMS}(\mathbf{x})}, \quad \mathbf{g} \in \mathbb{R}^d \text{（无 bias 项）}$$

### 4.3 反向传播梯度推导

定义 $r = \mathrm{RMS}(\mathbf{x})$，$\delta_i = \partial \mathcal{L} / \partial y_i$。

**Step 1**：对可学习参数 $g_i$：

$$\frac{\partial \mathcal{L}}{\partial g_i} = \delta_i \cdot \frac{x_i}{r}$$

**Step 2**：损失对 $r$ 的梯度：

$$\frac{\partial \mathcal{L}}{\partial r} = -\frac{1}{r^2} \sum_{i=1}^{d} \delta_i g_i x_i$$

**Step 3**：$r$ 对 $x_j$ 的偏导：

$$\frac{\partial r}{\partial x_j} = \frac{x_j}{d \cdot r}$$

**Step 4（最终梯度，合并直接路径与通过 $r$ 的路径）**：

$$\frac{\partial \mathcal{L}}{\partial x_j} = \frac{\delta_j g_j}{r} - \frac{x_j}{d \cdot r^3} \sum_{i=1}^{d} \delta_i g_i x_i$$

**与 LN 梯度的对比**：RMSNorm 梯度中**无均值梯度项**，LN 有三项，RMSNorm 只有两项，计算量更少，数学结构更简洁。两者均无 batch 耦合。

### 4.4 数值例子

输入 $\mathbf{x} = [1, 2, 3, 4]$，$\mathbf{g} = [1, 1, 1, 1]$，$\epsilon = 0$：

$$\mathrm{RMS}(\mathbf{x}) = \sqrt{7.5} \approx 2.7386, \quad \mathbf{y} \approx [0.365,\ 0.730,\ 1.095,\ 1.460]$$

向量**方向不变**，仅幅值缩放至 RMS 为 $1$ 的状态。

### 4.5 与 LayerNorm 的公式对比

|对比项|LayerNorm|RMSNorm|
|---|---|---|
|是否减均值|✅ 是|❌ 否|
|归一化分母|$\sqrt{\sigma^2 + \epsilon}$|$\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}$|
|可学习参数|$\gamma$（weight）$+ \beta$（bias）|$\mathbf{g}$（weight，无 bias）|
|计算步骤|基准|约 $40%$ 节省|
|梯度项数|3 项（含均值梯度项）|2 项|

### 4.6 工程实现（llama.cpp 风格）

```c
void rmsnorm(float* y, const float* x, const float* g, int d, float eps) {
    float sumsq = 0.0f;
    for (int i = 0; i < d; i++) sumsq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sumsq / d + eps); /* CUDA 中用 rsqrtf，约快 2× */
    for (int i = 0; i < d; i++) y[i] = x[i] * inv_rms * g[i];
}
```

工程要点：平方和循环可用 AVX2 `_mm256_fmadd_ps` 展开（吞吐量提升 4–8×）；不需要保存 mean / variance；Kernel Fusion（与相邻线性层合并）是 memory-bound 算子的核心优化手段。

### 4.7 在量化推理中的行为

Scale 参数 $\mathbf{g}$ 保持 FP16 / FP32 精度，在高精度域执行归一化，不放大量化误差。算子数量少于 LN，误差传播路径更短，量化精度更好。无法做 BN Folding，优化核心是 Kernel Fusion。

### 4.8 优缺点

**优点**：

- 计算量比 LN 低约 $40%$，推理吞吐量更高
- 梯度计算少一项，反向传播开销更低
- 无 bias 参数，可学习参数量从 $2d$ 降至 $d$
- 与 Pre-LN 架构配合时训练曲线平滑，无需 Warmup
- 量化推理中误差传播路径更短，精度更好
- LLaMA 系列大规模训练中验证稳定有效

**缺点**：

- 放弃均值中心化，对有显著均值偏移的分布处理能力理论上弱于 LN（实践影响可忽略）
- 无 bias 项，模型偏置表达能力略弱
- 无法做 BN Folding，推理时仍需在线计算
- 不产生正则化效果

---

## 5. 方法全面对比与选型

### 5.1 对比总表

|特性|BatchNorm|LayerNorm|RMSNorm|GroupNorm|
|---|---|---|---|---|
|归一化维度|Batch（per feature）|Feature（per sample）|Feature（per sample，无中心化）|分组 Channel|
|Batch 依赖|强依赖|无|无|无|
|适用场景|CNN|Transformer / RNN|LLM（LLaMA 系）|CNN（小 batch）|
|推理额外计算|无（可 Fold）|有（在线计算）|有（在线计算，更轻）|有|
|可学习参数|$\gamma, \beta \in \mathbb{R}^C$|$\gamma, \beta \in \mathbb{R}^d$|$\mathbf{g} \in \mathbb{R}^d$|$\gamma, \beta \in \mathbb{R}^C$|
|训练/推理一致性|不一致（EMA）|一致|一致|一致|
|计算量|中|中|低（约 $40%$ 节省）|中|
|正则化效果|有|无|无|无|
|单样本推理|❌ 不稳定|✅ 稳定|✅ 稳定|✅ 稳定|

### 5.2 选型决策树

```
需要归一化？
│
├── 任务是 CNN / 视觉模型？
│   ├── batch size ≥ 16？  →  BatchNorm（可 Fold，推理零开销，有正则化）
│   └── batch size 小？    →  GroupNorm（性能不随 batch 退化）
│
└── 任务是 Transformer / LLM？
    ├── 复现已有模型（BERT 等）？  →  LayerNorm + Post-LN
    ├── 从头训练 LLM？             →  RMSNorm + Pre-LN（LLaMA 默认）
    └── 极深网络（> 100 层）？     →  DeepNorm（参见 [[残差连接]]）
```

### 5.3 推理优化视角对比

|优化技术|BatchNorm|LayerNorm|RMSNorm|
|---|---|---|---|
|静态参数折叠（Folding）|✅ 可完全折叠|❌ 不可|❌ 不可|
|Kernel Fusion|收益一般（已 Fold）|收益显著|收益显著|
|量化友好性|Fold 后天然友好|好|好（路径更短）|
|分布式训练|需 SyncBN（All-Reduce）|无需同步|无需同步|
|单样本推理|❌ 不稳定|✅ 稳定|✅ 稳定|

---

## 6. 相关笔记

- [[梯度稳定性]]
- [[残差连接]]
