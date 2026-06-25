## 1. 宏观对比框架

训练与推理的根本差异在于**计算图的方向与范围**：

|维度|训练|推理|
|---|---|---|
|计算方向|Forward + Backward|Forward only|
|序列处理模式|Teacher Forcing（全序列并行）|Autoregressive（逐 token 串行）|
|显存占用|参数 + 梯度 + 优化器状态 + 激活值|参数 + KV Cache|
|计算瓶颈|compute-bound（大 batch，矩阵乘）|memory-bound（小 batch，GEMV）|
|目标函数|最小化 Cross-Entropy Loss|最大化生成质量/吞吐/延迟|
|随机性|Dropout、随机梯度噪声主动引入|推理时关闭 Dropout|
|位置信息|完整序列一次性处理|位置递增，需 KV Cache 复用|

---

## 2. 前向计算的根本差异：Teacher Forcing vs Autoregressive

### 2.1 训练：Teacher Forcing

训练时给定目标序列 $\mathbf{y} = (y_1, y_2, \ldots, y_T)$，在每个位置 $t$ 使用**真实的历史 token** 作为条件输入，而非模型自身的预测：

$$ \mathcal{L} = -\sum_{t=1}^{T} \log P(y_t \mid y_1, \ldots, y_{t-1};, \theta) $$

整个序列一次性输入模型：

$$ \mathbf{X}_{\text{train}} = [x_1, x_2, \ldots, x_T] \in \mathbb{R}^{T \times d} $$

注意力矩阵 $\mathbf{A} \in \mathbb{R}^{T \times T}$ 通过 **Causal Mask** 保证位置 $t$ 只 attend 到 $j \leq t$，但所有位置**并行计算**：

$$ \mathbf{A}_{ij} = \frac{\exp(s_{ij} / \sqrt{d_k})}{\sum_{k \leq i} \exp(s_{ik} / \sqrt{d_k})}, \quad s_{ij} = \mathbf{q}_i^\top \mathbf{k}_j $$

**关键特性：** 所有 $T$ 个位置的 Q/K/V 同时参与矩阵乘法，计算 $\mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{T \times T}$ 是一次 GEMM 操作，GPU 利用率高。

### 2.2 推理：Autoregressive Decoding

推理时模型自回归生成，位置 $t$ 的输入依赖模型在 $t-1$ 步的**自身输出**：

$$ \hat{y}_t \sim P(\cdot \mid \hat{y}_1, \ldots, \hat{y}_{t-1};, \theta) $$

每一步只有 **1 个新 token** 进入模型：

$$ \mathbf{x}_t \in \mathbb{R}^{1 \times d} $$

注意力计算变为：

$$ a_{tj} = \frac{\exp(\mathbf{q}_t^\top \mathbf{k}_j / \sqrt{d_k})}{\sum_{k=1}^{t} \exp(\mathbf{q}_t^\top \mathbf{k}_k / \sqrt{d_k})}, \quad j \leq t $$

$\mathbf{q}_t \in \mathbb{R}^{1 \times d_k}$ 与缓存的 $\mathbf{K} \in \mathbb{R}^{t \times d_k}$ 做向量-矩阵乘（GEMV），计算强度极低。

### 2.3 两种模式的计算图对比

```
训练（Teacher Forcing）：
┌────────────────────────────────────┐
│ x1  x2  x3  x4  x5  → 全序列并行  │
│  ↓   ↓   ↓   ↓   ↓                │
│ [Q1][Q2][Q3][Q4][Q5]               │
│      × [K1,K2,K3,K4,K5]            │
│      → T×T Attention (GEMM)        │
└────────────────────────────────────┘

推理（Autoregressive）：
Step 1: x1 → [Q1] × [K1]         → y1（1×1 Attention）
Step 2: x2 → [Q2] × [K1,K2]     → y2（1×2 Attention）
Step 3: x3 → [Q3] × [K1,K2,K3]  → y3（1×3 Attention）
         ...（每步 GEMV，串行依赖）
```

---

## 3. 显存占用差异

### 3.1 训练的显存构成

混合精度训练（AMP，FP16/BF16 前向 + FP32 参数副本）下，模型参数量 $N$ 对应的显存：

|组件|每参数字节数|总量（$N$ 参数）|
|---|---|---|
|FP16 参数（前向用）|2 B|$2N$ B|
|FP32 参数副本（梯度更新用）|4 B|$4N$ B|
|FP32 梯度|4 B|$4N$ B|
|Adam 优化器状态（一阶矩 + 二阶矩）|4+4 B|$8N$ B|
|**参数相关合计**|**18 B/param**|**$18N$ B**|

激活值显存（用于反向传播）是训练独有的额外开销：

$$ \text{激活显存（单层，序列长 }T\text{，batch }B\text{）} \approx B \cdot T \cdot d \cdot \text{层内中间张量数} $$

以标准 Transformer 层为例（不含 Activation Checkpointing），单层激活约占：

$$ \approx B \cdot T \cdot d \cdot 10 \times \text{dtype\_bytes} $$

对长序列（$T=4096$）、大 batch（$B=32$）、$d=4096$ 的 FP16 模型，**仅激活值**就约 $32 \times 4096 \times 4096 \times 10 \times 2 \text{ B} \approx 10 \text{ GB/层}$。

**Activation Checkpointing（梯度检查点）：** 放弃保存中间激活，反向传播时重新计算，显存从 $O(L \cdot B \cdot T \cdot d)$ 降至 $O(\sqrt{L} \cdot B \cdot T \cdot d)$，代价是约 $33\%$ 额外计算量。

### 3.2 推理的显存构成

推理无需梯度和优化器状态，且通常使用量化权重：

|组件|占用|
|---|---|
|模型权重（FP16）|$2N$ B|
|KV Cache|$2 \times L \times n_{kv} \times d_k \times s \times 2$ B|
|激活值（即时计算，可复用）|$O(B \times d)$，极小|

**数值对比（LLaMA-3-70B，$N \approx 70\text{B}$）：**

|场景|显存需求|
|---|---|
|训练（BF16+FP32 AMP + Adam）|$\approx 18 \times 70\text{B} \times 1\text{B} \approx 1260\text{ GB}$|
|推理（BF16 权重，batch=1，seq=4096）|$\approx 140\text{ GB（权重）} + 21\text{ GB（KV）} \approx 161\text{ GB}$|

---

## 4. 反向传播：推理阶段完全消除

### 4.1 训练中的梯度计算

设第 $l$ 层输出 $\mathbf{h}^{(l)}$，损失 $\mathcal{L}$ 对权重 $\mathbf{W}$ 的梯度：

$$ \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(l)}} \cdot \frac{\partial \mathbf{h}^{(l)}}{\partial \mathbf{W}^{(l)}} $$

链式法则从输出层向输入层逐层传播，每层需保留前向激活值用于梯度计算（这是激活值必须保存的根本原因）。

反向传播的计算量约为前向传播的 **2 倍**，因此训练总 FLOP 约为推理的 **3 倍**。

$$ \text{FLOP}_{\text{train}} \approx 3 \times \text{FLOP}_{\text{forward}} $$

### 4.2 推理中的计算图

推理时：

```python
with torch.no_grad():   # 关闭梯度追踪
    output = model(input_ids)
```

- 无需构建计算图（`grad_fn` 链）
- 中间激活不需要持久化，可立即释放或被后续操作覆盖
- 显存可复用（in-place 操作更激进）

PyTorch 的 `torch.no_grad()` / `torch.inference_mode()` 的差异：

|上下文管理器|禁用梯度追踪|禁用 `view` 追踪|性能|
|---|---|---|---|
|`no_grad()`|✓|✗|中|
|`inference_mode()`|✓|✓|最优|

---

## 5. Dropout 与归一化层的行为差异

### 5.1 Dropout

训练时以概率 $p$ 随机置零激活，推理时关闭并缩放：

$$ \text{训练：} \quad y_i = \begin{cases} 0 & \text{概率 } p \ x_i / (1-p) & \text{概率 } 1-p \end{cases} $$

$$ \text{推理：} \quad y_i = x_i \quad \text{（identity mapping）} $$

在 PyTorch 中通过 `model.train()` / `model.eval()` 切换，**推理必须调用 `model.eval()`**，否则 Dropout 仍激活导致结果随机且性能下降。

### 5.2 Batch Normalization（BN，Transformer 中不常用但需了解）

训练时使用当前 mini-batch 的统计量：

$$ \hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}, \quad \mu_{\mathcal{B}} = \frac{1}{m}\sum_{i=1}^m x_i $$

推理时使用训练过程中累积的**滑动平均统计量** $\bar{\mu}$、$\bar{\sigma}^2$（固定值）：

$$ \hat{x}_i = \frac{x_i - \bar{\mu}}{\sqrt{\bar{\sigma}^2 + \epsilon}} $$

**Transformer 使用 LayerNorm / RMSNorm 而非 BN：** LN/RMSNorm 归一化维度是特征维 $d$（而非 batch 维），统计量在推理和训练时的计算方式一致，无需区分模式。这是 LN 适合 NLP 序列变长场景的根本原因。

---

## 6. 位置编码与序列长度处理

### 6.1 训练时的位置编码

训练时模型见到固定长度（或动态 padding 到最大长度）的序列，RoPE 旋转角：

$$ \theta_k(m) = m \cdot \theta_k^{(0)}, \quad \theta_k^{(0)} = 10000^{-2k/d_k} $$

位置索引 $m$ 从 $0$ 到 $T_{\text{train}}-1$，模型对这个范围内的位置分布充分优化。

### 6.2 推理时的长度外推问题

推理时若序列长度 $s > T_{\text{train}}$，旋转角 $\theta_k(m)$ 进入训练未见区域，注意力分数分布失真。

**主流外推方案：**

**1. YaRN（Yet another RoPE extensioN）：** 对不同频率分量（高频/低频）分别施加不同的内插/外插策略：

$$ \theta_k^{\text{YaRN}}(m) = \begin{cases} m \cdot \theta_k^{(0)} & \text{高频分量（不插值）} \ m \cdot \theta_k^{(0)} / s & \text{低频分量（线性内插）} \end{cases} $$

其中 $s = T_{\text{infer}} / T_{\text{train}}$ 为缩放比例。

**2. 线性内插（Position Interpolation, PI）：** 将位置 $m$ 缩放为 $m \cdot T_{\text{train}} / T_{\text{infer}}$，强制所有位置落入训练区间。代价：高频分辨率下降。

**3. NTK-aware 插值：** 将 RoPE base 从 $10000$ 调整为 $10000 \cdot s^{d/(d-2)}$，使高频分量保持密度、低频分量线性压缩：

$$ \theta_k^{\text{NTK}}(m) = m \cdot \left(\alpha \cdot 10000\right)^{-2k/d_k}, \quad \alpha = s^{d/(d-2)} $$

---

## 7. Attention Mask 的差异

### 7.1 训练时的 Mask 构成

训练时同一 batch 内不同序列长度需要 Padding，Mask 由两部分叠加：

$$ \mathbf{M}_{ij} = \underbrace{\mathbf{M}^{\text{causal}}_{ij}}_{\text{因果掩码}} + \underbrace{\mathbf{M}^{\text{pad}}_{ij}}_{\text{Padding 掩码}} $$

$$ M^{\text{pad}}_{ij} = \begin{cases} 0 & \text{若 } j \text{ 为有效 token} \ -\infty & \text{若 } j \text{ 为 Padding token} \end{cases} $$

训练时通常通过 **Flash Attention** 的 `varlen` 模式处理变长序列，避免 Padding 计算浪费。

### 7.2 推理时的 Mask

**Prefill 阶段：** 与训练类似，需要因果掩码（无 Padding 掩码，因 Prompt 通常无 Padding）。

**Decode 阶段：** 每步只有 1 个新 token，KV Cache 中的历史 K/V 默认全部有效，**不需要 Causal Mask**（新 token 天然只有历史 K/V，不存在未来信息泄露问题）。Mask 逻辑退化为对 Padding 位置的屏蔽（若 KV Cache 中存在 Padding）。

---

## 8. 计算强度（Arithmetic Intensity）差异

### 8.1 训练阶段（GEMM 主导）

训练时 batch size $B$ 通常为 $16 \sim 512$，序列长度 $T = 512 \sim 4096$，注意力和 FFN 均为大矩阵乘：

**FFN 前向（单层，$\mathbf{X} \in \mathbb{R}^{BT \times d}$，$\mathbf{W}_1 \in \mathbb{R}^{d \times d_{ff}}$）：**

$$ \text{FLOP} = 2 \cdot BT \cdot d \cdot d_{ff}, \quad \text{Bytes} = (BT \cdot d + d \cdot d_{ff} + BT \cdot d_{ff}) \times \text{dtype} $$

当 $BT \gg d, d_{ff}$ 时，Arithmetic Intensity $\to \frac{2 \cdot BT \cdot d \cdot d_{ff}}{d \cdot d_{ff} \times 2} = BT$，达到数十至数百 FLOP/Byte，远超 GPU Ridge Point（A100: 156 FLOP/Byte），进入 **compute-bound** 区域。

### 8.2 推理 Decode 阶段（GEMV 主导）

Decode 时 $B=1$（或极小），单步 FFN：

$$ \mathbf{h} = \mathbf{x} \mathbf{W}_1, \quad \mathbf{x} \in \mathbb{R}^{1 \times d}, \quad \mathbf{W}_1 \in \mathbb{R}^{d \times d_{ff}} $$

$$ \text{FLOP} = 2 d d_{ff}, \quad \text{Bytes（读权重）} = d \cdot d_{ff} \times 2 $$

$$ \text{AI} = \frac{2 d d_{ff}}{2 d d_{ff}} = 1 \text{ FLOP/Byte} \ll 156 \text{ FLOP/Byte（Ridge Point）} $$

GPU 算力利用率 $\approx 1/156 < 1\%$，完全受 **HBM 带宽** 约束。

**提升 AI 的方法：增大有效 batch size。** 推理框架的核心目标之一即在延迟约束下尽可能提高 batch size：

$$ \text{AI}_{\text{decode}} = B \text{ FLOP/Byte（近似）} $$

当 $B = 156$ 时才达到 A100 的 Ridge Point。

---

## 9. 归一化融合（Norm Fusion）与推理专项优化

### 9.1 训练阶段的约束

训练时归一化层需要保存前向统计量（$\mu$, $\sigma$）用于反向传播梯度计算，**不能随意融合**或修改计算图。

### 9.2 推理阶段的 Kernel Fusion 机会

推理无反向传播，可将多个算子融合为单个 CUDA Kernel，消除中间结果的 HBM 读写：

**典型融合链：**

```
RMSNorm → Linear（QKV 投影）
```

融合前：RMSNorm 输出写 HBM → Linear 读 HBM，2次 HBM 访问 融合后：RMSNorm 输出保留在寄存器 → 直接输入 Linear，1次 HBM 访问

**典型融合收益（A100，$d=4096$，$T=1$）：**

|操作|未融合 HBM 访问|融合后 HBM 访问|
|---|---|---|
|RMSNorm + QKV 投影|$3 \times 4096 \times 2$ B|$4096 \times 2$ B|
|SwiGLU + W2 投影|$2 \times 4096 \times 4 \times 2$ B|$4096 \times 2$ B|

### 9.3 权重吸收（Weight Absorption）

RMSNorm 后接 Linear 时，推理阶段可将归一化缩放参数 $\boldsymbol{\gamma}$ 直接融入权重矩阵：

$$ \text{RMSNorm}(\mathbf{x}) \mathbf{W} = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \boldsymbol{\gamma} \cdot \mathbf{W} = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \cdot \mathbf{W}' $$

其中 $\mathbf{W}' = \text{diag}(\boldsymbol{\gamma}) \mathbf{W}$（在模型加载时预计算一次）。

这消除了 $\boldsymbol{\gamma}$ 的逐元素乘操作，**训练阶段不能这样做**（$\boldsymbol{\gamma}$ 需要独立更新梯度）。

---

## 10. 量化：训练与推理的不同用途

### 10.1 训练阶段的量化

训练量化（Quantization-Aware Training, QAT）在训练过程中模拟低精度推理，同时维持 FP32 梯度计算：

$$ \hat{W} = \text{round}!\left(\frac{W}{s}\right) \cdot s, \quad s = \frac{\max(|W|)}{2^{b-1}-1} $$

反向传播时使用 **Straight-Through Estimator（STE）** 跳过不可微的量化操作：

$$ \frac{\partial \mathcal{L}}{\partial W} \approx \frac{\partial \mathcal{L}}{\partial \hat{W}} $$

### 10.2 推理阶段的 PTQ（Post-Training Quantization）

推理使用离线量化，无需反向传播，重点在于最小化量化误差对精度的影响：

|方案|量化目标|推理收益|推理框架支持|
|---|---|---|---|
|GPTQ|权重（逐层最优化）|W4 权重，减少 HBM 带宽|vLLM, TGI|
|AWQ|权重（激活感知缩放）|W4，保护显著权重|vLLM, TGI|
|SmoothQuant|权重+激活|W8A8，提升 GEMM 吞吐|TensorRT-LLM|
|FP8|权重+激活|Hopper FP8 Tensor Core|TensorRT-LLM, vLLM|

**推理量化对训练阶段的依赖：** GPTQ/AWQ 需要少量校准数据（Calibration Set）做一次性离线分析，属于轻量级数据依赖，与完整训练流程无关。

---

## 11. 批处理策略差异

### 11.1 训练：静态批处理

训练 batch 在迭代开始前固定，序列长度通过 Padding 对齐至固定形状，以支持高效的矩阵乘法。

$$ \mathbf{X}_{\text{batch}} \in \mathbb{R}^{B \times T_{\max} \times d} $$

Padding token 的梯度通过 Mask 置零（不影响 Loss），但仍占用计算和显存资源。

### 11.2 推理：动态批处理与连续批处理

**Static Batching（朴素）：** 等待一个 batch 内所有请求全部完成才接收新请求，GPU 在长请求尾部大量空转。

**Continuous Batching（连续批处理）：** 当某个序列完成生成（遇到 EOS token）时，立即将该 slot 分配给新的等待请求，无需等待整个 batch。

```
时间轴 →
Batch Slot 0: [seq_A: prefill → decode → decode → EOS] [seq_D: prefill → ...]
Batch Slot 1: [seq_B: prefill → decode → EOS] [seq_C: prefill → ...]
              ↑ seq_B 完成即刻插入 seq_C，不等 seq_A
```

**Chunked Prefill：** 将长 Prompt 的 Prefill 切分为多个 chunk，与 Decode 请求交错执行，平衡 Prefill（compute-bound）和 Decode（memory-bound）的 GPU 利用率。

---

## 12. 数值精度策略差异

|精度方案|训练场景|推理场景|
|---|---|---|
|FP32|优化器状态、梯度累积|不使用（显存浪费）|
|BF16|前向/后向传播（主流）|权重+激活存储（主流）|
|FP16|前向/后向传播（较旧方案，需 Loss Scaling）|权重+激活存储|
|INT8/INT4|QAT 模拟量化|PTQ 部署，减少带宽|
|FP8|Hopper 架构训练探索中|Hopper 架构主流方向|

**Loss Scaling（训练特有）：** FP16 的动态范围（$\approx 6.5 \times 10^4$）导致小梯度下溢至零。Loss Scaling 将 Loss 乘以系数 $s$（通常 $2^{10} \sim 2^{16}$）后反向传播，梯度更新前除以 $s$：

$$ \nabla_\theta \mathcal{L} \leftarrow \frac{1}{s} \nabla_\theta (s \cdot \mathcal{L}) $$

推理无梯度计算，Loss Scaling 不存在。

---

## 13. 综合对比：显存分配示意

```
训练显存分配（以 7B 模型为例，BF16+FP32 AMP + Adam）：
┌────────────────────────────────────────────────────────┐
│ BF16 参数        14 GB  ████████████████               │
│ FP32 参数副本    28 GB  ████████████████████████████   │
│ FP32 梯度        28 GB  ████████████████████████████   │
│ Adam 一阶矩      28 GB  ████████████████████████████   │
│ Adam 二阶矩      28 GB  ████████████████████████████   │
│ 激活值（seq=2k） ~10 GB ██████████                     │
│ 合计            ~136 GB                                │
└────────────────────────────────────────────────────────┘

推理显存分配（7B 模型，BF16，batch=1，seq=4096）：
┌────────────────────────────────────────────────────────┐
│ BF16 参数        14 GB  ████████████████               │
│ KV Cache         ~3 GB  ████                           │
│ 激活值（极小）    <1 GB  █                              │
│ 合计             ~17 GB                                │
└────────────────────────────────────────────────────────┘
```

---

## 14. 参考资料

- Rajbhandari et al., _ZeRO: Memory Optimizations Toward Training Trillion Parameter Models_, SC 2020
- Chen et al., _Training Deep Nets with Sublinear Memory Cost_（Activation Checkpointing）, 2016
- Dao et al., _FlashAttention-2_, ICLR 2024
- Shoeybi et al., _Megatron-LM: Training Multi-Billion Parameter Language Models_, 2019
- Frantar et al., _GPTQ: Accurate Post-Training Quantization_, ICLR 2023
- Lin et al., _AWQ: Activation-aware Weight Quantization_, MLSys 2024
- Peng et al., _YaRN: Efficient Context Window Extension of Large Language Models_, 2023
- Yu et al., _Orca: A Distributed Serving System for Transformer-Based Generative Models_（Continuous Batching）, OSDI 2022
