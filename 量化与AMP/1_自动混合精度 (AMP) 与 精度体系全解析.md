## 一、 核心矛盾：精度 vs. 速度
在深度学习中，我们始终在做平衡：
- **高精度 (FP32)：** 算得准，但吃显存、跑得慢。
- **低精度 (FP16/BF16)：** 跑得极快、省显存，但容易“练废”（梯度消失或溢出）。
**AMP (Automatic Mixed Precision)** 的本质：在模型运算的各个环节**按需分配精度**——计算密集型用低精度提速，精度敏感型保留高精度稳盘。

---

## 二、 精度全家桶：AMP 的“原材料”
了解 AMP 之前，必须先看懂 GPU 里的数据格式。

|**格式**|**总位数**|**指数位 (Range)**|**尾数位 (Precision)**|**主要用途**|**硬件支持**|
|---|---|---|---|---|---|
|**FP32**|32 bit|8 bit|23 bit|训练基准、权重存储|所有 CPU/GPU|
|**TF32**|19 bit|8 bit|10 bit|**NVIDIA 专属**训练加速|Ampere (RTX 30/A100) 及以上|
|**FP16**|16 bit|5 bit|10 bit|传统 AMP 训练、推理|Volta (V100) 及以上|
|**BF16**|16 bit|8 bit|7 bit|**大模型(LLM)训练首选**|Ampere (A100) 及以上|
|**FP8**|8 bit|4-5 bit|2-3 bit|最新极速训练与推理|Hopper (H100) 及以上|
|**INT8**|8 bit|0 bit|8 bit|量化推理 (Deployment)|几乎所有现代硬件|

> **核心逻辑：** 指数位决定了数能表示多大（防止溢出），尾数位决定了数有多精细（防止舍入误差）。

### 2.1 TF32 (TensorFloat-32)：无痛加速的“黑科技”
TF32 是 NVIDIA 为了让用户“不改代码就能提速”搞出来的。
- **原理：** 它在内部计算时，保留了 FP32 的**指数位**（保证动态范围一样大），但把**尾数位**截断到了跟 FP16 一样。
- **优点：** 你的代码里依然写的是 `dtype=float32`，但 GPU 内部会偷偷用 TF32 加速矩阵运算。它比真正的 FP32 快得多，且不需要像 FP16 那样做 Loss Scaling。
### 2.2 BF16 (Brain Floating Point)：大模型的救星
由 Google Brain 开发，现在已经成为大语言模型（如 Llama, GPT 系列）的标配。
- **为什么它比 FP16 好？** FP16 最蛋疼的地方是指数位太短（只有 5 位），容易溢出。BF16 把指数位拉到了 8 位（和 FP32 一样），这意味着它**不需要 Loss Scaling** 就能训练得很稳。
- **代价：** 它的尾数位更短，精度比 FP16 低，但在深度学习这种容错率高的领域，范围（Range）往往比精度（Precision）更重要。
### 2.3 FP8：Hopper 架构的杀手锏
随着 H100 的普及，FP8 开始大放异彩。它分为两种模式：
- **E4M3：** 4 位指数，3 位尾数。精度高一点，适合**前向传播**。
- **E5M2：** 5 位指数，2 位尾数。范围广一点，适合**反向传播**。
- **意义：** 在保证模型不崩的前提下，把计算吞吐量又翻了一倍，显存占用再减半。
### 2.4 INT8 / INT4：极致的压缩
这些是**定点数（Integer）**。
- **场景：** 几乎不用于训练（因为没法表达微小的梯度更新），但在模型**部署（Inference）**阶段是神器。
- **效果：** 通过量化（Quantization）技术，可以将一个 175B 的模型压缩到单张显卡都能跑的程度，虽然会有一点精度损失，但速度和显存收益巨大。

---

## 三、 AMP 的三大支柱（运行机制）
AMP 并不是简单地把所有东西减半，它靠以下三招撑起强度：
### 1. 自动转型 (Autocasting)
AMP 会自动将算子分类：
- **白名单 (FP16)：** `Linear`, `Conv`。这些矩阵乘法在 Tensor Cores 上有巨大加成。
- **黑名单 (FP32)：** `Softmax`, `LayerNorm`, `Loss`。这些涉及指数运算或累加，用 FP16 必崩。
### 2. 主权重备份 (Master Weights)
为了防止 FP16 在权重更新时因为步长太小被“抹零”，AMP 在内存中维护一份 **FP32 的权重副本**。
- **前向+反向：** 使用 FP16。
- **更新：** 将梯度应用到 FP32 主权重上，再同步回 FP16。
### 3. 损失缩放 (Loss Scaling) —— 针对 FP16 的特效药
由于 FP16 范围太窄，微小的梯度常会变成 0（Underflow）。
- **缩放：** 把 Loss 乘以一个大因子 $S$（如 1024），把梯度“顶”进 FP16 的表示范围。
- **还原：** 在更新权重前，除以 $S$ 还原真实梯度。
    
$$Grad_{final} = \frac{Grad_{scaled}}{Scale}$$

---

## 四、 硬件选型与实战建议
### 1. 显卡支持对照表
- **V100 (Volta):** 支持 FP16，**必须使用 Loss Scaling**。
- **A100 / RTX 30 系列 (Ampere):** 支持 **BF16** 和 **TF32**。强烈建议用 BF16，可以告别 Loss Scaling 的烦恼。
- **H100 (Hopper):** 支持 **FP8**。开启后训练效率能再翻一倍。
### 2. PyTorch 代码模板
Python
```
scaler = torch.amp.GradScaler('cuda') # 仅 FP16 需要，BF16 可省略
for input, target in data_loader:
    optimizer.zero_grad()
    
    # 自动切换精度
    with torch.amp.autocast('cuda', dtype=torch.float16): # 或 torch.bfloat16
        output = model(input)
        loss = criterion(output, target)
    # 缩放、反向传播、更新
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 五、 总结：为什么要用 AMP？
1. **显存翻倍：** 原本只能跑 Batch Size 8，现在能跑 16 甚至 32。
2. **速度起飞：** 配合 Tensor Cores，训练耗时通常减半。
3. **几乎无损：** 最终模型的 Accuracy 与全 FP32 相比基本一致。
