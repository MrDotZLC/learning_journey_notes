# AWQ (Activation-aware Weight Quantization) 技术白皮书

# 一、 核心范式：从权重感知到输出空间的转换

在传统的量化策略（如 RTN）中，通常假设权重矩阵 $W$ 中的每个元素对模型精度的贡献是均等的。然而，AWQ 的核心洞察在于：权重的重要性是由其对应的激活值决定的。

## 1.1 现象观察：激活值的“极化”分布

通过可视化 LLM（如 Llama, Qwen）在推理时的中间激活值 $X$，可以观察到激活分布并非均匀分布，而是存在极少数通道（Channels），其数值远大于其他通道。

- 激活离群值（Outliers）：在 99% 的通道数值表现平稳时，约 1% 的通道数值可能达到常规值的 100 倍以上。
    
- 传播效应：在线性层 $Y = WX$ 的计算中，这些大激活值会与对应的权重列相乘。
    

## 1.2 敏感性分析：误差放大效应

由于 $Y = \sum_j w_{\cdot j} x_j$，可以推导出输出对权重的敏感度：

$$\frac{\partial Y}{\partial w_{\cdot j}} = x_j$$

这意味着，激活值 $x_j$ 的模量大小，直接决定了权重误差 $\Delta w$ 对输出误差 $\Delta Y$ 的放大倍数。即使某个权重列本身的量化误差很小，只要它碰巧与一个极大激活值相乘，产生的偏移也会导致整个网络崩塌。

---

# 二、 数学基石：二阶统计量与 Hessian 简化

AWQ 将量化问题形式化为最小化输出重建误差（Reconstruction Error）。

## 2.1 误差能量函数推导

定义量化噪声为 $\epsilon = \hat{W} - W$。输出误差能量 $E$ 为：

$$E = ||\epsilon X||_2^2 = \text{Tr}(\epsilon X X^T \epsilon^T) = \text{Tr}(\epsilon H \epsilon^T)$$

其中 $H = XX^T$ 是输入激活的二阶统计量矩阵（Hessian 矩阵的近似）。

## 2.2 对角化近似的严谨性

直接优化完整的 $H$ 复杂度为 $O(d^3)$，难以扩展。AWQ 提出对角化简化：

$$H \approx \text{diag}(h_1, h_2, \dots, h_d), \quad h_i = E[x_i^2]$$

逻辑支撑：Transformer 的 LayerNorm 弱化了通道间的协方差，且高维分布下非对角项对整体能量贡献较小。由此，问题降维为逐通道加权 MSE 优化：

$$E \approx \sum_i h_i ||\epsilon_i||^2$$

---

# 三、 核心算法：重参数化与 $\alpha$ 搜索机制

AWQ 不通过梯度下降修改权重，而是通过**等价缩放（Equivalent Scaling）**来优化量化器的动态范围。

## 3.1 数学等价性证明

引入缩放因子 $s_j > 1$。为了保持数学结果不变，必须在缩放权重列的同时，反向缩放输入通道：

$$Y = \sum_j (w_{\cdot j} \cdot s_j) \cdot (\frac{x_j}{s_j}) = \sum_j w'_{\cdot j} \cdot x'_j$$

量化后的实际计算为：

$$\hat{Y} = \sum_j \text{Quant}(w_{\cdot j} \cdot s_j) \cdot \frac{x_j}{s_j}$$

## 3.2 缩放因子 $s = s_x^\alpha$ 的边界推导

AWQ 提出启发式构造：$s_j = s_{x, j}^\alpha$，其中 $s_{x, j}$ 为通道激活强度（如 $E[|x|]$）。

- 当 $\alpha = 0$ 时（RTN）：显著通道误差随大激活 $x_j$ 放大而失控。
    
- 当 $\alpha = 1$ 时（完全均衡）：虽然显著权重的相对误差降至最低，但会导致同组（Group）内其他正常权重的量程被拉得过大，从而丢失量化分辨率，引入严重背景噪声。
    
- 结论：最优 $\alpha$ 必然是在“保护显著通道”与“维持全局分辨率”之间寻找平衡点，实验表明 $\alpha \approx 0.5$ 左右效果最佳。
    

---

# 四、 架构适配：攻克 Attention 与 Softmax 陷阱

Attention 层是对量化最敏感的结构。

## 4.1 Softmax 的误差指数放大

Attention 计算涉及 $e^z$。若权重量化导致 $Q$ 或 $K$ 产生微小扰动 $\delta$：

$$e^{(z+\delta)} = e^z \cdot e^\delta \approx e^z \cdot (1 + \delta)$$

由于 Softmax 的“赢家通吃”特性，局部的指数误差会传播到整个概率分布，导致 Attention Map 偏移。

## 4.2 AWQ 的稳定性本质

AWQ 采用误差避免而非误差补偿（如 GPTQ 会修改原始权重数值）。AWQ 保持了权重原有的分布特征，确保了 Logits 空间的稳定，从而在 INT3/INT4 下依然保持 Attention 的收敛性。

---

# 五、 工业部署与性能对比

## 5.1 算法特性对比

|**特性**|**GPTQ**|**AWQ**|
|---|---|---|
|数学原理|完整 Hessian 误差补偿|对角 Hessian 误差避免|
|数值风险|矩阵逆易发散（INT3 风险高）|极稳（不改权重分布）|
|部署开销|需特定排列，计算较杂|零开销（物理融合进算子）|

## 5.2 C++ 推理算子实现 (W4A16 示例)

C++

```
// 伪代码：AWQ W4A16 反量化与 GEMM 融合内核
__global__ void awq_gemm_kernel(
    const uint32_t* __restrict__ quantized_weights,
    const half* __restrict__ activations,
    const half* __restrict__ scales,
    const half* __restrict__ zeros,
    half* __restrict__ output,
    int M, int N, int K) {
    
    // 使用 Tensor Core 进行计算
    // 1. 从全局显存加载 4-bit 权重至寄存器
    // 2. 使用物理融合的 scale 和 zero 进行反量化
    // weight = (quant_weight - zero) * scale
    
    // 示例反量化逻辑
    float weight_f = (static_cast<float>(q_val) - zero_f) * scale_f;
    
    // 3. 执行 mma 指令
}
```

## 5.3 零开销重参数化 (Offline Fusion)

在推理前，缩放因子 $s$ 会被离线物理消除：

- 权重端：直接存储缩放并量化后的权重。
    
- 激活端：将 $1/s$ 融合进前一层的 LayerNorm 或线性算子中。
    
- 硬件加速：配合 W4A16 算子，在显存读取时即时反量化，推理吞吐量通常可提升 2.5x - 3x。
    

---

# 六、 总结：AWQ 的数值稳定性本质

“牺牲部分理论最优性（舍弃非对角项），换取数值鲁棒性。”

AWQ 通过识别并保护 1% 的关键特征，在精度、计算复杂度和硬件友好度之间找到了目前 LLM 部署的“黄金平衡点”。

是否需要我为您生成针对 NVIDIA Ada/Hopper 架构优化后的 AWQ 定制化 Triton Kernel 实现方案？