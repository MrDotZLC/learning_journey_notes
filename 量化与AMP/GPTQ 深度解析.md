**GPTQ （Generative Pre-trained Transformer Quantization）** 是目前大语言模型（LLM）部署领域最主流的**后训练量化（Post-Training Quantization, PTQ）** 算法之一，由 Frantar 等人于 2022 年提出（论文：*GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*，发表于 ICLR 2023）。

核心突破：**利用二阶矩阵信息（Hessian）补偿量化误差，将复杂优化问题转化为高效矩阵运算，使得在数小时内完成百亿参数模型的 INT4 量化成为可能。**

***

## 1. 背景与痛点

在 GPTQ 出现之前，LLM 量化主要面临两个极端的选择：
### 1.1 RTN（Round-to-Nearest）

直接四舍五入，将浮点权重映射到最近的量化网格点。

- **优点**：极快，无需校准数据。
- **缺点**：LLM 权重敏感度极不均匀，存在"承重墙"式的关键权重，微小扰动可导致输出崩塌。RTN 完全忽略敏感度差异，在 INT4 下（尤其是参数量 $<$ 10B 的模型）精度损失严重。

### 1.2 OBQ（Optimal Brain Quantization）

基于二阶导数，逐个权重进行最优量化与误差补偿。

- **优点**：精度极高。
- **缺点**：计算复杂度为 $O(d_{\text{row}} \cdot d_{\text{col}}^3)$。对于 175B 参数的 GPT-3，需要数千 GPU 小时，工程上不可行。

### 1.3 GPTQ 的定位

GPTQ 是 OBQ 的**算法简化与工程加速**版本：保留 Hessian 误差补偿的高精度特性，同时将计算复杂度降低至与正常推理相当的量级。

***

## 2. 核心数学原理

> 完整逐步推导见：[GPTQ 完整数学推导笔记](GPTQ%20完整数学推导笔记.md)

GPTQ 的数学核心源自 **Optimal Brain Surgeon（OBS）** 理论。优化目标不是"权重数值变化最小"，而是"**层输出结果变化最小**"。

### 2.1 符号定义

| 符号 | 维度 | 含义 |
|------|------|------|
| $W$ | $d_{\text{out}} \times d_{\text{in}}$ | 线性层原始权重矩阵 |
| $\hat{W}$ | $d_{\text{out}} \times d_{\text{in}}$ | 量化后的权重矩阵 |
| $X$ | $d_{\text{in}} \times n$ | 校准数据集的输入矩阵（$n$ 为样本数） |
| $\mathbf{w}$ | $1 \times d_{\text{in}}$ | $W$ 的某一行（逐行独立优化） |
| $Q(\cdot)$ | — | 量化算子 |
| $\mathbf{H}$ | $d_{\text{in}} \times d_{\text{in}}$ | Hessian 矩阵 |

### 2.2 目标函数

对矩阵 $W$ 的某一行 $\mathbf{w}$，寻找量化版本 $\hat{\mathbf{w}}$，使输出误差最小：

$$
\operatorname{argmin}_{\hat{\mathbf{w}}}L = \frac{1}{2}\|\mathbf{w}X - \hat{\mathbf{w}}X\|_2^2
$$

### 2.3 泰勒展开与 Hessian 矩阵

令 $\delta = \hat{\mathbf{w}} - \mathbf{w}$（行向量），将 $L$ 在 $\mathbf{w}$ 处展开：

$$
L(\mathbf{w} + \delta) \approx \underbrace{L(\mathbf{w})}_{=0} + \underbrace{\nabla L \cdot \delta^T}_{\approx 0} + \frac{1}{2}\,\delta\,\mathbf{H}\,\delta^T
$$

两项消去原因：$L(\mathbf{w})=0$（无量化误差）；$\nabla L \approx 0$（预训练已收敛）。问题化简为最小化二阶项：

$$
\min_{\delta}\;\frac{1}{2}\,\delta\,\mathbf{H}\,\delta^T
$$

Hessian 矩阵由校准集输入直接计算得到：

$$
\mathbf{H} = XX^T
$$

> **注**：若损失函数定义不含 $\frac{1}{2}$ 因子，则 $\mathbf{H} = 2XX^T$。两者在更新公式中等价（因子在分子分母中相消）。

> **直观理解**：$\mathbf{H}$ 本质是输入特征的协方差矩阵。某方向特征值越大，权重在该方向上越敏感，微小扰动对输出影响越大。

### 2.4 最优更新公式

量化第 $i$ 个权重 $w_i \to Q(w_i)$ 时，对同行剩余未量化权重的最优补偿为：

$$
\Delta\mathbf{w}_{\text{remaining}}
= -\frac{w_i - Q(w_i)}{[\mathbf{H}^{-1}]_{ii}} \cdot (\mathbf{H}^{-1})_{i,\,i:}
$$

**各项含义**：

- $w_i - Q(w_i)$：第 $i$ 个权重的量化误差。
- $[\mathbf{H}^{-1}]_{ii}$：$\mathbf{H}^{-1}$ 的第 $(i,i)$ 对角元素，作为归一化缩放因子。
- $(\mathbf{H}^{-1})_{i,\,i:}$：$\mathbf{H}^{-1}$ 第 $i$ 行中**从第 $i$ 列起**的子向量，描述 $w_i$ 与剩余未量化权重的耦合关系。

**结论**：量化一个权重产生的误差，应按 $\mathbf{H}^{-1}$ 第 $i$ 行的比例"分摊"到其他尚未量化的权重，使整层输出保持统计一致。

***

## 3. 算法流程：从数学到工程

直接对大矩阵逐个权重求逆并更新，开销无法接受。GPTQ 引入三项工程优化：

### 3.1 惰性批量更新（Lazy Batch-Updates）

OBQ 每量化一个权重就更新整个 $\mathbf{H}^{-1}$。GPTQ 改为**分块（Block）处理**：

1. 将列分为每 128 列一组（Block Size = 128）。
2. Block 内部：使用更新公式补偿，但仅更新 Block 内部权重，暂不更新 Block 外的大矩阵。
3. 一个 Block 处理完毕后，统一进行一次全局矩阵更新。

该策略大幅提升了 GPU 的计算密度（Arithmetic Intensity），减少了显存带宽瓶颈。

### 3.2 Cholesky 分解（数值稳定性）

$H = XX^T$ 在实际计算中可能出现奇异性（不可逆），且半精度（FP16）下直接求逆数值不稳定。GPTQ 的处理方式：

1. 加阻尼项（Damping）：$\mathbf{H}' = \mathbf{H} + \lambda I$（$\lambda$ 为小正数，保证正定）。
2. 对 $\mathbf{H}'$ 做 **Cholesky 分解**，从分解结果中高效、稳定地计算所需的 $\mathbf{H}^{-1}$ 子项。

### 3.3 执行循环

对每一层（Layer）：

1. **收集输入**：以校准数据集跑一遍前向传播，记录该层输入 $X$（维度 $d_{\text{in}} \times n$）。
2. **计算 Hessian**：$\mathbf{H} = XX^T$，加阻尼后做 Cholesky 分解。
3. **逐列循环**（在 Block 内）：
   - 读取 $[\mathbf{H}^{-1}]_{ii}$（对角元素）。
   - 量化当前列权重：$W_{:,i} \to \hat{W}_{:,i}$（对所有行同步量化）。
   - 计算误差：$\mathbf{err} = (W_{:,i} - \hat{W}_{:,i}) \;/\; [\mathbf{H}^{-1}]_{ii}$。
   - 误差补偿：$W_{:,\,j} \leftarrow W_{:,\,j} - \mathbf{err} \cdot [\mathbf{H}^{-1}]_{ji}$，对所有 $j > i$ 执行。
1. **存储**：将量化权重打包，按所选格式（INT4/INT3 等）写出。

***

## 4. 关键参数与配置

| 参数 | 建议值 | 功能说明 | 精度影响 |
|------|--------|----------|----------|
| **Bits** | 4-bit | 权重位宽；3-bit 精度损失明显，2-bit 极差 | ⭐⭐⭐⭐⭐ |
| **Group Size** | 128 | 每 128 个权重共享一组 Scale/Zero-point；越小精度越高，存储开销越大 | ⭐⭐⭐⭐ |
| **Act Order**（desc_act） | True | 按激活显著性对列降序排列后再量化，优先保护重要列 | ⭐⭐⭐ |
| **Damp \%** | 0.01 | 阻尼系数 $\lambda$，即 $\lambda = 0.01 \cdot \text{mean}(\text{diag}(H))$；防止 $\mathbf{H}$ 奇异 | ⭐ |

***

## 5. GPTQ vs AWQ

| 特性 | GPTQ | AWQ |
|------|------|-----|
| **核心思想** | **修改权重值**：利用 $\mathbf{H}^{-1}$ 将量化误差扩散补偿到其他权重 | **保护显著权重，调整 Scale**：识别重要 Channel，对其放大后再量化，等效于降低量化误差 |
| **校准集依赖** | 强依赖；校准集分布直接决定 $\mathbf{H}$，影响误差补偿方向 | 依赖；用于统计哪些 Channel 激活值较大（重要） |
| **过拟合风险** | 有（校准集与测试集分布差异大时，补偿方向可能失效） | 较小（不大幅修改权重数值，泛化更稳） |
| **推理性能** | 极快（ExLlamaV2 内核优化成熟） | 极快（AWQ 专用 CUDA 内核） |
| **适用场景** | 通用性强，追求逐层输出误差最小化 | 多模态或跨域任务泛化性稍好 |

***

## 6. 总结

GPTQ 是 LLM 量化的里程碑性工作，本质是一个**有约束的二次规划求解过程**：

$$
\text{量化误差}\xrightarrow{\;\mathbf{H}^{-1}\text{ 第 }i\text{ 行}\;}\text{分摊到剩余权重}\;\Rightarrow\;\text{层输出误差最小}
$$

- **一句话概括**：量化权重 $A$ 产生的误差，通过 $\mathbf{H}^{-1}$ 找到最优路径，由权重 $B, C, D, \ldots$ 协同抵消。
- **生产地位**：搭配 ExLlamaV2 推理后端，是当前 4-bit 量化部署的首选方案之一，实现了精度与推理速度的最优权衡。
