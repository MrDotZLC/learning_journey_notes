**GPTQ （Generative Pre-trained Transformer Quantization）** 是目前大语言模型（LLM）部署领域最主流的**后训练量化（Post-Training Quantization, PTQ）** 算法之一，由 Frantar 等人于 2022 年提出（论文：*GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*，发表于 ICLR 2023）。

核心突破：**利用二阶矩阵信息（Hessian）补偿量化误差，将复杂优化问题转化为高效矩阵运算，使得在数小时内完成百亿参数模型的 INT4 量化成为可能。**

***

## 1. 背景与痛点

### 1.1 RTN（Round-to-Nearest）

直接四舍五入，将浮点权重映射到最近的量化网格点。

- **优点**：极快，无需校准数据。
- **缺点**：LLM 权重敏感度极不均匀，存在"承重墙"式的关键权重，微小扰动可导致输出崩塌。RTN 完全忽略敏感度差异，在 INT4 下（尤其是参数量 $< 10\text{B}$ 的模型）精度损失严重。

### 1.2 OBQ（Optimal Brain Quantization）

基于二阶导数，逐个权重进行最优量化与误差补偿。

- **优点**：精度极高。
- **缺点**：计算复杂度为 $O(d_{\text{row}} \cdot d_{\text{col}}^3)$。对于 175B 参数的 GPT-3，需要数千 GPU 小时，工程上不可行。

### 1.3 GPTQ 的定位

GPTQ 是 OBQ 的**算法简化与工程加速**版本：保留 Hessian 误差补偿的高精度特性，同时通过惰性批量更新与 Cholesky 分解将实际计算开销降低至工程可接受的量级。

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
| $\delta$ | $1 \times d_{\text{in}}$ | 权重扰动行向量，$\delta = \hat{\mathbf{w}} - \mathbf{w}$ |
| $Q(\cdot)$ | — | 量化算子 |
| $\mathbf{H}$ | $d_{\text{in}} \times d_{\text{in}}$ | Hessian 矩阵，$\mathbf{H} = XX^T$ |
| $\mathbf{H}'$ | $d_{\text{in}} \times d_{\text{in}}$ | 阻尼后 Hessian，$\mathbf{H}' = \mathbf{H} + \lambda I$ |

### 2.2 目标函数

对矩阵 $W$ 的某一行 $\mathbf{w}$，寻找量化版本 $\hat{\mathbf{w}}$，使输出误差最小：

$$
\operatorname{argmin}_{\hat{\mathbf{w}}}\; L = \frac{1}{2}\|\mathbf{w}X - \hat{\mathbf{w}}X\|_2^2
$$

### 2.3 泰勒展开与 Hessian 矩阵

令 $\delta = \hat{\mathbf{w}} - \mathbf{w}$，将 $L$ 在 $\mathbf{w}$ 处展开：

$$
L(\mathbf{w} + \delta) \approx \underbrace{L(\mathbf{w})}_{=0} + \underbrace{\nabla L \cdot \delta^T}_{\approx 0} + \frac{1}{2}\,\delta\,\mathbf{H}\,\delta^T
$$

两项消去原因：$L(\mathbf{w})=0$（无量化误差）；$\nabla L \approx 0$（预训练已收敛）。问题化简为：

$$
\min_{\delta}\;\frac{1}{2}\,\delta\,\mathbf{H}\,\delta^T, \quad \mathbf{H} = XX^T
$$

> **注**：若损失函数定义不含 $\frac{1}{2}$ 因子，则 $\mathbf{H} = 2XX^T$。两者在更新公式中等价，因子在分子分母相消。

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

### 3.1 惰性批量更新（Lazy Batch-Updates）

#### 3.1.1 问题根源

OBQ 每量化一个权重，就对整个 $\mathbf{H}^{-1}$（$d \times d$）执行一次 rank-1 更新，总复杂度 $O(d^3)$。在 GPU 上，这等价于执行 $d$ 次微小矩阵操作——每次计算量极低，无法填充 Tensor Core，导致 GPU 长期处于 Memory-Bound 状态，实测利用率不足 $5\%$。

#### 3.1.2 分块思想与合法性

将 $d$ 列划分为大小 $B$（默认 128）的 Block，共 $K = \lceil d/B \rceil$ 个。GPTQ 的核心观察：对 Block $k$ 之后列的 rank-1 更新，可以**批量延迟**到 Block $k$ 全部量化完毕后，以一次矩阵乘法统一执行。

设 Block $k$ 覆盖列 $[kB, (k+1)B)$，处理完后的全局更新为：

$$
W_{:,\,(k+1)B:} \leftarrow W_{:,\,(k+1)B:}
- \underbrace{E}_{d_{\text{out}} \times B}
\cdot \underbrace{(\mathbf{H}^{-1})_{kB:(k+1)B,\,(k+1)B:}}_{B \times (d-(k+1)B)}
$$

其中 $E$ 的第 $j$ 列为 Block $k$ 中第 $j$ 列的误差 $e_j / [\mathbf{H}^{-1}]_{jj}$。

Block 内部（列 $i$ 与同 Block 后续列之间）的更新依然即时执行，保证 Block 内量化的正确性。Block 外的延迟更新是多次 rank-1 操作之和，等价于一次矩阵乘法，**数学上完全等价**。

#### 3.1.3 效率对比

| 操作模式 | 每步类型 | GPU 状态 |
|---|---|---|
| OBQ（逐个更新） | rank-1 更新，$O(d^2)$ FLOP | Memory-Bound，利用率 $< 5\%$ |
| GPTQ（分块更新） | GEMM，$B \times (d - kB)$ | Compute-Bound，Tensor Core 充分利用 |

Block Size $B = 128$ 在实践中兼顾 Block 内误差传播的充分性与 GEMM 规模的有效性。

### 3.2 Cholesky 分解（数值稳定性）

#### 3.2.1 直接求逆的失效场景

$\mathbf{H} = XX^T$ 面临两类数值问题：

- **奇异性**：若校准样本数 $n < d$，$XX^T$ 秩至多为 $n$，不可逆。即使 $n \geq d$，输入特征的线性相关性（如全零 padding token）仍可使 $\mathbf{H}$ 接近奇异。
- **FP16 精度累积误差**：Gauss-Jordan 消元在接近奇异矩阵上产生大量舍入误差，$[\mathbf{H}^{-1}]_{ii}$ 可能出现负值，使更新公式的分母失效。

#### 3.2.2 阻尼处理

引入自适应阻尼项，使 $\mathbf{H}$ 严格正定：

$$
\mathbf{H}' = \mathbf{H} + \lambda I, \quad \lambda = \rho \cdot \frac{\operatorname{tr}(\mathbf{H})}{d}
$$

$\rho$ 为 Damp \%（默认 0.01）。$\lambda$ 与 $\mathbf{H}$ 的平均特征值成比例，避免阻尼在大/小特征值矩阵上过强或过弱。

#### 3.2.3 Cholesky 分解原理

对严格正定矩阵 $\mathbf{H}'$，存在唯一下三角矩阵 $L$（对角元素为正）满足：

$$
\mathbf{H}' = LL^T
$$

递推求解 $L$ 的公式：

$$
l_{ii} = \sqrt{a_{ii} - \sum_{k=1}^{i-1} l_{ik}^2}
$$

$$
l_{ji} = \frac{1}{l_{ii}}\!\left(a_{ji} - \sum_{k=1}^{i-1} l_{jk}\, l_{ik}\right), \quad j > i
$$

由此导出逆矩阵关系：

$$
\mathbf{H}'^{-1} = (LL^T)^{-1} = (L^{-1})^T L^{-1}
$$

GPTQ 不显式构造完整 $\mathbf{H}'^{-1}$，而是只计算 $L^{-1}$，在 Block 更新时通过前向/后向替代（Forward/Backward Substitution）按需计算所需的 $[\mathbf{H}'^{-1}]_{ij}$，避免存储完整 $d \times d$ 矩阵。

#### 3.2.4 Cholesky 分解的优势

| 性质           | 直接求逆                 | Cholesky 分解                 |
| ------------ | -------------------- | --------------------------- |
| **数值稳定性**    | 接近奇异时误差大，需要 Pivoting | 对正定矩阵无条件稳定，无需 Pivoting      |
| **计算复杂度**    | $O(d^3)$             | $O(d^3/3)$，约快 3 倍           |
| **FP16 适用性** | 对角元素可能变负             | $l_{ii}$ 始终为正实数的平方根，FP16 友好 |
| **对称性利用**    | 不利用                  | 仅计算下三角，节省约一半存储              |

### 3.3 完整执行循环

对每一层（Layer）：

1. **收集输入**：以校准数据集跑一遍前向传播，记录该层输入 $X$（维度 $d_{\text{in}} \times n$）。
2. **计算 Hessian**：$\mathbf{H} = XX^T$；加阻尼 $\mathbf{H}' = \mathbf{H} + \lambda I$；对 $\mathbf{H}'$ 做 Cholesky 分解，计算 $L^{-1}$。
3. **分 Block 循环**（共 $K$ 个 Block）：
   - **Block 内**：
     - 读取 $[\mathbf{H}^{-1}]_{ii}$（对角元素）。
	   - 量化当前列权重：$W_{:,i} \to \hat{W}_{:,i}$（对所有行同步量化）。
	   - 计算误差：$\mathbf{err} = (W_{:,i} - \hat{W}_{:,i}) \;/\; [\mathbf{H}^{-1}]_{ii}$。
	   - 误差补偿：$W_{:,\,j} \leftarrow W_{:,\,j} - \mathbf{err} \cdot [\mathbf{H}^{-1}]_{ji}$，对所有后续列 $j > i$ 执行。
   - **Block 后**：将 Block 累计误差矩阵 $E$ 与 $\mathbf{H}'^{-1}$ 的对应子矩阵做矩阵乘，批量更新后续所有列。
4. **存储**：将量化权重打包，按所选格式（INT4/INT3 等）写出。

***

## 4. 关键参数与配置

| 参数 | 建议值 | 功能说明 | 精度影响 |
|------|--------|----------|----------|
| **Bits** | 4-bit | 权重位宽；3-bit 精度损失明显，2-bit 极差 | ⭐⭐⭐⭐⭐ |
| **Group Size** | 128 | 每 128 个权重共享一组 Scale/Zero-point；越小精度越高，存储开销越大 | ⭐⭐⭐⭐ |
| **Act Order**（desc_act） | True | 按激活显著性对列降序排列后再量化，优先保护重要列 | ⭐⭐⭐ |
| **Damp \%** | 0.01 | 阻尼系数 $\rho$，即 $\lambda = \rho \cdot \frac{\operatorname{tr}(\mathbf{H})}{d}$；防止 $\mathbf{H}$ 奇异 | ⭐ |
| **Block Size** | 128 | 惰性批量更新的分块列数；影响 GPU GEMM 规模与精度 | — |

***

## 5. GPTQ vs AWQ

| 特性 | GPTQ | AWQ |
|------|------|-----|
| **核心思想** | 修改权重值：利用 $\mathbf{H}^{-1}$ 将量化误差扩散补偿到其他权重 | 保护显著权重，调整 Scale：识别重要 Channel，对其放大后再量化，等效降低量化误差 |
| **校准集依赖** | 强依赖；校准集分布直接决定 $\mathbf{H}$，影响误差补偿方向 | 依赖；用于统计哪些 Channel 激活值较大（重要） |
| **过拟合风险** | 有（校准集与测试集分布差异大时，补偿方向可能失效） | 较小（不大幅修改权重数值，泛化更稳） |
| **推理性能** | 极快（ExLlamaV2 内核优化成熟） | 极快（AWQ 专用 CUDA 内核） |
| **适用场景** | 通用性强，追求逐层输出误差最小化 | 多模态或跨域任务泛化性稍好 |

***

## 6. 总结

GPTQ 是 LLM 量化的里程碑性工作，本质是一个**有约束的二次规划求解过程**，配合两项工程优化实现了工业级可用性：

$$
\underbrace{\text{量化误差}}_{\text{Hessian 驱动补偿}}
\xrightarrow{\text{Lazy Batch-Updates}}
\underbrace{\text{矩阵乘替代 rank-1 累积}}_{\text{GPU Compute-Bound}}
\xrightarrow{\text{Cholesky}}
\underbrace{\mathbf{H}'^{-1}\text{ 数值稳定}}_{\text{FP16 友好}}
$$

- **一句话概括**：量化权重 $A$ 产生的误差，通过 $\mathbf{H}^{-1}$ 找到最优路径，由权重 $B, C, D, \ldots$ 协同抵消。
- **生产地位**：搭配 ExLlamaV2 推理后端，是当前 4-bit 量化部署的首选方案之一，实现了精度与推理速度的最优权衡。
