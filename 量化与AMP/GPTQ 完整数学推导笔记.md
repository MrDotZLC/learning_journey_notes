## 1. 符号约定

| 符号 | 维度 | 含义 |
|------|------|------|
| $\mathbf{w}$ | $1 \times d$ | 权重矩阵中的**某一行**（行向量） |
| $X$ | $d \times n$ | 校准集输入矩阵，$n$ 为样本数，$d$ 为输入特征维度 |
| $\hat{\mathbf{w}}$ | $1 \times d$ | 量化后的对应行权重（行向量） |
| $\delta$ | $1 \times d$ | 权重扰动行向量，$\delta = \hat{\mathbf{w}} - \mathbf{w}$ |
| $\mathbf{H}$ | $d \times d$ | Hessian 矩阵，$\mathbf{H} = XX^T$ |
| $Q(\cdot)$ | — | 量化算子，将浮点数映射到最近的量化网格点 |
| $\mathbf{e}_i$ | $d \times 1$ | 第 $i$ 个标准基向量（第 $i$ 位为 1，其余为 0） |
| $[\mathbf{H}^{-1}]_{ii}$ | 标量 | $\mathbf{H}^{-1}$ 的第 $i$ 个对角线元素 |
| $(\mathbf{H}^{-1})_{i,:}$ | $1 \times d$ | $\mathbf{H}^{-1}$ 的第 $i$ 行（行向量） |
| $B$ | 标量 | Block Size，惰性批量更新的分块列数（典型值 128） |
| $L$ | $d \times d$ | Cholesky 下三角因子，满足 $\mathbf{H}' = LL^T$ |

> **作用域说明**：GPTQ 对权重矩阵 $W$（$d_{\text{out}} \times d_{\text{in}}$）**逐行独立**执行优化。下文所有推导均针对固定的**某一行** $\mathbf{w}$，不涉及行间耦合。

***

## 2. 核心目标：最小化输出扰动

对 Transformer 中某一线性层，原始权重行向量为 $\mathbf{w}$，量化后为 $\hat{\mathbf{w}}$。

定义该行对应的输出误差函数：

$$
L = \frac{1}{2} \| \mathbf{w}X - \hat{\mathbf{w}}X \|_2^2
$$

目标：寻找 $\hat{\mathbf{w}}$，使 $L$ 最小，即让量化前后的层输出尽可能一致。

***

## 3. 泰勒展开与二阶项转化

令 $\delta = \hat{\mathbf{w}} - \mathbf{w}$（行向量，维度 $1 \times d$），则 $L(\mathbf{w} + \delta)$ 在 $\mathbf{w}$ 处作泰勒展开：

$$
L(\mathbf{w} + \delta) \approx \underbrace{L(\mathbf{w})}_{=\,0} + \underbrace{\nabla_{\mathbf{w}} L \cdot \delta^T}_{\approx\,0} + \frac{1}{2}\, \delta\, \mathbf{H}\, \delta^T
$$

**三项消去理由**：

1. $L(\mathbf{w}) = 0$：原始权重对自身输出无量化误差，损失为零。
2. $\nabla_{\mathbf{w}} L \approx 0$：预训练已收敛，权重处于（局部）极小值，一阶梯度近似为零。
3. **结论**：优化目标退化为最小化二阶曲率项：

$$
\min_{\delta}\; \frac{1}{2}\, \delta\, \mathbf{H}\, \delta^T
$$

***

## 4. Hessian 矩阵 $\mathbf{H}$ 的详细推导

### 4.1 从范数到二次型

误差函数关于扰动 $\delta$（维度 $1 \times d$）展开：

$$
f(\delta) = \frac{1}{2} \| \delta X \|_2^2
$$

利用 $\|v\|_2^2 = v v^T$（此处 $v = \delta X$，维度 $1 \times n$）：

$$
f(\delta) = \frac{1}{2}\, (\delta X)(\delta X)^T
$$

### 4.2 展开为标准二次型

利用转置规则 $(AB)^T = B^T A^T$，有 $(\delta X)^T = X^T \delta^T$：

$$
f(\delta)
= \frac{1}{2}\;
\underbrace{\delta}_{1 \times d}\;
\underbrace{(X X^T)}_{d \times d}\;
\underbrace{\delta^T}_{d \times 1}
$$

### 4.3 读出 Hessian

标准二次型形如 $\frac{1}{2}\,\delta\,\mathbf{H}\,\delta^T$，对比得：

$$
\boxed{\mathbf{H} = X X^T}
$$

> **因子说明**：若损失函数定义不含 $\frac{1}{2}$（即 $L = \|\delta X\|_2^2$），则 $\mathbf{H} = 2XX^T$。本文采用含 $\frac{1}{2}$ 的标准化定义，故 $\mathbf{H} = XX^T$。两种形式在最终更新公式中完全等价，因子在分子分母同时出现时相消。

> **直观理解**：$\mathbf{H}$ 本质是输入特征的**协方差矩阵**（未中心化）。某方向特征值越大，权重在该方向上越敏感，微小扰动对输出影响越大。

***

## 5. 引入约束：逐个权重量化

量化第 $i$ 个权重 $w_i$ 时，产生强制约束：

$$
\delta w_i \;\triangleq\; Q(w_i) - w_i \quad \text{（已知标量，由量化算子决定）}
$$

用标准基向量表达该约束（选出 $\delta$ 的第 $i$ 个分量等于 $\delta w_i$）：

$$
\mathbf{e}_i^T \delta^T = \delta w_i
\quad\Longleftrightarrow\quad
\delta_i = \delta w_i
$$

> **符号说明**：$\mathbf{e}_i$ 为 $d \times 1$ 列向量；$\delta^T$ 为 $d \times 1$ 列向量；$\mathbf{e}_i^T \delta^T$ 为标量，等价于取 $\delta$ 的第 $i$ 个分量。

完整带约束优化问题：

$$
\min_{\delta}\; \frac{1}{2}\, \delta\, \mathbf{H}\, \delta^T
\quad \text{s.t.}\quad \delta_i = \delta w_i
$$

***

## 6. 拉格朗日乘数法求解补偿量

### 6.1 构造拉格朗日函数

$$
\mathcal{L}(\delta, \lambda)
= \frac{1}{2}\, \delta\, \mathbf{H}\, \delta^T
- \lambda\bigl(\mathbf{e}_i^T \delta^T - \delta w_i\bigr)
$$

### 6.2 对 $\delta^T$ 求偏导

将 $\mathcal{L}$ 视为关于列向量 $\delta^T$（维度 $d \times 1$）的函数，对其求偏导：

$$
\frac{\partial \mathcal{L}}{\partial \delta^T}
= \mathbf{H}\, \delta^T - \lambda\, \mathbf{e}_i
= \mathbf{0}
$$

**推导细节**：

- $\frac{\partial}{\partial \delta^T}\!\left(\frac{1}{2}\delta\mathbf{H}\delta^T\right) = \mathbf{H}\delta^T$（对称矩阵二次型的标准求导结果）。
- $\frac{\partial}{\partial \delta^T}\!\left(\lambda\,\mathbf{e}_i^T \delta^T\right) = \lambda\,\mathbf{e}_i$（线性项求导）。

令偏导为零，解出最优 $\delta^T$：

$$
\mathbf{H}\, \delta^T = \lambda\, \mathbf{e}_i
\implies
\delta^T = \lambda\, \mathbf{H}^{-1} \mathbf{e}_i
$$

### 6.3 解出拉格朗日乘子 $\lambda$

将 $\delta^T = \lambda\, \mathbf{H}^{-1} \mathbf{e}_i$ 代入约束条件 $\mathbf{e}_i^T \delta^T = \delta w_i$：

$$
\mathbf{e}_i^T \bigl(\lambda\, \mathbf{H}^{-1} \mathbf{e}_i\bigr) = \delta w_i
$$

注意 $\mathbf{e}_i^T \mathbf{H}^{-1} \mathbf{e}_i$ 等价于取 $\mathbf{H}^{-1}$ 的第 $(i,i)$ 元素：

$$
\mathbf{e}_i^T \mathbf{H}^{-1} \mathbf{e}_i = [\mathbf{H}^{-1}]_{ii}
$$

因此：

$$
\lambda\, [\mathbf{H}^{-1}]_{ii} = \delta w_i
\implies
\lambda = \frac{\delta w_i}{[\mathbf{H}^{-1}]_{ii}}
$$

### 6.4 代回得完整补偿向量

$$
\delta^T = \frac{\delta w_i}{[\mathbf{H}^{-1}]_{ii}}\, \mathbf{H}^{-1} \mathbf{e}_i
$$

注意 $\mathbf{H}^{-1} \mathbf{e}_i$ 等于取 $\mathbf{H}^{-1}$ 的**第 $i$ 列**（$d \times 1$ 列向量），
转置为行向量形式（与 $\delta$ 维度一致）：

$$
\boxed{
\delta = \frac{\delta w_i}{[\mathbf{H}^{-1}]_{ii}}\; (\mathbf{H}^{-1})_{i,\,:}
}
$$

其中 $(\mathbf{H}^{-1})_{i,\,:}$ 为 $\mathbf{H}^{-1}$ 的**第 $i$ 行**（行向量，维度 $1 \times d$）。

***

## 7. GPTQ 权重更新法则

量化第 $i$ 个权重时（$\delta w_i = Q(w_i) - w_i$），对**同行中所有剩余未量化权重**的最优补偿为：

$$
\boxed{
\Delta\mathbf{w}_{\text{remaining}}
= -\frac{w_i - Q(w_i)}{[\mathbf{H}^{-1}]_{ii}}\; (\mathbf{H}^{-1})_{i,\,i:}
}
$$

| 项                            | 含义                                                                 |
| ---------------------------- | ------------------------------------------------------------------ |
| $w_i - Q(w_i)$               | 第 $i$ 个权重的量化误差（正号，取负后为补偿方向）                                        |
| $[\mathbf{H}^{-1}]_{ii}$     | 归一化因子；对角元素越大，第 $i$ 个权重越"孤立"，误差越难扩散                                 |
| $(\mathbf{H}^{-1})_{i,\,i:}$ | $\mathbf{H}^{-1}$ 第 $i$ 行中**从第 $i$ 列起**的子向量；仅更新尚未量化的位置（$j \geq i$） 

> **物理直觉**：$\mathbf{H}^{-1}$ 第 $i$ 行描述了权重 $w_i$ 与其他权重在输出空间中的耦合关系。系 数越大的位置，该权重的量化误差对对应位置的影响越强，应分摊更多补偿量。

***

## 8. 惰性批量更新（Lazy Batch-Updates）

### 8.1 问题背景

OBQ 的做法是：每量化一个权重 $w_i$，就立刻对整个 $\mathbf{H}^{-1}$（$d \times d$）做一次 rank-1 更新。设权重总数为 $d$，则总更新次数为 $d$，每次更新复杂度 $O(d^2)$，总体复杂度 $O(d^3)$。对 $d \sim 4096$ 的线性层，这意味着 $\sim 6.9 \times 10^{10}$ 次浮点操作，在 GPU 上极其低效——每次 rank-1 更新的计算量极小，无法有效填充 CUDA Core，大部分时间用于显存读写（Memory-Bound 瓶颈）。

### 8.2 分块思想

GPTQ 的关键洞察：**Block 内的权重更新可以延迟到整个 Block 处理完毕后统一执行，而不影响最终结果的正确性。**

将 $d$ 列划分为若干大小为 $B$（默认 128）的 Block，设共 $K = \lceil d / B \rceil$ 个 Block，第 $k$ 个 Block 覆盖列 $[kB, (k+1)B)$。

### 8.3 两阶段执行

**阶段一：Block 内部量化（局部更新）**

对 Block $k$ 内的每一列 $i \in [kB, (k+1)B)$：

1. 读取 $[\mathbf{H}^{-1}]_{ii}$ 及第 $i$ 行的 Block 内子向量 $(\mathbf{H}^{-1})_{i,\,i:(k+1)B}$。
2. 量化：$\hat{w}_i \leftarrow Q(w_i)$，计算误差 $e_i = w_i - \hat{w}_i$。
3. 仅更新**同一 Block 内**尚未量化的权重（列 $j \in (i, (k+1)B)$）：

$$
w_j \leftarrow w_j - \frac{e_i}{[\mathbf{H}^{-1}]_{ii}} \cdot [\mathbf{H}^{-1}]_{ij}
$$

**阶段二：全局更新（Block 间传播）**

当 Block $k$ 的所有列均量化完毕后，将该 Block 的累计误差矩阵 $E \in \mathbb{R}^{B}$（每列的误差 $e_i / [\mathbf{H}^{-1}]_{ii}$ 堆叠而成）传播到后续所有未量化列（列 $j \geq (k+1)B$）：

$$
W_{:,\,j} \leftarrow W_{:,\,j} - \sum_{i \in \text{Block}_k} \frac{e_i}{[\mathbf{H}^{-1}]_{ii}} \cdot [\mathbf{H}^{-1}]_{ij}
$$

写成矩阵形式（对整个 Block 之后的子矩阵一次性更新）：

$$
W_{:,\,(k+1)B:} \leftarrow W_{:,\,(k+1)B:}
- \underbrace{E}_{d_{\text{out}} \times B}
\cdot \underbrace{(\mathbf{H}^{-1})_{kB:(k+1)B,\,(k+1)B:}}_{B \times (d - (k+1)B)}
$$

其中 $E$ 的第 $j$ 列为 $e_j / [\mathbf{H}^{-1}]_{jj}$。

### 8.4 为何可以延迟更新

设 Block $k$ 内已量化了列 $i$，按 OBQ 原始规则，应立刻更新后续所有列（包括 Block $k$ 内剩余列和 Block $k+1$ 之后的列）。GPTQ 的操作等价于：

- Block $k$ 内剩余列：立刻更新（阶段一），保证 Block 内量化顺序的正确性。
- Block $k+1$ 之后的列：**批量延迟**到 Block $k$ 结束时统一更新（阶段二）。

由于阶段一中每次更新均基于最新的 $w_j$ 值，且 Block 内部误差传播是严格正确的，因此延迟 Block 外更新**不影响 Block 外权重的最终补偿结果**——延迟的多次 rank-1 更新之和等于一次矩阵乘法。

### 8.5 计算密度提升

| 操作模式 | 每步计算量 | GPU 利用率 |
|---|---|---|
| OBQ（逐个更新） | $O(d^2)$ rank-1，极小 FLOP | 极低（Memory-Bound） |
| GPTQ（分块更新） | 每 Block 一次 $B \times (d - kB)$ 矩阵乘 | 高（Compute-Bound） |

矩阵乘法（GEMM）是 GPU 最擅长的操作，可完全利用 Tensor Core。Block Size $B = 128$ 在实践中兼顾了精度（Block 内误差传播充分）与效率（矩阵乘规模足够大）。

***

## 9. Cholesky 分解与数值稳定性

### 9.1 直接求逆的问题

GPTQ 更新法则需要 $\mathbf{H}^{-1}$ 的逐行/逐元素访问。直接对 $\mathbf{H} = XX^T$ 求逆面临两个问题：

**问题一：奇异性**  
若校准样本数 $n < d$，则 $XX^T$ 的秩至多为 $n$，矩阵奇异（不可逆）。即使 $n \geq d$，若输入特征存在线性相关性（如 padding token 的全零输入），$\mathbf{H}$ 仍可能接近奇异，逆矩阵数值极不稳定。

**问题二：半精度下数值误差累积**  
在 FP16 精度下，Gauss-Jordan 消元或 LU 分解在接近奇异矩阵上产生大量舍入误差，导致 $\mathbf{H}^{-1}$ 的对角元素 $[\mathbf{H}^{-1}]_{ii}$ 可能出现负值或极大值，使更新公式失效。

### 9.2 阻尼处理

对 $\mathbf{H}$ 添加正则化阻尼项，使其严格正定：

$$
\mathbf{H}' = \mathbf{H} + \lambda I, \quad \lambda = \rho \cdot \frac{1}{d}\operatorname{tr}(\mathbf{H})
$$

其中 $\rho$ 为 Damp \%（默认 0.01），$\frac{1}{d}\operatorname{tr}(\mathbf{H})$ 为 $\mathbf{H}$ 对角元素均值。这一自适应阻尼的好处在于：当 $\mathbf{H}$ 的特征值普遍较大时，$\lambda$ 也相应较大，避免阻尼过小；当特征值较小时，阻尼相对温和，不过度扭曲曲率信息。

阻尼后，$\mathbf{H}'$ 满足严格正定条件，保证 Cholesky 分解可行。

### 9.3 Cholesky 分解原理

**定义**：对任意对称正定矩阵 $A \in \mathbb{R}^{d \times d}$，存在唯一下三角矩阵 $L$（对角元素为正），使得：

$$
A = LL^T
$$

其中 $L$ 称为 $A$ 的 Cholesky 因子。

**逐元素推导（以 $3 \times 3$ 为例）**：

设 $A = LL^T$，展开：

$$
\begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix}
=
\begin{pmatrix} l_{11} & 0 & 0 \\ l_{21} & l_{22} & 0 \\ l_{31} & l_{32} & l_{33} \end{pmatrix}
\begin{pmatrix} l_{11} & l_{21} & l_{31} \\ 0 & l_{22} & l_{32} \\ 0 & 0 & l_{33} \end{pmatrix}
$$

逐行读出方程并求解：

$$
l_{11} = \sqrt{a_{11}}
$$

$$
l_{21} = \frac{a_{21}}{l_{11}}, \quad l_{31} = \frac{a_{31}}{l_{11}}
$$

$$
l_{22} = \sqrt{a_{22} - l_{21}^2}, \quad l_{32} = \frac{a_{32} - l_{31}l_{21}}{l_{22}}
$$

$$
l_{33} = \sqrt{a_{33} - l_{31}^2 - l_{32}^2}
$$

通用递推公式：

$$
l_{ii} = \sqrt{a_{ii} - \sum_{k=1}^{i-1} l_{ik}^2}
$$

$$
l_{ji} = \frac{1}{l_{ii}}\left(a_{ji} - \sum_{k=1}^{i-1} l_{jk} l_{ik}\right), \quad j > i
$$

### 9.4 从 $L$ 到 $\mathbf{H}'^{-1}$

由 $\mathbf{H}' = LL^T$，得：

$$
\mathbf{H}'^{-1} = (LL^T)^{-1} = (L^T)^{-1} L^{-1} = (L^{-1})^T L^{-1}
$$

**GPTQ 的实际做法**：不显式构造完整 $\mathbf{H}'^{-1}$，而是只计算 $L^{-1}$（上三角），然后在 Block 更新时直接用 $L^{-1}$ 的相应行/列元素代替 $\mathbf{H}'^{-1}$ 的对应位置。由于 $\mathbf{H}'^{-1} = (L^{-1})^T L^{-1}$，所需的 $[\mathbf{H}'^{-1}]_{ij}$ 等价于：

$$
[\mathbf{H}'^{-1}]_{ij} = \sum_{k} [L^{-1}]_{ki} [L^{-1}]_{kj}
$$

可通过前向/后向替代（Forward/Backward Substitution）逐列高效计算，无需存储完整 $d \times d$ 矩阵。

### 9.5 为何 Cholesky 优于直接求逆

| 性质 | 直接求逆（Gauss-Jordan） | Cholesky 分解 |
|---|---|---|
| **数值稳定性** | 需要主元选取（Pivoting），接近奇异时误差大 | 对正定矩阵无条件稳定，无需 Pivoting |
| **计算复杂度** | $O(d^3)$ | $O(d^3 / 3)$，约快 3 倍 |
| **结构利用** | 不利用对称性 | 利用对称正定性，只计算下三角 |
| **FP16 适用性** | 差（对角元素可能变负） | 良好（$l_{ii}$ 始终为正实数的平方根） |

***

## 10. 知识点速查

$$
\mathbf{H} = XX^T, \quad \mathbf{H}' = \mathbf{H} + \lambda I, \quad \mathbf{H}' = LL^T
$$

$$
\lambda = \frac{\delta w_i}{[\mathbf{H}^{-1}]_{ii}}, \quad
\delta = \lambda\,(\mathbf{H}^{-1})_{i,\,:}
$$

$$
\Delta\mathbf{w}_{\text{remaining}} = -\frac{w_i - Q(w_i)}{[\mathbf{H}^{-1}]_{ii}}\; (\mathbf{H}^{-1})_{i,\,i:}
$$

- $\mathbf{H}^{-1}$ 对角元素：控制误差扩散强度的归一化因子。
- $\mathbf{H}^{-1}$ 第 $i$ 行：决定误差按比例分摊给其他权重的方向。
- Lazy Batch-Updates：Block 内精确传播，Block 间延迟为矩阵乘，提升 GPU 利用率。
- Cholesky 分解：以 $O(d^3/3)$ 复杂度稳定计算 $\mathbf{H}'^{-1}$，FP16 友好。
