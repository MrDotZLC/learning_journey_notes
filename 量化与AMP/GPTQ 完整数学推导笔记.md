## 1. 符号约定

| 符号                        | 维度           | 含义                                               |
| ------------------------- | ------------ | ------------------------------------------------ |
| $\mathbf{w}$              | $1 \times d$ | 权重矩阵中的**某一行**（行向量）                               |
| $X$                       | $d \times n$ | 校准集输入矩阵，$n$ 为样本数，$d$ 为输入特征维度                     |
| $\hat{\mathbf{w}}$        | $1 \times d$ | 量化后的对应行权重（行向量）                                   |
| $\delta$                  | $1 \times d$ | 权重扰动行向量，$\delta = \hat{\mathbf{w}} - \mathbf{w}$ |
| $\mathbf{H}$              | $d \times d$ | Hessian 矩阵                                       |
| $Q(\cdot)$                | —            | 量化算子，将浮点数映射到最近的量化网格点                             |
| $\mathbf{e}_i$            | $d \times 1$ | 第 $i$ 个标准基向量（第 $i$ 位为 1，其余为 0）                   |
| $[\mathbf{H}^{-1}]_{ii}$  | 标量           | $\mathbf{H}^{-1}$ 的第 $i$ 个对角线元素                  |
| $(\mathbf{H}^{-1})_{i,:}$ | $1 \times d$ | $\mathbf{H}^{-1}$ 的第 $i$ 行                       |

> **作用域说明**：GPTQ 对权重矩阵 $W$（$d_{out} \times d_{in}$）**逐行独立**执行优化。下文所有推导均针对固定的**某一行** $\mathbf{w}$，不涉及行间耦合。

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
L(\mathbf{w} + \delta) \approx \underbrace{L(\mathbf{w})} + \underbrace{\nabla_\mathbf{w} L \cdot \delta^T} + \frac{1}{2}\, \delta\, \mathbf{H}\, \delta^T
$$

**三项的消去理由**：

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

> **直观理解**：$\mathbf{H}$ 本质是输入特征的**协方差矩阵**（未中心化）。$\mathbf{H}$ 在某方向上的特征值越大，权重在该方向上的微小变化对输出的影响越大——即该方向"越敏感"。

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

注意 $\mathbf{e}_i^T \mathbf{H}^{-1} \mathbf{e}_i$ 的含义：从左右两侧用 $\mathbf{e}_i$ "取出" $\mathbf{H}^{-1}$ 的第 $(i,i)$ 元素，即：

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

注意 $\mathbf{H}^{-1} \mathbf{e}_i$ 等于取 $\mathbf{H}^{-1}$ 的**第 $i$ 列**（$d \times 1$ 列向量）：

$$
\mathbf{H}^{-1} \mathbf{e}_i = (\mathbf{H}^{-1})_{:,\,i}
$$

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

**各项含义**：

| 项 | 含义 |
|---|---|
| $w_i - Q(w_i)$ | 第 $i$ 个权重的量化误差（取负号是因为 $\delta w_i = Q(w_i) - w_i$） |
| $[\mathbf{H}^{-1}]_{ii}$ | 归一化因子；对角元素越大，第 $i$ 个权重越"孤立"，误差越难扩散 |
| $(\mathbf{H}^{-1})_{i,\,i:}$ | $\mathbf{H}^{-1}$ 第 $i$ 行中**从第 $i$ 列起**的子向量；仅更新尚未量化的位置（$j \geq i$） |

> **物理直觉**：$\mathbf{H}^{-1}$ 第 $i$ 行描述了权重 $w_i$ 与其他权重在输出空间中的耦合关系。系数越大的位置，该权重的量化误差对对应位置的影响越强，应分摊更多补偿量。

***

## 8. 知识点速查

$$
\mathbf{H} = XX^T \quad\text{（输入协方差，无}\tfrac{1}{2}\text{ 时为 }2XX^T\text{）}
$$

$$
\lambda = \frac{\delta w_i}{[\mathbf{H}^{-1}]_{ii}}, \quad
\delta = \lambda\,(\mathbf{H}^{-1})_{i,\,:}
$$

- $\mathbf{H}^{-1}$ 的**对角元素**：控制误差"扩散强度"的归一化因子。  
- $\mathbf{H}^{-1}$ 的**第 $i$ 行**：决定误差如何按比例分摊给其他权重。  
- 补偿机制的作用：单个权重精度下降，但整层输出的宏观统计特性得以保持。
