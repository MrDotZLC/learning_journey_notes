## GPTQ 完整数学推导笔记
### 一、 核心目标：最小化输出扰动
对于 Transformer 中的某一层（线性层），其原始权重为 $W$。量化后的权重为 $\hat{W}$。
我们的目标是：**寻找一个 $\hat{W}$，使得在给定输入采样 $X$ 时，该层输出的均方误差最小。**
定义损失函数 $L$：
$$L = \frac{1}{2} \| WX - \hat{W}X \|_2^2$$

---

### 二、 泰勒展开与二阶项转化
为了研究权重变化对损失的影响，令 $\Delta W = \hat{W} - W$。将 $L$ 在 $W$ 处展开：
$$L(W + \Delta W) \approx L(W) + \nabla L^T \Delta W + \frac{1}{2} \Delta W^T \mathbf{H} \Delta W$$
1. **$L(W) = 0$**：原始权重无量化误差。
2. **$\nabla L \approx 0$**：预训练权重已收敛，处于极小值点附近，斜率为 0。
3. **结论**：问题转化为最小化二阶曲率项 $\frac{1}{2} \Delta W^T \mathbf{H} \Delta W$。

---

### 三、 Hessian 矩阵 $\mathbf{H}$ 的详细推导
对于不熟悉 Hessian 的读者，我们以**单行权重**向量 $\mathbf{w}$ 为例进行推导。
#### 1. 从范数到内积形式
设 $\delta = \mathbf{w} - \hat{\mathbf{w}}$ 为权重变化量向量（$1 \times d$）。输入 $X$ 维度为 $(d \times n)$。
误差函数可写为：
$$f(\delta) = \frac{1}{2} \| \delta X \|_2^2$$
根据 $L_2$ 范数平方等于向量自身内积的性质（$\|v\|_2^2 = vv^T$）：
$$f(\delta) = \frac{1}{2} (\delta X) (\delta X)^T$$
#### 2. 利用转置性质展开
根据矩阵转置规则 $(AB)^T = B^T A^T$，有 $(\delta X)^T = X^T \delta^T$。代入上式：
$$f(\delta) = \frac{1}{2} \underbrace{\delta}_{1 \times d} \underbrace{(X X^T)}_{d \times d} \underbrace{\delta^T}_{d \times 1}$$
#### 3. 定义 Hessian
在多元函数二次型 $f(\delta) = \frac{1}{2} \delta \mathbf{H} \delta^T$ 中，中间的矩阵 $\mathbf{H}$ 即为 Hessian 矩阵。
由此得到：
$$\mathbf{H} = XX^T$$
> **直观理解**：Hessian 矩阵是由输入数据的协方差决定的。它描述了权重空间中每个方向的“敏感度”。

---

### 四、 引入约束：逐个权重量化
在实际量化过程中，我们强制将第 $i$ 个权重 $w_i$ 变为量化值 $Q(w_i)$。
这产生了一个约束：$\mathbf{e}_i^T \delta = \delta w_i$（其中 $\delta w_i = Q(w_i) - w_i$）。
我们的任务是：**调整该行中其他尚未量化的权重，补偿由 $\delta w_i$ 带来的误差。**
数学表达为带约束的最小化问题：
$$\min_{\delta} \frac{1}{2} \delta \mathbf{H} \delta^T \quad \text{s.t. } \mathbf{e}_i^T \delta = \delta w_i$$
---
### 五、 利用拉格朗日乘数法求解补偿量
构造拉格朗日函数：
$$\mathcal{L}(\delta, \lambda) = \frac{1}{2} \delta \mathbf{H} \delta^T - \lambda (\mathbf{e}_i^T \delta - \delta w_i)$$
1. **求偏导令为 0**：
    $$\frac{\partial \mathcal{L}}{\partial \delta} = \mathbf{H} \delta^T - \lambda \mathbf{e}_i = 0 \implies \delta^T = \lambda \mathbf{H}^{-1} \mathbf{e}_i$$
2. **解出 $\lambda$**：
    将 $\delta$ 代入约束条件 $\mathbf{e}_i^T \delta = \delta w_i$：
    $$\mathbf{e}_i^T (\lambda \mathbf{H}^{-1} \mathbf{e}_i) = \delta w_i \implies \lambda [\mathbf{H}^{-1}]_{ii} = \delta w_i \implies \lambda = \frac{\delta w_i}{[\mathbf{H}^{-1}]_{ii}}$$
3. **得到最终补偿公式**：
    $$\delta = \frac{\delta w_i}{[\mathbf{H}^{-1}]_{ii}} (\mathbf{H}^{-1})_{i, :}$$

---

### 六、 结论：GPTQ 更新法则
在量化第 $i$ 个权重时，我们不仅要改变它，还要按以下公式更新**同一行中所有剩余权重**：
$$\Delta \text{Weights}_{remaining} = - \frac{w_i - Q(w_i)}{[\mathbf{H}^{-1}]_{ii}} \cdot (\mathbf{H}^{-1})_{i, i:}$$
#### Obsidian 知识点复习
> [!important] 核心精要
> 
> - **Hessian $H$** 代表输入特征的相关性。
>     
> - **$H^{-1}$ 的第 $i$ 列** 决定了误差如何分摊给其他权重。
>     
> - **补偿机制** 确保了虽然单个权重精度下降，但整层输出的宏观统计特性保持不变。
>     

---
