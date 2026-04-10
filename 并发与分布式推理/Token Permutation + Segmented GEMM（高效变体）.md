## 1. 问题建模：MoE 的稀疏路由本质

设：

- 输入 token 数：$N$
- 专家数：$E$
- Top-$k$ 路由（通常 $k=1$ 或 $k=2$）
- 隐藏维度：$d_{\text{model}}$
- FFN 扩展维度：$d_{\text{ff}}$

Router 输出：

$$
g_{t,i} = \text{softmax}(z_t)_i, \quad i \in [1, E]
$$

Top-$k$ 选择：

$$
\mathcal{E}(t) = \text{TopK}(g_t, k)
$$

每个 token 被映射到最多 $k$ 个 expert：

$$
X_i = \{ x_t \mid i \in \mathcal{E}(t) \}
$$

定义每个 expert 的负载：

$$
n_i = |X_i|
$$

约束：

$$
\sum_{i=1}^{E} n_i = kN
$$

> **核心问题**：$\{n_i\}$ 呈高度不均匀分布（长尾），导致：
>
> - Kernel 粒度不均
> - SM 利用率下降
> - Launch overhead 放大

***

## 2. 核心挑战

### 2.1 Warp / CTA 级别负载失衡

GPU 执行粒度：

- Warp：32 threads
- CTA（Thread Block）

若：

$$
n_i \ll M_{\text{tile}}
$$

则 Warp 空转，Tensor Core utilization 降低。

### 2.2 Memory Access 非合并（Uncoalesced）

原始 token layout：

$$
X = [x_1, x_2, \dots, x_N]
$$

路由后访问模式：

$$
x_{t_1}, x_{t_7}, x_{t_{23}}, \dots
$$

产生：

- 非连续 global memory access
- L2 miss 增加
- DRAM transaction 放大

### 2.3 Expert Capacity 与 Token Drop

工业实现中引入 **Expert Capacity** 上界约束：

$$
C_i = \left\lfloor \alpha \cdot \frac{kN}{E} \right\rfloor
$$

其中 $\alpha \geq 1$ 为 capacity factor（通常取 1.0～1.25）。

当 $n_i > C_i$ 时，超出部分的 token 被 **drop**（丢弃或走 residual bypass）：

$$
n_i^{\text{eff}} = \min(n_i, C_i)
$$

作用：

- 使 $\{n_i^{\text{eff}}\}$ 有界，GEMM tile 可静态分配
- 代价：精度损失，需配合 auxiliary load balancing loss 缓解

***

## 3. Token Permutation

### 3.1 数学建模

定义：

- 原始输入：$X \in \mathbb{R}^{N \times d}$
- 排列矩阵（Permutation Matrix）：$P \in \{0,1\}^{(kN) \times N}$

满足：

$$
X_{\text{perm}} = P X
$$

其中：

- 每一行只有一个 1（选择某个 token）
- 行按 expert 分组排列

### 3.2 Prefix-Sum + Offset 实现

工程实现不显式构造 $P$，而是：

**Step 1：Histogram**

$$
n_i = \sum_{t=1}^{N} \mathbf{1}(i \in \mathcal{E}(t))
$$

**Step 2：Prefix Sum（Exclusive Scan）**

$$
\text{offset}_i = \sum_{j < i} n_j
$$

**Step 3：写入位置计算**

对于 token $t$ 分配到 expert $i$：

$$
\text{pos}(t, i) = \text{offset}_i + \text{atomic\_inc}(\text{counter}_i)
$$

Exclusive scan 的语义：$\text{offset}_0 = 0$，$\text{offset}_i$ 为前 $i$ 个 expert 的累积 token 数，保证 expert $i$ 在 $X_{\text{perm}}$ 中的起始行为 $\text{offset}_i$。

### 3.3 GPU Kernel 设计要点

#### 3.3.1 避免 Atomic Contention

优化策略：

- 使用 warp-local buffer 聚合写操作，最后批量写回
- 或使用 block-level prefix sum 替代 per-thread atomic

#### 3.3.2 Vectorized Load/Store

```cpp
