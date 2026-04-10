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
// 128-bit 向量化读取，保证 coalesced access
float4 v = *reinterpret_cast<const float4*>(&x[t * d]);
*reinterpret_cast<float4*>(&x_perm[pos * d]) = v;
```

要求：`d` 满足 128-bit 对齐（即 $d \bmod 4 == 0$ for FP32，$d \bmod 8 == 0$ for FP16）。

#### 3.3.3 Layout 设计

两种主流布局：

| Layout       | 描述              | 优点             |
| ------------ | --------------- | -------------- |
| Expert-major | `[E][n_i][d]`   | GEMM 友好，连续内存访问 |
| Token-major  | `[kN][d]`       | 实现简单，Un-permute 高效 |

工业实现统一采用 **Expert-major contiguous layout**，配合 `offset[i]` 索引每段起点。

### 3.4 Un-permutation（推理/训练一致性）

逆操作：

$$
X_{\text{out}} = P^{\top} Y_{\text{perm}}
$$

其中 $P^{\top}$ 对应 gather 操作（按原始 token 顺序将 expert 输出收集回位）。

- 推理：scatter/gather，无梯度
- 训练：需对 $Y_{\text{perm}}$ 中对应 expert 的输出做梯度 accumulate（一个 token 被多个 expert 处理时）

***

## 4. Segmented / Grouped GEMM

### 4.1 问题形式化

每个 expert 执行独立矩阵乘：

$$
Y_i = X_i W_i
$$

其中：

- $X_i \in \mathbb{R}^{n_i \times d}$
- $W_i \in \mathbb{R}^{d \times d_{\text{ff}}}$

目标：

$$
Y = \text{Concat}(Y_1, Y_2, \dots, Y_E)
$$

### 4.2 Padding 方法的 FLOPs 浪费

设：

$$
n_{\max} = \max_i n_i
$$

Padding 后 FLOPs：

$$
\text{FLOPs}_{\text{pad}} = E \cdot n_{\max} \cdot d \cdot d_{\text{ff}}
$$

真实 FLOPs：

$$
\text{FLOPs}_{\text{real}} = \sum_{i=1}^{E} n_i \cdot d \cdot d_{\text{ff}}
$$

浪费比：

$$
\rho = \frac{E \cdot n_{\max}}{\sum_{i=1}^{E} n_i}
$$

当分布高度不均时 $\rho \gg 1$，Padding 方法不可接受。

### 4.3 Grouped GEMM 内核机制

#### 4.3.1 Kernel 输入结构

```cpp
// CUTLASS Grouped GEMM 接口（伪代码）
GemmCoord problem_sizes[E];    // 每个 expert 的 (m, n, k)
void* A_ptrs[E];               // A_ptrs[i] = X_i 起始地址
void* B_ptrs[E];               // B_ptrs[i] = W_i 起始地址
void* C_ptrs[E];               // C_ptrs[i] = Y_i 起始地址

for (int i = 0; i < E; ++i) {
    problem_sizes[i] = {n_i, d_ff, d};
    A_ptrs[i] = x_perm + offset[i] * d;
    B_ptrs[i] = w + i * d * d_ff;
    C_ptrs[i] = y_perm + offset[i] * d_ff;
}
```

#### 4.3.2 cuBLASLt vs CUTLASS 差异

| 属性             | cuBLASLt GemmBatched         | CUTLASS Grouped GEMM          |
| -------------- | ---------------------------- | ----------------------------- |
| 问题尺寸           | 所有 batch 相同 $(m, n, k)$      | 每组独立 $(m_i, n_i, k_i)$       |
| 适用场景           | $n_i$ 均匀                     | $n_i$ 不均匀（MoE 典型场景）           |
| 调度机制           | 静态 batch launch              | Persistent kernel + tile 队列  |
| 自定义算子支持        | 受限                           | 可定制 epilogue（如 fused activation） |

#### 4.3.3 执行模型：Persistent Kernel + Work Queue

核心思想：CTA 不与 expert 静态绑定，而是动态从任务队列领取 tile。

任务粒度（tile 数量）：

$$
T_i = \left\lceil \frac{n_i}{M_{\text{tile}}} \right\rceil
$$

总任务数：

$$
T = \sum_{i=1}^{E} T_i
$$

CTA 调度伪代码：

```cpp
__global__ void grouped_gemm_kernel(TaskQueue* queue, int T) {
    int task_id = atomicAdd(&queue->head, 1);
    while (task_id < T) {
        auto [expert_id, tile_row] = decode_task(task_id);
        compute_gemm_tile(expert_id, tile_row);
        task_id = atomicAdd(&queue->head, 1);
    }
}
```

作用：消除长尾 expert 导致的 CTA 空转，SM 保持饱和。

### 4.4 Tensor Core 对齐约束

Tensor Core（Ampere wmma/mma 指令）要求：

$$
m, n, k \equiv 0 \pmod{16} \quad (\text{FP16/BF16})
$$

$$
m, n, k \equiv 0 \pmod{32} \quad (\text{FP8, Hopper WGMMA})
$$

处理策略：

- **Tile-level internal padding**：tile 内部补零至对齐尺寸，开销仅在 tile 粒度，远小于 expert-level padding
- **Mixed tile size**（Ampere+）：对小 expert 使用更小 tile（如 $32 \times 32$），减少浪费

### 4.5 FP8 Grouped GEMM（Hopper / Blackwell）

Hopper 架构（H100）引入 WGMMA 指令，支持 FP8（E4M3/E5M2）Grouped GEMM：

- 峰值算力：FP8 约为 BF16 的 **2×**（H100 SXM：~2000 TFLOPS FP8 vs ~989 TFLOPS BF16）
- 约束：需要 per-tensor 或 per-block scale factor，CUTLASS 3.x 提供 `BlockScaledGroupedGemm`
- 工程成本：KV 量化与 expert weight 量化需协同设计

***

## 5. 端到端 Pipeline

### 5.1 完整流程（工业实现）

**Step 1：Routing**

$$
\mathcal{E}(t) = \text{TopK}(\text{softmax}(x_t W_r), k)
$$

**Step 2：Dispatch（Permutation）**

- histogram → prefix-sum → scatter

**Step 3：Expert Parallel（多 GPU）**

$$
X_i \xrightarrow{\text{All-to-All}} \text{GPU}_{r(i)}
$$

通信量：

$$
\mathcal{O}(kN \cdot d)
$$

瓶颈：NVLink（节点内）/ InfiniBand（节点间）带宽。

**Step 4：Grouped GEMM**

- persistent kernel + tile scheduling
- 可选：Kernel Fusion（GEMM + Activation）

典型 FFN 计算（以 SwiGLU 为例）：

$$
Y_i = \text{silu}(X_i W_{1,i}) \odot (X_i W_{3,i})
$$

$$
\text{output}_i = Y_i W_{2,i}
$$

**Step 5：Combine（Un-permute）**

$$
X_{\text{out}} = P^{\top} Y_{\text{perm}}
$$

多 GPU 时需先 All-to-All 将结果送回原 token 所在 GPU。

### 5.2 Kernel Fusion 机会

| 融合组合                       | 收益               |
| -------------------------- | ---------------- |
| GEMM + Activation（GELU/SwiGLU） | 消除中间 tensor HBM 读写 |
| Permutation + GEMM         | 难度高，需自定义 epilogue |
| GEMM + Quantization（FP8 output） | Hopper 原生支持      |

***

## 6. 性能模型

### 6.1 总耗时分解

$$
T_{\text{total}} = T_{\text{perm}} + T_{\text{gemm}} + T_{\text{unperm}} + T_{\text{comm}}
$$

多 GPU 场景下 $T_{\text{comm}}$ 不可忽略。

### 6.2 Permutation 带宽模型

$$
T_{\text{perm}} \approx \frac{k N d \cdot \text{sizeof(dtype)}}{BW_{\text{HBM}}}
$$

以 H100 SXM（$BW_{\text{HBM}} = 3.35 \text{ TB/s}$）、$N=4096$，$k=2$，$d=4096$，BF16 为例：

$$
T_{\text{perm}} \approx \frac{2 \times 4096 \times 4096 \times 2}{3.35 \times 10^{12}} \approx 20 \, \mu\text{s}
$$

→ **memory-bound 操作**。

### 6.3 GEMM 计算模型

$$
T_{\text{gemm}} \approx \frac{2 \sum_{i=1}^{E} n_i \cdot d \cdot d_{\text{ff}}}{\text{TensorCore TFLOPS}}
$$

（系数 2 来自 multiply-add 各计一次 FLOP）

→ **compute-bound（理想满载时）**。

### 6.4 关键平衡条件

系统高效运行要求：

$$
T_{\text{gemm}} \gg T_{\text{perm}} + T_{\text{comm}}
$$

否则 MoE 的稀疏计算优势被内存/通信开销吞噬。批量大（大 $N$）时此条件更易满足；batch size 小时（推理场景）是主要工程挑战。

***

## 7. 工程实现对比

| 框架                   | 技术路径                                 | 特点                  |
| -------------------- | ------------------------------------ | ------------------- |
| DeepSpeed-MoE        | Token Permutation + All-to-All       | 强分布式，支持 EP           |
| MegaBlocks           | Block-sparse GEMM（dMoE）              | 避免 permutation 开销    |
| vLLM                 | Permutation + Grouped GEMM           | 推理优化，PagedAttention  |
| NVIDIA CUTLASS 3.x   | Grouped GEMM + WGMMA（FP8）            | kernel 级，最高硬件利用率     |
| Tutel                | Adaptive Top-k + NCCL All-to-All     | 动态 capacity，训练/推理通用  |

### 7.1 MegaBlocks vs Permutation 路线

| 维度        | Permutation 路线          | MegaBlocks（dMoE）路线         |
| --------- | ----------------------- | -------------------------- |
| 核心思想      | 数据重排 → 密集 GEMM          | block-sparse GEMM，无需重排     |
| HBM 读写    | 额外一次 $X_{\text{perm}}$ 写读 | 省去 permutation，但 sparse 访问复杂 |
| Tensor Core 利用率 | 高（密集 tile）            | 取决于 block 大小              |
| 实现复杂度     | 中                       | 高（需自定义 sparse kernel）      |
| 适用场景      | 工业推理引擎主流               | 研究 / 训练阶段探索               |

***

## 8. 关键结论

### 8.1 本质分工

$$
\text{Token Permutation} \Rightarrow \text{memory layout optimization}
$$

$$
\text{Grouped GEMM} \Rightarrow \text{compute scheduling optimization}
$$

### 8.2 性能瓶颈迁移

| 阶段  | 瓶颈类型                                |
| --- | ----------------------------------- |
| 优化前 | compute-bound + load imbalance      |
| 优化后（单机） | memory-bound（Permutation + HBM BW） |
| 优化后（多机） | communication-bound（All-to-All）    |

### 8.3 工业界真实瓶颈（2025+）

在 Mixtral-8×7B、DeepSeek-V3、GPT-4 类 MoE 系统中：

- **单机推理**：HBM bandwidth 为主瓶颈（small batch），FP8 Grouped GEMM 为关键优化
- **多机推理**：All-to-All 通信为主瓶颈，需 overlap 通信与计算（async dispatch/combine）
- **趋势**：Expert Parallelism + Tensor Parallelism 混合，通信量分析转向 $O\!\left(\frac{kNd}{tp}\right)$

***

## 9. 图示占位

```
[图 1] Token Routing → Permutation → Grouped GEMM → Unpermute 数据流

描述：
- 上层：原始 token 序列 [x1, x2, ..., xN]，颜色标注路由 expert
- 中层：按 expert 分桶后的连续 layout，标注 offset[i]
- 下层：Grouped GEMM tile 调度示意，CTA 从 task queue 动态领取 tile
```

```
[图 2] Grouped GEMM CTA 调度模型

描述：
- 左侧：不同大小的 expert GEMM 任务（n_i 不等）
- 右侧：task queue 中 tile 序列，CTA 动态 dequeue
- 箭头：SM → tile，体现 persistent kernel 调度
```

```
[图 3] 性能瓶颈迁移示意

描述：
- X 轴：batch size N
- Y 轴：耗时占比
- 三条曲线：T_perm（内存）、T_gemm（计算）、T_comm（通信）
- 标注：小 batch 时 T_perm 主导；大 batch 时 T_gemm 主导；多机时 T_comm 主导
```