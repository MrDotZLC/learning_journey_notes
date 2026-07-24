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

## 2. 整体方案核心思想

### 2.1 本质矛盾

MoE 的设计目标是**稀疏激活**：每个 token 只经过 $k$ 个 expert，总计算量为 Dense FFN 的 $k/E$（理想情况）。但 GPU 的高效计算单元（Tensor Core）只擅长**密集矩阵乘法**，天然排斥稀疏、不规则的访问模式。

二者之间存在根本矛盾：

$$
\underbrace{\text{稀疏路由}}_{\text{模型设计}} \quad \longleftrightarrow \quad \underbrace{\text{密集 GEMM}}_{\text{硬件偏好}}
$$

### 2.2 解决思路：将稀疏问题转化为密集问题

直接对原始 token 序列做稀疏 gather + 独立 GEMM，会引发两个连锁问题：

1. **访存不规则**：路由后每个 expert 对应的 token 在原始序列中离散分布，global memory 访问非合并，HBM 带宽利用率极低。
2. **计算粒度碎片化**：$n_i$ 大小不一，直接逐 expert 启动独立 GEMM kernel，小 expert 会产生大量 launch overhead 且 SM 严重空转。

解决方案拆分为两个正交步骤：

| 步骤                | 操作                                           | 解决的问题        |
| ----------------- | -------------------------------------------- | ------------ |
| Token Permutation | 数据重排，使同一 expert 的 token 在内存中连续               | 访存非合并        |
| Grouped GEMM      | 所有 expert 的 GEMM 在单次 persistent kernel 中动态调度 | 计算碎片化 + 负载失衡 |

两步操作的组合，将一个**稀疏、不规则**的计算问题，转化为一个**密集、可被 Tensor Core 高效执行**的计算问题，代价是引入两次额外的 HBM 读写（permutation + un-permutation），这也是优化后系统的主要瓶颈所在。

### 2.3 为何不直接用 Block-Sparse GEMM

理论上可以绕过 permutation，直接对离散 token 做 block-sparse GEMM（MegaBlocks 路线）。代价是：

- sparse 访问模式使 Tensor Core pipeline 难以满载
- block-sparse kernel 实现复杂度高，硬件适配性差

工业推理引擎（vLLM、TensorRT-LLM）主流选择 Permutation 路线，以额外的内存带宽换取 Tensor Core 的密集计算效率。

***

## 3. 核心挑战

### 3.1 Warp / CTA 级别负载失衡

GPU 执行粒度：

- Warp：32 threads
- CTA（Thread Block）

若：

$$
n_i \ll M_{\text{tile}}
$$

则 Warp 空转，Tensor Core utilization 降低。

### 3.2 Memory Access 非合并（Uncoalesced）

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

### 3.3 Expert Capacity 与 Token Drop

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

## 4. Token Permutation

### 4.1 核心思想

Token Permutation 的本质是**用一次显式的数据搬运，换取后续所有计算的访存规整性**。

GPU global memory 的访问效率取决于**访问模式是否连续（coalesced）**。同一 warp 的 32 个线程若访问连续地址，可合并为少量 128B 事务；若地址离散，则退化为 32 次独立事务，有效带宽下降至峰值的 $1/32$。

路由后，expert $i$ 对应的 token 在原始序列 $X$ 中的位置是任意的。若直接以稀疏 index 访问，warp 内各线程的访问地址无规律，必然触发 uncoalesced access。

解决方案：在进入 GEMM 之前，执行一次 **scatter 操作**，将属于同一 expert 的 token **物理上搬移到连续地址**。代价是一次 $O(kNd)$ 的 HBM 写入；收益是此后所有 GEMM 的输入访存均为连续，整体带宽利用率大幅提升。

$$
\underbrace{X}_{\text{离散分布}} \xrightarrow{\text{scatter（一次性开销）}} \underbrace{X_{\text{perm}}}_{\text{按 expert 连续排列}} \xrightarrow{\text{Grouped GEMM}} Y_{\text{perm}} \xrightarrow{\text{gather}} \underbrace{Y}_{\text{还原原始顺序}}
$$

### 4.2 数学建模

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

### 4.3 Prefix-Sum + Offset 实现

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

### 4.4 GPU Kernel 设计要点

#### 4.4.1 避免 Atomic Contention

优化策略：

- 使用 warp-local buffer 聚合写操作，最后批量写回
- 或使用 block-level prefix sum 替代 per-thread atomic

#### 4.4.2 Vectorized Load/Store

```cpp
// 128-bit 向量化读取，保证 coalesced access
float4 v = *reinterpret_cast<const float4*>(&x[t * d]);
*reinterpret_cast<float4*>(&x_perm[pos * d]) = v;
```

要求：`d` 满足 128-bit 对齐（即 $d \bmod 4 == 0$ for FP32，$d \bmod 8 == 0$ for FP16）。

#### 4.4.3 Layout 设计

两种主流布局：

| Layout | 描述 | 优点 |
|--------|------|------|
| Expert-major | `[E][n_i][d]` | GEMM 友好，连续内存访问 |
| Token-major | `[kN][d]` | 实现简单，Un-permute 高效 |

工业实现统一采用 **Expert-major contiguous layout**，配合 `offset[i]` 索引每段起点。

### 4.5 Un-permutation（推理/训练一致性）

逆操作：

$$
X_{\text{out}} = P^{\top} Y_{\text{perm}}
$$

其中 $P^{\top}$ 对应 gather 操作（按原始 token 顺序将 expert 输出收集回位）。

- 推理：scatter/gather，无梯度
- 训练：需对 $Y_{\text{perm}}$ 中对应 expert 的输出做梯度 accumulate（一个 token 被多个 expert 处理时）

***

## 5. Segmented / Grouped GEMM

### 5.1 核心思想

Grouped GEMM 的本质是**用单次 persistent kernel 的动态调度，替代多次独立 kernel launch 的静态分配**。

朴素实现中，对 $E$ 个 expert 逐一调用 `cublasSgemm`，存在两个根本缺陷：

1. **Launch overhead 累积**：每次 kernel launch 约有 $5\text{–}20\,\mu\text{s}$ 的 CPU-GPU 调度延迟，$E$ 个 expert 串行启动时，overhead 占总耗时的比例不可忽视。
2. **静态资源分配导致失衡**：若为每个 expert 静态分配固定数量的 CTA，$n_i$ 小的 expert 会导致大量 CTA 空转，而 $n_i$ 大的 expert 则排队等待，SM 整体利用率低下。

Grouped GEMM 的解决思路：

$$
\underbrace{\text{多次独立 launch}}_{\text{串行，静态分配}} \longrightarrow \underbrace{\text{单次 persistent kernel}}_{\text{并发，动态调度}}
$$

所有 expert 的 GEMM tile 被统一放入一个**任务队列**，GPU 上的 CTA 持续运行并动态从队列中领取任务。这样：

- **消除 launch overhead**：只有一次 kernel launch
- **消除负载失衡**：CTA 不绑定 expert，任务自然均摊到所有活跃 SM
- **提升 Tensor Core 饱和度**：tile 粒度调度使 SM 始终有任务可执行，减少空转

### 5.2 问题形式化

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

### 5.3 Padding 方法的 FLOPs 浪费

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
\rho = \frac{E \cdot n_{\max}}{\displaystyle\sum_{i=1}^{E} n_i}
$$

当分布高度不均时 $\rho \gg 1$，Padding 方法不可接受。

### 5.4 Grouped GEMM 内核机制

#### 5.4.1 Kernel 输入结构

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

#### 5.4.2 cuBLASLt vs CUTLASS 差异

| 属性 | cuBLASLt GemmBatched | CUTLASS Grouped GEMM |
|------|----------------------|----------------------|
| 问题尺寸 | 所有 batch 相同 $(m, n, k)$ | 每组独立 $(m_i, n_i, k_i)$ |
| 适用场景 | $n_i$ 均匀 | $n_i$ 不均匀（MoE 典型场景） |
| 调度机制 | 静态 batch launch | Persistent kernel + tile 队列 |
| 自定义算子支持 | 受限 | 可定制 epilogue（如 fused activation） |

#### 5.4.3 执行模型：Persistent Kernel + Work Queue

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

### 5.5 Tensor Core 对齐约束

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

### 5.6 FP8 Grouped GEMM（Hopper / Blackwell）

Hopper 架构（H100）引入 WGMMA 指令，支持 FP8（E4M3/E5M2）Grouped GEMM：

- 峰值算力：FP8 约为 BF16 的 **2×**（H100 SXM：~2000 TFLOPS FP8 vs ~989 TFLOPS BF16）
- 约束：需要 per-tensor 或 per-block scale factor，CUTLASS 3.x 提供 `BlockScaledGroupedGemm`
- 工程成本：KV 量化与 expert weight 量化需协同设计

***

## 6. 端到端 Pipeline

### 6.1 完整流程（工业实现）

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

### 6.2 Kernel Fusion 机会

| 融合组合 | 收益 |
|----------|------|
| GEMM + Activation（GELU/SwiGLU） | 消除中间 tensor HBM 读写 |
| Permutation + GEMM | 难度高，需自定义 epilogue |
| GEMM + Quantization（FP8 output） | Hopper 原生支持 |

***

## 7. 性能模型

### 7.1 总耗时分解

$$
T_{\text{total}} = T_{\text{perm}} + T_{\text{gemm}} + T_{\text{unperm}} + T_{\text{comm}}
$$

多 GPU 场景下 $T_{\text{comm}}$ 不可忽略。

### 7.2 Permutation 带宽模型

$$
T_{\text{perm}} \approx \frac{k N d \cdot \text{sizeof(dtype)}}{BW_{\text{HBM}}}
$$

以 H100 SXM（$BW_{\text{HBM}} = 3.35\,\text{TB/s}$）、$N=4096$，$k=2$，$d=4096$，BF16 为例：

$$
T_{\text{perm}} \approx \frac{2 \times 4096 \times 4096 \times 2}{3.35 \times 10^{12}} \approx 20\,\mu\text{s}
$$

→ **memory-bound 操作**。

### 7.3 GEMM 计算模型

$$
T_{\text{gemm}} \approx \frac{2 \displaystyle\sum_{i=1}^{E} n_i \cdot d \cdot d_{\text{ff}}}{\text{TensorCore TFLOPS}}
$$

（系数 2 来自 multiply-add 各计一次 FLOP）

→ **compute-bound（理想满载时）**。

### 7.4 关键平衡条件

系统高效运行要求：

$$
T_{\text{gemm}} \gg T_{\text{perm}} + T_{\text{comm}}
$$

否则 MoE 的稀疏计算优势被内存/通信开销吞噬。批量大（大 $N$）时此条件更易满足；batch size 小时（推理场景）是主要工程挑战。

***

## 8. 工程实现对比

| 框架 | 技术路径 | 特点 |
|------|----------|------|
| DeepSpeed-MoE | Token Permutation + All-to-All | 强分布式，支持 EP |
| MegaBlocks | Block-sparse GEMM（dMoE） | 避免 permutation 开销 |
| vLLM | Permutation + Grouped GEMM | 推理优化，PagedAttention |
| NVIDIA CUTLASS 3.x | Grouped GEMM + WGMMA（FP8） | kernel 级，最高硬件利用率 |
| Tutel | Adaptive Top-k + NCCL All-to-All | 动态 capacity，训练/推理通用 |

### 8.1 MegaBlocks vs Permutation 路线

| 维度 | Permutation 路线 | MegaBlocks（dMoE）路线 |
|------|-----------------|----------------------|
| 核心思想 | 数据重排 → 密集 GEMM | block-sparse GEMM，无需重排 |
| HBM 读写 | 额外一次 $X_{\text{perm}}$ 写读 | 省去 permutation，但 sparse 访问复杂 |
| Tensor Core 利用率 | 高（密集 tile） | 取决于 block 大小 |
| 实现复杂度 | 中 | 高（需自定义 sparse kernel） |
| 适用场景 | 工业推理引擎主流 | 研究 / 训练阶段探索 |

***

## 9. 关键结论

### 9.1 本质分工

$$
\text{Token Permutation} \Rightarrow \text{memory layout optimization}
$$

$$
\text{Grouped GEMM} \Rightarrow \text{compute scheduling optimization}
$$

### 9.2 性能瓶颈迁移

| 阶段 | 瓶颈类型 |
|------|----------|
| 优化前 | compute-bound + load imbalance |
| 优化后（单机） | memory-bound（Permutation + HBM BW） |
| 优化后（多机） | communication-bound（All-to-All） |

### 9.3 工业界真实瓶颈（2025+）

在 Mixtral-8×7B、DeepSeek-V3、GPT-4 类 MoE 系统中：

- **单机推理**：HBM bandwidth 为主瓶颈（small batch），FP8 Grouped GEMM 为关键优化
- **多机推理**：All-to-All 通信为主瓶颈，需 overlap 通信与计算（async dispatch/combine）
- **趋势**：Expert Parallelism + Tensor Parallelism 混合，通信量分析转向 $\mathcal{O}\!\left(\dfrac{kNd}{tp}\right)$

***

## 10. 图示占位

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
