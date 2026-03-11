## 第 1 章·参考答案：GPU 硬件与内存体系

---

### 1.1 基础硬件架构

---

**Q1. GPU 的 SM（Streaming Multiprocessor）内部结构是什么？Warp 如何调度？**

**SM 核心组件（以 H100 SXM 为例）：**

|组件|数量/规格|职责|
|---|---|---|
|FP32 CUDA Core|128 个/SM|标量浮点/整数运算|
|Tensor Core（第四代）|4 组/SM|MMA 矩阵乘累加，支持 FP16/BF16/FP8/INT8|
|Register File|256 KB/SM|线程私有寄存器，最快存储层次|
|Shared Memory / L1|最大 228 KB/SM|Block 内线程共享，软件管理缓存|
|Warp Scheduler|4 个/SM|每周期各发射 1 条指令|
|SFU（特殊函数单元）|32 个/SM|三角函数、倒数等|

**Warp 调度机制：**

- 32 个线程构成 1 个 Warp，是 GPU 调度的**最小单位**。
- 调度器采用**零开销上下文切换**：当活跃 Warp 因全局内存访问停顿时，立即切换到其他就绪 Warp，以计算隐藏延迟。
- 每个 SM 可同时驻留多个 Warp（由 Occupancy 决定，见 Q70），Warp 数量越多，延迟隐藏越充分。
- 调度策略通常为 **GTO（Greedy-Then-Oldest）** 或 Round-Robin，具体由硬件实现。

---

**Q2. CUDA 的内存层次各层的带宽与延迟数量级是多少？**

|层次|带宽（H100 SXM）|访问延迟|作用域|
|---|---|---|---|
|Register File|~20 TB/s（聚合估算）|~1 cycle|线程私有|
|Shared Memory / L1|~20 TB/s|~20–30 cycles|Block 内共享|
|L2 Cache|~10 TB/s|~200 cycles|全 GPU 共享|
|HBM3（H100 SXM）|**3.35 TB/s**|~400–600 cycles|全 GPU 主显存|
|NVLink 4.0（GPU–GPU）|~900 GB/s 双向|微秒级|节点内多卡|
|PCIe 5.0（CPU–GPU）|~128 GB/s 双向|微秒级|主机–设备|

**核心结论：** Register 到 HBM 带宽差约 6×，延迟差约 400×。Kernel 优化的首要原则是**最大化片上数据复用**，减少 HBM Round-trip 次数。

---

**Q3. 什么是 Memory Coalescing？为什么非合并访问会严重降低性能？**

**定义：** 同一 Warp 的 32 个线程若访问**连续对齐的内存地址**，硬件将其合并为 1 次（或少数几次）128 字节的内存事务；否则每个线程产生独立事务，最坏情况触发 32 次事务。

**性能影响：**

- 合并访问：1 次事务，有效带宽利用率接近 100%。
- 非合并访问（如 Stride-N 或随机访问）：最差 32 次事务，实际带宽利用率下降至 $1/32 \approx 3%$。

**典型代码对比：**

```cpp
// ✅ 合并访问：线程 i 访问 A[i]，地址连续
int val = A[threadIdx.x];

// ❌ 非合并访问：线程 i 访问 A[i * stride]，地址跨步
int val = A[threadIdx.x * 32];
```

**规避方法：** 矩阵转置时使用 Shared Memory 中转，将非合并的列访问转换为合并的行访问后再写出。

---

**Q4. Shared Memory 的 Bank Conflict 是什么？如何消除？**

**Bank 结构：** Shared Memory 被划分为 **32 个 Bank**，每个 Bank 宽度 4 字节（可配置为 8 字节）。同一 Warp 内多个线程若访问**同一 Bank 的不同地址**，产生 Bank Conflict，访问被串行化。

**冲突判断：** 线程 $i$ 访问地址 $A_i$，若 $A_i \bmod 32$ 对多个线程相同且地址不同，则冲突。

**消除方法：**

1. **Padding（填充）：** 在 Shared Memory 数组每行末尾添加 1 个元素，改变步长。

```cpp
// 矩阵转置：每行 pad 1 列，消除 Bank Conflict
__shared__ float tile[BLOCK][BLOCK + 1];  // +1 padding
```

2. **Swizzle（地址重排）：** 通过位运算重映射访问地址，使同一 Warp 的线程散布到不同 Bank（FlashAttention-3 采用此方案）。

**特例：** 广播（Broadcast）——同一 Warp 多个线程访问**同一 Bank 的同一地址**，不产生冲突，硬件广播一次即可。

---

**Q5. H100 / A100 / H20 的关键硬件参数对比？**

|指标|A100 SXM4|H100 SXM5|H20|
|---|---|---|---|
|显存类型|HBM2e|HBM3|HBM3|
|显存容量|80 GB|80 GB|96 GB|
|显存带宽|2.0 TB/s|3.35 TB/s|4.0 TB/s|
|FP16 Tensor TFLOPS|312|989（稀疏）|148|
|FP8 Tensor TFLOPS|—|1979（稀疏）|296|
|NVLink 带宽（双向）|600 GB/s|900 GB/s|900 GB/s|
|SM 数量|108|132|132|

**定位差异：** H20 以高带宽、大显存为核心卖点，牺牲算力，专为显存带宽瓶颈的 Decode 推理场景设计（适合中国市场出口管制下的替代方案）。

---

**Q6. Warp Divergence 对性能的影响及规避方法？**

**原理：** SIMT 模型要求同 Warp 的 32 个线程执行相同指令。若线程因 `if/else`、`while` 等分支走向不同路径，GPU 将**串行执行所有分支**，非活跃线程被掩码屏蔽（Predicate Off），等待。

**性能损失：** 最坏情况（32 个线程走 32 条不同分支）吞吐降至 $1/32$。

**规避策略：**

1. **对齐分支到 Warp 边界**：保证同一 Warp 的 32 个线程走相同分支（按 Warp ID 而非 Thread ID 分支）。
2. **使用 Warp 级原语替代分支**：`__ballot_sync`、`__any_sync`、`__all_sync` 在 Warp 内进行条件聚合。
3. **展开循环，消除边界条件分支**：`#pragma unroll`。
4. **避免 Warp 内 Dynamic 索引计算差异过大**，尤其在 Reduction 时注意 tail 处理。

---

### 1.2 计算访存比分析

---

**Q7. 什么是 Arithmetic Intensity？如何用 Roofline Model 判断瓶颈？**

**Arithmetic Intensity（算术强度）定义：**

$$I = \frac{\text{FLOPs}}{\text{Bytes Accessed (HBM)}} \quad \left[\text{FLOP/Byte}\right]$$

**Roofline Model：**

性能上界由两个"屋顶"决定：

$$\text{Performance} = \min!\left(I \times BW_{\text{mem}},; P_{\text{peak}}\right)$$

其中 $BW_{\text{mem}}$ 为 HBM 带宽，$P_{\text{peak}}$ 为峰值算力。

**Ridge Point（脊点）：**

$$I^* = \frac{P_{\text{peak}}}{BW_{\text{mem}}}$$

以 H100 SXM 为例：$I^* = \dfrac{989 \text{ TFLOPS}}{3.35 \text{ TB/s}} \approx 295 \text{ FLOP/Byte}$

- $I < I^*$：**Memory-bound**，优化方向为减少 HBM 访问（Fusion、量化、提高数据复用）。
- $I > I^*$：**Compute-bound**，优化方向为提高算力利用率（Tensor Core、流水线）。

---

**Q8. LLM 推理的 Prefill 阶段和 Decode 阶段分别属于哪种瓶颈？**

|阶段|输入形状|主要算子|瓶颈类型|原因|
|---|---|---|---|---|
|**Prefill**|Batch × $S_{\text{in}}$（$S_{\text{in}}$ 大）|GEMM（大矩阵）|Compute-bound|$S_{\text{in}}$ 大时 GEMM 形状方正，Tensor Core 利用率高，$I \gg I^*$|
|**Decode**|Batch × 1（逐 Token）|GEMV（矩阵×向量）|Memory-bound|每步仅生成 1 Token，权重矩阵被读取一遍但计算量极少，$I \ll I^*$|

**Decode 阶段的 $I$ 估算：** 以 Linear 层为例，权重大小 $W \in \mathbb{R}^{d \times d}$，Batch=1 时：

- FLOPs $= 2d^2$
- Bytes $= 2d^2$（FP16 权重读取）
- $I = 1 \text{ FLOP/Byte}$，远小于 295 的脊点，深度 Memory-bound。

---

**Q9. GEMV 与 GEMM 的计算访存比差距？为何 Decode 受限于显存带宽？**

**GEMM（$M \times N \times K$，$M, N, K$ 均大）：**

$$I_{\text{GEMM}} = \frac{2MNK}{2(MK + NK + MN)} \approx \frac{M}{2} \quad (M=N=K)$$

典型值：$M=4096$ 时，$I \approx 2048 \text{ FLOP/Byte}$，Compute-bound。

**GEMV（$M \times K$，向量长度 $K$，Batch=1）：**

$$I_{\text{GEMV}} = \frac{2MK}{2MK + 2K} \approx 1 \text{ FLOP/Byte}$$

**结论：** Decode 阶段每步需从 HBM 读取模型**全部权重**（数十 GB），而计算量仅为读取量的 $\sim$1 FLOP/Byte。H100 HBM 带宽 3.35 TB/s，读取 70B FP16 模型权重（140 GB）需要 $\sim$42 ms，这直接决定了单步 Decode 的延迟下界。**增大 Batch Size 是提升 GEMV 计算密度、从 Memory-bound 向 Compute-bound 迁移的核心手段。**

## 第 2 章·参考答案：CUDA Kernel 开发与优化

---

### 2.1 基础 Kernel 实现

---

**Q10. 手写 Warp-level Reduce（Sum / Max）：使用 `__shfl_xor_sync` 实现，说明为什么比 Shared Memory Reduce 更快？**

**实现：**

```cpp
// Warp-level Sum Reduce，返回 Warp 内所有线程的总和
__device__ float warp_reduce_sum(float val) {
    // FULL_MASK = 0xffffffff，表示 Warp 内所有 32 个线程参与
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;  // 所有线程均持有最终结果
}

// Warp-level Max Reduce
__device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}
```

**`__shfl_xor_sync` 原理：** 线程 $i$ 与线程 $i \oplus \text{offset}$ 交换寄存器值，经过 $\log_2 32 = 5$ 轮后，每个线程持有全 Warp 的 Reduce 结果。

**为什么比 Shared Memory Reduce 更快：**

|对比项|Shared Memory Reduce|Warp Shuffle Reduce|
|---|---|---|
|数据路径|Register → Shared Memory → Register|Register → Register（寄存器直传）|
|同步开销|需要 `__syncthreads()`（Block 级屏障）|仅需 `__syncwarp()`（Warp 级，开销更小）|
|延迟|~20–30 cycles（Shared Memory 访问）|~1–4 cycles（寄存器级传输）|
|Bank Conflict 风险|存在|无|

---

**Q11. 手写 Block-level Reduce，需要处理哪些边界情况？**

**实现（以 Sum 为例）：**

```cpp
__device__ float block_reduce_sum(float val) {
    // 每个 Warp 先做 Warp-level Reduce
    val = warp_reduce_sum(val);

    // 用 Shared Memory 汇聚各 Warp 的结果
    __shared__ float warp_results[32];  // 最多 32 个 Warp/Block
    int lane   = threadIdx.x % 32;      // Warp 内线程索引
    int warpId = threadIdx.x / 32;      // Warp 索引

    // 每个 Warp 的 lane 0 写入 Shared Memory
    if (lane == 0)
        warp_results[warpId] = val;
    __syncthreads();

    // 第一个 Warp 对各 Warp 结果再做 Reduce
    int num_warps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < num_warps) ? warp_results[lane] : 0.0f;
    if (warpId == 0)
        val = warp_reduce_sum(val);

    return val;  // 线程 0 持有最终结果
}
```

**必须处理的边界情况：**

1. **Block 大小非 32 整数倍**：最后一个 Warp 的线程数不足 32，加载 `warp_results` 时需用 `threadIdx.x < num_warps` 保护，越界线程填 0（Sum）或 $-\infty$（Max）。
2. **`__syncthreads()` 位置**：必须在所有 Warp 写入 Shared Memory **之后**，读取之前。
3. **Warp 数超过 32**：Block 大小最大 1024 线程 = 32 Warp，`warp_results[32]` 足够；若动态 Block Size，需运行时计算。
4. **单 Warp Block**：`num_warps == 1` 时跳过第二轮 Reduce，直接返回。

---

**Q12. 如何实现 numerically stable 的 Online Softmax？推导 3-pass → 2-pass → 1-pass 的演化。**

**标准 Softmax（不稳定）：**

$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

当 $x_i$ 较大时 $e^{x_i}$ 上溢（overflow）。

**3-pass 稳定版本：**

- Pass 1：求全局最大值 $m = \max_i x_i$
- Pass 2：计算 $d = \sum_i e^{x_i - m}$
- Pass 3：输出 $e^{x_i - m} / d$

**2-pass 合并（Pass 1 + Pass 2 合并）：**

维护两个状态 $(m, d)$，对新元素 $x_j$：

$$m' = \max(m, x_j), \quad d' = d \cdot e^{m - m'} + e^{x_j - m'}$$

合并规则（两个分块 $(m_1, d_1)$、$(m_2, d_2)$ 的 Merge）：

$$m = \max(m_1, m_2), \quad d = d_1 \cdot e^{m_1 - m} + d_2 \cdot e^{m_2 - m}$$

**1-pass（Online Softmax，FlashAttention 的核心）：**

在遍历序列的同时**同步更新输出**。维护当前最大值 $m$ 和归一化因子 $d$，当 $m$ 更新时，对已计算的部分输出乘以修正因子 $e^{m_{\text{old}} - m_{\text{new}}}$：

$$O \leftarrow O \cdot e^{m_{\text{old}} - m_{\text{new}}}$$

此技巧使 Attention 的 Softmax 可以在单次遍历 KV Tile 时完成，无需将整个序列载入 SRAM，是 FlashAttention 实现 $O(N)$ 显存复杂度的关键（见 Q23）。

---

**Q13. 实现 Fused RMSNorm Kernel：为什么要 Fuse，省去了哪些 Global Memory 访问？**

**RMSNorm 公式：**

$$\text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}} \cdot \gamma_i$$

**非 Fused 实现的 Global Memory 访问：**

1. 读取 $x$（$d$ 个元素）→ 计算 RMS → 写回中间结果
2. 再次读取 $x$ 和中间结果 → 归一化 → 写出

共 **3 次 HBM 访问**（2 读 1 写 + 1 读 1 写）。

**Fused Kernel 实现：**

```cpp
__global__ void fused_rms_norm(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ out,
    int d, float eps)
{
    // 1. 每个线程负责若干元素，寄存器中累积平方和
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        sum_sq += x[i] * x[i];

    // 2. Block-level Reduce 求总平方和
    sum_sq = block_reduce_sum(sum_sq);

    // 3. 计算 RMS（单次除法）
    float rms_inv = rsqrtf(sum_sq / d + eps);

    // 4. 归一化并写出（x 只读一次）
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        out[i] = x[i] * rms_inv * gamma[i];
}
```

**收益：HBM 访问从 3 次降为 1 次读 + 1 次写**，对于 Memory-bound 的 Norm 算子，加速比接近 $3\times$。

---

**Q14. LayerNorm 的 Welford 在线算法如何实现？**

**问题：** 计算方差的朴素公式 $\text{Var} = E[x^2] - (E[x])^2$ 在数值上不稳定（两个大数相减）。Welford 算法以单次遍历在线更新均值与方差，数值稳定。

**Welford 递推（对第 $n$ 个元素 $x_n$）：**

$$\delta_1 = x_n - \mu_{n-1}$$ $$\mu_n = \mu_{n-1} + \frac{\delta_1}{n}$$ $$\delta_2 = x_n - \mu_n$$ $$M_n = M_{n-1} + \delta_1 \cdot \delta_2 \quad \Rightarrow \quad \text{Var}_n = \frac{M_n}{n}$$

**并行 Welford Merge（两个分块 $(n_a, \mu_a, M_a)$ 和 $(n_b, \mu_b, M_b)$ 合并）：**

$$n = n_a + n_b, \quad \delta = \mu_b - \mu_a$$ $$\mu = \mu_a + \delta \cdot \frac{n_b}{n}$$ $$M = M_a + M_b + \delta^2 \cdot \frac{n_a \cdot n_b}{n}$$

在 Kernel 中，每个线程用 Welford 累积局部统计量，再通过 Warp/Block Reduce 合并，最终一次性得到全局均值和方差，只需遍历数据一次。

---

### 2.2 GEMM 优化

---

**Q15. 朴素 GEMM 的瓶颈是什么？Tiled GEMM 的核心思路？**

**朴素 GEMM 瓶颈：**

计算 $C = A \times B$，$A \in \mathbb{R}^{M \times K}$，$B \in \mathbb{R}^{K \times N}$。每个输出元素 $C[i][j]$ 需读取 $A$ 的第 $i$ 行（$K$ 个元素）和 $B$ 的第 $j$ 列（$K$ 个元素）。

- 朴素实现中，$A$ 被重复读取 $N$ 次，$B$ 被重复读取 $M$ 次，全部来自 HBM。
- 算术强度 $I \approx \frac{2MNK}{2(MK + NK)} \approx \frac{MN}{M+N}$，当 $M, N$ 小时极低。

**Tiled GEMM（Shared Memory Tiling）：**

将 $A$、$B$ 分块（Tile），每次将大小为 $T \times T$ 的子块载入 Shared Memory，Block 内所有线程复用该子块计算。

```
对于每个输出 Tile C[bm, bn]：
    for k_tile in range(K / T):
        载入 A[bm, k_tile] → smem_A   // 每元素只从 HBM 读一次
        载入 B[k_tile, bn] → smem_B
        __syncthreads()
        for k in range(T):
            reg_c += smem_A[ty][k] * smem_B[k][tx]  // 在 Shared Memory 计算
        __syncthreads()
```

**收益：** 每个 HBM 读取被 $T$ 个线程复用，算术强度提升至 $\approx T/2$。典型 $T = 128$ 时，$I \approx 64$，大幅降低 HBM 压力。

---

**Q16. 什么是 Double Buffering？如何用 `cp.async` / TMA 实现异步预取？**

**Double Buffering 原理：**

将 Shared Memory 分为两个 Buffer（Ping / Pong）。在计算当前 Tile 的同时，**异步预取下一个 Tile**，使计算与数据搬运流水并行，消除等待。

```
Buffer A: [Tile 0 数据]  →  计算 Tile 0
Buffer B:               →  异步加载 Tile 1

下一轮：
Buffer A:               →  异步加载 Tile 2
Buffer B: [Tile 1 数据]  →  计算 Tile 1
```

**`cp.async`（Ampere+）：**

绕过寄存器，直接将 Global Memory 数据异步搬运至 Shared Memory，不阻塞 CUDA Core 执行：

```cpp
// 异步将 Global Memory → Shared Memory，不阻塞
__pipeline_memcpy_async(smem_dst, gmem_src, sizeof(float4));
__pipeline_commit();      // 提交异步操作
// ... 执行当前 Tile 计算 ...
__pipeline_wait_prior(0); // 等待最近一次提交完成
```

**TMA（Tensor Memory Accelerator，Hopper+）：**

比 `cp.async` 更高级的异步引擎，支持多维 Tensor 的批量搬运，完全由硬件控制地址生成，进一步解放 CUDA Core 的地址计算负担（见 Q122）。

---

**Q17. Tensor Core（WMMA / MMA / WGMMA）的使用方式与限制？Hopper WGMMA 与 Ampere MMA 的区别？**

**三代 API 对比：**

|API|架构|粒度|精度支持|调用者|
|---|---|---|---|---|
|WMMA（`nvcuda::wmma`）|Volta+|Warp 级（16×16×16）|FP16, BF16, INT8|C++ Fragment API|
|MMA PTX（`mma.sync`）|Ampere+|Warp 级（16×8×16 等）|FP16, BF16, FP8, INT8|PTX 内联汇编|
|WGMMA PTX（`wgmma.mma_async`）|Hopper|**Warpgroup 级**（128 线程，64×8×16 等）|FP16, BF16, FP8|PTX 内联汇编|

**Ampere MMA vs Hopper WGMMA 的关键区别：**

|特性|Ampere MMA|Hopper WGMMA|
|---|---|---|
|执行粒度|1 Warp（32 线程）|1 Warpgroup（4 Warp = 128 线程）|
|数据来源|寄存器 → 寄存器|**Shared Memory → 寄存器**（异步）|
|与 TMA 配合|不直接支持|原生支持，形成 Produce-Consume 流水|
|峰值利用率|中等|更高（配合 Warp Specialization）|

**使用限制：**

- 矩阵形状必须满足硬件支持的固定尺寸（如 16×8×16）。
- 寄存器消耗大，过多 MMA 操作易导致 Register Spill 到 Local Memory（性能骤降）。
- WGMMA 要求 Shared Memory 地址满足特定对齐与 Swizzle 要求。

---

**Q18. cuBLAS vs CUTLASS vs 手写 Kernel 的选型依据？**

|方案|适用场景|优势|劣势|
|---|---|---|---|
|**cuBLAS**|标准方形 GEMM，Batch GEMM|开箱即用，NVIDIA 深度优化，峰值性能|不可定制，无法 Fuse 其他算子|
|**CUTLASS**|需要定制 Epilogue、Fuse 算子、非标准形状|高度可组合，支持 Sparse/MoE，模板化|编译慢，学习曲线陡峭|
|**手写 Kernel**|特殊访存模式、极端优化需求|完全控制|开发成本极高，维护难|
|**Triton**|快速验证、跨硬件|Python 语法，自动调优|极限性能略逊于手写 CUDA|

**何时必须手写：**

- 算子融合逻辑极为复杂，CUTLASS Epilogue 无法表达（如 FlashAttention 的 Online Softmax 更新）。
- 访存模式非常规（如 PagedAttention 的非连续 KV Block 读取）。

---

**Q19. GEMM-SplitK 分解的适用场景？**

**问题背景：** Decode 阶段 Batch Size 小（$M = 1 \sim 32$），GEMM 形状为"瘦高矩阵"（$M \ll K$），单个 CTA（线程块）无法充分占满 GPU 的所有 SM，导致低 SM 利用率。

**SplitK 原理：** 将 $K$ 维度切分为 $S$ 份，每份由独立 CTA 计算部分和，最终通过 Reduction 合并：

$$C = \sum_{s=0}^{S-1} A[:, s \cdot K/S : (s+1) \cdot K/S] \times B[s \cdot K/S : (s+1) \cdot K/S, :]$$

- **收益**：CTA 数量从 $\lceil M/T_M \rceil \times \lceil N/T_N \rceil$ 扩大为 $S$ 倍，SM 利用率显著提升。
- **代价**：额外的 Reduction 步骤（通常用原子操作或独立 Reduction Kernel），以及 $S$ 倍的 $B$ 矩阵读取量。
- **典型值**：$S = 8 \sim 64$，在 Batch=1 的 Decode 场景可提升吞吐 2×–4×。

---

### 2.3 Kernel Fusion

---

**Q20. Kernel Fusion 的本质收益？FlashAttention 的 Fusion 策略？**

**本质收益：消除中间张量的 HBM Round-trip。**

以 `Softmax(QK^T/√d) · V` 为例，非 Fused 实现：

```
Q, K → [HBM写] S = QK^T → [HBM读] P = Softmax(S) → [HBM写] → [HBM读] O = PV
```

4 次 HBM 访问中间结果（$S$ 和 $P$ 均为 $N \times N$ 矩阵）。

**FlashAttention Fusion 策略：**

将 $Q, K, V$ 分 Tile 载入 SRAM，在片上完成 `QK^T → Softmax → PV` 的全部计算，中间结果 $S, P$ **始终驻留在 SRAM，不写回 HBM**。最终只写出输出 $O$（$N \times d$）。

- HBM 读写量：$O(N \cdot d)$（线性），而非 $O(N^2)$。
- 计算量不变（仍为 $O(N^2 \cdot d)$），但 IO 大幅降低，Attention 从 Memory-bound 变为接近 Compute-bound。

---

**Q21. 什么样的算子适合 Fusion？什么情况下 Fusion 有害？**

**适合 Fusion 的条件：**

- 相邻算子均为 **Memory-bound**（如 Elementwise、Norm、Activation），Fusion 将多次 HBM 读写压缩为一次。
- 算子间存在**生产-消费关系**，中间结果可在寄存器或 Shared Memory 中直接传递。
- 中间结果**体积远大于**计算量（高 IO/FLOPs 比值）。

**Fusion 有害的情况：**

|场景|原因|
|---|---|
|寄存器压力过大（Register Spilling）|融合算子后每线程需维护更多寄存器变量，超过上限后溢出到 Local Memory（HBM），性能反而下降|
|Occupancy 大幅降低|寄存器/Shared Memory 占用增加，每 SM 可驻留的 Warp 数减少，延迟隐藏能力下降|
|两个算子均 Compute-bound|Fusion 对 IO 无收益，徒增代码复杂度|
|算子形状不匹配|如 Reduction 后接 Broadcast，线程块映射方式不同，强行 Fusion 导致低效|

---

**Q22. CUDA Graph 的作用：如何消除 Kernel Launch Overhead？适用哪些场景？**

**Kernel Launch Overhead 来源：**

每次调用 CUDA Kernel（`kernel<<<grid, block>>>`）都需经过 CPU 驱动层，产生约 **5–20 μs** 的延迟。Decode 阶段每步执行数十个小 Kernel，Launch Overhead 可占端到端延迟的 10%–30%。

**CUDA Graph 原理：**

1. **录制阶段（Capture）**：在 `cudaStreamBeginCapture` 和 `cudaStreamEndCapture` 之间执行的所有 Kernel、内存操作被录制为一张有向无环图（DAG），不实际执行。
2. **实例化（Instantiate）**：将 Graph 编译为可执行的优化形式。
3. **执行阶段（Launch）**：调用 `cudaGraphLaunch`，整个 Graph 以**单次 CPU 调用**提交给 GPU，消除逐 Kernel 的 Launch 开销。

```cpp
// 录制
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernel_A<<<grid, block, 0, stream>>>(...);
kernel_B<<<grid, block, 0, stream>>>(...);
// ... 更多 Kernel
cudaStreamEndCapture(stream, &graph);

// 实例化
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

// 重复执行（只需一次 CPU 调用）
cudaGraphLaunch(graphExec, stream);
```

**适用场景：**

- Decode 阶段：计算图固定（每步形状相同），可复用同一 Graph，仅更新输入指针。
- 高频小 Kernel 场景：Launch Overhead 占比高时收益显著。

**不适用场景：**

- Prefill 阶段：输入序列长度每次不同，计算图形状动态变化，无法复用。
- 包含 CPU 条件分支的动态控制流（如动态 Batch 调度）。

## 第 3 章·参考答案：Attention 机制优化

---

### 3.1 FlashAttention 系列

---

**Q23. 标准 Attention 的内存复杂度为 $O(N^2)$，FlashAttention 如何将其降为 $O(N)$ SRAM 占用？核心思想（Tiling + Online Softmax）？**

**标准 Attention 的问题：**

$$\text{Attention}(Q, K, V) = \text{Softmax}!\left(\frac{QK^T}{\sqrt{d}}\right) V$$

朴素实现需将 $S = QK^T \in \mathbb{R}^{N \times N}$ 完整写入 HBM，再读回做 Softmax，再写出 $P = \text{Softmax}(S)$，再读回计算 $O = PV$。

- HBM 读写量：$O(N^2)$（中间矩阵 $S, P$）
- 对于 $N = 8192$，FP16 下 $S$ 占 $8192^2 \times 2 \approx 128\text{ MB}$，远超 H100 的 SRAM 容量（228 KB/SM）。

**FlashAttention 核心思想：Tiling + Online Softmax**

**① Tiling（分块计算）：**

将 $Q$ 按行分块（每块 $B_r$ 行），将 $K, V$ 按列分块（每块 $B_c$ 列）。每次只将一个 $Q$ 块和一个 $K/V$ 块载入 SRAM，在片上完成局部 Attention 计算。

**② Online Softmax（单次遍历完成归一化）：**

遍历所有 $K/V$ 块时，维护每行的 running max $m$ 和 running sum $\ell$：

对第 $j$ 个 $K$ 块，计算局部得分 $\tilde{S}_j = Q_i K_j^T / \sqrt{d}$，更新：

$$m_j = \max(m_{j-1},\ \text{rowmax}(\tilde{S}_j))$$

$$\ell_j = e^{m_{j-1} - m_j} \cdot \ell_{j-1} + \text{rowsum}(e^{\tilde{S}_j - m_j})$$

$$O_j = \text{diag}(e^{m_{j-1} - m_j})^{-1} \cdot O_{j-1} + e^{\tilde{S}_j - m_j} \cdot V_j$$

遍历结束后：$O = \text{diag}(\ell)^{-1} \cdot O$，得到正确归一化结果。

**复杂度对比：**

|指标|标准 Attention|FlashAttention|
|---|---|---|
|HBM 读写量|$O(N^2)$|$O(N \cdot d)$（线性）|
|SRAM 占用|$O(N^2)$（需存 $S, P$）|$O(B_r \cdot d + B_c \cdot d)$（常数级）|
|FLOPs|$O(N^2 \cdot d)$|$O(N^2 \cdot d)$（相同）|

**Tile 大小选择：** 需满足 $B_r \cdot d + B_c \cdot d \leq \text{SRAM}$，H100 上典型 $B_r = B_c = 64 \sim 128$。

---

**Q24. FlashAttention-2 相比 FA-1 的改进点？**

FA-1 存在两个主要低效问题，FA-2 针对性解决：

**改进 1：减少非 GEMM FLOPs（Rescaling 操作优化）**

FA-1 在每个 KV 块处理后对 $O$ 做 rescale（乘以修正因子），每个元素都有额外的乘法。FA-2 将 rescale **推迟到最后一步**，遍历过程中只更新 $m$ 和 $\ell$，最终一次性对 $O$ 做归一化。非 GEMM FLOPs 减少约 **50%**。

**改进 2：改进 Warp 并行策略（减少 Warp 间通信）**

FA-1 将不同的 $K/V$ 块分配给不同 Warp 并行处理，Warp 间需通过 Shared Memory 通信合并 $O, \ell, m$，产生同步开销。

FA-2 改为：**按 $Q$ 的行分块分配给不同 Warp**，每个 Warp 独立负责完整的 $Q$ 行，遍历所有 $K/V$ 块。Warp 间**不需要通信**，消除了 Shared Memory 同步瓶颈。

**改进 3：支持 Causal Masking 的 Tile 跳过**

对于因果掩码（Causal Mask），$Q$ 块 $i$ 只需处理 $K$ 块 $j \leq i$ 的部分，FA-2 显式跳过全为 $-\infty$ 的 Tile，节省约 **50% 的计算量**（训练/Prefill 阶段）。

**结果：** FA-2 在 A100 上的 MFU 从 FA-1 的 ~25–35% 提升至 ~50–73%（FP16）。

---

**Q25. FlashAttention-3 在 Hopper 架构上的改进：Warp Specialization、异步流水线、WGMMA？**

FA-3 专为 H100 的三项新硬件特性设计：

**改进 1：Warp Specialization（Warp 专业化）**

将 Warpgroup 分为两类角色：

- **Producer Warp**：专责通过 TMA 异步搬运 $Q, K, V$ 数据到 Shared Memory，不参与计算。
- **Consumer Warp**：专责执行 WGMMA 矩阵乘计算，不参与数据搬运。

两类 Warp 通过 **异步 Barrier**（`cuda::barrier`）协调，形成生产者-消费者流水。

**改进 2：异步流水线（两级 Double Buffering）**

```
Stage 1: Producer 加载 K_0, V_0 → SRAM_A
Stage 2: Consumer 计算 QK_0^T (WGMMA)   |  Producer 加载 K_1, V_1 → SRAM_B
Stage 3: Consumer 计算 QK_1^T (WGMMA)   |  Producer 加载 K_2, V_2 → SRAM_A
...
```

计算与数据搬运完全重叠，消除等待气泡。

**改进 3：WGMMA（Warpgroup-level MMA）**

相比 Ampere 的 Warp 级 MMA，WGMMA 以 128 线程（4 Warp）为单位执行，直接从 Shared Memory 读取数据，寄存器使用更高效，峰值 Tensor Core 利用率更高。

**改进 4：Softmax 与 GEMM 的流水重叠**

FA-3 将 Softmax（非 GEMM 操作）与下一轮 WGMMA 重叠执行，进一步隐藏非矩阵计算的延迟。

**结果：** FA-3 在 H100 上的 FP8 Attention 吞吐接近硬件峰值带宽的 **75%**。

---

**Q26. 为什么 Decode 阶段的 Attention 退化为 GEMV 问题？此时 FA 的收益是否仍然显著？**

**退化原因：**

Decode 阶段每步只生成 1 个新 Token，$Q \in \mathbb{R}^{1 \times d}$（或 Batch × 1 × d），而 $K, V \in \mathbb{R}^{S \times d}$（历史 KV Cache，序列长度 $S$ 很长）。

Attention 计算变为：

$$\text{score} = Q K^T \in \mathbb{R}^{1 \times S} \quad \Rightarrow \quad \text{GEMV（向量 × 矩阵）}$$

$$O = \text{Softmax}(\text{score}) \cdot V \in \mathbb{R}^{1 \times d} \quad \Rightarrow \quad \text{GEMV（向量 × 矩阵）}$$

**此时 FA 的收益分析：**

|指标|Prefill（$Q \in \mathbb{R}^{N \times d}$）|Decode（$Q \in \mathbb{R}^{1 \times d}$）|
|---|---|---|
|中间矩阵 $S$ 大小|$N \times S$（可能很大）|$1 \times S$（仅一行，很小）|
|HBM 节省|显著（$O(N \cdot S)$ → $O(N \cdot d)$）|有限（$S$ 已很小）|
|瓶颈|Memory-bound（读 $K, V$）|Memory-bound（读 $K, V$）|
|FA 收益|**显著**（IO 减少为主）|**有限**（主要收益是数值稳定性，IO 节省较小）|

**Decode 阶段的主要优化方向**不是 FA，而是：

- **GQA / MQA**：减少 KV Cache 大小（见 Q27）。
- **PagedAttention**：减少 KV Cache 碎片（见 Q32）。
- **KV Cache 量化**：降低 HBM 读取量（见 Q36）。
- **Fused Decode Attention Kernel**：如 vLLM 的 `paged_attention_v2`，专为非连续 KV 访问优化。

---

### 3.2 Attention 变体

---

**Q27. MHA vs GQA vs MQA 的区别？GQA 在 KV Cache 占用上的收益推导？**

**三种 Attention 的 KV 头数对比：**

|方案|Q 头数|K/V 头数|KV Cache 大小|代表模型|
|---|---|---|---|---|
|MHA（Multi-Head Attention）|$H$|$H$|基准 $1\times$|GPT-2、BERT|
|GQA（Grouped Query Attention）|$H$|$H / G$（$G$ 为分组数）|$1/G$|Llama-2/3、Mistral|
|MQA（Multi-Query Attention）|$H$|$1$|$1/H$|Falcon、PaLM|

**GQA KV Cache 节省推导：**

单请求、序列长度 $S$、头维度 $d$、数据类型 dtype 的 KV Cache：

$$M_{\text{MHA}} = 2 \times L \times H \times d \times S \times \text{sizeof(dtype)}$$

$$M_{\text{GQA}} = 2 \times L \times \frac{H}{G} \times d \times S \times \text{sizeof(dtype)} = \frac{M_{\text{MHA}}}{G}$$

以 Llama-3 70B 为例：$H = 64$，$G = 8$（即 8 个 Q 头共享 1 个 KV 头），KV Cache 减少为 MHA 的 $1/8$。

**精度影响：** GQA 相比 MHA 精度损失极小（通常 < 0.5 perplexity），MQA 损失稍大但仍可接受，工业界普遍采用 GQA。

**GQA 的计算方式：** 每组 $G$ 个 Q 头共享同一组 K/V，计算时将 K/V 广播（或通过 `expand`）至 $G$ 个 Q 头，不改变 Attention 的计算结构。

---

**Q28. MLA（Multi-head Latent Attention）的核心思路：低秩压缩 KV 的原理与 DeepSeek 中的实现？**

**动机：** GQA 通过减少 KV 头数降低 KV Cache，但以牺牲模型表达能力为代价。MLA 在**保持完整模型容量**的前提下压缩 KV Cache。

**核心思路：低秩联合压缩 K 和 V**

标准 MHA 中 K、V 分别投影：$K = X W_K,\ V = X W_V$，KV Cache 存储 $K, V \in \mathbb{R}^{S \times (H \cdot d)}$。

MLA 引入低秩潜变量 $c^{KV} \in \mathbb{R}^{S \times d_c}$（$d_c \ll H \cdot d$）：

$$c^{KV} = X W^{DKV} \quad \text{（Down-projection，压缩）}$$

$$K = c^{KV} W^{UK},\quad V = c^{KV} W^{UV} \quad \text{（Up-projection，推断时展开）}$$

**KV Cache 只存 $c^{KV}$**，而非完整的 $K, V$。

**显存收益：**

- MHA KV Cache：$2 \times H \times d \times S$
- MLA KV Cache：$d_c \times S$（$d_c$ 通常为 $H \times d / 8 \sim / 16$）

**DeepSeek-V2/V3 参数（示例）：** $H = 128$，$d = 128$，$d_c = 512$，压缩比约 $128 \times 128 \times 2 / 512 \approx 64\times$（相比 MHA）。

**推理时的计算策略：**

- 可在 Prefill 时预计算并缓存 $c^{KV}$，Decode 时只需存 $c^{KV}$，不展开 $K, V$，通过矩阵吸收（Absorption）技巧将 $W^{UK}$ 融入 $W_Q$，实现无额外计算开销。

---

**Q29. Sparse Attention（Sliding Window、BigBird）的适用场景？**

**动机：** 标准 Attention 计算复杂度 $O(N^2)$，对超长序列（$N > 32k$）代价极高。Sparse Attention 通过**限制每个 Token 只关注部分 Token**，将复杂度降至 $O(N \cdot k)$（$k$ 为关注窗口大小）。

**主要模式：**

|模式|原理|代表方案|
|---|---|---|
|**Sliding Window（局部窗口）**|每个 Token 只关注前后 $w/2$ 个 Token，形成带状 Attention 矩阵|Longformer、Mistral|
|**Global Token（全局 Token）**|特定 Token（如 `[CLS]`）关注全序列，其余只关注局部|BigBird、Longformer|
|**随机稀疏**|每个 Token 随机关注 $r$ 个 Token|BigBird|
|**Strided（跨步）**|以固定步长采样 Token|Sparse Transformer|

**Sliding Window Attention（Mistral/Mixtral）：**

窗口大小 $w$，每个 Token 只 Attend 最近 $w$ 个 Token，多层叠加后感受野为 $w \times L$（$L$ 为层数），可覆盖较远依赖。

- 优点：计算复杂度 $O(N \cdot w)$，KV Cache 大小从 $O(N)$ 降为 $O(w)$（每层只需保存 $w$ 个 KV）。
- 缺点：超出窗口的长距离依赖完全丢失（见 Q103 Attention Sink 问题）。

**BigBird：** 结合局部窗口 + 全局 Token + 随机稀疏三种模式，理论上可近似任意全注意力，适合文档级别的长文本理解任务。

**适用场景总结：**

- 超长文档理解、代码分析（$N > 32k$）：Sliding Window 足够，局部依赖为主。
- 需要全局语义整合（如问答、摘要）：BigBird 的全局 Token 机制更优。
- 生成任务（Decoder-only）：Sliding Window 结合 Attention Sink（StreamingLLM）实现无限流式生成（见 Q37）。

## 第 4 章·参考答案：KV Cache 管理

---

### 4.1 核心机制

---

**Q30. KV Cache 的作用与显存增长规律：推导单请求 $S$ tokens 的 KV Cache 显存占用公式。**

**KV Cache 的作用：**

自回归解码时，第 $t$ 步生成的 Token 需要 Attend 前 $t-1$ 个 Token 的 Key 和 Value。若不缓存，每步都需对全部历史 Token 重新计算 $K, V$，计算量随序列增长为 $O(S^2)$。KV Cache 将历史 $K, V$ 存入显存，每步只计算新 Token 的 $K, V$ 并追加，将计算量降为 $O(S)$，以**显存换计算**。

**显存占用推导：**

模型参数：层数 $L$，注意力头数 $H$，头维度 $d$，数据类型 dtype（FP16 = 2 字节，FP8 = 1 字节）。

每层、每个 Token 存储 $K$ 和 $V$ 各一份：

$$M_{\text{KV}} = \underbrace{2}_{\text{K+V}} \times L \times H \times d \times S \times \text{sizeof(dtype)}$$

**具体示例（Llama-3 70B，FP16，GQA H/G=8）：**

- $L = 80$，$H_{\text{KV}} = 8$（GQA），$d = 128$，sizeof(FP16) = 2 字节
- 单请求、$S = 4096$ tokens：

$$M = 2 \times 80 \times 8 \times 128 \times 4096 \times 2 = 2 \times 80 \times 8 \times 128 \times 4096 \times 2 \approx 1.34 \text{ GB}$$

- 若 Batch Size = 32，则 KV Cache 总占用 $\approx 42.9$ GB，接近 H100 80 GB 显存的 **54%**。

**增长规律：** KV Cache 与序列长度 $S$ 和批次大小 $B$ 均呈**线性增长**，是制约推理服务吞吐量的核心瓶颈。

---

**Q31. 为什么传统框架的 KV Cache 存在严重的内存碎片？**

**传统方案：** 为每个请求预分配一块**连续的最大长度**显存（按最大序列长度 $S_{\max}$ 预分配），生成过程中逐步填充。

**Internal Fragmentation（内部碎片）：**

预分配按 $S_{\max}$ 分配，而实际生成长度 $S_{\text{actual}} \leq S_{\max}$。已分配但未使用的部分形成内部碎片。

$$\text{碎片率} = 1 - \frac{S_{\text{actual}}}{S_{\max}}$$

若 $S_{\max} = 2048$，平均生成长度 $S_{\text{actual}} = 256$，碎片率高达 **87.5%**。

**External Fragmentation（外部碎片）：**

不同请求的 KV Cache 大小不同，完成的请求释放显存后留下大小各异的空洞，新请求所需的连续显存无法从碎片中拼凑，即使总空闲显存充足也无法分配。

```
显存示意（碎片化状态）：
[请求A: 512B][空闲: 200B][请求B: 1024B][空闲: 300B][请求C: 256B][空闲: 512B]
→ 总空闲 1012B，但无法满足需要 600B 连续空间的新请求
```

**后果：** 实际 GPU 显存利用率仅约 **20–40%**（vLLM 论文数据），大量显存被碎片浪费，制约并发请求数。PagedAttention 正是为解决此问题而设计（见 Q32）。

---

### 4.2 PagedAttention

---

**Q32. PagedAttention 的核心思路：类比 OS 虚拟内存，Block 大小如何选择？**

**核心思路：**

借鉴操作系统的**虚拟内存分页机制**：将 KV Cache 划分为固定大小的**物理 Block**（Physical Block），每个 Block 存储固定数量（$B$ 个，典型值 **16 tokens**）的 KV 向量；为每个请求维护一张**逻辑-物理 Block 映射表**（Block Table），逻辑上连续的 KV 序列映射到任意分散的物理 Block。

```
请求 A 的 Block Table：
逻辑块 0 → 物理块 7
逻辑块 1 → 物理块 3
逻辑块 2 → 物理块 15
...

请求 B 的 Block Table：
逻辑块 0 → 物理块 2
逻辑块 1 → 物理块 7  ← 与请求 A 共享（Prefix Sharing）
...
```

**消除碎片的原理：**

- **无 Internal Fragmentation**：仅最后一个逻辑 Block 可能不满，浪费最多 $(B-1)$ 个 Token 槽，期望浪费仅 $(B-1)/2 \approx 7.5$ 个 Token。
- **无 External Fragmentation**：所有物理 Block 大小相同，释放后可立即被任意新请求复用，如同 OS 的固定大小页框。

**Block 大小 $B$ 的选择权衡：**

|$B$ 值|优点|缺点|
|---|---|---|
|小（如 1）|内部碎片极少|Block Table 大，索引开销大，Cache 局部性差|
|大（如 256）|索引开销小，访存连续性好|内部碎片增多，Prefix Sharing 粒度粗|
|**16（典型值）**|平衡两者，与 GPU 缓存行对齐|—|

$B = 16$ 时，每个 Block 的 KV 数据大小为 $16 \times H \times d \times 2 \times \text{sizeof}$，通常为 512B–4KB，与 L2 Cache Line 对齐。

---

**Q33. PagedAttention 如何支持 Prefix Sharing（多请求共享同一 Prompt 的 KV Block）？**

**Prefix Sharing 原理：**

若多个请求拥有相同的前缀 Prompt（如 System Prompt），这些 Prompt Token 对应的 KV Block 内容完全相同。PagedAttention 通过**引用计数（Reference Counting）**让多个请求的 Block Table 指向**同一组物理 Block**，该 Block 只在显存中存储一份。

```
System Prompt: "You are a helpful assistant..."（256 tokens = 16 个 Block）

请求 A Block Table: [共享Block 0..15] → [私有Block 16, 17, ...]
请求 B Block Table: [共享Block 0..15] → [私有Block 23, 24, ...]
请求 C Block Table: [共享Block 0..15] → [私有Block 31, 32, ...]

物理显存中 Block 0..15 只有 1 份，被三个请求共享
```

**Copy-on-Write（写时复制）：**

共享 Block 为**只读**。当某请求需要修改（实际推理中 KV 追加只写新 Block，共享部分不修改），若发生写操作则触发 Copy-on-Write，为该请求复制一份私有副本。

**收益量化：**

- 若 System Prompt 长度 $S_p = 1024$ tokens，Batch Size = 64，则节省 KV Cache = $63 \times S_p$ 份，约 **98.4% 的 Prefix KV 显存复用**。
- SGLang 的 RadixAttention 将此思路推广为**前缀树（Radix Tree）结构**，支持任意公共前缀的自动识别与共享（见 Q66）。

---

**Q34. 相比连续 KV Buffer，PagedAttention 的 Attention Kernel 有哪些额外开销？**

标准 Attention Kernel 假设 KV 数据在显存中**连续存储**，可通过简单的指针偏移访问。PagedAttention 的非连续存储引入以下额外开销：

**① Block Table 查找开销：**

每次访问 KV 时需通过 Block Table 将逻辑地址转换为物理地址：

```cpp
// 每个 KV 访问需额外一次查表
int block_idx   = token_idx / block_size;
int block_offset = token_idx % block_size;
int physical_block = block_table[seq_id][block_idx];
float* kv_ptr = kv_cache + physical_block * block_size * kv_dim + block_offset * kv_dim;
```

增加约 1–2 次整数运算和 1 次全局内存读取（Block Table 本身在 HBM 中）。

**② Cache 局部性下降：**

连续 KV Buffer 的访问模式对 L2 Cache 友好（空间局部性好）；分散 Block 的访问模式可能导致更多 L2 Cache Miss，尤其在序列长、Block 分散时。

**③ Warp 内地址计算不一致：**

同一 Warp 的不同线程可能访问不同物理 Block，难以完全合并访问（Memory Coalescing 下降）。

**④ 量化开销（可忽略）：**

Block Table 本身占用极小（每个 Block 一个 int32，序列 4096 tokens / 16 = 256 个 Block，仅 1 KB）。

**实践结论：** vLLM 的测量表明，PagedAttention 相比连续 KV 的 Attention Kernel 性能损失约 **10–20%**，但其带来的内存利用率提升（从 20–40% 提升至 ~90%+）远超该开销，整体吞吐显著提升。

---

### 4.3 KV Cache 压缩

---

**Q35. Token Eviction 方法（H2O、SnapKV）的基本思路？**

**核心动机：** 并非所有历史 Token 对当前生成都同等重要，可以**丢弃低重要性 Token 的 KV**，将 KV Cache 大小限制在预算 $B_{\text{budget}}$ 以内。

**H2O（Heavy Hitter Oracle）：**

基于观察：Attention 分布高度集中，少数 Token（Heavy Hitters）持续获得大部分 Attention 权重。

- **重要性度量：** 累积 Attention Score $s_i = \sum_{t} \alpha_{t,i}$（Token $i$ 被所有后续 Token 关注的总权重）。
- **策略：** 维护大小为 $B_{\text{budget}}$ 的 KV Cache，每步驱逐累积 Score 最低的 Token，保留 Heavy Hitters 和最近的 $r$ 个 Token（Recent Window，防止驱逐刚生成的 Token）。

$$\text{保留集合} = \text{TopK}(s_i) \cup \text{Recent}(r)$$

**SnapKV：**

- 针对**长输入（RAG、长文档）场景**，在 Prefill 阶段即完成压缩，而非逐步驱逐。
- 观察到 Query 的注意力在不同层关注的 Token 位置具有**一致性**（Attention Pattern 可预测）。
- 策略：用输入末尾的若干 Observation Tokens 的 Attention 分布选取重要 KV，Prefill 结束后只保留被选中的 KV，大幅压缩初始 KV Cache 大小。

**精度-压缩率权衡：** H2O 在压缩率 5×–10× 时精度损失通常 < 5%（任务相关），超过 20× 时性能明显下降。

---

**Q36. KV Cache 量化（INT8 / FP8 KV）的精度损失分析？**

**量化方案：**

将 KV Cache 从 FP16（2 字节）量化为 INT8 或 FP8（1 字节），HBM 读取量减半，Decode 阶段带宽压力降低约 **50%**。

**精度损失来源：**

KV 中的 **Value（V）** 数值分布相对平滑，量化误差小；**Key（K）** 存在少量幅值极大的异常值（Outlier），对量化精度影响更大。

**主要量化方案对比：**

|方案|量化粒度|精度损失|实现复杂度|
|---|---|---|---|
|Per-tensor INT8|整个 KV Tensor 共享 Scale|较大（Outlier 影响全局 Scale）|低|
|Per-token INT8|每个 Token 的 KV 独立 Scale|较小|中|
|Per-channel INT8|每个头维度独立 Scale|更小|中|
|FP8（E4M3）|Per-tensor 或 Per-token|介于 FP16 与 INT8 之间|低（H100 原生支持）|

**工程实践：**

- H100 原生支持 FP8 Tensor Core，FP8 KV Cache 无需软件反量化，推理时直接参与计算。
- INT8 KV Cache 需在 Attention 计算前反量化为 FP16，引入额外计算，但带宽节省仍超过反量化开销。
- TensorRT-LLM、vLLM 均支持 FP8 KV Cache，典型精度损失（MMLU 等基准）< **0.5%**。

---

**Q37. StreamingLLM 的 Attention Sink 机制是什么？**

**问题背景：**

Sliding Window Attention（见 Q29）将 KV Cache 限制为最近 $w$ 个 Token，理论上可实现无限长序列生成。但实验发现：**直接丢弃窗口外的早期 Token 会导致困惑度（Perplexity）骤增**，模型输出崩溃。

**Attention Sink 现象：**

分析 Attention 分布发现，**序列最开始的几个 Token（通常是前 4 个）持续获得异常高的 Attention 权重**，无论输入内容如何。这些 Token 被称为 **Attention Sink**（注意力汇聚点）。

**原因：** Softmax 要求所有 Attention 权重之和为 1。当模型不需要关注任何特定 Token 时，多余的权重"涌入"最初的 Token 作为"垃圾桶"（Sink Token）。这是 Softmax 归一化的数学特性导致的，与 Token 的语义内容无关。

**StreamingLLM 解决方案：**

在 Sliding Window 的基础上，**始终保留前 $k$（默认 4）个 Sink Token 的 KV Cache**，不受窗口限制：

$$\text{KV Cache} = \text{Sink Tokens}(k) \cup \text{Recent Tokens}(w)$$

总 KV Cache 大小固定为 $k + w$，可实现**无限长序列流式生成**，且 Perplexity 与全 KV Cache 方案几乎相同（相差 < 0.1）。

**局限性：** 仅适合不依赖远距离历史的生成任务（如对话），对需要长程依赖的任务（如超长文档问答）无法使用。

## 第 5 章·参考答案：调度与批处理策略

---

### 5.1 Batching 机制

---

**Q38. Static Batching 与 Continuous Batching 的区别？后者如何消除 Padding 浪费？**

**Static Batching（静态批处理）：**

将若干请求组成一个固定 Batch，等待 Batch 内**所有请求全部完成**后才释放资源，接受下一批。

```
时间轴：
请求A ████████████████████ (生成 200 tokens，完成)
请求B ████████             (生成 100 tokens，完成) → 等待 A
请求C ████                 (生成 50 tokens，完成)  → 等待 A、B
────────────────────────────────────────────────→ 时间
         Batch 结束，GPU 空转等待最长请求
```

**问题：**

- **Padding 浪费**：短请求完成后 GPU Slot 空转，等待最长请求。
- **Head-of-Line Blocking**：一个超长请求拖慢整个 Batch 的周转时间。
- GPU 利用率低，尤其当请求长度分布方差大时。

**Continuous Batching（连续批处理 / Iteration-level Scheduling）：**

在**每个迭代步（每生成 1 个 Token）后**重新调度，完成的请求立即从 Batch 中移除，新请求立即插入填补空位。

```
时间轴（每列为一个迭代步）：
Slot 0: 请求A ██████████████████████ (200步完成) → 请求E ████...
Slot 1: 请求B ██████████             (100步完成) → 请求D ████████████...
Slot 2: 请求C ████                   (50步完成)  → 请求F ██████████████████...
────────────────────────────────────────────────→ 时间
         无空转，Slot 始终被占用
```

**消除 Padding 的原理：**

每个迭代步，Batch 内所有活跃请求各生成 1 个 Token，计算形状统一为 `[current_batch_size, 1, hidden_dim]`，无需 Padding。请求完成后 Slot 立即回收，新请求的 Prefill 可插入下一步执行。

**收益：** 相比 Static Batching，Continuous Batching 在请求长度分布不均时可将 GPU 利用率提升 **2–8×**（Orca 论文数据）。

---

**Q39. Chunked Prefill 的原理：将 Prefill 拆分为多个 Chunk 与 Decode 交错执行，有何收益与代价？**

**背景问题：**

Continuous Batching 中，当一个长 Prompt 请求进入时，其 Prefill 阶段需要处理大量 Token（如 $S = 8192$），耗时数百毫秒，在此期间**所有 Decode 请求被阻塞**，导致 TPOT（每 Token 输出延迟）出现尖刺。

**Chunked Prefill 原理：**

将长 Prompt 的 Prefill 拆分为大小为 $C$（如 $C = 512$）的 Chunk，**每个迭代步只处理一个 Chunk**，其余计算资源留给 Decode 请求，两者交错执行。

```
不使用 Chunked Prefill：
迭代 1: [Prefill 8192 tokens]  → Decode 请求全部阻塞 ~300ms
迭代 2: [Decode × N]
迭代 3: [Decode × N]
...

使用 Chunked Prefill（C=512）：
迭代 1: [Prefill chunk 0~511]   + [Decode × N]
迭代 2: [Prefill chunk 512~1023] + [Decode × N]
...
迭代16: [Prefill chunk 7680~8191] + [Decode × N]
```

**收益：**

- **降低 TPOT 尖刺**：Decode 请求不再被长 Prefill 完全阻塞，P99 TPOT 显著改善。
- **提升 GPU 利用率**：Prefill（Compute-bound）与 Decode（Memory-bound）混合调度，可更好利用 GPU 的计算和带宽资源。
- **改善 TTFT**：多个短 Prompt 请求可与大 Chunk 交错，减少等待时间。

**代价：**

- **TTFT 增加**：长 Prompt 的完整 Prefill 被拆分为多步，首 Token 时间延长（从 1 步变为 $\lceil S/C \rceil$ 步）。
- **调度复杂度上升**：需维护每个请求的 Prefill 进度（已处理的 Chunk 数），KV Cache 分批写入。
- **Chunk 大小 $C$ 需调优**：$C$ 过小导致 Prefill 效率低（GEMM 形状退化），$C$ 过大则 Decode 延迟改善不明显。典型值 $C = 256 \sim 2048$。

**实现：** vLLM v0.4+、SGLang 均支持 Chunked Prefill，是现代推理框架的标配。

---

**Q40. Prefill / Decode 分离（Disaggregated PD）架构的动机：两阶段分离部署如何提升集群利用率？**

**两阶段计算特性差异：**

|特性|Prefill|Decode|
|---|---|---|
|计算类型|GEMM（矩阵×矩阵）|GEMV（矩阵×向量）|
|瓶颈|Compute-bound|Memory-bound（HBM 带宽）|
|Batch 偏好|单请求大 Token 数|大 Batch Size|
|延迟敏感度|TTFT|TPOT|
|最优硬件|高 TFLOPS GPU（H100）|高 HBM 带宽 GPU（H20）|

**传统混合部署的问题：**

Prefill 和 Decode 共享同一 GPU 时相互干扰：

- Prefill 的大计算量占用 GPU，阻塞 Decode 请求，TPOT 抖动（即 Q39 所述问题）。
- 为保证 TPOT SLA 而限制 Prefill 并发，导致 GPU 利用率低。
- 两阶段对显存的需求模式不同，共享时调度困难。

**P/D 分离架构：**

```
┌─────────────────┐    KV Cache Transfer    ┌─────────────────┐
│  Prefill 实例群  │ ──────────────────────→ │  Decode 实例群   │
│  (P 节点)        │   GPUDirect RDMA /       │  (D 节点)        │
│  专注 TTFT 优化  │   NVLink / TCP          │  专注 TPOT 优化  │
└─────────────────┘                          └─────────────────┘
```

**工作流程：**

1. 请求到达 → 路由到 P 节点执行 Prefill，生成首 Token 和完整 KV Cache。
2. KV Cache 通过高速互联（GPUDirect RDMA，延迟 ~10–100 μs）传输到 D 节点。
3. D 节点接管，持续 Decode 生成后续 Token，直至请求完成。

**收益：**

- P 节点专注大 Batch Prefill，持续高 MFU（Compute-bound 充分利用算力）。
- D 节点专注大 Batch Decode，KV Cache 常驻，HBM 带宽充分利用。
- P/D 实例数比例（xPyD Ratio）可根据负载动态调整（见 Q94）。
- 2025 年已成为所有主流推理框架（vLLM、SGLang、TRT-LLM、NVIDIA Dynamo）的默认部署模式。

---

### 5.2 调度指标

---

**Q41. TTFT 与 TPOT 的区别及各自的优化路径？**

**定义：**

$$\text{TTFT}\ (\text{Time to First Token}) = t_{\text{first token}} - t_{\text{request arrive}}$$

$$\text{TPOT}\ (\text{Time Per Output Token}) = \frac{t_{\text{last token}} - t_{\text{first token}}}{S_{\text{out}} - 1}$$

$$\text{End-to-End Latency} = \text{TTFT} + \text{TPOT} \times (S_{\text{out}} - 1)$$

**区别与用户感知：**

|指标|含义|用户感知|主要决定因素|
|---|---|---|---|
|TTFT|等待首字的时间|响应速度（交互感）|Prefill 计算时间 + 排队时间|
|TPOT|每个字的生成间隔|流畅度（阅读跟得上）|Decode 计算时间（HBM 带宽）|

**各自优化路径：**

**降低 TTFT：**

- 减少 Prefill 计算量：减小输入长度（Prompt 压缩）、使用 Prefix Caching 复用历史 KV。
- 减少排队延迟：P/D 分离让 P 节点专注 Prefill；Chunked Prefill 减少被 Decode 阻塞的时间。
- 增大 Prefill 并发：增加 P 实例数，降低请求等待时间。

**降低 TPOT：**

- 减少单步 Decode 计算量：GQA/MQA 减少 KV 头数；量化降低权重读取量。
- 提升 HBM 带宽利用率：增大 Batch Size（提高 GEMV 的算术强度）；使用高带宽 GPU（H20）。
- 减少 Decode 阶段的调度开销：CUDA Graph 消除 Kernel Launch 开销（见 Q22）。

---

**Q42. 吞吐量与延迟之间的根本矛盾：增大 Batch Size 如何影响两个指标？**

**根本矛盾：**

- **提高吞吐量** → 需要聚合更多请求到同一 Batch → Batch Size 增大 → 每个请求等待时间更长 → **延迟上升**。
- **降低延迟** → 请求到达立即处理（Batch Size = 1）→ GPU 利用率低 → **吞吐量下降**。

**Batch Size 对两类指标的影响分析：**

|Batch Size $B$|Decode GEMV 变化|吞吐量|单请求延迟|
|---|---|---|---|
|1|$I \approx 1$ FLOP/Byte，深度 Memory-bound|低|低|
|增大至 $B^*$（脊点 Batch）|$I \approx I^*$，从 Memory-bound 转为 Compute-bound|线性增长|线性增长|
|超过 $B^*$|Compute-bound，吞吐增长趋于饱和|饱和|继续增长|

**脊点 Batch Size 估算（以 H100 + Llama-3 70B FP16 为例）：**

$$B^* \approx \frac{P_{\text{peak}}}{BW_{\text{mem}} \times \text{FLOPs/token/byte}} = \frac{989 \text{ TFLOPS}}{3.35 \text{ TB/s} \times 1} \approx 295$$

即 Batch Size 超过约 **295** 后，Decode 阶段转为 Compute-bound，吞吐不再随 Batch Size 线性增长。

**实践策略：**

- **在线服务（低延迟优先）**：设置最大 Batch Size 上限，保证 P99 TPOT SLA。
- **离线推理（高吞吐优先）**：使用最大可用 Batch Size，允许排队。
- **自适应调度**：根据当前负载动态调整 Batch Size（vLLM 的 Scheduler 策略）。

---

**Q43. 如何用 MFU（Model FLOP Utilization）评估系统效率？**

**MFU 定义：**

$$\text{MFU} = \frac{\text{实际 FLOPs/s（模型理论计算量 × 吞吐）}}{\text{GPU 峰值 FLOPs/s}}$$

展开为：

$$\text{MFU} = \frac{\text{Tokens/s} \times \text{FLOPs/token}}{P_{\text{peak}}}$$

**FLOPs/token 估算（Transformer，仅前向，忽略 Attention）：**

$$\text{FLOPs/token} \approx 2 \times N_{\text{params}}$$

其中 $N_{\text{params}}$ 为模型参数量（每个参数参与 1 次乘加 = 2 FLOPs）。

**示例（Llama-3 70B，H100 SXM，FP16）：**

- $N_{\text{params}} = 70 \times 10^9$，FLOPs/token $\approx 1.4 \times 10^{11}$
- 若系统吞吐 = 3000 Tokens/s（单卡）
- $P_{\text{peak}} = 989 \times 10^{12}$ FLOP/s（FP16 Tensor Core，稀疏）

$$\text{MFU} = \frac{3000 \times 1.4 \times 10^{11}}{989 \times 10^{12}} \approx 4.2\%$$

注意：Decode 阶段深度 Memory-bound，MFU 天然偏低（典型 3–10%）；Prefill 阶段 Compute-bound，MFU 可达 **40–60%**。

**MFU 的局限性与补充指标：**

|指标|含义|适用场景|
|---|---|---|
|MFU|算力利用率|评估 Prefill / 训练效率|
|MBU（Model Bandwidth Utilization）|带宽利用率 = 实际带宽 / 峰值带宽|评估 Decode 效率|
|Tokens/s/GPU|端到端吞吐|横向对比不同系统|
|Tokens/s/$|成本效率|选型决策|

Decode 阶段更应关注 **MBU**，而非 MFU；实际 MBU 可达 **60–85%**（vLLM + H100），这是 Decode 优化的更直观指标。

## 第 6 章·参考答案：模型量化

---

### 6.1 量化基础

---

**Q44. PTQ（Post-Training Quantization）与 QAT（Quantization-Aware Training）的区别？**

**核心对比：**

|维度|PTQ|QAT|
|---|---|---|
|**时机**|训练完成后，无需重新训练|训练过程中引入量化误差模拟|
|**数据需求**|少量校准数据（数百条）|完整训练数据集|
|**计算代价**|低（小时级）|高（与训练相当，GPU 天级）|
|**精度**|略低，低比特（W4 以下）时损失明显|更高，低比特下优势显著|
|**适用场景**|大模型快速部署（LLM 首选）|小模型、边缘设备、极低比特（W2/W3）|
|**代表方法**|GPTQ、AWQ、SmoothQuant|QLoRA 微调后量化、LLM-QAT|

**PTQ 流程：**

```
预训练模型权重 → 校准数据前向传播（收集激活分布）
→ 计算量化参数（Scale, Zero-point）→ 权重量化 → 量化模型
```

**QAT 核心技巧——Straight-Through Estimator（STE）：**

量化操作 $q(x) = \lfloor x / s \rceil \cdot s$ 的梯度几乎处处为 0，无法直接反传。STE 在前向时使用量化值，反向时将梯度**直通**量化操作，近似为：

$$\frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial q(x)}$$

这使模型权重在训练中"感知"量化误差并主动补偿。

---

**Q45. 对称量化与非对称量化的量化公式推导。**

**量化目标：** 将浮点数 $x \in [\alpha, \beta]$ 映射到整数 $x_q \in [q_{\min}, q_{\max}]$。

**对称量化（Symmetric Quantization）：**

假设浮点范围关于 0 对称，$\alpha = -\beta$，Zero-point $z = 0$：

$$s = \frac{\max(|\alpha|, |\beta|)}{q_{\max}}, \quad z = 0$$

$$x_q = \text{clip}!\left(\left\lfloor \frac{x}{s} \right\rceil,\ q_{\min},\ q_{\max}\right)$$

反量化：$\hat{x} = x_q \cdot s$

**非对称量化（Asymmetric Quantization）：**

浮点范围 $[\alpha, \beta]$ 不要求对称，Zero-point $z \neq 0$：

$$s = \frac{\beta - \alpha}{q_{\max} - q_{\min}}, \quad z = \text{clip}!\left(\left\lfloor -\frac{\alpha}{s} \right\rceil + q_{\min},\ q_{\min},\ q_{\max}\right)$$

$$x_q = \text{clip}!\left(\left\lfloor \frac{x}{s} \right\rceil + z,\ q_{\min},\ q_{\max}\right)$$

反量化：$\hat{x} = (x_q - z) \cdot s$

**对比：**

|特性|对称量化|非对称量化|
|---|---|---|
|Zero-point|$z = 0$，计算简单|$z \neq 0$，需额外存储|
|范围利用率|激活值偏正时范围浪费约 50%|精确覆盖实际范围，精度更高|
|硬件友好性|更好（INT8 GEMM 无需处理 zero-point）|稍差|
|适用对象|权重（分布通常对称）|激活值（如 ReLU 后全为正）|

**量化误差（舍入误差 + 截断误差）：**

$$\mathcal{E} = \underbrace{x_q \cdot s - x}_{\text{舍入误差，最大} \pm s/2} + \underbrace{\text{clip 引入的截断误差}}_{\text{超出范围部分}}$$

---

**Q46. Per-tensor、Per-channel、Per-group 量化粒度的精度-性能 Trade-off？**

量化粒度决定 Scale（和 Zero-point）的共享范围，粒度越细精度越高，但存储和计算开销越大。

|粒度|Scale 数量|精度|额外存储|硬件支持|
|---|---|---|---|---|
|**Per-tensor**|1 个/层|最低（Outlier 影响全局 Scale）|极小|最好|
|**Per-token**（激活）|1 个/token|中（适合激活值动态范围大的场景）|小|好（动态量化）|
|**Per-channel**（权重）|1 个/输出通道|高（每列独立 Scale，消除通道间差异）|小（$d_{\text{out}}$ 个 Scale）|好（cuBLAS 支持）|
|**Per-group**|1 个/group（如每 128 个权重）|更高（精细捕捉局部分布）|中（$N/g$ 个 Scale）|需软件支持|

**Per-group 量化的重要性（以 W4A16 为例）：**

仅 4 bit 存储权重时，Per-tensor 或 Per-channel 粒度的量化误差已无法接受。Per-group（group size = 128）在精度和压缩率之间取得最佳平衡：

- 额外 Scale 存储：$N / 128 \times 2$ 字节（FP16 Scale），约增加 **1.6%** 存储开销。
- 精度：接近 FP16 基线（典型 MMLU 下降 < 1%）。

**工业选型：**

- **权重量化（W4/W8）**：Per-channel 或 Per-group。
- **激活量化（A8）**：Per-token（动态量化，每步推理时实时计算 Scale）。
- **KV Cache 量化**：Per-channel（见 Q36）。

---

### 6.2 主流量化方法

---

**Q47. GPTQ 的核心思路：基于 OBQ 逐层量化，使用 Hessian 信息补偿误差？**

**核心目标：** 对每一层的权重矩阵 $W$，找到量化版本 $\hat{W}$，使输出误差最小：

$$\min_{\hat{W}} | WX - \hat{W}X |_F^2$$

其中 $X$ 为该层的输入激活（校准数据的统计量）。

**OBQ（Optimal Brain Quantization）理论基础：**

将权重逐列量化，每量化一个权重 $w_q$，通过最优更新补偿其余未量化权重，使总误差不增加：

$$\delta W = -\frac{w_q - \hat{w}_q}{\left[H^{-1}\right]_{qq}} \cdot \left[H^{-1}\right]_{:,q}$$

其中 $H = 2XX^T$ 为 Hessian 矩阵（二阶信息），$\hat{w}_q$ 为 $w_q$ 量化后的值。

**GPTQ 的工程简化（使 OBQ 实用化于 LLM）：**

1. **按列顺序量化**（而非 OBQ 的贪心选择顺序），避免 $O(d^3)$ 的动态规划。
2. **Cholesky 分解预计算** $H^{-1}$，避免每列量化都重新求逆，总复杂度降至 $O(d^2)$。
3. **Lazy Batch Update**：将多列的误差补偿合批处理，充分利用 GPU 并行。

**流程：**

```
输入: 层权重 W ∈ R^(d_out × d_in), 校准数据 X
1. 计算 H = 2XX^T，Cholesky 分解得 H^{-1}
2. 按列 j = 0..d_in:
   a. 量化 W[:, j] → Ŵ[:, j]（round-to-nearest）
   b. 更新剩余列: W[:, j+1:] -= (W[:, j] - Ŵ[:, j]) ⊗ H^{-1}[j, j+1:] / H^{-1}[j,j]
3. 输出: 量化权重 Ŵ
```

**性能：** GPTQ W4 在 Llama-2 70B 上相比 FP16 精度损失约 0.3–0.5 perplexity（WikiText-2），量化速度约 2–4 GPU 小时（A100）。

---

**Q48. AWQ（Activation-aware Weight Quantization）相比 GPTQ 的改进：保护 Salient Weights？**

**GPTQ 的问题：** GPTQ 对所有权重一视同仁地量化，但实验发现权重中约 **0.1%–1% 的 Salient（显著）权重**对输出影响极大（对应输入激活值幅度大的通道），量化这些权重导致显著精度损失。

**AWQ 核心观察：**

激活值 $X$ 存在少量幅值极大的通道（Outlier），这些通道对应的权重列对最终输出贡献最大。保护这些 Salient 权重列（不量化或高精度量化）可显著改善精度。

**AWQ 方案：Per-channel 缩放平衡量化难度**

对权重 $W$ 的每个输入通道 $i$，引入缩放因子 $s_i$：

$$\hat{Y} = (W \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot X) = \hat{W} \cdot \hat{X}$$

- 对 Salient 通道：增大 $s_i$，使对应权重 $W_{:,i} / s_i$ 幅值缩小，**量化误差减小**。
- 对应激活 $X_{i,:} \cdot s_i$ 幅值增大，但激活通常不量化（W4A16），无精度损失。

**最优缩放因子搜索：**

$$s_i^* = \arg\min_{s_i} | W \cdot \text{diag}(s)^{-1} \cdot \hat{X} - \hat{W} \cdot \hat{X} |$$

通过网格搜索（Grid Search）在少量校准数据上求解，无需梯度，速度极快（分钟级）。

**AWQ vs GPTQ 对比：**

|维度|GPTQ|AWQ|
|---|---|---|
|原理|Hessian 误差补偿|激活感知缩放|
|校准速度|较慢（小时级）|**更快**（分钟级）|
|W4 精度|好|相当或更好|
|硬件部署|需解压缩（dequant）|同上，但更易与激活量化结合|
|与 A8 结合|困难|**自然兼容** W4A8|

---

**Q49. SmoothQuant 的思路：将激活量化难度通过 per-channel 缩放迁移到权重侧？**

**问题背景（W8A8 的挑战）：**

INT8 权重量化（W8）容易，但激活（A8）量化困难：LLM 的中间激活存在大量**幅值极大的 Outlier**（某些通道数值可达正常通道的 100 倍），导致整体 Scale 被 Outlier 主导，正常通道精度极差。

**SmoothQuant 核心思路：数学等价变换，迁移量化难度**

对线性层 $Y = XW$，引入 per-channel 缩放向量 $s \in \mathbb{R}^{d_{\text{in}}}$：

$$Y = X W = \underbrace{(X \cdot \text{diag}(s)^{-1})}_{\text{平滑后的激活}\ \tilde{X}} \cdot \underbrace{(\text{diag}(s) \cdot W)}_{\text{吸收 Scale 后的权重}\ \tilde{W}}$$

**选择缩放因子 $s$：**

$$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}, \quad \alpha \in [0, 1]$$

- $\alpha \to 1$：所有量化难度迁移到权重，激活量化变容易，权重量化变难。
- $\alpha \to 0$：难度保留在激活侧（不改变）。
- 实践中 $\alpha = 0.5$（均等迁移），激活和权重都变得"平滑"，各自可用 INT8 量化。

**优势：**

- $s$ 在推理前离线计算，**推理时无额外开销**（$s$ 可融入前一层的权重或 LayerNorm 参数中）。
- 使 **W8A8 INT8 GEMM** 成为可能，直接利用 GPU INT8 Tensor Core，吞吐提升约 **1.5–2×**（相比 FP16）。
- 无需修改模型结构，兼容所有 Transformer 架构。

---

**Q50. W4A8 / W4A16 / FP8 / INT8 各方案的适用场景与硬件支持？**

|方案|权重精度|激活精度|计算精度|显存节省|速度收益|适用场景|
|---|---|---|---|---|---|---|
|**W8A8 INT8**|INT8|INT8|INT8 GEMM|~2×|**高**（1.5–2×）|吞吐优先，精度要求不极端|
|**W4A16**|INT4|FP16|FP16 GEMM（dequant）|~4×|中（带宽节省，Decode）|Decode 阶段带宽瓶颈|
|**W4A8**|INT4|INT8|INT8 GEMM|~4×|**最高**|吞吐优先 + 显存最小|
|**FP8 E4M3**|FP8|FP8|FP8 Tensor Core|~2×|高（H100 原生）|H100/H20，平衡精度与速度|
|**FP8 E5M2**|FP8|FP8|FP8 Tensor Core|~2×|高|需更大动态范围的场景|
|**W4A16（NVFP4）**|FP4|FP16|FP16/FP8|~8×|**极高**（Blackwell）|Blackwell 专用，MoE|

**硬件支持矩阵：**

|精度|A100|H100 / H20|B100 / GB200|
|---|---|---|---|
|INT8 GEMM|✅|✅|✅|
|FP8 Tensor Core|❌|✅|✅|
|FP4 Tensor Core|❌|❌|✅（NVFP4）|
|INT4 Tensor Core|✅（有限）|✅|✅|

**选型决策树：**

```
目标是降低显存（Decode带宽）？
├─ 是 → W4A16（GPTQ/AWQ）
目标是最大化吞吐（Prefill计算）？
├─ H100/H20 → FP8（W8A8 FP8 GEMM）
├─ A100 → W8A8 INT8（SmoothQuant）
├─ Blackwell → W4A8 或 NVFP4
精度要求极高？
└─ FP8 > INT8 > W4A16 > W4A8
```

---

**Q51. Blackwell 的 NVFP4（FP4 with block-level FP8 scale）机制与性能收益？**

**NVFP4 格式定义：**

标准 FP4（E2M1）动态范围极窄（仅 $[−6, 6]$），无法直接表示 LLM 权重分布。NVFP4 引入**分组缩放（Block Scaling）**：

- 每 **16 个连续权重**（1 个 Block）共享 1 个 **FP8（E4M3）的 Scale Factor**。
- 实际存储：4 bits/weight + 8 bits/16 weights = $4 + 0.5 = 4.5$ bits/weight（有效位宽）。
- 显存占用相比 FP16 减少 $16/4.5 \approx 3.6\times$。

**量化流程：**

$$\hat{w}_{\text{FP4}} = \text{quantize\_fp4}!\left(\frac{w}{s_{\text{FP8}}}\right), \quad s_{\text{FP8}} = \frac{\max(|w_{\text{block}}|)}{6}$$

**计算流程（推理时）：**

```
权重（NVFP4）× 激活（FP8）→ FP4 Tensor Core（Blackwell 原生）
→ 累加器（FP32）→ 输出（BF16/FP16）
```

**性能收益（相比 H100 FP8）：**

|指标|H100 FP8|B200 NVFP4|
|---|---|---|
|Tensor Core 峰值|1979 TFLOPS（稀疏）|~9000 TFLOPS（估算）|
|显存带宽|3.35 TB/s|8.0 TB/s|
|理论吞吐提升|基准|**~4–5×**|

**精度影响：** NVFP4 相比 FP8 精度损失约 0.5–1.5 perplexity（视任务而定），在 MoE 模型（Expert 权重量化）中损失更小，因为 MoE 的冗余性提供了天然的量化鲁棒性。

## 第 7 章·参考答案：解码加速算法

---

### 7.1 Speculative Decoding

---

**Q52. Speculative Decoding 的基本流程：Draft Model 生成候选 Token，Target Model 并行 Verify，Token 接受率 $\alpha$ 的定义？**

**核心动机：**

标准自回归解码每步只生成 1 个 Token，Target Model（大模型）的算力在 Decode 阶段严重浪费（Memory-bound，见 Q8）。Speculative Decoding 利用小 Draft Model 快速"猜测"多个候选 Token，再由 Target Model **并行验证**，在不改变输出分布的前提下实现加速。

**基本流程：**

```
Step 1 - Draft 阶段（小模型顺序生成）：
  Draft Model 自回归生成 γ 个候选 Token：
  x̃₁, x̃₂, ..., x̃ᵧ（每步约 2–10ms，成本极低）

Step 2 - Verify 阶段（大模型并行验证）：
  Target Model 以 [context, x̃₁, ..., x̃ᵧ] 为输入，
  一次前向传播（1 个 Prefill 步）得到 γ+1 个位置的概率分布：
  p(·|context), p(·|context, x̃₁), ..., p(·|context, x̃₁,...,x̃ᵧ)

Step 3 - Accept/Reject（逐 Token 验证）：
  对 i = 1..γ，以概率 min(1, p(x̃ᵢ)/q(x̃ᵢ)) 接受 x̃ᵢ
  若 x̃ᵢ 被拒绝，从修正分布中采样新 Token，停止验证
  若全部接受，从 p(·|context, x̃₁,...,x̃ᵧ) 额外采样 1 个 Token

其中 q(·) 为 Draft Model 的概率分布，p(·) 为 Target Model 的概率分布。
```

**Token 接受率 $\alpha$ 的定义：**

单个候选 Token $x̃$ 的接受概率为：

$$\alpha = \mathbb{E}!\left[\min!\left(1,\ \frac{p(x̃)}{q(x̃)}\right)\right]$$

其中期望对 Draft Model 的采样分布 $q$ 取。$\alpha$ 越大（Draft 与 Target 分布越接近），加速比越高。

**关键性质：** 每轮 Verify 无论接受几个 Token，**至少产出 1 个 Token**（最坏情况：全部拒绝，从修正分布采样 1 个），因此不会慢于标准解码。

---

**Q53. 接受率 $\alpha$ 与加速比的关系推导。**

**每轮期望接受 Token 数：**

设每轮 Draft 生成 $\gamma$ 个候选，每个独立地以概率 $\alpha$ 被接受（简化假设）。

第 $k$ 个 Token 被接受当且仅当前 $k-1$ 个均被接受，概率为 $\alpha^{k-1}$：

$$\mathbb{E}[\text{接受 Token 数}] = \sum_{k=1}^{\gamma} \alpha^{k-1} \cdot \alpha + \alpha^\gamma \cdot 1 = \sum_{k=1}^{\gamma} \alpha^k + \alpha^\gamma$$

化简（等比数列）：

$$= \frac{\alpha(1 - \alpha^\gamma)}{1 - \alpha} + \alpha^\gamma = \frac{\alpha - \alpha^{\gamma+1}}{1-\alpha} + \alpha^\gamma = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

**加速比推导：**

设 Draft Model 运行 $\gamma$ 步的时间代价相对于 Target Model 1 步为 $c$（即 $c = T_{\text{draft}} \times \gamma / T_{\text{target}}$），每轮 Speculative Decoding 的时间为：

$$T_{\text{spec}} = c \cdot T_{\text{target}} + T_{\text{target}} = (1 + c) \cdot T_{\text{target}}$$

（Draft 顺序生成 + Target 并行 Verify，两者串行）

每轮标准解码生成 1 个 Token 耗时 $T_{\text{target}}$，生成等量 Token 需要：

$$T_{\text{std}} = \frac{1 - \alpha^{\gamma+1}}{1-\alpha} \cdot T_{\text{target}}$$

**加速比：**

$$\boxed{\text{Speedup} = \frac{T_{\text{std}}}{T_{\text{spec}}} = \frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(1 + c)}}$$

**数值示例（$\alpha = 0.8,\ \gamma = 4,\ c = 0.1$）：**

$$\text{Speedup} = \frac{1 - 0.8^5}{(1-0.8)(1+0.1)} = \frac{1 - 0.328}{0.2 \times 1.1} = \frac{0.672}{0.22} \approx 3.05\times$$

**$\gamma$ 的最优选择：** 固定 $\alpha$ 和 $c$，对 $\gamma$ 求导可得最优草稿长度：

$$\gamma^* = \left\lfloor \frac{\ln(c(1-\alpha)/\alpha)}{\ln \alpha} \right\rfloor$$

当 Draft Model 足够小（$c \ll 1$）且 $\alpha$ 较高时，增大 $\gamma$ 收益显著。

---

**Q54. 为什么 Speculative Decoding 不改变输出分布（Rejection Sampling 的等效性）？**

**核心证明思路：**

对第 $i$ 个位置的 Token $x$，其在 Speculative Decoding 下的实际采样分布 $p'(x)$ 需证明等于 Target 分布 $p(x)$。

**情形 1：Token $x$ 被接受（接受概率 $\min(1, p(x)/q(x))$）：**

Token $x$ 从 Draft 分布 $q$ 采样后被接受，对输出的贡献概率：

$$q(x) \cdot \min!\left(1,\ \frac{p(x)}{q(x)}\right) = \min(q(x),\ p(x))$$

**情形 2：Token $x$ 从修正分布中采样（当某候选被拒绝时）：**

拒绝发生的总概率为 $\sum_x \max(0,\ q(x) - p(x))$，修正分布为：

$$p'_{\text{resample}}(x) = \frac{\max(0,\ p(x) - q(x))}{\sum_{x'}\max(0,\ p(x') - q(x'))}$$

两种情形的总贡献：

$$p'(x) = \min(p(x), q(x)) + \sum_x\max(0, q(x)-p(x)) \cdot p'_{\text{resample}}(x)$$

利用恒等式 $p(x) = \min(p(x), q(x)) + \max(0, p(x) - q(x))$ 及归一化条件，可证明：

$$p'(x) = p(x) \quad \forall x$$

**直观理解：** Accept/Reject + 修正采样的组合，等价于直接从 $p$ 采样，Draft Model 仅起加速作用，不影响输出分布。这是 Speculative Decoding 相比知识蒸馏等方法的关键优势——**无精度损失**。

---

**Q55. Ngram-based Draft、EAGLE、Medusa 各方案的对比？**

**三种方案的核心思路：**

**① Ngram-based Draft（无额外参数）：**

- Draft：从上下文历史中查找匹配当前 $n$ 个 Token 的 Ngram，用其后续 Token 作为候选。
- 优点：零额外参数，零额外计算，适合重复性高的场景（代码、模板化文本）。
- 缺点：$\alpha$ 低且不稳定（约 0.5–0.7），对创意生成无效。
- 代表：vLLM 的 `ngram_prompt_lookup`。

**② Medusa（并行解码头）：**

- 在 Target Model 的最后一层隐状态上添加 $K$ 个独立的**解码头**（每个头预测未来第 $k$ 步的 Token）。
- 一次前向同时产生 $K$ 个候选 Token（树形候选），再由原始 LM Head 验证。
- 优点：无需单独 Draft Model，推理开销低（仅增加 $K$ 个线性层）。
- 缺点：需要微调（训练解码头），$\alpha$ 约 0.6–0.75，多头之间相互独立（无自回归依赖）导致接受率不如 EAGLE。

**③ EAGLE（自回归 Draft 头）：**

- 在 Target Model 顶部添加 1 个轻量级**自回归 Draft 模型**（单层 Transformer），以 Target Model 的隐状态序列为条件，自回归地预测未来 Token。
- Draft Model 复用 Target Model 的特征（Feature），而非独立训练，使 Draft 分布更接近 Target 分布。
- EAGLE-2 进一步引入**动态草稿树**（基于当前上下文动态调整候选树的深度和宽度）。
- 优点：$\alpha$ 高（0.8–0.9+），加速比最高（通常 2.5–4×）。
- 缺点：需要在目标模型上微调 Draft 头，且与 Target Model 架构绑定。

**综合对比：**

|方案|额外参数|训练需求|典型 $\alpha$|加速比|适用场景|
|---|---|---|---|---|---|
|Ngram|无|无|0.5–0.7|1.2–1.8×|重复性文本、代码补全|
|Medusa|小（$K$ 个线性层）|需微调|0.6–0.75|1.5–2.5×|通用场景，延迟敏感|
|EAGLE-2|小（单层 Transformer）|需微调|0.8–0.9|**2.5–4×**|通用场景，最优加速|

---

### 7.2 其他算法

---

**Q56. Beam Search 与 Greedy Search 的显存和计算差异？**

**Greedy Search：**

每步选择概率最大的 Token，Beam Width = 1，Batch Size 不增长。

- 显存：仅维护 1 条序列的 KV Cache，$M_{\text{KV}} = O(S)$。
- 计算：每步 1 次 Target Model 前向，Decode 阶段为标准 GEMV。

**Beam Search（Beam Width = $B$）：**

维护 $B$ 条候选序列（Beam），每步扩展 $B \times V$（$V$ 为词表大小）个候选，选取得分最高的 $B$ 条保留。

- 显存：同时维护 $B$ 条序列的 KV Cache，$M_{\text{KV}} = B \times O(S)$。
- 计算：每步 $B$ 次前向（等价于 Batch Size = $B$ 的 Decode）。

**对比：**

|维度|Greedy Search|Beam Search（$B=4$）|
|---|---|---|
|KV Cache|$1\times$|$B\times$（4×）|
|计算量/步|$1\times$|$B\times$（4×）|
|输出质量|较低（局部最优）|更高（全局最优近似）|
|延迟|低|高（$B$ 倍）|
|LLM 实践|**常用**（生成任务）|较少（翻译、语音识别）|

**LLM 中 Beam Search 不常用的原因：** 对话/生成任务中 Beam Search 的输出质量提升有限，但显存和计算代价成 $B$ 倍增长，性价比低。采样方法（Top-k/Top-p）在多样性和质量上表现更好。

---

**Q57. Top-k / Top-p Sampling 的实现细节？**

**Greedy 与采样的关系：**

Greedy = $\text{argmax}$（确定性），Sampling = 按概率分布随机采样（随机性），Top-k/Top-p 是两者之间的折中。

**Top-k Sampling：**

只从概率最高的 $k$ 个 Token 中采样，截断长尾分布：

```python
def top_k_sampling(logits, k, temperature=1.0):
    logits = logits / temperature           # 温度缩放
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probs = F.softmax(top_k_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, 1)
    return top_k_indices[sampled_idx]
```

- **优点：** 实现简单，固定候选数量，GPU 效率高。
- **缺点：** $k$ 固定，无法自适应概率分布的"尖锐程度"。当分布很尖锐（大概率集中于 1–2 个 Token）时，$k = 50$ 仍引入大量低概率噪声。

**Top-p（Nucleus）Sampling：**

按概率从高到低排序，取累积概率刚超过 $p$ 的最小 Token 集合 $\mathcal{V}_p$，从中采样：

$$\mathcal{V}\_p = \min\left\{V' \subseteq V : \sum_{x \in V'} p(x) \geq p\right\}$$

```python
def top_p_sampling(logits, p, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # 移除累积概率超过 p 之后的 Token
    sorted_probs[cumulative_probs > p] = 0.0
    sorted_probs /= sorted_probs.sum()   # 重新归一化
    sampled_idx = torch.multinomial(sorted_probs, 1)
    return sorted_indices[sampled_idx]
```

- **优点：** 自适应分布形状，分布尖锐时候选少（精确），分布平坦时候选多（多样）。
- **缺点：** 需要排序（$O(V \log V)$），GPU 实现需注意 `torch.sort` 的并行效率。

**Temperature（温度）的作用：**

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- $T < 1$：分布变尖锐，输出更确定（倾向高概率 Token）。
- $T > 1$：分布变平坦，输出更随机（多样性增加）。
- $T \to 0$：退化为 Greedy Search。

**GPU 实现优化：**

- Top-k 可用 `torch.topk`（内置 GPU Kernel，$O(V)$ 近似堆排序）。
- 实际推理中词表 $V \sim 32k \sim 128k$，排序开销在 Decode 阶段占比约 1–5%，通常不是瓶颈。
- vLLM 将 Top-k/Top-p 采样 Fuse 进单个 CUDA Kernel，避免多次 HBM 读写。

**Top-k 与 Top-p 组合使用（推荐实践）：**

先做 Top-k（快速截断极端长尾），再做 Top-p（自适应调整候选集），双重过滤兼顾效率与质量：

```python
# 先 Top-k，再 Top-p
logits = top_k_filter(logits, k=50)
token  = top_p_sampling(logits, p=0.9)
```

## 第 8 章·参考答案：并行推理与分布式系统

---

### 8.1 并行策略

---

**Q58. Tensor Parallelism（TP）：以 Megatron-LM 风格说明 MLP 层如何按列/行切分，需要哪些 AllReduce 通信？**

**核心思路：** 将单层的权重矩阵沿某一维度切分到 $N$ 张 GPU 上，每卡只持有权重的 $1/N$，各卡并行计算后通过 AllReduce 聚合结果。

**MLP 层的 TP 切分（列-行切分）：**

标准 MLP：$Y = \text{GeLU}(XW_1)W_2$，其中 $W_1 \in \mathbb{R}^{d \times 4d}$，$W_2 \in \mathbb{R}^{4d \times d}$。

```
第一个线性层（W₁）：按列切分（Column Parallel）
  GPU 0: X × W₁[:, 0:2d]   → Z₀ ∈ R^(B×2d)   （本地 GeLU，无需通信）
  GPU 1: X × W₁[:, 2d:4d]  → Z₁ ∈ R^(B×2d)

  每卡独立持有完整输入 X（需 AllGather 或初始广播）
  每卡输出为完整输出的分块，不需要中间通信

第二个线性层（W₂）：按行切分（Row Parallel）
  GPU 0: Z₀ × W₂[0:2d, :]  → Y₀ ∈ R^(B×d)   （部分和）
  GPU 1: Z₁ × W₂[2d:4d, :] → Y₁ ∈ R^(B×d)   （部分和）

  AllReduce: Y = Y₀ + Y₁    ← 唯一的通信点
```

**Attention 层的 TP 切分：**

Q/K/V 投影按头维度切分（每卡负责 $H/N$ 个头），Output 投影按行切分，同样只需 **1 次 AllReduce**（在 Output 投影后）。

**每个 Transformer 层的通信量：**

- 前向：**2 次 AllReduce**（MLP 1 次 + Attention 1 次），每次通信量 $= 2 \times B \times S \times d \times \text{sizeof}$（乘 2 因为 AllReduce = ReduceScatter + AllGather）
- 反向（训练）：同样 2 次 AllReduce

**TP 适用原则：**

- TP 通信在同一节点内（NVLink），带宽高（900 GB/s），延迟低，适合 **TP ≤ 8**（单节点 8 卡）。
- 跨节点 TP（PCIe/InfiniBand）通信带宽骤降，通常不推荐。

---

**Q59. Pipeline Parallelism（PP）：GPipe vs 1F1B 调度的气泡率对比？**

**Pipeline Parallelism 基本思路：**

将模型的 $L$ 层按深度切分到 $P$ 台设备上，每台设备持有 $L/P$ 层，形成流水线。

**GPipe 调度：**

将 Batch 切分为 $M$ 个 Micro-batch，顺序执行所有前向，再顺序执行所有反向。

```
时间轴（P=4 台设备，M=4 个 Micro-batch）：
设备0: [F0][F1][F2][F3]                [B3][B2][B1][B0]
设备1:     [F0][F1][F2][F3]        [B3][B2][B1][B0]
设备2:         [F0][F1][F2][F3][B3][B2][B1][B0]
设备3:             [F0][F1][F2][F3][B3][B2][B1][B0]
       ←气泡→                              ←气泡→
```

**GPipe 气泡率：**

$$\text{Bubble Rate} = \frac{(P-1)}{M + P - 1}$$

当 $M \gg P$ 时气泡率趋近于 0，但需要大量 Micro-batch 才能摊薄气泡。

**1F1B（One Forward One Backward）调度：**

每台设备交替执行 1 次前向和 1 次反向（非流水线满载时），减少峰值激活值显存占用。

```
时间轴（P=4，M=8）：
设备0: [F0][F1][F2][F3][B0][F4][B1][F5][B2][F6][B3][F7][B4][B5][B6][B7]
设备1:     [F0][F1][F2][F3][B0][F4][B1]...（交错执行）
```

**气泡率（与 GPipe 相同）：**

$$\text{Bubble Rate}_{\text{1F1B}} = \frac{P-1}{M + P - 1}$$

**关键区别：显存占用**

|方案|峰值激活值显存|气泡率|
|---|---|---|
|GPipe|$O(M \times L/P)$（所有前向激活同时驻留）|$(P-1)/(M+P-1)$|
|1F1B|$O(P \times L/P) = O(L)$（仅 $P$ 个 Micro-batch 同时活跃）|$(P-1)/(M+P-1)$|

1F1B 将峰值激活显存从 $O(M)$ 降为 $O(P)$，是生产环境的标准选择。

**推理中的 PP（无反向传播）：**

推理时只有前向，1F1B 退化为简单流水：气泡率 $(P-1)/M$，$M$ 为并发请求数。PP 推理适合**跨节点部署超大模型**（模型无法单节点容纳时）。

---

**Q60. Sequence Parallelism（SP）的原理及适用场景？**

**动机：** 在 TP 中，输入序列 $X$ 需要被所有卡持有完整副本（通过 AllGather 广播），Dropout 和 LayerNorm 等算子在每卡上重复计算，既浪费显存又浪费算力。

**SP 原理（Megatron-LM V3 的 SP）：**

在 TP 的基础上，对序列维度同样进行切分：每卡只持有序列的 $1/N$ 片段（$S/N$ 个 Token），Attention 和 MLP 之外的算子（LayerNorm、Dropout）也在切分后的序列上执行。

```
通信模式变化（TP → SP+TP）：
  TP: AllReduce（= ReduceScatter + AllGather）
  SP: ReduceScatter（MLP/Attn 结束时聚合并切分）
    + AllGather（MLP/Attn 开始前展开序列）

通信量与 TP 相同，但激活显存从 O(B×S×d) 降为 O(B×S/N×d)
```

**Context Parallelism（CP，超长序列）：**

SP 仅切分非 Attention 算子的序列，CP 进一步将 Attention 的序列维度也切分（见 Q101）。适合 $S > 32k$ 的超长序列场景。

**适用场景：**

|技术|序列长度|显存收益|通信开销|
|---|---|---|---|
|TP（无 SP）|任意|无（激活全量复制）|AllReduce|
|SP（Megatron）|中长（8k–32k）|激活显存 $\div N$|ReduceScatter + AllGather（等量）|
|CP|超长（32k+）|KV Cache 显存 $\div N$|P2P Ring 通信|

---

**Q61. Expert Parallelism（EP）：MoE 模型中 All-to-All 通信的开销分析？**

**EP 基本结构：**

MoE 模型有 $E$ 个 Expert，EP 将 $E$ 个 Expert 均匀分配到 $N$ 张 GPU（每卡 $E/N$ 个 Expert）。每个 Token 由路由机制（Router）选择 Top-K 个 Expert 处理。

**Two-shot All-to-All 通信流程：**

```
Step 1 - Dispatch（分发）：
  每卡有 B×S/N 个 Token，根据路由结果
  将 Token 发送到对应 Expert 所在的 GPU
  → All-to-All #1（发送激活值）

Step 2 - Expert 计算：
  每卡对收到的 Token 执行 Expert FFN 计算

Step 3 - Combine（汇聚）：
  将 Expert 输出发送回原始 Token 所在的 GPU
  → All-to-All #2（接收激活值）
```

**通信量分析：**

每个 Token 激活向量大小 $= d \times \text{sizeof}$（如 $d = 7168$，FP16 = 2 B，则 $14336$ B/token）。

设 Batch Token 数为 $T$，Top-K = 2：

$$\text{All-to-All 单次通信量} = T \times K \times d \times \text{sizeof} = T \times 2 \times 14336 \text{ B}$$

以 $T = 4096$，$d = 7168$（DeepSeek-V3 规格）：

$$\text{单次} = 4096 \times 2 \times 14336 \approx 114 \text{ MB（单向）}$$

**延迟分析（H100 NVLink 900 GB/s）：**

$$t_{\text{A2A}} = \frac{114 \text{ MB}}{900 \text{ GB/s} / N} \approx \frac{114 \times 10^6}{900 \times 10^9 / 8} \approx 1 \text{ ms}$$

每层 2 次 All-to-All，Decode 阶段单步总 All-to-All 时间约 **2–5 ms**（视 EP 规模和 Batch Size），占端到端延迟 **10–30%**。

---

### 8.2 通信优化

---

**Q62. AllReduce 的 Ring-AllReduce 实现与带宽分析？**

**朴素 AllReduce（中心化）：** 所有节点将数据发送到 1 个 Master，Master 汇总后广播回去。通信瓶颈在 Master，带宽利用率随节点数 $N$ 线性下降。

**Ring-AllReduce（分散式）：**

将 $N$ 个节点排成一个逻辑环，分两个阶段执行：

**阶段 1 - ReduceScatter（$N-1$ 步）：**

每步每个节点向右邻发送 $M/N$ 大小的数据块，同时从左邻接收并累加。经过 $N-1$ 步后，每个节点持有全局 Reduce 结果的 $1/N$ 分片。

**阶段 2 - AllGather（$N-1$ 步）：**

每步每个节点向右邻发送已完成的分片，经过 $N-1$ 步后，每个节点持有完整的 Reduce 结果。

**带宽分析：**

- 总传输量（每个节点发送）：$2 \times M \times (N-1)/N \approx 2M$
- **与节点数 $N$ 无关**（渐近），带宽利用率接近 100%（所有链路同时满载）。
- 每步传输时间：$t_{\text{step}} = \frac{M/N}{B_{\text{link}}}$，总时间：$t_{\text{AR}} = 2(N-1) \times \frac{M/N}{B_{\text{link}}} \approx \frac{2M}{B_{\text{link}}}$

**Ring-AllReduce 的局限：** 延迟随 $N$ 线性增加（$2(N-1)$ 步），在小消息量场景下 Latency-bound（与 Bandwidth-bound 对立）。

---

**Q63. GEMM-ReduceScatter、AllGather-GEMM 的 Kernel Fusion 如何减少通信-计算串行等待？**

**传统 TP 的通信-计算串行：**

```
GEMM → [等待] → AllReduce → [等待] → 下一层
（计算完成后通信，通信期间 GPU 空闲）
```

**分解 AllReduce 的关键：**

$$\text{AllReduce} = \text{ReduceScatter} + \text{AllGather}$$

ReduceScatter 和 AllGather 各传输约一半数据，可以将计算穿插其中。

**GEMM-ReduceScatter Overlap（第一个线性层）：**

将输出矩阵沿序列维度切分为 $N$ 个分块，GEMM 每计算完一个分块（Tile），立即对该 Tile 发起 ReduceScatter，与下一个 Tile 的 GEMM 并行。

```
时间轴：
GEMM[Tile 0] → ReduceScatter[Tile 0]
GEMM[Tile 1]    ↕（重叠）
GEMM[Tile 2] → ReduceScatter[Tile 2]
...
最终合并：GEMM 与通信完全流水
```

**AllGather-GEMM Overlap（第二个线性层）：**

先发起 AllGather 获取完整输入，同时对已收到的分块执行 GEMM，两者流水进行。

**实现要点：**

- 使用双 CUDA Stream：Stream 0 执行 GEMM，Stream 1 执行通信，通过 CUDA Event 同步依赖关系。
- NCCL 的非阻塞通信（`ncclGroupStart/End`）配合 `cudaStreamWaitEvent`。
- Megatron-LM、TRT-LLM 均实现了此优化，在 NVLink 环境下可将 TP 通信开销降低 **50–80%**。

---

**Q64. NVLink 与 PCIe 的带宽差距对 TP 规模上限的影响？**

**带宽对比：**

|互联方式|带宽（双向）|延迟|适用规模|
|---|---|---|---|
|NVLink 4.0（H100 节点内）|900 GB/s|<1 μs|TP ≤ 8|
|NVLink Switch（NVL72）|3.6 TB/s（聚合）|<1 μs|TP ≤ 72|
|PCIe 5.0（跨 CPU）|128 GB/s|~1 μs|TP ≤ 2（不推荐）|
|InfiniBand NDR（跨节点）|400 Gb/s ≈ 50 GB/s|~1–5 μs|PP/EP|

**TP 上限分析：**

TP 每层需要 2 次 AllReduce，通信量约 $2 \times B \times S \times d \times \text{sizeof}$。

以 $B=32, S=1, d=8192, \text{FP16}$ 为例（Decode 阶段）： $$\text{通信量} = 2 \times 32 \times 1 \times 8192 \times 2 = 1048576 \text{ B} \approx 1 \text{ MB}$$

|互联|通信时间（1 MB）|单步 Decode 计算时间（估算）|通信占比|
|---|---|---|---|
|NVLink（900 GB/s）|~1.1 μs|~500 μs|**0.2%**|
|PCIe（128 GB/s）|~7.8 μs|~500 μs|**1.6%**|
|InfiniBand（50 GB/s）|~20 μs|~500 μs|**4%**|

**结论：**

- **TP 必须在 NVLink 域内**（同一节点 8 卡），带宽充足，通信开销可忽略。
- 超过单节点 8 卡时，TP 应切换为 **PP + EP**（跨节点使用延迟更低的 Pipeline 通信而非 AllReduce）。
- GB200 NVL72 通过 NVLink Switch 将 72 卡全互联，将 NVLink 域扩展至 72 卡，允许更大规模 TP/EP，是 2025 年超大模型推理的关键硬件方案。

## 第 9 章·参考答案：推理框架与工具链

---

### 9.1 主流框架

---

**Q65. vLLM 的核心创新点（PagedAttention + Continuous Batching）？与 TensorRT-LLM 的定位差异？**

**vLLM 的两项核心创新：**

**① PagedAttention（见 Q32）：**

- 将 KV Cache 分页管理，消除显存碎片，GPU 显存利用率从 ~20–40% 提升至 ~90%+。
- 支持 Prefix Sharing（多请求共享 System Prompt 的 KV Block），进一步节省显存。

**② Continuous Batching（见 Q38）：**

- Iteration-level Scheduling，请求完成后立即回收 Slot，新请求立即插入。
- 相比 Static Batching 吞吐提升 2–8×。

**vLLM 的架构组成：**

```
┌────────────────────────────────────────────────────┐
│                   vLLM Engine                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │
│  │  Scheduler   │  │  KV Cache    │  │  Worker  │  │
│  │ (Continuous  │  │  Manager     │  │  (GPU)   │  │
│  │  Batching)   │  │ (PagedAttn)  │  │  Model   │  │
│  └──────────────┘  └──────────────┘  └──────────┘  │
└────────────────────────────────────────────────────┘
```

**vLLM vs TensorRT-LLM 定位对比：**

|维度|vLLM|TensorRT-LLM|
|---|---|---|
|**定位**|通用推理服务框架|NVIDIA 官方高性能推理引擎|
|**核心优势**|调度灵活（PagedAttn + CB）、生态丰富|Kernel 极致优化、TRT 图优化、硬件利用率最高|
|**模型支持**|开箱支持 400+ 模型|需手动适配（Plugin 开发）|
|**Kernel 来源**|CUTLASS / FlashAttention / Triton|NVIDIA 内部手写 Kernel|
|**部署复杂度**|低（pip install）|高（需编译引擎、Plugin）|
|**适用场景**|快速部署、研究验证、多模型服务|生产环境极限性能、NVIDIA 硬件深度绑定|
|**典型吞吐差异**|基准|相同硬件下 TRT-LLM 通常高 10–30%|

---

**Q66. SGLang 相比 vLLM 的改进：RadixAttention（前缀 KV 复用树）的原理？**

**SGLang 的核心创新：RadixAttention**

vLLM 的 Prefix Sharing 只支持**静态、预定义的 System Prompt**共享，无法处理动态变化的公共前缀。SGLang 的 RadixAttention 将 KV Cache 组织为**基数树（Radix Tree）**，自动识别并复用任意请求间的公共前缀。

**Radix Tree 结构：**

```
根节点（空）
├── "You are a helpful assistant."（System Prompt A）
│   ├── "User: What is 2+2?"  → KV Block [3,7,12]
│   └── "User: Explain AI."   → KV Block [3,7,19]
└── "You are a coding expert."（System Prompt B）
    ├── "User: Write Python code." → KV Block [5,9,22]
    └── "User: Debug this code."   → KV Block [5,9,31]
```

**工作机制：**

1. 新请求到达 → 在 Radix Tree 中查找**最长公共前缀**（Longest Prefix Match）。
2. 命中的前缀 KV Block 直接复用（引用计数 +1），无需重新计算。
3. 新增的 Token 追加到 Tree 中作为新节点，供后续请求复用。
4. 内存不足时，按 **LRU（Least Recently Used）** 策略驱逐叶节点。

**相比 vLLM Prefix Caching 的优势：**

- vLLM：只支持完全匹配的固定前缀（Hash-based）。
- SGLang RadixAttention：支持**任意长度、任意内容**的公共前缀自动识别，多轮对话历史可被跨请求复用。

**其他 SGLang 改进：**

- **Zero-overhead Batch Scheduler**：调度逻辑与 GPU 计算重叠，减少调度延迟。
- **Torch.compile + CUDA Graph 结合**：兼顾动态形状与低 Launch 开销。
- **FP8 KV Cache 原生支持** + **MoE EP 优化**（DeepSeek 系列首选框架）。

---

**Q67. TensorRT-LLM 的 Plugin 机制与 In-flight Batching 如何工作？**

**Plugin 机制：**

TensorRT 本身是通用推理框架，对 LLM 特有算子（FlashAttention、RoPE、RMSNorm、KV Cache 管理）没有内置 Kernel。TRT-LLM 通过 **Plugin（自定义算子）** 机制将这些高度优化的 CUDA Kernel 注册到 TRT 的计算图中。

```cpp
// Plugin 注册示例（概念性）
class GPTAttentionPlugin : public IPluginV2DynamicExt {
    // 实现 FlashAttention + PagedKV + RoPE 的融合 Kernel
    void enqueue(...) override {
        flash_attention_with_paged_kv_cache<<<grid, block, smem, stream>>>(
            q, k_cache, v_cache, block_table, output, ...);
    }
};
```

**Plugin 的优势：** 可针对具体模型结构手写极致优化的 Kernel，避免通用框架的抽象开销。TRT-LLM 的 Attention Plugin 集成了 Paged KV Cache、ALiBi/RoPE、多种精度（FP16/BF16/FP8/INT8）于单个 Kernel。

**In-flight Batching（等价于 Continuous Batching）：**

TRT-LLM 将 Continuous Batching 称为 In-flight Batching，实现上有以下特点：

```
Iteration N:
  活跃请求: [Req A: Decode step 50] [Req B: Decode step 12] [Req C: Prefill 1024 tokens]
  ↓ Req B 完成（生成 EOS）
Iteration N+1:
  活跃请求: [Req A: Decode step 51] [Req D: Prefill 512 tokens（新加入）] [Req C: Decode step 1]
```

**Inflight Fused Chunked Context（IFCC）：**

TRT-LLM 进一步将 Chunked Prefill 与 In-flight Batching 结合（等价于 vLLM 的 Chunked Prefill），长 Prefill 分 Chunk 与 Decode 请求混合执行。

---

### 9.2 Profiling 与性能分析

---

**Q68. 使用 `nsys` 和 `ncu` 的区别：Timeline 分析 vs Kernel-level 指标采集？**

**两个工具的定位：**

|工具|全称|分析粒度|主要用途|
|---|---|---|---|
|`nsys`（Nsight Systems）|系统级 Profiler|整个应用 Timeline|定位**哪个阶段**慢、发现 CPU-GPU 交互问题|
|`ncu`（Nsight Compute）|Kernel 级 Profiler|单个 CUDA Kernel|分析**为什么**某个 Kernel 慢、硬件指标诊断|

**`nsys` 典型使用流程：**

```bash
nsys profile --trace=cuda,nvtx,osrt \
    --output=profile_output \
    python inference.py

# 查看 Timeline（GUI 或命令行）
nsys stats profile_output.nsys-rep
```

**`nsys` 能发现的问题：**

- CPU-GPU 之间的 `cudaMemcpy` 占用大量时间（应使用 Pinned Memory 或 Zero-copy）。
- Kernel 之间存在大量 Gap（CPU 调度开销，应使用 CUDA Graph）。
- GPU 空闲时间过长（计算与通信未 Overlap）。
- NCCL 通信占比过高。

**`ncu` 典型使用流程：**

```bash
ncu --set full \
    --kernel-name "flash_attention_kernel" \
    --launch-count 1 \
    python inference.py
```

**`ncu` 关键输出指标：**

|指标|含义|诊断方向|
|---|---|---|
|`sm__throughput`（SM Active %）|SM 活跃时间比例|低 → Occupancy 不足|
|`l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum`|Global Memory 读取量|超出理论值 → 非合并访问|
|`dram__bytes_read.sum`|HBM 实际读取量|与理论值对比|
|`sm__warps_active.avg.pct_of_peak_sustained_active`|Warp Occupancy|低 → 寄存器/SMEM 不足|
|`smsp__sass_thread_inst_executed_op_ffma_pred_on.sum`|FMA 指令数|与理论 FLOPs 对比|

**两者配合使用：** 先用 `nsys` 定位瓶颈 Kernel（宏观），再用 `ncu` 深入该 Kernel 诊断（微观）。

---

**Q69. 如何判断一个 Kernel 是 Memory-bound：查看 `ncu` 的哪些指标？**

**判断流程（Roofline 法）：**

**Step 1：计算实测算术强度 $I_{\text{actual}}$**

$$I_{\text{actual}} = \frac{\text{sm__sass_thread_inst_executed FMA 数} \times 2}{\text{dram__bytes 实际 HBM 访问量}}$$

**Step 2：与脊点 $I^*$ 比较（见 Q7）**

H100：$I^* \approx 295$ FLOP/Byte（FP16）

- $I_{\text{actual}} < I^*$：**Memory-bound**
- $I_{\text{actual}} > I^*$：**Compute-bound**

**`ncu` 关键指标组合诊断：**

|指标|Memory-bound 时的表现|
|---|---|
|`dram__throughput`（HBM 带宽利用率）|**高**（接近 90%+）|
|`sm__throughput`（SM 算力利用率）|**低**（< 50%）|
|`l2_hit_rate`|低（数据无法从 L2 命中，必须访问 HBM）|
|`stall_long_scoreboard`（等待内存的 Stall）|**高**|
|`achieved_occupancy`|可能低（但不一定）|

**典型 Memory-bound Kernel 的 `ncu` 报告特征：**

```
Memory Throughput: 3.1 TB/s  ← 接近 H100 峰值 3.35 TB/s（Memory-bound）
Compute Throughput: 12%      ← 算力严重浪费
DRAM Bandwidth Utilization: 92.3%
L2 Hit Rate: 23.4%           ← L2 命中率低，大量 HBM 访问
```

**常见 Memory-bound Kernel：** Elementwise（Add、Mul、GELU）、LayerNorm/RMSNorm、GEMV（Decode 阶段 Attention 和 Linear）。

---

**Q70. Occupancy 低对性能一定有影响吗？什么情况下低 Occupancy 也能高性能？**

**Occupancy 定义：**

$$\text{Occupancy} = \frac{\text{活跃 Warp 数/SM}}{\text{最大 Warp 数/SM}}$$

H100 每 SM 最大 64 个 Warp，Occupancy = 活跃 Warp 数 / 64。

**Occupancy 的作用：**

高 Occupancy → 更多就绪 Warp → 当一个 Warp 因内存延迟阻塞时，有更多 Warp 可立即调度 → **延迟隐藏**。

**Occupancy 低但性能高的情况：**

**① Compute-bound Kernel（如 GEMM）：**

大型 GEMM Kernel 每个 Warp 的寄存器用量极大（每线程 ~128–255 个寄存器），导致 Occupancy 低（可能仅 12.5%，即 8/64 Warp）。但 GEMM 是计算密集型，Warp 几乎不阻塞，Scheduler 始终有工作可做，**延迟隐藏需求低**。实测此时 Occupancy 从 12% 提升到 25% 对 GEMM 吞吐几乎没有影响。

**② Memory-bound Kernel 使用了 L2/L1 缓存：**

若数据集中在 L1/L2 Cache 中（高局部性），访存延迟从 ~600 cycles 降至 ~20–200 cycles，所需的 Warp 数量（延迟隐藏所需）随之减少，低 Occupancy 已足够。

**③ Kernel 受 Instruction-Level Parallelism（ILP）优化：**

单 Warp 内通过循环展开（Unroll）使多条指令并行执行，即使 Warp 数量少，指令流也能保持饱和。

**实践结论：**

|场景|Occupancy 重要性|优化重点|
|---|---|---|
|Memory-bound（L2 Miss 严重）|**高**（需要足够 Warp 隐藏 HBM 延迟）|提高 Occupancy|
|Compute-bound（GEMM）|**低**（计算不阻塞，无需隐藏延迟）|减少寄存器用量以外的优化|
|Memory-bound（高 L2 命中）|中|优先提高 L2 命中率|

**诊断工具：** `ncu` 的 `Latency Breakdown` 报告会显示 Warp Stall 的主要原因，若 `stall_long_scoreboard`（等待内存）占主导，则低 Occupancy 是问题；若 `stall_math_throttle`（等待计算）占主导，则 Compute-bound，Occupancy 无关紧要。

---

### 9.3 Triton

---

**Q71. Triton 与 CUDA 的核心编程模型差异（Block-level vs Thread-level）？**

**CUDA 编程模型（Thread-level）：**

程序员显式管理每个线程的行为：

- 手动计算每个 Thread 的全局索引。
- 手动管理 Shared Memory 的分配、加载、同步（`__syncthreads()`）。
- 手动处理 Bank Conflict、Warp Divergence、内存对齐。

```cuda
__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 手动计算索引
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}
```

**Triton 编程模型（Block-level / Tile-level）：**

程序员以**Block（Tile）为单位**编写逻辑，Triton 编译器自动处理底层细节：

- 每个 Triton 程序实例处理一个 Block 的数据（如 `BLOCK_SIZE = 1024` 个元素）。
- 通过 `tl.load` / `tl.store` 进行向量化访存，自动处理内存对齐和边界。
- 自动优化 Shared Memory 布局、Bank Conflict 消除、Warp 调度。

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                          # Block 索引（类比 blockIdx）
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)         # 向量化加载，自动 Coalescing
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)
```

**核心差异对比：**

|维度|CUDA|Triton|
|---|---|---|
|编程粒度|Thread 级（32 线程/Warp）|Block 级（$N$ 个元素/程序）|
|Shared Memory|手动分配与同步|自动管理（编译器优化）|
|Bank Conflict|手动消除|编译器自动处理|
|内存合并|手动保证连续访问|`tl.load` 自动向量化|
|自动调优|无|`triton.autotune` 自动搜索超参|
|跨硬件移植|需大量修改|AMD ROCm、Intel GPU 支持中|

---

**Q72. 何时选择 Triton 而非 CUDA 手写？**

**选择 Triton 的场景：**

**① 快速原型与算法验证：**

新 Attention 变体、新量化方案的 Kernel 原型，用 Triton 实现通常只需 CUDA 的 **1/5–1/10 代码量**，开发周期从数天缩短至数小时。FlashAttention 的 Triton 实现（`flash_attn_triton`）即是典型案例。

**② 跨硬件移植：**

Triton 编译器支持 NVIDIA（PTX）、AMD（HIP/ROCm）及实验性 Intel GPU 后端。同一份 Triton 代码可在不同硬件上运行，而 CUDA 代码严格绑定 NVIDIA。

**③ 自动调优（AutoTuning）：**

Triton 内置 `@triton.autotune` 装饰器，自动搜索 `BLOCK_SIZE`、`num_stages`、`num_warps` 等超参数的最优组合，无需手动 Benchmark：

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel(...):
    ...
```

**选择 CUDA 手写的场景：**

**① 极限性能需求（生产级 Kernel）：**

FlashAttention-3 使用 WGMMA + TMA + Warp Specialization，需精细控制 Warp 角色、异步 Barrier、Shared Memory Swizzle 等，Triton 目前无法表达这些 Hopper 特有的优化。

**② 特殊硬件特性利用：**

TMA、`cp.async`、`mbarrier`、Tensor Memory Descriptor 等 Hopper 原语，Triton 尚未完整支持（截至 2025 年支持仍不完整）。

**③ 寄存器级精细控制：**

需要手动 `#pragma unroll`、PTX 内联汇编、寄存器变量复用等极端优化时，CUDA/PTX 更可控。

**选型原则（简化）：**

```
是否需要跨硬件 / 快速验证 / 自动调优？
  └─ 是 → Triton

是否需要 Hopper 特有原语（WGMMA/TMA/Warp Spec）/ 极限性能？
  └─ 是 → CUDA / PTX 手写

其余情况 → Triton 足够，优先选择
```
