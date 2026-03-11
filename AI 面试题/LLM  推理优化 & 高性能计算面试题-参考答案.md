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
