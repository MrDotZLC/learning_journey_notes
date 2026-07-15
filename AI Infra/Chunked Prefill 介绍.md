## 1. Chunked Prefill 概述

Chunked Prefill（分块 Prefill）是一种**将长 Prompt 的 Prefill 阶段切分成多个 Chunk（块）依次执行**的调度策略。

其核心目标不是减少计算量，而是：
- **降低请求等待时间（TTFT）**
- **提升 Decode 请求优先级**
- **实现 Prefill 与 Decode 的流水执行**
- **提高在线推理系统吞吐**
    
它最早广泛应用于 **vLLM**，目前 TensorRT-LLM、SGLang、LMDeploy、KServe 等推理框架均采用类似思想。

---

## 2. Chunked Prefill  与 Chunked Tile 的区别

两者虽然都有 **Chunked**，但属于**不同层次**的优化，解决的问题完全不同。

|名称|优化层次|切分对象|目的|
|---|---|---|---|
|**Chunked Prefill**|**调度（Scheduling）**|Token 序列|降低 TTFT、与 Decode 交错执行|
|**Chunked Tile**|**Kernel/GEMM 实现**|矩阵 Tile|提高 GPU 利用率、减少访存|

---

## 3. 为什么需要 Chunked Prefill

**普通 Prefill 的问题：**

假设 Prompt 很长，如果整个 Prompt 一次计算，GPU 会连续执行：

```
8192 Token GEMM
8192 Token Attention
8192 Token KV Write
...
```

直到全部结束。

期间：
> **所有 Decode 请求都无法获得 GPU，且 Prefill 耗时很长。**

于是：
- TTFT 很高
- TPOT P99 很高
- 在线服务体验很差

---

## 4.  每个 Chunk 做什么

上一 Chunk 已经写好了 KV Cache，于是：

```
Chunk0
↓
写 Chunk0 KV
↓
Chunk1
↓
读取 Chunk0 KV
↓
写 Chunk1 KV
↓
Chunk2
↓
读取前两个 Chunk KV
```

整个过程保持与完整 Prefill 完全一致的因果 Attention。

---

## 5. Chunk 的收益

TTFT：

$$  
TTFT = Queue + Prefill + Sampling
$$

普通情况下，需要等待整个 Prefill。
Chunk 之后，调度粒度缩小，TTFT 显著下降。

---

## 6. Chunk Size 的影响

**Chunk 很大时：**

优点：
- GEMM 大
- GPU 利用率高

缺点：
- Decode 等待长
- TTFT 高

**Chunk 很小时：**

优点：
- Decode 几乎实时

缺点：
- Kernel 数量暴增
- Launch Overhead 增加
- GEMM 太小
- Tensor Core 利用率下降

---

## 7. Chunk Size 选择

**1. 计算 FLOPs**

一次乘加算两个FLOPs

1. QKV Linear
   $$XW_Q \in (S \times d)(d \times d)$$
   $$FLOPs_{QKV} = 3 \times 2 \times S \times d^2$$
2. Attention Scroe
   $$QK^T \in (S \times d)(d \times S)$$
   $$FLOPs_{QK}=2 \times S^2 \times d$$
3. Attention Value
   $$O_{attn} = Attention \times V \in (S \times S)(S \times d)$$
   $$FLOPs_{AV}=2 \times S^2 \times d$$
4. Attention
   $$FLOPs_{Attn} = FLOPs_{QK} + FLOPs_{AV} = 4S^2d$$
5. Output Projection（特征空间映射）
   融合所有头，使每个向量都包含所有头的信息
   $$O = O_{attn} \times W_O \in (S \times d)(d \times d)$$
   $$FLOPs_{O} = 2 \times S \times d^2$$
6. FFN
   第一层：
   $$XW_1 \in (S \times d)(d \times d_{ff})$$
   第二层：
   $$X_2W_2 \in (S \times d_{ff})(d_{ff} \times d)$$
   $$FLOPs_{FFN} = 2 \times S \times d \times d_{ff}$$
7. Prefill 的 FLOPs
   $$FLOPs_{Prefill}^{layer} = 6Sd^2 + 4S^2d + 2Sd^2 + 4Sdd_{ff} = 8Sd^2 + 4S^2d + 4Sdd_{ff}$$
   $$FLOPs_{Prefill} = L \times (8Sd^2 + 4S^2d + 4Sdd_{ff})$$
8. Decode 的 FLOPs
   $$FLOPs_{Decode} = L \times (8d^2 + 4Sd + 4dd_{ff})$$

**2. Prefill 时延对 Chunk Size 的影响**

在 H100 $\times$ 8（TP=8）上理论峰值约 $8 \times 989 \text{ TFLOPS} \approx 7.9 \text{ PFLOPS}$，MFU 约 $30\text{–}50\%$，实际耗时约：
$$t_{\text{Prefill}} \approx \frac{FLOPs_{Prefill}} {7.9 \times 10^{15} \times 0.4}$$

为了隐藏调度时延，应满足 $t_{\text{Prefill}} \geq t_{\text{Decode}}$ 。

**3. 碎片率对 Chunk Size 的影响**

Chunked Prefill 的碎片率：
$$碎片率 \approx \frac {B-1} {2C}$$
Chunked Size 对 GEMM 效率的影响：
H100 Tensor Core 的高效计算要求 $\text{Chunk Size} \ge 128 \: (Wave Quantization 效应)$

Chunked Size 要达到计算时延可以覆盖通信时延，最优数值要参考 $TPOT_{SLO}$。

P/D 同置下，Decode 请求的 P99 TPOT 约等于单 Chunk Prefill 的计算时间。

**4. 主流推理框架常见配置（会随模型、GPU、调度策略变化）：**

|Chunk Size|特点|
|---|---|
|64|极低延迟，Kernel 开销较高|
|128|较低延迟|
|256|常见折中|
|512|在线服务较常见|
|1024|吞吐优先|
|2048+|接近普通 Prefill|

实际最佳值通常需要根据模型规模、Batch Size、GPU 类型（如 H100、B200）以及业务目标（TTFT 或吞吐）进行调优。

---

## 8. Chunked Prefill 与 Continuous Batching 的关系

两者经常同时出现，但关注点不同：

|技术|作用对象|核心思想|
|---|---|---|
|Continuous Batching|请求调度|新请求无需等待整个 Batch 完成，可动态加入 Batch|
|Chunked Prefill|单个长 Prompt|将长 Prefill 拆分为多个 Chunk，释放 GPU 调度点|

二者结合后的调度流程可表示为：

```
Request A
Chunk0
↓
Request B
Chunk0
↓
Decode Batch
↓
Request C
Chunk0
↓
Request A
Chunk1
↓
Decode Batch
↓
Request B
Chunk1
```

Continuous Batching 提供**动态批处理能力**，Chunked Prefill 提供**细粒度的 GPU 抢占点**，共同降低在线推理的 TTFT，并改善 Decode 请求的尾延迟（如 TPOT P99）。

---

## 9. Chunked Prefill 与 P/D 分离（Prefill/Decode Disaggregation）

在 **P/D 分离**架构中：
- Prefill GPU 专门负责计算 Prompt 并生成 KV Cache。
- Decode GPU 专门负责自回归生成。
    
由于两类计算运行在不同 GPU 上，Decode 不再需要等待 Prefill GPU 释放计算资源，因此**Chunked Prefill 的主要价值不再是让 Decode 插队**。

此时 Chunked Prefill 的作用主要体现在：

- **降低 Prefill GPU 的单请求阻塞时间**，提高多个 Prefill 请求之间的公平性。
- **支持流式 KV Cache Transfer**：每完成一个 Chunk，即可将对应 KV Cache 发送到 Decode GPU，而无需等待整个 Prompt 完成。
- **减少首次可解码时间**：如果系统支持分块传输和提前消费 KV，Decode 侧可以更早开始工作（具体取决于实现）。
    
因此，在 P/D 分离系统中，Chunked Prefill 更多是一种**Prefill 流水化和 KV 流式传输机制**，而不是用于缓解 Decode 延迟的调度手段。
