## 1. Chunked Prefill 概述

Chunked Prefill（分块 Prefill）是一种**将长 Prompt 的 Prefill 阶段切分成多个 Chunk（块）依次执行**的调度策略。

其核心目标不是减少计算量，而是：
- **降低请求等待时间（TTFT）**
- **提升 Decode 请求优先级**
- **实现 Prefill 与 Decode 的流水执行**
- **提高在线推理系统吞吐**
    
它最早广泛应用于 **vLLM**，目前 TensorRT-LLM、SGLang、LMDeploy、KServe 等推理框架均采用类似思想。

---

## 2. 为什么需要 Chunked Prefill

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

## 3.  每个 Chunk 做什么

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

## 4. Chunk 的收益

TTFT：

$$  
TTFT = Queue + Prefill + Sampling
$$

普通情况下，需要等待整个 Prefill。
Chunk 之后，调度粒度缩小，TTFT 显著下降。

---

## 9. Chunk Size 的影响

设：

Chunk

$$  
C  
$$

### Chunk 很大

例如：

```
4096
```

优点：

- GEMM 大
    
- GPU 利用率高
    

缺点：

- Decode 等待长
    
- TTFT 高
    

---

### Chunk 很小

例如：

```
64
```

优点：

- Decode 几乎实时
    

缺点：

- Kernel 数量暴增
    
- Launch Overhead 增加
    
- GEMM 太小
    
- Tensor Core 利用率下降
    

因此：

Chunk 不宜过小。

---

## 10. Chunk Size 的经验值

主流推理框架常见配置（会随模型、GPU、调度策略变化）：

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

## 11. Chunked Prefill 与 Continuous Batching 的关系

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

## 12. Chunked Prefill 与 P/D 分离（Prefill/Decode Disaggregation）

在 **P/D 分离**架构中：

- Prefill GPU 专门负责计算 Prompt 并生成 KV Cache。
    
- Decode GPU 专门负责自回归生成。
    

由于两类计算运行在不同 GPU 上，Decode 不再需要等待 Prefill GPU 释放计算资源，因此**Chunked Prefill 的主要价值不再是让 Decode 插队**。

此时 Chunked Prefill 的作用主要体现在：

- **降低 Prefill GPU 的单请求阻塞时间**，提高多个 Prefill 请求之间的公平性。
    
- **支持流式 KV Cache Transfer**：每完成一个 Chunk，即可将对应 KV Cache 发送到 Decode GPU，而无需等待整个 Prompt 完成。
    
- **减少首次可解码时间**：如果系统支持分块传输和提前消费 KV，Decode 侧可以更早开始工作（具体取决于实现）。
    

因此，在 P/D 分离系统中，Chunked Prefill 更多是一种**Prefill 流水化和 KV 流式传输机制**，而不是用于缓解 Decode 延迟的调度手段。