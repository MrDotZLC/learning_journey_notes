## 1. 背景

随着 Large Language Model（LLM）支持的上下文长度从：
- GPT-3：2K Token
- GPT-4：8K～128K Token
- 长上下文模型：1M+ Token

Transformer 的 Attention 计算和激活显存成为训练长序列的主要瓶颈。

| 项目           | 复杂度       |
| ------------ | --------- |
| Attention 计算 | $O(S^2d)$ |
| Attention 显存 | $O(S^2)$  |
| KV Cache     | $O(Sd)$   |

当 $S=128K$ 时：

$$  
S^2=1.6\times10^{10}  
$$

单卡无法存储完整 Attention Matrix。

Context Parallelism（CP）提出：

> 将一个长 Context Sequence 切分到多个 GPU，每个 GPU 负责部分 Token 的 Attention 计算，通过通信完成全局 Attention。

---

## 2. 核心思想：Sequence 维度切分

Tensor Parallel（TP）切分模型参数：

```
        Hidden Dimension

GPU0 | W0
GPU1 | W1
GPU2 | W2
```

Context Parallelism 切分输入序列：

```
      Sequence Length

GPU0: token 0 ~ S/P
GPU1: token S/P ~ 2S/P
GPU2: token 2S/P ~ 3S/P
GPU3: token 3S/P ~ S
```

其中：
- $P$：CP degree
- 每个 GPU 保存：

$$  
S_{local}=\frac{S}{P}  
$$

因此 Attention Query：

$$  
Q_i\in R^{S/P\times d}  
$$

Key、Value 同样按照 Sequence 切分：

$$  
K_i,V_i\in R^{S/P\times d}  
$$

---

## 3. 为什么 Context Parallelism 需要通信

GPU0：

```
Q0
K0,V0
```

只能计算：

$$  
Attention(Q_0,K_0,V_0)  
$$

但是实际 Attention：

```
Q0 需要看到：

K0,V0
K1,V1
K2,V2
K3,V3
```

因为每个 Query Token 都需要访问完整 Context。

---

## 4. FlashAttention 与 Context Parallelism 关系

两者解决不同问题：

|技术|优化目标|
|---|---|
|FlashAttention|单 GPU Attention 显存和 IO|
|Context Parallelism|多 GPU 扩展 Context Length|

组合方式：

```
              Long Context

                   |
                   v

        Context Parallelism

                   |
       -------------------------
       |           |           |

     GPU0        GPU1        GPU2

       |
       v

 FlashAttention Kernel
```

实际系统通常：

```
CP + TP + PP + DP
```

联合使用。

---

## 5. Context Parallelism 通信模式

### 5.1 Ring Communication

[RingAttention 详细介绍](../Transformer/RingAttention%20详细介绍.md)

核心思想：将 All-Gather 与 Attention 计算**流水重叠**，消除通信等待。

优点：
- 通信量均衡
- GPU 利用率高

缺点：
- 延迟随 GPU 数增加

每个 GPU 需要接收：

$$  
\frac{P-1}{P}S  
$$

规模的 KV。

### 6.2 AllGather Attention

另一种方式：

步骤：

1. AllGather 所有 KV

```
GPU0:
K0+K1+K2+K3

GPU1:
K0+K1+K2+K3
```

2. 本地 Attention。

优点：
- 实现简单

缺点：
- KV 显存增加

每张 GPU：

$$  
O(Sd)  
$$

失去 CP 显存优势。
因此长 Context 通常采用 Ring Attention。

---

## 7. Context Parallelism 与 Sequence Parallelism 区别

|      | Context Parallelism               | Sequence Parallelism                     |
| ---- | --------------------------------- | ---------------------------------------- |
| 目的   | 扩展 Context Length                 | 减少激活显存                                   |
| 切分对象 | 输入 Token                          | Transformer 激活                           |
| 主要阶段 | **Attention 算子**（QKV 计算、Score 矩阵） | **非 Attention 算子**（LayerNorm、Dropout、残差） |
| 通信   | KV Exchange                       | AllGather                                |
| 代表   | Ring Attention                    | Megatron SP                              |

关系：

```
Sequence Parallelism
      ↓
减少 Transformer 激活

Context Parallelism
      ↓
扩展 Attention Sequence
```

---

## 8. CP 与 TP/PP/EP 的组合

现代 LLM 训练：

$$  
\text{GPU数量}
=
DP\times TP\times PP\times CP\times EP  
$$

例如：

```
128 GPUs

DP = 2
TP = 8
PP = 4
CP = 2

2×8×4×2=128
```

含义：

|并行|切分|
|---|---|
|DP|Batch|
|TP|Hidden/Model|
|PP|Layer|
|CP|Sequence|
|EP|Expert|

---

## 9. Context Parallelism 在推理中的应用

Decode 阶段，主要受 KV Cache 带宽限制，不需要 CP。

Prefill 阶段，整个序列长度需要本地 GPU 访问完整的  Q/K/V 矩阵，CP 可以极大缓解 Attention 中的显存压力，能够增大序列长度上限。

因此 CP 主要用于：

- 长文档理解
- Agent 长上下文
- RAG 大规模输入

---

## 10. 主流实现

### 10.1 Megatron-LM Context Parallel

NVIDIA Megatron-LM 支持：
- Context Parallel
- Tensor Parallel
- Pipeline Parallel

核心：
- Ring Attention
- P2P KV Communication

### 10.2 DeepSpeed Ulysses

DeepSpeed Ulysses 使用：All-to-All，重新分布 Attention Head。

流程：

```
Sequence Parallel
        ↓
   All-to-All
        ↓
Head Parallel Attention
```

特点：
- 通信一次完成
- 适合中等 CP degree

### 10.3 Ring Attention vs Ulysses

| |Ring Attention|Ulysses|
|---|---|---|
|通信|P2P Ring|All-to-All|
|显存|低|中|
|扩展性|强|中|
|实现|复杂|简单|
|适合|超长 Context|一般长 Context|

---

## 11. Context Parallelism 优缺点

**优点：**
- 支持百万级 Context
- 降低单 GPU 显存
- 与 FlashAttention 兼容
- 支持大模型训练

**缺点：**
- 增加 GPU 通信
- 通信隐藏困难
- 小 Context 收益有限
