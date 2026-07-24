## 1. Context Parallelism 技术全景

### 1.1 Context Parallelism 背景

随着 Large Language Model（LLM）支持的上下文长度从：
- GPT-3：2K Token
- GPT-4：8K～128K Token
- 长上下文模型：1M+ Token

Transformer 的 Attention 计算和激活显存成为训练长序列的主要瓶颈。

|项目|复杂度|
|---|---|
|Attention 计算|$O(S^2d)$|
|Attention 显存|$O(S^2)$|
|KV Cache|$O(Sd)$|

当 $S=128K$ 时：

$$  
S^2=1.6\times10^{10}  
$$

单卡无法存储完整 Attention Matrix。

Context Parallelism（CP）提出：

> 将一个长 Context Sequence 切分到多个 GPU，每个 GPU 负责部分 Token 的 Attention 计算，通过通信完成全局 Attention。

---

## 2. Context Parallelism 核心思想

### 2.1 Sequence 维度切分

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

### 3.1 局部 Attention 的问题

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

## 4. Ring Attention

### 4.1 基本思想

Ring Attention 是最经典 CP 实现。

GPU 组成 Ring：

```
GPU0 → GPU1 → GPU2 → GPU3
 ↑                 ↓
 └─────────────────┘
```

每轮：
1. GPU 保留自己的 Q
2. 接收邻居 KV
3. 计算 Attention Block
4. 发送 KV 到下一 GPU

例如：

第 0 轮：

```
GPU0:
Q0 × K0,V0

GPU1:
Q1 × K1,V1
```

交换 KV：

```
GPU0:
Q0 × K1,V1

GPU1:
Q1 × K2,V2
```

直到所有 KV Block 被计算。

---

### 4.2 Online Softmax

不能保存完整 Attention Matrix：

$$  
S\times S  
$$

因此使用 FlashAttention 类似的 Online Softmax。

维护：

- 当前最大值 $m$
- 当前归一化因子 $l$
- 当前输出 $O$

每收到一个 KV Block：

更新：

$$  
m_{new}=max(m,m_i)  
$$

$$  
l_{new}=e^{m-m_{new}}l+e^{m_i-m_{new}}l_i  
$$

最终得到完整 Attention 输出。

因此：

- 显存：

$$  
O(\frac{S^2}{P})  
$$

降低：

$$  
P  
$$

倍。

---

## 5. FlashAttention 与 Context Parallelism 关系

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

## 6. Context Parallelism 通信模式

### 6.1 Ring Communication

通信：

```
Send KV
Receive KV
Compute
```

优点：

- 通信量均衡
- GPU 利用率高

缺点：

- 延迟随 GPU 数增加

通信量：

每个 GPU 需要接收：

$$  
\frac{P-1}{P}S  
$$

规模的 KV。

---

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

容易混淆：

||Context Parallelism|Sequence Parallelism|
|---|---|---|
|目的|扩展 Context Length|减少激活显存|
|切分对象|输入 Token|Transformer 激活|
|主要阶段|Attention|LayerNorm/Dropout|
|通信|KV Exchange|AllGather|
|代表|Ring Attention|Megatron SP|

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

## $$  
\text{GPU数量}

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

# 9. Context Parallelism 在推理中的应用

## 9.1 长 Prompt Prefill

Decode：

单 Token：

$$  
O(S)  
$$

主要受 KV Cache 带宽限制。

Prefill：

一次处理：

$$  
S  
$$

Token。

Attention：

$$  
O(S^2)  
$$

因此 CP 主要用于：

- 长文档理解
    
- Agent 长上下文
    
- RAG 大规模输入
    

---

## 9.2 KV Cache Parallelism

长 Context 推理：

单 GPU：

```
KV Cache:

Layer × Sequence × Hidden
```

显存压力巨大。

CP：

```
GPU0:
KV token 0~N/P

GPU1:
KV token N/P~2N/P
```

减少单 GPU KV Cache。

---

# 10. 主流实现

## 10.1 Megatron-LM Context Parallel

NVIDIA Megatron-LM 支持：

- Context Parallel
    
- Tensor Parallel
    
- Pipeline Parallel
    

核心：

- Ring Attention
    
- P2P KV Communication
    

---

## 10.2 DeepSpeed Ulysses

DeepSpeed Ulysses 使用：

```
All-to-All
```

重新分布 Attention Head。

流程：

```
Sequence Parallel

        |
        v

All-to-All

        |
        v

Head Parallel Attention
```

特点：

- 通信一次完成
    
- 适合中等 CP degree
    

---

# 11. Ring Attention vs Ulysses

||Ring Attention|Ulysses|
|---|---|---|
|通信|P2P Ring|All-to-All|
|显存|低|中|
|扩展性|强|中|
|实现|复杂|简单|
|适合|超长 Context|一般长 Context|

---

# 12. Context Parallelism 优缺点

## 优点

- 支持百万级 Context
    
- 降低单 GPU 显存
    
- 与 FlashAttention 兼容
    
- 支持大模型训练
    

## 缺点

- 增加 GPU 通信
    
- 通信隐藏困难
    
- 小 Context 收益有限
    

---

# 13. 发展趋势

## 13.1 CP 与 Attention Kernel 深度融合

未来趋势：

```
CP Communication

        +

FlashAttention Kernel

        +

NVLink / InfiniBand


=> Distributed Attention
```

目标：

- 通信计算完全 Overlap
    
- 减少 KV Movement
    

## 13.2 Hierarchical Context Parallelism

针对大规模集群：

```
Node 内:

NVLink CP


Node 间:

InfiniBand CP
```

根据通信距离选择不同策略。

## 13.3 长上下文推理架构

未来推理系统：

```
Request Scheduler

       |

Context Parallel Engine

       |

FlashAttention

       |

Distributed KV Cache
```

Context Parallelism 将成为百万 Token LLM 服务的重要基础组件。
