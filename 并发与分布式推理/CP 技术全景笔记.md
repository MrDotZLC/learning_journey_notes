## 1. 概述

### 1.1 长上下文时代的计算瓶颈

Transformer 的 Self-Attention：

# $$  
\mathrm{Attention}(Q,K,V)

\mathrm{Softmax}  
\left(  
\frac{QK^T}{\sqrt{d_h}}  
\right)V  
$$

其中：

- $Q \in \mathbb{R}^{S \times d_h}$：Query
    
- $K \in \mathbb{R}^{S \times d_h}$：Key
    
- $V \in \mathbb{R}^{S \times d_h}$：Value
    
- $S$：Sequence Length
    
- $d_h$：Head Dimension
    

Attention Score：

$$  
A=QK^T  
$$

维度：

$$  
A\in\mathbb{R}^{S\times S}  
$$

因此：

计算复杂度：

$$  
O(S^2d_h)  
$$

显存复杂度：

$$  
O(S^2)  
$$

---

传统 Transformer 在短文本时代：

|模型|上下文长度|
|---|--:|
|GPT-2|1024|
|GPT-3|2048|
|LLaMA-1|2048|
|LLaMA-2|4096|

Attention 矩阵规模尚可接受。

但进入长上下文时代：

|模型|上下文长度|
|---|--:|
|GPT-4 Turbo|128K|
|Claude 3|200K|
|Gemini 1.5|1M|
|部分 Long Context LLM|1M+|

例如：

$$  
S=128K  
$$

Attention Matrix：

# $$  
S^2

(131072)^2  
$$

约：

$$  
1.7\times10^{10}  
$$

如果 FP16：

$$  
1.7\times10^{10}\times2  
\approx34GB  
$$

仅 Attention Matrix 就超过单张 GPU HBM。

---

因此长上下文带来三个核心问题：

|问题|本质|
|---|---|
|计算爆炸|$O(S^2)$ Attention|
|显存不足|KV Cache 和 Attention 激增|
|单卡无法承载|Sequence 维度成为瓶颈|

Tensor Parallelism（TP）主要切：

$$  
Hidden Dimension  
$$

Pipeline Parallelism（PP）切：

$$  
Layer Dimension  
$$

但：

$$  
Sequence Dimension  
$$

仍然集中在单个 GPU。

因此提出：

> Context Parallelism：沿 Sequence 维度切分上下文，使多个 GPU 协同完成长序列计算。

---

# 2. Context Parallelism 基本思想

## 2.1 Sequence Dimension 切分

假设：

输入：

$$  
X\in R^{S\times d}  
$$

使用：

$$  
P  
$$

张 GPU。

CP 将 Sequence 进行切分：

$$  
S=S_0+S_1+\cdots+S_{P-1}  
$$

每张 GPU 保存：

$$  
X_i\in R^{S/P\times d}  
$$

例如：

8 GPU：

128K context：

$$  
S=131072  
$$

每 GPU：

$$  
S_i=  
\frac{131072}{8}  
=16384  
$$

即：

```
GPU0 : token 0~16383
GPU1 : token 16384~32767
GPU2 : token 32768~49151
...
GPU7 : token 114688~131071
```

---

## 2.2 CP 与普通 Attention 的矛盾

Attention：

# $$  
Attention(Q,K,V)

Softmax(QK^T)V  
$$

对于 GPU0：

拥有：

$$  
Q_0,K_0,V_0  
$$

只能计算：

$$  
Q_0K_0^T  
$$

得到：

$$  
S_{00}  
$$

但是完整 Attention 需要：

$$  
Q_0K^T  
$$

其中：

$$  
K=  
[K_0,K_1,...K_{P-1}]  
$$

即：

GPU0 需要：

$$  
Q_0K_1^T  
$$

$$  
Q_0K_2^T  
$$

...

因此：

> CP 的核心问题不是切分 Q，而是如何让每个 GPU 获得完整 K/V 信息。

---

# 3. Context Parallel 的数学形式

原始 Attention：

$$  
O_i=  
\mathrm{Softmax}  
(  
Q_iK^T  
)V  
$$

展开：

$$  
K=  
[K_0,K_1,...K_{P-1}]  
$$

得到：

# $$  
Q_iK^T

[  
Q_iK_0^T,  
Q_iK_1^T,  
...  
Q_iK_{P-1}^T  
]  
$$

因此：

每个 GPU：

1. 持有自己的 $Q_i$
    
2. 依次获得其他 GPU 的 $K_j,V_j$
    
3. 计算局部 Attention
    
4. 合并结果
    

---

# 4. CP 的两大路线

目前主流 Context Parallel 有两个方向：

## 4.1 Ring Attention

代表：

- Stanford Ring Attention
    
- Megatron CP
    
- FlashAttention-2 CP
    

核心：

> KV 在 GPU Ring 中循环流动。

通信：

Ring AllGather：

```
GPU0 ---> GPU1 ---> GPU2 ---> GPU3
 ^                           |
 |---------------------------|
```

每轮：

GPU 获得一个 KV Block。

---

## 4.2 Ulysses Attention

代表：

- DeepSpeed Ulysses
    
- Megatron SP/CP 混合方案
    

核心：

> 通过 All-to-All 重新排列 Attention Head。

它不是移动 KV，而是：

重新分布：

$$  
Sequence  
\times  
Head  
$$

维度。

---

两者区别：

||Ring Attention|Ulysses|
|---|---|---|
|核心通信|Ring Send/Recv|All-to-All|
|切分维度|Sequence|Sequence + Head|
|通信模式|点对点|集合通信|
|GPU规模|适合大规模|中小规模|
|实现复杂度|较低|较高|
|代表|Megatron CP|DeepSpeed Ulysses|

---

## 8. Context Parallel 与 TP / SP / PP / EP 的组合方式

---

## 8.1 Transformer 并行维度总览

大模型训练和推理通常需要同时切分多个维度：

|并行方式|切分维度|解决问题|
|---|---|---|
|Tensor Parallelism（TP）|Hidden Dimension / Attention Head|单层计算过大|
|Pipeline Parallelism（PP）|Layer Dimension|模型参数过大|
|Data Parallelism（DP）|Batch Dimension|吞吐扩展|
|Expert Parallelism（EP）|Expert Dimension|MoE 专家扩展|
|Sequence Parallelism（SP）|Sequence Dimension（非 Attention 主体）|降低激活显存|
|Context Parallelism（CP）|Sequence Dimension（Attention）|长上下文扩展|

其中：

- TP 解决 **模型宽度问题**
    
- PP 解决 **模型深度问题**
    
- CP 解决 **序列长度问题**
    

---

# 8.2 CP + TP

## 8.2.1 TP 负责 Hidden，CP 负责 Sequence

Transformer 输入：

$$  
X\in R^{S\times d}  
$$

其中：

- $S$：Sequence Length
    
- $d$：Hidden Size
    

TP：

切 Hidden：

$$  
d=\sum_{i=0}^{P_{TP}-1}d_i  
$$

CP：

切 Sequence：

$$  
S=\sum_{j=0}^{P_{CP}-1}S_j  
$$

二维切分：

$$  
X_{i,j}  
\in  
R^{S_j\times d_i}  
$$

例如：

H100 集群：

```
TP=8
CP=4

总 GPU = 32
```

逻辑布局：

```
             TP Group

        GPU0 GPU1 ... GPU7
CP0      x    x        x

CP1      x    x        x

CP2      x    x        x

CP3      x    x        x
```

---

## 8.2.2 Attention 中 TP + CP 协作

Multi Head Attention：

$$  
Q=XW_Q  
$$

$$  
K=XW_K  
$$

$$  
V=XW_V  
$$

TP：

切 Head：

$$  
H=  
H_0+H_1+...+H_{P_{TP}}  
$$

每个 TP Rank：

拥有：

$$  
Q_i,K_i,V_i  
$$

CP：

切 Token：

GPU：

只拥有：

$$  
S/P_{CP}  
$$

因此：

一个 GPU：

实际负责：

$$  
Attention  
(  
Q_{head_i}^{seq_j},  
K_{head_i}^{allseq},  
V_{head_i}^{allseq}  
)  
$$

即：

- TP：
    
    - 分 Head
        
- CP：
    
    - 分 Token
        

---

## 8.2.3 TP + CP 通信关系

Attention 前：

TP 内：

需要同步：

$$  
Q,K,V  
$$

通常：

无需通信。

Attention：

CP 内：

需要交换：

$$  
K,V  
$$

采用：

- Ring Attention
    
- AllGather KV
    

Attention 后：

TP：

Linear Projection：

需要：

AllReduce / ReduceScatter。

结构：

```
             CP通信
GPU0 -------- GPU1
 |             |
 |             |
TP AllReduce  TP AllReduce
 |             |
GPU2 -------- GPU3
             CP通信
```

---

# 8.3 CP + SP

这是最容易混淆的组合。

## 8.3.1 Sequence Parallelism

SP 最早用于 Megatron-LM。

LayerNorm：

$$  
Y=LN(X)  
$$

如果 TP：

MLP：

Column Parallel：

$$  
Y=XW_1  
$$

每张卡：

拥有部分 Hidden。

但是：

LayerNorm 需要完整 Token。

SP 将：

Sequence：

进一步切分。

---

SP 主要作用：

降低：

- Activation Memory
    
- LayerNorm / Dropout 显存
    

但：

SP 不解决 Attention 的 $S^2$ 问题。

---

## 8.3.2 CP 与 SP 区别

||SP|CP|
|---|---|---|
|目标|降低激活显存|解决长 Attention|
|切分|Sequence|Sequence|
|对象|Transformer Block 激活|Attention Context|
|是否需要 KV 通信|否|是|
|典型位置|MLP/LN|Attention|

因此：

长上下文训练：

通常：

$$  
TP+SP+CP  
$$

例如：

```
64 GPU

TP=8
CP=4
DP=2

每个 DP:

8×4=32 GPU
```

---

# 8.4 CP + PP

PP：

按 Layer：

```
Stage0
Layer0-15

Stage1
Layer16-31

Stage2
Layer32-47
```

CP：

每个 Pipeline Stage 内：

继续切 Sequence。

二维：

```
Pipeline Stage

Stage0:
 GPU0 GPU1 GPU2 GPU3

Stage1:
 GPU4 GPU5 GPU6 GPU7


每个Stage内部:

CP group
```

---

问题：

Pipeline Bubble。

如果：

CP 增大：

每个 GPU：

token 数下降。

导致：

单 Micro Batch 计算减少。

可能增加：

Pipeline Bubble。

因此：

CP 与 PP 需要联合调优。

---

# 8.5 CP + EP（MoE）

MoE：

Router：

$$  
g(x)=TopK(W_rx)  
$$

Token：

发送给 Expert。

EP：

切 Expert：

```
GPU0:
Expert0 Expert1

GPU1:
Expert2 Expert3
```

CP：

切 Token：

```
CP0:
token 0~16k

CP1:
token16k~32k
```

组合后：

需要：

两类通信：

## Attention阶段

CP通信：

$$  
K,V  
$$

## MoE阶段

EP通信：

Token Dispatch：

$$  
AllToAll  
$$

因此：

MoE 长上下文：

通信压力：

```
Attention:
CP Ring

MoE:
EP AllToAll
```

两者可能竞争：

NVLink / IB。

---

# 9. CP 在 Prefill / Decode 阶段的工程优化

---

## 9.1 Prefill 阶段

Prefill：

输入：

$$  
S  
$$

个 Token。

计算：

Attention：

$$  
QK^T  
$$

复杂度：

$$  
O(S^2)  
$$

因此：

CP 最适合 Prefill。

---

例如：

128K Prompt：

单 GPU：

不可接受。

CP=8：

每 GPU：

16K Token。

Attention Block：

降低：

$$  
128K^2  
\rightarrow  
16K^2  
$$

显存压力降低：

约：

$$  
64\times  
$$

---

## 9.2 Decode 阶段

Decode：

每次生成：

1 token。

Attention：

Query:

$$  
Q\in R^{1\times d}  
$$

KV Cache：

$$  
K,V\in R^{S\times d}  
$$

计算：

$$  
QK^T  
$$

复杂度：

$$  
O(Sd)  
$$

瓶颈：

HBM Bandwidth。

---

CP 对 Decode 的收益有限。

原因：

Decode 需要：

完整 KV Cache。

如果 CP：

KV 分布：

```
GPU0:
KV 0~16K

GPU1:
KV16K~32K
```

每步：

Query：

需要访问全部 KV。

需要：

KV Gather。

通信：

成为瓶颈。

---

## 9.3 CP + KV Cache

两种方案：

---

## 方案 A：

KV Cache Replication

每 GPU：

保存完整 KV。

优点：

Decode 快。

缺点：

显存：

# $$  
Memory

P\times KV  
$$

---

## 方案 B：

KV Cache Sharding

CP：

切 KV。

GPU：

保存：

$$  
KV_i  
$$

Decode：

需要：

Remote KV。

通信：

$$  
O(P\times S)  
$$

适合：

超长 Context。

---

实际推理：

通常：

Prefill：

CP

Decode：

关闭 CP 或降低 CP Degree。

例如：

```
Prefill:
CP=8

Decode:
CP=1/2
```

这也是：

PD Disaggregation：

流行原因。

---

# 10. CP 通信量、性能模型与部署建议

---

# 10.1 Ring Attention 通信量

假设：

CP size：

$$  
P  
$$

KV：

大小：

$$  
M_{KV}  
$$

Ring Attention：

每 GPU：

发送：

$$  
P-1  
$$

轮。

每轮：

发送：

$$  
\frac{M_{KV}}{P}  
$$

因此：

总通信：

# $$  
T_{comm}

(P-1)  
\frac{M_{KV}}{P}  
$$

约：

$$  
\approx M_{KV}  
$$

即：

每层：

每 GPU：

通信接近一次 KV Cache 大小。

---

# 10.2 AllGather KV 通信

如果直接：

AllGather：

每 GPU：

发送：

$$  
\frac{P-1}{P}M_{KV}  
$$

总：

$$  
P\times  
\frac{P-1}{P}  
M_{KV}  
$$

约：

$$  
(P-1)M_{KV}  
$$

相比 Ring：

增加：

$$  
P  
$$

倍压力。

---

因此：

大规模 CP：

Ring 更优。

---

# 10.3 CP 性能模型

总时间：

$$  
T=  
T_{compute}  
+  
T_{comm}  
$$

计算：

# $$  
T_{compute}

\frac{FLOPs}  
{GPU_FLOPS}  
$$

通信：

# $$  
T_{comm}

\frac{Bytes}  
{Bandwidth}  
$$

CP 有收益条件：

$$  
T_{comm}  
<  
T_{compute}  
$$

即：

通信可以被计算隐藏。

---

# 10.4 CP 扩展瓶颈

## 小 CP

例如：

CP=2/4

优势：

- 通信低
    
- 利用率高
    

---

## 大 CP

例如：

CP=32/64

问题：

### 1. 通信增加

Ring：

$$  
O(P)  
$$

### 2. GPU 利用下降

每 GPU：

token：

$$  
\frac{S}{P}  
$$

下降。

### 3. Kernel 不饱和

FlashAttention：

tile：

$$  
(BLOCK_M,BLOCK_N)  
$$

需要足够：

Sequence。

---

# 10.5 工程部署建议

## 长上下文训练

推荐：

```
TP + CP + SP + DP
```

典型：

70B：

```
TP=8
CP=8
DP=4

总GPU=256
```

---

## 长上下文推理 Prefill

推荐：

```
PD Disaggregation

Prefill:
TP + CP

Decode:
TP
```

原因：

Prefill：

计算密集。

Decode：

带宽密集。

---

## CP 参数选择

经验：

|Context Length|CP|
|---|---|
|8K-32K|1|
|64K|2-4|
|128K|4-8|
|256K+|8-32|

约束：

$$  
\frac{S}{CP}  
\ge  
4096  
\sim8192  
$$

保证 Attention Kernel 饱和。

---

# 10.6 CP 未来方向

## 1. Hierarchical Context Parallel

两级通信：

节点内：

NVLink

节点间：

IB

例如：

```
CP=16

Node:
CP=4

Across Node:
CP=4
```

降低 IB 压力。

---

## 2. CP + KV Compression

结合：

- MQA
    
- GQA
    
- KV Quantization
    
- StreamingLLM
    

降低：

$$  
M_{KV}  
$$

---

## 3. CP + Speculative Decoding

Decode：

减少 KV 通信。

---

## 4. CP + Attention Kernel Fusion

未来趋势：

```
FlashAttention
+
Ring Communication
+
Tensor Core
+
Async Pipeline
```

实现：

Communication-Computation Overlap。

---

# 总结

Context Parallelism 的核心定位：

$$  
\boxed{  
CP=\text{沿 Sequence 维度扩展 Attention}  
}  
$$

与其他并行关系：

|技术|切分|
|---|---|
|TP|Hidden / Head|
|PP|Layer|
|EP|Expert|
|CP|Context Sequence|
|SP|Activation Sequence|

工程实践：

- **训练长上下文：TP + CP + SP**
    
- **Prefill：CP 收益最大**
    
- **Decode：CP 受 KV 通信限制**
    
- **超长 Context：Ring Attention 是主流**
    
- **未来方向：CP 与 KV 优化、通信隐藏结合**
    

CP 本质上是把 Transformer 的第三个维度：

$$  
(Layer, Hidden, Sequence)  
$$

中的 **Sequence Dimension 并行化**。