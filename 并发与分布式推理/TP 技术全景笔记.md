## 1. Tensor Parallelism 定义

Tensor Parallelism（张量并行，TP）是一种**模型并行（Model Parallelism）技术**，核心思想是：

> 将单个神经网络层内部的参数张量（Tensor）切分到多个 GPU 上，使多个 GPU 协同完成同一个 Layer 的计算。

与 Data Parallelism（DP）不同：

- **DP：每张 GPU 保存完整模型，不同 GPU 处理不同 Batch**
- **TP：每张 GPU 保存模型的一部分，共同完成一个 Batch**

对于大语言模型（LLM），TP 主要解决：

1. 单 GPU 显存无法容纳模型参数；
2. 单 GPU 计算吞吐不足；
3. 大规模 Transformer 推理/训练扩展问题。

典型应用：

- Megatron-LM
- DeepSpeed
- TensorRT-LLM
- vLLM
- NVIDIA NeMo

---

## 2. Transformer 中为什么需要 Tensor Parallelism

Transformer Layer 参数主要来自 Attention：

$$  
\mathbf{Q}=XW_Q  
$$

$$  
\mathbf{K}=XW_K  
$$

$$  
\mathbf{V}=XW_V  
$$

其中：

- $X\in R^{S\times H}$
- $S$：Sequence Length
- $H$：Hidden Size
- $W_Q,W_K,W_V\in R^{H\times H}$

FFN：

$$  
\text{FFN}(X)=W_2\sigma(W_1X)  
$$

其中：

$$  
W_1\in R^{H\times 4H}  
$$

$$  
W_2\in R^{4H\times H}  
$$

对于 GPT-3 175B：

- 参数：

$$  
175\times10^9  
$$

FP16：

$$  
175B\times2Byte  
\approx350GB  
$$

单张 H100：

- 80GB HBM

无法加载。

因此需要：

$$  
\text{Model Size}/N_{GPU}  
\leq GPU\ Memory  
$$

TP 将模型切分：

$$  
W=  
{W_0,W_1,...,W_{p-1}}  
$$

每张 GPU 保存 $\frac{1}{p}$ 参数。

---

## 3. Tensor Parallelism 基本思想

### 3.1 矩阵切分

假设：

$$  
Y=XW  
$$

其中：

$$  
X\in R^{m\times k}  
$$

$$  
W\in R^{k\times n}  
$$

输出：

$$  
Y\in R^{m\times n}  
$$

TP 有两种主要切分方式：

1. Column Parallel（列并行）
2. Row Parallel（行并行）

### 3.2 Column Parallel Linear（列并行）

#### 3.2.1 基本思想

按照权重矩阵的列切分：

$$  
W=  
[  
W_1,W_2,...,W_p  
]  
$$

其中：

$$  
W_i\in R^{k\times n/p}  
$$

GPU：

|GPU|参数|
|---|---|
|GPU0|$W_1$|
|GPU1|$W_2$|
|...|...|
|GPU p-1|$W_p$|

计算：

$$  
Y_i=XW_i  
$$

每个 GPU 得到：

$$  
Y_i\in R^{m\times n/p}  
$$

最后：

$$  
Y=  
[Y_1,Y_2,...,Y_p]  
$$

需要：

$$  
AllGather(Y_i)  
$$

#### 3.2.2 Column Parallel 通信

计算：

```
             X
             |
       --------------
       |      |     |
      W0     W1    W2
       |      |     |
      Y0     Y1    Y2
       |      |     |
       ----AllGather---
             |
             Y
```

通信：

$$  
Communication=  
O(mn)  
$$

### 3.3 Row Parallel Linear（行并行）

#### 3.3.1 基本思想

按照权重矩阵行切分：

$$  
W=  
\begin{bmatrix}  
W_1\  
W_2\  
...\  
W_p  
\end{bmatrix}  
$$

其中：

$$  
W_i\in R^{k/p\times n}  
$$

输入也切分：

$$  
X=  
[X_1,X_2,...,X_p]  
$$

计算：

$$  
Y_i=X_iW_i  
$$

最终：

$$  
Y=\sum_iX_iW_i  
$$

需要：

$$  
Reduce  
$$

实际使用：

$$  
AllReduce(Y_i)  
$$

#### 3.3.2 Row Parallel 通信

```
          X
          |
   ----------------
   |       |      |
  X0      X1     X2

   |       |      |

  W0      W1     W2

   |       |      |

  Y0      Y1     Y2

        AllReduce

            Y
```

通信：

$$  
O(mn)  
$$

---

## 4. Transformer 中 TP 如何应用

### 4.1 Attention TP

Attention：

$$  
Q=XW_Q  
$$

$$  
K=XW_K  
$$

$$  
V=XW_V  
$$

通常采用 QKV Column Parallel，切权重

$$  
W_Q=  
[  
W_{Q0},W_{Q1},...,W_{Qp}  
]  
$$

每个 GPU：

$$  
Q_i=XW_{Qi}  
$$

得到：

$$  
Q=[Q_0,...,Q_p]  
$$

$K,V$ 同理。

### 4.2 Attention Head 切分

Multi Head Attention：

$$  
Attention(Q,K,V)

Concat(head_1,...,head_h)  
$$

其中：

$$  
head_i  
\in R^{S\times d}  
$$

TP：

$$  
h_{GPU}

\frac{h}{p}  
$$

例如：

LLaMA：

- Head = 32
- TP=8

每 GPU：

$$  
4\ heads  
$$

因此：

GPU0：

```
Q head 0-3
K head 0-3
V head 0-3
```

GPU1：

```
Q head 4-7
...
```

Attention 内部：

无需通信。

---

## 5. Attention 输出投影 TP （Row Parallel）

Attention：

$$  
O=Attention(Q,K,V)W_O  
$$

因为 Concat 后：

$$  
O\in R^{S\times H}  
$$

通常 $W_O$，采用 **Row Parallel**，即：

$$  
W_O=  
\begin{bmatrix}  
W_0\  
W_1  
\end{bmatrix}  
$$

每 GPU 计算部分：

$$  
O_i=X_iW_i  
$$

最后：

$$  
AllReduce(O_i)  
$$

---

## 6. FFN 中 TP

Transformer FFN：

$$  
FFN(X)=W_2\sigma(W_1X)  
$$

其中：

第一层：

$$  
W_1:H\rightarrow4H  
$$

第二层：

$$  
W_2:4H\rightarrow H  
$$

### 6.1 第一层 Column Parallel

切：

$$  
W_1=  
[  
W_{10},W_{11},...,W_{1p}  
]  
$$

每 GPU：

$$  
Y_i=XW_{1i}  
$$

无需通信。

### 6.2 第二层 Row Parallel

输入：

$$  
Y=[Y_0,Y_1,...,Y_p]  
$$

计算：

$$  
Z_i=Y_iW_{2i}  
$$

最后：

$$  
Z=  
\sum_iZ_i  
$$

通信：

$$  
AllReduce  
$$

---

## 7. Megatron-LM TP 架构

一个 Transformer Block：

```
             X
             ↓
         LayerNorm
             ↓
      QKV Projection
             |
         Column TP
             ↓
		 Attention
             ↓
     Output Projection
             |
          Row TP
             ↓
         Residual
             ↓
            FFN
			 |
       W1 Column TP
             ↓
            GELU
             |
	     W2 Row TP
             ↓
          Residual
```

通信位置：

只有：

- Attention Output Projection
- FFN Output Projection

产生：

$$  
2\times AllReduce / Layer  
$$

---

## 8. TP 与其他并行方式组合

**TP + Sequence Parallel：** 解决 LayerNorm、Dropout 等无法 TP 的算子。SP 切 $Sequence$，只用于 Prefill，降低 Activation Memory。

**TP + Context Parallel：** 解决长上下文显存爆炸。CP 按 $Sequence$ 切 KV Cache 降低显存。

**TP + Expert Parallel：** 解决单卡装不下多个 Expert 的问题。EP 切分有多种策略：连续切分、锯齿切分（防止热点集中）、拓扑感知（优先节点内路由）

现代 LLM 使用：TP × PP × DP

例如 GPT-3：

```
TP=8
PP=8
DP=8

Total GPU:
8×8×8=512
```

|并行方式|切分对象|通信|优势|
|---|---|---|---|
|DP|Batch|Gradient AllReduce|简单|
|TP|Tensor|Layer 内通信|降低显存|
|PP|Layer|Activation|扩展模型深度|
|CP|Sequence|Attention|长上下文|
|EP|Expert|MoE Expert|扩大MoE容量|

---

## 9. TP 在 Decode 阶段的问题

**通信成为瓶颈**

Decode 阶段，每生成一个 Token，需要一次 AllReduce ，通信频繁。因此 TP 越大，通信比例越高。

Latency：

$$  
T=  
T_{compute}  
+  
T_{comm} 
=
\frac{M}{BW}  
$$

当 $T_{comm}>T_{compute}$，继续增加 GPU，收益下降。

---

## 10. 多头注意力的 TP 切分

### 10.1 MHA 的标准 TP 切分（Megatron-LM 方案）

MHA 中 Q/K/V 投影和输出投影的切分遵循 **Column Parallel → Row Parallel** 的经典模式。

**Q/K/V 投影（Column Parallel）：**

$W^Q \in \mathbb{R}^{d \times H d}$ 按 Head 维度（列）切分到 $P$ 张卡：

$$W^Q_i = W^Q[:, i \cdot Hd/P : (i+1) \cdot Hd/P] \in \mathbb{R}^{d \times (Hd/P)}$$

每卡计算 $H/P$ 个 Head 的 Q（$K, V$ 同理）。各卡完全独立，无需通信。

**Attention 计算：**

每卡在本地完成 $H/P$ 个 Head 的完整 Attention（Q/K/V 均已本地切分），无需通信。

**输出投影（Row Parallel）：**

$W^O \in \mathbb{R}^{Hd \times d}$ 按行（Head 维度）切分：

$$W^O_i = W^O[i \cdot Hd/P : (i+1) \cdot Hd/P, :] \in \mathbb{R}^{(Hd/P) \times d}$$

每卡输出局部1结果 $o_i = \text{Attn}_i \cdot W^O_i \in \mathbb{R}^{d}$，最终 All-Reduce 求和：

$$o = \sum_{i=1}^P o_i$$

**通信分析：** 仅需**一次 All-Reduce**（输出投影后），通信量 $= 2 \times B \times N \times d \times \text{sizeof}$（All-Reduce = Reduce-Scatter + All-Gather）。

### 10.2 GQA 下 TP 的约束

GQA 中 $H_{\text{KV}} = H / G$（KV Head 数），若 $P > H_{\text{KV}}$，则每个 KV Head 无法整除分配到所有卡——出现**TP > KV Head 数**的问题。

**约束：** $P$ 必须整除 $H_{\text{KV}}$，即 $P \leq H_{\text{KV}}$ 且 $H_{\text{KV}} \mod P = 0$。

以 LLaMA-3 70B（$H = 64$，$G = 8$，$H_{\text{KV}} = 8$）为例：最大 TP = 8（再大则 KV Head 无法整除）。

**若需更大 TP（如 TP = 16）的处理方案：**

方案 1（KV 复制）：每个 KV Head 复制到多张卡，各卡持有完整的 KV Head 副本，Q Head 正常切分。代价：KV 冗余存储。

方案 2（TP 与 DP 解耦）：Q 的 TP 维度独立于 KV 的 TP 维度，KV 用较小的 TP（如 8），Q 用更大的 TP，中间通过额外通信对齐。

TensorRT-LLM 和 vLLM 均采用方案 1，在 $P > H_{\text{KV}}$ 时自动触发 KV 复制。

### 10.3 KV Cache 在 TP 下的分布

KV Cache 按 KV Head 切分存放在各卡本地，Decode 时各卡直接读取本地 KV Cache，无需跨卡通信（这是 TP 切分 Attention 的主要优势之一）。

每卡 KV Cache 大小：

$$M_{\text{KV/card}} = 2 \times L \times \frac{H_{\text{KV}}}{P} \times d \times S \times \text{sizeof}$$

以上述 LLaMA-3 70B，TP=8，$S=8192$，FP16 为例：

$$M_{\text{KV/card}} = 2 \times 80 \times 1 \times 128 \times 8192 \times 2 \approx 335 \text{ MB/卡}$$

---

## 11. TP Size 如何选择

经验：

|模型规模|推荐 TP|
|---|---|
|7B|1-2|
|13B|2-4|
|70B|4-8|
|175B|8-16|
|MoE 大模型|8-64|

原则上，$TP\leq GPU\ 数量$，优先 TP = NVLink 域大小，例如 H100 DGX：

```
8 GPU
|
NVSwitch
|
高速 NVLink
```

通常：

$$  
TP=8  
$$

---

## 12. 总结

Tensor Parallelism 核心：

$$  
\boxed{  
\text{切分 Layer 内 Tensor，让多个 GPU 协同计算}  
}  
$$

Transformer 中：

|模块|切分|
|---|---|
|QKV Projection|Column Parallel|
|Attention Output|Row Parallel|
|FFN Up Projection|Column Parallel|
|FFN Down Projection|Row Parallel|

关键通信：

|位置|通信|
|---|---|
|Column 输出|AllGather|
|Row 输出|AllReduce|

工程实践：

- TP 主要依赖 GPU 间高速互联；
- NVLink/NVSwitch 适合节点内 TP；
- 跨节点 TP 通常受 InfiniBand 限制；
- Decode 阶段 TP 通信占比高，是扩展瓶颈；
- 大模型推理通常采用 **TP + PP + DP + CP/EP 混合并行**。
