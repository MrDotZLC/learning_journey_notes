# 1. Tensor Parallelism（TP）全景技术笔记

## 1.1 Tensor Parallelism 定义

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

# 2. Transformer 中为什么需要 Tensor Parallelism

## 2.1 LLM 参数规模问题

Transformer Layer 参数主要来自：

### Attention

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

每张 GPU 保存：

$$  
\frac{1}{p}  
$$

参数。

---

# 3. Tensor Parallelism 基本思想

## 3.1 矩阵切分

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
    

---

# 3.2 Column Parallel Linear（列并行）

## 3.21 基本思想

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

---

## 3.22 Column Parallel 通信

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

---

# 5. Row Parallel Linear（行并行）

## 5.1 基本思想

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

---

## 5.2 Row Parallel 通信

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

# 6. Transformer 中 TP 如何应用

## 6.1 Attention TP

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

通常采用：

## QKV Column Parallel

切：

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

同理：

$$  
K,V  
$$

---

## 6.2 Attention Head 切分

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

# 7. Attention 输出投影 TP

Attention：

$$  
O=Attention(Q,K,V)W_O  
$$

因为：

Concat 后：

$$  
O\in R^{S\times H}  
$$

通常：

$$  
W_O  
$$

采用：

## Row Parallel

即：

$$  
W_O=  
\begin{bmatrix}  
W_0\  
W_1  
\end{bmatrix}  
$$

每 GPU：

计算部分：

$$  
O_i=X_iW_i  
$$

最后：

$$  
AllReduce(O_i)  
$$

---

# 8. FFN 中 TP

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

---

## 8.1 第一层 Column Parallel

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

---

## 8.2 第二层 Row Parallel

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

# 9. Megatron-LM TP 架构

一个 Transformer Block：

```
             X

             |
        LayerNorm

             |

       QKV Projection
        Column TP

             |

       Attention

             |

       Output Projection
        Row TP

             |

        Residual


             |

          FFN

    W1 Column TP

             |

          GELU

             |

    W2 Row TP

             |

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

# 10. TP 通信原理

## 10.1 为什么需要通信

TP 切 Tensor 后：

每张 GPU 只拥有部分结果。

例如：

Column Parallel：

GPU0:

$$  
Y_0  
$$

GPU1:

$$  
Y_1  
$$

下一层需要：

$$  
Y=[Y_0,Y_1]  
$$

所以：

需要：

$$  
AllGather  
$$

---

## 10.2 AllReduce

AllReduce：

输入：

$$  
x_i  
$$

输出：

$$  
y=\sum_i x_i  
$$

流程：

```
GPU0 ----\
GPU1 ----- AllReduce ---> Sum
GPU2 ----/
```

用于：

Row Parallel。

---

# 11. TP 通信算法

## 11.1 Ring AllReduce

N 张 GPU：

分：

$$  
N  
$$

块。

两个阶段：

### Reduce-Scatter

每 GPU：

发送部分数据：

$$  
N-1  
$$

轮。

### AllGather

广播结果。

总通信量：

$$  
2\frac{N-1}{N}M  
$$

其中：

- $M$：Tensor 大小
    

---

## 11.2 NCCL 实现

NVIDIA GPU：

通常：

```
CUDA Kernel
      |
NCCL
      |
NVLink / NVSwitch / InfiniBand
```

---

# 12. TP 与其他并行方式组合

现代 LLM 使用：

## 12.1 3D Parallelism

```
             Model

              |

     ------------------

     TP × PP × DP


```

例如：

GPT-3：

```
TP=8

PP=8

DP=8


Total GPU:

8×8×8

=512
```

---

# 13. TP vs PP vs DP

|并行方式|切分对象|通信|优势|
|---|---|---|---|
|DP|Batch|Gradient AllReduce|简单|
|TP|Tensor|Layer 内通信|降低显存|
|PP|Layer|Activation|扩展模型深度|
|CP|Sequence|Attention|长上下文|
|EP|Expert|MoE Expert|扩大MoE容量|

---

# 14. Tensor Parallel 推理优化

## 14.1 推理为什么喜欢 TP

LLM Decode：

计算：

$$  
Y=XW  
$$

其中：

Batch 小：

$$  
B\approx1-32  
$$

单卡：

GEMM 小。

TP：

多个 GPU：

并行计算：

$$  
Latency  
\downarrow  
$$

---

# 15. TP 在 Decode 阶段的问题

## 15.1 通信成为瓶颈

Decode：

每生成一个 Token：

需要：

```
Linear
 ↓
AllReduce
 ↓
Next Layer
```

通信频繁。

因此：

TP 越大：

通信比例越高。

---

## 15.2 TP Scaling

Latency：

$$  
T=  
T_{compute}  
+  
T_{comm}  
$$

其中：

通信：

$$  
T_{comm}

\frac{M}{BW}  
$$

当：

$$  
T_{comm}>T_{compute}  
$$

继续增加 GPU：

收益下降。

---

# 16. TP Size 如何选择

经验：

|模型规模|推荐 TP|
|---|---|
|7B|1-2|
|13B|2-4|
|70B|4-8|
|175B|8-16|
|MoE 大模型|8-64|

原则：

$$  
TP\leq GPU\ 数量  
$$

并优先：

```
TP = NVLink 域大小
```

例如：

H100 DGX：

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

# 17. Tensor Parallel 发展趋势

## 17.1 TP + Sequence Parallel

解决：

LayerNorm、Dropout 等无法 TP 的算子。

SP：

切：

$$  
Sequence  
$$

降低 Activation Memory。

---

## 17.2 TP + Context Parallel

长上下文：

$$  
S=128k  
$$

Attention：

$$  
O(S^2)  
$$

CP：

切：

$$  
S  
$$

降低显存。

---

## 17.3 TP + Expert Parallel

MoE：

Dense:

```
TP
```

Expert:

```
EP
```

例如：

DeepSeek-V3：

```
TP
+
EP
+
DP
```

---

# 18. 总结

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