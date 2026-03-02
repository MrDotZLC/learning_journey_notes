# 1. Transformer 推理计算结构

Transformer 是当前 LLM 推理系统的核心结构。  
其单层结构如下：

```
Input
 ↓
RMSNorm
 ↓
Self Attention
 ↓
Residual Add
 ↓
RMSNorm
 ↓
MLP (SwiGLU)
 ↓
Residual Add
```

数学表达：

输入

[  
X \in \mathbb{R}^{T \times d_{model}}  
]

其中

|符号|含义|
|---|---|
|T|token长度|
|d_model|隐藏维度|

---

## 1.1 QKV 投影

线性投影：

[  
Q = XW_q  
]

[  
K = XW_k  
]

[  
V = XW_v  
]

权重：

[  
W_q,W_k,W_v \in \mathbb{R}^{d_{model}\times d_{model}}  
]

计算复杂度：

[  
F_{qkv} = 3Td_{model}^2  
]

---

## 1.2 Attention

Scaled dot product attention：

[  
S = \frac{QK^T}{\sqrt{d_h}}  
]

Softmax：

[  
P = softmax(S)  
]

输出：

[  
O = PV  
]

其中

[  
d_h = \frac{d_{model}}{H}  
]

复杂度：

[  
F_{attn} = 2T^2 d_h H  
]

即

[  
O(T^2 d_{model})  
]

---

## 1.3 MLP

LLM 通常使用 **SwiGLU**

结构：

[  
U = XW_u  
]

[  
G = XW_g  
]

激活：

[  
Z = (U \odot \sigma(U)) \odot G  
]

输出：

[  
Y = ZW_d  
]

复杂度：

[  
F_{mlp} = 4Td_{model}d_{ff}  
]

通常

```
d_ff ≈ 4 d_model
```

---

## 1.4 单层 FLOPs

总 FLOPs：

# [  
F_{layer}

3Td_{model}^2  
+  
2T^2d_{model}  
+  
8Td_{model}^2  
]

简化：

[  
F_{layer}  
≈  
11Td_{model}^2  
+  
2T^2d_{model}  
]

---

# 2. Prefill / Decode 性能模型

---

LLM 推理分为两个阶段。

```
Prefill
Decode
```

---

# 2.1 Prefill

Prefill 阶段：

```
一次性输入整个 prompt
```

复杂度：

[  
O(T^2)  
]

原因：

```
attention matrix
T × T
```

Prefill FLOPs：

[  
F_{prefill}  
≈  
L(11Td^2 + 2T^2d)  
]

---

# 2.2 Decode

Decode 阶段：

每次生成一个 token。

Attention：

```
new token × past tokens
```

复杂度：

[  
O(T)  
]

计算：

[  
QK^T  
]

其中

```
Q size = 1×d
K size = T×d
```

Decode FLOPs：

[  
F_{decode}  
≈  
L(11d^2 + 2Td)  
]

---

## 2.3 Decode 是 memory bound

Decode 的瓶颈：

```
读取 KV cache
```

KV cache 访问量：

[  
M = 2LHDT  
]

带宽限制：

[  
T_{decode}  
≈  
\frac{M}{BW}  
]

---

# 3. GPU Kernel 架构

---

# 3.1 GEMM Kernel

LLM 大部分计算是：

```
GEMM
```

矩阵：

[  
C = AB  
]

其中

[  
A \in \mathbb{R}^{M\times K}  
]

[  
B \in \mathbb{R}^{K\times N}  
]

---

# 3.2 GPU Tiling

GPU Kernel 使用 **分块计算**。

```
Block tile
Warp tile
Thread tile
```

示例：

```
Block tile : 128×128
Warp tile  : 64×64
Thread tile: 16×16
```

计算流程：

```
Load A tile → shared memory
Load B tile → shared memory
Compute
Repeat
```

---

# 3.3 Tensor Core

Tensor Core 计算单元执行：

[  
D = A \times B + C  
]

最小 tile：

```
16×16×16
```

Warp-level 指令：

```
wmma.mma.sync
```

FLOPs：

```
256 FMA / instruction
```

---

# 3.4 CUTLASS Kernel结构

CUTLASS GEMM pipeline：

```
Global Memory
   ↓
Shared Memory
   ↓
Registers
   ↓
Tensor Core
```

Pipeline：

```
Load tile
Compute tile
Prefetch next tile
```

形成 **software pipeline**。

---

# 4. Kernel Fusion

---

算子融合核心目标：

```
减少HBM访问
```

---

# 4.1 融合算子示例

MLP：

```
MatMul
Bias
SwiGLU
MatMul
```

融合：

```
FusedMLP
```

---

# 4.2 内存模型

未融合：

[  
M = MK + KN + 4MN  
]

融合：

[  
M = MK + KN + MN  
]

减少：

[  
3MN  
]

---

# 5. FlashAttention

---

FlashAttention 解决：

```
attention memory explosion
```

---

# 5.1 Attention Memory

传统 attention：

存储

[  
S = QK^T  
]

大小：

[  
N^2  
]

访存：

[  
O(N^2)  
]

---

# 5.2 FlashAttention 思想

使用 **block streaming**：

```
Q_block
K_block
V_block
```

只计算局部块。

避免：

```
store attention matrix
```

---

# 5.3 Online Softmax

Softmax：

# [  
softmax(x_i)

\frac{e^{x_i}}{\sum_j e^{x_j}}  
]

FlashAttention 使用：

```
running max
running sum
```

更新：

[  
m = max(m, m_{block})  
]

[  
l = e^{m_{old}-m}l + e^{m_{block}-m}l_{block}  
]

---

# 6. KV Cache

---

Decode 阶段缓存：

```
K,V
```

避免重复计算。

---

# 6.1 KV Cache Layout

常见布局：

```
[layer][head][token][dim]
```

内存：

[  
2LHDT  
]

---

# 6.2 KV Cache 规模

示例：

```
L = 80
H = 64
D = 128
T = 8192
dtype = FP16
```

内存：

[  
2×80×64×128×8192×2B  
≈ 21GB  
]

---

# 7. PagedAttention

---

KV cache 使用 **分页管理**。

Page：

```
16 tokens
```

数据结构：

```
sequence
 ├ page0
 ├ page1
 └ page2
```

Page table：

```
logical → physical
```

类似：

```
virtual memory
```

---

# 8. Continuous Batching

---

GPU 推理调度策略。

传统：

```
static batch
```

问题：

```
GPU idle
```

---

# 8.1 Continuous batching

调度流程：

```
请求进入
加入batch
逐token生成
完成即退出
```

GPU 始终执行：

```
decode step
```

---

# 9. Speculative Decoding

---

目标：

```
减少 autoregressive latency
```

---

# 9.1 算法

使用：

```
draft model
target model
```

流程：

```
draft → generate k tokens
target → verify
```

接受概率：

```
p
```

速度：

# [  
Speedup

\frac{1}{1-p}  
]

---

# 10. Quantization

---

量化减少：

```
memory bandwidth
```

---

# 10.1 线性量化

量化：

[  
q = round(\frac{x}{s})  
]

反量化：

[  
x = q s  
]

---

# 10.2 量化误差

误差：

[  
e = x - qs  
]

最大误差：

[  
|e| \le \frac{s}{2}  
]

---

# 11. 推理系统架构

---

典型推理系统：

```
API server
 ↓
Scheduler
 ↓
Model runner
 ↓
GPU kernels
```

典型框架：

- vLLM
    
- TensorRT-LLM
    
- TGI
    

---

# 12. 推理优化核心方向

LLM 推理优化五个方向：

|方向|技术|
|---|---|
|Kernel|Fusion / FlashAttention|
|Memory|KV cache / paging|
|Scheduling|continuous batching|
|Algorithm|speculative decoding|
|Precision|quantization|

核心目标：

```
减少HBM访问
提高GPU利用率
减少kernel launch
```

---

# 13. 推理优化技术关系图

```
LLM Inference Optimization
│
├ Kernel
│  ├ GEMM
│  ├ Fusion
│  └ FlashAttention
│
├ Memory
│  ├ KV cache
│  ├ PagedAttention
│  └ Quantization
│
├ Scheduling
│  └ Continuous batching
│
└ Algorithm
   └ Speculative decoding
```

---

如果继续深入，还可以补充三份 **推理优化工程级笔记**（每份约 1–2 万字）：

1️⃣ **FlashAttention CUDA Kernel 逐行解析（含完整代码）**  
2️⃣ **Tensor Core GEMM 内核设计（CUTLASS / WMMA）**  
3️⃣ **vLLM PagedAttention GPU Kernel 实现解析**

这三部分基本就是 **推理优化工程师面试和实际工作最核心的底层内容**。