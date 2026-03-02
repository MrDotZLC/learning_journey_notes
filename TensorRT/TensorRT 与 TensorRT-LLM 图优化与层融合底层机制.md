深度学习推理框架（TensorRT、TensorRT-LLM、TVM、XLA 等）本质上属于 **DL Compiler / Inference Compiler**。  
其核心任务是：
- 将高层计算图 (DAG)
- 映射到 GPU 的 **计算单元 + 内存层级**
- 并通过 **算子融合、调度优化、Kernel 自动选择**

最大化利用：

- SM 计算资源
    
- HBM 带宽
    
- L2 / Shared Memory / Register
    

图优化与层融合的根本目标只有一个：

> **减少 HBM 访存 + 提升 Arithmetic Intensity + 减少 Kernel Launch**

在 LLM 推理中，这直接影响：

- **TTFT (Time To First Token)**
    
- **Throughput (Tokens/s)**
    

# 1. GPU Roofline 视角：层融合的硬件本质

GPU 性能上限由 **Roofline Model** 决定。

## 1.1 Roofline 模型

GPU 计算性能上限：

[  
P = \min(P_{peak},\ I \times B_{mem})  
]

其中：

|符号|含义|
|---|---|
|(P)|实际性能|
|(P_{peak})|GPU 理论算力|
|(B_{mem})|显存带宽|
|(I)|算术强度|

---

## 1.2 算术强度 (Arithmetic Intensity)

算术强度定义：

[  
I = \frac{FLOPs}{Bytes}  
]

表示：

> 每访问 1 Byte 数据所执行的计算量

算子类型可分为：

|算子类型|Arithmetic Intensity|瓶颈|
|---|---|---|
|Elementwise|极低|Memory Bound|
|Softmax|低|Memory Bound|
|LayerNorm|低|Memory Bound|
|GEMM|高|Compute Bound|

---

## 1.3 层融合提升算术强度

若算子链：

```
LayerNorm → SiLU → Elementwise
```

非融合时：

```
HBM → Kernel1 → HBM → Kernel2 → HBM → Kernel3 → HBM
```

HBM 访问次数：

```
读 + 写 + 读 + 写 + 读 + 写
```

融合后：

```
HBM → Fused Kernel → HBM
```

中间数据只存在：

- **Register**
    
- **Shared Memory**
    

Arithmetic Intensity 提升：

[  
I_{fused} > I_{unfused}  
]

---

## 1.4 Kernel Launch 开销

每次 CUDA Kernel Launch 包含：

- CPU → GPU 调度
    
- 参数传输
    
- Stream 同步
    

典型开销：

```
2 – 5 μs
```

对于 LLM decoding：

```
每 token 数十个 kernel
```

Launch Overhead 会显著影响：

```
TTFT
```

融合后：

```
Kernel 数量 ↓
```

---

---

# 2. TensorRT 图优化编译流水线 (Builder Pipeline)

TensorRT 的优化过程发生在 **Engine Build Phase**。

整体流程：

```
ONNX / Network Definition
        │
        ▼
Graph Canonicalization
        │
        ▼
Pattern Matching
        │
        ▼
Layer Fusion
        │
        ▼
Tactic Search
        │
        ▼
Engine Plan
```

---

## 2.1 Canonicalization（图规范化）

目的：

> 消除前端框架差异，统一计算图

主要优化：

### 常量传播 (Constant Propagation)

若子图输入为常量：

[  
y = f(c_1, c_2)  
]

TensorRT 会在 **CPU build 阶段直接计算**：

```
subgraph → Constant Node
```

减少 runtime kernel。

---

### 公共子表达式消除 (CSE)

若存在：

```
A = f(X)
B = f(X)
```

优化为：

```
T = f(X)
A = T
B = T
```

减少重复计算。

---

## 2.2 Pattern Matching 与 Layer Fusion

TensorRT 会在 **DAG 中搜索子图模式**。

匹配方式：

```
Graph Pattern Matching
```

匹配成功：

```
Subgraph → Fused Layer
```

---

## 2.3 QKV Projection Fusion

Transformer 中：

[  
Q = XW_Q  
]

[  
K = XW_K  
]

[  
V = XW_V  
]

原始执行：

```
3 × GEMM
```

TensorRT 将权重拼接：

[  
W_{QKV} = [W_Q\ |\ W_K\ |\ W_V]  
]

计算：

[  
[Q,K,V] = X W_{QKV}  
]

优点：

- GEMM 数量减少
    
- GPU 利用率更高
    
- memory read 减少
    

---

## 2.4 Activation Fusion

例如：

```
GEMM → Bias → GELU
```

可融合为：

```
FusedGEMMKernel
```

在 **epilogue** 阶段计算：

```
Y = GELU(XW + b)
```

实现方式：

- cuBLASLt epilogue
    
- TensorRT fused kernel
    

---

## 2.5 Tactic Auto-Tuning

TensorRT 会从多个 backend 中选择实现：

可能实现来源：

- cuBLAS
    
- cuBLASLt
    
- cuDNN
    
- CUTLASS
    
- TensorRT custom kernels
    

搜索空间包括：

- tile size
    
- warp mapping
    
- shared memory usage
    
- pipeline depth
    

最终选择：

```
latency 最小 tactic
```

并写入：

```
Engine Plan
```

---

---

# 3. TensorRT-LLM 的极致融合策略

LLM 中最昂贵算子：

```
Attention
```

复杂度：

[  
O(N^2)  
]

Memory 访问极其密集。

---

## 3.1 标准 Attention 计算

[  
S = \frac{QK^T}{\sqrt{d_k}}  
]

[  
P = \text{Softmax}(S)  
]

[  
O = PV  
]

若序列长度：

```
N = 16k
```

矩阵大小：

```
N × N
```

需要写入：

```
Score matrix S
Softmax matrix P
```

显存压力巨大。

---

## 3.2 FlashAttention 核心思想

FlashAttention 将 Attention **完全融合为单 Kernel**。

关键思想：

```
Block Tiling + Online Softmax
```

---

### Step 1：Block 切分

将：

```
Q,K,V
```

分块：

```
Qi, Kj, Vj
```

每个 block 可放入：

```
Shared Memory
```

---

### Step 2：局部 Attention

计算：

[  
S_{ij} = Q_i K_j^T  
]

---

### Step 3：Online Softmax

维护：

- 最大值 (m)
    
- 指数和 (l)
    

更新：

[  
m_{new} = \max(m_{old}, m_{block})  
]

[  
l_{new} =  
l_{old} e^{m_{old}-m_{new}} +  
l_{block} e^{m_{block}-m_{new}}  
]

避免：

```
存储完整 S
```

---

### Step 4：累加输出

[  
O_i += P_{ij} V_j  
]

输出保存在：

```
Registers
```

---

## 3.3 复杂度变化

标准 Attention：

```
HBM IO = O(N²)
```

FlashAttention：

```
HBM IO = O(N)
```

---

## 3.4 TensorRT-LLM MHA Kernel

TensorRT-LLM 中：

```
QKV projection
RoPE
KV cache
FlashAttention
```

通常融合为：

```
Single Attention Kernel
```

减少：

- HBM IO
    
- kernel launch
    

---

## 3.5 MoE Expert Fusion

MoE 推理瓶颈：

```
大量小 GEMM
```

例如：

```
top-2 routing
```

若 token 数：

```
N
```

专家数：

```
E
```

传统执行：

```
N 次 expert kernel
```

---

### Token Sorting

TensorRT-LLM 先进行：

```
token → expert
```

排序：

```
token reordering
```

使同 expert token 连续。

---

### Grouped GEMM

然后执行：

```
Grouped GEMM
```

一次 kernel 计算多个 expert。

优点：

- 提升 GPU occupancy
    
- 减少 launch
    

---

---

# 4. 自动融合失败时的工程干预

TensorRT 无法识别：

- 新算子
    
- 新架构
    
- 自定义算子
    

必须使用：

```
Plugin
```

---

## 4.1 TensorRT Plugin 机制

TensorRT Plugin 是：

```
Custom Layer
```

开发者实现：

```
CUDA Kernel
```

并注册为图节点。

---

## 4.2 IPluginV3 架构

TensorRT 10.x 推荐接口：

```
IPluginV3
```

分为三个阶段接口：

|接口|职责|
|---|---|
|IPluginV3OneCore|插件元信息|
|IPluginV3OneBuild|构建期逻辑|
|IPluginV3OneRuntime|运行期执行|

---

---

# 5. 示例：RMSNorm + SiLU 融合算子

## 5.1 RMSNorm

设输入：

[  
x \in \mathbb{R}^D  
]

均方：

[  
v = \frac{1}{D}\sum_{i=1}^{D} x_i^2  
]

归一化因子：

[  
inv_rms =  
\frac{1}{\sqrt{v + \epsilon}}  
]

输出：

[  
x_{norm} = x \cdot inv_rms \odot \gamma  
]

---

## 5.2 SiLU

[  
\text{SiLU}(x)=x\sigma(x)  
]

[  
\sigma(x)=\frac{1}{1+e^{-x}}  
]

最终输出：

[  
y = x_{norm} \cdot  
\frac{1}{1+e^{-x_{norm}}}  
]

---

## 5.3 非融合访存

RMSNorm：

```
Read x
Write x_norm
```

SiLU：

```
Read x_norm
Write y
```

访存量：

```
4 × D × sizeof(T)
```

---

## 5.4 融合访存

```
Read x
Compute
Write y
```

访存量：

```
2 × D × sizeof(T)
```

减少：

```
50% HBM IO
```

---

---

# 6. CUDA Kernel 示例

```cpp
template <typename T>
__global__ void FusedRMSNormSiLUKernel(
    const T* input,
    const T* gamma,
    T* output,
    float epsilon,
    int D)
{
    int token = blockIdx.x;
    int tid   = threadIdx.x;

    const T* x = input + token * D;
    T* y = output + token * D;

    float local_sq_sum = 0.f;

    for (int i = tid; i < D; i += blockDim.x)
    {
        float v = static_cast<float>(x[i]);
        local_sq_sum += v * v;
    }

    typedef cub::BlockReduce<float,256> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp;

    float block_sum =
        BlockReduce(temp).Sum(local_sq_sum);

    __shared__ float inv_rms;

    if (tid == 0)
        inv_rms = rsqrtf(block_sum / D + epsilon);

    __syncthreads();

    for (int i = tid; i < D; i += blockDim.x)
    {
        float v = static_cast<float>(x[i]);

        float norm = v * inv_rms *
                     static_cast<float>(gamma[i]);

        float silu =
            norm / (1.f + expf(-norm));

        y[i] = static_cast<T>(silu);
    }
}
```

---

---

# 7. IPluginV3 Runtime 执行逻辑

关键接口：

```
IPluginV3OneRuntime::enqueue
```

负责：

```
Kernel Launch
```

示例：

```cpp
int32_t enqueue(
    PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept override
{
    int N =
        inputDesc[0].dims.d[0] *
        inputDesc[0].dims.d[1];

    int D =
        inputDesc[0].dims.d[2];

    dim3 grid(N);
    dim3 block(256);

    FusedRMSNormSiLUKernel<float>
        <<<grid, block, 0, stream>>>(
            (float*)inputs[0],
            (float*)mGammaWeights,
            (float*)outputs[0],
            mEpsilon,
            D);

    return cudaGetLastError()==cudaSuccess
           ? 0 : -1;
}
```

---

---

# 8. CUDA Graph 消除 Launch Overhead

对于无法继续融合的算子：

```
CUDA Graph
```

可将多个 Kernel：

```
Capture → Graph
```

运行时：

```
Graph Launch
```

CPU 开销接近：

```
0
```

示例：

```cpp
cudaGraph_t graph;
cudaGraphExec_t instance;

cudaStreamBeginCapture(stream,
    cudaStreamCaptureModeGlobal);

context->enqueueV2(bindings,
                   stream,
                   nullptr);

cudaStreamEndCapture(stream,&graph);

cudaGraphInstantiate(&instance,
                     graph,
                     NULL,
                     NULL,
                     0);

cudaGraphLaunch(instance,stream);
```

典型收益：

```
decode latency ↓
```

---

---

# 9. 推理编译器优化总结

TensorRT / TensorRT-LLM 的核心优化方向：

|优化层级|技术|
|---|---|
|Graph Level|Pattern Fusion|
|Kernel Level|FlashAttention / Fused MLP|
|Memory Level|Shared Memory / Register reuse|
|Scheduling|Tactic Auto-Tuning|
|Runtime|CUDA Graph|

最终目标：

```
减少 HBM IO
提高 SM 利用率
降低 Kernel Launch
```

这也是现代 **LLM 推理优化工程师**最核心的优化路径。