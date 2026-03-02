# 1. 算子融合的本质：对抗 Memory Wall
## 1.1 GPU 推理性能瓶颈
现代 GPU 推理性能通常受 **内存带宽限制** 而非计算能力限制。
GPU 执行时间近似为
$$
T = \max(T_{compute}, T_{memory})
$$
其中
$$
T_{compute} = \frac{F}{\pi}
$$
$$
T_{memory} = \frac{M}{\beta}
$$
符号定义：

| 符号      | 含义           |
| ------- | ------------ |
| $F$     | FLOPs（计算量）   |
| $M$     | 访存字节数        |
| $\pi$   | GPU 峰值 FLOPS |
| $\beta$ | GPU 内存带宽     |
引入 **算术强度 (Arithmetic Intensity)**：
$$
I = \frac{F}{M}
$$
则 **Roofline Model** 表示为
$$
P = \min(\pi, I \cdot \beta)
$$
解释：
* 当 $I < \frac{\pi}{\beta}$
  → **Memory Bound**
* 当
$$
I > \frac{\pi}{\beta}
$$
→ **Compute Bound**
## 1.2 Memory-bound 算子的典型特征
LLM 推理中大量算子属于：
* Elementwise
* Normalization
* Activation
典型算子：

| 算子        | FLOPs        | 访存  | 特性           |
| --------- | ------------ | --- | ------------ |
| ReLU      | 1            | 2   | 极低算术强度       |
| Add       | 1            | 3   | memory bound |
| LayerNorm | ~10          | 大量  | 强访存依赖        |
| Softmax   | exp + reduce | 高访存 | 典型 IO-bound  |
这些算子的性能主要由
$$
T \approx \frac{M}{\beta}
$$
决定。
## 1.3 Operator Fusion 的核心思想
目标：
**减少 Global Memory Traffic**
未融合：
```
A -> Op1 -> DRAM -> Op2 -> DRAM -> Op3 -> DRAM
```
融合：
```
A -> FusedKernel -> DRAM
```
核心原则：
> **中间结果停留在 Register / Shared Memory**
从而减少
* Global memory load
* Global memory store
---
# 2. 算子融合合法性条件
算子能否融合取决于 **数据依赖关系**。
设计算图
$$
G = (V, E)
$$
其中
* $V$ = 算子
* $E$ = 数据依赖
融合必须满足：
### 2.1 Producer-Consumer
如果
$$
Op_A \rightarrow Op_B
$$
且
```
Output(A) 仅被 B 使用
```
则可以融合。
### 2.2 无 Barrier
不能存在：
* 同步操作
* 通信算子
* 动态 shape barrier
例如：
```
MatMul -> AllReduce -> Add
```
无法完全融合。
### 2.3 无 Memory Alias
若两个算子写入同一 buffer：
```
A -> write X
B -> write X
```
则无法直接融合。

---
# 3. 算子融合分类
## 3.1 垂直融合 (Vertical Fusion)
典型：
```
MatMul
  ↓
BiasAdd
  ↓
Activation
```
融合为
```
FusedMatMulBiasActivation
```
数学表达：
$$
Y = \sigma(XW + b)
$$
Kernel 内部执行：
```
acc = dot(x, w)
acc += bias
acc = activation(acc)
```
仅 **一次 global write**。
## 3.2 水平融合 (Horizontal Fusion)
针对 **多个相同算子**。
Transformer 中：
$$
Q = XW_q
$$
$$
K = XW_k
$$
$$
V = XW_v
$$
可以融合为
$$
[XW_q ; XW_k ; XW_v]
$$
矩阵形式：
$$
W =
\begin{bmatrix}
W_q & W_k & W_v
\end{bmatrix}
$$
变为
$$
[XW]
$$
优势：
* 更大 GEMM
* 更高 GPU occupancy
## 3.3 Reduction Fusion
典型算子：
* Softmax
* LayerNorm
* RMSNorm
示例：
```
x -> square -> reduce -> sqrt -> divide
```
融合为单 Kernel。
利用：
* Warp Shuffle
* Block Reduction
避免：
```
Shared Memory Round-trip
```
---
# 4. 内存访问优化推导
考虑融合算子
$$
A = ReLU(XW + b)
$$
矩阵大小：
$$
X \in \mathbb{R}^{M \times K}
$$
$$
W \in \mathbb{R}^{K \times N}
$$
## 4.1 未融合访存
步骤：
### Step1
MatMul
读：
$$
MK + KN
$$
写：
$$
MN
$$
### Step2
BiasAdd
读：
$$
MN + N
$$
写：
$$
MN
$$
### Step3
ReLU
读：
$$
MN
$$
写：
$$
MN
$$
总访存：
$$
M_{unfused}
=
MK + KN + 4MN + N
$$
## 4.2 融合后
Kernel 内：
```
acc = dot(x,w)
acc += bias
acc = relu(acc)
```
访存：
读：
$$
MK + KN + N
$$
写：
$$
MN
$$
总访存：
$$
M_{fused}
=
MK + KN + MN + N
$$
减少：
$$
\Delta M = 3MN
$$
## 4.3 实际案例
假设
```
M = 1024
N = 4096
dtype = FP16
```
数据量：
$$
3MN \cdot 2\text{ bytes}
=
3 \times 1024 \times 4096 \times 2
\approx 24MB
$$
单层推理节省约 **24MB DRAM traffic**。
LLM 96 层：
```
≈ 2.3GB
```
---
# 5. Kernel Launch Overhead
GPU Kernel 启动存在固定开销：
$$
T_{launch} \approx 5-20\mu s
$$
若 3 个 kernel：
```
MatMul
Bias
ReLU
```
时间：
$$
T = 3T_{launch} + T_{compute}
$$
融合：
```
T = T_{launch} + T_{compute}
```
减少：
```
2 × kernel launch latency
```
在 **小 batch inference** 场景中尤为明显。

---
# 6. Transformer 中的典型融合
Transformer block 中：
```
MatMul(QKV)
 ↓
RoPE
 ↓
Attention
 ↓
Softmax
 ↓
MatMul(V)
 ↓
Linear
```
常见融合：
### QKV Fusion
```
XWq
XWk
XWv
```
→
```
Grouped GEMM
```
### Attention Fusion
FlashAttention：
```
QK^T
 ↓
Scale
 ↓
Mask
 ↓
Softmax
 ↓
PV
```
全部融合。

---
# 7. FlashAttention 的 IO Complexity
## 传统 Attention：
需要存储
$$
S = QK^T
$$
矩阵大小：
$$
N \times N
$$
访存：
$$
O(N^2)
$$
## FlashAttention 使用 **block streaming**：
分块：
```
Q_block
K_block
```
只计算局部块。
IO Complexity：
$$
O(Nd)
$$
其中
$$
d = head\ dimension
$$
从
```
N²
```
降低为
```
Nd
```
---
# 8. CUDA Kernel 融合结构
融合 Kernel 层级：
```
Register
Shared Memory
Global Memory
```
优化策略：

| 级别            | 用途         |
| ------------- | ---------- |
| Register      | 中间结果       |
| Shared Memory | block tile |
| Global Memory | 输入输出       |
## 示例：Warp-level Fusion
```cpp
#include <cuda_fp16.h>
__global__ void fused_linear_swiglu(
    const half* __restrict__ x,
    const half* __restrict__ w_up,
    const half* __restrict__ w_gate,
    half* __restrict__ out,
    int hidden)
{
    int row = blockIdx.x;
    int col = threadIdx.x;
    float acc_up = 0.0f;
    float acc_gate = 0.0f;
    for(int k=0;k<hidden;k++)
    {
        float xv = __half2float(x[row*hidden+k]);
        acc_up   += xv * __half2float(w_up[k*hidden+col]);
        acc_gate += xv * __half2float(w_gate[k*hidden+col]);
    }
    float sig = 1.f / (1.f + __expf(-acc_up));
    float res = (acc_up * sig) * acc_gate;
    out[row*hidden+col] = __float2half(res);
}
```
融合算子：
```
Linear
Sigmoid
Mul
Gate
```
---
# 9. Triton 融合机制
Triton 编译流程：
```
Python Kernel
     ↓
Triton IR
     ↓
MLIR
     ↓
LLVM IR
     ↓
PTX
```
编译器执行：
* Pattern Fusion
* Loop Tiling
* Vectorization
示例：
```
RMSNorm + SwiGLU
```
自动融合为单 Kernel。

---
# 10. 2025–2026 前沿趋势
## 10.1 NVIDIA Blackwell FP4
新数据格式：
```
NVFP4
```
结构：
```
4bit mantissa
shared scaling factor
```
Micro-block scaling：
```
16 elements → 1 scale
```
优势：

| 精度   | 内存   |
| ---- | ---- |
| FP16 | 2B   |
| FP8  | 1B   |
| FP4  | 0.5B |
内存下降：
```
4x vs FP16
```
## 10.2 Warp Specialization
FlashAttention-3 使用：
```
Producer Warp
Consumer Warp
```
Producer：
```
TMA copy
```
Consumer：
```
Tensor Core compute
```
形成 **异步流水线**。

---
# 11. 融合限制
并非所有算子都适合融合。
限制：

| 原因                  | 说明           |
| ------------------- | ------------ |
| Register pressure   | 寄存器溢出        |
| Shared memory limit | SMEM 不足      |
| 调度复杂                | Occupancy 下降 |
| 动态 shape            | 难以编译优化       |
| 通信算子                | 无法融合         |
过度融合可能导致：
```
register spill
```
反而 **降低性能**。

---
# 12. 推理框架中的融合实现
| 框架               | 融合方式                    |
| ---------------- | ----------------------- |
| TensorRT         | Pattern Graph Fusion    |
| TVM              | Auto-scheduler          |
| PyTorch Inductor | JIT Kernel Fusion       |
| Triton           | DSL Kernel Fusion       |
| TensorRT-LLM     | Attention + Comm Fusion |

---
# 13. 核心结论
算子融合的本质：
```
减少 DRAM 访问
增加 Arithmetic Intensity
```
主要收益来源：
1. 减少 global memory traffic
2. 减少 kernel launch
3. 提高 GPU occupancy
4. 提高 arithmetic intensity
在 LLM 推理中：
```
FlashAttention
RMSNorm fusion
MLP fusion
QKV fusion
```
是最关键的性能优化手段。
