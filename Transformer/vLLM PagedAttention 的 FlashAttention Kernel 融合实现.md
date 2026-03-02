# 1 PagedAttention + FlashAttention Kernel 融合实现
>PagedAttention 解决 **KV cache 管理问题**。  
>FlashAttention 解决 **Attention 访存复杂度问题**。

现代推理引擎（2024–2026）通常采用：
```
Paged KV Cache
+
FlashAttention Streaming Kernel
```
形成统一 GPU Kernel。
典型系统：

| 推理系统         | Attention Kernel                |
| ------------ | ------------------------------- |
| vLLM         | PagedAttention + FlashAttention |
| TensorRT-LLM | FMHA kernel                     |
| SGLang       | Paged FlashAttention            |
| TGI          | FlashAttention                  |
核心思想：
```
block streaming
+
online softmax
+
register accumulation
```
避免构造：
```
QK^T matrix
```
---
# 2 Tile Streaming Attention
## 2.1 Attention 复杂度
标准 Attention：
$$S = QK^T$$
维度：
$$Q \in \mathbb{R}^{T_q \times d}$$
$$K \in \mathbb{R}^{T_k \times d}$$
Score matrix：
$$S \in \mathbb{R}^{T_q \times T_k}$$
内存复杂度：
$$O(T_q T_k)$$
例如：

|seq_len|matrix size|
|---|---|
|4096|64MB|
GPU 显存压力巨大。
## 2.2 Streaming Attention
FlashAttention 使用 **Tile Streaming**：
将 KV 分块：
```
K = [K1 K2 K3 ...]
V = [V1 V2 V3 ...]
```
Kernel 计算流程：
```
for block in KV_blocks:
    load K_block
    score = Q * K_block
    update softmax
    load V_block
    update output
```
无需保存 score matrix。
内存复杂度：
$$O(T)$$
## 2.3 Tile 数学推导
设 block 大小：
$$B$$
则：
$$K =  
\begin{bmatrix}  
K_1 \  
K_2 \  
\vdots \  
K_n  
\end{bmatrix}$$
每次计算：
$$S_i = Q K_i^T$$
Softmax 需要：
$$P_i = \frac{e^{S_i}}{\sum_j e^{S_j}}$$
FlashAttention 使用 **online softmax**：
维护：
```
m = running max
l = running sum
```
递推公式：
$$m_{new} = \max(m_{old}, \max(S_i))$$
$$l_{new} =  
l_{old} e^{m_{old}-m_{new}}  
+  
\sum e^{S_i - m_{new}}$$
输出累加：
$$O_{new}
O_{old} e^{m_{old}-m_{new}}  
+  
\sum e^{S_i - m_{new}} V_i$$
这样即可 **逐块计算 attention**。

---
# 3 PagedAttention Tile Kernel
结合分页 KV cache：
```
logical sequence
   ↓
block table
   ↓
physical KV block
```
Kernel 逻辑：
```
for logical_block in seq:
    block_id = block_table[logical_block]
    K_ptr = kv_cache + block_id * block_stride
```
因此：
```
FlashAttention tile
+
Paged KV lookup
```
成为统一 kernel。

---
# 4 GPU Kernel Thread Mapping
Paged FlashAttention kernel 的核心设计：
```
Block -> head
Warp  -> query token
Thread -> head_dim
```
示例：

|GPU结构|任务|
|---|---|
|Thread Block|sequence head|
|Warp|query token|
|Thread|dimension slice|
示例：
```
head_dim = 128
warp = 32 threads
```
则：
```
thread -> 4 dims
```
---
# 5 Warp Specialization
FlashAttention-3（Hopper GPU）引入：
```
warp specialization
```
思想：
不同 warp 执行不同任务。
### Warp 类型
|Warp 类型|职责|
|---|---|
|Producer warp|global memory load|
|Consumer warp|TensorCore compute|
Pipeline：
```
Producer warp
    ↓
shared memory
    ↓
Consumer warp
```
实现：
```
load K/V tile
while compute previous tile
```
消除：
```
memory latency stall
```
---
# 6 Tensor Core Attention Kernel
现代 Attention kernel 会使用：
```
Tensor Core
```
矩阵乘：
$$QK^T$$
转化为：
```
mma.sync
```
典型 tile：
```
16 × 16 × 16
```
CUDA PTX：
```
mma.sync.aligned.m16n16k16
```
计算流程：
```
load Q tile
load K tile
mma.sync
accumulate score
```
Tensor Core throughput：

|GPU|TFLOPS FP16|
|---|---|
|A100|312|
|H100|989|
|B200|>2000|
Attention kernel 必须：
```
tile blocking
+
tensor core
```
才能接近峰值。

---
# 7 Shared Memory Tile Layout
为了避免 bank conflict：
K tile layout：
```
[K0 K1 K2 K3 ...]
```
存储为：
```
[dim][token]
```
这样 warp load：
```
thread0 -> dim0
thread1 -> dim1
```
可实现：
```
coalesced access
```
Shared memory 示例：
```
shared_K[BLOCK_K][HEAD_DIM]
```
访问：
```
shared_K[token][dim]
```
---
# 8 Triton Kernel 与 CUDA Kernel 对应
Triton kernel：
```
program_id -> block
```
CUDA kernel：
```
blockIdx -> block
threadIdx -> thread
```
对比：

|Triton|CUDA|
|---|---|
|program_id|blockIdx|
|tl.arange|threadIdx|
|tl.load|global load|
|tl.store|global store|
Triton 会自动：
```
vectorize
coalesce memory
schedule warps
```
示例：
```python
pid = tl.program_id(0)
q = tl.load(Q + pid)
for b in range(num_blocks):
    block_id = tl.load(block_table + b)
    k = tl.load(K + block_id)
    score = tl.sum(q * k)
```
编译 pipeline：
```
Python
↓
Triton IR
↓
LLVM IR
↓
PTX
↓
SASS
```
---
# 9 Paged FlashAttention Kernel 完整流程
最终 GPU kernel pipeline：
```
load Q
↓
lookup block_table
↓
for each KV block
    load K tile
    compute QK
    update softmax
    load V tile
    update output
```
关键优化：
```
register accumulation
warp shuffle reduction
tensor core mma
tile streaming
paged KV lookup
```
---
# 10 推理系统中的位置
完整推理 pipeline：
```
Token Embedding
↓
Linear
↓
Paged FlashAttention
↓
MLP (SwiGLU)
↓
KV cache append
```
其中最核心 kernel：
```
Paged FlashAttention
```
约占推理时间：
```
40% ~ 70%
```
---
# 11 当前推理引擎实现
|系统|Kernel 技术|
|---|---|
|vLLM|Paged FlashAttention|
|TensorRT-LLM|FMHA kernel|
|DeepSpeed-Inference|fused attention|
|SGLang|streaming attention|
核心优化方向：
```
KV cache layout
warp scheduling
tensor core usage
```
