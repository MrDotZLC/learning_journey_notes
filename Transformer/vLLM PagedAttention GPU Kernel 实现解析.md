## 1. LLM 推理中的 KV Cache 问题
在自回归推理中，每生成一个 token，需要访问所有历史 token 的 Key 与 Value。
Attention计算：
$$Attention(Q,K,V)
softmax\left(\frac{QK^T}{\sqrt{d}}\right)V$$
其中：

|符号|含义|
|---|---|
|Q|当前 token query|
|K|历史 token key|
|V|历史 token value|
假设：
$$T = sequence\ length$$
历史缓存：
$$K =  
\begin{bmatrix}  
K_1 \  
K_2 \  
\vdots \  
K_T  
\end{bmatrix}$$
$$V =  
\begin{bmatrix}  
V_1 \  
V_2 \  
\vdots \  
V_T  
\end{bmatrix}$$
这就是 **KV Cache**。

---
# 2. KV Cache 内存规模
KV cache 大小：
$$Memory
2 \times L \times H \times D \times T \times bytes$$
符号：

|符号|含义|
|---|---|
|L|transformer层数|
|H|attention heads|
|D|head dimension|
|T|sequence length|
示例：
```
L = 80
H = 64
D = 128
T = 4096
dtype = FP16 (2B)
```
计算：
$$Memory
2\times80\times64\times128\times4096\times2
≈
10.7 GB$$

仅一个 sequence。

---
# 3. 传统 KV Cache 内存布局
传统布局：
```
[layer][head][token][dim]
```
GPU地址：
$$addr =  
base +  
layer_offset +  
head_offset +  
token_offset$$
**优点**：
	地址连续
	访存简单
**缺点**：
	memory fragmentation

---
# 4. KV Cache Fragmentation
推理服务：variable length requests
例如：
```
request1 length = 200
request2 length = 3000
request3 length = 500
```
内存：
	`[request1][request2][request3]`
若 request2 结束：
	`[request1][free][request3]`
产生 memory hole，碎片化。
GPU 内存利用率可能下降到：40%

---
# 5. PagedAttention 设计思想
PagedAttention：
	KV cache → pages
类似：
	virtual memory paging
结构：
```
sequence
   │
   ├ page0
   ├ page1
   ├ page2
```
page_size = 16 tokens

---
# 6. Page 数据结构
KV cache page：
```cpp
struct KVPage {
    half K[PAGE_SIZE][NUM_HEADS][HEAD_DIM];
    half V[PAGE_SIZE][NUM_HEADS][HEAD_DIM];
};
```
全局存储：
	`[page0][page1][page2]...`
Sequence 记录：page list

---
# 7. Page Table
Sequence page table：
```cpp
struct SequencePageTable {
    int pages[MAX_PAGES];
};
```
token → page：
$$page_id =  
\left\lfloor  
\frac{token_id}{page_size}  
\right\rfloor$$
offset：
$$offset =  
token_id \mod page_size$$
GPU地址：
$$addr =  
page_base +  
offset$$
---
# 8. Attention Kernel 访问流程
Decode 阶段：
当前 token：
```
Q
```
访问：
```
K1 ... KT
```
PagedAttention：
```
for page in page_table
    load K_page
    compute QK
```
流程：
```
query
 ↓
page table lookup
 ↓
load KV pages
 ↓
QK^T
 ↓
softmax
 ↓
PV
```
---
# 9. GPU Kernel Block Mapping
PagedAttention Kernel：
```
1 block → 1 attention head
```
线程：
```
blockDim = HEAD_DIM
```
Warp：
```
32 threads
```
例如：
```
HEAD_DIM = 128
```
配置：
```
blockDim = 128
warps = 4
```
---
# 10. Kernel 完整执行流程
GPU kernel 逻辑：
```
load query
 ↓
pass1: compute attention scores
 ↓
warp reduction → max
 ↓
pass2: compute softmax
 ↓
load values
 ↓
accumulate output
```
分为两阶段：

|阶段|作用|
|---|---|
|score pass|计算QK|
|value pass|计算PV|

---
# 11. Warp-level Reduction
Softmax需要：
```
max(score)
sum(exp)
```
Warp reduction：
使用
```
__shfl_xor_sync
```
实现：
```cpp
float warp_reduce_max(float val)
{
    for(int offset=16; offset>0; offset/=2)
        val = max(val, __shfl_xor_sync(0xffffffff,val,offset));
    return val;
}
```
优势：
```
无需 shared memory
```
延迟：
```
~10 cycles
```
---
# 12. Shared Memory Layout
Shared memory 用于缓存：
```
query
K tile
V tile
```
示例布局：
```
shared_mem
 ├ query[HEAD_DIM]
 ├ K_tile[BLOCK_TOKENS][HEAD_DIM]
 └ V_tile[BLOCK_TOKENS][HEAD_DIM]
```
BLOCK_TOKENS：
```
16
```
减少：
```
HBM访问
```
---
# 13. Memory Coalescing
GPU高效访存要求：
```
32 threads → contiguous memory
```
布局：
```
token-major
```
地址：
$$addr =  
page_base +  
token_offset\times head_dim +  
lane$$
Warp读取：
```
128B transaction
```
实现：
```
coalesced memory load
```
---
# 14. Register 使用模型
每个 thread 使用 register 保存：
```
q_i
partial score
partial output
```
寄存器估计：
```
~40 registers / thread
```
GPU限制：
```
255 registers
```
Register spill 会导致：
```
local memory access
```
性能下降。

---
# 15. Block Scheduling
GPU SM 调度：
```
多个blocks并行
```
配置：
```
gridDim = num_heads × batch_size
```
例如：
```
num_heads = 64
batch = 16
```
- Grid：64 * 16 = 1024 blocks
- SM：假设为Blackwell B200 GPU，有 80 SM
- 理论上，每个 SM 承载：
$$\lceil 1024 / 80 \rceil \approx 13 \text{ blocks/SM}$$

---
# 16. 完整 CUDA Kernel 示例
简化示例：
```cpp
__global__ void paged_attention_kernel(
    float* query,
    float* key_cache,
    float* value_cache,
    int* page_table,
    float* output)
{
    int head = blockIdx.x;
    int lane = threadIdx.x;
    float q = query[head * HEAD_DIM + lane];
    float max_score = -1e30;
    // pass1: compute scores
    for(int p=0;p<num_pages;p++)
    {
        int page = page_table[p];
        float k = key_cache[page * PAGE_SIZE + lane];
        float score = q * k;
        max_score = max(max_score, score);
    }
    max_score = warp_reduce_max(max_score);
    float sum = 0.0;
    float out = 0.0;
    for(int p=0;p<num_pages;p++)
    {
        int page = page_table[p];
        float k = key_cache[page * PAGE_SIZE + lane];
        float score = q * k;
        float weight = exp(score - max_score);
        float v = value_cache[page * PAGE_SIZE + lane];
        sum += weight;
        out += weight * v;
    }
    out /= sum;
    output[head * HEAD_DIM + lane] = out;
}
```
---
# 17. IO Complexity
传统 attention：
$$O(T^2)$$
FlashAttention：
$$O(Td)$$
PagedAttention：
$$O(Td)$$
但优化：
```
memory allocation
fragmentation
```
---
# 18. GPU 带宽分析
Decode阶段：
每 token 读取：
```
K cache
V cache
```
数据量：
$$M =  
2 \times L \times H \times D \times T$$
若：
```
T = 4096
```
访问：
```
~40MB / token
```
GPU带宽：
```
H100 ≈ 3.3TB/s
```
理论 latency：
```
≈12 μs
```
因此 decode：
```
memory bound
```
---
# 19. 与 FlashAttention 的关系
FlashAttention：
```
attention compute kernel
```
PagedAttention：
```
KV memory manager
```
组合：
```
FlashAttention + PagedAttention
```
形成：
```
vLLM attention engine
```
---
# 20. vLLM Attention Engine
结构：
```
Query
 ↓
Paged KV cache
 ↓
FlashAttention kernel
 ↓
Output
```
支持：
```
continuous batching
dynamic memory
```
吞吐提升：
```
2×~4×
```
---
# 21. 总结
PagedAttention核心思想：
```
KV cache → paging
```
优势：
1️⃣ 消除 KV fragmentation  
2️⃣ 支持动态 batching  
3️⃣ GPU memory 利用率提升
结合 FlashAttention：
```
Kernel + Memory System
```
成为现代 LLM 推理系统关键技术。
