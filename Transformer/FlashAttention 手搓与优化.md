[FlashAttention 详细介绍](FlashAttention%20详细介绍.md)
[FlashAttention 算子解析](FlashAttention%20算子解析.md)

## 一、FlashAttention 核心思想

将 $K$、$V$ 分块，对每个 Q tile 用 **online softmax** 流式处理所有 KV tile，避免存储 $S[N,N]$：

$$ \text{HBM 访问量} = O(N \cdot d) \quad \text{（线性，无 } N^2 \text{ 项）} $$

online softmax 递推公式（处理第 $t$ 个 KV tile 时）：

$$ m^{(t)} = \max\left(m^{(t-1)},\ \max_j s^{(t)}_j\right) $$

$$ l^{(t)} = e^{m^{(t-1)} - m^{(t)}} \cdot l^{(t-1)} + \sum_j e^{s^{(t)}_j - m^{(t)}} $$

$$ O^{(t)} = e^{m^{(t-1)} - m^{(t)}} \cdot O^{(t-1)} + \sum_j e^{s^{(t)}_j - m^{(t)}} \cdot V_j $$

最终 $O = O^{(T)} / l^{(T)}$。

## 二、算子实现
### 2.1 通用工具函数
```cuda
// warp reduce 工具（所有版本共用）
__device__ __forceinline__ float warp_reduce_max(float val) {
    constexpr unsigned FULL_MASK = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    constexpr unsigned FULL_MASK = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// 辅助函数：Block 级别的 Max 规约
__device__ __forceinline__ float block_reduce_max(float val, float* shared_max) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // 1. Warp 内部规约
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    if (lane == 0) shared_max[wid] = val;
    __syncthreads();

    // 2. 最后一个 Warp 对各 Warp 结果规约
    val = (threadIdx.x < blockDim.x / 32) ? shared_max[lane] : 0.f;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
    }
    return val;
}

// 辅助函数：Block 级别的 Sum 规约
__device__ __forceinline__ float block_reduce_sum(float val, float* shared_sum) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (lane == 0) shared_sum[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared_sum[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    return val;
}
```
### 2.2 flash_attn_v1_naive
#### 2.2.1 代码
```cuda
#include "attention.h"

// FlashAttention v1：标准 Attention，无 tiling
// 每个 block 处理一行 Q，完整计算 softmax(Q*K^T)*V
// 内存访问模式：O(N²)，所有 K/V 全部加载到 Global Memory

// 计算流程：
// 1. S[i][j] = Q[i] · K[j] / sqrt(d)   QK^T
// 2. P[i][j] = softmax(S[i])             softmax
// 3. O[i]    = P[i] · V                  PV

static constexpr int WARP_SIZE = 32;

// 每个 block 处理一行 Q（对应输出 O 的一行）
// blockDim.x = head_dim（假设 head_dim <= 1024）
__global__ void flash_attn_v1_naive_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half*       __restrict__ O,
    int seq_len,
    int head_dim)
{
    float scale = 1.f / sqrtf((float)head_dim);
    const int q_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (q_idx >= seq_len) {
        return;
    }

    // ---- smem：存储 S（attention scores）和 P（softmax 结果）----
    extern __shared__ float s_scores[]; // [seq_len]，QK^T 结果

    // ---- step 1：计算 S[q_idx][j] = Q[q_idx] · K[j] * scale ----
    const __half* q_row = Q + q_idx * head_dim; 

    for (int kv_idx = 0; kv_idx < seq_len; kv_idx++) {
        const __half* k_row = K + kv_idx * head_dim;

        float dot = 0.f;
        // 每个线程负责 head_dim 中的一部分
        for (int d = tid; d < head_dim; d += blockDim.x) {
            dot += __half2float(q_row[d]) * __half2float(k_row[d]);
        }

        // block reduce
        __shared__ float shared_sum[32];
        dot = block_reduce_sum(dot, shared_sum);

        // 写入 smem（只有 lane 0 写）
        if (tid == 0) {
            s_scores[kv_idx] = dot * scale;
        }
    }
    __syncthreads();

    // ---- step 2：softmax ----
    // 2a. 求最大值（数值稳定性）
    float max_val = -FLT_MAX;
    for (int j = tid; j < seq_len; j += blockDim.x) {
        max_val = fmaxf(max_val, s_scores[j]);
    }
    __shared__ float shared_max[32];
    max_val = block_reduce_max(max_val, shared_max);
    __syncthreads();

    // 2b. exp(s - max) 并求和
    float sum_exp = 0.f;
    for (int j = tid; j < seq_len; j += blockDim.x) {
        s_scores[j] = expf(s_scores[j]  - max_val);
        sum_exp += s_scores[j];
    }
    __shared__ float shared_sum[32];
    sum_exp = block_reduce_sum(sum_exp, shared_sum);
    __shared__ float inv_sum;
    if (tid == 0) {
        inv_sum = 1.f / (sum_exp + 1e-6f);
    }

    // 2c. 归一化
    for (int j = tid; j < seq_len; j += blockDim.x) {
        s_scores[j] *= inv_sum;
    }
    __syncthreads();

    // ---- step 3：O[q_idx] = P * V ----
    // 每个线程负责输出的若干维度
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float out = 0.f;
        for (int j = 0; j < seq_len; j++) {
            out += s_scores[j] * __half2float(V[j * head_dim + d]);
        }
        O[q_idx * head_dim + d] = __float2half(out);
    }
}   


void flash_attn_v1_naive(
    const __half* Q, const __half* K, const __half* V,
    __half* O, int seq_len, int head_dim)
{
    // 每 block 处理一行 Q
    // blockDim.x = min(head_dim, 128)，保证至少 4 warps
    const int threads = min(head_dim, 128);
    const int smem_bytes = seq_len * sizeof(float);  // s_scores

    flash_attn_v1_naive_kernel<<<seq_len, threads, smem_bytes>>>(
        Q, K, V, O, seq_len, head_dim);
    CUDA_CHECK_LAST();
}
```

### 2.3 flash_attn_v2_tiled
#### 2.3.1 方案设计

|维度|v1|v2|
|---|---|---|
|smem|`s_scores[seq_len]`，随 seq_len 增长|固定大小 KV tile|
|K/V 访问|全部加载，$O(N^2)$ HBM|分块加载，$O(N)$ HBM|
|softmax|两遍（先求 max，再 exp+sum）|online 单遍|
|seq_len 限制|smem 随 seq_len 增长，有上限|无限制|

```
grid = (ceil(seq_len / Br),)        每 block 处理 Br 行 Q
block = (Br * 32,)                  Br 个 warp，每 warp 处理一行 Q
```

每 warp 独立维护自己负责行的 online softmax 状态 `(m, l, o)`，warp 间无需通信。

```
smem_Q [Br][head_dim]   Q tile，block 开始时加载一次，KV 循环内复用
smem_K [Bc][head_dim]   每次 KV tile 循环更新
smem_V [Bc][head_dim]   每次 KV tile 循环更新
```

smem 大小（$Br = Bc = 32$，$d = 64$）：

$$ (Br + 2 \times Bc) \times d \times 2 = (32 + 64) \times 64 \times 2 = 12288\ \text{Bytes} = 12\ \text{KB} $$
每 warp 负责一行 Q 与一行 K 的点积，lane 间分摊 head_dim：

$$ \text{dot}_{ij} = \sum_{d=0}^{63} Q[i][d] \times K[j][d] $$

每个 lane 负责 `head_dim / 32 = 2` 个元素，warp reduce sum 得到完整结果。

#### 2.3.2 代码
```cuda
#include "attention.h"
#include <float.h>
#include <mma.h>
using namespace nvcuda;

// FlashAttention v2：online softmax + smem KV tiling
// 相比 v1：
//   1. K/V 分块加载，HBM 访问从 O(N²) 降为 O(N)
//   2. online softmax，单遍完成，无需存储完整 S 矩阵
//   3. smem 固定大小，不随 seq_len 增长
//
// 当前实现：
//   1 warp -> 1 Q row
//   warp 独立加载 Q/K/V 到 shared memory
//   为节省寄存器，每个线程只保存自己的 o ，m_new 和 l/o 分两遍计算。
//   用共享内存 smem_S 缓存点积，防止重复计算。

// tile 参数
static constexpr int Br = 32;    // Q tile 行数（每 block 处理 Br 行 Q）
static constexpr int Bc = 32;    // KV tile 列数

__global__ void flash_attn_v2_tiled_kernel(
    const __half* __restrict__ Q,   // [seq_len, head_dim]
    const __half* __restrict__ K,   // [seq_len, head_dim]
    const __half* __restrict__ V,   // [seq_len, head_dim]
    __half*       __restrict__ O,   // [seq_len, head_dim]
    int seq_len,
    int head_dim)
{
    // ---- smem ----
    extern __shared__ __half smem[];
    __half* smem_Q = smem;                          // [Br][head_dim]
    __half* smem_K = smem_Q + Br * head_dim;        // [Bc][head_dim]
    __half* smem_V = smem_K + Bc * head_dim;        // [Bc][head_dim]
    float*  smem_S = reinterpret_cast<float*>(smem_V + Bc * head_dim);  // [Br][Bc]，缓存点积

    const int lane_id       = threadIdx.x % 32;
    const int warp_id       = threadIdx.x / 32;

    const float scale = 1.f / sqrtf((float)head_dim);

    // 该 warp 负责的全局 Q 行
    const int q_row = blockIdx.x * Br + warp_id;

    // ---- 加载 Q tile 到 smem（block 内所有线程协作）----
    // smem_Q [Br][head_dim]，每线程加载若干 half
    {
        for (int col = lane_id; col < head_dim; col += 32) {
            smem_Q[warp_id * head_dim + col] = (q_row < seq_len)
                              ? Q[q_row * head_dim + col]
                              : __float2half(0.f);
        }
    }
    __syncthreads();

    // online softmax state
    // 存所有 o 会爆 register，m_new 和 l/o 分两遍计算，
    // 用共享内存缓存点积，防止重复计算。
    float m = -FLT_MAX;
    float l = 0.f;
    float o0 = 0.f;   // lane 负责的第 1 个维度：d = lane_id
    float o1 = 0.f;   // lane 负责的第 2 个维度：d = lane_id + 32

    // 输出寄存器
    // float o[64] = {};

    // block级：KV tile loop
    for (int kv_start = 0; kv_start < seq_len; kv_start += Bc) {
        // warp 独立加载 K/V
        // warp_id -> KV row
        // lane_id -> col
        {
            const int kv_row = kv_start + warp_id;
            for (int col = lane_id; col < head_dim; col += 32) {
                smem_K[warp_id * head_dim + col] =
                    (kv_row < seq_len)
                    ? K[kv_row * head_dim + col]
                    : __float2half(0.f);

                smem_V[warp_id * head_dim + col] =
                    (kv_row < seq_len)
                    ? V[kv_row * head_dim + col]
                    : __float2half(0.f);
            }
        }
        __syncthreads();

        // compute
        if (q_row < seq_len) {
            // 计算 QK^T 并存 S
            for (int j = 0; j < Bc; j++) {
                const int kv_row = kv_start + j;
                if (kv_row >= seq_len) break;
                float dot = 0.f;
#pragma unroll
                for (int d = lane_id; d < head_dim; d += 32) {
                    dot += __half2float(smem_Q[warp_id * head_dim + d]) *
                            __half2float(smem_K[j * head_dim + d]);
                }
                dot = warp_reduce_sum(dot);

                if (lane_id == 0) {
                    smem_S[warp_id * Bc + j] = dot * scale;
                }
            }
            __syncthreads();

            // online softmax
            // 从 smem_S 读取，求 m_new
            float m_new = m;
#pragma unroll
            for (int j = 0; j < Bc && kv_start + j < seq_len; j++) {
                m_new = fmaxf(m_new, smem_S[warp_id * Bc + j]);
            }

            // exp(x - m_new) = exp(x - m_old) * exp(m_old - m_new)
            //                  ↓
            // scale_old = exp(m_old - m_new)
            float scale_old = expf(m - m_new);
            // l_new = scale_old * l_old + Σ exp(S_ij - m_new)
            float l_new = scale_old * l;

            // o_new = scale_old * o_old + Σ exp(S_ij - m_new) * V_j
            // 先对 o 进行缩放
            o0 *= scale_old;
            o1 *= scale_old;

            // 完成 l_new 和 o_new 中的 Σ exp(x_j - m_new)
#pragma unroll
            for (int j = 0; j < Bc && kv_start + j < seq_len; j++) {
                float p_ij = expf(smem_S[warp_id * Bc + j] - m_new);
                l_new += p_ij;
                o0 += p_ij * __half2float(smem_V[j * head_dim + lane_id]);
                o1 += p_ij * __half2float(smem_V[j * head_dim + lane_id + 32]);
            }

            m = m_new;
            l = l_new;
        }
        __syncthreads();
    }

    // write back
    // O_i = o_new / l_new
    if (q_row < seq_len) {
        const float inv_l = 1.f / (l + 1e-6f);
        O[q_row * head_dim + lane_id]      = __float2half(o0 * inv_l);
        O[q_row * head_dim + lane_id + 32] = __float2half(o1 * inv_l);
    }
}

void flash_attn_v2_tiled(const __half *Q, const __half *K, const __half *V,
                         __half *O, int seq_len, int head_dim) {
    // 1 block -> Br rows
    const int grid = (seq_len + Br - 1) / Br;
    // Br warps
    const int block = Br * 32;

    // Q + K + V
    const int smem_bytes =
        (Br + 2 * Bc) * head_dim * sizeof(__half) + Br * Bc * sizeof(float);

    flash_attn_v2_tiled_kernel<<<grid, block, smem_bytes>>>(Q, K, V, O, seq_len,
                                                            head_dim);
    CUDA_CHECK_LAST();
}
```

#### 2.3.3 数据分析

|seq_len|v1 (ms)|v2 (ms)|加速比|
|---|---|---|---|
|128|0.0794|0.0944|0.84× 慢|
|256|0.1765|0.1833|0.96× 持平|
|512|0.6691|0.3441|1.95× 快|
|1024|2.3804|1.2101|1.97× 快|

**规律正确**：seq_len 越大，v2 优势越明显，符合 $O(N^2)$ vs $O(N)$ HBM 访问的理论预测。

小 seq_len（128/256）v2 反而慢，原因是：

- smem 加载和 `__syncthreads()` 的固定开销在小矩阵下占比高
- v1 对小矩阵的 smem 利用率更高（`s_scores[seq_len]` 全部在 smem 内）

```
v2 1024: 0.2218 TFLOPS
```

两遍点积计算（求 m_new 一遍，更新 l/o 一遍）导致实际计算量是理论值的 2×，TFLOPS 计算分母用的是理论 FLOPs，实际计算量翻倍，所以数值偏低。

若用 smem_S 缓存方案（避免重复计算），TFLOPS 应接近翻倍。

### 2.4 flash_attn_v3_mma
#### 2.4.1 方案设计

|特性|状态|
|---|---|
|Tensor Core MMA|支持|
|`ldmatrix`|支持|
|`mma.sync`|支持|
|`head_dim=64`|固定|
|online softmax|支持|
|shared memory tiling|支持|
|warp-level kernel|支持|

$$  
D_{16\times8} = 

A_{16\times16}  
\cdot  
B_{16\times8}  
+  
C_{16\times8}  
$$
1个 warp 完成两个 MMA： 16×8 + 16×8，即可得到：$16 \times 16$ 。

$QK^T$的维度=$[16\times64]  
\cdot  
[64\times16]$

Tensor Core 单次只能：

$$  
k=16  
$$

因此需要：

$$  
64 / 16 = 4  
$$

次 MMA。
