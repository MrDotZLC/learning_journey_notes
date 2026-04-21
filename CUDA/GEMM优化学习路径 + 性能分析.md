## 一、硬件数据
### 1.1 架构：Turing（TU116）
### 1.2 SM 数量：24
### 1.3 CUDA Cores：1536（= 24 × 64）
### 1.4 Boost Clock：约 1770 MHz（不同厂商略有浮动）
### 1.5 显存：GDDR6，192-bit
### 1.6 算力峰值：1536×2×1.77×109=5.44 TFLOPS
### 1.7 理论显存带宽：288 GB/s
### 1.8 Ridge Point（屋脊点）：18.9FLOP/Byte
$$I_{\text{ridge}} = \frac{\text{算力峰值}}{\text{带宽峰值}} = \frac{5.44 \times 10^{12}}{288 \times 10^9} \approx 18.9 \ \text{FLOP/Byte}$$
### 1.9 GEMM 的算术强度：
GEMM $C = A \times B$ ，矩阵尺寸$M \times N \times K$：

$$\text{浮点运算量} = 2MNK \ \text{FLOP}$$
$$\text{数据搬运量（理论最小）} = (MK + KN + MN) \times \text{sizeof(float)} \ \text{Bytes}$$

当 M = N = K时：

$$I = \frac{2M^3}{3M^2 \times 4} = \frac{M}{6}$$

|M=N=K|算术强度|瓶颈|
|---|---|---|
|64|10.7 FLOP/Byte|带宽瓶颈|
|128|21.3 FLOP/Byte|算力瓶颈|
|512|85.3 FLOP/Byte|算力瓶颈|
|1024|170.7 FLOP/Byte|算力瓶颈|
|4096|682.7 FLOP/Byte|算力瓶颈|

## 二、SGEMM
### 2.1 sgemm_v1_naive
Naive kernel 每次从 Global Memory 读数据，无复用。

实际访存量（M=N=K=1024）：

$$ \text{每个线程读 K 次 A、K 次 B} \Rightarrow \text{总读取} = M \times N \times 2K \times 4 \ \text{Bytes} $$

$$ = 1024 \times 1024 \times 2048 \times 4 \approx 8 \ \text{GB} $$

在 288 GB/s 下，**仅访存就需要约 27.8ms**，而算力上限对应的时间是：

$$ t_{\text{compute}} = \frac{2 \times 1024^3}{5.44 \times 10^{12}} \approx 0.39 \ \text{ms} $$

两者相差 70 倍——Naive kernel 的瓶颈完全在 Global Memory 访问，计算单元绝大多数时间在等数据。这是 Shared Memory Tiling（v2）存在的根本原因。

```
__global__ void sgemm_v1_naive_kernel(const float *A, const float *B, float *C, int M,
                               int N, int K, float alpha, float beta) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    // 沿 K 轴累加：每次迭代从 Global Memory 各读一个元素
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

// 16x16 block
void sgemm_v1_naive(const float *A, const float *B, float *C, int M, int N,
                    int K, float alpha, float beta) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    sgemm_v1_naive_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
}
```

---

### 2.2 sgemm_v1_shared_memory_tile
#### 2.2.1 方案分析

**Naive 访存：** tile 大小 $T \times T$ 为例，朴素矩阵乘会从全局内存 GM 中读同一元素 $2K$ 次。
**Shared Memory Tile 共享内存：** block中每个线程搬运 $A$ 和 $B$ 一些元素到共享内存，每个元素只从 GM 中读取 1 次，所有线程共用。假设 Block 有 $T^2$ 个线程，沿 $K$ 轴循环 $K/T$ 次，每次搬一对 Tile。

访存量变化：

$$ \text{Naive：} \quad \text{reads} = 2 \cdot M \cdot N \cdot K $$

$$ \text{Tiling：} \quad \text{reads} = 2 \cdot M \cdot N \cdot K \cdot \frac{1}{T} \cdot \underbrace{T}_{\text{tile内每元素被}T\text{个线程复用}} = \frac{2MNK}{T} $$

每个元素从 Global Memory 只读一次，但在 smem 里被复用 $T$ 次，Global Memory 访问量降低为原来的 $\dfrac{1}{T}$。
#### 2.2.2 代码

```CUDA
#include "gemm.h"

// 策略：block tile = 32x32，每线程计算 C 的一个元素
//       A/B tile 协作搬入 smem，每元素 Global Memory 只读一次
//       Global Memory 访问量从 O(MNK) 降至 O(MNK/T)

static constexpr int TILE = 32;

__global__ void sgemm_v2_smem_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    // 全局索引
    const int block_row = blockIdx.y * TILE;
    const int block_col = blockIdx.x * TILE;

    // block 内线程索引
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    // C 的元素索引
    const int row = block_row + ty;
    const int col = block_col + tx;
    
    __shared__ float smem_A[TILE][TILE];
    __shared__ float smem_B[TILE][TILE];

    float acc = 0.f;

    // 沿 K 轴分 tile 循环
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // 搬运 A Tile
        int a_row = block_row + ty;
        int a_col = t * TILE + tx;
        smem_A[ty][tx] = (a_row < M && a_col < K) 
                            ? A[a_row * K + a_col] 
                            : 0.f;

        // 搬运 B Tile
        int b_row = t * TILE + ty;
        int b_col = block_col + tx;
        smem_B[ty][tx] = (b_row < K && b_col < N) 
                            ? B[b_row * N + b_col]
                            : 0.f; 

        __syncthreads();

        // 计算 C Tile 的一个元素
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += smem_A[ty][k] * smem_B[k][tx];
        }

        // 防止部分线程先执行后续迭代，导致 smem 被覆盖
        __syncthreads();
    }

    // 写回 C
    if (row < M && col < N) {
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

void sgemm_v2_smem(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    dim3 block(TILE, TILE);   // 32x32 = 1024 线程/block
    dim3 grid((N + TILE - 1) / TILE,
              (M + TILE - 1) / TILE);
    sgemm_v2_smem_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK_LAST();
}
```

#### 2.2.3 数据分析

| M=N=K | v1 latency | v2 latency | v2/v1   |
| ----- | ---------- | ---------- | ------- |
| 256   | 0.570ms    | 0.837ms    | 1.47x 慢 |
| 512   | 4.181ms    | 6.098ms    | 1.46x 慢 |
| 1024  | 38.88ms    | 81.46ms    | 2.10x 慢 |
| 2048  | 499.7ms    | 646.3ms    | 1.29x 慢 |
| 4096  | 3988ms     | 5101ms     | 1.28x 慢 |

v2 在所有规模下均慢于 v1，且大矩阵（1024）最严重。**根本原因在于：vscode的 cmake tools 默认为 Debug 编译，CMakeLists.txt 不生效，导致耗时异常。**

正确耗时为：

| M=N=K | v1 latency (ms) | v2 latency (ms) | v2 / v1          |
| :---- | :-------------- | :-------------- | :--------------- |
| 256   | 0.1025          | 0.0656          | **0.64×（快 36%）** |
| 512   | 0.8095          | 0.4934          | **0.61×（快 39%）** |
| 1024  | 6.3936          | 3.7388          | **0.58×（快 42%）** |
| 2048  | 91.1972         | 49.7004         | **0.55×（快 45%）** |

小矩阵时，cuBLAS 的 kernel 选择有启动开销，手写 smem kernel 在这个规模反而占优。
大矩阵时，当前 v2 每个线程只计算 C 的 1 个元素，寄存器里只有 1 个累加器 `acc`，无法隐藏 smem 读取延迟。线程级别的计算效率成为瓶颈，抵消了 smem tiling 的效果。
