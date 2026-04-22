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

#### 2.2.4 性能瓶颈推导
##### 2.2.4.1 当前 v2 的参数
```
nvcc -gencode arch=compute_75,code=sm_75 \
     -Xptxas=-v -cubin -I./include \
     ./kernels/gemm/sgemm_v2_smem.cu
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z20sgemm_v2_smem_kernelPKfS0_Pfiiiff' for 'sm_75'
ptxas info    : Function properties for _Z20sgemm_v2_smem_kernelPKfS0_Pfiiiff
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 42 registers, used 1 barriers, 8192 bytes smem, 396 bytes cmem[0]
```
	block = (32, 32) = 1024 线程
	每线程计算 C 的 1×1 个元素
	寄存器/线程 = 42
	smem = 8192 bytes（两个 float[32][32]）
##### 2.2.4.2 SM75 硬件限制

|资源|上限|
|---|---|
|每 SM 最大线程数|1024|
|每 SM 最大 warp 数|32|
|每 SM 寄存器总量|65536|
|每 SM smem|65536 bytes|
|每 SM 最大 block 数|16|
|每 block 最大线程数|1024|

各资源对驻留 block 数的限制：

$$ \text{寄存器限制：} \left\lfloor \frac{65536}{1024 \times 42} \right\rfloor = \left\lfloor 1.52 \right\rfloor = 1\ \text{block} $$

$$ \text{smem 限制：} \left\lfloor \frac{65536}{8192} \right\rfloor = 8\ \text{blocks} $$

$$ \text{线程数限制：} \left\lfloor \frac{1024}{1024} \right\rfloor = 1\ \text{block} $$

$$ \text{实际驻留：} \min(1, 8, 1) = \boxed{1\ \text{block/SM}} $$

$$ \text{Occupancy} = \frac{1 \times 1024}{1024} = 100\% $$

Occupancy 是 100%，但**只有 1 个 block 在驻留**，意味着：

- 只有 32 个 warp 可调度
- 没有额外 warp 可以在访存 stall 时切换执行
- **延迟隐藏能力为零**

##### 2.2.4.3 计算访存比分析

v2 每线程的工作：

$$ \text{计算量} = 2K\ \text{FLOP}\ (\text{K次FMA}) $$

$$ \text{smem 读取量} = 2K \times 4\ \text{Bytes}\ (\text{每次迭代读smem\_A和smem\_B各一次}) $$

$$ I_{v2} = \frac{2K\ \text{FLOP}}{2K \times 4\ \text{Bytes}} = 0.25\ \text{FLOP/Byte} $$

smem 的带宽约为 **19 TB/s**（SM75 理论值），0.25 FLOP/Byte 对应需要的算力：

$$ \text{需要算力} = 0.25 \times 19 \times 10^{12} = 4.75\ \text{TFLOPS} $$

而 SM75 的 FP32 算力只有 5.44 TFLOPS，两者已经非常接近——**v2 的计算和 smem 访问几乎是 1:1 的串行关系，smem 读取无法被计算隐藏**。

##### 2.2.4.4 指令级并行度（ILP）分析

v2 内层循环：

```cuda
for (int k = 0; k < TILE; k++) {
    acc += smem_A[ty][k] * smem_B[k][tx];
}
```

只有**一个累加器** `acc`，每次迭代的 FMA 依赖上一次的结果：

$$\text{acc}^{(k)} = \text{acc}^{(k-1)} + A_k \times B_k$$

这是一条**依赖链**，FMA 的延迟在 SM75 上约为 4 个时钟周期，每次迭代必须等待上一次 FMA 完成才能开始，**ILP = 1，流水线利用率极低**。

##### 2.2.4.5 解决办法：每个线程多干点活Thread Coarsening

**寄存器和线程数**：

每线程计算 $TM \times TN$ 个输出元素（取 $TM = TN = 4$）：

$$ \text{线程数/block} = \frac{64}{4} \times \frac{64}{4} = 256 $$

$$ \text{寄存器限制驻留块数：} \left\lfloor \frac{65536}{256 \times r} \right\rfloor $$

只要 $r < 256$（几乎必然满足），驻留 block 数 $\geq 1$，且线程数减少后每 SM 可驻留更多 block。

**计算访存比**：

每线程每次从 smem 读 $TM + TN = 8$ 个元素，完成 $TM \times TN \times 2 = 32$ 次 FLOP：

$$ I_{v3} = \frac{TM \times TN \times 2}{(TM + TN) \times 4} = \frac{32}{32} = 1\ \text{FLOP/Byte} $$

相比 v2 的 0.25 FLOP/Byte 提升 **4 倍**。

**ILP：**

$TM×TN=16$ 个独立累加器，彼此之间无依赖，编译器可以交错调度 FMA 指令填满流水线：

$$\text{ILP} = TM \times TN = 16$$

### 2.3 sgemm_v3_coarsen
#### 2.3.1 方案分析

**设计参数：**

$$ \text{block tile} = 64 \times 64, \quad \text{线程数/block} = 16 \times 16 = 256 $$

$$ \text{每线程负责} = TM \times TN = 4 \times 4\ \text{个输出元素} $$

$$ \text{smem} = (64 \times 32 + 32 \times 64) \times 4 = 16384\ \text{Bytes} = 16\ \text{KB} $$

$$ \text{smem 限制驻留块数} = \left\lfloor \frac{65536}{16384} \right\rfloor = 4\ \text{blocks/SM} $$
**线程到输出元素的映射：**

block 内 256 个线程排成 $16 \times 16$，每线程负责 $4 \times 4$ 输出：

$$ \text{thread}(ty, tx) \rightarrow C[block_{row} + ty \times 4 : ty \times 4 + 4, \ block_{col} + tx \times 4 : tx \times 4 + 4] $$

**外积计算原理：**

每次 K 轴迭代，从 smem 读：

- $A$ 的一列切片：$\text{regA}[TM]$，对应当前线程负责的 $TM$ 行
- $B$ 的一行切片：$\text{regB}[TN]$，对应当前线程负责的 $TN$ 列

外积更新 $TM \times TN$ 个累加器：

$$ \text{acc}[m][n] += \text{regA}[m] \times \text{regB}[n], \quad m \in [0, TM),\ n \in [0, TN) $$

每次 smem 读取 $TM + TN = 8$ 个元素，完成 $TM \times TN = 16$ 次 FMA，**16 个累加器彼此独立，ILP = 16**。

#### 2.3.2 代码
```cuda
#include "gemm.h"

// 策略：block tile=64×64，每线程算4×4输出元素
//       寄存器缓存regA[4]/regB[4]，外积展开16个独立FMA

static constexpr int BM = 64;   // tile M
static constexpr int BN = 64;   // tile N
static constexpr int BK = 64;   // tile K
static constexpr int TM = 4;    // 每线程负责的 M 维度元素数量
static constexpr int TN = 4;    // 每线程负责的 N 维度元素数量

static_assert(BM / TM * BN / TN == 256, "block must have 256 threads");

__global__ void sgemm_v3_coarsen_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    // block 起始位置索引
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // block 内线程索引
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    // 当前线程负责的输出元素起始坐标
    const int thread_row = ty * TM;
    const int thread_col = tx * TN;

    const int tid = ty * blockIdx.x + tx; // 0 ~ 255

    // ---- Shared Memory ------------------------------------------
    // smem_A: [BK, BM] = [32, 64]  注意：转置存储，方便列方向读取
    // smem_B: [BK, BN] = [32, 64]
    // 转置smem_A的原因：
    //   计算时读 smem_A 的列（M方向），若按[BM,BK]存储，
    //   同warp内不同线程读同列不同行 → bank conflict
    //   转置为[BK,BM]后，读同一K位置的不同M元素 → 连续行访问 → 无conflict

    __shared__ float smem_A[BK][BM]; // 转置存储，列方向读取
    __shared__ float smem_B[BK][BN];

    // 寄存器累加器
    float acc[TM][TN] = {0.f};

    // 搬运 A：smem_A[BK][BM]，按行主序，每线程负责连续8个元素
    // 搬运 B：smem_B[BK][BN]，按行主序，每线程负责连续8个元素
    // 每线程搬运的起始索引（步长256，共搬8次）
    // 2048 / 256 = 8，用循环处理

    for (int k = 0; k < (K + BK - 1) / BK; k++) {
        // ---- 协作搬运 A tile → smem_A[BK][BM]（转置存入）--------
        // A tile 原始形状：[BM, BK] = A[block_row:block_row+BM, k*BK:k*BK+BK]
        // 转置后存入 smem_A[BK][BM]：smem_A[k'][m'] = A[block_row+m'][k*BK+k']
        for (int i = tid; i < BM * BK; i += 256) {
            int m = i % BM;
            int k_ = i / BM;
            int global_r = block_row + m;
            int global_c = k * BK + k_;
            smem_A[k_][m] = (global_r < M && global_c < K)
                            ? A[global_r * K + global_c]
                            : 0.f;
        }

        // ---- 协作搬运 B tile → smem_B[BK][BN]（正常存入）--------
        for (int i = tid; i < BK * BN; i+= 256) {
            int n = i % BN;
            int k_ = i / BN;
            int global_r = k * BK + k_;
            int global_c = block_col + n;
            smem_B[k_][n] = (global_r < K && global_c < N)
                            ? B[global_r * N + global_c]
                            : 0.f;
        }

        __syncthreads();

        // ---- 外积累加 -------------------------------------------
        // 对当前 K slice 的每个 k' 位置：
        // 从当前线程负责的输出元素起始坐标，连续读取 TM/TN 个元素，进行外积累加
        float reg_A[TM], reg_B[TN];

        #pragma unroll
        for (int k_ = 0; k_ < BK; k_++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                reg_A[m] = smem_A[k_][thread_row + m];
            }
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                reg_B[n] = smem_B[k_][thread_col + n];
            }

            // 外积更新 acc[TM][TN] += reg_A[TM] * reg_B[TN]
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    acc[m][n] += reg_A[m] * reg_B[n];
                }
            }
        }

        __syncthreads();
    }

    // ---- 写回全局内存 C --------------------------------------
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int global_r = block_row + thread_row + m;
            int global_c = block_col + thread_col + n;
            if (global_r < M && global_c < N) {
                C[global_r * N + global_c] =
                    alpha * acc[m][n] + beta * C[global_r * N + global_c];
            }
        }
    }
}

void sgemm_v3_coarsen(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    dim3 block(BN / TN, BM / TM);   // (16, 16) = 256 线程
    dim3 grid((N + BN - 1) / BN,
              (M + BM - 1) / BM);
    sgemm_v3_coarsen_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK_LAST();
}
```

### 2.3.3 数据分析
