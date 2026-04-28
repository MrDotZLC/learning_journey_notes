## 一、硬件数据
1. 架构：Turing（TU116）
2. SM 数量：24
3. CUDA Cores：1536（= 24 × 64）
4. Boost Clock：约 1770 MHz（不同厂商略有浮动）
5. 显存：GDDR6，192-bit
6. 算力峰值：1536×2×1.77×109=5.44 TFLOPS
7. 理论显存带宽：288 GB/s
8. Ridge Point（屋脊点）：18.9FLOP/Byte
$$I_{\text{ridge}} = \frac{\text{算力峰值}}{\text{带宽峰值}} = \frac{5.44 \times 10^{12}}{288 \times 10^9} \approx 18.9 \ \text{FLOP/Byte}$$
9. GEMM 的算术强度：
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

## 二、CUDA Core SGEMM
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

**外积计算带来的问题**：

外积计算的循环中，smem_A 要按列读取，导致大量等待“气泡“，将数据存入转置后的 smem_A ，虽然写入时会有 4 路 bank conflict，但能够提升 4 倍读取效率。

#### 2.3.2 代码
```cuda
#include "gemm.h"

// 策略：block tile=64×64，每线程算4×4输出元素
//       寄存器缓存regA[4]/regB[4]，外积展开16个独立FMA

static constexpr int BM = 64;   // tile M
static constexpr int BN = 64;   // tile N
static constexpr int BK = 32;   // tile K
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

    const int tid = ty * blockDim.x + tx; // 0 ~ 255

    // ---- Shared Memory ------------------------------------------
    // smem_A: [BK, BM] = [32, 64]  注意：转置存储，方便列方向读取
    // smem_B: [BK, BN] = [32, 64]
    // 转置smem_A的原因：
    //   计算时读 smem_A 的列（M方向），若按[BM,BK]存储，
    //   同warp内不同线程读同列不同行 → 4 路 bank conflict
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

#### 2.3.3 数据分析
##### 2.3.3.1 v2 vs. v3

| M=N=K | v2 TFLOPS | v3 TFLOPS | 提升倍数  |
| ----- | --------- | --------- | ----- |
| 256   | 0.512     | 0.822     | 1.61x |
| 512   | 0.540     | 1.209     | 2.24x |
| 1024  | 0.574     | 1.300     | 2.26x |
| 2048  | 0.345     | 0.715     | 2.07x |

理论预测提升 4 倍，实际约 2 倍。差距原因：

smem 搬运阶段的开销没有减少。v3 的 BK=32，每次 K 迭代搬运：
$$ \text{smem\_A} = BM \times BK = 64 \times 32 = 2048\ \text{元素} $$ $$ \text{smem\_B} = BK \times BN = 32 \times 64 = 2048\ \text{元素} $$

256 个线程各搬 16 个元素，搬运本身占用了相当比例的执行时间，计算阶段的 ILP 提升被搬运开销稀释。

#### 2.3.4 瓶颈分析

v3 的搬运阶段是当前瓶颈，每线程用标量 `float` 搬运，每次 Global Memory 事务只读 4 bytes。 `float4` 一次读 16 bytes，在相同访存延迟下吞吐量提升 4 倍：

| 量                           | v3标量读 | v4 float4读 | 单位       |
| --------------------------- | ----- | ---------- | -------- |
| tile总数据量（smem_A + smem_B）   | 16384 | 16384      | bytes    |
| 每线程每次读取量                    | 4     | 16         | bytes/线程 |
| 每轮256线程搬运量                  | 1024  | 4096       | bytes/轮  |
| 完成tile搬运总轮次                 | 16    | 4          | 轮        |
| 每轮L2事务数（128bytes/事务）        | 8     | 32         | 次/轮      |
| 完成tile搬运总事务数                | 128   | 128        | 次        |
| 完成tile搬运总指令数（smem_A+smem_B） | 128   | 32         | 条        |

---

**结论**：

- 总事务数：相同（128次），带宽利用率不变
- 总指令数：减少 **4倍**（128→32条）

### 2.4 sgemm_v4_vec
#### 2.4.1 方案分析

v3 中 Tile A 的元素是逐个写入 SM，不用考虑写入方向，v4 是连续写入 4 个 float，减少 4 倍指令数，写入方向需要和读取方向保持一致，即沿着 K 轴方向。索引计算改为：

$$ k\_ = i \bmod BK, \quad m = i / BK $$

读取到寄存器进行外积计算时，暂不需要 Vectorized，因为 smem → 寄存器的读取**不经过L2事务**，瓶颈不在事务数和指令数，在**bank访问模式**。

#### 2.4.2 代码

```cuda
#include "gemm.h"

// 策略：在 v3 基础上，搬运阶段用 float4 替代标量 float
//       A tile：float4 读 Global Memory，分散写入转置 smem
//       B tile：float4 读 Global Memory，float4 写入 smem

static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr int BK = 32;
static constexpr int TM = 4;
static constexpr int TN = 4;

static_assert(BK % 4 == 0, "BK must be divisible by 4 for float4"); // 保证 Tile A 能沿 K 轴写入float4
static_assert(BN % 4 == 0, "BN must be divisible by 4 for float4"); // 保证 Tile B 能沿 N 轴写入float4

__global__ void sgemm_v4_vec_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta) 
{
    // block 对应的起始元素索引
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // block 内线程索引
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    // Tile 内线程负责计算的 C 起始元素索引 
    const int thread_row = ty * TM;
    const int thread_col = tx * TN;

    const int tid = ty * blockDim.x + tx;

    __shared__ float smem_A[BK][BM];
    __shared__ float smem_B[BK][BN];

    float acc[TM][TN] = {0.f};
    float reg_A[TM], reg_B[TN];

    for (int k = 0; k < (K + BK - 1) / BK; ++k) {
        // ---- 搬运 A tile：float4 读，分散写入转置 smem -----------
        // 遍历方向：M为外层，K为内层（与Global Memory连续方向一致）
        // 每线程处理4个元素（float4），步长=256线程*4个元素
        const int block_A_r = block_row;
        const int block_A_c = k * BK;
        for (int i = tid * 4; i < BK * BM; i += blockDim.x * blockDim.y * 4) {
            int k_ = i % BK;
            int m = i / BK;
            int global_r = block_A_r + m;
            int global_c = block_A_c + k_;
            float4 val = {0.f, 0.f, 0.f, 0.f};

            if (global_r < M && global_c + 3 < K) {
                val = *reinterpret_cast<const float4 *>(
                    &A[global_r * K + global_c]);
            } else {
                if (global_r < M) {
                    val.x = (global_c     < K) ? A[global_r * K + global_c    ] : 0.f;
                    val.y = (global_c + 1 < K) ? A[global_r * K + global_c + 1] : 0.f;
                    val.z = (global_c + 2 < K) ? A[global_r * K + global_c + 2] : 0.f;
                    val.w = (global_c + 3 < K) ? A[global_r * K + global_c + 3] : 0.f;
                }
            }

            // 转置写入 semem_A
            smem_A[k_    ][m] = val.x;
            smem_A[k_ + 1][m] = val.y;
            smem_A[k_ + 2][m] = val.z;
            smem_A[k_ + 3][m] = val.w;
        }

        // ---- 搬运 B tile：float4 读，float4 写入 smem -----------
        for (int i = tid * 4; i < BK * BN; i += blockDim.x * blockDim.y * 4) {
            int n = i % BN;
            int k_ = i / BN;
            int global_r = k * BK + k_;
            int global_c = block_col + n;
            float4 val = {0.f, 0.f, 0.f, 0.f};

            if (global_r < K && global_c + 3 < N) {
                val = *reinterpret_cast<const float4 *>(
                    &B[global_r * N + global_c]);
            } else {
                if (global_r < K) {
                    val.x = (global_c     < N) ? A[global_r * N + global_c    ] : 0.f;
                    val.y = (global_c + 1 < N) ? A[global_r * N + global_c + 1] : 0.f;
                    val.z = (global_c + 2 < N) ? A[global_r * N + global_c + 2] : 0.f;
                    val.w = (global_c + 3 < N) ? A[global_r * N + global_c + 3] : 0.f;
                }
            }

            *reinterpret_cast<float4*>(&smem_B[k_][n]) = val;
        }

        __syncthreads();

        // 计算外积
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

    // 写回 C
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int global_r = block_row + thread_row + m;
        int global_c = block_col + thread_col;

        if (global_r < M && global_c + 3 < N) {
            float4 out = {acc[m][0], acc[m][1], acc[m][2], acc[m][3]};
            if (beta != 0.f) {
                float4 old =
                    *reinterpret_cast<float4 *>(&C[global_r * N + global_c]);
                out.x = alpha * out.x + beta * old.x;
                out.y = alpha * out.y + beta * old.y;
                out.z = alpha * out.z + beta * old.z;
                out.w = alpha * out.w + beta * old.w;
            } else {
                out.x *= alpha;
                out.y *= alpha;
                out.z *= alpha;
                out.w *= alpha;
            }
            *reinterpret_cast<float4*>(&C[global_r * N + global_c]) = out;
        } else {
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                int col = global_c + n;
                if (global_r < M && col < N) {
                    C[global_r * N + col] =
                        alpha * acc[m][n] + beta * C[global_r * N + col];
                }
            }
        }
    }
}

void sgemm_v4_vec(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN,
              (M + BM - 1) / BM);
    sgemm_v4_vec_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK_LAST();
}
```

#### 2.4.3 数据分析

|M=N=K|v3 TFLOPS|v4 TFLOPS|提升倍数|
|---|---|---|---|
|256|0.661|0.882|1.33x|
|512|0.958|1.625|1.70x|
|1024|1.113|1.836|1.65x|
|2048|0.666|1.129|1.70x|

| M=N=K | v1   | v2   | v3   | v4       |
| ----- | ---- | ---- | ---- | -------- |
| 1024  | 0.33 | 0.57 | 1.11 | **1.84** |
| 2048  | 0.19 | 0.34 | 0.67 | **1.13** |

#### 2.4.4 瓶颈分析

##### 2.4.3.1 理论峰值与实际效率

v4 在 1024 规模达到 1.8914 TFLOPS，与理论峰值 5.44 TFLOPS 的比值：

$$ \eta = \frac{\text{实际 TFLOPS}}{\text{理论峰值 TFLOPS}} = \frac{1.8914}{5.44} = 34.8\% $$

距离峰值仍有 3 倍空间。

##### 2.4.4.2 Roofline 定位

v4 的算术强度（M=N=K=1024）：

$$ I = \frac{2MNK}{(MK + KN + MN) \times 4\ \text{bytes}} = \frac{2 \times 1024^3}{3 \times 1024^2 \times 4} = \frac{2 \times 1024}{12} \approx 170.7\ \text{FLOP/Byte} $$

Ridge Point：

$$ I^* = \frac{5.44 \times 10^{12}}{288 \times 10^9} = 18.9\ \text{FLOP/Byte} $$

$I = 170.7 \gg I^* = 18.9$，**v4 深度处于计算瓶颈区**，理论上带宽不是限制因素。

但实际只达到 34.8% 峰值，说明瓶颈在**计算单元内部**，而非访存。

##### 2.4.4.3 Occupancy 分析

每个 block 的资源占用：

$$ \text{寄存器} = 59 \times 256 = 15104 $$

$$ \text{smem} = 16384\ \text{bytes} $$

SM75 限制：

- 寄存器/SM：65536，可容纳 $\lfloor 65536 / 15104 \rfloor = 4$ 个 block
- smem/SM：65536 bytes，可容纳 $\lfloor 65536 / 16384 \rfloor = 4$ 个 block
- 最大 block/SM：16
- 最大线程/SM：1024，每 block 256 线程，4 block = 1024 线程 ✓

实际 Occupancy：

$$ \text{warp/SM} = 4\ \text{block} \times \frac{256}{32}\ \text{warp/block} = 32\ \text{warp} $$

$$ \text{Occupancy} = \frac{32}{32} = 100\% $$

Occupancy 满载，SM 有足够的 warp 切换来掩盖延迟。**Occupancy 不是瓶颈。** 但效率仅 34.8%，说明瓶颈在**计算单元内部的指令执行**。

##### 2.4.4.4 bank conflict 分析

###### Tile A 写入阶段
```
// 转置写入 semem_A
smem_A[k_    ][m] = val.x;
smem_A[k_ + 1][m] = val.y;
smem_A[k_ + 2][m] = val.z;
smem_A[k_ + 3][m] = val.w;
```

`smem_A` 的列宽 $BM = 64$。

$$\text{Bank}(k\_, m) = (k\_ \times 64 + m) \pmod{32} = m \pmod{32}$$
$$m = i / BK$$
- **核心变量**：由于 `BK = 32`，对于第一个 Warp ($tid \in [0, 31]$)，每个线程处理 `i = tid * 4`。
    - $T_0: i=0 \Rightarrow m=0, k\_=0$。写入 `smem_A[0,1,2,3][0]`。
    - $T_1: i=4 \Rightarrow m=0, k\_=4$。写入 `smem_A[4,5,6,7][0]`。
    - ...
    - $T_7: i=28 \Rightarrow m=0, k\_=28$。写入 `smem_A[28,29,30,31][0]`。
    - $T_8: i=32 \Rightarrow m=1, k\_=0$。写入 `smem_A[0,1,2,3][1]`。
-  **计算结果**：$T_0$ 到 $T_7$ 这 8 个线程的 $m$ 全部等于 0。因此，这 8 个线程在执行 `smem_A[k_][m] = val.x` 时，全部指向 `Bank 0`。
- **结论**：**8 路 Bank Conflict**。

###### 计算阶段加载 `reg_A` 冲突分析

```
reg_A[m] = smem_A[k_][thread_row + m];
```

- **线程分布**：Warp 内线程的 `thread_row` 取决于 `ty`。
- **参数**：`blockDim.x = 16` (BN/TN)，意味着 Warp 前 16 个线程的 `ty = 0`，后 16 个线程的 `ty = 1`。
- **地址计算**：
    - $T_0 \dots T_{15}$: `thread_row = 0 * 4 = 0`。访问 `smem_A[k_][0+m]`。
    - $T_{16} \dots T_{31}$: `thread_row = 1 * 4 = 4`。访问 `smem_A[k_][4+m]`。
- **Bank 映射**：
    $$ \text{Bank} = (\text{Base} + \text{thread\_row} + m) \pmod{32}$$
    
	- 前 16 个线程全部访问同一个地址（同一 Bank 的同一位置），触发 **Broadcast（广播）** 机制，无冲突。
	- 后 16 个线程全部访问另一个相同地址，触发 **Broadcast**，无冲突。
- **结论**：**无冲突**。现代架构（Maxwell 之后）对于 Warp 内不同线程访问同一地址具有极好的广播支持。

###### 计算阶段加载 `reg_B` 冲突分析

```
reg_B[n] = smem_B[k_][thread_col + n];
```

- **线程分布**：`thread_col = tx * 4`。在一个 Warp 中，前 16 个线程的 `tx` 为 $0 \dots 15$，后 16 个线程的 `tx` 也是 $0 \dots 15$（假设 `blockDim.x = 16`）。
- **Bank 映射**：
    
    $$ \text{Bank}(tx) = (k\_ \times 64 + tx \times 4 + n) \pmod{32}$$
    
    - $T_0 (tx=0) \Rightarrow \text{Bank } 0$
    - $T_1 (tx=1) \Rightarrow \text{Bank } 4$
    - $T_2 (tx=2) \Rightarrow \text{Bank } 8$
    - ...
    - $T_7 (tx=7) \Rightarrow \text{Bank } 28$
    - $T_8 (tx=8) \Rightarrow \text{Bank } (32) \pmod{32} = 0$
- **冲突判定**：在同一个 Warp 的前 16 个线程中，$T_0$ 和 $T_8$ 都访问了 Bank 0。但注意，它们访问的是 **Bank 0 的不同地址**（$T_0$ 访问偏移 0，$T_8$ 访问偏移 32）。
- **结论**：**2 路 Bank Conflict**。由于 16 个线程就覆盖了一次 32 个 Banks 的循环（步长为 4），整个 Warp（32 线程）会请求同一个 Bank 4 次，但在每个半 Warp（Half-Warp）执行周期内是 2 路冲突。

###### 总结

|**阶段**|**涉及变量**|**冲突情况**|**原因简述**|
|---|---|---|---|
|**Tile A 写入**|`smem_A`|**8 路冲突**|8 个线程拥有相同的 $m$，映射到同一 Bank|
|**Tile B 写入**|`smem_B`|**无冲突**|`float4` 对齐写入且 $BN$ 是 32 倍数|
|**Tile A 读取**|`reg_A`|**无冲突**|同 Warp 内线程访问相同地址，触发 Broadcast|
|**Tile B 读取**|`reg_B`|**2 路冲突**|线程间步长为 4 个 float，32 线程循环命中 Bank|

###### 解决方案
- padding：
	- 将 `smem_A` 声明为 `smem_A[BK][BM + 1]`
	- 将 `smem_B` 声明为 `smem_A[BN][BK + 1]`（不推荐，需修改较多逻辑）
- swizzle
  对第 $r$ 行施加列偏移 $\delta(r) = r \times 4$，用 XOR 代替加法（无进位，单周期）。写入 smem_A 时 $k\_$ 步长为 4，令 $r = k_ / 4$，偏移即为 $k\_$ 本身。为防止 XOR 影响列索引 bit[5]（控制 0/32 段的选择，影响 smem_B 等其他访问模式），屏蔽 $k\_$ 的无关位，只保留 bit[4:2]：
$$\text{col\_swz} = m \oplus \left(k\_ \mathbin{\&} \text{0x1C}\right)$$
读取时施加相同变换即可无损还原逻辑索引。

##### 2.4.4.4 计算阶段的指令级分析

v4 计算外积的核心循环：

```cpp
for (int k_ = 0; k_ < BK; k_++) {       // 32次
    for (int m = 0; m < TM; m++)         // 4次：读 smem_A → reg_A
        reg_A[m] = smem_A[k_][...];
    for (int n = 0; n < TN; n++)         // 4次：读 smem_B → reg_B
        reg_B[n] = smem_B[k_][...];
    for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++)     // 16次 FMA
            acc[m][n] += reg_A[m] * reg_B[n];
}
```

每次 k_ 迭代的指令构成：

|指令类型|数量|延迟（SM75）|
|---|---|---|
|smem load（reg_A）|4|约 20 cycle|
|smem load（reg_B）|4|约 20 cycle|
|FFMA|16|4 cycle（throughput）|

smem load 到 FFMA 的依赖链：

```
reg_A[m] = smem_A[...]     ← 20 cycle latency
                    ↓
acc[m][n] += reg_A[m] * reg_B[n]   ← 必须等 reg_A 就绪
```

隐藏 smem load 延迟需要足够的独立指令填充。每次 k_ 迭代只有 16 个 FFMA 可用于掩盖 4 次 smem load 的延迟，**ILP 不足以完全掩盖 smem 读取延迟**。
搬运与计算**完全串行**，两段时间均存在资源空闲。这是比指令级 ILP 更上层的结构性浪费。

##### 2.4.4.6 方案分析

由分析可知，现有bank conflict 和 smem load 延迟问题，有如下 3 个方案：
- **swizzle** 解决 bank conflict
- **padding** 解决 bank conflict 
- **double buffering** 解决 smem load 延迟

### 2.5 sgemm_v5_swizzle
#### 2.5.1 方案分析

消除 v4 smem_A 的下入bank conflict，将不同行的同一逻辑列 m 映射到不同 bank，即对每一行施加一个不同的列偏移。
$$ \text{col\_swz} = m + \delta(r) \pmod{64} $$
加法的 bank 等价于 $(m + \delta) \bmod 32$，需要 8 行的 $\delta$ 各不相同且步长均匀覆盖 ${0,4,8,...,28}$，硬件上 XOR 无进位，计算更合适，即：
$$ \text{col\_swz} = m \oplus k\_ $$
但 $k\_$ 最大为 28（bit[4:0]），而列索引 $m \in [0,63]$ 占 6 bit，XOR 的有效范围需对齐。$k\_$ 的 bit[1:0] 恒为 0（4 的倍数），bit[4:2] 才是有效变化位，对应列索引的 bit[4:2]：

$$ \text{col\_swz} = m \oplus (k\_ \: \mathbin{\&} \underbrace{0\text{x}1\text{C}}_{0001\ 1100_2}) $$

`& 0x1C` 的作用是**屏蔽 $k\_$ 的 bit[1:0] 和 bit[7:5]**，只保留 bit[4:2]，防止 XOR 影响列索引的高位（bit[5]，对应 0/32 的选择），避免引入新的 conflict。

**验证 8 行互不碰撞**：

$k_ \in {0,4,8,12,16,20,24,28}$，mask = $k_ \mathbin{\&} 0\text{x}1\text{C}$ = ${0,4,8,12,16,20,24,28}$，$m=0$ 时 col_swz = mask，bank = mask，8 个不同值，✓。

**验证同一行内不引入新 conflict**：

固定 $k\_$，mask 为常数，col_swz = $m \oplus \text{const}$，相当于对列索引做常数 XOR，不改变同一行内不同 $m$ 之间的 bank 间距，原本无 conflict 的访问模式保持不变，✓。

#### 2.5.2 代码
```cuda
#include "gemm.h"

// v5_swizzle：在 v4 基础上，对 smem_A 写入/读取施加 swizzle
//   消除 smem_A 写入的 8-way bank conflict
//   smem_B 的 2-way conflict 在不转置条件下不可消除，保持原样
//
// Swizzle 公式：col_swz = m ^ (k_ & 0x1C)
//   作用：让不同 k_ 行的相同逻辑列映射到不同 bank
//   & 0x1C：只保留 k_ 的 bit[4:2]，防止影响列索引 bit[5]

static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr int BK = 32;
static constexpr int TM = 4;
static constexpr int TN = 4;

static_assert(BK % 4 == 0, "BK must be divisible by 4 for float4");
static_assert(BN % 4 == 0, "BN must be divisible by 4 for float4");

__global__ void sgemm_v5_swizzle_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int thread_row = ty * TM;
    const int thread_col = tx * TN;

    const int tid = ty * blockDim.x + tx;

    __shared__ float smem_A[BK][BM];
    __shared__ float smem_B[BK][BN];

    float acc[TM][TN] = {0.f};
    float reg_A[TM], reg_B[TN];

    for (int k = 0; k < (K + BK - 1) / BK; k++) {

        const int block_A_r = block_row;
        const int block_A_c = k * BK;
        for (int i = tid * 4; i < BK * BM; i += blockDim.x * blockDim.y * 4) {
            int k_ = i % BK;
            int m  = i / BK;
            int global_r = block_A_r + m;
            int global_c = block_A_c + k_;

            float4 val = {0.f, 0.f, 0.f, 0.f};
            if (global_r < M && global_c + 3 < K) {
                val = *reinterpret_cast<const float4 *>(
                    &A[global_r * K + global_c]);
            } else if (global_r < M) {
                val.x = (global_c     < K) ? A[global_r * K + global_c    ] : 0.f;
                val.y = (global_c + 1 < K) ? A[global_r * K + global_c + 1] : 0.f;
                val.z = (global_c + 2 < K) ? A[global_r * K + global_c + 2] : 0.f;
                val.w = (global_c + 3 < K) ? A[global_r * K + global_c + 3] : 0.f;
            }

            // swizzle
            smem_A[k_    ][m ^ ((k_    ) & 0x1C)] = val.x;
            smem_A[k_ + 1][m ^ ((k_ + 1) & 0x1C)] = val.y;
            smem_A[k_ + 2][m ^ ((k_ + 2) & 0x1C)] = val.z;
            smem_A[k_ + 3][m ^ ((k_ + 3) & 0x1C)] = val.w;
        }

        const int block_B_r = k * BK;
        const int block_B_c = block_col;
        for (int i = tid * 4; i < BK * BN; i += blockDim.x * blockDim.y * 4) {
            int n  = i % BN;
            int k_ = i / BN;
            int global_r = block_B_r + k_;
            int global_c = block_B_c + n;

            float4 val = {0.f, 0.f, 0.f, 0.f};
            if (global_r < K && global_c + 3 < N) {
                val = *reinterpret_cast<const float4 *>(
                    &B[global_r * N + global_c]);
            } else if (global_r < K) {
                val.x = (global_c     < N) ? B[global_r * N + global_c    ] : 0.f;
                val.y = (global_c + 1 < N) ? B[global_r * N + global_c + 1] : 0.f;
                val.z = (global_c + 2 < N) ? B[global_r * N + global_c + 2] : 0.f;
                val.w = (global_c + 3 < N) ? B[global_r * N + global_c + 3] : 0.f;
            }
            *reinterpret_cast<float4 *>(&smem_B[k_][n]) = val;
        }

        __syncthreads();

        // 计算外积
        #pragma unroll
        for (int k_ = 0; k_ < BK; k_++) {
            const int swz = k_ & 0x1C;

            #pragma unroll
            for (int m = 0; m < TM; m++) {
                reg_A[m] = smem_A[k_][(thread_row + m) ^ swz];
            }
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                reg_B[n] = smem_B[k_][thread_col + n];
            }

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

    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int global_r = block_row + thread_row + m;
        int global_c = block_col + thread_col;

        if (global_r < M && global_c + 3 < N) {
            float4 out = {acc[m][0], acc[m][1], acc[m][2], acc[m][3]};
            if (beta != 0.f) {
                float4 old = *reinterpret_cast<float4 *>(
                    &C[global_r * N + global_c]);
                out.x = alpha * out.x + beta * old.x;
                out.y = alpha * out.y + beta * old.y;
                out.z = alpha * out.z + beta * old.z;
                out.w = alpha * out.w + beta * old.w;
            } else {
                out.x *= alpha;
                out.y *= alpha;
                out.z *= alpha;
                out.w *= alpha;
            }
            *reinterpret_cast<float4 *>(&C[global_r * N + global_c]) = out;
        } else {
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                int col = global_c + n;
                if (global_r < M && col < N) {
                    C[global_r * N + col] =
                        alpha * acc[m][n] + beta * C[global_r * N + col];
                }
            }
        }
    }
}

void sgemm_v5_swizzle(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    dim3 block(BN / TN, BM / TM);   // (16, 16)
    dim3 grid((N + BN - 1) / BN, 
              (M + BM - 1) / BM);
    sgemm_v5_swizzle_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK_LAST();
}   
```

#### 2.5.3 数据分析

|M=N=K|v4 TFLOPS|v5 TFLOPS|v5 vs v4|
|---|---|---|---|
|256|1.0661|0.9384|**-12.0%**|
|512|1.5767|1.6023|+1.6%|
|1024|1.8914|1.8636|-1.5%|
|2048|1.1081|1.1161|+0.7%|

swizzle 的收益在误差范围内（±2%），**对整体性能无显著影响**。

#### 2.5.4 优化无效的原因分析

**smem_A 写入的 8-way conflict 本身权重低。** 写入 smem_A 发生在搬运阶段，每个 tile 迭代只执行一次，而计算阶段的 smem 读取执行 BK=32 次。搬运占总执行时间的比例本身就小，消除其 conflict 的绝对收益有限。

**SM75 的 warp 调度掩盖了部分 conflict 延迟。** 8-way conflict 理论上需要 8 轮串行内存事务，但 SM 内其他 warp 的指令可以填充这段等待，实际暴露的延迟低于理论值。

**256 规模的回退（-12%）** 是小矩阵下 L2/L1 cache 命中率高，smem 路径本身占比下降，swizzle 引入的额外 XOR 指令和索引计算反而成为相对开销。

### 2.6 sgemm_v5_padding

#### 2.6.1 方案分析

根据前面的分析：

- **smem_A 写入 8-way conflict**：跨行访问，行宽 64 是 32 的整数倍，padding +1 有效
- **smem_B 读取 2-way conflict**：同行内 tx 与 tx+8 碰撞，padding 对同行内冲突无效（§3.3 已证）

所以 padding 方案只处理 smem_A，smem_B 保持原样。

---

**方案**：`smem_A[BK][BM + 1]` = `smem_A[32][65]`

验证 padding +1 后 tid=0..7 的 bank：

$$ \text{addr}(k_, m) = k_ \times 65 + m $$

| tid | $k\_$ | $m$ | addr | bank |
| --- | ----- | --- | ---- | ---- |
| 0   | 0     | 0   | 0    | 0    |
| 1   | 4     | 0   | 260  | 4    |
| 2   | 8     | 0   | 520  | 8    |
| 3   | 12    | 0   | 780  | 12   |
| 4   | 16    | 0   | 1040 | 16   |
| 5   | 20    | 0   | 1300 | 20   |
| 6   | 24    | 0   | 1540 | 24   |
| 7   | 28    | 0   | 1820 | 28   |

8 个线程分散到 8 个不同 bank，无 conflict。smem 增量：

$$ 32 \times 1 \times 4 = 128\ \text{bytes},\quad \text{total} = 16384 + 128 = 16512\ \text{bytes} < 65536\ \text{bytes} $$

#### 2.6.2 代码

只改一行
```cuda
__shared__ float smem_A[BK][BM + 1]; // padding 1 行避免 bank conflict
__shared__ float smem_B[BK][BN];
```

#### 2.6.3 数据分析

|M=N=K|v4 TFLOPS|padding TFLOPS|delta|
|---|---|---|---|
|256|1.0625|0.9844|**-7.4%**|
|512|1.5733|1.5835|+0.6%|
|1024|1.8920|1.8533|-2.0%|
|2048|1.1428|1.1349|-0.7%|

两个现象需要解释：

1. **512 及以上**：与 v4 差异在 ±2% 以内，在测量噪声范围内，**无显著收益**
2. **256 规模**：明确退步 -7.4%

#### 2.6.4 优化无效的原因分析

256 规模下 grid 只有 $4 \times 4 = 16$ 个 block，SM75 共 24 个 SM，**大量 SM 空闲，每个活跃 SM 驻留的 warp 数远低于 32**，warp 切换掩盖延迟的能力大幅下降，conflict 的实际停顿开始暴露。

与此同时 padding 引入了新的开销：smem_A 行宽从 64 变为 65，**不再是 2 的幂次**，编译器无法用移位替代乘法计算行地址，地址计算指令数增加。这个额外开销在大规模下被 FFMA 计算掩盖，在 256 规模下无法被掩盖，导致明确退步。

padding 消除了 smem_A 写入的 8-way conflict，但 conflict 本身不在关键路径上，消除它的收益接近零，而引入的非 2 的幂次行宽开销在小规模下反而成为负担。

### 2.6 sgemm_v5_double_buf
#### 2.6.1 方案分析

v4 每个 tile 迭代的执行时间线：

```
[搬运 Global→smem] → [sync] → [计算 smem→reg→FFMA] → [sync]
      ↑ CUDA core 空闲                ↑ Global Memory 带宽空闲
```

用两块 smem 交替，将 tile_{k+1} 的 Global Memory 读取与 tile_k 的计算重叠：

```
预搬运 tile_0 → smem[0]
k=0: ldg tile_1→寄存器  ||  计算 smem[0]  →  sync  →  寄存器→smem[1]  →  sync
k=1: ldg tile_2→寄存器  ||  计算 smem[1]  →  sync  →  寄存器→smem[0]  →  sync
...
```

SM75 无 `cp.async`，用**寄存器 prefetch** 模拟：`ldg` 发射后不阻塞后续指令，在计算阶段执行期间数据从 Global Memory 到达寄存器，计算结束后再 `sts` 写入 smem。

**`ldg` 的作用**：
1. **只读缓存访问**
    - `ldg` 是 Load Global (read-only cache) 指令，数据会先尝试载入 **L1 只读缓存** 或 **L2**。
    - 对于重复访问的只读数据（如矩阵 A/B 的元素），可减少 L1/Shared Memory 的占用。
2. **寄存器直接载入**
    - 和普通 `global memory load` 类似，`ldg` 会将数据加载到寄存器，但**不会污染 Shared Memory 或 L1 写回缓存**。
3. **潜在的延迟隐藏**
    - 在某些 GPU 架构上（Kepler 及之后），`ldg` 可以和计算部分重叠，但这并不是严格的异步，只是对缓存访问延迟的优化。

smem 翻倍后每 SM 可驻留 block 数：

$$ \left\lfloor \frac{65536}{32768} \right\rfloor = 2\ \text{block},\quad \text{Occupancy} = \frac{2 \times 8}{32} = 50\% $$

warp 调度掩盖延迟的能力减半。**净收益取决于 overlap 节省的时间是否覆盖 Occupancy 损失，需实测判断。**

#### 2.6.2 代码

```cuda
#include "gemm.h"

static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr int BK = 32;
static constexpr int TM = 4;
static constexpr int TN = 4;

static_assert(BK % 4 == 0);
static_assert(BN % 4 == 0);

// ---------------------------------------------------------------
// 改动 1：用 PTX ld.global.ca 显式加载，无分支，编译器可乱序调度
// __ldg 走 texture cache 路径（只读），SM75 上等价于
// ld.global.nc.v4.f32，warp scheduler 可在数据未就绪时切换
// ---------------------------------------------------------------
__device__ __forceinline__ float4 ldg128_safe(
    const float* ptr, int r, int c, int rows, int cols)
{
    float4 val = {0.f, 0.f, 0.f, 0.f};
    if (r >= rows) return val;
    if (c + 3 < cols) {
        val = __ldg(reinterpret_cast<const float4*>(&ptr[r * cols + c]));
    } else {
        val.x = (c     < cols) ? ptr[r * cols + c    ] : 0.f;
        val.y = (c + 1 < cols) ? ptr[r * cols + c + 1] : 0.f;
        val.z = (c + 2 < cols) ? ptr[r * cols + c + 2] : 0.f;
        val.w = (c + 3 < cols) ? ptr[r * cols + c + 3] : 0.f;
    }
    return val;
}

// ---------------------------------------------------------------
// 改动 2：__launch_bounds__ 限制每线程寄存器数
// 256 线程/block，目标 occupancy = 2 blocks/SM（SM75 共 64KB smem）
// double buf 占 32KB，2 blocks × 32KB = 64KB，恰好满
// 限制寄存器 ≤ 64，保证 2 blocks 并发
// ---------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void sgemm_v5_double_buf_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int thread_row = ty * TM;
    const int thread_col = tx * TN;

    // double buffer smem
    __shared__ float smem_A[2][BK][BM];
    __shared__ float smem_B[2][BK][BN];
    

    float acc[TM][TN] = {};
    float reg_A[TM], reg_B[TN];
    // 故意填充非零垃圾值
    #pragma unroll
    for (int i = 0; i < TM; i++) reg_A[i] = 99999.f;
    #pragma unroll
    for (int i = 0; i < TN; i++) reg_B[i] = 99999.f;

    // prefetch 寄存器：每线程 2 次 float4 × A/B
    // BK*BM / (256*4) = 32*64/1024 = 2
    float4 pA[2], pB[2];

    const int num_tiles  = (K + BK - 1) / BK;

    // ---- 预加载第 0 个 tile 到 smem[0] ----
    #pragma unroll
    for (int i = 0, idx = tid * 4; i < 2;
            i++, idx += 256 * 4) {
        int k_ = idx % BK, m = idx / BK;
        float4 v = ldg128_safe(A, block_row + m, 0 * BK + k_, M, K);
        smem_A[0][k_  ][m] = v.x;
        smem_A[0][k_+1][m] = v.y;
        smem_A[0][k_+2][m] = v.z;
        smem_A[0][k_+3][m] = v.w;
    }
    #pragma unroll
    for (int i = 0, idx = tid * 4; i < 2;
            i++, idx += 256 * 4) {
        int n = idx % BN, k_ = idx / BN;
        float4 v = ldg128_safe(B, 0 * BK + k_, block_col + n, K, N);
        *reinterpret_cast<float4*>(&smem_B[0][k_][n]) = v;
    }
    __syncthreads();

    // 主循环
    #pragma unroll
    for (int k = 0; k < num_tiles; k++) {
        const int cur     = k & 1;
        const int next    = cur ^ 1;
        const bool has_next = (k + 1 < num_tiles);

        // ---- 发射 ldg，prefetch tile_{k+1} 到寄存器 ----
        if (has_next) {
            #pragma unroll
            for (int i = 0, idx = tid * 4; i < 2;
                    i++, idx += 256 * 4) {
                int k_ = idx % BK, m = idx / BK;
                pA[i] = ldg128_safe(
                    A, block_row + m, (k+1) * BK + k_, M, K);
            }
            #pragma unroll
            for (int i = 0, idx = tid * 4; i < 2;
                    i++, idx += 256 * 4) {
                int n = idx % BN, k_ = idx / BN;
                pB[i] = ldg128_safe(
                    B, (k+1) * BK + k_, block_col + n, K, N);
            }
        }

        // ---- FMA（掩盖上方 ldg 的 latency）----
        #pragma unroll
        for (int k_ = 0; k_ < BK; k_++) {
            #pragma unroll
            for (int m = 0; m < TM; m++)
                reg_A[m] = smem_A[cur][k_][thread_row + m];
            #pragma unroll
            for (int n = 0; n < TN; n++)
                reg_B[n] = smem_B[cur][k_][thread_col + n];
            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int n = 0; n < TN; n++)
                    acc[m][n] += reg_A[m] * reg_B[n];
        }

        // ---- sync + sts：此时 ldg 数据已就绪 ----
        if (has_next) {
            __syncthreads();
            #pragma unroll
            for (int i = 0, idx = tid * 4; i < 2;
                 i++, idx += 256 * 4) {
                int k_ = idx % BK, m = idx / BK;
                smem_A[next][k_  ][m] = pA[i].x;
                smem_A[next][k_+1][m] = pA[i].y;
                smem_A[next][k_+2][m] = pA[i].z;
                smem_A[next][k_+3][m] = pA[i].w;
            }
            #pragma unroll
            for (int i = 0, idx = tid * 4; i < 2;
                 i++, idx += 256 * 4) {
                int n = idx % BN, k_ = idx / BN;
                *reinterpret_cast<float4*>(&smem_B[next][k_][n]) = pB[i];
            }
            __syncthreads();
        }
    }

    // ---- 写回 C ----
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        const int gr = block_row + thread_row + m;
        const int gc = block_col + thread_col;
        if (gr >= M) continue;
        // 前 4 列
        if (gc + 3 < N) {
            float4 out = {acc[m][0], acc[m][1], acc[m][2], acc[m][3]};
            if (beta != 0.f) {
                float4 old =
                    __ldg(reinterpret_cast<const float4 *>(&C[gr * N + gc]));
                out.x = alpha*out.x + beta*old.x;
                out.y = alpha*out.y + beta*old.y;
                out.z = alpha*out.z + beta*old.z;
                out.w = alpha*out.w + beta*old.w;
            } else {
                out.x *= alpha; out.y *= alpha;
                out.z *= alpha; out.w *= alpha;
            }
            *reinterpret_cast<float4*>(&C[gr * N + gc]) = out;
        } else {
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                int col = gc + n;
                if (col < N)
                    C[gr * N + col] =
                        alpha * acc[m][n] + beta * C[gr * N + col];
            }
        }
    }
}

void sgemm_v5_double_buf(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    // 改动 5：显式配置 smem 上限，保证 32KB 分配成功
    cudaFuncSetAttribute(
        sgemm_v5_double_buf_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);

    dim3 block(BN / TN, BM / TM);  // 16×16 = 256
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_v5_double_buf_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK_LAST();
}
```

#### 2.6.3 数据分析

|M=N=K|v4_vec (TFLOPS)|v5_double_buf (TFLOPS)|增量|vs 峰值 (5.44T)|
|---|---|---|---|---|
|256|1.0638|1.0670|+0.3%|19.6%|
|512|1.6309|1.6061|-1.5%|29.5%|
|1024|1.8906|1.9253|+1.8%|35.4%|
|2048|1.1422|1.1466|+0.4%|21.1%|

Double Buffering 的理论模型假设 Global Memory Load 延迟可被 FMA 计算完全掩盖。实际有效的条件是：

$$ T_{\text{compute}}(\text{tile}_k) \geq T_{\text{ldg}}(\text{tile}_{k+1}) $$

对 v5 的参数 $BM=BN=64,\ BK=32,\ TM=TN=4$，计算量与搬运量：

$$ \text{FMA per tile} = BK \times TM \times TN = 32 \times 4 \times 4 = 512 \text{ FMA/thread} $$

$$ \text{LDG per tile (A+B)} = \frac{BK \times BM}{256 \times 4} \times 2 + \frac{BK \times BN}{256 \times 4} \times 2 = 2 + 2 = 4 \text{ float4/thread} $$

每次 float4 LDG 在 SM75 上的延迟约 200~400 cycle（L2 miss），而 512 FMA 在 peak 情况下需约：

$$ \frac{512 \text{ FMA}}{2 \text{ FMA/cycle}} = 256 \text{ cycles} $$

**计算窗口（256 cycles）< LDG 延迟（200–400 cycles）**，即 FMA 不足以掩盖 LDG。

此外，寄存器 prefetch 路径（SM75 无 `cp.async`）需要占用 `pA[2]` + `pB[2]` 共 16 个额外寄存器，寄存器压力从 v4 的 59 上升，导致 occupancy 降低抵消了延迟隐藏收益。

v5 的 sync 时序：

```
loop k:
  发射 LDG(tile_{k+1}) → 寄存器
  执行 FMA(tile_k) 从 smem[cur]        ← 试图掩盖 LDG
  __syncthreads()                       ← 等待所有 warp 到达
  写寄存器 → smem[next]
  __syncthreads()                       ← 保证 STS 完成
```

第一个 `__syncthreads()` 是全 block 屏障，若某个 warp 的 FMA 先完成，它必须等待最慢的 warp。在 256 线程（8 个 warp）下，warp divergence 使实际等待时间接近最坏情况。**两次 sync 仍然将搬运和计算串行化**，仅将 LDG latency 从 sync 之后移到 sync 之前，并未真正重叠。

#### 2.6.4 瓶颈分析

在 M=N=K=1024 时，v5 达到 1.9253 TFLOPS，对应峰值利用率 35.4%，主要瓶颈：

1. **smem_B 的 2-way bank conflict**（继承自 v4，未解决）
2. **double buffering 掩盖失效：** LDG 延迟（L2 miss）约 200–400 cycles，计算窗口不足以完全掩盖 LDG 延迟。
3. **寄存器压力升高**：`pA[2]/pB[2]` 额外占用 8 个寄存器，`--launch_bounds__(256,2)` 约束上限 64 个，marginal occupancy 不改善
4. **M=2048 性能断崖**：1.1466 TFLOPS，远低于 1024 规模的 1.9253，推测 L2 cache 失效（BM=64 的 block tile 数量增大，L2 working set 超出 2MB）

### 2.7 sgemm_v6_large_tile
#### 2.7.1 方案分析

v5（BM=BN=64, BK=32, TM=TN=4）在 M=N=K=1024 达到 1.9253 TFLOPS，占峰值 35.4%。剩余 64.6% 的损失来源需要逐层定位。

**计算访存比（tile 级）：**

$$ \text{AI}_{v5} = \frac{2 \times 64 \times 64}{(64 + 64) \times 4} = 16.0\ \text{FLOP/B} $$

Ridge Point = 18.9 FLOP/B，v5 的 AI 低于 Ridge Point，**仍处于 memory-bound 区间**。这是根本瓶颈——在 smem 复用率不足的情况下，无论如何优化计算侧（ILP、warp tiling）都无法突破带宽天花板。

优化方向1：提升 AI  → 增大 BM、BN

优化方向2：增强 double buffering 掩盖效果 → 增大计算窗口 → 增大 BK、TM×TN

三个方向的调整受 smem 和寄存器两个硬约束：

**smem 约束（SM75，64 KB/SM，目标 ≥2 blocks/SM）：**

$$ \text{smem\_per\_block} = 2 \times (BK \times BM + BK \times BN) \times 4 \leq \frac{64\text{ KB}}{2} = 32\text{ KB} $$

$$ BK \times (BM + BN) \leq 4096 $$

在 $BM = BN = 128$ 下：

$$ BK \leq \frac{4096}{256} = 16 $$

BK 最大取 16，smem = $2 \times 16 \times 256 \times 4 = 32\text{ KB}$，恰好满足 2 blocks/SM。

**寄存器约束（目标 2 blocks/SM，256 线程/block）：**

$$ \text{reg/thread} \leq \frac{65536}{256 \times 2} = 128 $$

`acc[TM][TN] = acc[8][8]` 占 64 个寄存器，加上 `reg_A[8]`、`reg_B[8]`、prefetch 寄存器，总计约 85–90 个，**满足 128 的上限**。

#### 2.7.2 代码
```cuda
#include "gemm.h"

static constexpr int BM = 128;
static constexpr int BN = 128;
static constexpr int BK = 16;
static constexpr int TM = 8;
static constexpr int TN = 8;

static_assert(BK % 4 == 0);
static_assert(BN % 4 == 0);

// 每线程 LDG 次数
static constexpr int LDG_A = (BK * BM) / (256 * 4);  // = 2
static constexpr int LDG_B = (BK * BN) / (256 * 4);  // = 2

__device__ __forceinline__ float4 ldg128_safe(
    const float* ptr, int r, int c, int rows, int cols)
{
    float4 val = {0.f, 0.f, 0.f, 0.f};
    if (r >= rows) return val;
    if (c + 3 < cols) {
        val = __ldg(reinterpret_cast<const float4*>(&ptr[r * cols + c]));
    } else {
        val.x = (c     < cols) ? ptr[r * cols + c    ] : 0.f;
        val.y = (c + 1 < cols) ? ptr[r * cols + c + 1] : 0.f;
        val.z = (c + 2 < cols) ? ptr[r * cols + c + 2] : 0.f;
        val.w = (c + 3 < cols) ? ptr[r * cols + c + 3] : 0.f;
    }
    return val;
}

// 256 线程/block，smem=16KB → 4 blocks/SM（SM75 64KB smem 下）
__global__ __launch_bounds__(256, 2)
void sgemm_v6_large_tile_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int tx  = threadIdx.x;   // [0, BN/TN) = [0, 16)
    const int ty  = threadIdx.y;   // [0, BM/TM) = [0, 16)
    const int tid = ty * blockDim.x + tx;

    const int thread_row = ty * TM;
    const int thread_col = tx * TN;

    __shared__ float smem_A[2][BK][BM];
    __shared__ float smem_B[2][BK][BN];

    float acc[TM][TN] = {};
    float reg_A[TM], reg_B[TN];
    // 故意填充非零垃圾值
#pragma unroll
    for (int i = 0; i < TM; i++) reg_A[i] = 99999.f;
#pragma unroll
    for (int i = 0; i < TN; i++) reg_B[i] = 99999.f;

    // prefetch 寄存器：每线程 2 次 float4 × A/B
    // BK*BM / (256*4) = 16*128/1024 = 2
    float4 p_A[LDG_A], p_B[LDG_B];

    const int num_tiles  = (K + BK - 1) / BK;

    // ---- 预加载第 0 个 tile 到 smem[0] ----
#pragma unroll
    for (int i = 0, idx = tid * 4; i < LDG_A; i++, idx += 256 * 4) {
        int k_ = idx % BK, m = idx / BK;
        p_A[i] = ldg128_safe(A, block_row + m, 0 * BK + k_, M, K);
        smem_A[0][k_    ][m] = p_A[i].x;
        smem_A[0][k_ + 1][m] = p_A[i].y;
        smem_A[0][k_ + 2][m] = p_A[i].z;
        smem_A[0][k_ + 3][m] = p_A[i].w;
    }
#pragma unroll
    for (int i = 0, idx = tid * 4; i < LDG_B; i++, idx += 256 * 4) {
        int n = idx % BN, k_ = idx / BN;
        p_B[i] = ldg128_safe(B, 0 * BK + k_, block_col + n, K, N);
        *reinterpret_cast<float4*>(&smem_B[0][k_][n]) = p_B[i];
    }
    __syncthreads();

// 主循环
#pragma unroll
    for (int k = 0; k < num_tiles; k++) {
        const int cur     = k & 1;
        const int next    = cur ^ 1;
        const bool has_next = (k + 1 < num_tiles);

        // ---- 发射 ldg，prefetch tile_{k+1} 到寄存器 ----
        if (has_next) {
#pragma unroll
            for (int i = 0, idx = tid * 4; i < 2;
                    i++, idx += 256 * 4) {
                int k_ = idx % BK, m = idx / BK;
                p_A[i] = ldg128_safe(
                    A, block_row + m, (k+1) * BK + k_, M, K);
            }
#pragma unroll
            for (int i = 0, idx = tid * 4; i < 2;
                    i++, idx += 256 * 4) {
                int n = idx % BN, k_ = idx / BN;
                p_B[i] = ldg128_safe(
                    B, (k+1) * BK + k_, block_col + n, K, N);
            }
        }

        // ---- FMA（掩盖上方 ldg 的 latency）----
#pragma unroll
        for (int k_ = 0; k_ < BK; k_++) {
#pragma unroll
            for (int m = 0; m < TM; m++)
                reg_A[m] = smem_A[cur][k_][thread_row + m];
#pragma unroll
            for (int n = 0; n < TN; n++)
                reg_B[n] = smem_B[cur][k_][thread_col + n];
#pragma unroll
            for (int m = 0; m < TM; m++)
#pragma unroll
                for (int n = 0; n < TN; n++)
                    acc[m][n] += reg_A[m] * reg_B[n];
        }

        // ---- sync + sts：此时 ldg 数据已就绪 ----
        if (has_next) {
            __syncthreads();
#pragma unroll
            for (int i = 0, idx = tid * 4; i < 2;
                 i++, idx += 256 * 4) {
                int k_ = idx % BK, m = idx / BK;
                smem_A[next][k_  ][m] = p_A[i].x;
                smem_A[next][k_+1][m] = p_A[i].y;
                smem_A[next][k_+2][m] = p_A[i].z;
                smem_A[next][k_+3][m] = p_A[i].w;
            }
#pragma unroll
            for (int i = 0, idx = tid * 4; i < 2;
                 i++, idx += 256 * 4) {
                int n = idx % BN, k_ = idx / BN;
                *reinterpret_cast<float4*>(&smem_B[next][k_][n]) = p_B[i];
            }
            __syncthreads();
        }
    }

    // ---- 写回 C ----
#pragma unroll
    for (int m = 0; m < TM; m++) {
        const int gr = block_row + thread_row + m;
        const int gc = block_col + thread_col;
        if (gr >= M) continue;

        int remaining_cols = N - gc;

        if (remaining_cols >= 8) {
            // 拆成两个 float4 写回
            float4 out0 = {acc[m][0], acc[m][1], acc[m][2], acc[m][3]};
            float4 out1 = {acc[m][4], acc[m][5], acc[m][6], acc[m][7]};

            if (beta != 0.f) {
                float4 old0 = __ldg(reinterpret_cast<const float4*>(&C[gr * N + gc + 0]));
                float4 old1 = __ldg(reinterpret_cast<const float4*>(&C[gr * N + gc + 4]));

                out0.x = alpha*out0.x + beta*old0.x;
                out0.y = alpha*out0.y + beta*old0.y;
                out0.z = alpha*out0.z + beta*old0.z;
                out0.w = alpha*out0.w + beta*old0.w;

                out1.x = alpha*out1.x + beta*old1.x;
                out1.y = alpha*out1.y + beta*old1.y;
                out1.z = alpha*out1.z + beta*old1.z;
                out1.w = alpha*out1.w + beta*old1.w;
            } else {
                out0.x *= alpha; 
                out0.y *= alpha; 
                out0.z *= alpha; 
                out0.w *= alpha;
                out1.x *= alpha; 
                out1.y *= alpha; 
                out1.z *= alpha; 
                out1.w *= alpha;
            }

            *reinterpret_cast<float4*>(&C[gr * N + gc + 0]) = out0;
            *reinterpret_cast<float4*>(&C[gr * N + gc + 4]) = out1;

        } else {
            // 边界列不足 8，逐列写回
#pragma unroll
            for (int n = 0; n < TN && n < remaining_cols; n++) {
                C[gr * N + gc + n] =
                    alpha * acc[m][n] + beta * C[gr * N + gc + n];
            }
        }
    }
}

void sgemm_v6_large_tile(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    // cudaFuncSetAttribute(
    //     sgemm_v6_large_tile_kernel,
    //     cudaFuncAttributePreferredSharedMemoryCarveout,
    //     cudaSharedmemCarveoutMaxShared);

    dim3 block(BN / TN, BM / TM);  // 16×16 = 256
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_v6_large_tile_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK_LAST();
}
```

#### 2.7.3 数据分析

|M=N=K|v4_vec|v5_double_buf|v6 (BK=16)|峰值利用率|vs v4|
|---|---|---|---|---|---|
|256|1.0695|0.9347|0.5369|9.9%|−49.8%|
|512|1.5688|1.6267|1.6994|31.2%|+8.3%|
|1024|1.8902|1.9050|**2.5315**|**46.5%**|**+33.9%**|
|2048|1.1435|1.1473|1.6317|30.0%|+42.8%|

- **M=N=K=256：wave quantization 主导**

$$ \text{grid}_{v6} = \left\lceil\frac{256}{128}\right\rceil^2 = 4\ \text{blocks},\quad \text{grid}_{v4} = \left\lceil\frac{256}{64}\right\rceil^2 = 16\ \text{blocks} $$

SM75 共 24 SM，v6 仅 4 blocks，SM 利用率 $4/24 \approx 17\%$。v4 的 16 blocks 利用率 $67\%$，因此 v6 在此规模的性能（0.4863）不及 v4（1.0646）的一半。

- **M=N=K=512：v6 首次超越 v4**

$$ \text{grid}_{v6} = 16\ \text{blocks},\quad \text{SM 利用率} = 67\% $$

v6（1.6696）> v4（1.5742），差距 +6.1%。AI 提升（32 vs 16 FLOP/B）开始发挥作用，但 SM 利用率仍未饱和，收益有限。

- **M=N=K=1024：性能峰值**

$$ \text{grid}_{v6} = 64\ \text{blocks},\quad \text{blocks/SM} \approx 2\text{–}3,\quad \text{SM 利用率} = 100\% $$

v6（2.5081）vs v4（1.8965），提升 **+32.2%**。AI 超过 Ridge Point：

$$ \text{AI}_{v6} = 32.0\ \text{FLOP/B} > 18.9\ \text{FLOP/B (Ridge Point)} $$

Roofline 上界切换到 compute-bound：

$$ \text{上界} = \min(5.44,\ 0.288 \times 32.0) = \min(5.44,\ 9.22) = 5.44\ \text{TFLOPS} $$

实测 2.508 TFLOPS，利用率 46.1%，剩余损失来自 smem_B bank conflict（2-way，TN=8）。

- **M=N=K=2048：性能回落**

v6（1.6329）相比 1024 的 2.5081 下降 **-34.9%**，但仍比 v4（1.1406）高 +43.2%。回落原因：

working set 超出 L2：

$$ (A + B + C) = 3 \times 2048^2 \times 4\ \text{B} = 48\ \text{MB} \gg 1.5\ \text{MB (L2)} $$

每次 LDG 均为 DRAM miss，实际带宽受 288 GB/s 限制。此时 AI=32 的优势部分被 DRAM 延迟抵消，但大 tile 减少了 block 总数（256 blocks vs v4 的 1024 blocks），**总 LDG 次数降低 4 倍**，仍保留显著优势。

#### 2.7.4 瓶颈分析
##### 2.7.4.1 理论上限

SM75（GTX 1660 Ti）的硬件参数：

$$ \text{Peak FP32} = 5.44\ \text{TFLOPS},\quad \text{BW} = 288\ \text{GB/s},\quad \text{Ridge Point} = 18.9\ \text{FLOP/B} $$

v6 的 AI：

$$ \text{AI} = \frac{2 \times BM \times BN}{(BM + BN) \times 4} = \frac{2 \times 128 \times 128}{256 \times 4} = 32.0\ \text{FLOP/B} $$

AI > Ridge Point，Roofline 上界为 compute-bound：

$$ \text{Roofline} = 5.44\ \text{TFLOPS} $$

实测 1024 规模 2.5315 TFLOPS，利用率 **46.5%**，损失 **53.5%**。

损失来自四个层次，从硬件到算法依次剥离：

$$ \underbrace{5.44}_{\text{Peak}} \xrightarrow{-\text{occupancy}} \xrightarrow{-\text{bank conflict}} \xrightarrow{-\text{loop overhead}} \xrightarrow{-\text{其他}} 2.53 $$

##### 2.7.4.2 Occupancy 损失

SM75 资源上限：65536 寄存器/SM，64 KB smem/SM，32 warps/SM，1024 线程/SM。

v6 每 block：256 线程（8 warps），smem = 32 KB。

smem 约束：

$$ \left\lfloor \frac{64\ \text{KB}}{32\ \text{KB}} \right\rfloor = 2\ \text{blocks/SM} \Rightarrow 16\ \text{warps/SM} $$

寄存器约束（`__launch_bounds__(256, 2)` 限定上限 128/thread）：

$$ \left\lfloor \frac{65536}{256 \times 96} \right\rfloor = 2\ \text{blocks/SM} \Rightarrow 16\ \text{warps/SM} $$

实际寄存器数（`acc[64] + reg_A[8] + reg_B[8] + p_A[8] + p_B[8]` ≈ 96）与 smem 约束吻合，两者均限定为 **2 blocks/SM = 16 warps/SM**。

理论 warp occupancy：

$$ \frac{16}{32} = 50\% $$

occupancy = 50% 对应的 TFLOPS 上界（假设其他完美）：

$$ 5.44 \times 50\% = 2.72\ \text{TFLOPS} $$

实测 2.53，说明 **occupancy 是主导损失**，将上界从 5.44 压缩到 2.72。

##### 2.7.4.3 bank conflict 

smem_A 布局 `[2][BK][BN] = [2][16][128]`。
写入阶段：按列写入，4 路冲突；
读取阶段：warp 前 16 个线程访问同一个 bank $(ty \times 8) \: mod \: 32$，后 16 个访问另一个$((ty + 1) \times 8) \: mod \: 32$，Broadcast，无冲突。

smem_B 布局 `[2][BK][BN] = [2][16][128]`：
写入阶段：`n = idx % BN, k_ = idx / BN`，同一 warp 所有线程访问同一行，无冲突。
读取阶段：同一列 $tx$ 读取相同 bank $(tx \times 8 + n) \: mod \: 32$，Broadcast；同一行读取不同列，无冲突。

同一 warp 内 32 个线程，但 block 为 16×16，warp 按行优先排列（tx 方向），实际一个 warp 内 tx ∈ [0,15]（前 16 线程）或 tx ∈ [0,15]（后 16 线程共享同一 ty），一行线程有 4 路冲突，一个 warp 则有 8 路冲突。

smem 读基本被 FMA 掩盖，bank conflict 对 TFLOPS 几乎无影响。

##### 2.7.4.4 Loop Overhead 损失

每次迭代两次 `__syncthreads()`，共 $K/BK = 64$ 次迭代：

$$ \text{sync overhead} = 64 \times 2 \times 20\ \text{cycles} = 2560\ \text{cycles（估算）} $$

FMA 总周期（理想）：

$$ \frac{2 \times 1024^3}{5.44 \times 10^{12}} \times 1.8 \times 10^9 \approx 700{,}000\ \text{cycles/block} $$

overhead 占比 $\approx 0.4\%$，可忽略。

##### 2.7.4.5 Wave quantization（1024 规模）

$$ \text{grid} = \frac{1024}{128} \times \frac{1024}{128} = 64\ \text{blocks} $$

$$ 64 / 24\ \text{SM} = 2.67\ \text{blocks/SM（非整数）} $$

64 blocks 在 24 SM 上的分配：$24 \times 2 = 48$ blocks 第一波，$64 - 48 = 16$ blocks 第二波。第二波只占 $16/24 \approx 67\%$ 的 SM，产生 tail effect。

有效利用率修正：

$$ \frac{64}{3 \times 24} \times 3 = \frac{64}{72} \approx 88.9\%\ \text{的时间 SM 满负荷} $$

最后一波 16 blocks 在 16 个 SM 上运行时，另外 8 个 SM 空闲，等效损失约 $\frac{8}{24} \times \frac{T_{\text{last\_wave}}}{T_{\text{total}}}$。tail wave 占总时间的比例约 $1/3$，损失：

$$ \frac{8}{24} \times \frac{1}{3} \approx 11\%\ \text{的 SM 利用率损失} $$

这是 2.72 → 2.5315 之间 7.2% 损失的主要来源。

##### 2.7.4.5 总结

| 瓶颈                   | 当前损失     | 可解决性  | 解决方案                      |
| -------------------- | -------- | ----- | ------------------------- |
| Occupancy 50%        | 主导（~50%） | 部分可解  | 减小 smem（降 BK）或切换 WMMA     |
| smem_B bank conflict | ~0%      | 基本不可解 | 已被 FMA 掩盖                 |
| Loop overhead        | ~0.4%    | 基本不可解 | BK 已是约束下最大值，overhead 不可避免 |
| Wave quantization    | ~6.8%    | 基本不可解 | SM 7.5下，已经拉满了             |

occupancy 是主导瓶颈，提升路径只有两条：

**路径 A：降低 smem 占用，换取更高 occupancy**

降低BK，减少smem，但是occupancy 提升被 BK 减小带来的其他代价（loop overhead 翻倍、double buf 窗口缩小）完全抵消。**路径 A 在当前框架内已无空间。**

**路径 B：切换 Tensor Core（WMMA）**

将 acc 寄存器存入 Tensor Core 的 fragment 中，缓解寄存器压力。

## 三、WMMA HGEMM（Tensor Core MMA）

[Tensor Core 介绍](Tensor%20Core%20介绍.md)
### 3.1  wmma_v1_naive

#### 3.1.1 参数设计
SM75 Tensor Core 唯一支持的 fragment 尺寸，三个参数均由硬件固定，无设计空间：

$$ (WM,\ WN,\ BK) = (16,\ 16,\ 16) $$

v1 无 smem，K 维每步直接调用一次 `mma_sync`，步长必须与 fragment K 尺寸对齐：

$$ BK = WK = 16 $$

SM75 硬性上限：

$$ \text{warps/block} \leq 32\ \Rightarrow\ \text{WARP\_NUM\_X} \times \text{WARP\_NUM\_Y} \leq 32 $$

v1 无 smem，寄存器消耗极低，选取：

$$ \text{WARP\_NUM\_X} = \text{WARP\_NUM\_Y} = 4\ \Rightarrow\ 16\ \text{warps/block} = 512\ \text{threads/block} $$

选 16 而非 32 的原因：保留余量，避免后续版本引入 smem 后触发 occupancy 硬降。

由 warp 数与 fragment 尺寸直接推导：

$$ BM = \text{WARP\_NUM\_Y} \times WM = 4 \times 16 = 64 $$

$$ BN = \text{WARP\_NUM\_X} \times WN = 4 \times 16 = 64 $$

### 3.2 代码

```cuda


```

## 四、MMA PTX SGEMM

## 五、CUTLASS SGEMM
