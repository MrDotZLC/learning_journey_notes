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

##### 2.3.3.2 v3 vs. cuBLAS

|M=N=K|vs_cublas|
|---|---|
|256|2.03x|
|512|2.88x|
|1024|3.08x|
|2048|2.04x|

小规模下 v3 大幅超过 cuBLAS，原因同 v2：小矩阵时 cuBLAS 的 kernel 选择有启动开销。

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

v3 中 Tile A 的元素是逐个写入 SM，不用考虑写入方向，v4 是连续写入 4 个 float，写入方向需要和读取方向保持一致，即沿着 K 轴方向。，索引计算改为：

$$ k_ = i \bmod BK, \quad m = i / BK $$

读取到寄存器进行外积计算时，暂不需要 Vectorized，因为 smem → 寄存器的读取**不经过L2事务**，瓶颈不在事务数和指令数，在**bank访问模式**。读取的数据量是固定的，单次读取 float/float4 仅能减少指令数，但会引入更严重的bank conflict。

由于每个线程沿着 K 轴，每次从 smem 取 TM/TN 个元素，bank conflict为：

$$\text{bank\_conflict}_{smem_A} = \frac {32} {TM} = \frac {32} {4} = 8$$

32线程只访问8个bank（bank 0..7） 
→ 每个bank被4个线程同时访问 
→ 4-way bank conflict 

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

|M=N=K|v1|v2|v3|v4|cuBLAS|
|---|---|---|---|---|---|
|1024|0.33|0.57|1.11|**1.84**|0.43|
|2048|0.19|0.34|0.67|**1.13**|0.34|

v4 在所有规模下都超过 cuBLAS FP16，这是因为：

1. cuBLAS 选择的 kernel 针对通用场景，不一定对当前矩阵规模最优
2. GTX 1660 Ti 的 FP16 Tensor Core 路径在这个规模下未必被充分利用

#### 2.4.4 瓶颈分析

v4 在 1024 规模达到 1.84 TFLOPS，与理论峰值 5.44 TFLOPS 的比值：

$$ \frac{1.84}{5.44} = 33.8\% $$

距离峰值仍有 3 倍空间，瓶颈转移到了**计算阶段的 smem 读取延迟**：

外积循环中 `regA` 和 `regB` 从 smem 读取：

```cuda
for (int k_ = 0; k_ < BK; k_++) {
    for (int m = 0; m < TM; m++)
        regA[m] = smem_A[k_][thread_row + m];  // TM次smem读
    for (int n = 0; n < TN; n++)
        regB[n] = smem_B[k_][thread_col + n];  // TN次smem读
    // 16次FMA
}
```

v4 外积计算阶段，每次 k_ 迭代读 `smem_B[k_][thread_col + n]`。

v4 的线程分配方式（block=16×16），warp0 内 tid=0..15 的 thread_col（tx=0..15，TN=4）：

$$ \text{thread\_col} = 0, 4, 8, \ldots, 60 \quad \text{（16个不同值）} $$

对应 smem_B 的 index = 0..60，bank_id = index % 32：

bank 0..28 各被两个线程访问，地址不同：

$$ \rightarrow \text{2-way bank conflict} \rightarrow \text{每次 smem 读需要 2 个串行周期} $$

$$ \lfloor 60 / 32 \rfloor = 1 \Rightarrow \text{bank 0..28 各被访问 2 次} \Rightarrow \text{2-way conflict} $$

消除方法：**将 warp 内不同地址数控制在 32 以内，且 index < 32**。

### 2.5 sgemm_v5_warp
#### 2.5.1 方案分析

消除 v4 的 smem bank conflict，将每次 smem 读取从 2 个串行周期降至 1 个周期。即将 warp 单次处理数据减半，调度次数翻倍。

$$ \text{block tile} = 64 \times 64,\quad \text{线程数} = 256\ \text{（与v4相同）} $$

$$ \text{warp tile} = WARP\_TILE\_M \times WARP\_TILE\_N $$

约束条件：

$$ \frac{WARP\_TILE\_M}{TM} \times \frac{WARP\_TILE\_N}{TN} = 32 \quad \text{（每warp恰好32线程）} $$

$$ \text{warp数} = \frac{BM}{WARP\_TILE\_M} \times \frac{BN}{WARP\_TILE\_N} = \frac{256}{32} = 8 $$

消除 bank conflict 的核心约束：

**warp 内不同地址的 index 必须全部 < 32**

warp 内不同 thread_col 的数量 = $\frac{WARP\_TILE\_N}{TN}$，对应 index 步长 = TN=4：

$$ \text{index最大值} = \left(\frac{WARP\_TILE\_N}{TN} - 1\right) \times TN = WARP\_TILE\_N - TN $$

要求：

$$ WARP\_TILE\_N - TN < 32 \Rightarrow WARP\_TILE\_N \leq 32 $$

取 $WARP\_TILE\_N = 32$，则：

$$ \frac{WARP\_TILE\_M}{TM} \times \frac{32}{TN} = 32 \Rightarrow \frac{WARP\_TILE\_M}{4} \times 8 = 32 \Rightarrow WARP\_TILE\_M = 16 $$

$$ \text{warp数} = \frac{64}{16} \times \frac{64}{32} = 4 \times 2 = 8\ \checkmark $$

#### 2.5.2 代码

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

// warp tile 参数
static constexpr int WARP_TILE_M = 16;   // 每warp负责的M方向元素数
static constexpr int WARP_TILE_N = 32;   // 每warp负责的N方向元素数
static constexpr int WARP_ROWS   = BM / WARP_TILE_M;   // 4
static constexpr int WARP_COLS   = BN / WARP_TILE_N;   // 2

static_assert(WARP_TILE_M / TM * WARP_TILE_N / TN == 32,
              "warp tile must contain exactly 32 threads");
static_assert(WARP_ROWS * WARP_COLS * 32 == 256,
              "block must have 256 threads");
static_assert(BK % 4 == 0, "BK must be divisible by 4");
static_assert(BN % 4 == 0, "BN must be divisible by 4");


__global__ void sgemm_v5_warp_kernel(
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

    const int tid = ty * blockDim.x + tx;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // warp 在 block 内的位置
    const int warp_row = warp_id / WARP_COLS;   // 0..3
    const int warp_col = warp_id % WARP_COLS;   // 0..1

    // lane 在 warp tile 内的位置
    // warp tile 内列方向有 WARP_TILE_N/TN = 8 个线程
    const int lane_row = lane_id / (WARP_TILE_N / TN);   // 0..3
    const int lane_col = lane_id % (WARP_TILE_N / TN);   // 0..7

    // Tile 内线程负责计算的 C 起始元素索引 
    const int thread_row = warp_row * WARP_TILE_M + lane_row * TM;
    const int thread_col = warp_col * WARP_TILE_N + lane_col * TN;

    __shared__ float smem_A[BK][BM];
    __shared__ float smem_B[BK][BN];

    float acc[TM][TN] = {0.f};
    float reg_A[TM], reg_B[TN];
    // // 故意填充非零垃圾值
    // #pragma unroll
    // for (int i = 0; i < TM; i++) reg_A[i] = 99999.f;
    // #pragma unroll
    // for (int i = 0; i < TN; i++) reg_B[i] = 99999.f;

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
                    val.x = (global_c     < N) ? B[global_r * N + global_c    ] : 0.f;
                    val.y = (global_c + 1 < N) ? B[global_r * N + global_c + 1] : 0.f;
                    val.z = (global_c + 2 < N) ? B[global_r * N + global_c + 2] : 0.f;
                    val.w = (global_c + 3 < N) ? B[global_r * N + global_c + 3] : 0.f;
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

void sgemm_v5_warp(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN,
              (M + BM - 1) / BM);
    sgemm_v5_warp_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK_LAST();
}

```

### 2.5.3 数据分析
