[Tensor Core 介绍](Tensor%20Core%20介绍.md)
工具代码来自[cuda_hgemm](https://github.com/Bruce-Lee-LY/cuda_hgemm/tree/master/src/common)
算子代码来自[LeetCUDA](https://github.com/xlite-dev/LeetCUDA/blob/main/kernels/hgemm/wmma/hgemm_wmma.cu)（强无敌的开源库）
WMMA代码使用简单但不灵活，可能存在bank conflict 且无法优化。MMA PTX更底层、更灵活，结合Swizzle机制，能够很好避免bank conflict。
本篇重点介绍WMMA、MMA PTX、Swizzle。
# 一、API 介绍
## 1.1 cublasGemmEx 函数介绍
### 概念
`cublas` 是基础线性代数算子库，与常规语言的行主序内存排布有所不同，`cublas` 中是列主序，在通用矩阵乘法的扩展接口 `cublasGemmEx` 的使用中也体现了这一点（网络上的科普有很多误区）。
`cublasGemmEx` 是通用矩阵乘法（GEMM）的扩展接口，核心目标是**支持多数据类型、混合精度和 Tensor Core，加速现代 GPU 上的高性能矩阵乘法。**
`cublasGemmEx` 计算的是标准 GEMM：
$$C = \alpha \cdot op(A) \cdot op(B) + \beta \cdot C$$
其中：
- A：`m × k`
- B：`k × n`
- C：`m × n`
- `op(X)` = `X` 或 `Xᵀ`
```
  cublasStatus_t cublasGemmEx(
    cublasHandle_t handle,        // cuBLAS 句柄，管理 stream / math mode 等状态
    cublasOperation_t transA,     // A 是否转置（CUBLAS_OP_N/T/C）
    cublasOperation_t transB,     // B 是否转置（CUBLAS_OP_N/T/C）
    int m,                        // 结果矩阵 C 的行数
    int n,                        // 结果矩阵 C 的列数
    int k,                        // A 与 B 相乘的内维度
    const void* alpha,            // 标量 α，类型需与 computeType 匹配
    const void* A,                // 矩阵 A 的 device 指针（列主序）
    cudaDataType Atype,           // A 在内存中的数据类型
    int lda,                      // A 的 leading dimension（列主序下为行数）
    const void* B,                // 矩阵 B 的 device 指针（列主序）
    cudaDataType Btype,           // B 在内存中的数据类型
    int ldb,                      // B 的 leading dimension
    const void* beta,             // 标量 β，类型需与 computeType 匹配
    void* C,                      // 矩阵 C 的 device 指针（输入 + 输出）
    cudaDataType Ctype,           // C 在内存中的数据类型
    int ldc,                      // C 的 leading dimension
    cublasComputeType_t computeType, // 计算/累加精度，决定 Tensor Core 使用
    cublasGemmAlgo_t algo          // GEMM 算法选择（默认或 Tensor Core）
  );
```
### 例子1：显式转置（A^T）
![](assets/Pasted%20image%2020260202064543.png)
```
// 行主序 A(2x3), B(3x4), C(2x4)
float *A, *B, *C;
float alpha = 1.0f;
float beta  = 0.0f;

auto m_C_t = std::make_shared<Matrix>(m_M, m_N, "Matrix C T");
// 列主序读取并转置 == 逻辑行主序
cublasGemmEx(
    handle,
    CUBLAS_OP_T, CUBLAS_OP_T,   // 显式转置
    2, 4, 3,                    // m=2, n=4, k=3   (C 的尺寸)
    &alpha,
    A, CUDA_R_16F, K,           // A: 3x2, lda = 3
    B, CUDA_R_16F, N,           // B: 4x3, ldb = 4
    &beta,
    m_C_t->getDevPtr(), CUDA_R_16F, M,           // C: 4*2, ldc = 2
    CUBLAS_COMPUTE_16F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
);
// 在算子外，将C转置回来。
dim3 block_size(32, 8);
dim3 grid_size((M - 1) / block_size.x + 1, (N - 1) / block_size.y + 1);
transpose_naive<<<grid_size, block_size>>>(m_C_t->getDevPtr(), C, N, M);
```
### 例子2：公式推导
![](assets/Pasted%20image%2020260202065510.png)
```
// 行主序 A(2x3), B(3x4), C(2x4)
float *A, *B, *C;
float alpha = 1.0f;
float beta  = 0.0f;

auto m_C_t = std::make_shared<Matrix>(m_M, m_N, "Matrix C T");
// Cᵀ(4x3) = Bᵀ(4x3) × Aᵀ(3x2)
cublasGemmEx(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,   // 非显式转置
    4, 2, 3,                    // m=2, n=4, k=3   (C 的尺寸)
    &alpha,
    B, CUDA_R_16F, N,           // B: 4x3, lda = 4
    A, CUDA_R_16F, K,           // A: 3x2, ldb = 3
    &beta,
    m_C_t->getDevPtr(), CUDA_R_16F, N,           // C: 4*2, ldc = 2
    CUBLAS_COMPUTE_16F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
);
// 在算子外，将C转置回来。
dim3 block_size(32, 8);
dim3 grid_size((M - 1) / block_size.x + 1, (N - 1) / block_size.y + 1);
transpose_naive<<<grid_size, block_size>>>(m_C_t->getDevPtr(), C, N, M);
```
  
## 1.2 wmma api 介绍 
CUDA 提供的 **warp 级 Tensor Core 编程接口**，本质是**手写 cuBLAS Tensor Core 内核**。
**1 warp 计算 1 个矩阵 tile**，block 至少 32 threads，多 warp/block ≠ 多 tile（需手动设计）。
数学语义：D = A × B + C。
### 1.2.1 fragment 的三种用法
声明乘加运算 $C/D=AB+C$中的A、B、C/D，
#### 1.2.1.1 matrix_a / matrix_b
保存 A（M×K）或 B（K×N）的一个 tile，只用于 `load_matrix_sync`
```
wmma::fragment<
    wmma::matrix_a | wmma::matrix_b,
    M, N, K,
    half | bf16 | tf32,
    wmma::row_major | wmma::col_major // layout 表示内存布局，行/列主序
>
```
#### 1.2.1.2 accumulator
保存 C/D（M×N）的一个 tile，FP16/BF16/TF32 → **FP32 累加**
```
wmma::fragment<
    wmma::accumulator,
    M, N, K,
    float // accumulator 必须是 `float`
>
```
#### 1.2.1.3 fragment 精度与 Matrix 大小
wmma中，fragment 精度和 Matrix 大小都有固定搭配，不能随意设置。
![](assets/Pasted%20image%2020260202122232.png)
### 1.2.2 fill_fragment
初始fragment。
```
wmma::fill_fragment(C_frag, 0.0f);
```
### 1.2.3 load_matrix_sync
warp所有线程从 global / shared memory 加载 tile。
`lda`（leading dimension）表示行主序一行的元素个数，列主序同理。
```
wmma::load_matrix_sync(
	A_frag, 
	A_ptr,   // 数据源指针，必须256位对齐
	lda      // 连续行/列的元素步长，只有累加器 accumulator 需要填写
); 
```
### 1.2.4 mma_sync（Tensor Core）
warp 同步执行乘加操作。
```
wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
```
### 1.2.5 store_matrix_sync
写回内存
自动做 **float → half**
支持 row / col major
```
wmma::store_matrix_sync(C_ptr, C_frag, ldc, wmma::mem_row_major);
```
## 1.3 MMA 相关的 PTX 指令
**CUDA 11+ / Hopper 及 CUTLASS 中**的一个操作接口。
`mma.sync` PTX 指令或者 CUTLASS kernel template 的 `mma` 模板。
### 1.3.1 mma
#### 1.3.1.1 API 介绍
```
mma.sync.aligned.m{M}n{N}k{K}.{type_a}.{type_b}.{type_c}.{layout_a}.{layout_b}.{layout_c} d, a, b, c;

m{M}n{N}k{K}：tile 尺寸，表示矩阵 A(M×K) × B(K×N) → C(M×N)
{type_a/b/c}：数据类型（f16/bf16/s8/u8/tf32 等）
{layout_a/b/c}：矩阵存储布局（row 或 col）
aligned：数据对齐，确保 tensor core 可以高效加载
sync：warp 内线程协作同步执行
```
![](assets/Pasted%20image%2020260207160249.png)
mma.sync中传入A、B、C的寄存器地址，其中B是转置后的（合并访存），如下三图所示。
![](assets/Pasted%20image%2020260207161107.png)
![](assets/Pasted%20image%2020260207161113.png)
![](assets/Pasted%20image%2020260207161118.png)
#### 1.3.1.2 例子
`mma.sync.aligned.m16n16k16.row.col.row d, a, b, c;`
- **d**：输出累加 fragment
- **a, b**：输入矩阵 fragment
- **c**：累加源 fragment（初始值可以为零）
- 执行：
$$D_{16×16} = A_{16×16} × B_{16×16} + C_{16×16}$$
> 注意：每个 warp 内 32 个线程合作完成这个操作。
#### 1.3.1.3 数据类型支持

|数据类型|说明|
|---|---|
|`f16`|FP16|
|`bf16`|BF16|
|`tf32`|TensorFloat32|
|`s8`/`u8`|INT8/UINT8|
|`f32`|FP32（通过 accumulate）|
#### 1.3.1.4 Warp 内协作
- 每条 `mma.sync` 指令执行需要 **整个 warp（32 threads）**
- GPU 自动做 **lane 分配和矩阵分片**
- 因此 PTX MMA 指令**不能单线程使用**

### 1.3.2 ldmatrix 
#### 1.3.2.1 api介绍
**全称**：Load Matrix to Shared Memory / Register Tile
把 global memory 或 shared memory 中的 tile 高效加载到 寄存器 tile（fragment）。
```
# api:
ldmatrix.sync.aligned.shape.num{.trans}{.ss}.type     r, [p];
ldmatrix.sync.aligned.m8n16.num{.ss}.dst_fmt.src_fmt  r, [p];
ldmatrix.sync.aligned.m16n16.num.trans{.ss}.dst_fmt.src_fmt r, [p];
# param
.shape ={.m8n8,.m16n8}; 
.num ={.x1,.x2,.x4}; 
.ss ={.shared{::cta}}; 
.type ={.b16,.b8};

# example
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r0,%r1,%r2,%r3}, [%r_shared];

```
#### 1.3.2.2 参数介绍

| 字段 / 修饰符   | 可选值 / 范围                     | 含义             | 说明 / 典型用途                          |
| ---------- | ---------------------------- | -------------- | ---------------------------------- |
| `sync`     | 必选                           | Warp 内同步       | 确保 warp 32 个线程协作完成 tile 加载         |
| `aligned`  | 必选                           | 对齐访问           | 数据地址必须按 tile 对齐，提高访存效率             |
| `shape`    | `.m8n8`, `.m8n16`, `.m16n16` | tile 行列尺寸      | 定义要加载的矩阵 tile 尺寸                   |
| `num`      | `.x1`, `.x2`, `.x4`          | 一条指令加载 tile 数量 | 控制寄存器输出和 warp 内分片方式                |
| `.trans`   | 可选                           | tile 转置        | 将矩阵在载入寄存器时转置（行列互换）                 |
| `.ss`      | 可选                           | stride swizzle | 优化 shared memory bank conflict     |
| `type`     | `.b16`, `.b8`                | 数据类型           | 16-bit / 8-bit 等，用于 Tensor Core 输入 |
| `.dst_fmt` | `.b8x16`                     | 输出寄存器格式        | 控制寄存器 fragment 的 unpack/pack 格式    |
| `.src_fmt` | `.b6x16_p32`, `.b4x16_p64`   | 输入 memory 格式   | 控制 packed memory 读取方式，低精度矩阵场景使用    |
| `{r}`      | 寄存器列表                        | 输出目标寄存器        | warp 内每个线程分片 tile，组合成 fragment     |
| `[p]`      | memory 地址                    | 源地址            | tile 在 shared/global memory 中的起始地址 |
 
#### 1.3.2.3 参数规则
idmatrix参数是相互协作的、遵循下述表格规则的。
图表1-1：每个 .shape 的矩阵加载实例。
![](assets/Pasted%20image%2020260206135707.png)
图表1-2：6 位或 4 位数据加载的有效用法
 ![](assets/Pasted%20image%2020260206135727.png)
图表1-3：每个矩阵所需的八个地址由八个线程提供，具体取决于.num的值
 ![](assets/Pasted%20image%2020260206170958.png)
图表1-4：用于单个 8×8 矩阵（16 位元素）的 stmatrix 片段布局。
 ![](assets/Pasted%20image%2020260206173243.png)
1. **type、num、寄存器r之间的关系**
   type为b16时，shape只能是8×8，如果矩阵是16 * 16，num则是x4，分成4个8×8 tile，寄存器{%r0,%r1,%r2,%r3}分别存储tile的首地址，每个线程对应 tile 的行地址，如图表1-3。
   ![](assets/Pasted%20image%2020260206174908.png)

2. **源地址的语义**
   是每个 warp 提供的是「矩阵 tile 的基地址集合」，而不是「每个线程要加载的地址」。
   参考图表1-4，每行4个线程，每个线程处理2个half。
   每个线程计算寄存器地址（图表1-3）和处理数据的位置（图表1-4），是不相关的，不要混淆。
   ![](assets/Pasted%20image%2020260206174846.png)![](assets/Pasted%20image%2020260206174752.png)
   
## 1.4 Swizzle介绍


# 二、WMMA
## 2.1 V1 Naive Kernel（m16n16k16）
什么都不考虑。
![](assets/Pasted%20image%2020260205123640.png)
![](assets/Pasted%20image%2020260205123408.png)

## 2.2 V2 Shared Memory（m16n16k16，mma4x2）
每个线程处理 1tile（16 * 16），每个元素需要从 Global Memory 中加载**16次**，使用 Shared Memory 缓存矩阵，将数据加载的资源消耗**降低至1/16**。
把 tile 扩大至64 * 32，A 缓存64 * 16，B 缓存16 * 32，每 tile 分成8个 warp，一个 warp 处理16 * 16。
![](assets/Pasted%20image%2020260205123925.png)
![](assets/Pasted%20image%2020260205211317.png)
## 2.3 V3 （m16n16k16，mma4x2，warp2x4）
让每个 warp 多干点活（16 * 16 -> 2 * 4 * 16 * 16）。
![](assets/Pasted%20image%2020260205222154.png)
![](assets/Pasted%20image%2020260205233757.png)

## 2.4 V4 double buffer async（m16n16k16，mma4x2，warp2x4）
### 2.4.1 思路
tile_n的赋值和tile_n-1的计算是异步的，不等待所有数据全部传输，实现数据传输与计算分tile异步进行。
![](assets/Pasted%20image%2020260205234904.png)
**cp.async** 是Ampere 架构（SM80）引入的新功能。
传统赋值操作（“=”）会阻塞线程，而 cp.async 不会阻塞线程，完全隐藏load延时。
![](assets/Pasted%20image%2020260205234211.png)
### 2.4.2 耗时对比
- L = 每轮 K tile 的 global memory load latency
- C = 每轮 K tile 的 compute latency
- N = K 方向 tile 数量（num_tiles）

| 方法             | 时间公式                       | 理想加速                    |
| -------------- | -------------------------- | ----------------------- |
| 单缓冲            | $(N \cdot (L + C))$        | baseline                |
| 双缓冲            | $(L + N \cdot \max(L, C))$ | 避免部分 idle               |
| cp.async + 双缓冲 | $(L + N \cdot C)$          | 理论最优，load latency几乎完全隐藏 |
### 2.4.3 代码
![](assets/Pasted%20image%2020260206000938.png)

# 三、MMA
**注：受限于设备，本小节的代码没有实际运行，读者自行验证。**
## 3.1 V1 Naive Kernel（m16n8k16）
GMem到SMem的数据传输，没有conflict。
SMem到Reg的数据传输，一次warp处理有2路conflict，ldmatrix有4个（沿用前例），共8路。
![](assets/Pasted%20image%2020260207174551.png)
![](assets/Pasted%20image%2020260207174634.png)
![](assets/Pasted%20image%2020260207182117.png)
![](assets/Pasted%20image%2020260207182501.png)
## 3.2 V2 Kernel（SMem Padding，m16n8k16，warp4x4）
![](assets/Pasted%20image%2020260207174527.png)
![](assets/Pasted%20image%2020260207181932.png)
# 四、Swizzle
[Swizzle 介绍](Swizzle%20介绍.md)
对索引或数据布局做可逆的置换（permutation）。
![](assets/Pasted%20image%2020260207180148.png)
