[Tensor Core 介绍](Tensor%20Core%20介绍.md)
工具代码来自[cuda_hgemm](https://github.com/Bruce-Lee-LY/cuda_hgemm/tree/master/src/common)
算子代码来自[LeetCUDA](https://github.com/xlite-dev/LeetCUDA/blob/main/kernels/hgemm/wmma/hgemm_wmma.cu)（强无敌的开源库）
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


# 二、V1 Naive Kernel

