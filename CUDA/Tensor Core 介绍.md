# 1. Tensor Core 和 WMMA
## 1.1 Tensor Core
**Tensor Core** 是 NVIDIA 在 Volta 架构（2017）首次引入的专用矩阵运算单元，之后在 Turing、Ampere、Hopper 等架构上继续演进。其核心目标是加速深度学习中的 **矩阵乘法-累加（Matrix Multiply-Accumulate, MMA）** 运算，这是神经网络计算的核心。

与传统 CUDA 核心不同，Tensor Core 专为 **高吞吐量矩阵计算** 设计，可在单个时钟周期内完成大量乘加操作。
## 1.2 WMMA (Warp Matrix Multiply-Accumulate)
- **定义**：WMMA 是 CUDA 提供的 **API/指令级接口**，允许程序员在 GPU 上使用 Tensor Core 执行 MMA 操作。
- **特点**：
    - 属于 **Warp 级别** 的操作，每个 warp（32 个线程）可以合作完成一个小矩阵乘加（例如 16x16x16）。
    - CUDA 里有 `wmma` namespace，比如 `wmma::fragment` 用来表示矩阵块。
- **作用**：
    - CUDA 开发者通过 WMMA 调用 Tensor Core 的 MMA 硬件，而不用手动写低级汇编。
    - 是软件到硬件的桥梁。

## 名词解释
| 名称          | 全称<br>                          | 类型       | 功能                          | 层级/作用            |
| ----------- | ------------------------------- | -------- | --------------------------- | ---------------- |
| Tensor Core | -                               | 硬件单元     | 执行 MMA                      | 底层硬件             |
| MMA         | Matrix Multiply-Accumulate      | 算法操作     | 矩阵乘加 (D=A*B+C)              | Tensor Core 做的运算 |
| WMMA        | Warp Matrix Multiply-Accumulate | CUDA API | 调用 Tensor Core 的 MMA（warp级） | 软件接口/编程层         |

| 名称    | 全称                                              | 功能         | 公式              | 数据类型           | BLAS等级  |
| ----- | ----------------------------------------------- | ---------- | --------------- | -------------- | ------- |
| GEMM  | General Matrix Multiply                         | 通用矩阵乘法     | C = α·A·B + β·C | 任意（FP32/FP16等） | Level-3 |
| HGEMM | Half-precision General Matrix Multiply          | 半精度矩阵乘法    | C = α·A·B + β·C | FP16/BF16      | Level-3 |
| HGEMV | Half-precision General Matrix-Vector Multiply   | 半精度矩阵-向量乘法 | y = α·A·x + β·y | FP16/BF16      | Level-2 |
| SGEMM | Single-precision General Matrix Multiply        | 单精度矩阵乘法    | C = α·A·B + β·C | FP32           | Level-3 |
| SGEMV | Single-precision General Matrix-Vector Multiply | 单精度矩阵-向量乘法 | y = α·A·x + β·y | FP32           | Level-2 |
|       |                                                 |            |                 |                |         |

# 2. Tensor Core 的工作原理
Tensor Core 的基本运算是 **矩阵乘法累加**：
$$D = A \times B + C$$  
其中：
- (A, B) 是输入矩阵
- (C) 是累加矩阵（可初始化为 0）
- (D) 是输出矩阵

## 2.1 数据类型支持
Tensor Core 支持多种数据类型，主要用于平衡 **精度** 与 **性能**：

|架构|支持的数据类型|
|---|---|
|Volta|FP16 输入 + FP32 输出累加|
|Turing|FP16, INT8, INT4 输入 + FP32 输出累加|
|Ampere|FP16, BF16, TF32, INT8, INT4|
|Hopper|FP8, FP16, BF16, TF32, INT8, INT4|

> **说明**：Tensor Core 的引入让深度学习训练和推理的吞吐量相比传统 CUDA 核心提升 3~12 倍，尤其在 FP16/BF16 训练中效果显著。
## 2.2 并行策略
Tensor Core 以 **矩阵块（tile）为单位** 并行执行：
- Volta: 每个 Tensor Core 执行 (4 \times 4) FP16 矩阵乘法
- Ampere: 支持 (8 \times 8) 或 (16 \times 16) tile
- Hopper: 支持更大 tile，增加 FP8 和稀疏矩阵加速

这种 tile 并行允许 **SIMD（单指令多数据）式执行**，充分利用 GPU 的吞吐能力。
# 3. Tensor Core 架构
Tensor Core 通常位于每个 **SM（Streaming Multiprocessor）** 内，和 CUDA 核心、共享内存、寄存器紧密结合。
结构特点：
1. **矩阵乘加单元（MMA Unit）**：执行 D = A×B+C 运算
2. **加载/存储路径优化**：从寄存器或共享内存高效读取矩阵 tile
3. **可配置精度累加**：例如 FP16 输入，FP32 累加

在 Ampere 架构中：
- 每个 SM 包含多个 Tensor Core
- Tensor Core 可以与 CUDA 核心同时工作
- 支持稀疏矩阵优化（Sparse Tensor Core），在 Transformer 模型推理中性能提升约 2 倍
# 4. Tensor Core 的应用场景
Tensor Core 的设计目标主要集中在深度学习：
1. **训练**
    - CNN、RNN、Transformer 等网络
    - 支持混合精度训练（FP16/BF16 + FP32 累加）
    - 大幅减少显存占用，提高吞吐量
2. **推理**
    - FP16、INT8、INT4 推理加速
    - Transformer、LLM 模型推理（ChatGPT、BERT、GPT 系列）
3. **科学计算**
    - 高性能矩阵运算（线性代数、物理仿真）
    - 可以用 FP16 或 TF32 提升矩阵运算速度
# 5. 编程和优化
## 5.1 CUDA / cuBLAS / cuDNN
Tensor Core 可以通过 NVIDIA 的深度学习库直接调用：
- **cuBLAS**: GEMM 运算 (矩阵乘法)
- **cuDNN**: CNN、RNN 层加速
- **CUTLASS**: 高度可定制矩阵乘法模板
## 5.2 关键优化策略
1. **数据对齐**：矩阵 tile 对齐 16/32 字节可提升吞吐
2. **混合精度训练**：FP16/BF16 输入 + FP32 累加
3. **利用共享内存**：减少全局内存访问
4. **稀疏矩阵优化**：AMPERE/Hopper 架构可利用稀疏 Tensor Core
# 6. 性能提升示例
以 Ampere 架构为例（A100 GPU）：
- 单精度 FP32 GEMM：≈ 19.5 TFLOPS
- Tensor Core FP16 GEMM：≈ 312 TFLOPS（16 倍提升）
- Tensor Core TF32 GEMM：≈ 156 TFLOPS
- INT8 GEMM：≈ 624 TOPS

> 可见 Tensor Core 对 **AI 推理和训练** 有革命性提升。

# 7. 总结
Tensor Core 是 NVIDIA 针对 **深度学习矩阵运算**专门设计的硬件单元，通过：
- 高吞吐量的矩阵乘加运算
- 多精度和稀疏矩阵支持
- 与 CUDA 核心协同工作

极大地提升了 AI 模型训练和推理性能。  
它的核心优势在于 **tile 并行 + 专用矩阵硬件 + 灵活精度支持**。
