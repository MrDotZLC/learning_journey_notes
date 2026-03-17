## 1. Tensor Core 总览
### 1.1 Tensor Core（硬件层）
**Tensor Core** 是 NVIDIA 在 Volta 架构（2017）首次引入的**专用矩阵运算硬件单元**，用于高吞吐执行 **矩阵乘法-累加（Matrix Multiply-Accumulate, MMA）**：
$$D = A \times B + C$$
其核心目标是：
- 以远高于 CUDA Core 的吞吐率执行小规模、稠密矩阵计算
- 通过**低/混合精度**换取数量级性能提升
- 面向深度学习训练、推理及部分 HPC 场景
Tensor Core 并非独立核心，而是**集成在 SM（Streaming Multiprocessor）内部**，以 **warp 级协同**方式工作。
### 1.2 相关名词与算子对照
#### 概念层级

|名称|全称|类型|功能|所在层级|
|---|---|---|---|---|
|Tensor Core|-|硬件单元|执行 MMA|GPU 硬件|
|MMA|Matrix Multiply-Accumulate|算法|D=A×B+C|Tensor Core 运算本质|

#### BLAS / GEMM 术语

|名称|全称|公式|数据类型|BLAS 等级|
|---|---|---|---|---|
|GEMM|General Matrix Multiply|C=αAB+βC|任意|Level-3|
|HGEMM|Half GEMM|同上|FP16/BF16|Level-3|
|SGEMM|Single GEMM|同上|FP32|Level-3|
|GEMV|Matrix-Vector|y=Ax|各精度|Level-2|

## 2. Tensor Core 的工作原理
### 2.1 基本计算模型
Tensor Core 原生支持 **tile 级 MMA**。大规模 GEMM 会被拆分为多个固定尺寸 tile，由多个 warp 并行完成。
- 输入矩阵 A、B 通常来自 **寄存器或共享内存**
- 累加矩阵 C 多采用更高精度（如 FP32）
- 输出 D 写回寄存器/内存
### 2.2 数据类型支持（随架构演进）

|架构|支持类型|
|---|---|
|Volta|FP16 → FP32 累加|
|Turing|FP16, INT8, INT4|
|Ampere|FP16, BF16, TF32, INT8, INT4（含稀疏）|
|Hopper|FP8（E4M3/E5M2）, FP16, BF16, TF32|

> Tensor Core 的性能提升主要来自**低精度输入 + 高吞吐并行**，而非单次运算更快。

### 2.3 并行与 Tile 策略
- **Warp 级并行**：一个 warp 驱动一次 MMA
- **Tile 分层**：
    - Block tile → Shared Memory
    - Warp tile → Registers
    - MMA tile → Tensor Core

不同架构支持的 tile 大小不同：
- Volta：4×4
- Ampere：8×8 / 16×16
- Hopper：更大 tile + WGMMA
## 3. Tensor Core 架构位置
Tensor Core 位于 SM 内部，与 CUDA Core、寄存器、共享内存紧密耦合：
- 多个 Tensor Core/SM
- 可与 CUDA Core **并行执行**（控制流、激活函数等）
- Ampere/Hopper 支持 **Sparse Tensor Core**，2:4 结构化稀疏可带来 ~2× 吞吐
## 4. Tensor Core 开发技术栈（全景）
```
应用 / 框架
PyTorch / TF / JAX
        ↓
混合精度与数值策略
AMP / TF32 / FP8 Engine
        ↓
高性能库
cuBLAS / cuBLASLt / cuDNN / TensorRT
        ↓
Kernel 生成与模板
CUTLASS / Triton / TVM
        ↓
CUDA 编程模型
CUDA C++ / WMMA / Inline PTX
        ↓
指令级
PTX mma → SASS HMMA / WGMMA
        ↓
硬件
SM → Tensor Core
```
开发者主要决策点在 **数据类型、算子形态、kernel 路线选择**。
## 5. Tensor Core 开发的不同路线
### 5.1 库驱动（工程首选）
- **AMP + cuBLAS / cuBLASLt / cuDNN**
- 自动触发 Tensor Core
- 稳定、可维护、覆盖 90% 场景
### 5.2 模板驱动（高阶优化）
- **CUTLASS**
- 显式控制 tile、pipeline、epilogue
- 适合非标准 GEMM / Transformer 内核
### 5.3 DSL / 编译器驱动
- **Triton / TVM / MLIR**
- 快速开发、自动映射 Tensor Core
- 性能依赖生成质量
### 5.4 手写 WMMA / PTX（极限）
[Tensor Core：HGEMM 半精度矩阵乘](Tensor%20Core：HGEMM%20半精度矩阵乘.md)
- 仅用于研究或极限优化
- 可维护性差、强架构绑定
## 6. 必须配合的关键技术
- **混合精度**：FP16/BF16/TF32/FP8 + FP32 累加
- **内存优化**：Shared Memory、double buffering、cp.async
- **算子融合**：bias / activation / scaling（epilogue）
- **量化技术**：INT8/FP8 scale、zero-point
- **Profiling**：Nsight Compute（Tensor Core 利用率）
## 7. 性能示例（A100）

|精度|峰值性能|
|---|---|
|FP32|~19.5 TFLOPS|
|TF32 Tensor Core|~156 TFLOPS|
|FP16 Tensor Core|~312 TFLOPS|
|INT8|~624 TOPS|

## 8. 推理工程师（Inference Engineer）需要重点掌握的内容
推理工程师关注的核心目标与训练阶段不同：
> **在可接受精度下降的前提下，最大化吞吐、最小化延迟、最小化成本。**
Tensor Core 在推理阶段几乎是**必选硬件路径**，但使用方式与训练有明显差异。
### 8.1 推理工程师的 Tensor Core 全景职责
从工程视角，推理工程师需要覆盖以下层次：
```
模型结构 → 数值精度 → 算子形态 → Kernel → 硬件利用率
```
Tensor Core 贯穿其中的 **数值精度、算子实现、kernel 选择** 三层。
### 8.2 推理阶段最重要的数值体系
#### 8.2.1 FP16 / BF16 推理
- 最基础、风险最低
- 常用于中小模型或延迟敏感场景
#### 8.2.2 INT8 / INT4 量化推理
- Post-Training Quantization (PTQ)
- Quantization-Aware Training (QAT)
- Per-tensor vs Per-channel scale
- Tensor Core 在 INT8/INT4 推理中提供极高吞吐
#### 8.2.3 FP8（Hopper / LLM 推理）
- Hopper 支持 FP8（E4M3/E5M2）
- 动态 scaling 与饱和控制
- 配合 Transformer Engine / TensorRT
### 8.3 推理阶段的算子视角

|算子|Tensor Core 角色|
|---|---|
|GEMM / Batched GEMM|核心计算|
|Attention (QKV)|多个 GEMM 叠加|
|Linear / MLP|Tensor Core 主战场|
|Conv（推理）|隐式 GEMM|

### 8.4 推理阶段的关键工程技术
- Kernel 融合（epilogue fusion、attention fusion）
- Layout 与 padding 处理
- 内存与带宽优化（L2 cache, KV Cache）
### 8.5 推理工程师常用工具链

|工具|作用|
|---|---|
|TensorRT|推理图优化 + Tensor Core|
|cuBLASLt|高度可控 GEMM|
|CUTLASS|自定义推理 kernel|
|Nsight Compute|Tensor Core 利用率|
|Nsight Systems|端到端延迟|

### 8.6 推理工程师能力分级
- 初级：会用 FP16 / AMP，TensorRT，知道 Tensor Core
- 中级：掌握 INT8，理解启用条件，判断是否用上 Tensor Core
- 高级：设计 Tensor Core 友好模型，优化关键算子，延迟/吞吐/精度权衡
### 8.7 推理工程师总结
> Tensor Core 是推理性能和成本的物理上限，而不是一个可选加速器。
