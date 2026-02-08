# 一、CUTLASS 是什么
NVIDIA 开源的一个 **基于 C++ 模板的 CUDA 线性代数库**，一个用于**构建自定义、高性能 CUDA 内核的模板化元编程框架**。
[Cutlass+vscode 环境搭建](Cutlass+vscode%20环境搭建.md)

---
# 二、设计目标
CUTLASS 的核心目标可以概括为四点：
1. **性能接近手写汇编级内核**
    - 充分利用 Tensor Core
    - 精确控制 warp / thread / memory 访问
2. **高度可组合**
    - 通过模板参数组合不同数据类型、tile 大小、指令路径
3. **架构可移植**
    - 支持多代 NVIDIA GPU（Volta / Turing / Ampere / Hopper）
4. **作为 cuBLAS 的“参考实现与实验平台”**
    - 许多 cuBLAS 内核思想最早来自 CUTLASS

---
# 三、整体架构概览
CUTLASS 采用**分层抽象架构**，从硬件指令逐层向上封装：
```
┌──────────────────────────┐
│   Device-level GEMM      │  ← 用户最终调用
├──────────────────────────┤
│   Kernel-level GEMM      │
├──────────────────────────┤
│   Threadblock-level GEMM │
├──────────────────────────┤
│   Warp-level GEMM        │
├──────────────────────────┤
│   MMA / SIMT instruction │  ← Tensor Core / FMA
└──────────────────────────┘
```
每一层都通过 **C++ 模板参数** 精确描述计算形态。

---
# 四、核心思想：模板元编程 + Tile 化计算
## 1️⃣ Tile（分块）是核心
CUTLASS 的一切都围绕 **Tile-based GEMM**：

|层级|Tile|
|---|---|
|Threadblock|TB_M × TB_N × TB_K|
|Warp|W_M × W_N × W_K|
|Instruction|MMA_M × MMA_N × MMA_K|
每一层的 tile 尺寸都由模板参数静态确定。

---
## 2️⃣ 模板参数极其丰富
一个 GEMM Kernel 的模板参数通常包括：
- 数据类型（A/B/C）
- 布局（RowMajor / ColumnMajor）
- 计算类型（accumulator type）
- Tile 尺寸（threadblock / warp / instruction）
- MMA 运算类型（Tensor Core / SIMT）
- Pipeline stage 数量
- Epilogue（输出阶段）

这也是 CUTLASS **学习曲线陡峭**的根本原因。

---
# 五、主要模块详解
### 1. `cutlass::gemm`
**最核心模块**，用于 GEMM：
- `device::Gemm`（用户层）
- `kernel::Gemm`
- `threadblock::Mma`
- `warp::Mma`
- `arch::Mma`

典型层级调用路径：
```
device::Gemm
  → kernel::Gemm
    → threadblock::Mma
      → warp::Mma
        → arch::Mma (wmma / mma.sync)
```
## 2. `cutlass::arch`
描述底层硬件指令：
- `OpClassTensorOp`
- `Sm80`, `Sm90`
- `mma.sync.aligned.m16n8k16`

这是 **Tensor Core 的抽象层**。

---
### 3. `cutlass::layout`
描述矩阵存储方式：
- `RowMajor`
- `ColumnMajor`
- `TensorNHWC`
- `TensorNCxHWx`

用于泛化 GEMM 到卷积、Transformer 等场景。

---
### 4. `cutlass::epilogue`
**输出阶段（极其重要）**
在 GEMM 完成后执行：
- bias
- scale
- activation（ReLU / GELU）
- 类型转换（fp32 → fp16 / int8）
    
例如：
```
C = alpha * Acc + beta * C
```
Epilogue 是 CUTLASS 可扩展性最强的部分之一。

---
### 5. `cutlass::conv`
将卷积转换为 **Implicit GEMM**：
- 前向卷积
- 反向数据
- 反向权重

广泛用于深度学习框架。

---
# 六、典型使用方式
## 1️⃣ 直接使用 device::Gemm（最简单）
```cpp
using Gemm = cutlass::gemm::device::Gemm<
    half, RowMajor,
    half, RowMajor,
    half, RowMajor
>;

Gemm gemm_op;
gemm_op(arguments);
```
优点：
- 上手快  
    缺点：
- 参数固定、可控性有限

---
## 2️⃣ 自定义 Kernel（高级用法）
需要你指定：
- Threadblock tile
- Warp tile
- MMA instruction
- Pipeline stages

这是 **性能调优工程师的主要工作模式**。

---
# 七、源码浅析
[Cutlass 源码浅析](Cutlass%20源码浅析.md)

# 八、性能特征
### 优势
- 性能可逼近 cuBLAS
- Tensor Core 利用率极高
- 静态展开、零运行时分支
- 可用于极端定制场景（如 Transformer）

### 代价
- 编译时间长
- 二进制体积大
- 调试困难
- 学习曲线陡峭

---
# 九、CUTLASS vs cuBLAS

|维度|CUTLASS|cuBLAS|
|---|---|---|
|使用难度|高|低|
|可定制性|极高|很低|
|性能|接近/等同|官方最优|
|使用场景|内核开发、研究|工程直接调用|
一句话总结：

> **cuBLAS 是“成品”，CUTLASS 是“机床”**

---

