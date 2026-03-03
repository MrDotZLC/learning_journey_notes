tags:
  - onnx
  - inference-ir
  - tensorrt
  - graph-optimization
  - cuda
  - quantization
  - custom-plugin
created: 2025-01
aliases:
  - ONNX推理
  - ONNX IR
  - TensorRT Engine
  - Conv-BN融合
  - 推理部署
面向：C++ 后端 → 推理部署 / 推理优化工程师  
知识前置：GPU Architecture，CUDA C++，Tensor Core，Transformer

***

## 1. ONNX 核心架构
### 1.1 本质定义
**ONNX (Open Neural Network Exchange)** 是跨框架模型的中间表示（Intermediate Representation, IR）标准。
工程抽象：
```
Framework Graph → ONNX IR → Runtime / Accelerator Execution
```
核心目标：
- 跨框架互操作
- 静态图优化基础
- 推理引擎统一入口
### 1.2 计算图数学模型
ONNX 模型抽象为有向无环图（DAG）：
$$  
\mathcal{G} = (\mathcal{V}, \mathcal{E})  
$$

| 符号            | 含义                |
| ------------- | ----------------- |
| $\mathcal{V}$ | 算子节点（Operator）    |
| $\mathcal{E}$ | 张量数据依赖            |
| $\mathcal{W}$ | Initializer（静态权重） |

执行模型：数据驱动调度（Dataflow Execution Model）
### 1.3 IR Version vs Opset Version

|概念|作用|
|---|---|
|IR Version|Protobuf Schema 结构版本|
|Opset Version|算子语义版本|

兼容条件：
$$  
IR_{model} \le IR_{runtime}  
\quad \land \quad  
Opset_{model} \le Opset_{runtime}  
$$
两者独立演进，不可混淆。
### 1.4 Protobuf 层级结构

|层级|类型|关键接口|
|---|---|---|
|模型|ModelProto|model.graph()|
|图|GraphProto|graph.node(i)|
|节点|NodeProto|node.op_type()|
|张量|TensorProto|tensor.raw_data()|

***

## 2. Opset 版本管理
### 2.1 版本兼容约束
设：
- $V_{exp}$：导出版本
- $V_{eng}$：运行时支持版本
兼容条件：
$$  
V_{exp} \le V_{eng}  
$$
否则：
```
Parser Failure
Unsupported Attribute
Kernel Not Found
```
### 2.2 关键版本节点

|Opset|重要变更|
|---|---|
|13|Softmax 轴语义修正|
|17|LayerNormalization 标准化|
|19|Quantization 扩展增强|
|21|引入 float8e4m3fn / float8e5m2 类型|

Opset 支持数据类型 ≠ Runtime 自动支持对应 Kernel。

***

## 3. 图优化：Conv–BN 融合（深入版）
### 3.1 图优化背景
在推理阶段，神经网络执行成本主要来自两类资源：

|资源|瓶颈来源|
|---|---|
|计算（Compute）|Tensor Core / CUDA Core|
|内存带宽（Memory Bandwidth）|Global Memory 读写|

对于大多数 CNN / Transformer 层：
> **显存带宽往往比算力更早成为瓶颈**

典型执行流程：
```
Conv Kernel
    ↓ (写回显存)
BatchNorm Kernel
```
两次 Global Memory 往返：
```
Conv output → global memory → BN input
```
优化目标：
```
Conv + BN → 单算子
```
减少：
- Kernel Launch
- Global Memory Read / Write
- CUDA Stream 同步
### 3.2 Conv 与 BatchNorm 数学模型
#### Conv
设：
- 输入张量
$$  
X \in \mathbb{R}^{N \times C_{in} \times H \times W}  
$$
- 卷积核
$$  
W \in \mathbb{R}^{C_{out} \times C_{in} \times K_h \times K_w}  
$$
输出：
$$  
Y = W * X + b  
$$
其中：
- $*$ 表示卷积
- $b \in \mathbb{R}^{C_{out}}$
#### BatchNorm（推理阶段）
BN 对 **每个通道独立归一化**
$$  
Z = \gamma \cdot \frac{Y - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta  
$$
参数：

|符号|含义|
|---|---|
|$\mu$|running mean|
|$\sigma^2$|running variance|
|$\gamma$|scale|
|$\beta$|bias|
|$\epsilon$|数值稳定项|

这些值在推理阶段 **全部为常量**。
### 3.3 融合推导
原始表达：
$$  
Y = W X + b  
$$
BN：
$$  
Z = \gamma \frac{Y - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta  
$$
代入：
$$  
Z = \gamma \frac{(WX + b) - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta  
$$
展开：
$$  
Z = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} WX  
+  
\frac{\gamma (b - \mu)}{\sqrt{\sigma^2 + \epsilon}}  
+  
\beta  
$$
重新整理为卷积形式：
$$  
Z = W'X + b'  
$$
其中
$$  
W' = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} W  
$$
$$  
b' =  
\frac{\gamma (b - \mu)}{\sqrt{\sigma^2 + \epsilon}} + \beta  
$$
### 3.4 向量化表达（实际实现）
BN 参数是 **逐通道 scaling**。
定义：
$$  
\alpha_c = \frac{\gamma_c}{\sqrt{\sigma_c^2 + \epsilon}}  
$$
则
```
for each output channel c:
W'[c,:,:,:] = α_c * W[c,:,:,:]
b'[c] = α_c * (b[c] - μ_c) + β_c
```
这一步在 **图优化阶段一次性完成**。
### 3.5 计算图变换
原始 ONNX 计算图：
```
 X
 │
Conv
 │
BatchNorm
 │
 Z
```
优化后：
```
 X
 │
Conv(W', b')
 │
 Z
```
BatchNorm 节点被 **删除**。
### 3.6 ONNX Graph Transform 示例
原始节点：
```
node {
  op_type: "Conv"
}
node {
  op_type: "BatchNormalization"
}
```
优化后：
```
node {
  op_type: "Conv"
  weight = W'
  bias   = b'
}
```
### 3.7 C++ 实现示例（权重融合）
```cpp
void fuse_conv_bn(
    std::vector<float>& W,
    std::vector<float>& b,
    const std::vector<float>& gamma,
    const std::vector<float>& beta,
    const std::vector<float>& mean,
    const std::vector<float>& var,
    float eps,
    int Cout,
    int kernel_size)
{
    for (int c = 0; c < Cout; ++c)
    {
        float alpha = gamma[c] / std::sqrt(var[c] + eps);
        for (int k = 0; k < kernel_size; ++k)
        {
            W[c * kernel_size + k] *= alpha;
        }
        b[c] = alpha * (b[c] - mean[c]) + beta[c];
    }
}
```
复杂度：
```
O(Cout × KernelSize)
```
仅执行 **一次**。
### 3.8 性能收益分析
设：
```
FeatureMap Size = N × C × H × W
```
BatchNorm 需要：
```
read(Y)  + write(Z)
```
ConvBN 融合后：
```
只写一次
```
显存访问减少：
```
≈ 2 × FeatureMap Size
```
在 GPU 上：
> Memory Bandwidth Reduction → 10%–30% 推理速度提升
### 3.9 融合限制条件
必须满足：

|条件|原因|
|---|---|
|BN 为 inference mode|训练 BN 不可融合|
|参数为常量 initializer|必须可提前计算|
|Conv 权重静态|动态权重无法融合|
|无中间节点|Conv → BN 必须直接相连|

不满足示例：
```
Conv
 │
Relu
 │
BatchNorm
```
无法融合。
### 3.10 推理引擎实现方式
主流推理框架均自动执行该优化：

|框架|实现位置|
|---|---|
|ONNX Runtime|Graph Optimizer|
|TensorRT|Network Fusion Pass|
|OpenVINO|Graph Rewrite Pass|

融合发生阶段：
```
ONNX Load
   ↓
Graph Optimization
   ↓
Kernel Selection
```
### 3.11 推理优化中的地位
Conv–BN Fusion 属于：
```
Operator Fusion
```
同类优化：

|优化|目的|
|---|---|
|Conv + BN|减少 Memory IO|
|Conv + ReLU|减少 Kernel Launch|
|MatMul + Bias|减少 Tensor Load|
|Attention Fusion|减少中间张量|

本质：
> **将算子级图优化转换为 Kernel 级执行优化**
### 3.12 Transformer 中的类似优化
在 Transformer 推理中：
```
LayerNorm + Linear
```
可等价变换为：
```
Scaled Linear
```
以及：
```
QKV Linear Fusion
```
```
3 × GEMM → 1 × GEMM
```
### 3.13 与 Constant Folding 的关系
ConvBN 融合依赖：
```
Constant Folding
```
原因：
BN 参数必须为：
```
Initializer Tensor
```
否则无法提前计算：
```
W' , b'
```
### 3.14 总结
Conv–BN 融合本质：
```
图变换 + 权重重参数化
```
优化效果：

|指标|变化|
|---|---|
|Kernel 数量|减少|
|显存访问|减少|
|计算量|不变|
|推理延迟|降低|

核心思想：
> 将 **运行时计算** 转移到 **模型构建阶段**。

***

## 4. TensorRT 引擎构建
### 4.1 构建流程
```
IBuilder
  → INetworkDefinition (kEXPLICIT_BATCH)
  → ONNX Parser
  → IBuilderConfig
  → buildSerializedNetwork
  → IRuntime.deserializeCudaEngine
```
### 4.2 Dynamic Shape Profile（必须）
```cpp
auto profile = builder->createOptimizationProfile();
profile->setDimensions("input",
    nvinfer1::OptProfileSelector::kMIN,
    Dims4(1,3,224,224));
profile->setDimensions("input",
    nvinfer1::OptProfileSelector::kOPT,
    Dims4(8,3,224,224));
profile->setDimensions("input",
    nvinfer1::OptProfileSelector::kMAX,
    Dims4(32,3,224,224));
config->addOptimizationProfile(profile);
```
### 4.3 构建失败常见原因

|原因|现象|
|---|---|
|未设置 Profile|Build 失败|
|Workspace 不足|Kernel 选择失败|
|Plugin 未注册|Fallback / Crash|
|INT8 无校准|精度异常|

***

## 5. 精度策略与硬件依赖

|精度|硬件依赖|特点|
|---|---|---|
|FP32|全平台|基准精度|
|FP16|Volta+|Tensor Core 加速|
|INT8|Turing+|需 Calibration|
|FP8|Hopper+ / Blackwell|Transformer Engine|

主流推理 GPU 生态由 **NVIDIA** 主导。
精度选择需考虑：
- Tensor Core 对齐约束
- Reformat Kernel 开销
- 数值敏感层（Softmax / LayerNorm）

***

## 6. Custom Plugin 开发要点
### 必须实现接口
- getOutputDimensions
- supportsFormatCombination
- configurePlugin
- enqueue
- getSerializationSize
- serialize
- clone
缺失任一接口 → 构建阶段失败。
### ONNX 映射约束
```
NodeProto.op_type == Plugin.getPluginType()
```

***

## 7. LLM 推理优化前沿
### 7.1 FP8 推理现实约束
- 权重需 Scaling Factors
- LayerNorm / Softmax 多保持 FP16
- Kernel 覆盖率依赖 TensorRT 版本
### 7.2 KV Cache 优化

|策略|目的|
|---|---|
|PagedAttention|减少显存碎片|
|连续布局|降低 TLB Miss|
|cudaMallocAsync|减少同步阻塞|

### 7.3 Zero-Copy / RDMA
核心机制：
- Pinned Memory（`cudaHostRegister`）
- GPUDirect RDMA
- 多节点通信优化

***

## 8. 知识图谱结构
```
ONNX (IR)
 ├── Protobuf Schema
 ├── Opset
 ├── Graph Optimization
 │     ├── Constant Folding
 │     ├── Conv-BN Fusion
 │     └── Quantization
 └── Runtime Mapping
        └── TensorRT
```
