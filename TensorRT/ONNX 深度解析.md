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
# 1. ONNX 核心架构
## 1.1 本质定义
**ONNX (Open Neural Network Exchange)** 是跨框架模型的中间表示（Intermediate Representation, IR）标准。
工程抽象：
```
Framework Graph → ONNX IR → Runtime / Accelerator Execution
```
核心目标：
- 跨框架互操作
- 静态图优化基础
- 推理引擎统一入口
## 1.2 计算图数学模型
ONNX 模型抽象为有向无环图（DAG）：
$$  
\mathcal{G} = (\mathcal{V}, \mathcal{E})  
$$

|符号|含义|
|---|---|
|$\mathcal{V}$|算子节点（Operator）|
|$\mathcal{E}$|张量数据依赖|
|$\mathcal{W}$|Initializer（静态权重）|
执行模型：数据驱动调度（Dataflow Execution Model）
## 1.3 IR Version vs Opset Version
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
## 1.4 Protobuf 层级结构
|层级|类型|关键接口|
|---|---|---|
|模型|ModelProto|model.graph()|
|图|GraphProto|graph.node(i)|
|节点|NodeProto|node.op_type()|
|张量|TensorProto|tensor.raw_data()|
***
# 2. Opset 版本管理
## 2.1 版本兼容约束
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
## 2.2 关键版本节点
|Opset|重要变更|
|---|---|
|13|Softmax 轴语义修正|
|17|LayerNormalization 标准化|
|19|Quantization 扩展增强|
|21|引入 float8e4m3fn / float8e5m2 类型|
Opset 支持数据类型 ≠ Runtime 自动支持对应 Kernel。
***
# 3. 图优化：Conv–BN 融合
## 3.1 融合前提
- BN 处于 inference mode
- running_mean / running_var 为常量
- γ / β 为 initializer
- Conv 权重静态化
## 3.2 数学推导
原始计算：
$$  
Y = W X + b  
$$
$$  
Z = \gamma \frac{Y - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta  
$$
代入：
$$  
Z = \gamma \frac{(W X + b) - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta  
$$
重排：
$$  
W' = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} W  
$$
$$  
b' = \frac{\gamma (b - \mu)}{\sqrt{\sigma^2 + \epsilon}} + \beta  
$$
推理图化简：
```
Conv + BN → Conv(W', b')
```
***
# 4. TensorRT 引擎构建
## 4.1 构建流程
```
IBuilder
  → INetworkDefinition (kEXPLICIT_BATCH)
  → ONNX Parser
  → IBuilderConfig
  → buildSerializedNetwork
  → IRuntime.deserializeCudaEngine
```
## 4.2 Dynamic Shape Profile（必须）
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
## 4.3 构建失败常见原因
|原因|现象|
|---|---|
|未设置 Profile|Build 失败|
|Workspace 不足|Kernel 选择失败|
|Plugin 未注册|Fallback / Crash|
|INT8 无校准|精度异常|
***
# 5. 精度策略与硬件依赖
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
# 6. Custom Plugin 开发要点
## 必须实现接口
- getOutputDimensions
- supportsFormatCombination
- configurePlugin
- enqueue
- getSerializationSize
- serialize
- clone
缺失任一接口 → 构建阶段失败。
## ONNX 映射约束
```
NodeProto.op_type == Plugin.getPluginType()
```
***
# 7. LLM 推理优化前沿
## 7.1 FP8 推理现实约束
- 权重需 Scaling Factors
- LayerNorm / Softmax 多保持 FP16
- Kernel 覆盖率依赖 TensorRT 版本
## 7.2 KV Cache 优化
|策略|目的|
|---|---|
|PagedAttention|减少显存碎片|
|连续布局|降低 TLB Miss|
|cudaMallocAsync|减少同步阻塞|
## 7.3 Zero-Copy / RDMA
核心机制：
- Pinned Memory（`cudaHostRegister`）
- GPUDirect RDMA
- 多节点通信优化
***
# 8. 知识图谱结构
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
