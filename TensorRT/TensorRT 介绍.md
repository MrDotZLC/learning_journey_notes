# 一、TensorRT 的定位
**一句话概括**：
 > TensorRT 是一个- 面向推理阶段的编译器 + 运行时（compiler + runtime）*。
 它**不负责训练**，而是：
- 接收来自- *PyTorch / TensorFlow / ONNX** 等框架的模型
* 对模型做- *图级、算子级、数值级优化**
* 生成一个高度优化、与硬件强绑定的- *推理引擎（Engine）**
* 在部署时以极低开销执行
 
---
 
# 二、TensorRT 的整体架构
```
训练框架
(PyTorch / TF)
 ↓
模型导出
(ONNX / UFF / API)
 ↓
TensorRT Builder
- 图优化
- 精度校准
- Kernel 选择
 ↓
TensorRT Engine
(序列化)
 ↓
TensorRT Runtime
(GPU / DLA 上执行)
```

 **TensorRT 主要分为两大阶段**：
## 1️⃣ Build（构建/编译阶段）
- 一次性，耗时较长
* 与 GPU 架构、精度强相关
* 生成 Engine
## 2️⃣ Runtime（运行阶段）
- 高频调用
* 极低延迟
* 几乎只做 kernel 调度和数据搬运
 ---
# 三、核心组件详解
## 1. Builder（构建器）
 负责把网络**编译成最优推理引擎**：
- 解析网络结构
* 选择最优 kernel（cuDNN / cuBLAS / 自定义）
* 融合算子
* 搜索最优执行计划（tactic search）
 关键参数：
- 最大 batch size
* 精度模式（FP32 / FP16 / INT8）
* Workspace size（用于 kernel 搜索）
 ---
## 2. Network Definition（网络定义）
 TensorRT 内部的计算图表示：
- Layer-based（Conv / FC / Activation / Plugin）
* 静态或动态 shape
* 支持子图融合
 ---
## 3. Engine（推理引擎）
 TensorRT 的**核心产物**：
- 已优化好的、不可再修改的计算图
* 与**GPU 架构 + 精度 + batch/shape** 强绑定
* 可序列化为`.engine` 文件
 > Engine ≈ 已编译好的“二进制程序”
 ---
## 4. Runtime（运行时）
- 加载 Engine
* 接收输入 Tensor
* 在 CUDA Stream 中执行
* 输出结果
 ---
## 5. Plugin（插件机制）
 用于扩展 TensorRT 不支持的算子：
- 自定义 CUDA kernel
* 支持反序列化
* 工业级部署中非常常见（如 Swish、Deformable Conv）
 ---
# 四、TensorRT 的核心优化技术
## 1️⃣ 计算图级优化（Graph Optimization）
[TensorRT 与 TensorRT-LLM 图优化与层融合底层机制](TensorRT%20与%20TensorRT-LLM%20图优化与层融合底层机制.md)
- **算子融合（Layer Fusion）**
* Conv + Bias + ReLU → 单 kernel
* 消除冗余节点
* 常量折叠（Constant Folding）
 📈 减少 kernel launch 次数，显著降低延迟
 ---
## 2️⃣ Kernel 自动选择（Tactic Selection）
* 为每个算子测试多种 CUDA 实现
* 根据实际硬件 benchmark
* 选择最优 kernel 组合
 📌 构建慢，但运行极快
 ---
## 3️⃣ 精度优化（Precision Optimization）

| 精度   | 特点        | 使用场景         |
| ---- | --------- | ------------ |
| FP32 | 精度最高      | 调试 / 高精度     |
| FP16 | 速度快、精度损失小 | 主流推理         |
| INT8 | 极致性能      | 边缘设备 / 大规模部署 |
### INT8 校准（Calibration）
[TensorRT INT8 Calibration 原理](../量化与AMP/TensorRT%20INT8%20Calibration%20原理.md)
* 使用少量代表性数据
* 统计激活值分布
* 自动计算 scale / zero-point
 ---
## 4️⃣ 内存与带宽优化
* Tensor 内存复用
* 最小化 Host ↔ Device 拷贝
* 高效使用 Tensor Core
 ---
## 5️⃣ 动态 Shape 支持
- 通过**Optimization Profile**
* 兼顾灵活性与性能
* 广泛用于 NLP / CV 多分辨率输入
 ---
# 五、TensorRT 支持的输入来源
## 1. ONNX（最推荐）
[ONNX 深度解析](ONNX%20深度解析.md)
* 框架无关
* 支持最好
* 工业部署首选
## 2. PyTorch
* `torch2trt`
- `torch.compile + TensorRT`
* PyTorch → ONNX → TensorRT（最稳定）
## 3. TensorFlow
- SavedModel → TensorRT（TF-TRT）
* 现多用于 legacy 项目
 ---
# 六、典型工作流程（以 PyTorch 为例）
```text
1. PyTorch 训练模型
2. 导出 ONNX
3. TensorRT Builder 构建 Engine
4. 序列化 Engine
5. 部署时加载 Engine
6. Runtime 执行推理
```
 ---
# 七、TensorRT 的优势
 ✅**极致性能**
-  通常比原生 PyTorch 推理快 2–10 倍
* 延迟敏感场景优势明显
 ✅**工业级稳定性**
-  NVIDIA 官方维护
* 大规模生产验证
 ✅**硬件深度绑定**
-  Tensor Core
* DLA（Jetson）
 ---
# 八、局限与挑战
 ⚠️**调试困难**
-  Engine 是黑盒
* 错误多发生在 build 阶段
 ⚠️**算子支持不全**
-  新模型（如部分 Transformer 算子）需 Plugin
 ⚠️**平台强绑定**
-  Engine 不同 GPU 通常不可复用
 ⚠️**学习曲线陡**
-  C++ API 复杂
* 内存管理要求高
 ---
# 九、典型应用场景
- 🚗 自动驾驶（感知 / 规划推理）
* 📷 实时视觉（检测 / 分割 / OCR）
* 🧠 NLP 在线推理（BERT / GPT 推理）
* 🤖 机器人 & Jetson 边缘设备
* ☁️ 云端高并发推理服务
 ---
# 十、总结一句话
 >**TensorRT 本质是一个为 NVIDIA 硬件量身定制的深度学习推理编译器，通过图优化、算子融合和精度压缩，把训练好的模型“压榨”到极致性能。**
