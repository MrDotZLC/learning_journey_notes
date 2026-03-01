# 环境
|项目|版本|
|---|---|
|TensorRT|10.15.1|
|CUDA|12.6|
|GPU|GTX 1660 Ti (Turing, SM 7.5)|
|C++|17|
**GTX 1660 Ti 硬件特性：**
- 无 FP16 Tensor Core → FP16 加速来自显存带宽减半，非计算加速
- 有 INT8 Tensor Core → INT8 有实质性计算加速
---
# 项目结构
```plaintext
trt_practice/
├── CMakeLists.txt          # 顶层，add_subdirectory
├── common/
│   └── logger.hpp          # 共享 TRT Logger
└── resnet18/
    ├── CMakeLists.txt
    ├── resnet18.onnx
    └── src/
        ├── calibrator.hpp / calibrator.cpp   # INT8 校准器
        ├── builder.hpp    / builder.cpp       # Engine 构建
        ├── infer.hpp      / infer.cpp         # 推理 + Benchmark
        └── main.cpp
```
---
# 导出 ONNX 模型
```python
import torch
import torchvision.models as models
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy, 'resnet18.onnx',
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=17
)
```
---
# CMakeLists.txt
## 顶层
```cmake
cmake_minimum_required(VERSION 3.18)
project(trt_practice LANGUAGES CXX)

# 公共头文件路径
include_directories(${CMAKE_SOURCE_DIR}/common)

add_subdirectory(resnet18)
```
## resnet18/CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.18)
project(trt_practice LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_path(TRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/x86_64-linux-gnu /usr/local/include
    REQUIRED
)

find_path(CUDA_INCLUDE_DIR cuda_runtime.h
    PATHS /usr/local/cuda/include
    REQUIRED
)

include_directories(
    ${TRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

find_library(TRT_LIB     nvinfer        PATHS /usr/lib/x86_64-linux-gnu  REQUIRED)
find_library(TRT_PLUGIN  nvinfer_plugin PATHS /usr/lib/x86_64-linux-gnu  REQUIRED)
find_library(TRT_ONNX    nvonnxparser   PATHS /usr/lib/x86_64-linux-gnu  REQUIRED)
find_library(CUDA_RT_LIB cudart         PATHS /usr/local/cuda/lib64      REQUIRED)

file(GLOB SRCS src/*.cpp)
add_executable(trt_resnet18 ${SRCS})

target_link_libraries(trt_resnet18
    ${TRT_LIB}
    ${TRT_PLUGIN}
    ${TRT_ONNX}
    ${CUDA_RT_LIB}
)

# 将源码目录路径编译进可执行文件，避免路径硬编码
target_compile_definitions(trt_resnet18 PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_options(trt_resnet18 PRIVATE -Wno-deprecated-declarations)
```
---
# 核心概念
## TRT 构建阶段对象链
```plaintext
IBuilder
  ├── INetworkDefinition   （从 ONNX 解析的网络结构）
  ├── IBuilderConfig       （精度、workspace、profile）
  │     └── IOptimizationProfile  （动态 shape min/opt/max）
  └── buildSerializedNetwork()
        ├── Layer fusion（Conv+BN+ReLU → 单 kernel）
        ├── Kernel auto-tuning（枚举候选 CUDA kernel，选最快）
        └── INT8 Calibration（统计激活分布，确定 scale）
```
## TRT 推理阶段对象链
```plaintext
IRuntime
  └── ICudaEngine          （反序列化 engine，加载编译好的 kernel）
        └── IExecutionContext  （推理状态，动态 shape 绑定）
```
## 动态 Shape Profile
```cpp
// 必须声明 min/opt/max，TRT 针对 opt 选择最优 kernel
profile->setDimensions("input", kMIN, Dims4{1,  3, 224, 224});
profile->setDimensions("input", kOPT, Dims4{8,  3, 224, 224});
profile->setDimensions("input", kMAX, Dims4{16, 3, 224, 224});
```
## 推理异步流程
```plaintext
H2D（cudaMemcpyAsync）→ enqueueV3 → D2H（cudaMemcpyAsync）→ cudaStreamSynchronize
```
同一 stream 内顺序执行，保证依赖关系正确。

---
# INT8 量化原理
**映射公式：**
```plaintext
x_int8 = clamp( round(x_fp32 / scale), -128, 127 )
```
**Calibration 流程：**
1. TRT 调用 `getBatch()` 获取真实数据（GPU 指针）
2. TRT 对每层激活值统计分布
3. 用 KL 散度最小化（EntropyCalibrator2）确定各层 scale
4. scale 写入 cache 文件，下次跳过校准
**注意：** TRT 10.15 将隐式量化（`kINT8` flag + Calibrator）标记为废弃，推荐迁移至 Q/DQ 显式量化。当前阶段仍可用。
---
# 关键代码
## common/logger.hpp
```cpp
#pragma once
#include <NvInfer.h>
#include <iostream>
class Logger : public nvinfer1::ILogger {
public:
    explicit Logger(Severity minSeverity = Severity::kWARNING)
        : mMinSeverity(minSeverity) {}
    void log(Severity severity, const char* msg) noexcept override {
        if (severity > mMinSeverity) return;
        switch (severity) {
        case Severity::kINTERNAL_ERROR: std::cerr << "[TRT INTERNAL_ERROR] "; break;
        case Severity::kERROR:          std::cerr << "[TRT ERROR]          "; break;
        case Severity::kWARNING:        std::cerr << "[TRT WARNING]        "; break;
        case Severity::kINFO:           std::cout << "[TRT INFO]           "; break;
        case Severity::kVERBOSE:        std::cout << "[TRT VERBOSE]        "; break;
        }
        std::cout << msg << "\n";
    }
private:
    Severity mMinSeverity;
};
```
## src/builder.cpp（精度设置核心片段）
```cpp
switch (precision) {
case Precision::FP16:
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    break;
case Precision::INT8:
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);  // fallback：不支持 INT8 的层退回 FP16
    calibrator = std::make_unique<Int8Calibrator>(8, 3, 224, 224, "calib_cache.bin");
    config->setInt8Calibrator(calibrator.get());
    break;
case Precision::FP32:
default:
    break;
}
```
## src/infer.cpp（推理核心片段）
```cpp
// 每次推理前必须重新设置（context 不缓存）
m_context->setInputShape("input", nvinfer1::Dims4{batchSize, k_C, k_H, k_W});
m_context->setTensorAddress("input",  m_device_input);
m_context->setTensorAddress("output", m_device_output);
cudaMemcpyAsync(m_device_input, inputHost.data(), inputBytes,
                cudaMemcpyHostToDevice, m_stream);         // H2D
m_context->enqueueV3(m_stream);                            // 推理（异步）
cudaMemcpyAsync(outputHost.data(), m_device_output, outputBytes,
                cudaMemcpyDeviceToHost, m_stream);          // D2H，注意方向
cudaStreamSynchronize(m_stream);                           // 等待完成
```
## src/infer.cpp（CUDA Event 计时片段）
```cpp
cudaEvent_t ev_start, ev_stop;
cudaEventCreate(&ev_start);
cudaEventCreate(&ev_stop);
cudaEventRecord(ev_start, m_stream);
m_context->enqueueV3(m_stream);
cudaEventRecord(ev_stop, m_stream);
cudaEventSynchronize(ev_stop);
float ms;
cudaEventElapsedTime(&ms, ev_start, ev_stop);  // GPU 时间，单位 ms
```
---
# 实验结果（batch=8，GTX 1660 Ti）
|精度|Engine大小|mean延迟|p50|p99|吞吐量|cosine_sim|max_abs_diff|
|---|---|---|---|---|---|---|---|
|FP32|51 MB|8.74 ms|8.69 ms|10.67 ms|915 img/s|基准|基准|
|FP16|37 MB|5.47 ms|5.23 ms|7.05 ms|1463 img/s|0.999996|0.020|
|INT8|11 MB|4.65 ms|4.42 ms|6.17 ms|1719 img/s|0.995874|1.125|
**加速比：** FP16 = 1.6x，INT8 = 1.9x（相对 FP32）
**INT8 精度说明：** 使用随机数据 Calibrator，max_abs_diff=1.125 偏高。生产环境用真实 ImageNet 数据（≥500张）校准后 cosine_sim 通常可达 0.999+。

---
# TRT API 版本差异
|API|TRT 8.x|TRT 10.x|
|---|---|---|
|推理执行|`enqueueV2(bindings[], stream, nullptr)`|`enqueueV3(stream)`|
|绑定地址|`bindings[]` 数组|`setTensorAddress(name, ptr)`|
|设置输入 shape|`setBindingDimensions(idx, dims)`|`setInputShape(name, dims)`|
|序列化|`buildEngineWithConfig()` → `serialize()`|`buildSerializedNetwork()`|
|FP16/INT8|`BuilderFlag::kFP16/kINT8`（已废弃）|Q/DQ 显式量化（推荐，待迁移）|
