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

add_subdirectory(0_resnet18_onnx)
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
## src/builder.hpp
```c++
#pragma once
#include <NvInfer.h>
#include <string>
#include "logger.hpp"

/**
 * 推理精度枚举
 *
 * FP32：全精度浮点，基准精度，无损失
 * FP16：半精度浮点，GTX 1660 Ti 无 FP16 Tensor Core，加速不明显
 * INT8：8位整数，GTX 1660 Ti 有 INT8 Tensor Core，有实质加速，精度略有损失
 */
enum class Precision { FP32, FP16, INT8 };

/**
 * 将 Precision 枚举转为字符串，用于日志输出
 */

inline const char *precisionStr(Precision p) {
    switch (p) {
        case Precision::FP32:
            return "FP32";
        case Precision::FP16:
            return "FP16";
        case Precision::INT8:
            return "INT8";
    }
    return "UNKNOWN";
}

/**
 * 从 ONNX 文件构建 TensorRT Engine 并序列化到磁盘
 *
 * 构建过程（发生在此函数内部）：
 *   1. IBuilder       解析 ONNX，创建网络定义
 *   2. IBuilderConfig 设置精度、workspace、动态 shape profile
 *   3. buildSerializedNetwork()
 *        ├── Layer fusion（算子融合）：如 Conv+BN+ReLU 合并为一个 kernel
 *        ├── Kernel auto-tuning：对每个算子枚举候选 CUDA kernel，选最快的
 *        └── INT8 模式下执行 Calibration，确定每层 scale
 *   4. 序列化为二进制写入磁盘（.engine 文件）
 *
 * 注意：
 *   - engine 与 GPU 架构绑定（SM 7.5 编译的不能在 SM 8.0 运行）
 *   - 构建耗时较长（数秒到数分钟），推理时直接加载 .engine 文件
 *
 * @param onnxPath    输入 ONNX 模型路径
 * @param enginePath  输出 engine 文件路径
 * @param precision   推理精度
 * @param logger      TRT Logger 实例（Builder 和 Runtime 共用）
 */

void buildEngine(const std::string& onnxPath,
                 const std::string& enginePath,
                 Precision precision,
                 Logger& logger);
```
## src/builder.cpp
```cpp
#include "builder.hpp"
#include "calibrator.hpp"
#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

void buildEngine(const std::string &onnxPath, const std::string &enginePath,
                 Precision precision, Logger &logger) {
    std::cout << "\n[Builder] onnx=" << onnxPath
              << "  precision=" << precisionStr(precision) << "\n";
    // ── 1. 创建 Builder
    // ───────────────────────────────────────────────────────
    //
    // IBuilder 是构建阶段的总入口。
    // 所有构建对象（network、config、profile）都由它创建。
    // createInferBuilder 需要传入 logger，TRT 内部消息通过它回调输出。
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger));
    if (!builder) {
        throw std::runtime_error("[Builder] createInferBuilder failed");
    }

    // ── 2. 创建网络定义
    // ───────────────────────────────────────────────────────
    //
    // INetworkDefinition 保存从 ONNX 解析出的网络结构（层、张量、连接关系）。
    //
    // kEXPLICIT_BATCH：
    //   显式 batch 模式，batch 维度作为正常维度参与 shape 推导。
    //   TRT 10.x 只支持此模式，旧版 kIMPLICIT_BATCH 已废弃。

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(0U));
    if (!network) {
        throw std::runtime_error("[Builder] createNetworkV2 failed");
    }

    // ── 3. 解析 ONNX
    // ──────────────────────────────────────────────────────────
    //
    // IParser 读取 ONNX 文件，将其中的算子逐一转换为 TRT 的 ILayer，
    // 填充到 network 中。
    // 支持的 ONNX opset 版本取决于 TRT 版本，TRT 10.x 支持到 opset 20。
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger));
    if (!parser) {
        throw std::runtime_error("[Builder] createParser failed");
    }
    if (!parser->parseFromFile(
            onnxPath.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << "[Builder] Parse error: "
                      << parser->getError(i)->desc() << "\n";
        }
        throw std::runtime_error("[Builder] ONNX parse failed");
    }
    std::cout << "[Builder] ONNX parsed OK\n";
    std::cout << "[Builder] Network inputs : " << network->getNbInputs()
              << "\n";
    std::cout << "[Builder] Network outputs: " << network->getNbOutputs()
              << "\n";

    // ── 4. 构建配置
    // ───────────────────────────────────────────────────────────
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    if (!config) {
        throw std::runtime_error("[Builder] createBuilderConfig failed");
    }
    // Workspace（工作区）：
    //   TRT kernel auto-tuning 时需要临时显存来测试候选 kernel。
    //   设置过小会导致部分 kernel 无法测试，影响性能选择。
    //   1 GiB 对 ResNet18 足够。
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1UL << 30);

    // ── 5. 动态 Shape Profile
    // ─────────────────────────────────────────────────
    //
    // Profile 告知 TRT 输入 tensor 的形状范围：
    //   kMIN：最小 batch，TRT 保证此 shape 可正确运行
    //   kOPT：最优 batch，TRT 针对此 shape 选择最快 kernel（最重要）
    //   kMAX：最大 batch，TRT 保证此 shape 可正确运行
    //
    // 推理时 batch 大小必须在 [kMIN, kMAX] 范围内，
    // 越接近 kOPT 性能越好。
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN,
                           nvinfer1::Dims4{1, 3, 224, 224});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims4{8, 3, 224, 224});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims4{16, 3, 224, 224});
    config->addOptimizationProfile(profile);

    // ── 6. 精度设置
    // ───────────────────────────────────────────────────────────
    std::unique_ptr<Int8Calibrator> calibrator;
    switch (precision) {
        case Precision::FP16:
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            break;

        case Precision::INT8:
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            config->setFlag(nvinfer1::BuilderFlag::kFP16);  // fallback
            calibrator = std::make_unique<Int8Calibrator>(8, 3, 224, 224,
                                                          "calib_cache.bin");
            config->setInt8Calibrator(calibrator.get());
            break;

        case Precision::FP32:
        default:
            break;
    }

    // ── 7. 构建并序列化 Engine
    // ────────────────────────────────────────────────
    //
    // buildSerializedNetwork 是最耗时的步骤，内部执行：
    //   a. 图优化（层融合、常量折叠等）
    //   b. 对每个算子枚举 CUDA kernel 候选，实际运行计时，选最快的
    //   c. INT8 模式下调用 calibrator 执行校准
    //
    // 返回 IHostMemory：序列化后的二进制 engine 数据（在 CPU 内存中）
    std::cout << "[Builder] Building engine, please wait...\n";

    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
        throw std::runtime_error("[Builder] buildSerializedNetwork failed");
    }

    // ── 8. 写入文件
    // ───────────────────────────────────────────────────────────
    std::ofstream fout(enginePath, std::ios::binary);
    if (!fout) {
        throw std::runtime_error("[Builder] Cannot write: " + enginePath);
    }

    fout.write(static_cast<const char *>(serialized->data()),
               serialized->size());
    std::cout << "[Builder] Engine saved: " << enginePath
              << "  size=" << serialized->size() / 1024 / 1024 << " MB\n";
}

```
## src/infer.hpp
```
#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <memory>
#include "logger.hpp"

/**
 * TensorRT 推理会话（Inference Session）
 *
 * 对象生命周期与内部结构：
 *
 *   IRuntime
 *     └── ICudaEngine          反序列化 engine，包含编译好的 CUDA kernel
 *           └── IExecutionContext  执行上下文，持有推理状态
 *                                  单线程复用，多线程需每线程独立创建
 *
 * 显存布局：
 *   mDeviceInput  → GPU 输入缓冲区  [maxBatch * 3 * 224 * 224 * sizeof(float)]
 *   mDeviceOutput → GPU 输出缓冲区  [maxBatch * 1000 * sizeof(float)]
 *
 * 推理流程（异步）：
 *   1. setInputShape()       告知 context 本次实际 batch 大小
 *   2. setTensorAddress()    绑定输入/输出 GPU 指针
 *   3. cudaMemcpyAsync()     H2D（Host to Device）异步拷贝输入数据
 *   4. enqueueV3()           将推理任务提交到 CUDA Stream（异步，立即返回）
 *   5. cudaMemcpyAsync()     D2H（Device to Host）异步拷贝输出数据
 *   6. cudaStreamSynchronize() 等待 stream 中所有任务完成
 *
 * CUDA Stream（流）：
 *   GPU 任务队列，同一 stream 内任务顺序执行，不同 stream 可并行。
 *   使用异步接口 + stream 可以将 H2D / 推理 / D2H 流水线化，
 *   提高 GPU 利用率（本实现单 stream，流水线优化留作扩展）。
 */
class InferSession {
public:
    /**
     * @param enginePath  .engine 文件路径
     * @param logger      TRT Logger，需与构建时同一实例或同类型实例
     * @param maxBatch    最大 batch 大小，须 ≤ 构建时的 kMAX
     */
    explicit InferSession(const std::string &enginePath, Logger &logger, int maxBatch = 16);
    ~InferSession();

    // 禁止拷贝（持有 GPU 资源，拷贝语义不安全）
    InferSession(const InferSession&)               = delete;
    InferSession& operator=(const InferSession&)     = delete;

    /**
     * 执行一次推理
     *
     * @param inputHost  CPU 端输入数据，NCHW 格式，FP32
     *                   大小必须 = batchSize * 3 * 224 * 224
     * @param batchSize  本次实际 batch 大小，须在 [kMIN, kMAX] 范围内
     * @return           CPU 端输出 logits，大小 = batchSize * 1000，1000 维向量
     *                      （ResNet 系列最初是为 ImageNet (ILSVRC) 大规模视觉识别挑战赛设计的，
     *                      ImageNet 数据集共有 1000 个类别，故这里默认设为 1000 维）
     */
    std::vector<float> infer(const std::vector<float>& inputHost,
                             int batchSize);
    
    /**
     * 性能基准测试（Benchmark）
     *
     * 使用 CUDA Event 计时（比 std::chrono 精确，避免 CPU 调度 jitter）。
     * 统计指标：mean / p50 / p99 延迟，吞吐量（img/s）。
     *
     * @param batchSize  测试用 batch 大小
     * @param nWarmup    预热次数（排除 GPU 首次启动 JIT 开销）
     * @param nRun       正式计时次数
     */
    void benchmark(int batchSize, int nWarmup = 50, int nRun = 200);

private:
    // 根据 maxBatch 分配 GPU 输入/输出缓冲区
    void allocBuffers();

    Logger& m_logger;
    int m_max_batch;

    std::unique_ptr<nvinfer1::IRuntime>             m_runtime; // 负责反序列化 engine
    std::unique_ptr<nvinfer1::ICudaEngine>          m_engine;  // 包含编译好的 CUDA kernel，线程安全，可多线程共享 
    std::unique_ptr<nvinfer1::IExecutionContext>    m_context; // 执行上下文，持有推理状态，单线程复用

    // GPU 缓冲区
    void* m_device_input =      nullptr;
    void* m_device_output =     nullptr;

    // 模型固定参数（ResNet18 ImageNet）
    static constexpr int k_C     = 3;
    static constexpr int k_H     = 224;
    static constexpr int k_W     = 224;
    static constexpr int k_CLS   = 1000;  // 输出类别
    
    // CUDA Stream
    cudaStream_t m_stream = nullptr;
};
```
## src/infer.cpp（推理核心片段）
```cpp
#include "infer.hpp"
#include <fstream>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <stdexcept>

// ── 辅助：读取二进制文件到 buffer ────────────────────────────────────────────
static std::vector<char> readFile(const std::string& path)
{
    std::ifstream fin(path, std::ios::binary | std::ios::ate);
    if (!fin) {
        throw std::runtime_error("[Infer] Cannot open: " + path);
    }

    // ate：打开时定位到文件末尾，tellg() 直接得到文件大小
    size_t sz = fin.tellg();
    fin.seekg(0);

    std::vector<char> buf(sz);
    fin.read(buf.data(), sz);
    return buf;
}

// ── 构造函数 ──────────────────────────────────────────────────────────────────
InferSession::InferSession(const std::string& enginePath,
                           Logger& logger,
                           int maxBatch)
    : m_logger(logger), m_max_batch(maxBatch) {
    // 1. 读取 engine 文件
    auto buf = readFile(enginePath);

    // 2. 创建 Runtime
    //    IRuntime 是推理侧入口，负责反序列化 engine。
    //    与构建侧 IBuilder 完全独立，生产部署时只需 Runtime。
    m_runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!m_runtime) {
        std::cerr << "[Infer] createInferRuntime returned nullptr\n";
        std::cerr << "[Infer] TRT version: " << NV_TENSORRT_VERSION << "\n";
        throw std::runtime_error("[Infer] createInferRuntime failed");
    }
    
    // 3. 反序列化 Engine
    //    deserializeCudaEngine 将二进制数据还原为 ICudaEngine 对象，
    //    并将编译好的 CUDA kernel 加载到 GPU。
    //    此步骤比构建快得多（秒级），适合每次启动时执行。
    m_engine.reset(m_runtime->deserializeCudaEngine(buf.data(), buf.size()));
    if (!m_engine)
        throw std::runtime_error("[Infer] deserializeCudaEngine failed");

    // 4. 创建执行上下文
    //    IExecutionContext 持有推理状态（动态 shape 绑定、中间激活值显存等）。
    //    同一 engine 可创建多个 context 用于多线程并发推理，
    //    本实现单线程，只创建一个。
    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        throw std::runtime_error("[Infer] createExecutionContext failed");
    }

    // 5. 创建 CUDA Stream
    //    Stream 是 GPU 任务队列，异步操作都提交到 stream 中顺序执行。
    cudaStreamCreate(&m_stream);

    // 6. 分配 GPU 缓冲区
    allocBuffers();

    std::cout << "[Infer] Session ready: " << enginePath << "\n";
}

// ── 析构函数 ──────────────────────────────────────────────────────────────────
InferSession::~InferSession()
{
    // 释放 GPU 资源
    if (m_device_input)  cudaFree(m_device_input);
    if (m_device_output) cudaFree(m_device_output);
    if (m_stream)       cudaStreamDestroy(m_stream);
    // mContext / mEngine / mRuntime 由 unique_ptr 自动析构
}

// ── 分配 GPU 缓冲区 ───────────────────────────────────────────────────────────
void InferSession::allocBuffers() {
    // 按最大 batch 分配，推理时实际使用的 batch ≤ mMaxBatch
    size_t input_bytes = m_max_batch * k_C * k_H * k_W * sizeof(float);
    size_t output_bytes = m_max_batch * k_CLS * sizeof(float);

    cudaMalloc(&m_device_input, input_bytes);
    cudaMalloc(&m_device_output, output_bytes);

    std::cout << "[Infer] GPU buffers allocated:"
              << " input="  << input_bytes  / 1024 << " KB"
              << " output=" << output_bytes / 1024 << " KB\n";
}

// ── 单次推理 ──────────────────────────────────────────────────────────────────
std::vector<float> InferSession::infer(const std::vector<float>& inputHost,
                                       int batchSize) {
    // 1. 设置本次推理的动态输入 shape
    //    动态 shape 必须在每次推理前设置，告知 context 实际 batch 大小。
    //    TRT 据此确定输出 shape 和中间缓冲区大小。
    m_context->setInputShape("input", nvinfer1::Dims4{batchSize, k_C, k_H, k_W});

    // 2. 绑定 tensor 地址（TRT 10.x API）
    //    context 不缓存地址，每次推理前必须重新设置。
    m_context->setTensorAddress("input", m_device_input);
    m_context->setTensorAddress("output", m_device_output);

    size_t input_bytes = batchSize * k_C * k_H * k_W * sizeof(float);
    size_t output_bytes = batchSize * k_CLS * sizeof(float);

    // 3. H2D 异步拷贝（Host to Device）
    //    将 CPU 数据异步传输到 GPU，不阻塞 CPU。
    //    数据在 stream 中排队，保证在推理之前完成。
    cudaMemcpyAsync(m_device_input, inputHost.data(), input_bytes, cudaMemcpyHostToDevice, m_stream);

    // 4. 异步推理
    //    enqueueV3 将推理任务提交到 stream，立即返回，不等待 GPU 完成。
    //    GPU 在后台执行，CPU 可继续做其他工作（本实现直接等待）。
    if (!m_context->enqueueV3(m_stream)) {
        throw std::runtime_error("[Infer] enqueueV3 failed");
    }

    // 5. D2H 异步拷贝（Device to Host）
    //    将 GPU 输出结果异步拷贝回 CPU。
    //    因为在同一 stream 中，保证在推理完成后才执行。
    std::vector<float> outputHost(batchSize * k_CLS);
    cudaMemcpyAsync(outputHost.data(), m_device_output, output_bytes, cudaMemcpyDeviceToHost, m_stream);

    // 6. 同步：等待 stream 中所有任务完成
    cudaStreamSynchronize(m_stream);

    return outputHost;
}

// ── Benchmark ─────────────────────────────────────────────────────────────────
void InferSession::benchmark(int batchSize, int nWarmup, int nRun) {
    // 构造固定输入数据
    std::vector<float> input(batchSize * k_C * k_H * k_W, 0.5f);

    // 设置 shape 和地址（benchmark 期间固定不变）
    m_context->setInputShape("input", nvinfer1::Dims4{batchSize, k_C, k_H, k_W});
    m_context->setTensorAddress("input", m_device_input);
    m_context->setTensorAddress("output", m_device_output);

    size_t input_bytes = batchSize * k_C * k_H * k_W * sizeof(float);
    cudaMemcpyAsync(m_device_input, input.data(), input_bytes, cudaMemcpyHostToDevice, m_stream);
    cudaStreamSynchronize(m_stream);

    // ── Warmup（预热）──────────────────────────────────────────────────────
    // GPU kernel 首次启动存在初始化开销（JIT 编译缓存、显存页锁定等），
    // warmup 排除此干扰，使计时结果稳定。
    for (int i = 0; i < nWarmup; ++i) {
        m_context->enqueueV3(m_stream);
        cudaStreamSynchronize(m_stream);
    }

    // ── CUDA Event 计时 ────────────────────────────────────────────────────
    // CUDA Event 在 GPU 时间线上打点，比 std::chrono 更精确：
    //   - chrono 包含 CPU 调度 jitter（线程被抢占导致的误差）
    //   - CUDA Event 直接测量 GPU 执行时间，精度约 0.5 μs
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    std::vector<float> latencies(nRun);
    for (int i = 0; i < nRun; ++i) {
        // EventRecord：在 stream 当前位置插入时间戳
        cudaEventRecord(ev_start, m_stream);
        m_context->enqueueV3(m_stream);
        cudaEventRecord(ev_stop, m_stream);

        // EventSynchronize：等待 evStop 完成（只等这一个 event，不等整个 stream）
        cudaEventSynchronize(ev_stop);

        // ElapsedTime：计算两个 event 之间的 GPU 时间，单位 ms
        cudaEventElapsedTime(&latencies[i], ev_start, ev_stop);
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    // ── 统计 ───────────────────────────────────────────────────────────────
    float mean = std::accumulate(latencies.begin(), latencies.end(), 0.f) / nRun;

    std::vector<float> sorted = latencies;
    std::sort(sorted.begin(), sorted.end());
    float p50 = sorted[nRun * 50 / 100];
    float p99 = sorted[nRun * 99 / 100];
    
    // 吞吐量：每秒处理的图片数
    float throughput = batchSize / (mean / 1000.f); // 张/ms -> 张/s

    std::cout << "[Benchmark]"
              << "  batch="      << batchSize
              << "  mean="       << mean      << " ms"
              << "  p50="        << p50       << " ms"
              << "  p99="        << p99       << " ms"
              << "  throughput=" << throughput << " img/s\n";
}
```
## src/calibrator.hpp
```cpp
#pragma once
#include <NvInfer.h>
#include <string>
#include <vector>

/**
 * INT8 校准器（INT8 Calibrator）
 *
 * 背景：
 *   FP32 有效范围约 ±3.4e38，INT8 只有 [-128, 127]。
 *   量化（Quantization）就是把 FP32 激活值线性映射到 INT8：
 *
 *       x_int8 = clamp( round(x_fp32 / scale), -128, 127 )
 *
 *   scale 的确定需要在真实数据上统计每一层激活值的分布，
 *   这个过程叫做校准（Calibration）。
 *
 * 校准器类型：
 *   TRT 提供 4 种校准器，区别在于 scale 的计算方式：
 *
 *   IInt8MinMaxCalibrator      取激活值的 [min, max]
 * 作为范围，简单但可能有较大误差 IInt8EntropyCalibrator     最小化量化前后的 KL
 * 散度，精度较好（旧版默认） IInt8EntropyCalibrator2    EntropyCalibrator
 * 的改进版，当前推荐 IInt8LegacyCalibrator      旧版兼容，不推荐
 *
 *   本实现继承 IInt8EntropyCalibrator2。
 *
 * 校准流程：
 *   1. TRT 调用 getBatch() 获取一批数据（在 GPU 上）
 *   2. TRT 对该批数据执行前向传播，收集每层激活值分布
 *   3. 重复直到 getBatch() 返回 false
 *   4. TRT 根据统计结果计算每层的 scale，写入 cache
 *
 * 生产环境：
 *   getBatch() 中替换为真实 ImageNet 数据，建议 ≥ 500 张。
 *   此处用随机数据仅验证流程，INT8 精度无参考价值。
 */

class Int8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    /**
     * @param batch_size  每次校准的 batch 大小
     * @param channels   输入通道数
     * @param height     输入高度
     * @param width      输入宽度
     * @param cache_file  校准缓存文件路径，存在则跳过校准直接读取
     */
    Int8Calibrator(int batch_size, int channels, int height, int width,
                   const std::string &cache_file);
    ~Int8Calibrator();
    
    // -- TRT 回调接口--
    int getBatchSize() const noexcept override;

    /**
     * 将下一批数据的 GPU 指针写入 bindings[]。
     * @param bindings  输出数组，bindings[i] = 第 i 个输入 tensor 的 GPU 指针
     * @param names     各输入 tensor 的名称
     * @param nb_bindings tensor 数量
     * @return true 表示有数据；false 表示数据耗尽，校准结束
     */
    bool getBatch(void* bindings[], const char* names[],
                  int nbBindings) noexcept override;

    // 读取已有 cache，返回 nullptr 表示无 cache，TRT 将重新校准
    const void* readCalibrationCache(size_t& length) noexcept override;

    // TRT 校准完成后调用，将 scale 数据写入 cache 文件
    void writeCalibrationCache(const void* cache,
                               std::size_t length) noexcept override;
private:
    int m_batch_size;               // 每批的样本数量
    int m_input_size;               // 每批输入的元素数量（batch_size * channels * height * width）
    int m_current_batch = 0;        // 当前校准批次索引
    int m_total_batchs = 10;        // 校准轮数，生产环境：ceil(500 / batchSize)

    void*               m_device_input = nullptr; 	// GPU 端输入缓冲区
    std::vector<float>  m_host_input;               // CPU 端数据
	
	std::string         m_cache_file;
    std::vector<char>   m_calibration_cache;       // 读入的 cache 数据

};    
```
## src/calibrator.cpp
```
#include "calibrator.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>

// 构造
Int8Calibrator::Int8Calibrator(int batch_size, int channels, int height,
                               int width, const std::string &cache_file)
    : m_batch_size(batch_size),
      m_input_size(batch_size * channels * height * width),
      m_cache_file(cache_file) {
    // 分配 CPU 端数据缓冲区
    m_host_input.resize(m_input_size);

    // 分配 GPU 端数据缓冲区
    cudaMalloc(&m_device_input, m_input_size * sizeof(float));
}

// 析构
Int8Calibrator::~Int8Calibrator() {
    // 释放 GPU 显存，与 cudaMalloc 配对
    if (m_device_input) {
        cudaFree(m_device_input);
    }
}

int Int8Calibrator::getBatchSize() const noexcept {
    return m_batch_size;
}

bool Int8Calibrator::getBatch(void *bindings[], const char *names[],
                             int nbBindings) noexcept {
    // 数据耗尽，通知 TRT 校准结束
    if (m_current_batch >= m_total_batchs) {
        return false;
    }

    // 生成随机数据
    // 生产环境：此处替换为从磁盘加载真实图片，并做归一化预处理：
    //   x = (pixel / 255.0 - mean) / std
    //   mean = [0.485, 0.456, 0.406]
    //   std  = [0.229, 0.224, 0.225]
    std::mt19937 rng(m_current_batch);  // 用 batch 序号做种子，保证可复现
    std::normal_distribution<float> dist(0.f, 1.f);  // 标准正态分布
    for (auto &v : m_host_input) {
        v = dist(rng);
    }

    // CPU → GPU 拷贝（同步）
    // cudaMemcpy(dst, src, bytes, direction)
    cudaMemcpy(m_device_input, m_host_input.data(),
               m_input_size * sizeof(float), cudaMemcpyHostToDevice);

    // 将 GPU 指针写入 bindings[0]（ResNet18 只有一个输入）
    bindings[0] = m_device_input;

    std::cout << "[Calibrator] batch " << m_current_batch + 1 << " / "
              << m_total_batchs << "\n";

    ++m_current_batch;
    return true;
}

const void* Int8Calibrator::readCalibrationCache(size_t& length) noexcept {
    m_calibration_cache.clear();

    std::ifstream fin(m_cache_file, std::ios::binary);
    
    if (!fin) {
        length = 0;
        return nullptr;
    }

    m_calibration_cache.assign(std::istreambuf_iterator<char>(fin),
                               std::istreambuf_iterator<char>());
    length = m_calibration_cache.size();

    std::cout << "[Calibrator] Loaded cache: " << m_cache_file
              << " (" << length << " bytes), skipping calibration.\n";

    return m_calibration_cache.data();
}

void Int8Calibrator::writeCalibrationCache(const void* cache,
                               std::size_t length) noexcept {
    std::ofstream fout(m_cache_file, std::ios::binary);
    fout.write(static_cast<const char*>(cache), length);
    
    std::cout << "[Calibrator] Written cache: " << m_cache_file
              << " (" << length << " bytes).\n";
}

```
## src/main.cpp
```
#include "builder.hpp"
#include "infer.hpp"
#include <NvInferPlugin.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

// ── 精度对比工具函数 ──────────────────────────────────────────────────────────

/**
 * 余弦相似度（Cosine Similarity）
 *
 * 公式：cos(θ) = (A · B) / (||A|| * ||B||)
 *
 * 含义：
 *   衡量两个向量的方向差异，与向量模长无关。
 *   范围 [-1, 1]，越接近 1 表示输出分布越一致。
 *   用于精度对比比 MSE 更能反映分类结果的一致性：
 *   即使绝对值有偏差，只要各类别的相对大小顺序一致，cos 仍接近 1。
 */
static float cosineSim(const std::vector<float>& a,
                       const std::vector<float>& b) {
    double dot = 0, norm_A = 0, norm_B = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot    += static_cast<double>(a[i]) * b[i];
        norm_A += static_cast<double>(a[i]) * a[i];
        norm_B += static_cast<double>(b[i]) * b[i];
    }
    return static_cast<float>(dot / (std::sqrt(norm_A) * std::sqrt(norm_B) + 1e-12));
}

/**
 * 最大绝对误差（Max Absolute Difference）
 *
 * 公式：max( |a[i] - b[i]| )
 *
 * 含义：
 *   找出所有输出元素中偏差最大的一个。
 *   反映量化引入的最坏情况误差。
 */
static float maxAbsDiff(const std::vector<float>& a,
                        const std::vector<float>& b) {
    float max_d = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_d = std::max(max_d, std::abs(a[i] - b[i]));
    }
    return max_d;
}

/**
 * 均方误差（Mean Squared Error）
 *
 * 公式：MSE = (1/N) * Σ (a[i] - b[i])²
 *
 * 含义：
 *   衡量整体误差的平均水平，对大误差敏感（平方放大）。
 */
static float mse(const std::vector<float>& a,
                 const std::vector<float>& b) {
    double sum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return static_cast<float>(sum / a.size());
}

// ── 最大值索引 ────────────────────────────────────────────────────────────────
static int argmax(const std::vector<float>& logits, int offset, int numCls) {
    // 找 [offset, offset+numCls) 范围内最大值的下标
    int mx_idx = 0;
    float mx = logits[offset];
    for (int i = 1; i < numCls; ++i) {
        if (logits[offset + i] > mx) {
            mx = logits[offset + i];
            mx_idx = i;
        }
    }
    return mx_idx;
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    Logger logger;

    // 初始化 TRT 内置插件库，必须在任何 Runtime 创建前调用
    initLibNvInferPlugins(&logger, "");

    const std::string onnxPath = std::string(PROJECT_SOURCE_DIR) + "/resnet18.onnx";
    const int BATCH = 8;
    const int C = 3, H = 224, W = 224;
    const int NUM_CLS = 1000;

    // ── Step 1: 构建三种精度 Engine ──────────────────────────────────────────
    struct Task
    {
        std::string engine;
        Precision   prec;
    };

    std::vector<Task> tasks = {
        {"resnet18_fp32.engine", Precision::FP32},
        {"resnet18_fp16.engine", Precision::FP16},
        {"resnet18_int8.engine", Precision::INT8},
    };

    for (auto& t : tasks)
    {
        std::cout << "\n========== Build: " << t.engine << " ==========\n";
        buildEngine(onnxPath, t.engine, t.prec, logger);
    }

    // ── Step 2: 准备统一输入数据 ─────────────────────────────────────────────
    // 使用相同输入保证精度对比公平，值域 [0,1] 模拟归一化后的图片数据
    std::vector<float> input(BATCH * C * H * W);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i % 255) / 255.f;
    }
    
    // ── Step 3: 推理 + 精度对比 + Benchmark ─────────────────────────────────
    std::vector<float> fp32_out;

    for (auto &t : tasks) {
        std::cout << "\n========== Infer: " << t.engine << " ==========\n";
        InferSession sess(t.engine, logger);

        // 精度验证推理
        auto out = sess.infer(input, BATCH);

        if (t.prec == Precision::FP32) {
            fp32_out = out;
            std::cout << "[Accuracy] FP32 baseline, argmax class (batch[0]): "
                      << argmax(fp32_out, 0, NUM_CLS) << "\n";
        } else {
            // 如果 fp32Out 为空，说明 FP32 推理结果没有正确保存
            if (fp32_out.empty())
            {
                std::cerr << "[ERROR] fp32Out is empty, FP32 must run first.\n";
                continue;
            }
            // 与 FP32 基准对比
            float cs   = cosineSim(fp32_out, out);
            float diff = maxAbsDiff(fp32_out, out);
            float err  = mse(fp32_out, out);

            std::cout << "[Accuracy vs FP32]"
                      << "  cosine_sim="   << cs
                      << "  max_abs_diff=" << diff
                      << "  mse="          << err << "\n";

            // 最大值索引 是否一致（对分类任务最直接的精度指标）
            bool allMatch = true;
            for (int b = 0; b < BATCH; ++b) {
                int fp32_argmax = argmax(fp32_out, b * NUM_CLS, NUM_CLS);
                int cur_argmax  = argmax(out,     b * NUM_CLS, NUM_CLS);
                if (fp32_argmax != cur_argmax) {
                    allMatch = false;
                    std::cout << "  [!] batch[" << b << "] argmax mismatch:"
                              << " FP32=" << fp32_argmax
                              << " " << precisionStr(t.prec) << "=" << cur_argmax << "\n";
                }
            }
            if (allMatch) {
                std::cout << "  Top-1 all match FP32 across "
                          << BATCH << " samples.\n";
            }
        }
        // 性能 Benchmark
        sess.benchmark(BATCH, 50, 200);
    }

    std::cout << "\n========== Done ==========\n";
    return 0;
}
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
