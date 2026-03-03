## 环境

|项目|版本|
|---|---|
|GPU|GTX 1660 Ti (Turing, SM 7.5)|
|TensorRT|10.15.1|
|CUDA|12.6|
|nsys|2024.5.1|

***

## 1. NVTX 标注 + nsys Profiling
### 思路
默认 `nsys timeline` 只显示 CUDA kernel 乱码名称。加入 NVTX 标注后，可在 timeline 上看到清晰的阶段标签（H2D / Infer / D2H）。
### 变动代码
**`src/infer.cpp` — 顶部加头文件：**
```cpp
#include <nvtx3/nvToolsExt.h>
```
**`src/infer.cpp` — `infer()` 函数加标注：**
```cpp
nvtxRangePushA("H2D");
cudaMemcpyAsync(m_device_input, m_pinned_input, input_bytes,
                cudaMemcpyHostToDevice, m_stream);
nvtxRangePop();
nvtxRangePushA("Infer");
if (!m_context->enqueueV3(m_stream))
    throw std::runtime_error("[Infer] enqueueV3 failed");
nvtxRangePop();
nvtxRangePushA("D2H");
cudaMemcpyAsync(m_pinned_output, m_device_output, output_bytes,
                cudaMemcpyDeviceToHost, m_stream);
nvtxRangePop();
```
**`src/infer.cpp` — `benchmark()` 计时循环加标注：**
```cpp
for (int i = 0; i < nRun; ++i) {
    nvtxRangePushA("benchmark_iter");
    cudaEventRecord(ev_start, m_stream);
    m_context->enqueueV3(m_stream);
    cudaEventRecord(ev_stop, m_stream);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&latencies[i], ev_start, ev_stop);
    nvtxRangePop();
}
```
**`0_resnet18_onnx/CMakeLists.txt` — 末尾加 profile / stats target：**
```cmake
add_custom_target(profile
    COMMAND nsys profile
        --output ${CMAKE_CURRENT_BINARY_DIR}/nsys_report
        --trace cuda,nvtx,osrt,cudnn,cublas
        --sample=none
        --force-overwrite=true
        $<TARGET_FILE:trt_resnet18>
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS trt_resnet18
    COMMENT "Running nsys profile..."
)
add_custom_target(stats
    COMMAND sqlite3 ${CMAKE_CURRENT_BINARY_DIR}/nsys_report.sqlite
        < ${CMAKE_CURRENT_SOURCE_DIR}/stats.sql
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Querying NVTX stats from nsys_report.sqlite..."
)
```
**`0_resnet18_onnx/stats.sql`（新建）：**
```sql
SELECT text,
       COUNT(*) as count,
       AVG(end - start) / 1000000.0 as avg_ms,
       MIN(end - start) / 1000000.0 as min_ms,
       MAX(end - start) / 1000000.0 as max_ms
FROM NVTX_EVENTS
WHERE text IN ('H2D', 'Infer', 'D2H', 'benchmark_iter')
GROUP BY text
ORDER BY avg_ms DESC;
```
### WSL2 注意事项
- CUPTI kernel-level tracing 不可用（无 `CUPTI_ACTIVITY_KIND_KERNEL` 表）
- NVTX 事件正常采集，存入 `NVTX_EVENTS` 表
- `.nsys-rep` 需使用与命令行版本一致的 GUI（2024.5.1）打开

***

## 2. Pageable vs Pinned Memory
### 问题发现
`nsys stats` 查询结果：

|阶段|优化前|优化后|改善|
|---|---|---|---|
|H2D|1.49 ms|0.036 ms|41×|
|Infer|1.16 ms|~1.1 ms|持平|
|D2H|**22.306 ms**|**0.014 ms**|**1593×**|

### 根因
WSL2 下 `std::vector`（Pageable Memory）执行 `cudaMemcpyAsync` 会退化为同步拷贝，触发内核态内存锁定，导致巨大开销。
### 变动代码
**`src/infer.hpp` — 新增 Pinned Memory 成员：**
```cpp
float* m_pinned_input  = nullptr;
float* m_pinned_output = nullptr;
```
**`src/infer.cpp` — `allocBuffers()` 分配 Pinned Memory：**
```cpp
void InferSession::allocBuffers() {
    size_t input_bytes  = m_max_batch * k_C * k_H * k_W * sizeof(float);
    size_t output_bytes = m_max_batch * k_CLS * sizeof(float);
    cudaMalloc(&m_device_input,  input_bytes);
    cudaMalloc(&m_device_output, output_bytes);
    cudaMallocHost(&m_pinned_input,  input_bytes);
    cudaMallocHost(&m_pinned_output, output_bytes);
}
```
**`src/infer.cpp` — 析构函数释放：**
```cpp
InferSession::~InferSession() {
    if (m_device_input)  cudaFree(m_device_input);
    if (m_device_output) cudaFree(m_device_output);
    if (m_pinned_input)  cudaFreeHost(m_pinned_input);
    if (m_pinned_output) cudaFreeHost(m_pinned_output);
    if (m_stream)        cudaStreamDestroy(m_stream);
}
```
**`src/infer.cpp` — `infer()` 使用 Pinned Memory：**
```cpp
memcpy(m_pinned_input, inputHost.data(), input_bytes);
nvtxRangePushA("H2D");
cudaMemcpyAsync(m_device_input, m_pinned_input, input_bytes,
                cudaMemcpyHostToDevice, m_stream);
nvtxRangePop();
nvtxRangePushA("D2H");
cudaMemcpyAsync(m_pinned_output, m_device_output, output_bytes,
                cudaMemcpyDeviceToHost, m_stream);
nvtxRangePop();
cudaStreamSynchronize(m_stream);
return std::vector<float>(m_pinned_output,
                          m_pinned_output + batchSize * k_CLS);
```
### Pinned Memory 使用原则

|场景|建议|
|---|---|
|ResNet18（~10MB）|无限制|
|多模型并发|注意总量，避免挤压系统内存|
|LLM（GB 级）|谨慎评估物理内存容量|

***

## 3. Multi-Batch 性能曲线
### 变动代码
**`src/infer.hpp` — `benchmark()` 改为返回结构体：**
```cpp
struct BenchResult {
    float mean_ms;
    float p50_ms;
    float p99_ms;
    float throughput;
};
BenchResult benchmark(int batchSize, int nWarmup = 50, int nRun = 200);
```
**`src/infer.cpp` — `benchmark()` 末尾改为 return：**
```cpp
return BenchResult{mean, p50, p99, throughput};
```
**`src/main.cpp` — Multi-Batch 循环：**
```cpp
std::vector<int> batch_sizes = {1, 2, 4, 8, 16};
std::cout << std::left
          << std::setw(10) << "Precision"
          << std::setw(8)  << "Batch"
          << std::setw(12) << "mean(ms)"
          << std::setw(12) << "p50(ms)"
          << std::setw(12) << "p99(ms)"
          << std::setw(16) << "throughput"
          << "\n"
          << std::string(70, '-') << "\n";
for (int bs : batch_sizes) {
    auto r = sess.benchmark(bs, 50, 200);
    std::cout << std::left
              << std::setw(10) << precisionStr(t.prec)
              << std::setw(8)  << bs
              << std::setw(12) << r.mean_ms
              << std::setw(12) << r.p50_ms
              << std::setw(12) << r.p99_ms
              << std::setw(16) << r.throughput
              << "\n";
}
```
### 结果

|Batch|FP32 (img/s)|FP16 (img/s)|INT8 (img/s)|
|---|---|---|---|
|1|583|813|770|
|2|861|992|1190|
|4|868|1383|1591|
|8|956|1597|1942|
|16|1014|1655|**2070**|

### 结论
- `batch = 8` 附近 GPU 接近饱和，`batch = 16` 边际收益有限
- INT8 在 `batch = 1` 时略慢于 FP16（kernel launch overhead 占比高）
- `batch ≥ 4` 后 INT8 优势明显
- INT8 最大加速比 **2.04×**（vs FP32，batch=16）

***

## 4. Workspace 调优
### 变动代码
**`src/builder.hpp` — 加 workspace 参数：**
```cpp
void buildEngine(const std::string& onnxPath,
                 const std::string& enginePath,
                 Precision precision,
                 Logger& logger,
                 size_t workspaceBytes = 1UL << 30);
```
**`src/builder.cpp` — 替换 workspace 设置：**
```cpp
config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspaceBytes);
```
**`src/main.cpp` — Workspace 调优实验：**
```cpp
struct WsTask { std::string engine; size_t workspace; };
std::vector<WsTask> ws_tasks = {
    {"resnet18_int8_ws32m.engine",   32UL  << 20},
    {"resnet18_int8_ws128m.engine",  128UL << 20},
    {"resnet18_int8_ws256m.engine",  256UL << 20},
    {"resnet18_int8_ws512m.engine",  512UL << 20},
    {"resnet18_int8_ws1024m.engine", 1UL   << 30},
};
for (auto& [wt, build_sec] : ws_results) {
    InferSession sess(wt.engine, logger);
    auto r = sess.benchmark(8, 50, 200);
    std::cout << std::left
              << std::setw(32) << wt.engine
              << std::setw(14) << (wt.workspace >> 20)
              << std::setw(14) << build_sec
              << std::setw(12) << r.mean_ms
              << std::setw(16) << r.throughput
              << "\n";
}
```
### 结论
ResNet18 规模较小，workspace 对 build 时间与推理性能均无明显影响。  
`32MB` workspace 已足够。该参数主要影响大型模型（如 Transformer / LLM）的 kernel tactic 搜索空间。

***

## 5. INT8 Calibration：随机数据 vs 真实数据
### 变动代码
**`src/calibrator.hpp` — 新增成员：**
```cpp
std::string              m_calib_data_dir;
std::vector<std::string> m_file_list;
int                      m_file_index = 0;
Int8Calibrator(int batch_size, int channels, int height, int width,
               const std::string& cache_file,
               const std::string& calib_data_dir);
```
**`src/calibrator.cpp` — 构造函数扫描文件：**
```cpp
#include <filesystem>
namespace fs = std::filesystem;
Int8Calibrator::Int8Calibrator(..., const std::string& calib_data_dir)
    : ..., m_calib_data_dir(calib_data_dir) {
    for (auto& entry : fs::directory_iterator(calib_data_dir)) {
        if (entry.path().extension() == ".bin")
            m_file_list.push_back(entry.path().string());
    }
    std::sort(m_file_list.begin(), m_file_list.end());
    m_total_batchs = static_cast<int>(m_file_list.size()) / m_batch_size;
    m_host_input.resize(m_input_size);
    cudaMalloc(&m_device_input, m_input_size * sizeof(float));
}
```
**`src/calibrator.cpp` — `getBatch()` 读取真实文件：**
```cpp
bool Int8Calibrator::getBatch(void* bindings[], const char* names[],
                              int nbBindings) noexcept {
    if (m_current_batch >= m_total_batchs) return false;
    int single_size = m_input_size / m_batch_size;
    for (int i = 0; i < m_batch_size; ++i) {
        int file_idx = m_current_batch * m_batch_size + i;
        if (file_idx >= static_cast<int>(m_file_list.size())) break;
        std::ifstream fin(m_file_list[file_idx], std::ios::binary);
        fin.read(reinterpret_cast<char*>(
                     m_host_input.data() + i * single_size),
                 single_size * sizeof(float));
    }
    cudaMemcpy(m_device_input, m_host_input.data(),
               m_input_size * sizeof(float), cudaMemcpyHostToDevice);
    bindings[0] = m_device_input;
    ++m_current_batch;
    return true;
}
```
**`src/builder.cpp` — 传入真实数据目录：**
```cpp
calibrator = std::make_unique<Int8Calibrator>(
    8, 3, 224, 224,
    "calib_cache.bin",
    std::string(PROJECT_SOURCE_DIR) + "/calib_data"
);
```
**`prepare_calib_data.py`（新建）— 预处理脚本：**
```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
def preprocess(image):
    if image is None: return None
    image = image.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    return arr
def main():
    out_dir = os.path.join(os.path.dirname(__file__), "calib_data")
    os.makedirs(out_dir, exist_ok=True)
    dataset = load_dataset("zh-plus/tiny-imagenet", split="valid", streaming=True)
    count, target = 0, 500
    pbar = tqdm(total=target, desc="Saving calib data", unit="img")
    for sample in dataset:
        arr = preprocess(sample["image"])
        if arr is None: 
            continue
        arr.tofile(os.path.join(out_dir, f"calib_{count:04d}.bin"))
        count += 1
        pbar.update(1)
        if count >= target:
            break
    pbar.close()
if __name__ == "__main__":
    main()
```
**`.gitignore`：**
```
**/calib_data/
```
### 精度对比

|指标|随机数据|真实数据|
|---|---|---|
|cosine_sim|0.9959|**0.9980**|
|max_abs_diff|1.125|**0.458**|
|mse|0.0315|**0.0143**|
|Top-1 match|全部一致|全部一致|

### 生产环境建议
- 校准图片数量 **≥ 500**，覆盖真实部署数据分布
- `calib_cache.bin` 必须与模型版本绑定，模型更新需重新校准
