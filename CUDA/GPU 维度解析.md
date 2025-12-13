在 CUDA 编程模型中，**grid** 与 **block** 的维度设计直接影响**并行度、内存访问效率、SM 资源利用率以及整体性能**。下面从**模型定义 → 设计原则 → 常见模式 → 性能考量 → 实战示例**进行系统性说明。
## 一、CUDA 中 grid / block 的基本概念
### 1. Grid（网格）
- **Grid** 是一次 kernel 启动的整体执行域
- 由 **多个 thread block** 组成
- 维度最多 **3 维**：`gridDim.x / y / z`[[为什么 Grid 和 Block 维度最多 3 维]]
- 不同 block **彼此独立**，不能直接同步或通信
### 2. Block（线程块）
- **Block** 是线程调度与资源分配的基本单位
- 一个 block 内：
    - 线程可通过 `__syncthreads()` 同步
    - 可共享 shared memory
- 维度最多 **3 维**：`blockDim.x / y / z`
- 每个 block 通常映射到 **一个 SM（或一个 SM 的一部分）**
### 3. Thread（线程）
- 最小执行单元
- 使用 `threadIdx.{x,y,z}` 进行索引
- 硬件以 **warp（32 线程）** 为最小调度单位
## 二、维度设计的核心目标
设计 grid / block 维度，本质是在回答三个问题：
1. **一个线程负责多少数据？**
2. **线程如何映射到数据结构？**
3. **如何让 SM 保持高占用（Occupancy）？**
总结为四个目标：

|目标|含义|
|---|---|
|并行度|覆盖足够多的数据元素|
|对齐 warp|blockDim 是 32 的倍数|
|内存友好|连续线程访问连续内存|
|资源平衡|不浪费寄存器 / shared memory|
## 三、block 维度设计原则（重点）
### 1. blockDim.x 优先考虑 128 / 256 / 512
**原因：**
- warp = 32 线程
- 128 = 4 warps
- 256 = 8 warps（最常见）
- 512 = 16 warps（适合算密集）
经验规则：
```text
通用 kernel：256
内存带宽受限：128 或 256
计算密集型：256 或 512
```
> 很少使用 1024（上限），因为：
> - 单 block 占用资源过多
> - SM 上并发 block 数下降
### 2. blockDim 形状要匹配数据维度
#### 1D 数据（向量、数组）
```cpp
blockDim = (256, 1, 1)
gridDim  = ((N + 255) / 256, 1, 1) // 向上取整
```
#### 2D 数据（图像、矩阵）
```cpp
blockDim = (16, 16, 1)   // 256 threads
gridDim  = (ceil(W/16), ceil(H/16), 1)
```
#### 3D 数据（体数据、3D stencil）
```cpp
blockDim = (8, 8, 8)     // 512 threads
gridDim  = (ceil(X/8), ceil(Y/8), ceil(Z/8))
```
### 3. block 内线程数必须 ≤ 硬件限制
典型限制（以现代 NVIDIA GPU 为例）：

| 项目         | 上限     |
| ---------- | ------ |
| block 总线程数 | 1024   |
| blockDim.x | ≤ 1024 |
| blockDim.y | ≤ 1024 |
| blockDim.z | ≤ 64   |
## 四、grid 维度设计原则
### 1. gridDim 用于“覆盖数据”，而非性能调优
- grid 只负责 **把数据切块**
- gridDim 过大不会降低性能（只要 kernel 内有边界判断）
典型写法：
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) { ... }
```
### 2. gridDim.x 应远大于 SM 数量
目标：
```text
grid 中 block 数 ≫ SM 数
```
原因：
- SM 执行 block 是动态调度
- block 数太少 → SM 空闲
## 五、典型维度设计模式（高频）
### 模式 1：一线程一元素（最常见）
```cpp
dim3 block(256);
dim3 grid((N + 255) / 256);
```
适用：
- 向量运算
- embedding lookup
- layernorm / bias add
### 模式 2：2D tile（矩阵 / 图像）
```cpp
dim3 block(16, 16);
dim3 grid((W + 15) / 16, (H + 15) / 16);
```
优点：
- 空间局部性好
- 易配合 shared memory
### 模式 3：block 对应一个逻辑单元
例如：
- 一个 block 处理一个 attention head
- 一个 block 处理一个 token 的 reduction
```cpp
dim3 block(256);
dim3 grid(num_heads);
```
## 六、性能层面的关键影响因素
### 1. Occupancy（占用率）
block 太大：
- 寄存器 / shared memory 用量高
- SM 并发 block 数下降
block 太小：
- warp 数不足
- 隐藏访存延迟能力弱
> 实际目标不是 100% occupancy，而是**足够高 + 高效计算**
### 2. 内存访问合并（Coalescing）
正确设计：
```text
threadIdx.x 连续 → 访问连续内存地址
```
错误设计：
```text
threadIdx.y 改变 → 跨 stride 访问
```
### 3. Warp 分支发散
block 维度影响 warp 内线程执行路径：
- 尽量让 warp 内线程走同一分支
- 避免在 y / z 维度做条件分支
## 七、一个完整示例（矩阵加法）
```cpp
__global__ void matAdd(float* A, float* B, float* C, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W && y < H) {
        int idx = y * W + x;
        C[idx] = A[idx] + B[idx];
    }
}

// launch
dim3 block(16, 16);
dim3 grid((W + 15) / 16, (H + 15) / 16);
matAdd<<<grid, block>>>(A, B, C, W, H);
```
## 八、实战经验总结（可直接记）
1. **先定 block，再算 grid**
2. **block 总线程数 ≈ 256 是最稳妥选择**
3. **block 形状要贴合数据维度**
4. **grid 只负责覆盖，不负责优化**
5. **warp 对齐 > 理论并行度**
6. **shared memory kernel 优先 2D tile**