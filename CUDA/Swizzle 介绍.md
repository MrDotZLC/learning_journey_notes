# 一、硬件视角：为什么 GPU “需要” swizzle
## 1. Shared Memory 的物理结构
以主流 NVIDIA 架构（Volta+）为例：
* Shared memory 被划分为 **32 个 bank**
* 每个 bank：
  * 宽度通常为 **4 bytes（32-bit）**
* 一个 warp = 32 threads
  → **理想情况**：warp 中每个线程命中一个不同的 bank
### 地址到 bank 的映射
```
bank_id = (address / 4) % 32
```
👉 **问题根源**：
当多个线程访问地址 stride 相同、对齐关系相同 → bank_id 冲突。
## 2. 为什么二维 tile 特别容易冲突
典型 GEMM tile：
```cpp
__shared__ float A[BM][BK];  // row-major
```
访问模式：
```cpp
A[ty][k]
```
当：
* `ty` 不变
* `k` 连续
  → stride = BK
  → 若 BK 是 32 的倍数，**整 warp 命中同一个 bank**
这是 swizzle 最常见的触发点。
# 二、Swizzle 的数学本质
**Swizzle = 一个低成本、可逆的索引置换函数**
满足：
1. 逻辑访问顺序不变
2. 物理地址分布更均匀
3. warp 内 bank_id 分散
常见形式：
### 1️⃣ XOR swizzle
```cpp
new_x = x ^ f(y)
```
### 2️⃣ Bit permutation
```cpp
new_idx = (idx & mask1) | ((idx << s) & mask2)
```
### 3️⃣ Padding swizzle
```cpp
stride = original_stride + 1
```
👉 GPU 偏爱 **位运算 swizzle**：快、无分支、可编译期展开。
# 三、Shared Memory Swizzle（最核心）
## 1. 经典 32×32 tile 冲突分析
```cpp
__shared__ float tile[32][32];
float v = tile[ty][tx];
```
若 `tx` = lane_id：
```
address = base + (ty * 32 + tx) * 4
bank = (ty * 32 + tx) % 32 = tx
```
👉 没冲突（理想情况）
但转置时：
```cpp
float v = tile[tx][ty];
```
```
bank = (tx * 32 + ty) % 32 = ty
```
👉 **warp 内所有线程访问同一个 bank**（32-way conflict）
## 2. Padding Swizzle（结构级）
```cpp
__shared__ float tile[32][33];
```
```
bank = (tx * 33 + ty) % 32
```
33 ≡ 1 (mod 32)
→ bank = (tx + ty) % 32
→ 完全打散
优点：
* 简单
* 易理解
缺点：
* 多占 shared memory
* 不适合 Tensor Core 对齐
## 3. XOR Index Swizzle（逻辑级，主流）
```cpp
int sx = tx ^ ((ty & 1) << 4);
float v = tile[ty][sx];
```
### 原理
* `(ty & 1)` 控制是否 swizzle
* `<< 4` 影响 bank 高位
* 保证：
  ```
  bank = (ty * stride + sx) % 32
  ```
  warp 内均匀分布
👉 **Ampere / Hopper kernel 中最常见**
# 四、Warp-Level Swizzle（计算模式）
## 1. Butterfly / XOR 网络
```cpp
int peer = lane ^ 1;
val += __shfl_sync(0xffffffff, val, peer);
```
这是一个 **对数复杂度通信网络**：
```
lane 0 ↔ 1
lane 2 ↔ 3
...
```
用于：
* reduction
* prefix-sum
* FFT
* attention score 累加
## 2. Lane Mapping Swizzle
```cpp
int lane = threadIdx.x & 31;
int row = lane >> 3;
int col = lane & 7;
```
👉 通过 bit swizzle：
* 一个 warp 映射为 8×4 或 16×2 tile
* 每个线程“假装”是二维线程
# 五、Tensor Core Swizzle（工业级）
## 1. 为什么 Tensor Core 必须 swizzle
Tensor Core：
* 按 **fragment** 取数据
* fragment 中：
  * 每个 lane 负责固定元素
* 若按普通 row-major：
  * bank conflict
  * fragment 装载失败
## 2. WMMA 中的隐藏 swizzle
```cpp
wmma::load_matrix_sync(a_frag, shmem_ptr, stride);
```
背后发生：
* index permutation
* lane → element 映射
* shared memory swizzle
👉 **你看到的是 row-major，硬件看到的是 swizzled layout**
## 3. 手写 MMA（CUTLASS 风格）
典型 layout 名称：
* `RowMajorInterleaved`
* `ColumnMajorTensorOp`
* `SwizzledSharedLayout`
本质都是：
```
logical (m, n)
→ swizzle(m, n)
→ physical address
```
# 六、Global Memory / Cache Swizzle
## 1. Block-linear / Z-order
```
(x, y) → morton(x, y)
```
作用：
* 提升 2D spatial locality
* 常见于：
  * 图形
  * Texture
  * L2 cache
CUDA 中：
* 显式少
* 多为硬件隐式 swizzle
# 七、如何判断“该不该 swizzle”
### Nsight Compute 指标
重点看：
* `shared_load_bank_conflicts`
* `shared_store_bank_conflicts`
* `sm__pipe_tensor_active`
### 决策流程
1. 先 tiling + coalescing
2. 再 padding
3. 最后 XOR swizzle
4. Tensor Core → 查官方 layout
# 八、工程经验总结
> **Swizzle 是“硬件友好型作弊”**
> 不改变算法，只改变“看起来的顺序”，让 GPU 跑得更快。
**牢记三点**：
1. Swizzle 永远服务于 **warp**
2. Swizzle 本质是 **bit manipulation**
3. Swizzle 强依赖架构，需 profile 验证
