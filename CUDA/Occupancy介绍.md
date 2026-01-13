# 一、Occupancy 的本质定义（非常关键）

## 1.1 数学定义

> **Occupancy = 当前 SM 上活跃 Warp 数 / 该 SM 支持的最大 Warp 数**
$$\text{Occupancy} = \frac{\text{Active Warps per SM}}{\text{Max Warps per SM}}$$  
	
**注：不足一个block的warp会被丢弃，不参与Occupancy的计算。**
示例（Ampere）：
- 最大 warp 数：64
- 实际驻留 warp：32  
    → Occupancy = 50%
### 1.2 Occupancy 的真实作用
Occupancy **不是性能指标本身**，而是：
> **用于隐藏访存与流水线延迟的调度资源**

GPU 通过 **warp-level latency hiding** 工作：
- 一个 warp stall（等内存）
- 调度器切换到另一个 ready warp

**Occupancy 决定了“可切换 warp 的数量上限”**
# 二、Occupancy 的硬件边界（架构事实）
## 2.1 SM 的硬性上限（Ampere 示例）

| 资源            | 每 SM 上限        |
| ------------- | -------------- |
| Threads       | 2048           |
| Warps         | 64             |
| Blocks        | 32             |
| Registers     | 65536 × 32-bit |
| Shared Memory | 164 KB（可配置）    |
> Occupancy **永远不可能超过这些物理上限**

# 三、决定 Occupancy 的四大资源约束
Occupancy 实际上是**四个约束取最小值**的结果。**不足一个block的warp会被丢弃，不参与Occupancy的计算。**
## 3.1 Block 数限制
$$\text{Active Blocks}_{threads}

\left\lfloor  
\frac{\text{Max Threads per SM}}{\text{Threads per Block}}  
\right\rfloor  $$
示例：
- 1024 threads / block  
  → SM 只能放 2 blocks  
  → 最多 64 warps（刚好满，Occupancy100%）
- 32 threads / block
  → 理论为 64 block，但
  → 最多 32 block（Occupancy=32/64= 50%）
  → 小 block（如 32 threads）会被 block 上限限制
## 3.2 Register 使用限制（最常见瓶颈）
### 3.2.1 物理事实
- Registers **按 warp 分配**
- 每个线程使用 `R` 个寄存器
$$\text{Regs per Block}
=
R \times \text{Threads per Block}$$  
$$\text{Active Blocks}_{regs}

\left\lfloor  
\frac{\text{Regs per SM}}{\text{Regs per Block}}  
\right\rfloor  $$
### 3.2.2 示例
- 80 registers / thread，256 threads / block，65536 regs / SM
- 每个 warp 寄存器 = 65536 / (80 * 32) = 25.6
- 每 block 寄存器 = 80 × 256 = 20480，SM 总寄存器 = 65536
  → 只能放 3 blocks（61440）
  → 3 × 8 warps = 24 warps
  → Occupancy = 24 / 64 = 37.5%
### 3.4 Shared Memory 使用限制
$$\text{Active Blocks}_{smem}

\left\lfloor  
\frac{\text{Shared Memory per SM}}{\text{Shared per Block}}  
\right\rfloor  $$
示例：
- 每 block 使用 48 KB，256 threads / block
- 每 SM 164 KB  
    → 只能放 3 blocks
    → Occupancy = 24 / 64 = 37.5%
### 3.5 综合公式（概念）
$$\text{Active Blocks per SM}
=
\min(  
B_{threads},  
B_{regs},  
B_{smem},  
B_{max}  
)  $$
# 四、Occupancy ≠ 性能（必须强调）
## 4.1 常见误区

|误区|纠正|
|---|---|
|Occupancy 越高越快|错|
|100% Occupancy 是目标|错|
|低 Occupancy 一定慢|错|
## 4.2 什么时候高 Occupancy 有用？

|场景|需求|
|---|---|
|内存受限|高 occupancy 隐藏 DRAM 延迟|
|分支多|更多 warp 填补空泡|
|长 latency 指令|需要 warp 切换|
## 4.3 什么时候低 Occupancy 反而更快？

| 场景             | 原因             |
| -------------- | -------------- |
| 计算密集           | 指令级并行 > warp 级 |
| Tensor Core    | pipeline 饱和优先  |
| 高寄存器需求         | 减少 spill       |
| L1 / cache 命中高 | 延迟低            |
典型案例：
- Tensor Core kernel
- cuBLAS GEMM
- cuDNN convolution
Occupancy 常在 **25%–50%**
# 五、编译期与运行期对 Occupancy 的影响
## 5.1 编译期因素

| 因素              | 影响          |
| --------------- | ----------- |
| `-maxrregcount` | 人为限制寄存器     |
| 内联展开            | ↑ registers |
| 循环展开            | ↑ registers |
| 使用 double       | ↑ registers |
| 大结构体            | ↑ registers |
## 5.2 运行期因素

|因素|影响|
|---|---|
|动态 shared memory|↓ blocks|
|Launch 参数|直接决定|
|多 kernel 并发|SM 资源竞争|
# 六、实际Occupancy

| 维度                  | 理论 Occupancy             | 实际（Achieved）Occupancy            |
| ------------------- | ------------------------ | -------------------------------- |
| 本质                  | **资源允许的上限**              | **调度器实际使用情况**                    |
| 是否静态                | 是（launch 时确定）            | 否（运行中统计）                         |
| 是否受控制流影响            | 否                        | 是                                |
| 是否受 memory stall 影响 | 否                        | 是                                |
| 计算口径                | Active warps / Max warps | Issued / Active / Eligible warps |
| 是否等于性能              | 否                        | 仍然不等于                            |
## 6.1 定义
在 NVIDIA 工具里，更准确的名字是：  
**Achieved Active Warp Occupancy** 或 **Issued Warp Occupancy**

> **定义**：在运行过程中，实际处于可执行或被发射状态的 warp 比例

## 6.2 影响因素
实际 Occupancy 会被以下因素 **动态拉低**：
1. **Memory Stall**：等待 global / shared / L2
2. **Instruction Dependency**：RAW / WAW hazard
3. **Branch Divergence**：warp 部分线程 inactive
4. **Execution Unit 饱和**：scheduler 无法发射
5. **Barrier / __syncthreads()**
## 6.3 举例
![Pasted image 20260104152620](Pasted%20image%2020260104152620.png)
图片出处：[比飞鸟贵重的多-CUDA调优指南](https://www.bilibili.com/video/BV1EhrTYcEdf/?spm_id_from=333.1387.collection.video_card.click&vd_source=058215bc88ce2096996ca1d20cfeab0a) 
# 七、工程化调优流程
## 7.1 标准步骤
1. **先保证正确性**
2. 通过 profiler 判断瓶颈：
    - memory-bound？
    - compute-bound？
3. 查看：
    - register usage
    - occupancy
4. 判断是否需要：
    - 提高 occupancy
    - 或降低 occupancy 换 ILP
## 7.2 常见调优手段

| 目标          | 手段           |
| ----------- | ------------ |
| ↑ occupancy | 减少 registers |
| ↓ registers | 减少展开         |
| ↓ spill     | 提高 registers |
| ↑ 吞吐        | 增大 block     |
| ↑ cache     | 降低 occupancy |
# 八、一个完整示例分析

```cpp
__global__ void kernel(float* a) {
    float x[32];   // 寄存器爆炸
    ...
}
```
- 编译后：
    - 120 registers / thread
    - Occupancy ≈ 25%
- 优化：
    - 使用 shared memory
    - 或缩小数组  
        → Occupancy 提升但不一定更快
# 九、总结一句话

> **Occupancy 是“能同时驻留多少 warp”，  
> 性能是“这些 warp 是否在做有用的工作”。**