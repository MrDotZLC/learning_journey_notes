## 一、Achieved Occupancy（实际占用率）
### 1. 定义
> **Achieved Occupancy = 实际在执行期间活跃的 warp 数 / 理论可驻留的最大 warp 数**
注意：
- **不是你在编译期算出来的 occupancy**
- 是运行时统计结果（时间平均）
### 2. 它反映什么
- **SM 中“同时可切换的 warp 数量”**
- 决定了 **warp-level latency hiding 能力**
### 3. 高 / 低意味着什么

| 数值     | 含义                 |
| ------ | ------------------ |
| ≥ 60%  | 通常足够隐藏大多数延迟        |
| 30–60% | 边缘状态，依赖访存模式        |
| < 30%  | 明显 TLP 不足，线程数不够或提高 |
⚠ **Achieved occupancy 高 ≠ 性能一定好**
### 4. 常见限制因素
- 寄存器使用过多
- shared memory 占用过大
- block 太小
- 指令依赖严重导致 warp 早退
## 二、DRAM Utilization（显存利用率）
### 1. 定义
> **DRAM Utilization = 实际 DRAM 带宽 / 理论峰值带宽**
### 2. 它反映什么
- kernel 是否 **带宽受限（memory-bound）**
- 是否真正把显存“跑满”
### 3. 高 / 低意味着什么

| 数值     | 含义          |
| ------ | ----------- |
| ≥ 70%  | 带宽瓶颈 kernel |
| 30–70% | 混合型         |
| < 30%  | 计算受限或访存效率低  |
### 4. 低 DRAM 利用率的典型原因
- 访问不合并（uncoalesced）[合并访存介绍](%E5%90%88%E5%B9%B6%E8%AE%BF%E5%AD%98%E4%BB%8B%E7%BB%8D.md)
- L2/L1 命中率高（其实是好事）
- kernel 太短（启动 / 尾部开销占比高）
## 三、SM Utilization（SM 利用率）
### 1. 定义
> **SM Utilization = 至少有一个 warp 在 SM 上执行的周期占比**
### 2. 它反映什么
- **GPU 是否“忙着干活”**
- 是否存在明显空转
### 3. 高 / 低意味着什么

|数值|含义|
|---|---|
|≥ 90%|GPU 很忙|
|70–90%|基本健康|
|< 70%|SM 经常空闲|
### 4. 低 SM 利用率的常见原因
- kernel launch 配置不当
- grid 太小（不能覆盖所有 SM）
- warp stall 严重（memory / dependency）
## 四、Warp Issue Efficiency（warp 发射效率）
### 1. 定义
> **Warp Issue Efficiency = 实际发射 warp 指令的 cycle / 理论可发射 cycle**
或者理解为：
> **一个 cycle 中，scheduler 是否成功给执行单元发射了 warp**
### 2. 它反映什么（非常关键）
- **调度器是否“有活可发”**
- 是 **延迟隐藏是否成功** 的直接体现
### 3. 高 / 低意味着什么

|数值|含义|
|---|---|
|≥ 80%|pipeline 填得很满|
|50–80%|有一定 stall|
|< 50%|大量周期浪费|
### 4. Warp issue efficiency 低的根因
- 可运行 warp 不足（低 occupancy）
- 指令依赖（ILP 不足）
- memory latency（cache miss）
## 五、L2 Hit Rate（L2 缓存命中率）
### 1. 定义
> **L2 Hit Rate = L2 cache 命中访问 / 总 L2 访问**
### 2. 它反映什么
- 全芯片层级的 **数据复用效率**
- DRAM 压力大小
### 3. 高 / 低意味着什么

|数值|含义|
|---|---|
|≥ 70%|数据复用好，DRAM 压力小|
|40–70%|正常|
|< 40%|流式 / 随机访问|
### 4. 重要反直觉点
> **L2 hit rate 高，DRAM utilization 低，可能是好事**
说明：
- 数据大多被 L2 吃掉
- kernel 不再带宽受限
## 六、这 5 个指标之间的因果链（非常重要）
可以按下面顺序理解：
```
Occupancy
   ↓
Warp availability
   ↓
Warp Issue Efficiency
   ↓
SM Utilization
   ↓
Throughput
```
同时：
```
Memory Access Pattern
   ↓
L2 Hit Rate
   ↓
DRAM Utilization
   ↓
Memory Stall
   ↓
Warp Issue Efficiency
```
## 七、典型“性能诊断模式”
### 模式 1：低 occupancy + 低 issue efficiency
➡ **TLP 不足[如何提高TLP和ILP](%E5%A6%82%E4%BD%95%E6%8F%90%E9%AB%98TLP%E5%92%8CILP.md)**
- 减寄存器
- 减 shared memory
- 增 block 数
### 模式 2：高 occupancy + 低 issue efficiency
➡ **memory / dependency stall**
- 看 L2 hit rate
- 看 DRAM utilization
- 考虑 ILP、prefetch[如何提高TLP和ILP](%E5%A6%82%E4%BD%95%E6%8F%90%E9%AB%98TLP%E5%92%8CILP.md)
### 模式 3：高 SM utilization + 高 DRAM utilization
➡ **典型 memory-bound kernel**
- 优化访存
- 提高算强（arithmetic intensity）
### 模式 4：低 DRAM utilization + 低 SM utilization
➡ kernel 太小 / launch 开销主导
## 八、一句话工程总结
> - **Achieved Occupancy**：我有多少 warp 可切
> - **Warp Issue Efficiency**：我每个周期能不能把 warp 发出去
> - **SM Utilization**：SM 有没有在干活
> - **L2 Hit Rate**：数据有没有被缓存住
> - **DRAM Utilization**：显存是不是瓶颈