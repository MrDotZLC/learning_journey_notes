
# 0. 问题定义
## 0.1 什么是 LLM Serving？
**LLM Serving（Large Language Model Serving）**  
指的是：
> 在**在线系统**中，为多个用户并发地运行大语言模型推理，并以**低延迟、可预测性能**返回结果。

与训练或离线推理不同，Serving 的核心挑战在于：
- 请求是**实时到达**的
- 输入 / 输出长度**事先未知**
- 用户期望**流式返回（streaming）**
## 0.2 Serving 系统的核心性能指标（首次定义）
- **吞吐（Throughput）**  
    单位时间内生成的 token 数，常用指标：tokens / second
- **TTFT（Time To First Token）**  
    从请求到达，到第一个 token 返回的时间
- **TPOT（Time Per Output Token）**  
    相邻两个 token 之间的平均时间间隔
- **p99 latency**  
    99% 请求的延迟上界，用于衡量**长尾稳定性**
## 0.3 Serving 面临的三大不可控因素
1. 请求到达时间不可预测
2. Prompt（输入文本）长度不可预测
3. 输出 token 数不可预测

> **Serving 优化的目标不是“消除不可控”，  
> 而是把不可控限制在可管理的边界内。**
# 1. 朴素 Batch 推理（最原始方案）
## 1.1 什么是 Batch 推理？
**Batch 推理**指的是：
> 把多个请求拼成一个 batch，  
> 一次性送入 GPU 进行 forward 计算。

这是训练和离线推理中最常见的方式。
## 1.2 朴素 Serving 架构
```text
请求队列
 → 收集 N 个请求
 → pad 到最大长度
 → 执行一次 model.forward
 → 返回结果
```
## 1.3 为什么该方案在 Serving 中失败？
- **Prefill 和 Decode 混在一起**（后文定义）
- 一个超长请求 → 整个 batch 退化
- batch 越大，TTFT 越高
- p99 latency 没有上界

> 该方案**只适合离线，不适合线上 Serving**
# 2. 关键分水岭 —— Prefill / Decode 拆分
## 2.1 什么是 Prefill？
**Prefill（也称 Prompt Encoding）**  
指的是：
> 模型第一次看到完整 prompt 时，对所有 token 进行一次 forward 计算，  
> 并生成后续生成所需的 KV Cache。

特征：
- Attention 复杂度是 **O(L²)**（L 为 prompt 长度）
- 计算量大，但只执行一次
- 对延迟不敏感
## 2.2 什么是 Decode？
**Decode（也称 Autoregressive Generation）**  
指的是：
> 在已有 KV Cache 的基础上，每一步生成 **1 个新 token**。

特征：
- 每一步 Attention 复杂度是 **O(T)**（T 为已生成 token 数）
- 单步计算轻，但频率极高
- 对延迟极其敏感（直接影响 streaming）
## 2.3 为什么必须拆分？

|维度|Prefill|Decode|
|---|---|---|
|计算规模|大|小|
|并行度|高|低|
|延迟敏感性|低|极高|
> **不拆分，Decode 一定会被 Prefill 阻塞**

这是 Serving 架构中的**第一条铁律**。
# 3. Prefill 阶段的工程化优化（吞吐优先）
## 3.1 Prefill Scheduler 的角色定义
**Prefill Scheduler**  
负责决定：
- 哪些请求进入 Prefill
- 何时执行 Prefill
- Prefill batch 的大小
## 3.2 Prefill 的设计原则（首次明确）
- 可排队（允许延迟）
- 可 padding
- 大 batch
- **不能抢占 Decode 的算力**
## 3.3 常见 Prefill 优化手段（解释先行）
- **Prompt Length Bucket**  
    按 prompt 长度区间（如 512 / 1k / 2k）分组，减少 padding
- **Admission Control（准入控制）**  
    在 Decode 压力过大时，暂缓新的 Prefill
# 4. Decode 阶段的 Continuous Batching
## 4.1 什么是 Continuous Batching？
**Continuous Batching（连续批处理）**  
指的是：
> batch 不再固定，  
> 请求可以在任意时间加入或离开 batch。

这是 Decode 阶段的核心机制。
## 4.2 为什么 Decode 需要 Continuous Batching？
- 单请求每 step 只生成 1 token
- 单请求无法填满 GPU
- 静态 batch 会快速 shrink
## 4.3 初步收益
- GPU 空转显著减少
- Decode 吞吐明显上升
- Streaming 更平滑
## 4.4 新问题：Attention 长度失控

> Continuous batching **并没有解决**  
> batch 内 attention 计算量差异巨大的问题。
# 5. Decode 阶段“长度问题”的本质
## 5.1 两种“长度”的首次严格区分
- **Prompt 长度**  
    输入文本的 token 数，仅影响 Prefill
- **Generation Step（生成步数）**  
    已生成 token 的数量，决定 Decode Attention 的循环次数
## 5.2 关键结论
> Decode 的性能瓶颈  
> **只与 generation step 有关，与 prompt 无关**

#  6. Position Batching（按生成位置分组）
## 6.1 什么是 Position Batching？
**Position Batching**  
指的是：
> 在 Decode 阶段，  
> batch 内请求按 generation step 对齐或近似对齐。

目标是：
- 让 batch 内 attention 计算复杂度接近
#  7. Step 是否必须完全一致？
## 7.1 Step 容差（Δstep）的定义
**Δstep（Step Tolerance）**  
指的是：
> batch 内允许的最大 generation step 差值。

## 7.2 工业经验结论

| Δstep | 评价   |
| ----- | ---- |
| 0     | 过于严格 |
| **1** | 最优   |
| 2     | 可接受  |
| ≥3    | 退化   |
#  8：为什么 Δstep = 1 是工程最优？
## 8.1 收益
- batch 填充率提升
- 调度等待减少
- p99 latency 收敛
## 8.2 代价
- 少量 padding
- GPU kernel 内部分支
#  9. PagedAttention（核心系统前提）
## 9.1 传统 KV Cache 的问题
- 必须连续显存
- 长度需提前预估
- 显存碎片严重
## 9.3 什么是 PagedAttention？
**PagedAttention**  
是一种将 KV Cache 按固定大小 block 分页管理的机制：
- 固定大小 block（如 16 / 32 token）
- 全局 block pool
- 每请求维护 block table（逻辑页表）
#  10. Decode Kernel 如何处理 step 不一致
## 10.1 Kernel 的抽象执行逻辑
```cpp
for each request:
    for each kv_block:
        if block_start >= request.step:
            break
        attend()
```
## 10.2 为什么 divergence 可控？
- divergence 发生在 block 粒度
- Δstep 很小
- attention 主要受内存带宽限制
#  11. Prefill / Decode 的流水线并行
## 11.1 什么是流水线并行？
**Pipeline Parallelism（流水线并行）**  
指 Prefill 与 Decode 在时间上交错执行，而非串行。
## 11.2 优先级规则
> **Decode 永远高于 Prefill**

#  12. 最终调度器骨架
```python
while True:
    decode_batch = collect_decode_tokens(delta_step=1)
    launch_decode(decode_batch)

    if decode_pressure_low():
        prefill_batch = collect_prefill_requests()
        launch_prefill(prefill_batch)

    recycle_kv_cache()
```
#  13：p99 latency 的最终控制手段
## 13.1 p99 的根源
> batch 内 **最长 attention loop**
## 13.2 Position batching 的本质作用
- 控制 batch 内计算复杂度上界
- 隔离极端请求
- 让 latency 分布可预测
#  14. 工程铁律
1. Prefill / Decode 必须拆
2. Decode 永远优先
3. Decode 使用 Continuous Batching
4. Position batching 控制 attention 复杂度
5. Δstep ≈ 1
6. 必须使用 PagedAttention
7. 优化目标是 **p99 稳定性**
## 15. 总结
> **LLM Serving 的核心不是“更快的模型”，  
> 而是“更可控的调度”。**

