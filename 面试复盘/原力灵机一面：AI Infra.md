## 面经：

## 针对性知识补充
### 1. 你现在的能力分布

按照你之前准备过的内容，我会这样判断：

|知识|你的状态|这 2 小时|
|---|---|---|
|CUDA execution / Memory / Kernel|**较强**|不复习|
|GEMM / CUTLASS / Tensor Core|**较强**|不复习|
|FlashAttention|**较强**|只准备如何和系统结合|
|TensorRT|**较强**|不复习基础|
|KV Cache|**较强**|准备系统设计|
|PagedAttention|**较熟**|准备 scheduler / memory manager|
|Continuous Batching|**较熟**|准备综合题|
|Prefill / Decode / TTFT / TPOT|**较熟**|准备场景题|
|TP / PP / DP|**基础有**|**重点补通信关系**|
|FSDP / ZeRO|**不够系统**|**重点补**|
|NCCL Collective|**知道概念**|**重点补**|
|NCCL Hang / Jitter|**明显薄弱**|**最高优先级**|
|RDMA|**明显薄弱**|**重点补**|
|RoCE / PFC / ECN|**明显薄弱**|**重点补到能回答场景题**|
|GPU-NIC / PCIe Topology|**薄弱**|**重点补**|
|大规模训练 Scaling|**薄弱**|**重点补**|
|生产环境故障排查|**薄弱**|**重点补**|
|RL / 机器人|**不需要深挖**|只准备系统链路|
|Linux/C++|**已有后端基础**|只准备排障工具|

尤其是 **NCCL/RDMA**，这份 JD 已经明确把“**NCCL hang、抖动分析**”写进任职要求，所以它不是普通加分项，而是非常明显的面试信号。

---

### 2. 120 分钟具体安排

不要平均分配。

|时间|内容|目标|
|--:|---|---|
|0–10 min|分布式训练总图|DP/TP/PP/FSDP/ZeRO 一张图|
|10–30 min|NCCL Collective + 通信模型|AllReduce/AllGather/RS/AllToAll|
|30–50 min|NCCL Hang + RDMA/RoCE|**最高优先级**|
|50–65 min|Scaling / GPU-NIC / NUMA|多机多卡综合题|
|65–85 min|LLM Serving 系统设计|KV + Scheduler + Batching|
|85–105 min|5 道综合场景题|练回答框架|
|105–120 min|项目串联 + 自我介绍追问|把已有 CUDA 项目接到 AI Infra|

下面直接给你这 2 小时应该记住的内容。

---

### 3. 0–10 分钟：只建立一张“分布式训练地图”

你不需要重新学习 DP/TP/PP。

只记住：

```text
                 Distributed Training
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
       DP               TP               PP
   Batch 切分       Tensor 切分        Layer 切分
        │                │                │
    AllReduce       AllReduce/AG        P2P
        │                │                │
        └────────────────┼────────────────┘
                         ↓
                     FSDP / ZeRO
                         │
                  Parameter Sharding
                         │
                 AllGather / RS
```

然后记一句：

> **DP 主要解决数据扩展，TP 解决单层模型放不下一张 GPU 或提高单层计算并行度，PP 解决模型跨层切分，FSDP/ZeRO 解决参数、梯度和 optimizer state 的显存问题。**

面试官继续问：

> “这些可以组合吗？”

直接回答：

> 可以，例如 DP + TP + PP 是典型 3D Parallel，MoE 还可以叠加 Expert Parallel；真正的系统设计需要根据模型规模、GPU 显存、NVLink/PCIe/NIC topology 和通信开销决定各维度 degree。

---

### 4. 10–30 分钟：NCCL 只学“通信关系”

你不需要重新学 NCCL API。

#### 4.1 四个 Collective 必须张口就来

##### AllReduce

```text
GPU0: A ─┐
GPU1: B ─┼─→ A+B+C+D → 所有 GPU
GPU2: C ─┤
GPU3: D ─┘
```

##### AllGather

```text
GPU0: A
GPU1: B
GPU2: C
GPU3: D

↓ AllGather

每张 GPU：A B C D
```

##### ReduceScatter

```text
A B C D
E F G H
I J K L
M N O P

↓ ReduceScatter

GPU0: A+E+I+M
GPU1: B+F+J+N
GPU2: C+G+K+O
GPU3: D+H+L+P
```

##### AllToAll

```text
GPU0 ──→ GPU0/GPU1/GPU2/GPU3
GPU1 ──→ GPU0/GPU1/GPU2/GPU3
GPU2 ──→ GPU0/GPU1/GPU2/GPU3
GPU3 ──→ GPU0/GPU1/GPU2/GPU3
```

最重要的一句话：

> **AllGather 是“把数据复制给所有人”，ReduceScatter 是“规约后各拿一部分”，AllToAll 是“每个人给每个人不同的数据”，MoE 的 Expert Parallel 最典型的是 AllToAll。**

NCCL 当前官方文档仍把这些作为核心 collective，并强调 NCCL 是 topology-aware 的 GPU collective communication library，而不是完整的进程管理框架。([NVIDIA Docs](https://docs.nvidia.com/deeplearning/nccl/index.html?utm_source=chatgpt.com "NVIDIA Deep Learning NCCL Documentation"))

---

#### 4.2 必须会回答：AllReduce 为什么可以拆？

AllReduce=ReduceScatter+AllGather

回答：

> ReduceScatter 先把全局 reduction 结果分片到各 GPU，AllGather 再把这些分片广播给所有 GPU，所以最终每张 GPU 都拿到完整的 reduction result。

这句话能自然接到：

> FSDP 为什么喜欢 ReduceScatter？

因为 FSDP 不需要每张 GPU 都保留完整 gradient。

---

### 5. 30–50 分钟：这是你最应该补的——NCCL Hang

如果面试官问：

> **“训练任务 NCCL hang 了，你怎么排查？”**

你直接按照下面这套说。

---

#### 5.1 第一层：先判断是“应用问题”还是“系统问题”

```text
NCCL Hang
│
├── ① 看 Rank 日志
│   │
│   ├── 所有 Rank 都卡在同一个 Collective
│   │   └── 继续检查 Collective 是否一致
│   │
│   └── 某个 Rank 没进入 Collective
│       └── 查 DataLoader / CUDA Kernel / OOM / 异常 / 代码分支
│
├── ② 检查 Collective 一致性
│   │
│   ├── Sequence 是否一致
│   ├── Tensor Shape / Dtype 是否一致
│   ├── Step 数是否一致
│   └── 不同 Rank 是否执行了不同代码路径
│       └── 不一致 → Collective Mismatch → Hang
│
├── ③ 检查 CUDA / GPU
│   │
│   ├── CUDA OOM
│   ├── illegal memory access
│   ├── device-side assert
│   ├── CUDA Kernel 是否异常
│   ├── nvidia-smi
│   └── dmesg → NVRM / Xid
│       └── 怀疑异步 CUDA 错误 → CUDA_LAUNCH_BLOCKING=1
│
├── ④ 打开 Debug 日志
│   │
│   ├── TORCH_DISTRIBUTED_DEBUG=DETAIL
│   ├── NCCL_DEBUG=INFO
│   └── NCCL_DEBUG_SUBSYS=INIT,COLL,NET
│       └── 对比不同 Rank 的 Collective Sequence
│
├── ⑤ 判断单机 / 多机
│   │
│   ├── 单机也 Hang
│   │   └── 重点查 CUDA / GPU / NCCL / 拓扑
│   │
│   └── 单机正常，多机 Hang
│       └── 重点查网络
│           ├── InfiniBand / RoCE
│           ├── GPUDirect RDMA
│           ├── NIC / HCA
│           ├── NCCL NET
│           └── 网络配置
│
├── ⑥ 检查 GPU / NIC 拓扑
│   │
│   └── nvidia-smi topo -m
│       ├── GPU ↔ GPU
│       ├── GPU ↔ NIC
│       ├── NVLink
│       ├── PCIe
│       └── NUMA
│
├── ⑦ 用 nccl-tests 隔离问题
│   │
│   ├── AllReduce
│   ├── AllGather
│   └── ReduceScatter
│       │
│       ├── nccl-tests 正常
│       │   └── 重点查训练代码 / FSDP / Collective 逻辑
│       │
│       └── nccl-tests 也 Hang
│           └── 重点查 NCCL / CUDA / GPU / 网络 / 驱动
│
└── ⑧ 如果是 FSDP
    │
    ├── Forward
    │   └── All-Gather
    │
    ├── Backward
    │   ├── All-Gather
    │   └── Reduce-Scatter
    │
    └── 检查
        ├── FSDP Wrapping 粒度
        ├── Prefetch
        ├── Communication-Computation Overlap
        └── 所有 Rank 的 Collective Sequence 是否一致
```

非常重要：

> **不要一看到 NCCL hang 就修改 NCCL 环境变量。**

NVIDIA 当前 NCCL troubleshooting 文档明确建议先通过 diagnostics 缩小范围，再考虑 tuning；NCCL 2.31 起还提供 active diagnostics，可以实际验证 GPU/GPU 和 GPU/network communication path。([NVIDIA Docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html?utm_source=chatgpt.com "Troubleshooting — NCCL 2.32.3 documentation"))

---

#### 5.2 第二层：GPU / PCIe / NVLink

```text
GPU P2P
 ↓
NVLink / PCIe
 ↓
GPU-NIC topology
```

重点知道：

```bash
nvidia-smi topo -m
```

以及：

```bash
nvidia-smi topo -p2p p
```

NCCL 本身依赖 GPU Direct，并会根据 GPU/NIC 的 PCIe topology 选择通信路径；PCIe ACS/IOMMU 配置异常可能导致性能下降甚至 hang。([NVIDIA Docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html?utm_source=chatgpt.com "GPU troubleshooting — NCCL 2.32.3 documentation"))

你可以说：

> 如果单机多卡通信异常，我先排除 GPU P2P、NVLink/PCIe topology、ACS/IOMMU 和 GPU 本身的问题。

---

#### 5.3 第三层：NIC / RDMA

如果是多节点：

```text
GPU
 ↓
PCIe
 ↓
NIC
 ↓
RDMA
 ↓
Switch
 ↓
Remote NIC
 ↓
Remote GPU
```

排查：

```bash
ib_write_bw
ib_write_lat
```

以及 NIC counters。

NCCL 当前 diagnostics 已经能够进行 network bandwidth check，并使用 `ib_write_bw` 检查物理 InfiniBand/RDMA 路径。([NVIDIA Docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/diagnostics.html?utm_source=chatgpt.com "Diagnostics — NCCL 2.32.3 documentation"))

---

### 6. 你只需要把 RDMA/RoCE 学到这个程度

不要在这 20 分钟里深入 RDMA verbs。

#### 6.1 RDMA 是什么？

普通网络：

```text
GPU
 ↓
CPU
 ↓
NIC
 ↓
Network
```

GPUDirect RDMA：

```text
GPU Memory
     ↕
    NIC
     ↕
  Network
```

核心：

> **减少 CPU involvement 和额外数据拷贝，让 NIC 可以直接访问 GPU memory。**

---

#### 6.2 InfiniBand vs RoCE

一句话：

> **InfiniBand 是专用高性能网络体系，RoCEv2 是把 RDMA 放到 Ethernet/IP 网络上。**

然后面试官如果问：

> RoCE 为什么容易出现抖动？

回答：

> 因为 RoCE 基于 Ethernet，共享网络存在 congestion；拥塞会导致 queue buildup、packet loss/retransmission 或 ECN/PFC 触发，从而影响 RDMA latency 和 tail latency，最终表现为 NCCL throughput 下降或者 iteration time 抖动。

---

#### 6.3 PFC / ECN

只记：

```text
ECN
 ↓
标记拥塞
 ↓
发送端降低发送压力
```

```text
PFC
 ↓
Pause
 ↓
让上游暂时停止发送
```

不要深挖协议。

---

### 7. 你一定要记住这个“生产 NCCL 排障回答”

面试时可以直接按照这个结构：

> **我会先区分 application-level hang 和 communication/system-level hang。**

> 第一层检查所有 rank 是否存活、是否都进入 collective、collective 顺序以及 count/dtype/communicator 是否一致；如果应用逻辑没问题，再检查 NCCL initialization、GPU P2P、NVLink/PCIe topology 和 GPU-NIC topology。

> 多节点情况下继续检查 NIC、RDMA path 和网络带宽/延迟，例如 `ib_write_bw`、`ib_write_lat`，再检查 RoCE 的 PFC、ECN、NIC counters 和交换机侧 congestion。

> 同时结合 `NCCL_DEBUG=INFO`、`NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,NET` 定位 NCCL 初始化、bootstrap 和网络阶段；新版本 NCCL 还可以使用 RAS/diagnostics 判断 rank 是否异常以及实际通信路径是否健康。([NVIDIA Docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/diagnostics.html?utm_source=chatgpt.com "Diagnostics — NCCL 2.32.3 documentation"))

这个回答已经足够应付相当一部分“生产环境 NCCL hang”问题。

---

### 8. 50–65 分钟：补 Scaling 综合题

这个岗位很可能问：

> **“为什么 8 GPU 扩到 64 GPU 后，性能没有线性提升？”**

你的回答应该是：

Tstep=Tcompute+Tcommunication+Tsynchronization+TpipelineT_{step}=T_{compute}+T_{communication}+T_{synchronization}+T_{pipeline}

GPU 数增加：

Tcompute↓T_{compute}\downarrow

但：

TcommunicationT_{communication}

不一定同比下降。

因此：

> GPU 越多，通信、同步、pipeline bubble、load imbalance 等固定或增长成本占比越高，scaling efficiency 就会下降。

---

#### 8.1 如果问“怎么优化？”

直接说：

```text
Scaling 差
 ↓
1. Communication volume
2. Communication bandwidth
3. Communication latency
4. Compute/Communication overlap
5. Load balance
6. Topology
7. Parallelism degree
```

然后具体：

- TP 尽量放在 NVLink 域内；
- PP 减少跨节点通信；
- DP 跨节点时优化 AllReduce；
- FSDP 调整 shard/wrap 和 communication overlap；
- MoE 关注 AllToAll；
- GPU-NIC affinity；
- RDMA；
- NCCL topology；
- 增加 computation/communication overlap。

---

### 9. 一个非常容易被问的综合题：为什么 TP 不应该无限扩大？

回答：

> TP degree 增大后，每张 GPU 的 computation 降低，但 collective communication 增加；当单 GPU computation 已经足够小，communication latency 和 synchronization overhead 会占主导，因此 scaling efficiency 下降。

用公式：

Tstep=Tcompute(P)+Tcomm(P)T_{step}=T_{compute}(P)+T_{comm}(P)

随着 $P$ 增加：

Tcompute(P)↓T_{compute}(P)\downarrow

但：

Tcomm(P)↑T_{comm}(P)\uparrow

最终出现 diminishing return。

这个逻辑你一定要会，因为它能连接 **TP、NCCL、NVLink、RDMA、Scaling** 五个知识点。

---

### 10. 65–85 分钟：LLM Serving 系统设计

这一块你已有知识，不需要学习新概念，只需要**把零散知识组织成系统**。

如果问：

> **“设计一个低延迟、高吞吐的大模型推理服务。”**

直接画：

```text
Client
  ↓
Load Balancer
  ↓
Request Queue
  ↓
Scheduler
  ↓
Continuous Batching
  ↓
┌───────────────────────┐
│       GPU Worker      │
│                       │
│  Prefill / Decode     │
│         ↓             │
│    KV Cache Manager   │
│         ↓             │
│ Attention / GEMM      │
│         ↓             │
│ CUDA Kernel / Graph   │
└───────────────────────┘
  ↓
Response
```

然后从 5 个方向回答：

#### 调度

Continuous Batching：

> iteration-level 调度，每轮 decode 完成后动态加入/移除 request。

#### 显存

> PagedAttention 把 KV Cache 按 block 管理，减少动态 request 导致的 fragmentation。

#### 计算

> FlashAttention、Tensor Core、kernel fusion、quantization。

#### GPU

> CUDA Graph、memory pool、减少 launch overhead。

#### 指标

> TTFT、TPOT、E2E latency、QPS、GPU utilization、KV Cache utilization、P99。

---

### 11. 重点准备一个你可能被追问的问题：为什么 Batch 越大不一定越好？

回答：

> Batch 增大可以提高 GPU utilization 和吞吐，但会增加 KV Cache 占用、调度等待和单 request latency；对于实时场景还可能导致 queueing delay 增加，因此需要在 throughput 和 TTFT/TPOT/P99 之间做 trade-off。

然后继续：

```text
Batch ↑
 ├─ GPU Utilization ↑
 ├─ Throughput ↑
 ├─ KV Memory ↑
 ├─ Queueing ↑
 └─ P99 Latency ↑
```

这个就是很典型的 AI Infra 思维。

---

### 12. 85–105 分钟：只练 5 道综合场景题

这 20 分钟非常重要。

不要看答案，先自己用 **1～2 分钟/题** 口述。

---

#### 场景题 1：256 GPU 训练突然变慢

> **问题：训练从 64 GPU 扩展到 256 GPU 后，step time 只降低 20%，你怎么定位？**

你的回答框架：

```text
Step Time
 ↓
Compute?
Communication?
Synchronization?
Load Balance?
 ↓
Nsight Systems
 ↓
NCCL Timeline
 ↓
Collective duration
 ↓
Bandwidth / Latency
 ↓
GPU-NIC topology
 ↓
RDMA
 ↓
RoCE congestion
```

最后补：

> 如果通信正常，再看 DP/TP/PP degree 是否合理以及 computation/communication overlap 是否被破坏。

---

### 13. 场景题 2：NCCL 偶发 Hang

> **问题：100 次训练中偶尔有一次 NCCL hang，你怎么处理？**

标准回答：

```text
1. 判断是否 rank dead
2. 检查 collective 顺序
3. 检查 shape/count/dtype
4. 检查 async error
5. NCCL INFO/TRACE
6. RAS
7. GPU P2P
8. GPU-NIC topology
9. RDMA
10. RoCE counters
11. kernel/system log
```

特别加一句：

> **因为是偶发问题，我会重点关注 tail latency、网络 congestion、GPU/NIC error、rank desynchronization，而不是只看平均 bandwidth。**

这句话很像生产环境经验。

---

### 14. 场景题 3：LLM Decode 延迟越来越高

> **问题：刚开始 TPOT 30ms，随着并发增加逐渐变成 80ms，怎么分析？**

你应该回答：

```text
Concurrency ↑
 ↓
Batch ↑
 ↓
KV Cache ↑
 ↓
Memory Bandwidth Pressure ↑
 ↓
Decode Kernel latency ↑
 ↓
Scheduler Queueing ↑
```

然后检查：

- KV Cache 是否接近 capacity；
- batch size；
- sequence length；
- memory bandwidth；
- GPU utilization；
- kernel latency；
- P99；
- scheduler waiting time。

最后：

> Decode 通常比 Prefill 更容易受 memory bandwidth 和 KV Cache access 影响，所以我会重点 profile attention/KV cache，而不是只看 GEMM TFLOPS。

---

### 15. 场景题 4：GPU 利用率只有 40%

不要回答：

> “优化 CUDA kernel。”

正确回答：

```text
GPU Util 40%
 ↓
Nsight Systems
 ↓
GPU idle?
 ↓
CPU scheduling?
Kernel launch?
NCCL?
Memory allocation?
Synchronization?
 ↓
Nsight Compute
 ↓
Kernel:
Compute Bound?
Memory Bound?
Occupancy?
Warp Stall?
```

这和你原来的 CUDA 知识正好能接起来。

---

### 16. 场景题 5：P99 延迟突然增加，但平均延迟没变

这是一个很好的 AI Infra 场景题。

回答：

> 平均值不变而 P99 上升，说明不是所有 request 都变慢，而是 tail request 出现异常，因此我会优先检查 request length、batch composition、queueing delay、KV Cache pressure、GPU contention、网络 congestion 和 CPU scheduling。

然后：

```text
P99 ↑
 ↓
Queueing?
Request Shape?
Batch?
KV Cache?
GPU contention?
Network?
CPU?
```

**这个比“平均 latency 优化”更符合 JD 的稳定性治理。**

---

### 17. 105–120 分钟：把你自己的项目重新包装

这是最后 15 分钟最重要的事情。

你不要把自己包装成：

> “我做过一个完整的大规模 AI Infra 生产系统。”

这很容易被追问穿。

你应该把自己的优势定位成：

> **“我原来的优势是 C++/CUDA 和 GPU 性能优化，近期进一步向 LLM inference/AI Infra 延伸，因此对单 GPU kernel、推理 runtime 和多 GPU 通信之间的性能关系比较关注。”**

然后把自己的项目串成：

```text
CUDA Kernel
 ↓
GPU Performance
 ↓
TensorRT
 ↓
LLM Inference
 ↓
KV Cache
 ↓
Continuous Batching
 ↓
TP
 ↓
NCCL
 ↓
Distributed Inference
```

这样你即使没有真实的大规模集群生产经验，也不会显得知识结构断裂。

---

### 18. 如果面试官追问：“你没有生产集群经验，怎么做 NCCL 故障排查？”

**不要硬装。**

你可以这样回答：

> 我目前没有大规模生产集群的实际值班经验，所以不会把它描述成自己已经处理过生产事故；但我会按照应用、NCCL、GPU topology、NIC/RDMA、网络几个层次定位。比如 NCCL hang，我会先确认 rank 和 collective 是否一致，再通过 NCCL 日志和 RAS 判断 communicator 状态，然后检查 GPU P2P、PCIe/NVLink、GPU-NIC topology，多节点再用 RDMA bandwidth/latency test 和 NIC counters 排查网络路径。

这个回答实际上比：

> “我之前处理过 NCCL hang。”

安全得多。

而且现在 NCCL 官方已经提供 RAS 和 active diagnostics，能够检查 rank 状态、GPU/驱动/NCCL 配置一致性、P2P 和 network path，因此你讲这套方法有明确的官方依据。([NVIDIA Docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/diagnostics.html?utm_source=chatgpt.com "Diagnostics — NCCL 2.32.3 documentation"))

---

### 19. 最后 5 分钟：强制记住这 12 句话

如果进面试前只能看一页，就看这里。

#### ① DP

> **DP 切 Batch，核心通信是 Gradient AllReduce。**

#### ② TP

> **TP 切 Tensor，计算减少，但需要 Collective 同步；TP degree 不能无限扩大。**

#### ③ PP

> **PP 切 Layer，主要问题是 Pipeline Bubble 和 stage imbalance。**

#### ④ FSDP

> **FSDP 通过 Parameter/Gradient/Optimizer Sharding 降显存，用 AllGather 和 ReduceScatter 换通信。**

#### ⑤ AllReduce

> **AllReduce = ReduceScatter + AllGather。**

#### ⑥ MoE

> **Expert Parallel 的核心通信是 AllToAll，因为 Token 需要跨 GPU Dispatch 到对应 Expert。**

#### ⑦ NCCL

> **NCCL 是 GPU collective communication library，本身不是完整的 process launcher。** ([NVIDIA Docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/setup.html?utm_source=chatgpt.com "Setup — NCCL 2.32.3 documentation"))

#### ⑧ RDMA

> **GPUDirect RDMA 的核心是让 NIC 直接访问 GPU memory，减少 CPU 和额外拷贝。**

#### ⑨ RoCE

> **RoCE 把 RDMA 放到 Ethernet 上，因此需要关注 congestion、PFC、ECN 和 NIC counters。**

#### ⑩ NCCL Hang

> **先查 Rank/Collective/Application，再查 GPU P2P/Topology，再查 NIC/RDMA/Network。**

#### ⑪ Continuous Batching

> **Continuous Batching 是 iteration-level scheduling，不是简单扩大 batch size。**

#### ⑫ 性能优化

> **任何性能问题都先拆 Compute、Memory、Communication、CPU、Synchronization、Scheduling，再决定优化方向。**

---

### 20. 这场面试你最应该避免的事情

**不要再花时间：**

- 重新背 CUDA thread hierarchy；
- 重新背 Shared Memory；
- 重新背 FlashAttention；
- 重新背 TensorRT；
- 重新背 Attention；
- 重新背量化；
- 重新背 Transformer。

这些已经不是你这场面试的主要风险。

**真正应该补的是：**

```text
                    你已有优势
                       │
             CUDA / GPU / Inference
                       │
                       ↓
                 ┌───────────┐
                 │ AI Infra  │
                 └───────────┘
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
      Distributed    Network      Serving
       Training       Infra        System
          │            │            │
       FSDP/ZeRO     RDMA/RoCE    Scheduler
       TP/PP/DP      NCCL Hang     KV Cache
          │            │          Batching
          └────────────┼────────────┘
                       ↓
                 综合场景题
                       ↓
              性能 + 稳定性 + 扩展性
```

**如果只能选择一个主题重点突击：选 `NCCL Hang + RDMA/RoCE + 多机多卡性能排查`。**

因为你的 CUDA/推理知识已经能支撑前半场，而这恰好是这份 JD 与你现有知识结构之间最大的缺口；并且 JD 直接点名了 **“NCCL hang、抖动分析”**。

你现在甚至不需要再找资料：NCCL 最新文档已经把 **Diagnostics → GPU/Topology → Networking → Runtime → Performance → RAS** 明确组织成排障路径；其中 RAS 可以监控 rank/process 状态，active diagnostics 可以实际验证 GPU/GPU 和 GPU/network communication path。([NVIDIA Docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html?utm_source=chatgpt.com "Troubleshooting — NCCL 2.32.3 documentation"))
