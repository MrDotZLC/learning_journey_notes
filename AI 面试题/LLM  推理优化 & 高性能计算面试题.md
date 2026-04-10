## 1. GPU 硬件与内存体系

### 1.1 基础硬件架构

- **Q1.** GPU 的 SM（Streaming Multiprocessor）内部结构是什么？Warp 如何调度？
- **Q2.** CUDA 的内存层次（Register → L1/Shared Memory → L2 → HBM/GDDR）各层的带宽与延迟数量级是多少？
- **Q3.** 什么是 Memory Coalescing（内存合并访问）？为什么非合并访问会严重降低性能？
- **Q4.** Shared Memory 的 Bank Conflict 是什么？如何通过 Padding 或 Swizzle 消除？
- **Q5.** H100 / A100 / H20 各自的 HBM 带宽、Tensor Core TFLOPS、NVLink 带宽分别是多少？
- **Q6.** Warp Divergence（束散）对性能的影响及规避方法？

### 1.2 计算访存比分析

- **Q7.** 什么是 Arithmetic Intensity（算术强度）？如何用 Roofline Model 判断一个 Kernel 是 Compute-bound 还是 Memory-bound？
- **Q8.** LLM 推理的 Prefill 阶段和 Decode 阶段分别属于哪种瓶颈？原因是什么？
- **Q9.** GEMV 与 GEMM 的计算访存比差距有多大？为何 Decode 阶段吞吐量受限于显存带宽？

---

## 2. CUDA Kernel 开发与优化

### 2.1 基础 Kernel 实现

- **Q10.** 手写 Warp-level Reduce（Sum / Max）：使用 `__shfl_xor_sync` 实现，说明为什么比 Shared Memory Reduce 更快？
- **Q11.** 手写 Block-level Reduce，需要处理哪些边界情况？
- **Q12.** 如何实现 numerically stable 的 Online Softmax？推导 3-pass → 2-pass → 1-pass 的演化过程。
- **Q13.** 实现 Fused RMSNorm Kernel：为什么要 Fuse，省去了哪些 Global Memory 访问？
- **Q14.** LayerNorm 的 Welford 在线算法如何实现？
- **Q_J.** Warp Reduce 的 `mask` 参数在非满 Warp 场景（Block 尾部）如何正确处理？错误使用会导致什么问题？

### 2.2 GEMM 优化

- **Q15.** 朴素 GEMM 的瓶颈是什么？Tiled GEMM 的核心思路（Shared Memory Tiling）？
- **Q16.** 什么是 Double Buffering（Ping-Pong Buffer）？如何用 `cp.async` / TMA 实现异步数据预取？
- **Q17.** Tensor Core（WMMA / MMA / WGMMA）的使用方式与限制？Hopper 的 WGMMA 与 Ampere MMA 的区别？
- **Q18.** cuBLAS vs CUTLASS vs 手写 Kernel 的选型依据？何时需要手写？
- **Q19.** GEMM-SplitK 分解的适用场景（瘦矩阵 / Decode 阶段小 Batch）？
- **Q_K.** Register Tiling（Thread-level Tiling）的原理是什么？如何在 GEMM 中提升寄存器级数据复用？
- **Q_L.** 什么是 Epilogue Fusion？CUTLASS 的 Epilogue Visitor Tree（EVT）如何将 Bias、Activation、量化融合进 GEMM Kernel？

### 2.3 Kernel Fusion

- **Q20.** Kernel Fusion 的本质收益是什么（减少 HBM Round-trip）？举例说明 FlashAttention 的 Fusion 策略。
- **Q21.** 什么样的算子适合 Fusion？什么情况下 Fusion 反而有害（Register Spilling）？
- **Q22.** CUDA Graph 的作用：如何消除 Kernel Launch Overhead？适用哪些场景？
- **Q_M.** 什么是 Persistent Kernel？与普通 Kernel 的区别是什么？在 LLM 推理中如何应用？

---

## 3. Attention 机制优化

### 3.1 FlashAttention 系列

- **Q23.** 标准 Attention 的内存复杂度为 $O(N^2)$，FlashAttention 如何将其降为 $O(N)$ SRAM 占用？核心思想（Tiling + Online Softmax）？
- **Q24.** FlashAttention-2 相比 FA-1 的改进点：减少非 GEMM FLOPs、改进 Warp 并行策略？
- **Q25.** FlashAttention-3 在 Hopper 架构上的改进：Warp Specialization、异步流水线、WGMMA 的利用？
- **Q26.** 为什么 Decode 阶段的 Attention 退化为 GEMV 问题？此时 FA 的收益是否仍然显著？

### 3.2 Attention 变体

- **Q27.** MHA vs GQA vs MQA 的区别？GQA 在 KV Cache 占用上的收益推导？
- **Q28.** MLA（Multi-head Latent Attention）的核心思路：低秩压缩 KV 的原理与 DeepSeek 中的实现？
- **Q29.** Sparse Attention（如 Sliding Window、BigBird）的适用场景？

### 3.3 MLA 矩阵吸收与位置编码
- Q30. MLA 的矩阵吸收（Absorption）推导：为何推理时可消除 Up-projection 的计算开销？
- Q31. RoPE 与 ALiBi 的原理对比，及其对 KV Cache 复用策略（Prefix Caching）的影响

### 3.4 Decode 阶段 Attention 优化
- Q32. PagedAttention 原理：为何 KV Cache 存在碎片化问题？分页机制如何解决？
- Q33. Flash-Decoding：为何 FA 在 Decode 阶段并行度不足？分块归约如何提升吞吐？

### 3.5 长序列与分布式 Attention
- Q34. Ring Attention / Context Parallelism：超长序列跨设备 Attention 的切分方案与通信分析
- Q35. Multi-head Attention 的 Tensor Parallelism 切分：Column/Row 并行与 GQA 下的特殊处理
- 

---

## 4. KV Cache 管理

### 4.1 核心机制

- **Q30.** KV Cache 的作用与显存增长规律：给定模型参数（层数 $L$、头数 $H$、头维度 $d$、数据类型），推导单请求 $S$ tokens 的 KV Cache 显存占用公式：

$$M_{\text{KV}} = 2 \times L \times H \times d \times S \times \text{sizeof(dtype)}$$

- **Q31.** 为什么传统框架的 KV Cache 存在严重的内存碎片？Internal Fragmentation 与 External Fragmentation 分别指什么？
- **Q30-b.** GQA / MQA 对 KV Cache 显存的节省推导：设 MHA 的 KV 头数为 $H$，GQA 分 $G$ 组，每组共享一对 KV 头，缩减比为 $G/H$；以 LLaMA-3 70B（$H=64$，$G=8$）代入 Q30 的公式，对比 MHA 节省 87.5%，说明 GQA 成为工程默认选择的根本原因。（与 Q27 形成完整闭环）
- **Q30-c.** MLA（Multi-head Latent Attention）的 KV Cache 压缩比推导：缓存低秩向量 $c$（维度 $d_c$）而非展开的 $K, V$，压缩比为 $d_c / (H \cdot d)$；DeepSeek-V2 中 $d_c = 512$，$H \cdot d = 16384$，压缩比约 $1/32$；与 GQA 路径对比，分析两种方案的工程取舍。（与第 3 章 Q28、Q30 形成闭环）
- **Q30-d.** Prefill 阶段与 Decode 阶段 KV Cache 增长行为的差异：Prefill 阶段 $S_{\text{prompt}}$ 个 Token 的 KV 在单次前向中批量写入，显存在 Prefill 结束时跳变至峰值；Decode 阶段每步追加 1 个 Token，线性递增。两阶段对 Block 分配策略的要求不同，说明此差异如何驱动 Chunked Prefill（Q39）的设计。

### 4.2 PagedAttention

- **Q32.** PagedAttention 的核心思路：类比 OS 虚拟内存页表机制，Block 大小如何选择（典型值 16 tokens/block）？
- **Q33.** PagedAttention 如何支持 Prefix Sharing（多请求共享同一 Prompt 的 KV Block）？
- **Q34.** 相比连续 KV Buffer，PagedAttention 的 Attention Kernel 有哪些额外开销？
- **Q34-b.** RadixAttention（SGLang）相比 PagedAttention 的 Prefix Sharing 有何本质改进？Radix Tree 的最长公共前缀匹配（LCP）机制与 LRU 驱逐策略是什么？对多轮对话、Tree-of-Thought、RAG 等场景的覆盖能力如何？（与 Q66 形成闭环）
- **Q34-c.** KV Block 的引用计数管理与安全释放时机：共享 Block 何时可以回收？显存压力下的驱逐优先级如何排序？错误提前释放会导致什么后果，为何此类 Bug 难以复现？

### 4.3 KV Cache 压缩

- **Q35.** Token Eviction 方法（H2O、SnapKV）的基本思路：基于 Attention Score 保留"Heavy Hitter" Tokens？
- **Q36.** KV Cache 量化（INT8 / FP8 KV）的精度损失分析？
- **Q36-b.** H100 上 FP8 KV Cache 的量化与反量化时机：为何 H100 无需软件反量化 Kernel？与 INT8 KV Cache 方案的工程差异是什么？实测带宽节省与精度损失的数量级？
- **Q37.** StreamingLLM 的 Attention Sink 机制是什么？
- **Q37-b.** KV Cache 分级存储方案（HBM $\to$ CPU DRAM $\to$ NVMe SSD）：各级有效带宽的数量级，以及恢复延迟对 TTFT 的叠加影响；适用场景与精度无损的前提条件？

---

## 5. 调度与批处理策略

### 5.1 Batching 机制

- **Q38.** Static Batching 与 Continuous Batching（Iteration-level Scheduling）的区别？后者如何消除 Padding 浪费？
- **Q39.** Chunked Prefill 的原理：将长 Prompt 的 Prefill 拆分为多个 Chunk，与 Decode 请求交错执行，有何收益与代价？
- **Q39-KV.** Chunked Prefill 执行期间 KV Block 的按需分配策略（与 Q39 关联）：是否需要预分配全量显存？与 Decode 请求共批时如何隔离 Block Pool？Chunk 大小与内部碎片率 ≈(B−1)/(2C)\approx (B-1)/(2C) ≈(B−1)/(2C) 的量化关系？
- **Q40.** Prefill / Decode 分离（Disaggregated PD）架构的动机：两阶段计算特性不同，分离部署如何提升集群利用率？
- **Q40-b.** xPyD Ratio（P 实例数 : D 实例数）的调优依据：如何根据 ISL/OSL（输入/输出序列长度）比例和计算耗时模型推导最优比值？静态配比与动态扩缩容的工程权衡？
- **Q40-c.** KV Cache Transfer 的实现机制与延迟分析：GPUDirect RDMA、NVLink、TCP 三种传输路径的带宽与延迟量级？Transfer 延迟对 TTFT 的叠加影响？NIXL 相比 NCCL 在此场景的优化点？

### 5.2 调度指标

- **Q41.** TTFT（Time to First Token）与 TPOT（Time Per Output Token）的区别及各自的优化路径？
- **Q42.** 吞吐量（Tokens/sec/GPU）与延迟（Latency）之间的根本矛盾：增大 Batch Size 如何影响两个指标？如何估算 Decode 阶段从 Memory-bound 转为 Compute-bound 的临界 Batch Size（脊点 Batch）？
- **Q43.** 如何用 MFU（Model FLOP Utilization）评估系统效率？Decode 阶段为何应改用 MBU（Model Bandwidth Utilization）作为主要指标？
- **Q44-Sched.** 调度器的抢占（Preemption）机制：当 KV Cache 显存耗尽时，vLLM 如何通过 Swap 或 Recompute 策略处理被抢占请求？两种策略的延迟代价与适用场景？
- **Q45-Sched.** Goodput 的定义与 SLO 感知调度：与原始吞吐量（Throughput）的区别？在 TTFT / TPOT 双 SLO 约束下，调度器如何最大化满足 SLO 的请求比例而非最大化原始 Token 吞吐？

---

## 6. 模型量化

### 6.1 量化基础

- **Q44.** PTQ（Post-Training Quantization）与 QAT（Quantization-Aware Training）的区别？
- **Q45.** 对称量化与非对称量化的量化公式推导：

$$x_q = \text{clip}!\left(\left\lfloor \frac{x}{s} \right\rceil + z,; q_{\min},; q_{\max}\right)$$

- **Q46.** Per-tensor、Per-channel、Per-token、Per-group 量化粒度的精度-性能 Trade-off？
- **Q46-b.** 动态量化（Dynamic Quantization）与静态量化（Static Quantization）的区别？激活值为何更常用动态量化？其推理时额外开销如何？

### 6.2 主流量化方法

- **Q47.** GPTQ 的核心思路：基于 OBQ（Optimal Brain Quantization）逐层量化，使用 Hessian 信息补偿误差？
- **Q47-b.** GPTQ 的 Lazy Batch Update 与 Cholesky 分解优化：为何朴素 OBQ 对 LLM 不可行（$O(d^3)$ 复杂度），GPTQ 如何将其降至 $O(d_{\text{in}}^2)$？
- **Q48.** AWQ（Activation-aware Weight Quantization）相比 GPTQ 的改进：保护 Salient Weights 的机制？
- **Q48-b.** AWQ 的 Per-channel 缩放因子 $s_i^{*}$ 如何搜索？为什么不能直接用梯度优化，而用 Grid Search？
- **Q49.** SmoothQuant 的思路：将激活值的量化难度通过 per-channel 缩放迁移到权重侧？
- **Q49-b.** SmoothQuant 的迁移系数 $\alpha$ 的选择对精度的影响？为何默认取 $\alpha = 0.5$？缩放向量 $s$ 如何在推理时做到零开销（融入 LayerNorm 或前置权重矩阵）？
- **Q50.** W4A8 / W4A16 / FP8 / INT8 各方案的适用场景与硬件支持（A100 vs H100 vs Blackwell）？
- **Q50-b.** W4A16 推理的 Dequantization 开销分析：权重量化存储后必须在矩阵乘前解压回 FP16，该操作在 Decode（Memory-bound）与 Prefill（Compute-bound）阶段的性能影响分别是什么？
- **Q51.** Blackwell 的 NVFP4（FP4 with block-level FP8 scale）机制与性能收益？
- **Q51-b.** NVFP4 的两级缩放方案（Block-level FP8 Scale + Tensor-level FP32 Scale）的存储格式推导：每 16 个 FP4 权重共享 1 个 FP8 Scale，有效位宽约 4.5 bits/weight；与 H100 FP8 相比，B200 的 FP4 Tensor Core 峰值算力提升倍数与实测吞吐收益的差距原因？

### 6.3 旋转/变换类量化方法（新增）

- **Q52-Q.** QuaRot / SpinQuant 的核心思路：通过随机 Hadamard 变换旋转权重矩阵以消除 Outlier，从而使权重、激活和 KV Cache 均可量化至 4-bit；与 SmoothQuant 的本质区别（旋转等价变换 vs. 缩放等价变换）？
- **Q53-Q.** AutoRound（EMNLP 2024）与 GPTQ 的差异：引入可学习的 Rounding Offset $v$、Clipping Range $[\alpha, \beta]$，用 Signed Gradient Descent 最小化块级输出重建误差；为何在极低比特（W3/W2）下优于 GPTQ？

### 6.4 KV Cache 量化（新增）

- **Q54-Q.** KV Cache 量化（FP8 / INT8）的量化时机与反量化开销：KV 写入时量化、Attention 计算前反量化的完整数据流；H100 上 FP8 KV Cache 硬件原生支持与 Ampere 上软件模拟的性能差异？
- **Q55-Q.** Per-tensor vs. Per-head vs. Per-token KV Cache 量化粒度的精度对比：Key 和 Value 的分布特性（Key 通道方差大、Value 分布较平）是否要求不同粒度？KIVI（2-bit KV Cache）的思路？
- **Q56-Q.** KV Cache 量化与 FlashAttention 后端的兼容性问题：为何 FlashAttention-2 不支持 FP8 KV Cache，而 FlashAttention-3（FA3）与 FlashInfer 可以原生支持？

---

## 7. 解码加速算法

### 7.1 Speculative Decoding

- **Q52.** Speculative Decoding 的基本流程：Draft Model 生成候选 Token，Target Model 并行 Verify，Token 接受率 $\alpha$ 的定义？
- **Q53.** 接受率 $\alpha$ 与加速比的关系推导：若 $\gamma$ 为 Draft 步数，则期望每轮接受 Token 数与加速比公式推导？
- **Q54.** 为什么 Speculative Decoding 不改变输出分布（Rejection Sampling 的等效性证明）？
- **Q55.** Ngram-based Draft、Medusa、EAGLE（含 EAGLE-2/3）各方案的核心思路与对比？
- **Q55-b.** Tree-based Speculative Decoding（SpecInfer、EAGLE 树形验证）相比链式 Draft 的优势：候选树如何构建？Tree Attention 的 Mask 形式？期望接受 Token 数如何提升？
- **Q55-c.** Self-Speculative Decoding（LayerSkip / Draft & Verify）的核心思路：用目标模型自身的早退出层作为 Draft，无需额外模型的工程代价与精度损失分析？
- **Q55-d.** Speculative Decoding 在高 Batch Size 下性能退化的根本原因：随 Batch 增大系统从 Memory-bound 转为 Compute-bound，Draft 与 Verify 的计算代价比 $c$ 如何变化？何时 Speculative Decoding 反而降低吞吐？

### 7.2 其他解码算法

- **Q56.** Beam Search 与 Greedy Search 的显存和计算差异？LLM 推理中 Beam Search 为何不常用？
- **Q57.** Top-k / Top-p Sampling 的实现细节与 GPU 优化？Temperature 的作用？
- **Q57-b.** Repetition Penalty 与 Min-p Sampling 的实现原理？Min-p 相比 Top-p 的自适应优势？

---

## 8. 并行推理与分布式系统

### 8.1 并行策略

- **Q58.** Tensor Parallelism（TP）：以 Megatron-LM 风格说明 MLP 层如何按列/行切分，需要哪些 AllReduce 通信？
- **Q58-b.** GQA 与 MQA 下 Tensor Parallelism 的特殊处理：当 KV 头数小于 TP 度时，如何避免 KV 头复制的正确性问题？与 MHA 切分方案的差异？
- **Q59.** Pipeline Parallelism（PP）：GPipe vs 1F1B 调度的气泡率对比？训练与推理场景下气泡率公式的差异？
- **Q59-b.** Interleaved 1F1B（虚拟流水段）相比标准 1F1B 的气泡率改进：设 $V$ 为虚拟段数，气泡率如何从 $(P-1)/(M+P-1)$ 降低？代价是什么？
- **Q60.** Sequence Parallelism（SP）的原理及适用场景（中长序列）？与 TP 联合使用时通信模式如何从 AllReduce 变为 ReduceScatter + AllGather？
- **Q61.** Expert Parallelism（EP）：MoE 模型中 Two-shot All-to-All 通信的开销分析？Decode 阶段（小 Batch）与 Prefill 阶段通信量的数量级差异？
- **Q61-b.** EP 与 TP 联合部署（N-D 并行）时的通信层次：All-to-All 与 AllReduce 如何在节点内/跨节点调度？DeepSeek-V3 的 EP=320 实际配置说明了什么？
- **Q_N.** ZeRO（Zero Redundancy Optimizer）的三个阶段（ZeRO-1/2/3）在推理中是否适用？ZeRO-Inference 与训练 ZeRO 的核心差异？

### 8.2 通信优化

- **Q62.** AllReduce 的 Ring-AllReduce 实现与带宽分析：$N$ 个节点、每个节点数据量 $M$，总通信量为 $2M(N-1)/N \approx 2M$，与 $N$ 无关？在小消息量场景下为何变为 Latency-bound？
- **Q63.** GEMM-ReduceScatter、AllGather-GEMM 的 Kernel Fusion 如何减少通信-计算串行等待？与 Sequence Parallelism 联合时的 Overlap 方案？
- **Q64.** NVLink 与 PCIe 的带宽差距对 TP 规模上限的影响？GB200 NVL72 的全互联方案如何改变 TP/EP 的规模边界？
- **Q_O.** 通信拓扑感知调度（Topology-aware AllReduce）：为何 Ring 拓扑在多机环境下次优？Tree AllReduce（如 Recursive Halving-Doubling）与 Ring 的延迟-带宽权衡？NCCL 如何自动选择拓扑？
- **Q_P.** 推理服务中 Prefill 实例与 Decode 实例之间的 KV Cache Transfer 通信（P/D 分离场景）：GPUDirect RDMA 路径的带宽与 AllReduce 路径的共享竞争如何影响整体吞吐？

---

## 9. 推理框架与工具链

### 9.1 主流框架

- **Q65.** vLLM 的核心创新点（PagedAttention + Continuous Batching）？与 TensorRT-LLM 的定位差异？
- **Q66.** SGLang 相比 vLLM 的改进：RadixAttention（前缀 KV 复用树）的原理？
- **Q67.** TensorRT-LLM 的 Plugin 机制与 In-flight Batching 如何工作？
- **Q67-b.** vLLM / SGLang / TensorRT-LLM 三者在生产部署时的选型框架：如何根据模型规模、QPS 目标、运维能力和硬件约束做出选择？

### 9.2 Profiling 与性能分析

- **Q68.** 使用 `nsys` 和 `ncu` 的区别：Timeline 分析 vs Kernel-level 指标采集？
- **Q69.** 如何判断一个 Kernel 是 Memory-bound：查看 `ncu` 的哪些指标（Memory Throughput、L2 Hit Rate、DRAM BW Utilization）？
- **Q70.** Occupancy（占用率）低对性能一定有影响吗？什么情况下低 Occupancy 也能高性能？
- **Q70-b.** 给定一个实际的 `ncu` 报告（SM Active 30%、HBM BW 91%、L2 Hit Rate 18%），写出完整的诊断流程与优化路径？
- **Q70-c.** CUDA Graph 捕获（Capture）的条件与限制：哪些操作无法被 Graph 捕获？LLM 推理中动态 Batch Size 与 Graph Replay 如何共存（cudaGraphExecUpdate / Multi-Graph 方案）？

### 9.3 Triton

- **Q71.** Triton 与 CUDA 的核心编程模型差异（Block-level vs Thread-level）？
- **Q72.** 何时选择 Triton 而非 CUDA 手写（快速原型验证、跨硬件移植）？
- **Q72-b.** Triton 编译器的 `num_stages` 与 `num_warps` 超参数对性能的影响机制？`@triton.autotune` 的搜索代价与缓存机制？
- **Q72-c.** Triton 在 Hopper 架构上的现状：对 TMA / WGMMA 的支持程度（截至 2025 年上半年），以及 FlashAttention-3 为何仍选择 CUDA 而非 Triton 实现？

---

## 10. 系统设计题

### 10.1 典型题目

- **Q73.** 设计一个支持 100 QPS、P99 TTFT < 500ms、Batch 动态变化的 LLM 推理服务，说明关键组件与调优策略。
- **Q74.** 给定 8 × H100 节点，部署一个 70B 参数模型，选择 TP/PP 策略并分析通信瓶颈。
- **Q75.** KV Cache 显存告警，但计算 GPU 利用率只有 40%，根因分析与优化路径？
- **Q76.** 如何在不更换硬件的前提下，将现有服务的吞吐提升 2×？给出逐步排查与优化的思路。
- **Q77-SD.** 设计一个面向超长 CoT（平均输出 4096 tokens）的推理服务：与常规对话服务（平均 OSL 256 tokens）相比，KV Cache 规划、调度策略、SLO 设计各有哪些差异？
- **Q78-SD.** 多模型共享 GPU 集群（Dense 70B + MoE 8×7B 同时在线）：如何设计资源隔离与 KV Cache 显存分区方案，避免相互干扰？
- **Q79-SD.** 设计一个 P/D 分离推理系统的 xPyD 配比调优方案：输入 ISL 均值 2048 tokens、输出 OSL 均值 512 tokens，如何根据计算耗时模型推导 Prefill 实例与 Decode 实例的最优比值，并说明动态扩缩容的触发阈值设计？

### 10.2 答题框架

|步骤|内容|
|---|---|
|需求澄清|延迟 SLA、吞吐目标、硬件约束、模型规格|
|瓶颈定位|Compute-bound / Memory-bound / IO-bound|
|方案设计|算法层 → 系统层 → 硬件层|
|指标量化|MFU、TTFT、TPOT、Tokens/s/GPU|
|权衡说明|精度损失、工程复杂度、可维护性|

---

## 11. C++ 与系统编程

- **Q77.** `std::atomic` 的 Memory Order 模型（`memory_order_relaxed` vs `acquire/release` vs `seq_cst`）？各级别在 x86 与 ARM 平台上的实际开销差异？
- **Q78.** Lock-free Queue 的实现（Michael-Scott Queue）与 ABA 问题？Tagged Pointer 方案的 16-byte CAS 硬件要求？Hazard Pointer 的正确实现范式？
- **Q79.** NUMA 架构下内存分配对延迟的影响？如何通过 `numactl`、`numa_alloc_onnode`、`mbind` 将内存与线程 Pin 到特定 NUMA Node？`nvidia-smi topo -m` 如何确认 GPU 的 NUMA 亲和性？
- **Q80.** Zero-copy DMA 传输的实现原理（`cudaHostAlloc` Pinned Memory）？Pageable Memory 的额外拷贝路径与 Pinned Memory 的直接 DMA 路径对比？`cudaHostAllocMapped` 的适用场景与代价？
- **Q81.** 多线程推理服务中 Thread Pool 的设计与线程亲和性（CPU Affinity）绑定？各功能线程池（IO / Scheduler / CUDA Launch / Sampler）的职责划分与 NUMA 对齐原则？
- **Q82.** `mmap` vs `read` 的权衡：大模型权重加载的最优策略？`O_DIRECT` 与 `mmap` 的语义冲突问题？SafeTensors 格式如何利用 `mmap` 实现 MoE Expert 权重懒加载？
- **Q83-CPP.** `std::pmr`（Polymorphic Memory Resource）在推理引擎中的应用：如何用 `std::pmr::monotonic_buffer_resource` 实现请求级零碎片内存分配，请求结束后 $O(1)$ 批量回收？
- **Q84-CPP.** CUDA Stream 与 Host 线程的同步机制：`cudaStreamSynchronize` vs `cudaEventRecord` + `cudaEventSynchronize` vs `cudaStreamAddCallback` 三种方式的延迟与 CPU 忙等开销对比？推理引擎中如何用 `cudaEvent` 实现 Prefill → Decode KV 传递的无锁同步？
- **Q85-CPP.** `cudaIpcMemHandle` 跨进程显存共享：P/D 分离架构中，同节点 Prefill 进程与 Decode 进程如何通过 IPC Handle 零拷贝共享 KV Cache Block？与 RDMA 路径的适用场景边界？
- **Q86-CPP.** C++ 内存序与 GPU Kernel 启动的混合并发模型：推理引擎的调度线程（CPU）向 CUDA Launch 线程提交任务时，如何正确使用 `acquire/release` 配对保证 KV Block 指针的可见性，避免 GPU 读取未初始化 Block？

---

## 12. MoE 架构推理

### 12.1 MoE 基础

- **Q83.** Dense 模型与 Sparse MoE 的计算量对比：给定总参数 $N$、激活专家比例 $k/E$，单 Token 的实际 FLOPs 约为等规模 Dense 模型的多少？
- **Q84.** Top-K Routing 的 Gating 函数实现：Softmax-based vs. Sigmoid-based，Expert Load Balancing Loss 的形式？
- **Q85.** Expert Capacity（专家容量）与 Token Drop 的关系：Capacity Factor 如何取值？
- **Q83-b.** MoE 模型的显存占用构成分析：总参数 $N_{\text{total}}$ 中 Expert 权重占比极高，但推理时激活参数仅为 $N_{\text{active}}$；以 DeepSeek-V3（671B 总参数、37B 激活参数）为例，说明其显存需求与计算需求的"解耦"特性，以及对硬件选型（高带宽 vs. 高显存）的影响。
- **Q83-c.** MoE 中 Shared Expert 机制的设计动机：为何 DeepSeek-V2/V3 引入永久激活的 Shared Expert？与 Top-K Routed Expert 的分工如何？对 Load Balancing Loss 的影响？
- **Q84-b.** Expert 路由崩溃（Routing Collapse）的成因与检测：如何通过 Expert 利用率直方图诊断崩溃？推理阶段的路由分布与训练阶段的分布偏移（Distribution Shift）如何影响性能？

### 12.2 Expert Parallelism（EP）

- **Q86.** EP 的核心通信模式是 Two-shot All-to-All：第一次按路由结果将 Tokens 分发到对应 Expert 所在 GPU，第二次将计算结果汇回，分析完整通信量公式与延迟构成。
- **Q87.** Wide EP（大规模 Expert Parallelism）的适用场景：何时 EP 度应超过 TP 度？
- **Q88.** EP 与 TP 组合时的通信分析：All-to-All 与 AllReduce 如何在 N-D 并行中调度？
- **Q86-b.** EP All-to-All 与 Expert 计算的 Overlap（重叠）实现：将 Token 划分为多个 Micro-batch，使用 CUDA Stream 交错执行 All-to-All 和 FFN 计算；DeepSeek-V3 的 DualPipe 方案如何将通信延迟近乎完全隐藏？overlap 成立的条件（计算时间 ≥ 通信时间）如何量化验证？
- **Q86-c.** EP 场景下 KV Cache 的布局问题：EP 将 Expert 权重分布到不同 GPU，但 Attention 层（含 KV Cache）通常使用 TP 切分；当 EP 与 TP 共存时，KV Cache 按 TP 域存储还是按 EP 域存储？P/D 分离架构下 Prefill 实例的 EP 配置如何影响 KV Transfer 的目标节点选择？

### 12.3 MoE 量化与 Kernel 优化

- **Q89.** MoE 层的 GEMM 为什么是"非均匀矩阵乘"（每个 Expert 的 Token 数不同）？如何用 GroupGEMM / Batched GEMM 处理？
- **Q90.** Structured Sparsity（结构化稀疏，如 2:4 稀疏 Tensor Core）与 MoE 稀疏性的区别？
- **Q89-b.** FP8 量化对 MoE Expert 权重的适用性分析：Expert FFN 权重分布是否均匀（与 Dense 模型相比）？为何部分 Expert 的权重 Outlier 比例更高？Per-Expert 量化粒度与 Per-tensor 量化粒度的精度权衡？
- **Q89-c.** MoE 推理的 Expert 权重预加载策略：Expert 权重在单卡显存中的驻留策略（全量常驻 vs. 按需换入）；当 EP 度不足以将所有 Expert 分布到不同 GPU 时（每 GPU 持有多个 Expert），如何调度 DRAM 与 HBM 之间的 Expert 权重换入以降低延迟？
- **Q90-b.** MoE 模型的 Decode 阶段瓶颈分析：小 Batch Decode 时，Expert 计算退化为 GEMV（Memory-bound），All-to-All 通信与 GEMV 的时间占比；与 Dense 模型 Decode 的带宽瓶颈相比，MoE 的额外开销来源是什么？如何估算"MoE Tax"（MoE 相比同激活参数 Dense 模型的吞吐损失）？

---

## 13. P/D 分离架构（Disaggregated Prefill-Decode）

### 13.1 核心动机与架构

- **Q91.** Prefill 与 Decode 两阶段的计算特性（Compute-bound vs. Memory-bound、主要算子、KV Cache 读写行为）有何根本差异？传统混合部署因此引发哪三类干扰问题（Prefill 阻塞 Decode、显存竞争、最优 Batch Size 矛盾）？
- **Q92.** P/D 分离已成为 2025 年主流推理栈的默认方案：vLLM、SGLang、TensorRT-LLM、NVIDIA Dynamo、MoonCake、llm-d 各自的实现方式与 KV Transfer 机制？NIXL（NVIDIA Inference Xfer Library）的定位与核心能力？
- **Q93.** KV Cache Transfer 的三种实现路径（NVLink 节点内、GPUDirect RDMA 跨节点、TCP 降级）的带宽与延迟量级？Transfer 延迟如何叠加到 TTFT？为何生产环境必须配备高速互联？
- **Q93-b.** NVLink 节点内传输的正确带宽参数：H100 单 GPU NVLink 总线双向带宽 900 GB/s 与 GPU-to-GPU 点对点可用带宽（~300 GB/s 双向）的区别？误用总线带宽会导致 Transfer 时间估算偏差多大？
- **Q93-c.** NIXL 与 NCCL 的定位区别：两者分别针对哪类通信模式（点对点 KV Transfer vs. 集合通信 AllReduce）？为何不能直接用延迟百分比对比？NIXL 对非连续 PagedAttention Block 的 Scatter-Gather DMA 支持是其核心优化点？

### 13.2 调度设计

- **Q94.** xPyD Ratio（P 实例数 : D 实例数）的调优依据：如何根据单实例 Prefill 吞吐 $R_P$、Decode 吞吐 $R_D$ 及 ISL/OSL 比例推导平衡条件 $x/y = (R_D \cdot \text{ISL}) / (R_P \cdot \text{OSL})$？不同 ISL/OSL 场景（长输入短输出 vs. 短输入长输出）下的典型配比举例？
- **Q94-b.** xPyD 静态配比与动态扩缩容的工程权衡：全局调度器如何依据 P/D 队列积压实时调整实例数？D 实例是否可临时承担 Prefill（"Prefill-fallback"）？扩缩容的触发阈值如何量化设计（队列深度 vs. 时延 SLO 违约率）？
- **Q95.** P/D 分离收益最显著的三类场景（超大模型 120B+、长输入序列 ISL > 10k、稀疏 MoE 架构）的量化分析：以长输入场景（ISL=16k，OSL=512，P99 TPOT 目标 < 100ms）对比混合部署与 P/D 分离的 TPOT 差异？
- **Q95-b.** P/D 分离在 MoE 架构下的额外收益：P 节点与 D 节点可采用不同 EP（Expert Parallelism）规模，DeepSeek-V3 为何在 P 节点使用更大 EP？两阶段对 All-to-All 通信特性的差异（大 Batch 可 Overlap vs. 小 Batch 延迟敏感）？
- **Q96.** KV Cache Transfer 与 EP All-to-All 的带宽竞争根源及四类缓解方案（网络隔离、QoS 优先级降级、Transfer 时序错开、KV 压缩减少传输量）的实现机制与适用场景？
- **Q96-b.** KV 感知路由（KV-aware Routing）：全局调度器如何利用 Prefix Cache 命中信息将请求路由到已持有相关 KV Block 的 D 实例，从而避免重复 Transfer？NVIDIA Dynamo 的 Smart Router 与 SGLang 的 RadixAttention 在此机制上的实现差异？
- **Q97-PD.** P/D 分离架构中的容错与一致性设计：Prefill 实例完成计算但 KV Transfer 失败时如何处理（重试 vs. 本地降级为混合模式）？D 实例崩溃时持有的 KV Cache 如何恢复（Recompute vs. Checkpoint）？两种恢复路径的延迟代价与适用场景？
- **Q98-PD.** P/D 分离的显存规划差异：P 实例 KV Cache 仅需在 Transfer 完成前驻留（短暂峰值），D 实例 KV Cache 需要长期驻留（随 Decode 步数线性增长）；两类实例应如何分别配置 KV Cache 内存池大小？P 实例回收 KV Block 的时机与 Transfer 完成事件的同步机制？

---

## 14. 长上下文推理

### 14.1 位置编码扩展

- **Q97.** RoPE 的数学原理：对 Query/Key 施加旋转矩阵，使注意力得分仅依赖相对位置 $m-n$，推导形式：

$$\mathbf{q}_m^T \mathbf{k}_n = \text{Re}!\left[\left(\mathbf{W}_q \mathbf{x}_m \odot e^{im\theta}\right)^* \cdot \left(\mathbf{W}_k \mathbf{x}_n \odot e^{in\theta}\right)\right]$$

- **Q98.** RoPE 外推问题：训练长度之外的位置 $\theta$ 分量溢出，YaRN / LongRoPE / Llama3 RoPE Scaling 各自的补偿策略？
- **Q99.** ALiBi 与 RoPE 的外推能力对比？

### 14.2 超长上下文系统

- **Q100.** Ring Attention（序列并行）的原理：将序列维度切分到多 GPU，通过 P2P Ring 通信交换 KV，避免将全序列集中到单 GPU？
- **Q101.** Context Parallelism（CP）与 Sequence Parallelism（SP）的区别？
- **Q102.** 超长上下文（128k+）时 KV Cache 的显存压力与 Chunked Prefill 的配合？
- **Q102-KV.** 128k+ 上下文时单请求 KV Cache 显存压力的量化分析（与 Q102 关联）：以 LLaMA-3 70B GQA FP16 为例推导 $S = 128\text{k}$ 时 KV Cache 大小，逐步分析 FP8 量化、Token Eviction、Context Parallelism 三种应对路径的精度与延迟代价。
- **Q103.** Sliding Window Attention 在长上下文中的 Attention Sink 失效问题？

---

## 15. 推理时计算扩展（Test-Time Compute Scaling）

### 15.1 核心概念

- **Q104.** 什么是 Test-Time Compute Scaling？与 Training-Time Scaling 的本质区别？
- **Q105.** Chain-of-Thought（CoT）/ Extended Thinking 对推理系统的负载特征有何改变（输出 Token 数激增，Decode 阶段成为更严重瓶颈）？
- **Q106.** o1 / DeepSeek-R1 类推理模型的输出长度分布对 KV Cache 规划的影响？

### 15.2 系统层响应

- **Q107.** 针对长 CoT 的 Speculative Decoding：Draft 模型接受率在长推理链上是否稳定？
- **Q108.** 推理模型的 SLO 设计：TTFT vs. Total Latency 的权衡如何变化？

---

## 16. 模型结构轻量化

### 16.1 知识蒸馏

- **Q109.** 逻辑蒸馏（Logit Distillation）vs. 特征蒸馏（Feature Distillation）的优劣？
- **Q110.** 推理场景下蒸馏（如 DeepSeek-R1 → Qwen 系列）的常见方法？

### 16.2 结构剪枝

- **Q111.** Unstructured Pruning vs. Structured Pruning（Head Pruning、Layer Dropping）对推理加速的实际贡献差异？
- **Q112.** 2:4 稀疏格式（NVIDIA Sparse Tensor Core）的激活方式与精度损失分析？

### 16.3 模型架构设计题

- **Q113.** 给定延迟 SLA = 50ms / Token，如何在 7B 模型的基础上通过蒸馏 + 量化组合达到目标，说明决策链？

---

## 17. 多模态推理（VLM/MLM）

- **Q114.** Vision Encoder（如 ViT）的输出 Token 数量对 Prefill 显存和计算的影响（典型值：每张 224×224 图片 = 196 ~ 256 Image Tokens）？
- **Q115.** Image Token 的 KV Cache 是否应与 Text Token 区别对待（不同 Eviction 策略）？
- **Q116.** 多模态模型中 Prefill 计算量远大于纯文本场景，如何调整 Chunked Prefill 的 Chunk Size？

---

## 18. 网络通信与互联

### 18.1 集合通信

- **Q117.** AllReduce、AllGather、ReduceScatter、All-to-All 的语义与典型使用场景各是什么？
- **Q118.** Ring-AllReduce 的通信量分析：$N$ 个节点、每个节点数据量 $M$，总通信量为 $2M(N-1)/N \approx 2M$，与 $N$ 无关？

### 18.2 通信-计算 Overlap

- **Q119.** Tensor Parallelism 中 GEMM 与 AllReduce 的 Overlap 方案：GEMM-ReduceScatter + AllGather-GEMM 流水线如何实现？
- **Q120.** NCCL 的底层实现：为何 NVLink 通信可直接触发而 PCIe 通信需要 CPU 中介？
- **Q121.** NIXL（NVIDIA Inference Xfer Library）相比 NCCL 在 KV Transfer 场景的优化点？

---

## 19. 新硬件特性

### 19.1 Hopper 架构（H100）

- **Q122.** TMA（Tensor Memory Accelerator）的作用：异步批量数据搬运，解放 CUDA Core 的地址计算负担？
- **Q123.** Warp Specialization 的思路：将 Warp 分为 Producer（负责数据搬运）和 Consumer（负责计算），形成软件流水？
- **Q124.** H100 的 FP8 Tensor Core：E4M3 vs. E5M2 的精度-范围权衡？

### 19.2 Blackwell 架构（B100 / GB200）

- **Q125.** NVFP4（FP4 with block-scale FP8）的存储格式：每 16 个 FP4 值共享一个 FP8 Scale Factor，有效位宽约 4.5 bits/weight？
- **Q126.** GB200 NVL72：72 个 Blackwell GPU 通过 NVLink Switch 全互联，单节点聚合 HBM 约 13.5 TB，适合哪类推理场景？
- **Q127.** Blackwell 的 FP4 Tensor Core 峰值算力相比 H100 FP8 的提升倍数？

---

## 20. 高频考点优先级速查

| 优先级 | 考点                                   | 涵盖岗位          |
| --- | ------------------------------------ | ------------- |
| ⭐⭐⭐ | KV Cache + PagedAttention            | 所有推理岗         |
| ⭐⭐⭐ | FlashAttention 原理                    | 所有推理岗         |
| ⭐⭐⭐ | Continuous Batching                  | 所有推理岗         |
| ⭐⭐⭐ | P/D 分离架构（2025 默认方案）                  | 所有推理岗         |
| ⭐⭐⭐ | Roofline / Compute vs Memory Bound   | 性能优化岗         |
| ⭐⭐⭐ | GEMM Tiling + Shared Memory          | Kernel 开发岗    |
| ⭐⭐⭐ | MoE + Expert Parallelism             | 分布式推理岗        |
| ⭐⭐⭐ | Test-Time Compute / 推理模型负载特征         | 系统 + 调度岗      |
| ⭐⭐⭐ | RoPE 原理 + 长上下文扩展                     | 算法 + 系统岗      |
| ⭐⭐  | Speculative Decoding                 | 算法 + 系统岗      |
| ⭐⭐  | 量化（GPTQ / AWQ / FP8）                 | 部署优化岗         |
| ⭐⭐  | TP / PP 并行策略                         | 分布式推理岗        |
| ⭐⭐  | Warp Reduce / CUDA 同步原语              | Kernel 开发岗    |
| ⭐⭐  | Ring Attention / Context Parallelism | 长上下文岗         |
| ⭐⭐  | NVFP4 / Blackwell 特性                 | 量化 + 硬件岗      |
| ⭐⭐  | KV Transfer（RDMA / GPUDirect）        | 分布式推理岗        |
| ⭐⭐  | Warp Specialization / TMA            | Kernel 开发岗    |
| ⭐⭐  | 推理框架选型（vLLM / SGLang / TRT-LLM）      | 所有推理岗         |
| ⭐⭐  | nsys / ncu 诊断流程（实战级）                 | 性能分析岗         |
| ⭐   | Triton 编程模型                          | Kernel 开发岗    |
| ⭐   | VLM 多模态推理特性                          | 多模态系统岗        |
| ⭐   | 2:4 结构化稀疏                            | 量化 + Kernel 岗 |
