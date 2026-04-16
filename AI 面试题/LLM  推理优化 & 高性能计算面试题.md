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

$$\mathbf{q}_m^T \mathbf{k}_n = \text{Re}\!\left[\left(\mathbf{W}_q \mathbf{x}_m \odot e^{im\theta}\right)^* \cdot \left(\mathbf{W}_k \mathbf{x}_n \odot e^{in\theta}\right)\right]$$

- **Q98.** RoPE 外推问题：训练长度之外的位置 $\theta$ 分量溢出，YaRN / LongRoPE / Llama3 RoPE Scaling 各自的补偿策略？
- **Q99.** ALiBi 与 RoPE 的外推能力对比？
- **Q99-b.** RoPE 与 ALiBi 对 Prefix Caching（前缀 KV 复用）的兼容性差异：RoPE 的旋转变换将绝对位置嵌入 KV 向量，为何只要 Token 绝对位置不变即可跨请求复用？动态插入内容（如 RAG 文档）会破坏哪些 Token 的 KV 缓存？ALiBi 为何天然兼容 Prefix Caching？（与 Q31、Q34-b 形成闭环）

### 14.2 超长上下文系统

- **Q100.** Ring Attention（序列并行）的原理：将序列维度切分到多 GPU，通过 P2P Ring 通信交换 KV，避免将全序列集中到单 GPU？
- **Q101.** Context Parallelism（CP）与 Sequence Parallelism（SP）的区别？
- **Q101-b.** CP 的精确通信量推导：设 CP 度为 $P$，序列长度 $N$，GQA KV 头数 $H_{\text{KV}}$，头维度 $d$，推导每卡总通信量 $(P-1)/P \times 2N H_{\text{KV}} d \cdot b$；代入 LLaMA-3 70B 参数，在 NVLink 节点内（点对点带宽 ~300 GB/s）估算单步传输时间，说明计算-通信 Overlap 成立的条件；跨节点（InfiniBand ~25 GB/s）场景下 Overlap 效率的变化？
- **Q102.** 超长上下文（128k+）时 KV Cache 的显存压力与 Chunked Prefill 的配合？
- **Q102-KV.** 128k+ 上下文时单请求 KV Cache 显存压力的量化分析（与 Q102 关联）：以 LLaMA-3 70B GQA FP16 为例推导 $S = 128\text{k}$ 时 KV Cache 大小，逐步分析 FP8 量化、Token Eviction、Context Parallelism 三种应对路径的精度与延迟代价。
- **Q102-b.** 超长上下文下 Chunked Prefill 的 Chunk Size 选择原则：Block 大小 $B$、Chunk Size $C$ 与内部碎片率 $\approx (B-1)/(2C)$ 的量化关系；Chunk Size 对 GEMM 效率（Wave Quantization 效应，$M \geq 128$ 要求）与 Decode TPOT P99 的双向约束；在 TPOT SLO 约束下给出 $C^*$ 的选择框架，并对比 TPOT SLO = 100ms / 20ms / 5ms 三种场景的推荐值；CP 联合部署时每卡实际处理的 Token 数 $C/P$ 对 Chunk Size 下界的影响？
- **Q103.** Sliding Window Attention 在长上下文中的 Attention Sink 失效问题？
- **Q104-LC.** 长上下文下 KV Cache 分级存储（HBM $\to$ CPU DRAM $\to$ NVMe SSD）的触发阈值与精度无损条件：各级有效带宽与恢复延迟的数量级；以 LLaMA-3 70B 128k 为例估算从 DRAM 恢复全量 KV Cache 对 TTFT 的叠加影响；精度无损的前提条件（Block 完整性、传输前完成保证）；与 FP8 量化和 Prefetch 策略的组合方案？（与 Q37-b 形成闭环）

---

## 15. 推理时计算扩展（Test-Time Compute Scaling）

### 15.1 核心概念

- **Q104.** 什么是 Test-Time Compute Scaling？与 Training-Time Scaling 的本质区别？
- **Q105.** Chain-of-Thought（CoT）/ Extended Thinking 对推理系统的负载特征有何改变（输出 Token 数激增，Decode 阶段成为更严重瓶颈）？
- **Q106.** o1 / DeepSeek-R1 类推理模型的输出长度分布对 KV Cache 规划的影响？

### 15.2 系统层响应

- **Q107.** 针对长 CoT 的 Speculative Decoding：Draft 模型接受率在长推理链上是否稳定？
- **Q108.** 推理模型的 SLO 设计：TTFT vs. Total Latency 的权衡如何变化？

### 15.3 采样策略与资源分析（新增）

- **Q109-TTC.** Best-of-N 与 Self-Consistency 的系统资源对比：并行采样（$N$ 个独立请求）与串行采样（单请求多次生成）对 KV Cache 占用、Batch 利用率和吞吐量的影响差异？Reward Model 评分的计算开销如何叠加到端到端延迟？
- **Q110-TTC.** Test-Time Compute 的收益边际递减规律：对于不同难度的任务（数学证明 vs. 开放式问答），扩展推理 Token 数的质量提升曲线有何差异？如何估算特定任务的"最优推理预算"？

### 15.4 工程实现细节（新增）

- **Q111-TTC.** Thinking Token 的流式传输与用户体验设计：推理模型在生成"思考过程"（thinking tokens）期间是否向客户端流式输出？不同框架（OpenAI o1、DeepSeek-R1、Anthropic Extended Thinking）的行为差异？思考内容的可见性设计对 TTFT 感知的影响？
- **Q112-TTC.** 推理模型的 KV Cache 动态增长与抢占调度：Extended Thinking 场景下 KV Cache 按步线性增长（每步 +1 token），当单请求 KV 超过预分配 Block Pool 时，vLLM / SGLang 的动态扩容机制与抢占触发条件？Swap-Out 的 PCIe 带宽瓶颈如何量化为延迟惩罚？
- **Q113-TTC.** Test-Time Compute Scaling 与 P/D 分离架构的联动：长 CoT（OSL > 8k）场景下，D 节点的 KV Cache 规划策略与常规服务有何不同？P 节点完成 Prefill 后向 D 节点传输 KV，D 节点需为后续数千步 Decode 预留多大 Block Pool？动态扩容与静态预留的工程权衡？

---

## 16. 模型结构轻量化

### 16.1 知识蒸馏

- **Q109.** 逻辑蒸馏（Logit Distillation）vs. 特征蒸馏（Feature Distillation）的优劣？
- **Q110.** 推理场景下蒸馏（如 DeepSeek-R1 → Qwen 系列）的常见方法？
- **Q110-b.** 序列蒸馏（Sequence-level Distillation）与在线蒸馏（On-policy Distillation）的 Exposure Bias 问题：离线 SFT 数据训练的 Student 在推理时为何产生分布漂移？RLVR 如何作为第二阶段修正？
- **Q110-c.** 知识蒸馏的温度参数 $\tau$ 的作用：为何 $\tau > 1$ 能放大"暗知识"（Dark Knowledge）？$\tau$ 过高或过低对 Student 学习的影响？

### 16.2 结构剪枝

- **Q111.** Unstructured Pruning vs. Structured Pruning（Head Pruning、Layer Dropping）对推理加速的实际贡献差异？
- **Q111-b.** Head Importance 评估方法：如何用 Gradient × Activation 或 Taylor Expansion 近似估计每个 Attention Head 的重要性？剪枝后是否需要恢复性微调（Recovery Fine-tuning）？
- **Q112.** 2:4 稀疏格式（NVIDIA Sparse Tensor Core）的激活方式与精度损失分析？
- **Q112-b.** 2:4 稀疏与量化的组合方案（Sparse + INT8 / FP8）：两者能否叠加？在 A100 / H100 上的硬件支持情况？叠加后精度损失的累积规律？
- **Q112-c.** SparseGPT（NeurIPS 2022）的核心思路：如何将非结构化剪枝的逐权重误差补偿推广至 2:4 结构化稀疏，在单次前向中完成校准？与 GPTQ 误差补偿方案的异同？

### 16.3 模型架构设计题

- **Q113.** 给定延迟 SLA = 50ms / Token，如何在 7B 模型的基础上通过蒸馏 + 量化组合达到目标，说明决策链？
- **Q113-b.** 多目标约束下的轻量化决策：若同时存在 TTFT < 200ms（P99）、TPOT < 30ms（P99）、精度损失 < 1%（MMLU）三重约束，优化顺序与方案选择的思路？如何量化评估 Pareto 前沿上的方案？

---

## 17. 多模态推理（VLM/MLM）

### 17.1 Vision Encoder 与 Token 化

- **Q114.** Vision Encoder（如 ViT）的输出 Token 数量对 Prefill 显存和计算的影响：标准 ViT-L/14 对 224×224 图片输出 256 个 Image Token（含 CLS Token）；现代高分辨率 VLM（LLaVA-NeXT、InternVL2、Qwen2-VL）如何通过动态分辨率（Dynamic Resolution）和图像切片（Image Tiling）将单张图片的 Image Token 数提升至 2880+；Image Token 数量对 Prefill 阶段 TTFT 和显存的量化影响推导？
- **Q114-b.** 动态分辨率（Dynamic Resolution / Naive Dynamic Resolution）的工程实现：Qwen2-VL 的 NaViT 风格变长 Token 如何在 Batch 中 Padding 或 Packing？不同分辨率输入在同一 Batch 内的效率差异？ViT 前向的计算量如何随分辨率平方增长，并对 TTFT 产生前期瓶颈？
- **Q114-c.** 视觉编码器（ViT）本身在 VLM 推理中的 TTFT 占比：在高分辨率（如 4096² 的文档图像）场景下 ViT 前向耗时可占总 TTFT 的 80% 以上（Apple FastVLM 实测数据）；ViT 计算量如何与 LLM Prefill 计算量解耦？轻量化 ViT（ConvNeXT、FastViT、SigLIP）替换 ViT-L/G 的精度-速度权衡？

### 17.2 Image Token KV Cache 管理

- **Q115.** Image Token 的 KV Cache 是否应与 Text Token 区别对待：Image Token 在 LLM 层的注意力行为（浅层 Attention 权重集中、深层逐渐被 Text Token 主导）是否支持差异化 Eviction？FastV（ECCV 2024 Oral）的核心机制：在第 2 层之后裁剪 50% Image Token，对大多数任务精度影响 < 1% 但 FLOPs 减少 45%；SparseVLM（ICML 2025）的文本感知（Text-aware）Token 稀疏化策略与 FastV 的差异？
- **Q115-b.** Image Token 的 Prefix Caching 可行性分析：相同图片被多请求复用时，Image Token 的 KV Cache 是否可跨请求命中（VLCache / SGLang 的实现思路）？RoPE 位置编码对 Image Token 绝对位置的绑定是否破坏 Prefix Cache 跨请求复用（与 Text Token 的一致性）？同一图片被插入不同位置时 KV Cache 是否需要重算？
- **Q115-c.** 多帧视频 VLM 的 KV Cache 管理：视频理解中逐帧生成的 Image Token 数量（如 64 帧 × 256 Token/帧 = 16384 Token）对 KV Cache 显存的冲击；时序冗余（相邻帧 Token 高度相似）如何支撑跨帧 Token 合并（Token Merging）与帧级 Eviction 策略？

### 17.3 VLM 推理的 Prefill 优化

- **Q116.** 多模态模型中 Prefill 计算量远大于纯文本场景，如何调整 Chunked Prefill 的 Chunk Size：Image Token 不可拆分（跨 Chunk 会破坏 2D 位置编码的空间连续性），导致最小不可分割单元为整张图片的 Token 数；高分辨率图片（2880 Token）远超常规 Chunk Size（512），如何制定"图片感知（Image-aware）Chunking"策略？TTFT 与 Decode TPOT 的双向约束下，包含多张图片的请求如何调度？
- **Q116-b.** vLLM 对 VLM 的 Chunked Prefill 支持现状（2024-2025）：标准 Chunked Prefill 不能跨 Image Token 边界切分的工程约束；多模态 Prefix Caching（Image KV Block 复用）在 vLLM / SGLang 中的实现状态；Vision Encoder 前向（CPU 或独立 CUDA Stream）与 LLM Prefill 的异步流水线设计？
- **Q116-c.** Visual Token Compression（视觉 Token 压缩）作为 Prefill 加速的替代路径：Perceiver Resampler / Q-Former 在编码器侧将 ViT 输出压缩至固定 Token 数（如 32~256），LLaVA-style 直接投影 vs. Q-Former 压缩的精度-速度权衡？Token 压缩后 KV Cache 的压缩比如何影响 Decode 阶段 TPOT？

### 17.4 多模态 KV Cache 量化与位置编码

- **Q117-VLM.** 多模态位置编码（M-RoPE）的推理影响：Qwen2-VL 将 RoPE 分解为文本维（1D）、图像高度维、图像宽度维的三维位置编码；与纯文本 RoPE 相比，M-RoPE 对 Prefix Caching 的影响（图片位置信息绑定于绝对坐标）？ViT Patch 坐标与 LLM RoPE 的对齐方式？
- **Q118-VLM.** VLM 中混合模态批处理的 Attention Mask 结构：Image Token 与 Text Token 之间的双向 / 单向 Attention 策略（图片 Token 相互全注意力、文本 Token 单向 Causal Attention）；Flash Attention 如何高效支持此非标准 Mask？对 PagedAttention Block 分配的影响？

---

## 第 18 章·参考答案：网络通信与互联

---

### 1. 集合通信

---

**Q117. AllReduce、AllGather、ReduceScatter、All-to-All 的语义与典型使用场景各是什么？**

**四种集合通信原语的语义：**

设 $N$ 个节点，每个节点持有数据块 $x_i \in \mathbb{R}^M$。

**1.1 AllReduce**

每个节点将本地数据与其他所有节点的数据进行聚合（如 Sum），每个节点最终持有**相同的全局聚合结果**：

$$y = \bigoplus_{i=0}^{N-1} x_i \quad \text{（每个节点均持有 } y\text{）}$$

```
输入:  节点0=[A0], 节点1=[A1], 节点2=[A2], 节点3=[A3]
输出:  节点0=[A0+A1+A2+A3], 节点1=[A0+A1+A2+A3], ...（每节点相同）
```

- **通信量**：单节点收发 $2M(N-1)/N \approx 2M$（Ring-AllReduce）
- **典型场景**：Tensor Parallelism 中各 GPU 的部分 GEMM 结果求和

**1.2 AllGather**

每个节点将本地持有的 $1/N$ 数据分片广播给所有节点，每个节点最终持有**所有节点数据的拼接**：

$$y = [x_0, x_1, \ldots, x_{N-1}] \quad \text{（每个节点均持有完整拼接，大小为 } NM\text{）}$$

```
输入:  节点0=[A], 节点1=[B], 节点2=[C], 节点3=[D]
输出:  每个节点=[A, B, C, D]
```

- **通信量**：每节点发送 $M$，接收 $(N-1)M$，总接收量 $M(N-1)$
- **典型场景**：TP 中 ReduceScatter 后恢复完整激活；Sequence Parallelism 中恢复完整序列

**1.3 ReduceScatter**

先对所有节点数据按位置执行归约，再将结果**均分**为 $N$ 份，每个节点仅保留第 $i$ 份：

$$y_i = \bigoplus_{j=0}^{N-1} x_j!\left[,i \cdot \frac{M}{N} : (i+1) \cdot \frac{M}{N}\right]$$

```
输入:  节点0=[A0,B0], 节点1=[A1,B1], 节点2=[A2,B2], 节点3=[A3,B3]
       （每节点持有完整向量的不同副本）
输出:  节点0=[A0+A1+A2+A3], 节点1=[B0+B1+B2+B3]
       （每节点只有归约结果的 1/N 分片）
```

- **通信量**：每节点发送/接收约 $M(N-1)/N \approx M$
- **关键关系**：$\text{AllReduce} \equiv \text{ReduceScatter} + \text{AllGather}$
- **典型场景**：TP 中 GEMM 结果的第一步归约（配合 AllGather 形成 Overlap 流水）

**1.4 All-to-All（全互换）**

每个节点将本地数据的不同部分**个性化地**发送给对应节点，同时从每个节点接收一部分数据：

```
输入:  节点0=[给0的, 给1的, 给2的, 给3的]
       节点1=[给0的, 给1的, 给2的, 给3的]
       ...
输出:  节点0=[从0来的, 从1来的, 从2来的, 从3来的]
```

- **通信量**：每节点发送 $M$，接收 $M$（总流量 $2MN$ Bytes，含自身的本地块不移动）
- **典型场景**：MoE Expert Parallelism 的 Token 分发（Dispatch）与汇聚（Combine）

**四种通信原语对比：**

|原语|每节点输出大小|通信量（每节点发送）|主要用途|
|---|---|---|---|
|AllReduce|$M$（完整聚合）|$\approx 2M$|TP 梯度/激活聚合|
|AllGather|$NM$（完整拼接）|$M$|恢复完整激活/权重|
|ReduceScatter|$M/N$（归约分片）|$\approx M$|TP 中间步骤|
|All-to-All|$M$（个性化路由）|$M$|MoE EP Token 路由|

---

**Q118. Ring-AllReduce 的通信量分析：总通信量为 $2M(N-1)/N \approx 2M$，与 $N$ 无关？**

**Ring-AllReduce 两阶段详解：**

**阶段 1：ReduceScatter（$N-1$ 步）**

将每个节点的向量 $x_i \in \mathbb{R}^M$ 切分为 $N$ 个 Chunk，每个 Chunk 大小 $M/N$（元素数）。

- 第 $k$ 步（$k = 1, \ldots, N-1$）：节点 $i$ 将自己当前持有的第 $(i - k \bmod N)$ 号 Chunk 传给节点 $(i+1) \bmod N$，同时接收节点 $(i-1) \bmod N$ 传来的对应 Chunk，并执行累加。

经过 $N-1$ 步后，节点 $i$ 持有所有节点在第 $i$ 号位置 Chunk 的完整归约结果。

每节点每步发送 $M/N$ Bytes，共 $N-1$ 步，**每节点总发送量**：

$$V_{\text{RS}} = (N-1) \cdot \frac{M}{N} = \frac{M(N-1)}{N}$$

**阶段 2：AllGather（$N-1$ 步）**

每个节点持有一个已归约的分片，通过 $N-1$ 步环形传播将所有分片广播到每个节点（只传输，不累加）。

$$V_{\text{AG}} = (N-1) \cdot \frac{M}{N} = \frac{M(N-1)}{N}$$

**每节点总发送量：**

$$V_{\text{total}} = V_{\text{RS}} + V_{\text{AG}} = \frac{2M(N-1)}{N}$$

当 $N \to \infty$：

$$\lim_{N \to \infty} \frac{2M(N-1)}{N} = 2M$$

**关键结论：Ring-AllReduce 每节点通信量渐近 $2M$，与节点数 $N$ 无关**。所有链路同时工作，带宽利用率接近 $100\%$。

**与中心化 AllReduce 的对比：**

|方案|Master 节点峰值带宽需求|总通信时间|
|---|---|---|
|中心化 AllReduce|$2M(N-1)$（随 $N$ 线性增长）|$O(NM/B)$|
|Ring-AllReduce|$2M$（常数）|$O(M/B)$（与 $N$ 无关）|

其中 $B$ 为单条链路带宽（Bytes/s）。

**延迟 vs 带宽的权衡：**

Ring-AllReduce 的总步数为 $2(N-1)$，每步存在启动延迟 $\alpha$（约数 μs 量级），总延迟开销为：

$$T = 2(N-1)\alpha + \frac{2M(N-1)}{NB}$$

- **大消息**（$M$ 大）：带宽项主导，Ring-AllReduce 接近最优（Bandwidth-bound）。
- **小消息**（$M$ 小）：延迟项 $2(N-1)\alpha$ 主导，随 $N$ 线性增长（Latency-bound）。

对于 Decode 阶段的 AllReduce（通信量约 1–4 MB，$N$ 较大），延迟项不可忽略，可改用 **Recursive Halving-Doubling**（树形/蝶形）AllReduce，步数降至 $2\log_2 N$，延迟开销为 $O(\log_2 N \cdot \alpha)$，但通信量略高于 Ring（需额外数据填充）。

---

**Q128（新）. GPUDirect RDMA 的工作原理：P 节点如何零拷贝地将 KV Cache 直接写入 D 节点的 HBM？**

**标准 RDMA（无 GPUDirect）的数据路径：**

```
P 节点 GPU HBM
    ↓ D2H（PCIe，GPU → CPU DRAM）
P 节点 CPU DRAM（Pinned Memory）
    ↓ RDMA Send（InfiniBand NIC DMA from CPU DRAM）
D 节点 CPU DRAM（Pinned Memory）
    ↓ H2D（PCIe，CPU DRAM → GPU HBM）
D 节点 GPU HBM
```

共经历**4 次数据移动**，4 次跨 PCIe 总线，延迟高、CPU 占用大。

**GPUDirect RDMA 的数据路径：**

GPUDirect RDMA 通过在 GPU 驱动与 RDMA NIC 驱动之间建立**对等 BAR（Base Address Register）映射**，使 NIC 的 DMA 引擎可直接寻址 GPU HBM 的物理地址：

```
P 节点 GPU HBM
    ↓ NIC DMA（通过 PCIe P2P，NIC 直接读 GPU HBM）
InfiniBand 网络（RDMA Write）
    ↓ NIC DMA（通过 PCIe P2P，NIC 直接写 D 节点 GPU HBM）
D 节点 GPU HBM
```

仅 **1 次网络传输**，CPU 不经手数据（CPU 仅参与控制面：注册内存、建立 QP）。

**关键技术要素：**

**内存注册（Memory Registration）：**

GPU HBM 的物理页在使用前必须向 RDMA 驱动注册，使 NIC 获取对应的物理地址映射：

```cpp
// 注册 GPU 显存为 RDMA 可访问区域
ibv_mr* mr = ibv_reg_mr(
    pd,
    gpu_ptr,          // GPU HBM 虚拟地址（cuMalloc 返回）
    size,
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
);
// mr->lkey / mr->rkey 用于后续 RDMA 操作
```

**RDMA Write 操作（单边操作，D 节点无需 CPU 参与接收）：**

```cpp
// P 节点发起 RDMA Write，直接写入 D 节点 GPU HBM
ibv_send_wr wr = {
    .opcode     = IBV_WR_RDMA_WRITE,
    .sg_list    = &sge,          // 源：P 节点 GPU HBM 地址
    .remote_addr = remote_addr,  // 目标：D 节点 GPU HBM 地址（预先交换）
    .rkey       = remote_rkey,   // D 节点 GPU HBM 的远程访问密钥
};
ibv_post_send(qp, &wr, &bad_wr);
// P 节点 CPU 仅发送 WQE，不参与数据路径
```

**性能参数（InfiniBand NDR 400，H100）：**

|路径|带宽|延迟|CPU 数据参与|
|---|---|---|---|
|标准 RDMA（经 CPU DRAM）|~25 GB/s（PCIe 5.0 限制）|~30–80 μs|有（两次 PCIe 传输）|
|GPUDirect RDMA|~50 GB/s（NDR 400 单端口）|~3–6 μs|无|

**NIXL 与 GPUDirect RDMA 的关系：**

NIXL 在跨节点 KV Transfer 场景下，其 Scatter-Gather 传输底层即依赖 GPUDirect RDMA。NIXL 在 RDMA 层之上封装了 SGE 链表构建、内存注册池（避免每次注册/注销的开销）和完成事件到推理调度器的回调机制。

---

**Q129（新）. InfiniBand 网络中 ECMP（等价多路径）与自适应路由对 All-to-All 通信的影响？**

**背景：MoE EP All-to-All 的流量特征**

MoE Expert Parallelism 中，All-to-All 通信将 Token 从所有节点路由到对应的 Expert 所在节点，再将结果路由回来。流量特征：

- **多对多**：$N$ 个节点同时互相发送/接收，形成 $N \times N$ 流量矩阵。
- **不均匀**：不同 Expert 的 Token 负载不均衡（Load Imbalance），导致部分节点间流量远高于其他。
- **突发性**：每个 Transformer Layer 产生一次 All-to-All，持续时间短（数 ms），但带宽需求大。

**ECMP（Equal-Cost Multi-Path）路由：**

InfiniBand Fat-Tree 网络中存在多条等价路径。ECMP 按**固定哈希**（通常基于源-目标端口对）将流量分配到不同路径：

- **问题**：哈希碰撞导致多个大流被映射到同一路径，而其他路径空闲（Hash Polarization），造成局部拥塞。
- **后果**：All-to-All 中某些 GPU 对之间的有效带宽大幅低于链路峰值，延迟不确定性高。

**自适应路由（Adaptive Routing，InfiniBand 支持）：**

InfiniBand 交换机可根据**实时端口队列深度**动态选择下一跳，将流量引导至当前负载较轻的路径：

```
传统 ECMP：  流 A、B、C 均哈希到路径 1 → 路径 1 拥塞，路径 2/3 空闲
自适应路由：  流 A→路径1，流 B→路径2，流 C→路径3 → 负载均衡
```

**对 All-to-All 吞吐的实际影响（64 节点，MoE All-to-All，InfiniBand HDR）：**

|路由策略|平均带宽利用率|尾延迟（P99）|
|---|---|---|
|ECMP|~55–65%|高（碰撞时 2–3× 平均）|
|自适应路由|**~80–90%**|**低**（接近平均）|

**推理系统的实践意义：**

- DeepSeek V3/R1 等大规模 MoE 推理集群在 InfiniBand 网络配置中明确启用自适应路由，以支持高频 All-to-All 通信。
- GB200 NVL72 通过将 All-to-All 移入 NVLink 域内，根本上规避了 InfiniBand 路由策略对 MoE 延迟的影响。

---

### 2. 通信-计算 Overlap

---

**Q119. Tensor Parallelism 中 GEMM 与 AllReduce 的 Overlap 方案：GEMM-ReduceScatter + AllGather-GEMM 流水线如何实现？**

**传统 TP 的串行瓶颈：**

```
时间轴（Layer L）：
[GEMM W1] → [AllReduce] → [GeLU] → [GEMM W2] → [AllReduce] → 下一层
              ↑ GPU 停止计算等待通信完成（串行）
```

**核心思路：分解 AllReduce**

$$\text{AllReduce} \equiv \text{ReduceScatter} + \text{AllGather}$$

将两个原本不可分割的通信操作变为可与 GEMM Tile 流水的细粒度通信，形成两种 Overlap 模式。

**GEMM-ReduceScatter Overlap（第二个线性层 W2）：**

将输出矩阵按**序列维度（Sequence）** 切分为 $N$ 个 Tile，GEMM 逐 Tile 计算，每完成一个 Tile 立即通过 CUDA Event 触发该 Tile 的 ReduceScatter，与下一个 Tile 的 GEMM 并行执行：

```
时间轴：
Stream 0（计算）: [GEMM Tile 0] [GEMM Tile 1] [GEMM Tile 2] [GEMM Tile 3]
Stream 1（通信）:              [RS Tile 0]   [RS Tile 1]   [RS Tile 2]   [RS Tile 3]
                              ←─────────── 重叠 ───────────────────────────────→
```

**AllGather-GEMM Overlap（第一个线性层 W1）：**

上一层 ReduceScatter 结束后，每个节点持有 $1/N$ 的激活分片。AllGather 按 Chunk 传输，每收到一个 Chunk 立即对该 Chunk 执行 GEMM，无需等待 AllGather 全部完成：

```
时间轴：
Stream 1（通信）: [AG Chunk 0]   [AG Chunk 1]   [AG Chunk 2]   [AG Chunk 3]
Stream 0（计算）:               [GEMM Chunk 0] [GEMM Chunk 1] [GEMM Chunk 2] [GEMM Chunk 3]
                               ←─────────── 重叠 ─────────────────────────────────────→
```

**CUDA 实现关键（双 Stream + CUDA Event）：**

```cpp
cudaStream_t compute_stream, comm_stream;
cudaEvent_t tile_done[N_TILES];

for (int i = 0; i < N_TILES; i++) {
    // 计算 Stream：执行第 i 个 Tile 的 GEMM
    launch_gemm_tile(compute_stream, tile_input[i], weight, tile_output[i]);
    cudaEventRecord(tile_done[i], compute_stream);

    // 通信 Stream：等待 Tile i 计算完成后立即启动 ReduceScatter
    cudaStreamWaitEvent(comm_stream, tile_done[i], 0);
    ncclReduceScatter(tile_output[i], reduced_output[i],
                      tile_elem_count, ncclFloat16, ncclSum,
                      nccl_comm, comm_stream);
}
cudaStreamSynchronize(compute_stream);
cudaStreamSynchronize(comm_stream);
```

**实际加速效果（H100，TP=8，Llama-3 70B，Prefill）：**

|方案|单层时间（归一化）|通信暴露占比|
|---|---|---|
|串行 AllReduce|100%|~15–20%|
|GEMM-RS + AG-GEMM Overlap|**~83–85%**|**近似 0%**（完全隐藏）|

**Overlap 成立的前提：**

- GEMM Tile 的计算时间 $\geq$ 对应 Tile 的 ReduceScatter 通信时间，否则通信尾部无法被隐藏。
- Batch Size 足够大（GEMM 充分 Compute-bound，计算时间远长于通信时间）。
- Decode 阶段（GEMV，计算极快，约 $\mu$s 量级）通信成为主导，此 Overlap 方案收益极有限。

---

**Q120. NCCL 的底层实现：为何 NVLink 通信可直接触发而 PCIe 通信需要 CPU 中介？**

**NVLink 通信（GPU 直连）：**

NVLink 是 NVIDIA 专有的 GPU-GPU 高速互联，物理上将 NVLink 控制器集成在 GPU 芯片内部，形成**点对点高速串行链路**：

```
GPU 0 ←──── NVLink ────→ GPU 1
  ↑                          ↑
NVLink Controller         NVLink Controller
（GPU 芯片内集成）          （GPU 芯片内集成）
```

**直接触发的原因：**

- GPU 内置 DMA 引擎（Copy Engine）可通过 NVLink 直接读写对端 GPU 的 HBM（Peer-to-Peer，P2P），地址空间由 CUDA 统一虚拟地址（UVA）管理。
- CUDA Kernel 或 Copy Engine 在运行时直接发起传输，无需经过 Host CPU。
- NCCL 使用 `cuMemcpyPeerAsync` 或内部封装的 NVLink P2P 接口，单次传输启动延迟 **< 1 μs**。

```cpp
// NVLink P2P 示例（无 CPU 数据路径参与）
cudaMemcpyPeerAsync(
    dst_ptr, dst_device,   // 目标：GPU 1 的 HBM 指针
    src_ptr, src_device,   // 源：  GPU 0 的 HBM 指针
    size_bytes, stream);   // 直接走 NVLink，CPU 不经手数据
```

**PCIe 通信（需 CPU 中介的根本原因）：**

PCIe 总线连接 GPU 与 CPU 的 Root Complex，不同 GPU 通常挂在同一或不同的 PCIe Root Complex 下，GPU 之间**不存在直接物理链路**：

```
GPU 0 ──PCIe──→ PCIe Root Complex (CPU) ←──PCIe── GPU 1
```

**需要 CPU 中介的原因：**

- 标准 PCIe 协议下，跨 Root Complex 的 P2P 读写在许多平台上**不被硬件支持**（需 BIOS/驱动启用 ACS，Access Control Services）。
- 未启用 GPUDirect P2P 时，数据路径为：GPU 0 HBM → CPU DRAM → GPU 1 HBM，经历两次 DMA 拷贝，CPU 参与协调。
- 即使启用 GPUDirect P2P（要求两块 GPU 在同一 PCIe Switch 下），也仅消除了 CPU DRAM 的中转拷贝，带宽仍受 PCIe 总线限制。

**NCCL 的拓扑感知后端选择：**

```
NCCL 初始化时调用 ncclGetUniqueId 并探测拓扑：
  同节点，NVLink 可达     → NVLink P2P（最优）
  同节点，GPUDirect P2P  → PCIe P2P（带宽受限）
  同节点，无 P2P 支持     → 通过 CPU DRAM 中转
  跨节点，IB 可用         → InfiniBand RDMA（GPUDirect RDMA）
  跨节点，仅以太网        → TCP/IP（最慢）
```

**性能量化对比（H100 系统）：**

|通信路径|带宽|延迟|CPU 数据路径|
|---|---|---|---|
|NVLink P2P（8× H100 NVSwitch）|900 GB/s（双向总计）|< 1 μs|无|
|PCIe P2P（GPUDirect，同 Switch）|~64 GB/s（PCIe 5.0 ×16）|~5–10 μs|无（硬件支持时）|
|PCIe CPU 中转|~25–30 GB/s（受 CPU 内存带宽限制）|~20–50 μs|有（两次 DMA）|
|InfiniBand NDR 200 RDMA|~50 GB/s（单端口双向）|~1–3 μs|无（GPUDirect RDMA）|

---

**Q121. NIXL（NVIDIA Inference Xfer Library）相比 NCCL 在 KV Transfer 场景的优化点？**

**NCCL 的设计定位与局限：**

NCCL 为**同构训练集群**设计，优化目标是同步的大规模 AllReduce / AllGather / ReduceScatter，能够解决下述：

- 假设通信数据在**物理连续显存** 中（Contiguous Buffer），不支持 Scatter-Gather 描述符。
- 针对**对称通信**优化（所有节点通信量相同、角色对等），Communicator 拓扑固定。
- Communicator 初始化（`ncclCommInitRank`）需要所有节点 Barrier 同步，开销大，不适合动态变化的通信拓扑。
- 同步语义（`ncclGroupEnd` 处阻塞）与推理引擎的异步调度模型不兼容。

**KV Cache Transfer 的特殊需求（PagedAttention 场景）：**

```
KV Cache 物理存储示意（PagedAttention）：
  物理 Block 0: [Layer 0~3, Token  0~15, K/V]  → HBM 地址 0x1000_0000
  物理 Block 7: [Layer 0~3, Token 16~31, K/V]  → HBM 地址 0x5000_0000（不连续）
  物理 Block 3: [Layer 0~3, Token 32~47, K/V]  → HBM 地址 0x2000_0000（散布各处）
```

KV Cache 的 Block 在显存中**非连续分布**（PagedAttention 的本质）。用 NCCL 传输需要额外步骤：

1. P 节点：将散布的 Block Gather 为连续临时 Buffer（1 次 D2D 拷贝）。
2. NCCL 发送连续 Buffer。
3. D 节点：将连续 Buffer Scatter 回目标 Block 位置（1 次 D2D 拷贝）。

共引入 **2 次额外显存拷贝**，每次拷贝消耗约 2–3 ms（335 MB KV Cache，3.35 TB/s HBM）。

**NIXL 的核心优化：**

**① 原生 Scatter-Gather DMA（消除额外拷贝）：**

NIXL 直接接受非连续内存描述符列表，由 RDMA NIC 的硬件 DMA 引擎（SGE，Scatter-Gather Element）原生处理非连续地址，无需预先 Gather：

```python
# NIXL Python 接口（示意）
sg_list = nixl.build_sg_list([
    MemDesc(addr=block_table[0], size=block_size),
    MemDesc(addr=block_table[7], size=block_size),
    MemDesc(addr=block_table[3], size=block_size),
    # ... 逻辑上连续、物理上散布的 KV Block
])
transfer_handle = nixl.isend(sg_list, dst_rank=decode_rank, tag=req_id)
# 异步，立即返回
```

**② 针对推理流量特征的协议优化：**

- **单向非对称通信**：P 节点发送、D 节点接收，NIXL 针对此优化连接建立（仅单向握手）和零拷贝接收缓冲区管理。
- **小-中消息优化**：KV Block 通常为 16–128 KB，NIXL 在此消息尺寸下的 Latency 优化好于 NCCL（NCCL 的零拷贝路径针对 ≥ 1 MB 的大消息优化）。
- **动态目标节点**：不同请求的 KV 可能发往不同 D 节点，NIXL 支持每次传输动态指定目标；NCCL Communicator 在初始化后目标拓扑固定，灵活性差。

**③ 与推理调度器深度集成：**

NIXL 提供**完全异步的基于事件的 API**（类似 RDMA Completion Queue），KV Transfer 完成后通知调度器（而非 NCCL 的同步 Barrier 语义），与 Continuous Batching 的逐迭代（iteration-level）调度无缝配合，使调度器可在 KV Transfer 完成后立即将对应请求放入 Decode Batch。

**性能对比（P/D 分离，Llama-3 70B，KV Cache 约 335 MB/请求，InfiniBand NDR 400）：**

|方案|端到端传输时间|额外显存拷贝|调度灵活性|
|---|---|---|---|
|NCCL（Gather + 传输 + Scatter）|~15 ms|**2 次**（~4–6 ms 额外）|差（固定 Comm Group）|
|NIXL（原生 Scatter-Gather）|**~8–9 ms**|**0 次**|好（动态目标，异步）|

NIXL 在 KV Transfer 场景端到端延迟比 NCCL 低约 40–50%，收益来源：消除 2 次额外显存拷贝（~4–6 ms）+ 小消息路径延迟优化。

---

## 第 19 章·参考答案：新硬件特性

---

### 3. H100 新特性

---

**Q122. TMA（Tensor Memory Accelerator）的工作原理：如何替代 `cp.async` 实现多维张量的异步加载？**

**`cp.async` 的局限（Ampere 时代）：**

`cp.async` 允许 GPU 线程发起异步 HBM→SRAM 拷贝，主线程继续计算（计算-访存重叠）。局限：

- 地址计算（多维 Tensor 的 Stride 偏移）由**软件线程**承担，消耗寄存器文件（Register File）和 ALU 资源，减少可用于计算的寄存器数量。
- 每条 `cp.async` 指令单次拷贝 4 / 8 / 16 Bytes，大 Tile（如 128×64 的 FP16 = 16 KB）需要发射 1024 条指令，造成 Instruction Issue 流水线压力，占用 Warp 的 Issue Slot。
- 仅支持线性（一维）地址，二维 Tile 的行间 Stride 需要外层循环手动计算偏移，代码复杂。
- 每个参与加载的 Warp 均需发射 `cp.async` 指令，指令并发量受 Warp 数量限制。

**TMA（Tensor Memory Accelerator，Hopper H100 引入）：**

TMA 是 H100 每个 SM 中的**专用硬件功能单元**，独立于 CUDA Core 和 Tensor Core，可自主完成多维张量的异步加载/存储，**彻底卸载软件线程的地址计算和指令发射开销**。

**核心概念：Tensor Map（`CUtensorMap`，张量描述符）：**

在 Host 端（推理初始化阶段）预先创建描述符，编码 Tensor 的完整多维布局：

```cpp
// Host 端创建 Tensor Map（一次性，推理全程复用）
CUtensorMap tma_desc;
cuTensorMapEncodeTiled(
    &tma_desc,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    2,                // rank（维度数）：此处为 2D [rows, cols]
    global_addr,      // HBM 中全局 Tensor 的基地址
    global_dims,      // 全局 Tensor 形状，如 {M, K}
    global_strides,   // 各维度步长（Bytes）：{K * sizeof(half), sizeof(half)}
    box_dims,         // 单次 TMA 加载的 Tile 形状，如 {Bm, Bk}
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,   // Shared Memory Bank Conflict 消除
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);
```

**Kernel 内的 TMA 加载（单条 PTX 指令加载整个 Tile）：**

```cuda
__shared__ alignas(128) half smem_tile[Bm][Bk];
__shared__ uint64_t tma_barrier;

// 仅 Producer Warp 的一个线程发射（整个 Thread Block 的加载由此一条指令驱动）
if (threadIdx.x == 0) {
    __mbarrier_init(&tma_barrier, 1);
    // 发起异步 TMA 加载：从全局坐标 [row_offset, col_offset] 加载 Bm×Bk Tile
    // PTX: cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_tile), "l"(&tma_desc),
           "r"(row_offset), "r"(col_offset), "r"(&tma_barrier)
    );
}
__syncthreads();

// Consumer 线程等待 TMA 完成（mbarrier phase-based synchronization）
uint32_t phase = 0;
__mbarrier_wait(&tma_barrier, phase);

// 使用 smem_tile 执行 WGMMA
wgmma::mma_async(smem_tile, ...);
```

**TMA vs `cp.async` 对比：**

|维度|`cp.async`（Ampere）|TMA（Hopper）|
|---|---|---|
|地址计算|软件线程（消耗 ALU/寄存器）|**硬件 TMA 单元**（零软件开销）|
|多维支持|仅 1D（需手动 Stride 循环）|**原生 1D–5D**（硬件处理）|
|单次传输大小|4–16 Bytes|**整个 Tile（任意大小，上限约 256 KB）**|
|同步机制|`cp.async.wait_group/wait_all`|**mbarrier（细粒度，Tile 级，支持 Phase）**|
|Warp 指令开销|每个 Warp 均需发射 $n$ 条|**单线程发射 1 条指令**（TMA 硬件自主完成）|
|Swizzle 支持|无（Bank Conflict 需软件处理）|**硬件原生支持 128B Swizzle**|
|与 WGMMA 配合|间接（需手动调度 Ping-Pong）|**深度集成**（TMA + WGMMA + mbarrier 三件套）|

**对 FlashAttention-3 的意义：**

FA-3 利用 TMA 将 Q、K、V 的 Tile 加载完全交给 Producer Warp 中的单个线程（一条 TMA 指令加载整个 Tile），Consumer Warp Group（WGMMA 计算）与 TMA 加载通过 mbarrier 精确同步并完全重叠，MFU 从 FA-2 的 ~50–60% 提升至 **~75%**。

---

**Q123. Warp Specialization（Warp 专用化）的 Producer-Consumer 设计模式？**

**背景：传统 CUDA Kernel 的 Warp 同质化瓶颈：**

传统 CUDA Kernel 中所有 Warp 执行相同的代码路径（Load + Compute 交替），两类工作的硬件资源需求存在根本冲突：

- **数据加载阶段**：需要 Copy Engine / 内存子系统带宽，Tensor Core 空转。
- **GEMM 计算阶段**：Tensor Core 满负荷，内存带宽空闲。
- 两者串行交替，任意时刻只有一类硬件在工作，利用率理论上不超过 50%。

**Warp Specialization（Hopper 推荐设计模式）：**

将同一 Thread Block 内的 Warp 静态分配为两类角色，各自只执行一类工作：

```
┌─────────────────────────────────────────────┐
│              Thread Block                   │
│                                             │
│  ┌───────────────┐     ┌─────────────────┐  │
│  │  Producer     │     │  Consumer       │  │
│  │  Warp(s)      │     │  Warp Group(s)  │  │
│  │               │     │                 │  │
│  │ ① TMA 加载    │     │ ① WGMMA 执行   │  │
│  │   K/V/权重    │     │   矩阵乘累加    │  │
│  │   Tile        │     │                 │  │
│  │ ② Softmax     │     │ ② 累加器维护   │  │
│  │   Row-max/sum │     │   输出 Tile     │  │
│  │   标量运算    │     │                 │  │
│  └──────┬────────┘     └────────┬────────┘  │
│         │    Shared Memory      │            │
│         └───────────────────────┘            │
│             （mbarrier 同步）                │
└─────────────────────────────────────────────┘
```

**Producer Warp（数据供应者）：**

- 发射 TMA 异步加载指令（仅 1 条即可加载整个 Tile），不阻塞等待完成。
- 加载完成后通过 `mbarrier::arrive` 通知 Consumer。
- 不执行 WGMMA，其 Tensor Core 资源全部留给 Consumer Warp Group。

**Consumer Warp Group（计算执行者）：**

- 调用 `mbarrier::wait` 等待 Producer 的完成信号，随后立即发射 WGMMA 指令（异步执行）。
- 不发射内存加载指令，寄存器文件全部分配给 WGMMA 累加器（最大化 Occupancy 和累加器精度）。

**双缓冲 Ping-Pong 流水（2-Stage Pipeline）：**

SRAM 中分配两组 Buffer（Stage 0 和 Stage 1 交替使用）：

```
时间轴（K 维度迭代）：
Producer: [TMA 加载 Tile_K[0]→buf0] [TMA 加载 Tile_K[1]→buf1] [TMA 加载 Tile_K[2]→buf0] ...
                     ↓ arrive(0)               ↓ arrive(1)
Consumer:            [WGMMA buf0→acc]           [WGMMA buf1→acc]           [WGMMA buf0→acc] ...
                     ←────────────────── 加载与计算完全重叠 ──────────────────────────────→
```

**为何能消除等待：**

- Producer TMA 加载 Tile_K[1] 时，Consumer 同时执行 Tile_K[0] 的 WGMMA。
- 只要 TMA 加载时间 $\leq$ WGMMA 计算时间，加载延迟被完全隐藏。
- H100 Tensor Core 在 FP8 下计算速度极快（~1979 TFLOPS），而 TMA 带宽（HBM ~3.35 TB/s）是主要限制，两者时间接近，双缓冲效果显著。

**FlashAttention-3 中的具体分工：**

|Warp 角色|工作内容|使用的硬件单元|
|---|---|---|
|Producer Warp|TMA 加载 K Tile、V Tile；计算 Softmax row-max 和 row-sum|TMA 单元；CUDA Core（标量运算）|
|Consumer Warp Group|`WGMMA(Q, K^T)` → Score 矩阵；`WGMMA(Score_softmax, V)` → Output|Tensor Core（WGMMA 异步指令）|

**收益量化（H100 FP8 GEMM / FlashAttention-3）：**

|方案|MFU|说明|
|---|---|---|
|无 Warp Specialization（FA-2 风格）|~50–60%|Load 与 Compute 串行，Tensor Core 等待|
|有 Warp Specialization（FA-3）|**~72–75%**|TMA 加载与 WGMMA 完全重叠|

---

**Q124. H100 FP8 格式：E4M3 vs E5M2 的动态范围与精度权衡？**

**FP8 的两种格式（基于 Nvidia 标准）：**

浮点数格式：$1$ bit 符号 + $E$ bits 指数 + $M$ bits 尾数，共 8 bits。指数采用 Bias 编码，Bias 值不同于 IEEE 754 标准（由 NVIDIA 特别定义，含 NaN 与 Inf 的处理差异）。

| 格式       | 指数位 $E$ | 尾数位 $M$ | 指数 Bias | 动态范围                                    | 最小正规格化值                              |
| -------- | ------- | ------- | ------- | --------------------------------------- | ------------------------------------ |
| **E4M3** | 4       | 3       | 7       | $\approx [1.95 \times 10^{-3},\ 448]$   | $2^{-6} \approx 0.0156$              |
| **E5M2** | 5       | 2       | 15      | $\approx [1.53 \times 10^{-5},\ 57344]$ | $2^{-14} \approx 6.1 \times 10^{-5}$ |

**最大可表示值推导（E4M3）：**

E4M3 中指数域最大非特殊值为 $1110_2 = 14$（$1111_2$ 保留用于 NaN），尾数域最大值为 $111_2$，表示 $1 + 7/8 = 15/8$：

$$\text{E4M3}_{\max} = \frac{15}{8} \times 2^{15 - 7} = \frac{15}{8} \times 128 = 448$$

**各自适用场景：**

**E4M3FN（权重与激活值，前向推理）：**

- 尾数 3 bits，相邻值间距约为 E5M2 的一半，精度更高。
- 动态范围足以覆盖 LLM 权重（多集中于 $[-1, 1]$）和 SmoothQuant 后的激活值（Outlier 控制在 $[-448, 448]$ 内）。
- H100 FP8 Tensor Core 前向推理标准格式。

**E5M2（梯度，反向传播）：**

- 动态范围 $\approx 57344$，比 E4M3 大两个数量级，适应梯度分布跨度宽的特点。
- 精度稍低（尾数 2 bits），但梯度噪声在随机梯度下降中本身可被接受。
- FP8 混合精度训练标准方案：前向 E4M3，反向 E5M2。

**H100 FP8 Tensor Core 使用路径：**

```
前向推理：
  Activation (E4M3FN) × Weight (E4M3FN) → FP32 Accumulate → BF16/FP16 Output

训练前向：
  Activation (E4M3FN) × Weight (E4M3FN) → FP32 Accumulate → BF16 Output

训练反向：
  Gradient (E5M2) × Weight (E4M3FN) → FP32 Accumulate → BF16 Weight Gradient
```

**Scale Factor 的必要性：**

FP8 动态范围远小于 FP16（E4M3 最大 448 vs FP16 最大 65504），直接量化会导致大量值超出范围（Overflow 截断为最大值）或精度损失（数值聚集在低端）。

H100 提供硬件 `AMAX` 指令，可在 Kernel 内高效计算 Tensor 最大绝对值，用于计算 Per-tensor 或 Per-token 的 Scale Factor：

$$\text{scale} = \frac{\text{FP8}_{MAX}}{\text{AMAX}(X)} = \frac{448}{\max_i |x_i|}$$

$$x_{\text{fp8}} = \text{round\_to\_fp8}(x \cdot \text{scale})$$

---

**Q130（新）. H100 Thread Block Cluster 与 Distributed Shared Memory（DSMEM）的工作原理及在 FlashAttention-3 中的应用？**

**背景：传统 Shared Memory 的局限**

传统 CUDA 编程中，Shared Memory 仅在同一 Thread Block 内共享，跨 Thread Block 的通信必须经由 Global Memory（HBM），延迟高（约 200–800 ns），不适合细粒度的 Tile 级数据共享。

**Thread Block Cluster（Hopper 引入）：**

H100 允许将多个（2–16 个）Thread Block 组成一个 **Cluster**，同一 Cluster 内的 Thread Block 被调度到**物理相邻的 SM** 上执行，可通过 DSMEM 机制直接访问彼此的 Shared Memory，无需经过 HBM：

```cpp
// 定义 Cluster 大小（编译时或运行时）
__cluster_dims__(2, 1, 1)   // 2 个 Thread Block 构成 1 个 Cluster
__global__ void fused_kernel(...) {
    // 获取 Cluster 内的 Block 索引
    uint32_t block_rank = __cluster_block_rank();

    // 直接读写邻居 Block 的 Shared Memory（DSMEM）
    void* peer_smem = __cluster_map_shared_rank(local_smem_ptr, peer_block_rank);
    // peer_smem 指向邻居 Block 的 Shared Memory 地址，延迟约 30–50 ns
}
```

**延迟对比：**

|访问路径|延迟|带宽|
|---|---|---|
|本地 Shared Memory|~23 cycles（~10 ns）|~19 TB/s（per SM）|
|DSMEM（同 Cluster 内相邻 SM）|~30–50 cycles（~20 ns）|~6–8 TB/s|
|Global Memory（HBM）|~600–800 cycles（~200–300 ns）|3.35 TB/s（共享）|

**在 FlashAttention-3 中的应用：**

FA-3 利用 Cluster 将 $Q$ 的不同 Tile 分配给同 Cluster 内的不同 SM，而 $K/V$ 通过 DSMEM 在 Cluster 内广播共享，避免每个 SM 独立从 HBM 加载相同的 $K/V$ Block：

```
Cluster（2 SM）：
  SM 0：处理 Q[0:64, :]     ← K/V Tile 从 SM 0 的 SMEM 读取
  SM 1：处理 Q[64:128, :]   ← K/V Tile 通过 DSMEM 从 SM 0 的 SMEM 读取（无需重新从 HBM 加载）
```

**收益**：$K/V$ 加载次数减半（Cluster 内 SM 共享），HBM 带宽需求降低 $\sim 30\%$，进一步提升长序列下的 FlashAttention 吞吐。

---

### 4. Blackwell 新特性

---

**Q125. NVFP4（FP4 with block-level FP8 scale）的存储格式与 Tensor Core 支持。**

**NVFP4 的数值格式：**

NVFP4（NVIDIA 定义，规范名称 FP4 E2M1）：1 bit 符号 + 2 bits 指数 + 1 bit 尾数，共 4 bits，指数 Bias = 1。

可表示值域（有限正数）：

$${0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}$$（及对应负数，共 15 个有限值 + 1 个 NaN）

**块级 FP8 Scale 的必要性：**

NVFP4 动态范围极小（最大值仅 6.0，最小非零值 0.5），直接量化 LLM 激活值/权重必然导致严重精度损失。解决方案：每 **16 个连续的 FP4 元素**共享 1 个 **FP8 E4M3** Scale Factor：

$$x_{\text{FP4},k} = \text{round\_to\_fp4}\!\left(\frac{x_k}{s}\right), \quad s = \frac{6.0}{\max_{k \in \text{block}}|x_k|}$$

其中 $s$ 以 FP8 E4M3 存储，每 16 个 FP4 元素 + 1 个 FP8 Scale 的实际平均位宽为：

$$\text{平均位宽} = 4 + \frac{8}{16} = 4.5 \text{ bits/element}$$

**存储布局：**

```
权重矩阵（NVFP4 格式，行主序）：
┌──────────────────────────────────────────┐
│  FP4[0..15]  │  FP8 Scale[0]  │  ...     │  ← 每 16 个 FP4 值对应 1 个 FP8 Scale
│  FP4[16..31] │  FP8 Scale[1]  │  ...     │
│  ...                                     │
└──────────────────────────────────────────┘
存储密度：(16×4 + 8) bits / 16 elements = 4.5 bits/element
vs FP16：16 bits/element → 压缩比 ≈ 3.56×
vs INT8：8 bits/element  → 压缩比 ≈ 1.78×
```

**Blackwell FP4 Tensor Core 数据流：**

```
输入路径：
  权重（NVFP4）：HBM → L2 Cache → SRAM
  激活（FP8 E4M3）：HBM → L2 Cache → SRAM

Tensor Core 内置 FP4 解压单元（硬件自动）：
  每 16 个 FP4 权重值 × FP8 Scale → 等效 FP8 × 16
  FP8 激活 × 等效 FP8 权重 → FP32 Accumulate（MMA）
  → 输出 BF16/FP16
```

**NVFP4 的 WGMMA 指令（PTX 示意）：**

```
// Blackwell wgmma.mma_async，FP4 × FP8 输入
wgmma.mma_async.sync.aligned.m64n256k128.f32.e2m1.e4m3
    d_reg,           // FP32 累加器（寄存器文件）
    a_smem_fp4,      // A 矩阵（SRAM，NVFP4 压缩格式，含 Scale）
    b_smem_fp8,      // B 矩阵（SRAM，FP8 E4M3）
    scale_a,         // A 的 FP8 Block Scale（每 16 元素一个）
    scale_b;         // B 的 FP8 Block Scale
```

**与 H100 FP8 的核心差异：**

|特性|H100 FP8 E4M3|B200 NVFP4|
|---|---|---|
|权重存储位宽|8 bits|**4 bits**（+0.5 bits Scale 开销）|
|Scale 粒度|Per-tensor 或 Per-token|**Per-16-elements（FP8 E4M3 Scale）**|
|MMA 输入组合|E4M3 × E4M3 / E5M2 × E4M3|**E2M1（FP4）× E4M3（FP8）**|
|理论峰值（Dense）|~989 TFLOPS（H100）|**~9000 TFLOPS（B200，估算）**|
|显存带宽节省|2× vs FP16|**~3.6× vs FP16**（含 Scale 开销）|
|精度损失（感知质量）|< 0.5%|0.5–1.5%（取决于模型和 PTQ 方案）|

---

**Q126. GB200 NVL72 系统的硬件规格与推理意义。**

**GB200 NVL72 规格：**

|参数|数值|
|---|---|
|GPU 数量|**72 × B200 GPU**|
|CPU 数量|72 × Grace CPU（ARM Neoverse V2）|
|GPU-CPU 封装方式|每对 Grace + B200 构成 1 个 GB200 模块（MCM）|
|NVLink Switch 芯片|NVSwitch 4（第四代，72 GPU 全互联）|
|总 HBM3e 显存|**72 × 192 GB = 13.824 TB**|
|每 GPU HBM 带宽|8.0 TB/s|
|总 NVLink 带宽|**~1.8 TB/s（单向）/ 3.6 TB/s（双向）**|
|单 GPU 峰值（NVFP4 Dense）|~9 PFLOPS|
|总系统峰值（NVFP4 Dense）|**~648 PFLOPS**|
|CPU-GPU 互联（C2C）|NVLink-C2C，900 GB/s（双向）|
|Grace CPU 内存|480 GB LPDDR5X（每 CPU）|

**NVL72 的关键推理意义：**

**① 超大 NVLink 域（72 GPU 全互联，规避 InfiniBand 瓶颈）：**

H100 单节点 NVLink 域最大 8 GPU，跨节点依赖 InfiniBand（单端口 ~50 GB/s，比节点内 NVLink 慢约 18×）。GB200 NVL72 通过 NVSwitch 4 将 72 GPU 构成**单一 NVLink 全互联域**：

- **TP = 72** 全程 NVLink，GEMM-AllReduce Overlap 效果无需 InfiniBand 降级。
- **MoE EP = 72**，All-to-All 通信延迟 < 1 μs（NVLink），而 InfiniBand 场景下约 5–20 μs。
- P/D 分离场景：同一 NVL72 机柜内的 KV Transfer 通过 NVLink 进行，峰值带宽远超 IB。

**② 13.8 TB 总显存（单 NVLink 域内装载超大模型）：**

|系统|总显存|可装载 FP16 模型规模|节点间互联|
|---|---|---|---|
|8× H100 单节点|640 GB|~320B 参数|NVLink（域内）|
|16× H100（跨节点）|1.28 TB|~640B 参数|InfiniBand|
|**GB200 NVL72**|**13.8 TB**|**~6.9T 参数**|**NVLink（域内）**|

**③ Grace CPU 紧耦合（NVLink-C2C）：**

每个 B200 GPU 通过 NVLink-C2C 与 Grace CPU 以 900 GB/s 互联（比 PCIe 5.0 × 16 的 ~128 GB/s 高约 7×）。Grace CPU 的 480 GB LPDDR5X 内存可作为 GPU HBM 的高速扩展（CPU 内存存放部分权重，访问带宽显著高于传统 CPU 内存 + PCIe 路径）。

**④ 对 Decode 吞吐的影响（实际 vs 理论）：**

Decode 阶段受 HBM 带宽主导（Memory-bound）。B200 每 GPU HBM 带宽为 8.0 TB/s（H100 为 3.35 TB/s），提升约 2.4×；同时 192 GB 显存（H100 80 GB）允许更大 KV Cache 和更大 Batch Size，综合 Decode 吞吐提升约 **3–5×**。

---

**Q127. NVFP4 的理论峰值 TFLOPS 相比 H100 FP8 的提升倍数推算？**

**推算原理：**

Tensor Core 的峰值 TFLOPS 与每个时钟周期处理的操作数成正比。相同 MMA 指令形状（如 `m16n8k16`）下，寄存器位宽固定，FP4 操作数是 FP8 操作数的 2 倍，因此：

$$\text{TFLOPS}_{\text{FP4}} = \text{TFLOPS}_{\text{FP8}} \times 2$$

**从 H100 FP8 推算 B200 NVFP4（逐步推导）：**

**Step 1：H100 FP8 Dense 峰值**

$$\text{H100 FP8 Dense} = 989 \text{ TFLOPS}$$

**Step 2：B200 架构相比 H100 的基准提升倍数**

NVIDIA 官方发布 B200 FP8 Dense 峰值约 4.5 PFLOPS（= 4500 TFLOPS），故架构提升倍数：

$$k = \frac{4500}{989} \approx 4.55\times$$

（来源：SM 数量提升 + 每 SM Tensor Core 通量提升 + 时钟频率提升的综合效果）

**Step 3：B200 NVFP4 Dense 峰值**

$$\text{B200 NVFP4 Dense} \approx 4500 \times 2 = 9000 \text{ TFLOPS}$$

**Step 4：B200 NVFP4 Sparse（2:4 结构化稀疏，每个 MMA 跳过 50% 零元素）**

$$\text{B200 NVFP4 Sparse} \approx 9000 \times 2 = 18{,}000 \text{ TFLOPS}$$

**与 H100 FP8 的倍数关系：**

$$\frac{\text{B200 NVFP4 Dense}}{\text{H100 FP8 Dense}} = \frac{9000}{989} \approx \mathbf{9.1\times}$$

$$\frac{\text{B200 NVFP4 Sparse}}{\text{H100 FP8 Sparse}} = \frac{18{,}000}{1979} \approx \mathbf{9.1\times}$$

**系统级实际提升（瓶颈分析）：**

|瓶颈因素|B200 vs H100 提升|受影响的工作负载|
|---|---|---|
|Tensor Core 峰值（FP4 vs FP8）|**~9.1×**|Prefill（Compute-bound GEMM）|
|HBM 带宽（8.0 vs 3.35 TB/s）|**~2.4×**|Decode（Memory-bound，权重加载主导）|
|HBM 容量（192 vs 80 GB/卡）|**~2.4×**|最大 Batch Size / KV Cache 容量|
|NVLink 带宽（NVL72 域 vs 8× H100）|**~4×**|TP/EP 通信延迟|

**关键结论：**

- **Prefill 场景**（Compute-bound）：接近 9.1× 理论峰值（GEMM 充分利用 FP4 Tensor Core）。
- **Decode 场景**（Memory-bound）：约 **2.4×**（HBM 带宽主导，非计算峰值）。
- **Decode 综合吞吐**：GB200 NVL72 相比 8× H100 节点约 **3–5×**（综合 HBM 带宽提升 + 显存容量增大支持更大 Batch + NVL72 消除跨节点通信瓶颈三重效果）。

---

**Q131（新）. Blackwell NVSwitch 4 的交换架构与其相比 NVSwitch 3（H100）的带宽提升来源？**

**NVSwitch 演进对比：**

|特性|NVSwitch 3（Hopper/H100）|NVSwitch 4（Blackwell/B200）|
|---|---|---|
|最大互联 GPU 数|8（单节点 NVL8）|**72（NVL72 机柜）**|
|每芯片端口数|64 × NVLink 4 端口|64 × NVLink 5 端口|
|每端口带宽|~25 GB/s（NVLink 4）|~50 GB/s（NVLink 5）|
|单芯片聚合带宽|~7.2 TB/s|**~13 TB/s**|
|NVL72 总交换带宽|—|**~1.8 PB/s**（多片 NVSwitch 4 互联）|

**NVSwitch 4 带宽提升来源：**

NVLink 5 相比 NVLink 4 的单端口带宽翻倍（~50 GB/s vs ~25 GB/s），原因是信号速率提升（PAM4 编码 + 更高串行速率）和更宽的物理通道数。

**对推理的意义：**

NVL72 中 72 个 B200 GPU 通过多颗 NVSwitch 4 芯片实现全互联，任意两 GPU 间的通信不经过 CPU 或 InfiniBand，带宽和延迟均远优于 InfiniBand 方案，是 MoE EP All-to-All 和大规模 TP 在单机柜内可行的硬件基础。

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
