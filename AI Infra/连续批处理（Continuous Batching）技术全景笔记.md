## 1. 背景与问题定义

### 1.1 什么是 LLM Serving？

**LLM Serving（Large Language Model Serving）** 指在在线系统中，为大量并发用户请求提供大语言模型推理服务，并在满足延迟约束的情况下返回生成结果。

与训练（Training）和离线推理（Offline Inference）不同，Serving 具有以下特点：
- 请求实时到达，无法提前批量规划；
- Prompt 长度未知，输入规模变化大；
- 输出长度未知，生成过程具有随机性；
- 用户通常要求 Streaming 输出，即逐 token 返回结果。

因此，LLM Serving 的核心问题不是单次推理速度最大化，而是：

> 在动态请求环境下，通过调度策略最大化 GPU 利用率，同时保证 TTFT、TPOT 和 p99 latency 满足 SLA。

### 1.2 LLM Serving 核心性能指标

#### Throughput

**Throughput（吞吐量）** 表示单位时间生成 token 的数量，是衡量系统处理能力的核心指标。

通常定义为：

$$  
\begin{aligned}  
\text{Throughput}  
&= \frac{N_{\text{output tokens}}}{T_{\text{total}}}  
\end{aligned}  
$$

其中：
- $N_{\text{output tokens}}$ 表示生成 token 总数量；
- $T_{\text{total}}$ 表示服务运行时间。

当并发请求增加时，Continuous Batching 通过提高 GPU batch 利用率提升 Throughput。

#### TTFT（Time To First Token）

**TTFT** 表示请求到达后，第一个输出 token 返回所需要的时间。

$$  
\begin{aligned}  
\text{TTFT}  
&= T_{\text{queue}}
- T_{\text{prefill}}
- T_{\text{schedule}}  
    \end{aligned}  
    $$

其中：
- $T_{\text{queue}}$ 表示请求等待调度时间；
- $T_{\text{prefill}}$ 表示 Prompt Encoding 计算时间；
- $T_{\text{schedule}}$ 表示调度器等待执行时间。

TTFT 主要影响用户首次响应体验。

#### TPOT（Time Per Output Token）

**TPOT** 表示 Decode 阶段生成相邻两个 token 的平均时间间隔：

$$  
\begin{aligned}  
\text{TPOT}  
&= \frac{T_{\text{decode}}}{N_{\text{output tokens}}}  
\end{aligned}  
$$

其中：
- $T_{\text{decode}}$ 表示整个 Decode 阶段耗时；
- $N_{\text{output tokens}}$ 表示生成 token 数。


TPOT 决定 Streaming 输出的连续性。

#### p99 Latency

**p99 latency** 表示 99% 请求能够完成的最大延迟。

相比平均延迟：

- 平均 latency 反映整体性能；
- p99 latency 反映系统长尾稳定性。

LLM Serving 中，由于请求长度和生成长度存在巨大差异，p99 通常比平均值更重要。

---

## 2. LLM Serving 的核心挑战

### 2.1 请求生命周期不可预测

在线 Serving 中，请求具有以下不确定性：

|因素|影响阶段|问题|
|---|---|---|
|请求到达时间|Scheduler|无法提前组成固定 batch|
|Prompt 长度|Prefill|计算量变化|
|生成长度|Decode|生命周期变化|

其中生成长度差异是 Batch 调度问题的主要来源。

设 batch 中第 $i$ 个请求生成 token 数为 $L_{i}$，则静态 batch 完成时间由最长请求决定：

$$  
\begin{aligned}  
T_{\text{batch}}  
&= \max(L_{1},L_{2},...,L_{B})  
\end{aligned}  
$$

其中：
- $B$ 表示 batch size；
- $L_{i}$ 表示第 $i$ 个请求需要生成的 token 数。

当存在一个长请求时，短请求完成后仍然占用 batch slot，导致 GPU 利用率下降。

### 2.2 Serving 优化目标

LLM Serving 优化不是简单降低单次推理时间，而是在多个目标之间进行权衡：

$$  
\begin{aligned}  
\max \quad  
& \text{Throughput}  
\\  
\text{s.t.}\quad  
& \text{TTFT} \leq S_{\text{TTFT}}  
\\  
& \text{TPOT} \leq S_{\text{TPOT}}  
\\  
& \text{p99 latency}\leq S_{\text{p99}}  
\end{aligned}  
$$

其中：
- $S_{\text{TTFT}}$ 表示 TTFT 服务约束；
- $S_{\text{TPOT}}$ 表示 TPOT 服务约束；
- $S_{\text{p99}}$ 表示尾延迟约束。

---

## 3. Prefill 与 Decode 分离
### 3.1 为什么 LLM 推理需要拆分？

LLM 推理天然分为两个阶段：
1. Prefill：
    - 输入阶段；
    - 一次处理整个 Prompt；
    - 计算并保存 KV Cache。
2. Decode：
    - 自回归生成阶段；
    - 每次生成一个 token；
    - 重复执行直到 EOS。

两个阶段具有完全不同的硬件特征：

|维度|Prefill|Decode|
|---|---|---|
|输入规模|完整 Prompt|单 token|
|计算特点|大矩阵计算|小 batch 迭代|
|主要瓶颈|Compute Bound|Memory Bandwidth Bound|
|并行度|高|低|
|延迟敏感度|较低|极高|

---

## 4. Prefill 阶段

### 4.1 Prefill 定义

**Prefill（Prompt Encoding）** 指模型首次处理完整输入序列的阶段。

输入：

$$  
X=[x_{1},x_{2},...,x_{L}]  
$$

其中 $L$ 为 Prompt token 数。

模型一次 forward：
- 计算所有 token 的隐藏状态；
- 生成 Key Cache 和 Value Cache；
- 为后续 Decode 提供 Attention 历史信息。

### 4.2 Prefill 计算特征

Self-Attention 计算：

$$  
\begin{aligned}  
Attention(Q,K,V)  
&=  
Softmax(\frac{QK^{T}}{\sqrt{d}})V  
\end{aligned}  
$$

其中：
- $Q$ 表示 Query；
- $K$ 表示 Key；
- $V$ 表示 Value；
- $d$ 表示 Head Dimension。

由于 Prompt 内 token 两两交互：

$$  
\begin{aligned}  
\text{Complexity}_{\text{prefill}}  
&=O(L^{2})  
\end{aligned}  
$$

因此 Prefill：
- 计算量大；
- GPU Tensor Core 利用率高；
- 更适合大 batch。

### 4.3 Prefill 调度原则

Prefill Scheduler 负责：
- 决定哪些请求进入 Prefill；
- 控制 Prefill batch 大小；
- 避免阻塞 Decode。

核心原则：

> Prefill 可以等待，但不能长期阻塞 Decode。

常见优化：

| 优化方式                 | 作用                     |
| -------------------- | ---------------------- |
| Prompt Length Bucket | 减少不同长度 Prompt padding  |
| Admission Control    | Decode 压力高时限制新 Prefill |
| Chunked Prefill      | 拆分长 Prompt，避免阻塞        |

---

## 5. Decode 阶段
### 5.1 Decode 定义

Decode 指在已有 KV Cache 的基础上，每次生成一个新的 token。

第 $t$ 步：

输入：

$$  
x_{t}  
$$

读取：

$$  
K_{1:t-1},V_{1:t-1}  
$$

生成：

$$  
x_{t+1}  
$$

### 5.2 Decode 计算特征

Decode 每一步只计算一个 token：
- GEMM batch 较小；
- Tensor Core 利用率降低；
- 大量时间消耗在读取 KV Cache。

Attention：

$$  
\begin{aligned}  
Attention_{decode}  
&=  
Softmax(\frac{q_{t}K_{1:t}^{T}}{\sqrt{d}})V_{1:t}  
\end{aligned}  
$$

其中：
- 当前 Query 数量固定为 1；
- KV 长度随着生成增长。

因此：

$$  
\begin{aligned}  
\text{Complexity}_{\text{decode}}  
&=O(T)  
\end{aligned}  
$$

其中 $T$ 表示当前序列历史 token 总长度。

### 5.3 为什么 Prefill 和 Decode 必须拆分？

如果二者混合执行：
- 长 Prompt Prefill 会占用 GPU；
- Decode token 无法及时生成；
- Streaming 延迟增加；
- p99 latency 恶化。

因此：

> Prefill / Decode 分离是 LLM Serving 架构的第一条基础原则。

---

## 6. Batch 调度演进路线

LLM Serving Batch 机制经历三个阶段：

```
Static Batching
        ↓
Dynamic Batching
        ↓
Continuous Batching
```

演进核心：

> 调度粒度从 request-level 降低到 iteration-level。

|阶段|调度粒度|核心机制|主要问题|
|---|---|---|---|
|Static Batching|Request|固定 batch 一次执行|长请求阻塞|
|Dynamic Batching|Request|时间窗口+batch size|生命周期仍绑定|
|Continuous Batching|Iteration|每一步重新调度|需要复杂 KV 管理|

---

## 7. Static Batching

### 7.1 根本问题

Static Batching（静态批处理）将多个请求组成固定 batch，并持续执行直到整个 batch 完成。

执行过程：

```text
Request Queue

    ↓

Batch Assembly

    ↓

Forward Iteration 1

    ↓

Forward Iteration 2

    ↓

...

    ↓

最长请求完成

    ↓

释放 Batch
```

设 batch 中共有 $B$ 个请求，第 $i$ 个请求需要生成 $L_{i}$ 个 token，则 batch 生命周期：

$$  
\begin{aligned}  
T_{\text{batch}}  
&=\max_{i=1}^{B}L_{i}  
\end{aligned}  
$$

当：

$$  
L_{1}\gg L_{2},L_{3},...,L_{B}  
$$

则短请求完成后：
- GPU slot 无法释放；
- KV Cache 持续占用；
- batch 有效利用率下降。

因此 Static Batching 不适合在线 LLM Serving。

---

## 8. Dynamic Batching
### 8.1 定义

Dynamic Batching（动态批处理）在请求进入模型前进行聚合。

与 Static Batching 区别：

|机制|Static Batching|Dynamic Batching|
|---|---|---|
|batch 形成时间|固定|动态|
|等待策略|等待固定数量|时间窗口或数量触发|
|调度粒度|request|request|

### 8.2 Dynamic Batching 调度条件

设请求到达时间为 $t_{i}$，调度器维护等待队列：

$$  
\mathcal{Q}={r_{1},r_{2},...,r_{N}}  
$$

当满足以下任一条件时执行：

$$  
\begin{aligned}  
\text{Launch}  
&=  
(|\mathcal{Q}|\geq B_{\max})  
\lor  
(t_{\text{now}}-t_{\text{oldest}}\geq\Delta T)  
\end{aligned}  
$$

其中：
- $B_{\max}$ 表示最大 batch size；
- $\Delta T$ 表示最大等待时间；
- $t_{\text{oldest}}$ 表示最早请求时间。

### 8.3 限制

Dynamic Batching 只解决：

> 请求什么时候进入 batch。

但是没有解决：

> batch 内请求什么时候退出。

因此执行期间仍然存在：

```
Seq1  ███████████████████████

Seq2  ████████░░░░░░░░░░░░░░░

Seq3  ██████░░░░░░░░░░░░░░░░░
```

其中：
- `█` 表示有效计算；
- `░` 表示等待最长请求。

根本原因：

> batch 生命周期仍由最长序列决定。

---

## 9. Continuous Batching 核心机制
### 9.1 定义

**Continuous Batching（连续批处理）** 又称：
- In-flight Batching；
- Iteration-level Scheduling。

核心思想：

> 将调度粒度从 request 降低到 iteration，每一次 Decode step 后重新决定下一次 batch 内容。

传统：

```
Request
    ↓
Batch
    ↓
Execute until finish
```

Continuous Batching：

```
Iteration 1
 ↓
重新调度
 ↓
Iteration 2
 ↓
重新调度
 ↓
Iteration 3
```

### 9.2 核心调度流程

Decode iteration 执行：

```cpp
void scheduler_step(RequestPool& pool,
                    ActiveBatch& batch)
{
    // 1. 删除完成请求
    for (auto request : batch)
    {
        if (request.finished())
        {
            release_kv_cache(request);
            remove(request);
        }
    }

    // 2. 插入等待请求
    while (!pool.empty())
    {
        auto request = pool.pop();

        if (can_allocate_kv(request))
        {
            batch.push(request);
        }
        else
        {
            break;
        }
    }

    // 3. 执行一次 decode iteration
    model.forward(batch);
}
```

核心变化：

| 操作         | Static Batching | Continuous Batching |
| ---------- | --------------- | ------------------- |
| 请求加入       | batch 开始前       | 任意 iteration        |
| 请求退出       | 整个 batch 完成     | 生成结束立即退出            |
| KV Cache释放 | 延迟              | 立即                  |

### 9.3 Continuous Batching 的收益

**GPU 利用率提升**

假设：
- batch size 为 $B$；
- 平均请求长度为 $\bar{L}$；
- 最大请求长度为 $L_{\max}$。

Static Batching 有效利用率：

$$  
\begin{aligned}  
U_{\text{static}}  
&=  
\frac{B\bar{L}}  
{BL_{\max}}  
\end{aligned}  
$$

Continuous Batching 中：

$$  
\begin{aligned}  
U_{\text{continuous}}  
&\approx1  
\end{aligned}  
$$

因为请求完成后立即补充新请求。

---

## 10. Ragged Batching

### 10.1 为什么需要 Ragged Batching？

Continuous Batching 后：

同一个 batch 中：
- 请求 A 已生成 100 token；
- 请求 B 已生成 500 token；
- 请求 C 已生成 20 token。

因此无法构造规则矩阵：

```
[
 token token token token ...
 token token token
 token token
]
```

传统 Tensor：

$$  
[B,L,H]  
$$

要求固定长度 $L$。

而实际：

$$  
L_{1}\neq L_{2}\neq ...\neq L_{B}  
$$

### 10.2 Ragged Batching

Ragged Batching（非规则 Batch）通过：
- Flatten token；
- 保存序列边界；
- Kernel 内恢复关系。

例如：

原始：

```
Seq1: A B C
Seq2: D E
Seq3: F G H I
```

Flatten：

```
A B C D E F G H I
```

同时保存：

$$  
\begin{aligned}  
\text{cu\_seqlens}[0]&=0  
\\  
\text{cu\_seqlens}[1]&=3  
\\  
\text{cu\_seqlens}[2]&=5  
\\  
\text{cu\_seqlens}[3]&=9  
\end{aligned}  
$$

其中：
- `cu_seqlens` 表示每个 sequence 的起止位置；
- Attention Kernel 根据边界恢复不同序列。

### 10.3 FlashAttention 对 Ragged Batching 的支持

FlashAttention-2 提供：`varlen attention` 接口支持：
- 不同长度序列；
- 无 padding；
- 直接使用 `cu_seqlens`。

相比 padding：

计算量：

传统：

$$  
\begin{aligned}  
F_{\text{pad}}  
&=B\times L_{\max}  
\end{aligned}  
$$

Ragged：

$$  
\begin{aligned}  
F_{\text{ragged}}  
&=\sum_{i=1}^{B}L_{i}  
\end{aligned}  
$$

当长度差异大时，可以减少大量无效计算。

---

## 11. Decode 阶段 Position Batching

### 11.1 问题来源

Continuous Batching 解决：

> 请求动态加入和退出。

但没有完全解决：

> batch 内 Attention 长度差异。

Decode Attention：

$$  
\begin{aligned}  
Attention_{i}  
&=  
Softmax(q_{i}K_{1:L_{i}}^{T})V_{1:L_{i}}  
\end{aligned}  
$$

其中：$L_{i}$ 表示第 $i$ 个请求当前 KV Cache 长度。

若：

$$  
L_{1}\gg L_{2}  
$$

则：
- 长请求读取更多 KV Cache；
- batch 内 kernel 执行时间由最长 Attention 决定。

### 11.2 Position Batching 定义

**Position Batching**：

> 按 Decode generation position 对请求分组，使同一 batch 内请求的 KV Cache 长度接近。

目标：

降低：

$$  
\max(L_{i})-\min(L_{i})  
$$

### 11.3 Step Tolerance

定义：

$$  
\begin{aligned}  
\Delta_{\text{step}}  
&=  
\max(L_{i})-\min(L_{i})  
\end{aligned}  
$$

其中：

- $\Delta_{\text{step}}$ 表示 batch 内允许的位置差异。
    

工程中通常：

|$\Delta_{\text{step}}$|特点|
|---|---|
|0|严格同步，等待增加|
|1|平衡吞吐和延迟|
|2|允许更高填充率|
|较大|退化为普通 Continuous Batching|

不存在公开统一结论证明 $\Delta_{\text{step}}=1$ 在所有模型和负载下最优，需要根据 SLA 和负载调节。

---

## 12. Chunked Prefill

### 12.1 Prefill Stall 问题

Continuous Batching 主要优化 Decode 阶段，但仍存在一个问题：

> 一个长 Prompt 的 Prefill 可能独占 GPU，导致正在 Decode 的请求暂停。

例如：

```text
Decode Request A
    █ █ █ █ █ █ █ █

Long Prefill Request B
            █████████████████████

Decode Request C
    █ █ █ █ █ █ █ █
```

如果 Request B 的 Prompt 长度为数万 token：
- Prefill kernel 执行时间较长；
- Decode token 无法及时生成；
- TTFT 和 TPOT 恶化。

因此需要将 Prefill 从“一次性执行”改为“分段执行”。

### 12.2 Chunked Prefill 定义

**Chunked Prefill**：

> 将长 Prompt 切分为多个固定大小 chunk，每次 iteration 只处理部分 Prompt，使 Prefill 与 Decode 在时间维度交错执行。

设：
- Prompt 长度为 $N$；
- Chunk size 为 $C$。

需要执行的 chunk 数：

$$  
\begin{aligned}  
K  
&=  
\left\lceil  
\frac{N}{C}  
\right\rceil  
\end{aligned}  
$$

第 $k$ 个 chunk 处理 token 范围：

$$  
[(k-1)C,\min(kC,N))  
$$

其中：
- $k\in[1,K]$；
- $C$ 控制单次 Prefill 计算规模。

### 12.3 Chunked Prefill 执行流程

普通 Prefill：

```text
Prompt
[1 ---------------- N]

一次 Forward

生成 KV Cache
```

Chunked Prefill：

```text
Chunk 1
[1 ----- C]

Forward
↓
KV Cache

Chunk 2
[C+1 --- 2C]

Forward
↓
KV Cache

...

Chunk K
```

每个 iteration：
1. Scheduler 检查 Decode 请求；
2. 优先执行 Decode；
3. 剩余 GPU 预算执行 Prefill chunk；
4. 更新 KV Cache。

### 12.4 Chunked Prefill 的优化目标

Chunk size 存在 Trade-off：

|Chunk Size|优点|缺点|
|---|---|---|
|大|Prefill Throughput 高|阻塞 Decode|
|小|Decode latency 低|Prefill kernel 利用率下降|

因此目标：

$$  
\begin{aligned}  
\min  
\quad  
&  
\text{TTFT}  
+  
\lambda\text{TPOT}  
\end{aligned}  
$$

其中：$\lambda$ 表示系统对 Decode 延迟的权重。

---

## 13. KV Cache 显存管理

### 13.1 KV Cache 为什么成为核心瓶颈？

LLM Serving 中：
- 权重（Weights）固定；
- Activation 生命周期短；
- KV Cache 随请求数量和长度动态增长。

因此：

> Continuous Batching 的最大 batch size 通常由 KV Cache 显存决定。

### 13.2 单请求 KV Cache 大小

对于 Transformer：

设：
- $L_{seq}$：当前序列长度；
- $N_{layer}$：Transformer 层数；
- $N_{head}$：Attention Head 数量；
- $d_{head}$：单个 Head 维度；
- FP16 数据类型大小为 $2$ bytes。

KV Cache：

$$  
\begin{aligned}  
M_{kv}  
&=  
2  
\times  
L_{seq}  
\times  
N_{layer}  
\times  
N_{head}  
\times  
d_{head}  
\times  
2  
\text{ bytes}  
\end{aligned}  
$$

其中：
- 第一个 $2$ 表示：Key 和 Value。
- 第二个 $2$ 表示：FP16 byte size。

### 13.3 Batch KV Cache 总量

对于 batch：

$$  
\begin{aligned}  
M_{kv,total}  
&=  
\sum_{i=1}^{B}M_{kv}^{(i)}  
\end{aligned}  
$$

必须满足：

$$  
\begin{aligned}  
M_{kv,total}  
&\leq  
M_{GPU}
-
M_{weights}
-
M_{activation}  
\end{aligned}  
$$

其中：
- $M_{GPU}$：GPU 总显存；
- $M_{weights}$：模型权重占用；
- $M_{activation}$：运行时激活占用。

---

## 14. PagedAttention

### 14.1 传统 KV Cache 分配问题

传统实现：

每个请求预先分配连续 KV Cache：

```text
Request A:
[------------------------]

Request B:
[------------]

Request C:
[----------------]
```

问题：

#### 内部碎片

需要按照最大长度预留：
- 实际生成 500 token；
- 预留 4096 token。

大量空间浪费。

#### 外部碎片

不同请求生命周期不同：

```text
GPU Memory:

AAAA BBBB CCCC
AAAA      CCCC
AAAA DDDD CCCC
```

释放后产生不连续空间。

### 14.2 PagedAttention 核心思想

PagedAttention 借鉴操作系统 Virtual Memory：

将 KV Cache 划分为固定大小 Block，结构：

```text
Logical KV Cache

Request A:
Block0 → Block3 → Block8

Block Table

Logical Block
      |
      ↓
Physical Block Pool

GPU Memory:
B0 B1 B2 B3 B4 B5 ...
```

每个请求维护：
- Block Table；
- 逻辑 Block 到物理 Block 映射。

### 14.3 PagedAttention 优势

#### 消除预分配浪费

传统：

$$  
M_{alloc}=L_{max}\times size  
$$

PagedAttention：

$$  
M_{alloc}
=
\lceil  
\frac{L_{seq}}{B_{block}}  
\rceil  
\times size  
$$

其中：$B_{block}$ 为 block token 数。

#### 支持动态请求生命周期

请求结束，立即释放 Block：

```text
Before:

A A A B B B C C C

Request B finished

After:

A A A _ _ _ C C C
```

释放 Block 可立即分配给新请求。

---

## 15. Memory-aware Dynamic Batching

### 15.1 静态 Batch Size 的问题

传统 Serving：

固定：

```text
max_batch_size = 128
```

但是实际：
- 请求长度变化；
- KV Cache 占用变化；
- GPU 空闲程度变化。

固定 batch 无法适应动态负载。

### 15.2 Memory-aware Scheduling

核心思想：

> 将 batch size 从静态超参数变成实时控制变量。

优化目标：

$$  
\begin{aligned}  
\max_{B(t)}  
\quad  
&  
\text{Throughput}(B(t))  
\\  
\text{s.t.}  
\quad  
& 
M_{kv}(B(t))  
\leq  
M_{available}(t)  
\\  
&  
Latency(B(t))  
\leq  
SLA  
\end{aligned}  
$$

其中：
- $B(t)$：时间 $t$ 时刻 batch size；
- $M_{available}(t)$：当前可用显存。

### 15.3 调度策略

Scheduler 持续监控：
- KV Cache 使用量；
- 活跃请求数量；
- Decode latency；
- GPU 利用率。

动态调整：

```text
GPU Memory Low
        ↓
减少新请求

GPU Memory Available
        ↓
增加 batch
```

---

## 16. 主流框架实现

|框架|Continuous Batching 实现|核心特点|
|---|---|---|
|vLLM|Continuous Batching + PagedAttention|Paged KV 管理，Iteration Scheduler|
|TensorRT-LLM|In-flight Batching|NVIDIA 官方推理优化方案|
|HuggingFace TGI|Continuous Batching|基于 ORCA 思想|
|LMDeploy|Persistent Batching|TurboMind 推理优化|
|SGLang|RadixAttention + Continuous Batching|Prefix Cache 优化|

---

## 17. 完整演进路线

```text
Static Batching

问题：
固定 batch 生命周期
最长请求拖慢整体

        ↓

Dynamic Batching

优化：
时间窗口 + batch size

问题：
batch 生命周期仍绑定

        ↓

Continuous Batching

优化：
Iteration-level Scheduling

解决：
请求动态加入/退出

        ↓

Ragged Batching

优化：
消除 padding

        ↓

Chunked Prefill

优化：
避免 Prefill Stall

        ↓

PagedAttention

优化：
KV Cache 动态管理

        ↓

Memory-aware Scheduling

优化：
实时控制 batch size
```

---

## 18. 工程调优参数

以 vLLM 类系统为例：

|参数|作用|影响|
|---|---|---|
|`max_num_seqs`|最大并发序列数|影响 Decode Throughput|
|`max_num_batched_tokens`|单 iteration 最大 token 数|影响 Prefill/Decode 平衡|
|`gpu_memory_utilization`|KV Cache 显存比例|影响并发容量|
|`max_model_len`|最大上下文长度|影响 KV Cache 消耗|
|`chunked_prefill_size`|Prefill chunk 大小|影响 TTFT/TPOT Trade-off|

---

## 19. 性能分析

Continuous Batching 的收益：提升 GPU 利用率、降低长尾延迟。

### 19.1 提升 GPU 利用率

Static Batching：

$$  
\begin{aligned}  
U  
&=  
\frac{\sum_iL_i}  
{B\times max(L_i)}  
\end{aligned}  
$$

长度差异越大：

- 空闲 slot 越多；
- 利用率越低。

Continuous Batching：

通过动态补充：

$$  
U\rightarrow1  
$$

### 19.2 降低长尾延迟

p99 latency 的主要来源：

- 长请求；
- Prefill 阻塞；
- KV Cache 分配等待。

解决方式：

|问题|机制|
|---|---|
|长请求拖慢 batch|Continuous Batching|
|长度差异|Position Batching|
|Prefill 阻塞|Chunked Prefill|
|显存碎片|PagedAttention|

---

## 20. 工程铁律

1. **Prefill 与 Decode 必须逻辑分离。**
2. **Decode 优先级高于 Prefill。**
3. **Continuous Batching 是在线 Serving 的核心调度机制。**
4. **Ragged Batching 用于解决动态 batch 的长度不规则问题。**
5. **Chunked Prefill 用于解决长 Prompt 阻塞 Decode。**
6. **PagedAttention 是大规模并发 Serving 的显存管理基础。**
7. **优化目标不是最大 batch，而是在 SLA 约束下最大化吞吐。**

最终：

> LLM Serving 的核心竞争力不是单次 forward 更快，而是在动态请求环境中，通过调度、显存管理和计算组织，使 GPU 长时间保持高利用率，同时保证尾延迟稳定。
