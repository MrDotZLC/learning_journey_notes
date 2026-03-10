## 1. 动态 Shape

**动态 Shape（Dynamic Shape）** 指模型推理时，输入张量的一个或多个维度在编译期（compile time）未知，仅在运行期（runtime）确定。

|特性|Static Shape|Dynamic Shape|
|---|---|---|
|维度确定时机|编译期|运行期|
|典型场景|batch=1, seq=512 固定|变长序列、动态 batch|
|编译器优化空间|极大（常量折叠、循环展开）|受限（需保守假设或运行时分支）|
|代表引擎/模式|TensorRT static mode|TensorRT dynamic mode、TVM、XLA|

---

## 2. 为何在 LLM 推理中普遍存在

LLM 推理分为两阶段：

- **Prefill**：一次性处理完整 prompt，token 数 $S_{\text{in}}$ 由用户输入决定，每次请求各异
- **Decode**：自回归逐 token 生成，序列长度 $S_t$ 随步骤 $t$ 单调递增

设标准输入张量形状为：

$$ X \in \mathbb{R}^{B \times S \times H} $$

其中：

- $B$：batch size，在线服务中随并发请求数变化
- $S$：sequence length，Prefill 由 prompt 长度决定，Decode 阶段逐步增长
- $H$：hidden dimension，由模型结构静态确定

$B$ 与 $S$ 均为运行期才能确定的动态维度。多维度同时动态时，问题复杂度显著上升——编译器必须对所有维度组合保持正确性，而非仅对单一维度做自适应。

---

## 3. 对推理引擎的核心挑战

### 3.1 内存分配策略

静态 shape 下，所有 buffer 可在引擎初始化时一次性分配。动态 shape 下有三种策略：

|策略|机制|优点|缺点|
|---|---|---|---|
|按上界预分配|按 $S_{\max}$ 分配全量 buffer|零运行时分配开销|内存利用率低|
|运行时按需分配|实际 shape 确定后调用 `cudaMalloc`|内存利用率高|`cudaMalloc` 引入延迟，影响吞吐|
|内存池（Memory Pool）|预建固定大小 pool，按需取用|兼顾效率与延迟|需设计 pool 管理逻辑|

TensorRT 的 `IRuntime` 默认采用内存池策略；`cudaMallocAsync` + CUDA stream-ordered allocator 是当前主流的异步内存管理方案。

### 3.2 Kernel 选择与编译

高性能 CUDA Kernel（如 FlashAttention）针对特定 shape 做了 tiling 参数调优。最优 tile size 由以下因素共同决定：

$$ \text{tile\_size}^* = \underset{t}{\arg\max} ; \text{Utilization}(t,\ S,\ \text{SRAM},\ N_{\text{warp}}) $$

$S$ 未知时，编译器无法静态选取最优 $t$，需在运行时根据实际 $S$ 查询 **kernel dispatch table**，将请求 dispatch 到对应预编译的 kernel 实例。

> `[占位图：kernel dispatch table 示意图]` 横轴为 sequence length 范围分段，纵轴为对应 tile_size 和 kernel 实例，箭头表示运行时 dispatch 路径。

### 3.3 计算图优化受限

以三算子融合为例：

```
LayerNorm → Linear → GeLU
```

融合 kernel 需在编译期确定 loop bound（即 $S$）。$S$ 动态时，编译器面临两种妥协：

- **生成含运行时 branch 的通用 kernel**：保留 `if (s < threshold)` 等条件分支，寄存器压力上升，性能劣于静态融合版本
- **推迟融合至运行时 JIT**：首次执行触发编译，引入 warm-up 延迟（对在线服务尤为不利）

---

## 4. 主流解决方案

### 4.1 Shape Bucketing（形状分桶）

将动态维度离散化为有限个预设值（bucket），对每个 bucket 单独编译静态图。

$$ \mathcal{B} = {128,\ 256,\ 512,\ 1024,\ 2048} $$

推理时将实际 $S$ 向上映射至最近 bucket，不足部分用 padding token 填充：

$$ S_{\text{bucket}} = \min{b \in \mathcal{B} \mid b \geq S} $$

padding 引入的无效计算比例：

$$ \text{waste} = \frac{S_{\text{bucket}} - S}{S_{\text{bucket}}} $$

**Bucket 划分策略**：bucket 间距应与实际请求长度分布匹配。若请求集中在短序列，可采用非均匀分桶：${32, 64, 128, 256, 512, 2048}$，在短序列区间加密以降低 waste。

### 4.2 TensorRT Dynamic Shape Mode

TensorRT 通过 `IOptimizationProfile` 声明 shape 的三元组 $(\text{kMIN},\ \text{kOPT},\ \text{kMAX})$，引擎针对 `kOPT` 做深度优化，在 $[\text{kMIN},\ \text{kMAX}]$ 范围内保持正确性：

```cpp
IOptimizationProfile* profile = builder->createOptimizationProfile();

// 输入: [batch, seq_len, hidden=768]
profile->setDimensions("input", OptProfileSelector::kMIN, Dims3{1,    1, 768});
profile->setDimensions("input", OptProfileSelector::kOPT, Dims3{8,  512, 768});
profile->setDimensions("input", OptProfileSelector::kMAX, Dims3{16, 2048, 768});

config->addOptimizationProfile(profile);
```

运行时通过 `setInputShape` 通知实际 shape，引擎按实际维度执行：

```cpp
// 实际输入 shape: [4, 256, 768]
context->setInputShape("input", Dims3{4, 256, 768});
context->executeV2(bindings);
```

**多 Profile 策略**：可注册多个 `IOptimizationProfile`，分别针对短序列（`kOPT=128`）和长序列（`kOPT=1024`）优化，运行时按实际 $S$ 选择最匹配的 profile，相当于在 TRT 框架内实现分桶。

### 4.3 PagedAttention（KV Cache 动态管理）

vLLM 提出的 PagedAttention 将 KV Cache 切分为固定大小的 **page**（默认 16 tokens/page），通过 block table 间接寻址，彻底解耦序列长度与物理内存分配：

$$ N_{\text{pages}}(S) = \left\lceil \frac{S}{P} \right\rceil, \quad P = 16\ \text{(tokens/page)} $$

> `[占位图：PagedAttention 内存布局示意图]` 左侧为逻辑 KV Cache（连续序列视图），右侧为物理 page 池（离散分配），中间为 block table 映射关系。

物理内存按 page 粒度按需分配，无需预知 $S_{\max}$，从根本上消除静态预分配的 padding 浪费。

### 4.4 Symbolic Shape 编译（TVM Relax / torch.compile）

TVM Relax IR 和 `torch.compile`（基于 Triton 后端）支持将动态维度表示为符号变量，编译期做符号推导，运行时代入实际值触发 JIT，已编译的 kernel 按 shape 缓存复用：

```python
# torch.compile dynamic=True：追踪符号维度，自动缓存 kernel
@torch.compile(dynamic=True)
def forward(x: torch.Tensor) -> torch.Tensor:
    # x.shape = [B, S, H]，B 和 S 为符号维度
    return model(x)
```

首次执行某 shape 时触发编译并缓存，后续相同 shape 直接复用，warm-up 延迟集中在首次请求。

---

## 5. 各方案横向对比

|方案|编译开销|运行时开销|内存效率|适用场景|
|---|---|---|---|---|
|Static Shape|一次编译|零|低（padding 浪费）|固定 shape 离线推理|
|Shape Bucketing|$\|\mathcal{B}\|$ 次编译|极低（查表）|中（bucket 间距决定）|离线批处理、受控在线服务|
|TRT Dynamic Mode|一次编译（per profile）|低|高|在线服务主流方案|
|PagedAttention|无（运行时逻辑）|中（block table 间接寻址）|极高|LLM Decode KV Cache 管理|
|Symbolic JIT|首次运行触发|低（缓存命中后）|高|研究环境、灵活部署场景|

---

## 6. 与 KV Cache 的交叉影响

Decode 阶段每步生成 1 个 token，KV Cache 形状为：

$$ K,\ V \in \mathbb{R}^{B \times n_{\text{heads}} \times S_t \times d_{\text{head}}} $$

$S_t$ 随时间步 $t$ 线性增长，是 LLM 推理中最典型的动态 shape 场景。静态预分配策略下，需在请求开始时按 $S_{\max}$ 预留全量 KV Cache：

$$ M_{\text{KV}} = 2 \cdot B \cdot L \cdot n_{\text{heads}} \cdot S_{\max} \cdot d_{\text{head}} \cdot \text{sizeof(dtype)} $$

其中 $L$ 为模型层数，系数 $2$ 对应 $K$ 与 $V$。

以 LLaMA-3 8B（$L=32,\ n_{\text{heads}}=32,\ d_{\text{head}}=128$，FP16）、$B=16$、$S_{\max}=4096$ 为例：

$$ M_{\text{KV}} = 2 \times 16 \times 32 \times 32 \times 4096 \times 128 \times 2\ \text{bytes} \approx 68.7\ \text{GB} $$

单卡 A100（80 GB）在此配置下几乎无剩余显存用于模型权重，静态分配的不可行性由此直观体现。PagedAttention 将 $S_{\max}$ 替换为实际 $S_t$ 的动态增长，使显存占用从预分配上界降至实际使用量，是解决该问题的核心手段。

---

## 7. 面试高频问题

### 7.1 概念理解类

**Q1：Static Shape 和 Dynamic Shape 的本质区别是什么？为什么 Dynamic Shape 会导致性能下降？**

核心区别在于维度信息的可见时机。Static Shape 下，编译器掌握所有维度的精确值，可执行常量折叠、循环展开、最优 tiling 选取等深度优化；Dynamic Shape 下，编译器只知维度的约束范围（如 $S \in [1, 2048]$），必须生成对任意合法值均正确的通用代码路径，导致：

- 无法静态选取最优 kernel tiling 参数
- 融合 kernel 需插入运行时 branch
- 部分算子融合机会丧失

**Q2：LLM 推理中动态 Shape 主要来源于哪些维度？Prefill 和 Decode 阶段各有何特点？**

主要来源为 batch size $B$ 和 sequence length $S$：

|阶段|动态维度|变化特征|
|---|---|---|
|Prefill|$S_{\text{in}}$（prompt 长度）|请求间随机变化，同一请求内固定|
|Decode|$S_t$（已生成序列长度）|同一请求内随步骤 $t$ 单调递增|
|在线服务|$B$（并发请求数）|随到达/完成事件动态变化|

**Q3：Shape Bucketing 的 waste 如何量化？bucket 划分策略的依据是什么？**

单次请求的 padding waste：

$$ \text{waste}(S) = \frac{S_{\text{bucket}}(S) - S}{S_{\text{bucket}}(S)}, \quad S_{\text{bucket}}(S) = \min{b \in \mathcal{B} \mid b \geq S} $$

在请求长度分布 $p(S)$ 下，期望 waste：

$$ \mathbb{E}[\text{waste}] = \int \frac{S_{\text{bucket}}(S) - S}{S_{\text{bucket}}(S)} \cdot p(S), dS $$

划分依据：对历史请求长度做统计，在高密度区间加密 bucket（减小间距），低密度区间稀疏 bucket（减少编译数量），在编译成本与 waste 之间取平衡。

---

### 7.2 工程实现类

**Q4：TensorRT 中如何正确配置动态 Shape？kMIN/kOPT/kMAX 各自的作用是什么？**

三元组含义：

|参数|含义|对引擎的影响|
|---|---|---|
|`kMIN`|允许的最小 shape|定义合法输入下界，低于此值行为未定义|
|`kOPT`|最优化目标 shape|引擎针对此 shape 做深度 kernel 选取和内存规划|
|`kMAX`|允许的最大 shape|定义合法输入上界，buffer 按此分配|

`kOPT` 应设为线上最高频的 shape，偏离 `kOPT` 越远性能下降越明显。多 profile 策略可针对不同 shape 区间分别设置 `kOPT`。

**Q5：`cudaMalloc` 和内存池在动态 Shape 场景下的性能差异来自哪里？**

`cudaMalloc` 是同步调用，需等待 GPU 驱动完成物理页分配后才返回，单次调用延迟可达数十至数百微秒。动态 shape 下每次推理可能触发多次 `cudaMalloc`，累积延迟不可忽视。

内存池方案（如 `cudaMallocAsync` + stream-ordered allocator）将分配操作插入 CUDA stream，与 kernel 执行异步重叠，消除同步等待；pool 内的分配退化为指针偏移操作，延迟接近零。

**Q6：PagedAttention 如何解决 KV Cache 的动态 Shape 问题？与静态预分配相比优势在哪里？**

PagedAttention 将 KV Cache 的逻辑连续视图映射到离散物理 page：

$$ \text{block\_table}[i] \rightarrow \text{physical\_page}[j], \quad j \in {0,\ldots,N_{\text{pool}}-1} $$

每次 Decode 步骤仅在序列跨越 page 边界时分配新 page（$S_t \mod P = 0$），其余步骤零分配开销。

对比静态预分配：

|维度|静态预分配|PagedAttention|
|---|---|---|
|显存占用|$\propto B \cdot S_{\max}$|$\propto B \cdot S_{\text{actual}}$|
|分配时机|请求到达时一次性|按 page 粒度增量分配|
|碎片化|内部碎片严重|仅最后一个 page 存在内部碎片|
|支持请求抢占|困难（内存已绑定）|天然支持（page 可回收重分配）|

---

### 7.3 系统设计类

**Q7：在线推理服务中，如何在延迟（latency）和吞吐（throughput）之间权衡动态 Shape 策略的选取？**

核心矛盾：延迟敏感场景希望零编译开销（倾向 Dynamic Mode）；吞吐敏感场景可接受 padding 代价以换取静态 kernel 的极致性能（倾向 Bucketing）。

决策框架：

```
if SLA_latency < 50ms:
    优先 TRT Dynamic Mode + 多 Profile
    辅以 PagedAttention 管理 KV Cache
else:
    Shape Bucketing（按请求长度分布划分 bucket）
    bucket 数量 = 编译时间预算 / 单次编译时间
```

实践中两者常结合：用 Bucketing 处理 Prefill（序列长度在请求开始时已知），用 Dynamic Mode + PagedAttention 处理 Decode（序列长度实时增长）。

**Q8：多个请求并发时，动态 Shape 如何影响 Continuous Batching 的实现？**

Continuous Batching 要求在同一 batch 中合并处于不同 Decode 步骤（即不同 $S_t$）的请求。各请求 $S_t$ 各异，batch 内无法用单一静态 shape 表示。

解决路径：

- **Padded Batching**：对 batch 内所有请求 pad 至 $S_{\max}^{\text{batch}}$，引入大量无效计算
- **Ragged Tensor**（变长张量）：将 batch 维度展平为 token 列表，以总 token 数 $T = \sum_{i} S_t^{(i)}$ 为单一动态维度，消除 padding；需要专门的 kernel 支持（如 FasterTransformer 的 `remove_padding` 机制）
- **PagedAttention + token-level scheduling**：vLLM 的做法，每个 Decode 步骤将所有活跃请求的当前 token 聚合为一个 $[T, H]$ 的矩阵统一处理，$T$ 为动态维度
