## 1. 背景

### 1.1 MoE 与 Expert Parallelism

Expert Parallelism（EP）是 **Mixture-of-Experts（MoE）模型中的一种分布式并行策略，通过沿 Expert 维度切分模型参数，使超大规模 MoE 模型能够跨多 GPU 存储和计算。**

传统 Dense Transformer 使用固定 FFN：

$$  
Y=\sigma(XW_{1})W_{2}  
$$

其中：

- $X\in R^{T\times d}$：输入 token hidden states；
- $T$：token 数量；
- $d$：hidden dimension；
- $W_{1},W_{2}$：FFN 权重。

FFN 参数量：

$$  
Params_{FFN}\approx 2Ld d_{ff}  
$$

其中：

- $L$：Transformer Layer 数量；
- $d_{ff}$：FFN intermediate dimension。

随着模型规模增加：

$$  
d,d_{ff},L\uparrow  
$$

导致：

- 参数量增加；
- 显存需求增加；
- 推理成本增加。

---

### 1.2 MoE 引入 Expert

MoE 将 Dense FFN 替换为多个 Expert：

$$  
FFN(x)  
\rightarrow  
{E_{1}(x),E_{2}(x),...,E_{N_E}(x)}  
$$

其中：

- $N_E$：Expert 数量；
- $E_i$：第 $i$ 个 Expert。

Router 根据 token 动态选择 Expert：

$$  
G=XW_r  
$$

$$  
P_i=  
Softmax(G_i)  
$$

Top-K Router：

$$  
I=TopK(P)  
$$

最终输出：

$$  
Y=  
\sum_{i\in I}  
P_iE_i(X)  
$$

当：

- Expert 数量 $N_E$ 很大；
- 每个 token 激活 $K$ 个 Expert；

则：

参数规模：

$$  
Params\propto N_E  
$$

计算规模：

$$  
Compute\propto K  
$$

实现：

> **模型参数容量增长与单 token 计算量解耦。**

---

## 2. EP 问题来源

### 2.1 Expert 参数无法单卡存储

假设：

- Expert 数量：$N_E$
- 单 Expert 参数量：$P_E$

总参数：

$$  
P_{total}=N_EP_E  
$$

单 GPU 显存：

$$  
Memory_{GPU}=N_EP_E  
$$

当：

$$  
N_EP_E>Memory_{HBM}  
$$

模型无法部署。

EP 将 Expert 分布到多个 GPU：

当 EP degree 为 $P_E$ 时：

$$  
Memory_{GPU}
=
\frac{N_EP_E}{P_E}  
$$

每张 GPU 保存：

$$  
\frac{1}{P_E}  
$$

的 Expert 参数。

### 2.2 Expert 计算负载不均衡

Router 输出：

$$  
p_i=P(E_i|x)  
$$

理想情况：

$$  
tokens_i
=
\frac{T\times K}{N_E}  
$$

实际：

热门 Expert：

$$  
tokens_i  
\gg  
\frac{T\times K}{N_E}  
$$

导致：

- 部分 GPU 计算饱和；
- 部分 GPU 空闲；
- 通信等待增加。

该问题称：

**Expert Load Imbalance。**

### 2.3 Token Dispatch 通信

由于 Expert 分布在不同 GPU，token 需要发送到对应 Expert：

例如：

```text
GPU0:

Token0 → Expert3
Token1 → Expert1


Expert3 位于 GPU3
Expert1 位于 GPU1
```

因此：

```text
GPU0
 |
 | All-to-All
 |
GPU1/GPU3
```

EP 的核心通信：

$$  
All\text{-}to\text{-}All  
$$

区别：

|并行方式|主要通信|
|---|---|
|Tensor Parallel|AllReduce|
|Pipeline Parallel|Send/Recv|
|Expert Parallel|All-to-All|

---

## 3. EP 核心思想

### 3.1 Expert 参数切分

EP 将 Expert 沿 GPU 维度划分。

当：

- Expert 数量 $N_E$；
- EP degree $P_E$；

每 GPU Expert 数：

$$  
N_{local}

\frac{N_E}{P_E}  
$$

例如：

|参数|数值|
|---|---|
|Expert 数量|64|
|EP degree|8|
|每 GPU Expert|8|

## 3.2 Token Routing

完整流程：

```text
Input Tokens

      |
      v

 Router

      |
      v

 Expert Assignment

      |
      v

 Token Permutation

      |
      v

 All-to-All Dispatch

      |
      v

 Local Expert Compute

      |
      v

 All-to-All Combine

      |
      v

 Restore Token Order
```

---

## 4. 数学原理

### 4.1 Router

输入：

$$  
X\in R^{T\times d}  
$$

Router：

$$  
G=XW_r  
$$

Softmax：

$$  
P_i
=
\frac{e^{G_i}}  
{\sum_j e^{G_j}}  
$$

选择：

$$  
I=TopK(P)  
$$

输出：

$$  
Y=  
\sum_{i\in I}  
P_iE_i(X)  
$$

#### 4.2 Capacity Factor

为了限制 Expert 最大 token 数，引入 Capacity：

$$  
C=  
\frac{T\times K}{N_E}  
\times CF  
$$

其中：

- $T$：token 数；
- $K$：激活 Expert 数；
- $N_E$：Expert 总数；
- $CF$：capacity factor。

例如：

当：

- $T=8192$
- $K=2$
- $N_E=64$
- $CF=1.25$

则：

$$  
C=  
\frac{8192\times2}{64}  
\times1.25

320  
$$

每个 Expert 最多处理：

$$  
320  
$$

个 token。

超过容量：

- drop token；
- reroute；
- padding。

---

## 5. EP 算法流程

### 5.1 Token Dispatch

#### Step 1：Router 计算

得到：

```text
Token → Expert ID
```

例如：

```text
T0 → Expert3

T1 → Expert1

T2 → Expert3
```

#### Step 2：Token Permutation

按照 Expert 排序：

原始：

```text
T0(E3)

T1(E1)

T2(E3)
```

转换：

```text
Expert1:

T1


Expert3:

T0,T2
```

目的：

让相同 Expert token 连续，提高 GEMM 效率。

#### Step 3：All-to-All Dispatch

将 token 发送到 Expert 所在 GPU。

通信量近似：

$$  
V_{comm}  
\approx  
2Td\times sizeof(dtype)  
$$

其中：

- 第一次 All-to-All：Dispatch；
- 第二次 All-to-All：Combine。

#### Step 4：Expert Computation

每个 Expert 执行 FFN：

$$  
Y_i=X_iW_i  
$$

由于不同 Expert token 数不同：

例如：

$$  
X_1\in R^{512\times d}  
$$

$$  
X_2\in R^{64\times d}  
$$

产生大量不规则 GEMM。

因此需要：

**Grouped GEMM。**

---

## 6. Grouped GEMM

### 6.1 问题来源

普通 GEMM：

```text
Expert0:

GEMM(X0,W0)


Expert1:

GEMM(X1,W1)


Expert2:

GEMM(X2,W2)
```

问题：

- 多次 Kernel Launch；
- 小矩阵效率低；
- Tensor Core 利用率下降。

### 6.2 核心思想

Grouped GEMM 将多个 Expert GEMM 合并：

$$  
{X_1W_1,X_2W_2,...,X_nW_n}  
$$

通过一个 Kernel 调度：

```text
Grouped GEMM Kernel

 ├── Expert0 GEMM
 ├── Expert1 GEMM
 ├── Expert2 GEMM
 └── ...
```

### 6.3 性能收益

普通 GEMM：

$$  
T=  
\sum_iT_{launch}^{i}  
+  
\sum_iT_{compute}^{i}  
$$

Grouped GEMM：

$$  
T=  
T_{launch}  
+  
\sum_iT_{compute}^{i}  
$$

减少：

- Kernel Launch overhead；
- SM 空闲；
- 小矩阵计算损失。

---

## 7. EP 工程实现

### 7.1 EP 与其他并行组合

大模型通常采用：

$$  
DP+TP+PP+EP  
$$

|并行方式|切分对象|
|---|---|
|Data Parallel|Batch|
|Tensor Parallel|矩阵维度|
|Pipeline Parallel|Layer|
|Expert Parallel|Expert|

典型结构：

```text
GPU Cluster

      DP

      |

 TP × PP × EP
```

### 7.2 DeepSpeed-MoE

Microsoft DeepSpeed

核心组件：

- Router；
- Token Dispatcher；
- All-to-All；
- Expert GEMM。

### 7.3 Megatron-LM EP

NVIDIA Megatron-LM

支持：

$$  
TP+PP+EP  
$$

例如：

$$  
TP=8  
$$

$$  
PP=4  
$$

$$  
EP=8  
$$

总 GPU：

$$  
8\times4\times8=256  
$$

---

## 8. 性能分析

### 8.1 计算复杂度

Dense FFN：

$$  
FLOPs_{Dense}  
\approx  
2Tdd_{ff}  
$$

MoE：

$$  
FLOPs_{MoE}  
\approx  
2TKdd_{ff}  
$$

比例：

$$  
\frac{FLOPs_{MoE}}  
{FLOPs_{Dense}}  
\approx  
\frac{K}{N_E}  
$$

当：

$$  
K\ll N_E  
$$

MoE 获得更高参数容量。

### 8.2 通信瓶颈

EP 通信：

$$  
V_{EP}  
\approx  
2Td\times sizeof(dtype)  
$$

Prefill：

- token 数量大；
- GEMM 计算充分；

通信容易隐藏。

Decode：

- batch 小；
- token 少；

计算降低：

$$  
T_{compute}\downarrow  
$$

通信比例升高：

$$  
\frac{T_{comm}}  
{T_{total}}\uparrow  
$$

因此 EP Decode 优化难度更高。

---

## 9. 优缺点

|优点|说明|
|---|---|
|突破显存限制|Expert 参数跨 GPU 存储|
|扩大模型容量|支持千亿、万亿参数|
|降低激活计算|只执行 Top-K Expert|

|缺点|说明|
|---|---|
|All-to-All 通信压力|跨 GPU token 交换|
|负载不均衡|Router 热点|
|系统复杂|需要 Dispatcher|
|Decode 延迟高|通信占比增加|

---

## 10. 应用场景

### 10.1 MoE 大模型训练

代表：

|模型|特点|
|---|---|
|Switch Transformer|稀疏 Expert|
|Mixtral|Sparse MoE|
|DeepSeek MoE 系列|大规模 Expert Parallel|

### 10.2 推理部署

典型：

```text
Prefill:

EP 较适合

Decode:

EP 通信压力较大
```

优化方向：

- Expert Placement；
- Expert Cache；
- Dynamic Batching；
- Communication Compute Overlap。

---

## 11. 发展趋势

### 11.1 Fine-grained MoE

趋势：

增加 Expert 数：

$$  
N_E\uparrow  
$$

减少 Expert 单体规模：

$$  
ExpertSize\downarrow  
$$

优势：

- 更精细能力分工；
- 提升参数容量。

问题：

- All-to-All 通信增加。

---

### 11.2 EP 与硬件协同

优化方向：

|方向|作用|
|---|---|
|NVLink/NVSwitch|降低 GPU 内通信|
|GPUDirect RDMA|优化跨节点通信|
|Grouped GEMM Kernel|提升 Expert 计算|
|动态路由|降低负载不均衡|

---

## 12. 总结

Expert Parallelism 的本质：

> **EP 将 MoE 模型中的 Expert 参数分布到多个 GPU，通过 All-to-All 完成 token 路由，使模型参数规模扩展与单 token 激活计算解耦。**

关键公式：

参数存储：

$$  
Memory_{GPU}

\frac{N_EP_E}{EP}  
$$

MoE 计算：

$$  
FLOPs_{MoE}  
\approx  
2TKdd_{ff}  
$$

EP 通信：

$$  
V_{EP}  
\approx  
2Td\times sizeof(dtype)  
$$

工程核心：

1. Router 决定 token 到 Expert 的映射；
2. All-to-All 完成跨 GPU token Dispatch；
3. Grouped GEMM 提升 Expert 计算效率；
4. Load Balance 决定大规模 EP 扩展效率；
5. EP 通常与 TP、PP、DP 组合形成现代 LLM 分布式推理与训练架构。
