
## 1. 问题背景

序列长度 $N > 128k$ 时，单卡显存无法容纳完整的 Q/K/V 矩阵（FP16，$d=128$，$H=32$，$N=128k$，单矩阵 $= 128k \times 32 \times 128 \times 2 \approx 1\text{ GB}$，三矩阵 $\approx 3\text{ GB}$，还不含激活）。

**Tensor Parallelism（按头切分）** 无法解决此问题，因为每个头仍需访问完整的序列长度。

**Context Parallelism（CP）** 将序列维度切分到多卡：

- 设 $P$ 张卡，每卡负责 $N/P$ 个 Token 的 $Q, K, V$。

## 2. 朴素 CP 的通信问题

每张卡有局部 $Q_i \in \mathbb{R}^{(N/P) \times d}$，但需要访问**全局** $K, V \in \mathbb{R}^{N \times d}$。朴素方案为先 All-Gather $K, V$，再本地计算。

All-Gather 通信量：$2 \times N \times H \times d \times \text{sizeof}$，以 $N=128k$，$P=8$，FP16 为例：

$$2 \times 128k \times 32 \times 128 \times 2 \approx 2\text{ GB}$$

在 $P=8$ 的 NVLink 环境（NVLink 带宽 ~900 GB/s）下通信时间约 $2\text{ ms}$，远大于计算时间——通信成为瓶颈。

## 3. Ring Attention

核心思想：将 All-Gather 与 Attention 计算**流水重叠**，消除通信等待。

$P$ 张卡形成逻辑环，每步：

1. 每卡用本地 $Q_i$ 与当前持有的 $K_j, V_j$ 计算局部 Attention（Online Softmax 累积）
2. 同时，通过 P2P Send/Recv 将 $K_j, V_j$ 传递给下一卡

经过 $P$ 步后，每卡的 $Q_i$ 已与所有 $K, V$ 做完 Attention，合并统计量得到最终输出。

**通信-计算重叠条件：**

每步计算时间：

$$T_{\text{compute}} = \frac{2 \times (N/P)^2 \times H \times d}{P_{\text{FLOPS}}}$$

每步通信时间（P2P，NVLink）：

$$T_{\text{comm}} = \frac{2 \times (N/P) \times H \times d \times \text{sizeof}}{B_{\text{NVLink}}}$$

要完全隐藏通信：$T_{\text{compute}} \geq T_{\text{comm}}$，即：

$$\frac{N/P}{P_{\text{FLOPS}} / (B_{\text{NVLink}} \times \text{sizeof})} \geq 1 \quad \Rightarrow \quad \frac{N}{P} \geq \frac{P_{\text{FLOPS}}}{B_{\text{NVLink}} \times \text{sizeof}}$$

H100（$P_{\text{FLOPS}}^{\text{FP16}} \approx 989\text{ TFLOPS}$，$B_{\text{NVLink}} \approx 900\text{ GB/s}$）：

$$\frac{N}{P} \geq \frac{989 \times 10^{12}}{900 \times 10^9 \times 2} \approx 550k$$

即每卡分配 $\geq 550k$ Token 时通信可被完全隐藏，Ring Attention 对**超长序列**（单卡 $> 64k$）最为有效。

## 4. Causal Mask 下的负载均衡问题

因果掩码下，第 $i$ 个 Token 仅 Attend 前 $i$ 个 Token，序列前部 Token 的计算量远小于后部，朴素 CP 切分导致负载不均。

**解决方案：** 将序列按"锯齿形"分配给各卡（Zigzag 分配），每卡同时持有一段头部 Token 和一段尾部 Token，使各卡的有效计算量近似相等。

## 5. 与 Tensor Parallelism 的组合

实际系统（如 Megatron-LM）同时使用 TP（按头切分）和 CP（按序列切分），形成二维并行：

- TP 组内（同一节点，NVLink 互联）：按 Head 维度切分。
- CP 组跨节点（跨机，InfiniBand 互联）：按序列维度切分。

两者正交，总并行度 $= P_{\text{TP}} \times P_{\text{CP}}$。
