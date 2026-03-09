## 1. 核心直觉与历史背景
### 1.1 起源
- **1991**：Jacobs et al. 提出原始 MoE 框架（用于监督学习的模块化网络）
- **2017**：Shazeer et al. _Outrageously Large Neural Networks_ —— 将 Sparse MoE 引入 LSTM，提出 Top-K Gating + Noisy Gating
- **2022**：Switch Transformer（Google）—— Top-1 路由，简化训练
- **2024**：DeepSeek-V2/V3、Mixtral 8x7B —— MoE 进入主流 LLM
### 1.2 核心动机
**目标**：在**不线性增加计算量**的前提下，扩大模型参数量（容量）。
**基本思路**：
```
输入 x
  │
  ▼
路由器（Router）──→ 选择 k 个专家
  │
  ├──→ Expert_1(x) ──→ 加权求和 ──→ 输出
  ├──→ Expert_3(x) ──┘
  └──→ (其余 N-k 个专家不激活)
```
**稠密模型（Dense）vs 稀疏 MoE**：
|维度|Dense Transformer|Sparse MoE|
|---|---|---|
|参数量|$P$|$N \times P_{\text{expert}}$|
|每 token 激活参数|$P$（全部）|$\approx P/N \times k$（部分）|
|FLOPs/token|高|低（与激活专家数成比例）|
|显存需求|低|高（需加载全部专家权重）|
---
## 2. 数学形式化
### 2.1 基本结构
设有 $N$ 个专家 ${E_1, E_2, \dots, E_N}$，输入 token 表示为 $\mathbf{x} \in \mathbb{R}^d$。
**MoE 层输出**：
$$\text{MoE}(\mathbf{x}) = \sum_{i=1}^{N} G(\mathbf{x})_i \cdot E_i(\mathbf{x})$$
其中 $G(\mathbf{x}) \in \mathbb{R}^N$ 为门控权重向量。
### 2.2 Softmax 门控（稠密，所有专家激活）
$$G(\mathbf{x}) = \text{Softmax}(\mathbf{x} \cdot \mathbf{W}_g)$$
- $\mathbf{W}_g \in \mathbb{R}^{d \times N}$：可学习门控矩阵
- 所有专家均参与计算，无稀疏性
### 2.3 Top-K 稀疏门控（Shazeer et al., 2017）
**步骤一**：计算原始门控分数
$$H(\mathbf{x}) = \mathbf{x} \cdot \mathbf{W}_g \in \mathbb{R}^N$$
**步骤二**：保留 Top-K，其余置 $-\infty$
$$H'(\mathbf{x})_i = \begin{cases} H(\mathbf{x})_i & \text{if } i \in \text{TopK}(H(\mathbf{x}), k) \ -\infty & \text{otherwise} \end{cases}$$
**步骤三**：Softmax 归一化
$$G(\mathbf{x}) = \text{Softmax}(H'(\mathbf{x}))$$
**最终输出**（仅 $k$ 个专家参与计算）：
$$\text{MoE}(\mathbf{x}) = \sum_{i \in \text{TopK}} G(\mathbf{x})_i \cdot E_i(\mathbf{x})$$
### 2.4 Noisy Top-K 门控（训练时探索）
在训练阶段引入噪声，防止路由器坍塌到固定专家：
$$H_{\text{noisy}}(\mathbf{x})_i = H(\mathbf{x})_i + \epsilon_i \cdot \text{Softplus}!\left(\mathbf{x} \cdot \mathbf{W}_{\text{noise}}\right)_i$$
$$\epsilon_i \sim \mathcal{N}(0, 1)$$
- $\mathbf{W}_{\text{noise}} \in \mathbb{R}^{d \times N}$：可学习噪声幅度矩阵
- 推理时关闭噪声
---
## 3. 负载均衡问题（Load Balancing）
### 3.1 问题：专家坍塌（Expert Collapse）
若不加约束，路由器倾向于**始终选同几个专家**（马太效应）：
- 被选专家梯度更新多 → 能力更强 → 被选概率更高
- 未被选专家得不到训练 → 退化为废专家（dead expert）
### 3.2 辅助损失（Auxiliary Loss）
**Switch Transformer（Fedus et al., 2022）** 提出负载均衡损失：
$$\mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot p_i$$
其中：
$$f_i = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}[\text{token } t \text{ 路由到专家 } i]$$
$$p_i = \frac{1}{T} \sum_{t=1}^{T} G(\mathbf{x}_t)_i$$
- $f_i$：专家 $i$ 实际接收的 token 分数（离散，不可微）
- $p_i$：门控对专家 $i$ 的平均软概率（可微，用于梯度传播）
- $\alpha$：辅助损失系数，典型值 $10^{-2}$
- 均匀分布时 $\mathcal{L}_{\text{aux}}$ 最小（$= \alpha$，由均值不等式可证）
**证明均匀时最小**：
由 Cauchy-Schwarz 不等式，$\sum_i f_i p_i \geq \frac{1}{N}$，等号当且仅当 $f_i = p_i = \frac{1}{N}$ 时成立。
### 3.3 Expert Capacity（容量上限）
每个专家设置 token 容量上限，溢出的 token 直接跳过该专家（pass-through）：
$$C = \frac{T}{N} \times \text{capacity_factor}$$
- $T$：batch 内总 token 数
- capacity_factor 典型值：$1.0 \sim 1.5$（训练），$2.0$（推理）
- 超出容量的 token：在 Top-1 路由中直接用残差输出，在 Top-2 中路由到第二专家
---
## 4. 典型架构设计
### 4.1 MoE 在 Transformer 中的位置
标准做法：**替换 FFN 层**（Feed-Forward Network），Attention 层保持稠密不变。
```
Transformer Block:
  ┌─────────────────────────┐
  │  Multi-Head Attention    │  ← 稠密，所有 token 共享
  │  LayerNorm               │
  │  MoE FFN Layer           │  ← 替换原 FFN
  │    ├─ Router             │
  │    ├─ Expert_1 (FFN)     │
  │    ├─ Expert_2 (FFN)     │
  │    └─ ...Expert_N (FFN)  │
  │  LayerNorm               │
  └─────────────────────────┘
```
每个 Expert 本质是独立的 FFN：
$$E_i(\mathbf{x}) = \text{SiLU}(\mathbf{x} \mathbf{W}_{i,1}) \cdot (\mathbf{x} \mathbf{W}_{i,3}) \cdot \mathbf{W}_{i,2}$$
（GLU 变体，与 LLaMA FFN 结构相同）
### 4.2 主流模型参数对比
|模型|总参数|激活参数|专家数 $N$|Top-K|专家间隔层|
|---|---|---|---|---|---|
|Switch Transformer|1.6T|~10B|2048|1|每层|
|Mixtral 8x7B|46.7B|12.9B|8|2|每层|
|DeepSeek-V2|236B|21B|160|6|每层|
|DeepSeek-V3|671B|37B|256|8|每层|
|Qwen2-57B-A14B|57B|14B|64|8|每层|
### 4.3 共享专家（Shared Expert）
DeepSeek-V2 引入：部分专家**始终激活**（不经过路由），其余专家稀疏路由。
$$\text{MoE}(\mathbf{x}) = \sum_{i=1}^{K_s} E_i^{\text{shared}}(\mathbf{x}) + \sum_{j \in \text{TopK}_r} G(\mathbf{x})_j \cdot E_j^{\text{routed}}(\mathbf{x})$$
**作用**：共享专家捕获通用知识，路由专家捕获特化知识，缓解专家间知识冗余。
---
## 5. 路由策略演进
### 5.1 Token-Choice 路由（主流）
每个 token 独立选择专家（如上述 Top-K）。
**问题**：不同 token 可能争抢同一专家，导致容量溢出。
### 5.2 Expert-Choice 路由（Zhou et al., 2022）
反向：每个专家选择它最想处理的 Top-C token。
$$\text{对专家 } i: \text{选择 } \text{Top-}C_i({G(\mathbf{x}_t)_i}_{t=1}^T)$$
**优点**：天然负载均衡，无需辅助损失。  
**缺点**：每个 token 不保证被处理（可能被所有专家放弃），自回归生成场景不适用（无法逐 token 推理）。
### 5.3 辅助无损负载均衡（DeepSeek-V3）
不用辅助损失，改用**偏置项（bias）动态调整路由**：
$$H'(\mathbf{x})_i = H(\mathbf{x})_i + b_i$$
- $b_i$ 根据专家负载动态更新：负载高则 $b_i$ 减小，负载低则 $b_i$ 增大
- 避免辅助损失干扰主任务梯度
---
## 6. 训练挑战与工程实现
### 6.1 通信瓶颈（All-to-All）
多 GPU 训练时，不同 token 被路由到不同设备上的专家，需要 All-to-All 通信：
```
Device 0: [token_1, token_5] → Expert_0, Expert_3（在 Device 0）
           [token_2, token_7] → Expert_5（在 Device 1）─→ 跨设备通信
Device 1: [token_3, token_6] → Expert_2（在 Device 0）─→ 跨设备通信
```
**代价**：All-to-All 通信延迟随专家数 $N$ 和节点数线性增长，成为训练瓶颈。
### 6.2 并行策略
|并行类型|说明|适用场景|
|---|---|---|
|**Expert Parallelism（EP）**|不同设备存放不同专家|专家数多、显存紧张|
|**Tensor Parallelism（TP）**|单个专家权重切分到多设备|单专家参数量大|
|**Data Parallelism（DP）**|不同 batch 在不同设备|常规扩展|
|**Pipeline Parallelism（PP）**|不同层在不同设备|层数多|
实际部署通常组合 EP + DP（或 EP + TP + DP）。
### 6.3 推理优化
```
问题：MoE 推理时需将所有专家权重加载到显存
     Mixtral 8x7B 总参数 46.7B → ~93GB (FP16)
     但每 token 只用 ~12.9B 参数
优化策略：
1. 专家卸载（Expert Offloading）
   - 将不活跃专家权重存在 CPU 内存
   - 预测下一步路由，提前 prefetch 专家权重
   - 延迟代价：PCIe 带宽 ~32 GB/s
2. 专家量化（Expert Quantization）
   - 专家权重 INT4/INT8 量化
   - 路由器权重保持 FP16（参数量小，影响有限）
3. 专家缓存（Expert Caching）
   - LRU 缓存热门专家在 GPU 显存
   - 长文本中专家使用分布存在局部性
```
---
## 7. 完整 MoE 层实现（PyTorch）
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
class ExpertFFN(nn.Module):
    """单个专家：标准 FFN（GLU 变体，与 LLaMA 结构一致）"""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)   # gate projection
        self.w2 = nn.Linear(d_ff, d_model, bias=False)   # down projection
        self.w3 = nn.Linear(d_model, d_ff, bias=False)   # up projection
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(xW1) ⊙ (xW3) * W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
class MoELayer(nn.Module):
    """
    稀疏 MoE 层（Top-K 路由 + 辅助负载均衡损失）
    Args:
        d_model:        输入维度
        d_ff:           专家 FFN 内部维度
        num_experts:    总专家数 N
        top_k:          每 token 激活专家数 k
        aux_loss_alpha: 辅助损失系数 α
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        aux_loss_alpha: float = 1e-2,
    ):
        super().__init__()
        self.N = num_experts
        self.k = top_k
        self.alpha = aux_loss_alpha
        # 路由器：线性层，输出 N 维 logits
        self.router = nn.Linear(d_model, num_experts, bias=False)
        # 专家列表
        self.experts = nn.ModuleList(
            [ExpertFFN(d_model, d_ff) for _ in range(num_experts)]
        )
    def _compute_aux_loss(
        self,
        router_probs: torch.Tensor,   # (T, N)，门控软概率
        dispatch_mask: torch.Tensor,  # (T, N)，bool，token 是否路由到专家 i
    ) -> torch.Tensor:
        """
        Switch Transformer 辅助负载均衡损失
        L_aux = α * N * Σ_i (f_i * p_i)
        """
        T = router_probs.size(0)
        # f_i: 专家 i 实际接收的 token 比例（离散，不可微）
        f = dispatch_mask.float().mean(dim=0)  # (N,)
        # p_i: 门控对专家 i 的平均软概率（可微）
        p = router_probs.mean(dim=0)           # (N,)
        aux_loss = self.alpha * self.N * (f * p).sum()
        return aux_loss
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output:   (batch_size, seq_len, d_model)
            aux_loss: scalar，需加入总损失
        """
        B, L, D = x.shape
        T = B * L
        # 展平 batch 和 seq 维度：(T, D)
        x_flat = x.view(T, D)
        # ── 路由计算 ──────────────────────────────────────────
        # logits: (T, N)
        logits = self.router(x_flat)
        # 软概率（用于辅助损失）
        router_probs = F.softmax(logits, dim=-1)  # (T, N)
        # Top-K：取最高 k 个专家的索引
        topk_logits, topk_indices = torch.topk(logits, self.k, dim=-1)  # (T, k)
        # 稀疏门控权重：只保留 top-k，其余为 0
        topk_weights = F.softmax(topk_logits, dim=-1)  # (T, k)，归一化
        # dispatch_mask: (T, N)，token t 是否路由到专家 i
        dispatch_mask = torch.zeros(T, self.N, dtype=torch.bool, device=x.device)
        dispatch_mask.scatter_(1, topk_indices, True)
        # ── 辅助损失 ──────────────────────────────────────────
        aux_loss = self._compute_aux_loss(router_probs, dispatch_mask)
        # ── 专家计算 ──────────────────────────────────────────
        output = torch.zeros_like(x_flat)  # (T, D)
        for expert_idx in range(self.N):
            # 找出路由到该专家的 token 下标
            token_indices = dispatch_mask[:, expert_idx].nonzero(as_tuple=True)[0]
            if token_indices.numel() == 0:
                continue  # 该专家本步未被激活，跳过
            # 取出对应 token
            expert_input = x_flat[token_indices]          # (num_tokens, D)
            # 专家计算
            expert_output = self.experts[expert_idx](expert_input)  # (num_tokens, D)
            # 找出这些 token 对应的门控权重（在 top-k 中的位置）
            # topk_indices[token_indices]: (num_tokens, k)
            # 找到 expert_idx 在 top-k 中的列位置
            expert_pos = (topk_indices[token_indices] == expert_idx).nonzero(as_tuple=True)[1]
            weights = topk_weights[token_indices, expert_pos].unsqueeze(-1)  # (num_tokens, 1)
            # 加权累加
            output.index_add_(0, token_indices, weights * expert_output)
        # 恢复 shape
        output = output.view(B, L, D)
        return output, aux_loss
# ── 使用示例 ─────────────────────────────────────────────────
if __name__ == "__main__":
    moe = MoELayer(
        d_model=512,
        d_ff=2048,
        num_experts=8,
        top_k=2,
        aux_loss_alpha=1e-2,
    )
    x = torch.randn(2, 16, 512)          # batch=2, seq_len=16, d_model=512
    out, aux_loss = moe(x)
    # 训练时：总损失 = 主任务损失 + 辅助损失
    # loss = task_loss + aux_loss
    print(f"output shape: {out.shape}")   # (2, 16, 512)
    print(f"aux_loss:     {aux_loss.item():.4f}")
```
---
## 8. MoE 与 Dropout 的对比
### 8.1 表面相似性与本质差异
两者都涉及"部分神经元/模块不激活"，但动机、机制、效果完全不同。
|维度|Dropout|Sparse MoE|
|---|---|---|
|**目的**|正则化，防止过拟合|扩大模型容量，提升计算效率|
|**激活方式**|训练时随机丢弃，推理时全部激活|训练和推理均稀疏激活|
|**选择依据**|无条件随机（均匀分布）|条件路由（依赖输入内容）|
|**参数利用**|同一组参数，部分暂时不用|不同专家有独立参数|
|**推理行为**|恢复全量激活（乘以保留率 $1-p$）|始终只激活 Top-K 专家|
|**容量增益**|无（参数量不变）|有（参数量 = $N \times$ 单专家参数）|
|**计算量变化**|训练减少，推理不变|训练和推理均减少|
### 8.2 Dropout 的数学形式
训练时：
$$\tilde{\mathbf{h}} = \mathbf{m} \odot \mathbf{h}, \quad m_i \sim \text{Bernoulli}(1-p)$$
推理时（期望等价）：
$$\hat{\mathbf{h}} = (1-p) \cdot \mathbf{h}$$
**本质**：对同一组参数做集成（ensemble），等价于训练了 $2^n$ 个共享权重的子网络。
### 9.3 MoE 路由的数学形式（对比）
$$\text{MoE}(\mathbf{x}) = \sum_{i \in \text{TopK}} G(\mathbf{x})_i \cdot E_i(\mathbf{x})$$
**本质**：输入内容决定激活哪些参数，是**条件计算（Conditional Computation）**，不是正则化。
### 9.4 能否将 Dropout 用于 MoE？
可以，但需区分施加位置：
```
位置 1：专家内部 FFN（标准 Dropout）
   → 对专家权重本身做正则化，正常使用
位置 2：路由器输出（对门控权重 Dropout）
   → 等价于随机强制 token 探索更多专家
   → 有助于缓解专家坍塌，但效果不如辅助损失稳定
位置 3：专家级别 Dropout（随机丢弃整个专家）
   → 类似 DropBlock，训练时增强鲁棒性
   → 不常用，工程复杂度高
```
## 9. 关键问题总结
|问题|原因|解决方案|
|---|---|---|
|**专家坍塌**|路由器倾向固定专家|辅助损失 / Noisy Gating / 偏置动态调整|
|**负载不均**|token 分布不均匀|Expert Capacity + 辅助损失|
|**显存压力**|所有专家需常驻显存|专家卸载 / 量化 / 缓存|
|**通信瓶颈**|All-to-All 跨设备路由|EP+DP 并行 / 减少专家并行节点数|
|**训练不稳定**|路由离散操作不可微|软概率辅助损失代替硬路由梯度|
|**推理延迟**|专家串行/并行调度|批量合并同一专家的 token（batched GEMM）|
---
## 10. 面试高频问题与标准答案
### Q1：MoE 为什么能在参数量增大的同时保持 FLOPs 不变？
**答**：
MoE 将 FFN 层替换为 $N$ 个独立专家，但每个 token 只激活其中 $k$ 个。
$$\text{FLOPs}_{\text{MoE}} = k \times \text{FLOPs}_{\text{single expert}} \approx \frac{k}{N} \times \text{FLOPs}_{\text{dense FFN (N experts worth)}}$$
参数量扩大了 $N$ 倍，但计算量只有 $k/N$ 倍。关键前提：$k \ll N$（如 $k=2, N=8$）。
---
### Q2：Top-1 和 Top-2 路由各有何优劣？
||Top-1（Switch Transformer）|Top-2（Mixtral）|
|---|---|---|
|**FLOPs**|最低|2× Top-1|
|**训练稳定性**|较差（梯度方差大）|较好|
|**专家利用率**|低（更易坍塌）|较高|
|**负载均衡难度**|低|中|
|**表达能力**|弱（单一专家决策）|强（两专家加权组合）|
**一句话**：Top-1 省计算但难训练；Top-2 是工程与效果的平衡点，目前主流。
---
### Q3：辅助损失（Auxiliary Loss）为什么用 $f_i \times p_i$ 而不是直接最小化 $f_i$ 的方差？
**答**：
$f_i$（实际 token 分配比例）是**离散量**，通过 argmax/TopK 得到，对路由器参数 $\mathbf{W}_g$ **不可微**，无法直接反向传播。
因此引入可微的软概率 $p_i = \text{mean}(\text{Softmax}(\mathbf{x} \mathbf{W}_g)_i)$，用 $f_i \times p_i$ 构造代理目标：
- $f_i$ 提供方向信号（哪个专家负载过重）
- $p_i$ 提供梯度通路
当两者均匀时乘积最小（均值不等式），达到负载均衡目的。
---
### Q4：Expert Capacity 溢出时，token 怎么处理？
**答**：溢出 token 执行 **pass-through**（跳过该 MoE 层），直接将输入 $\mathbf{x}$ 通过残差连接传到下一层：
$$\text{output} = \mathbf{x} + \text{MoE}(\mathbf{x})_{\text{routed tokens}}$$
溢出 token 的 $\text{MoE}(\mathbf{x}) = \mathbf{0}$，等价于该层对其无贡献。
**工程含义**：capacity_factor 设太小会导致大量 token 被丢弃，质量下降；设太大则浪费显存/计算。
---
### Q5：为什么 MoE 的推理延迟有时比同 FLOPs 的 Dense 模型更高？
**答**：三个原因：
**① 内存带宽瓶颈（Memory-Bound）**
推理时（小 batch）往往是内存带宽瓶颈而非算力瓶颈。MoE 需加载 $N$ 个专家权重，但每步只计算 $k$ 个，权重加载量远大于计算量。
$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Loaded}} \downarrow$$
**② All-to-All 通信（多卡）**
多卡部署时 token 跨设备路由引入额外通信延迟。
**③ 专家计算碎片化**
不同 token 路由到不同专家，难以合并为大矩阵乘法（GEMM），GPU 利用率低。
**解决思路**：
```
batch 内将路由到同一专家的 token 聚合 → 一次大 GEMM
↑ 这是 MoE 推理 kernel 优化的核心思路（grouped GEMM）
```
---
### Q6：MoE 与多头注意力（MHA）中的"多头"有何异同？
||MHA 多头|MoE 多专家|
|---|---|---|
|**并行方式**|所有头同时计算（稠密）|仅激活 Top-K 专家（稀疏）|
|**参数独立性**|各头有独立 $W_Q, W_K, W_V$|各专家有独立 FFN 权重|
|**选择机制**|无路由，全部激活|路由器条件选择|
|**捕获信息**|不同子空间的注意力模式|不同语义领域的知识|
|**输出融合**|拼接后线性变换|加权求和|
**共同点**：都是"分而治之"的模块化思想，用多个独立子模块覆盖不同的表示空间。
---
### Q7：如何理解"专家专业化"（Expert Specialization）？
**答**：
训练收敛后，路由器倾向于将语义相似的 token 路由到同一专家：
```
实验观察（Mixtral 论文）：
- 数学/代码 token → 集中路由到特定专家
- 自然语言 token → 路由到另一组专家
- 多语言 token → 不同语言有偏好专家
```
**本质**：路由器通过梯度下降自发学习到"哪类输入交给哪个专家处理更高效"，无需人工标注。
**验证方式**：
```python
# 统计每个专家接收的 token 类型分布
# 可视化路由矩阵热力图：行=token类型，列=专家编号
router_logits  # (T, N)
# 按 token 的语义类别分组，观察路由分布是否聚类
```
---
### Q8：DeepSeek-V3 的辅助无损（Auxiliary-Loss-Free）负载均衡是怎么实现的？
**答**：
传统辅助损失的问题：$\mathcal{L}_{\text{aux}}$ 会干扰主任务梯度，可能损害模型性能。
DeepSeek-V3 方案：对每个专家维护一个**偏置项 $b_i$**，加入路由 logit：
$$s_i = \mathbf{x} \cdot \mathbf{W}_g^{(i)} + b_i$$
**$b_i$ 的更新规则**（不参与梯度，单独更新）：
$$b_i \leftarrow b_i - \gamma \cdot \text{sign}(L_i - \bar{L})$$
- $L_i$：专家 $i$ 的当前负载（接收 token 数）
- $\bar{L}$：所有专家平均负载
- $\gamma$：更新步长（超参）
**逻辑**：负载高于平均 → $b_i$ 减小 → 该专家被选中概率降低 → 负载自动回落。不引入额外损失项，主任务梯度干净。
---
### Q9：MoE 在训练时的梯度稀疏性问题
**答**：
每个 token 只更新被路由到的 $k$ 个专家的参数，未被选中的专家在该 step **梯度为零**。
**后果**：
- 每个专家的有效 batch size = $\frac{k}{N} \times$ 全局 batch size
- 专家参数更新频率低 → 收敛慢
- 极端情况（专家坍塌）→ 部分专家几乎不更新
**缓解方案**：
```
1. 增大全局 batch size（保证每个专家的有效 batch 足够大）
   DeepSeek-V3 训练 batch size ≈ 15,360 tokens × 极大梯度累积步数
2. Noisy Gating（增加探索，避免固定专家）
3. 辅助损失 / 偏置调整（均匀分配 token）
4. 专家权重初始化：各专家独立随机初始化（非共享），
   防止早期训练对称性导致所有专家相同
```
---
### Q10：如果让你从头设计一个 MoE 系统，关键决策点是什么？
**决策清单**：
```
1. 专家数 N 与激活数 k
   - 推理预算优先 → 固定 k，增大 N
   - 显存有限 → 减小 N 或引入专家卸载
2. MoE 层的位置
   - 每层替换 FFN（Mixtral 方案，最彻底）
   - 间隔替换（每隔 2 层，节省显存）
   - 仅替换高层（低层共享语法知识，高层专业化）
3. 路由策略
   - 在线部署 → Token-Choice（逐 token 路由）
   - 离线批处理 → Expert-Choice（均衡更好）
   - 极大规模 → 辅助无损偏置调整（DeepSeek-V3）
4. 负载均衡
   - 小规模实验：辅助损失（简单）
   - 大规模生产：偏置动态调整（避免梯度污染）
5. 共享专家
   - 有通用知识需求 → 加入 1~2 个共享专家
   - 纯专业化场景 → 全路由专家
6. 并行策略
   - 单机多卡：TP + DP
   - 多机：EP（专家并行）+ DP，注意 All-to-All 通信开销
```
---
## 11. 知识点速查卡（面试背诵）
```
MoE 三要素
├── 专家（Expert）：独立 FFN，有各自参数
├── 路由器（Router）：线性层 + Softmax/TopK
└── 稀疏激活：每 token 只算 k/N 个专家
核心公式
├── 输出：Σ G(x)_i · E_i(x)，仅 Top-K 项非零
├── 长度惩罚：lp(Y) = ((5+|Y|)/6)^α
└── 辅助损失：α·N·Σ f_i·p_i
主要挑战 → 解决方案
├── 专家坍塌    → 辅助损失 / Noisy Gating
├── 负载不均    → Expert Capacity + 辅助损失
├── 推理延迟    → Grouped GEMM / 专家缓存
├── 显存压力    → 专家量化 / 卸载
└── 梯度稀疏    → 大 batch + 均衡路由
关键论文
├── Jacobs 1991         → 原始 MoE
├── Shazeer 2017        → Sparse MoE + Noisy Top-K
├── Fedus 2022          → Switch Transformer（Top-1）
├── Mixtral 2024        → 8x7B，Top-2，主流基准
└── DeepSeek-V3 2024    → 671B，辅助无损负载均衡
```