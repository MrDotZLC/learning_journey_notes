---
tags: [RLHF, PPO, 策略梯度, 强化学习, 优势估计]
aliases: [PPO笔记, 近端策略优化]
created: 2025-03-09
status: complete
---

## PPO 深度笔记（Proximal Policy Optimization）

> 主文档：[[RLHF（Post-Training）]]　｜　相关：[[SFT（Supervised Fine-Tuning）深度解析]] · [[RM（Reward Model） 深度介绍]]

---

### ⚡ 速查卡

| 维度 | 内容 |
|---|---|
| 核心目标 | 在 KL 约束下最大化 RM 奖励 |
| 算法类型 | On-policy 策略梯度，重要性采样 |
| 模型数量 | 4 个（Actor / Critic / Ref / RM） |
| 关键机制 | Clip、GAE、KL 惩罚 |
| 核心挑战 | 显存开销、超参敏感、Critic 收敛 |

---

## 第一章 从 REINFORCE 到 PPO

### 1.1 策略梯度基础

**目标**：最大化期望奖励

$$J(\theta) = \mathbb{E}_{y \sim \pi_\theta}[R(y)]$$

**REINFORCE 梯度（log-derivative trick）**：

$$\nabla_\theta J = \mathbb{E}_{y \sim \pi_\theta}\left[R(y) \cdot \nabla_\theta \log\pi_\theta(y)\right]$$

**推导**：

$$\nabla_\theta \mathbb{E}[R] = \nabla_\theta \int R(y)\pi_\theta(y)\,dy = \int R(y)\nabla_\theta\pi_\theta(y)\,dy$$

$$= \int R(y)\cdot\pi_\theta(y)\cdot\underbrace{\frac{\nabla_\theta\pi_\theta(y)}{\pi_\theta(y)}}_{\nabla_\theta\log\pi_\theta(y)}\,dy = \mathbb{E}\left[R(y)\cdot\nabla_\theta\log\pi_\theta(y)\right]$$

**REINFORCE 的问题**：高方差，样本效率低，步长难以控制。

### 1.2 引入基线（Baseline）降低方差

对于任意与动作无关的基线 $b$：

$$\mathbb{E}_{y\sim\pi_\theta}\left[b \cdot \nabla_\theta\log\pi_\theta(y)\right] = b\nabla_\theta\underbrace{\sum_y\pi_\theta(y)}_{=1} = 0$$

因此基线不影响梯度期望，但显著降低方差：

$$\nabla_\theta J = \mathbb{E}\left[(R(y) - b) \cdot \nabla_\theta\log\pi_\theta(y)\right]$$

最优基线 $b^* = V(s)$（当前状态的价值函数），此时 $R - V = A$（优势函数）。

### 1.3 TRPO：信任域约束

TRPO（Trust Region Policy Optimization，2015）通过 KL 散度约束保证单调提升：

$$\max_\theta \; \mathbb{E}_t\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t\right] \quad \text{s.t.} \quad \mathbb{E}_t\left[\mathbb{KL}[\pi_{\theta_{old}} \| \pi_\theta]\right] \leq \delta$$

**缺陷**：需要计算 Fisher 信息矩阵（Hessian of KL）并求逆，时间复杂度 $O(n^2)$，无法用于亿级参数模型。

### 1.4 PPO：一阶近似 TRPO

用 clip 机制近似 TRPO 的信任域约束，保持一阶优化：

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(\underbrace{\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}}_{\rho_t} \hat{A}_t,\; \text{clip}\left(\rho_t, 1-\epsilon, 1+\epsilon\right)\hat{A}_t\right)\right]$$

---

## 第二章 RLHF 中的 PPO

### 2.1 RL 概念映射

| RL 术语 | RLHF 中的对应 |
|---|---|
| 环境（Environment） | 人类 / Reward Model |
| 智能体（Agent） | 语言模型 $\pi_\theta$ |
| 状态（State $s_t$） | 当前 prompt + 已生成 token $y_{1:t-1}$ |
| 动作（Action $a_t$） | 生成下一个 token $y_t$ |
| 奖励（Reward） | 仅在序列末尾给出 $r(x,y)$（稀疏） |
| 回合（Episode） | 一次完整的 prompt → response |
| 状态转移 | $s_{t+1} = [s_t; y_t]$（确定性拼接） |

**关键特点**：动作空间极大（词表大小，通常 32K~100K），奖励极稀疏（只在末尾），序列极长（数百至数千 token）。

### 2.2 完整优化目标

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) - \beta \cdot \mathbb{KL}\left[\pi_\theta(y|x) \,\|\, \pi_{ref}(y|x)\right] \right]$$

**KL 散度 per-token 展开**：

$$\mathbb{KL}[\pi_\theta \| \pi_{ref}] = \sum_{t=1}^{T} \log \frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{ref}(y_t|x,y_{<t})}$$

**每步实际奖励**（将 KL 惩罚分配到每一步）：

$$\tilde{r}_t = \begin{cases} r_\phi(x,y) - \beta\log\dfrac{\pi_\theta(y_T|x,y_{<T})}{\pi_{ref}(y_T|x,y_{<T})} & t = T \\ -\beta\log\dfrac{\pi_\theta(y_t|x,y_{<t})}{\pi_{ref}(y_t|x,y_{<t})} & t < T \end{cases}$$

### 2.3 四模型布局

```
┌─────────────────────────────────────────────────────────────┐
│  Actor (Policy Network)  π_θ                                │
│    - 被更新的主模型                                           │
│    - 生成 response，计算 log prob                            │
├─────────────────────────────────────────────────────────────┤
│  Critic (Value Network)  V_φ                                │
│    - 被更新，估计状态价值 V(s_t)                              │
│    - 通常与 Actor 共享 backbone，只替换最后一层               │
├─────────────────────────────────────────────────────────────┤
│  Reference Policy  π_ref                                    │
│    - 冻结（SFT 模型）                                        │
│    - 只用于计算 KL 惩罚                                      │
├─────────────────────────────────────────────────────────────┤
│  Reward Model  r_ψ                                          │
│    - 冻结                                                   │
│    - 只在序列末尾调用一次                                     │
└─────────────────────────────────────────────────────────────┘

显存占用（70B 模型为例，fp16）：
  Actor:   ~140GB
  Critic:  ~140GB（若共享 backbone 则省去）
  Ref:     ~140GB
  RM:      ~140GB（若比 Actor 小则省去部分）
  总计：   ~560GB → 需要 7×80GB A100
```

---

## 第三章 PPO-Clip 机制深度分析

### 3.1 重要性采样（Importance Sampling）

PPO 使用旧策略 $\pi_{\theta_{old}}$ 采集数据，然后对新策略 $\pi_\theta$ 更新多次（off-policy 复用）。

重要性权重修正：

$$\mathbb{E}_{a \sim \pi_\theta}[f(a)] = \mathbb{E}_{a \sim \pi_{\theta_{old}}}\left[\frac{\pi_\theta(a)}{\pi_{\theta_{old}}(a)} f(a)\right]$$

当 $\pi_\theta$ 和 $\pi_{\theta_{old}}$ 差异过大时，重要性权重 $\rho = \pi_\theta/\pi_{\theta_{old}}$ 方差爆炸。

### 3.2 Clip 机制的数学含义

$$\mathcal{L}^{CLIP} = \mathbb{E}_t\left[\min\left(\rho_t \hat{A}_t,\; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

**分情况分析**（$\epsilon = 0.2$）：

| $\hat{A}_t$ | $\rho_t$ 范围 | 有效目标 | 含义 |
|---|---|---|---|
| $> 0$（好动作） | $\rho_t \leq 1+\epsilon$ | $\rho_t \hat{A}_t$ | 正常增大概率 |
| $> 0$（好动作） | $\rho_t > 1+\epsilon$ | $(1+\epsilon)\hat{A}_t$（常数） | 梯度为 0，不再增大 |
| $< 0$（坏动作） | $\rho_t \geq 1-\epsilon$ | $\rho_t \hat{A}_t$ | 正常减小概率 |
| $< 0$（坏动作） | $\rho_t < 1-\epsilon$ | $(1-\epsilon)\hat{A}_t$（常数） | 梯度为 0，不再减小 |

**几何解释**：clip 在 $\rho_t$ 超出 $[1-\epsilon, 1+\epsilon]$ 范围后将目标函数"压平"，使梯度变为 0，从而限制每步更新幅度。

### 3.3 PPO 的熵正则化

为防止策略过快收敛到确定性策略（探索停止），PPO 通常加入熵奖励：

$$\mathcal{L}^{total} = \mathcal{L}^{CLIP} + c_1 \mathcal{L}^{VF} - c_2 H[\pi_\theta]$$

其中 $H[\pi_\theta] = -\sum_a \pi_\theta(a)\log\pi_\theta(a)$ 是策略熵。

熵下降过快是 Reward Hacking 的早期信号（见 [[RM（Reward Model） 深度介绍]] 第四章）。

---

## 第四章 GAE（Generalized Advantage Estimation）深度分析

### 4.1 优势函数的多种估计方式

$$A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$$

不同 $\lambda$ 对应不同的估计方式：

**$\lambda = 0$（单步 TD，高偏差低方差）**：

$$\hat{A}_t^{(1)} = r_t + \gamma V(s_{t+1}) - V(s_t) = \delta_t$$

**$\lambda = 1$（蒙特卡洛，低偏差高方差）**：

$$\hat{A}_t^{(\infty)} = \sum_{l=0}^{T-t-1} \gamma^l r_{t+l} - V(s_t)$$

**GAE（插值）**：

$$\hat{A}_t^{GAE(\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$（TD 残差）。

### 4.2 GAE 的递推计算

$$\hat{A}_t = \delta_t + (\gamma\lambda)\hat{A}_{t+1}$$

从后往前递推，计算复杂度 $O(T)$：

```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    rewards: [T]，每步奖励（包含 KL 惩罚）
    values:  [T+1]，Critic 估计的状态值（最后一个用于 bootstrap）
    """
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values[:-1]  # GAE + V = Q 的估计
    return advantages, returns
```

### 4.3 语言模型 RL 中 $\gamma$ 和 $\lambda$ 的选取

**$\gamma$（折扣因子）**：

语言生成任务中，一句话的所有 token 在时间上紧密关联，未来奖励不应大幅折扣。通常 $\gamma = 0.99 \sim 1.0$。

**$\lambda$（GAE 系数）**：

语言任务中 Critic 在训练初期不准确，$\lambda$ 过小（依赖 Critic）会引入大偏差。通常 $\lambda = 0.95$，在低偏差方向偏移。

**RLHF 特殊性**：只有序列末尾有 RM 奖励，其余步骤 $r_t = -\beta \cdot \text{KL}_t$（通常为负）。这使得 GAE 的传播具有特殊形状——末尾奖励通过指数衰减向前传播。

### 4.4 Returns 的计算

$$\text{returns}_t = \hat{A}_t + V(s_t)$$

Returns 作为 Critic 的训练目标：

$$\mathcal{L}^{VF} = \mathbb{E}_t\left[(V_\phi(s_t) - \text{returns}_t)^2\right]$$

**Value Clipping**（可选）：类似 PPO-Clip，对 Critic 更新也做 clip，防止 Critic 单步更新过大：

$$\mathcal{L}^{VF}_{clip} = \max\left[(V_\phi - ret)^2,\; (\text{clip}(V_\phi, V_{old}-\epsilon_v, V_{old}+\epsilon_v) - ret)^2\right]$$

---

## 第五章 完整训练流程与工程细节

### 5.1 训练流程伪代码

```python
def ppo_rlhf_training(
    actor, critic, ref_model, reward_model,
    dataset, config
):
    optimizer_actor  = Adam(actor.parameters(),  lr=config.actor_lr)
    optimizer_critic = Adam(critic.parameters(), lr=config.critic_lr)

    for iteration in range(config.num_iterations):

        # ═══════════════════════════════════════════════
        # Phase 1: Rollout（经验收集，不计算梯度）
        # ═══════════════════════════════════════════════
        actor.eval()
        critic.eval()
        rollout_buffer = []

        for prompt_batch in sample_batches(dataset, config.rollout_batch_size):
            with torch.no_grad():
                # 1.1 生成 response
                responses, logprobs_old = actor.generate_with_logprobs(prompt_batch)

                # 1.2 计算参考策略 log prob（用于 KL）
                logprobs_ref = ref_model.log_prob(prompt_batch, responses)

                # 1.3 RM 打分（只调用一次，在序列末尾）
                rm_scores = reward_model.score(prompt_batch, responses)

                # 1.4 构造每步奖励（KL 惩罚分配到每步）
                kl_per_token = logprobs_old - logprobs_ref      # per-token KL
                rewards = -config.beta * kl_per_token           # 每步负奖励
                rewards[:, -1] += rm_scores                     # 末尾加 RM 分

                # 1.5 Critic 估计状态价值
                values = critic(prompt_batch, responses)         # [B, T]

                # 1.6 计算 GAE
                advantages, returns = compute_gae(
                    rewards, values,
                    gamma=config.gamma, lam=config.lam
                )
                advantages = normalize(advantages)              # 标准化优势

            rollout_buffer.append({
                "prompts": prompt_batch, "responses": responses,
                "logprobs_old": logprobs_old, "advantages": advantages,
                "returns": returns
            })

        # ═══════════════════════════════════════════════
        # Phase 2: PPO Update（策略更新，计算梯度）
        # ═══════════════════════════════════════════════
        actor.train()
        critic.train()

        for epoch in range(config.ppo_epochs):      # 通常 4 个 epoch
            for mini_batch in shuffle_and_split(rollout_buffer):
                # 2.1 重新计算当前策略的 log prob
                logprobs_new = actor.log_prob(
                    mini_batch["prompts"], mini_batch["responses"]
                )

                # 2.2 重要性采样比率
                ratio = (logprobs_new - mini_batch["logprobs_old"]).exp()

                # 2.3 PPO-Clip 策略损失
                adv = mini_batch["advantages"]
                policy_loss = -torch.min(
                    ratio * adv,
                    ratio.clamp(1 - config.eps, 1 + config.eps) * adv
                ).mean()

                # 2.4 Critic 损失（Value Function Loss）
                values_new = critic(
                    mini_batch["prompts"], mini_batch["responses"]
                )
                value_loss = F.mse_loss(values_new, mini_batch["returns"])

                # 2.5 熵奖励（鼓励探索）
                entropy = actor.entropy(
                    mini_batch["prompts"], mini_batch["responses"]
                ).mean()

                # 2.6 总损失
                loss = (policy_loss
                        + config.c1 * value_loss
                        - config.c2 * entropy)

                # 2.7 梯度更新
                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                optimizer_actor.step()
                optimizer_critic.step()
```

### 5.2 关键超参数详解

| 超参数 | 典型值 | 过大 | 过小 |
|---|---|---|---|
| $\beta$（KL系数） | 0.01~0.1 | 模型几乎不动 | Reward Hacking |
| $\epsilon$（clip范围） | 0.2 | 更新不稳定 | 更新太保守 |
| $\gamma$（折扣因子） | 0.99~1.0 | N/A（语言任务通常 ≈1） | 短视，忽略末尾奖励 |
| $\lambda$（GAE系数） | 0.95 | 高方差（蒙特卡洛） | 高偏差（依赖Critic） |
| PPO epochs | 4 | Critic 跟不上 | 样本利用率低 |
| Actor LR | 1e-6~5e-6 | 训练不稳定 | 收敛太慢 |
| Critic LR | 1e-5~5e-5 | Critic 震荡 | Critic 收敛慢 |
| $c_1$（Critic Loss系数） | 0.5~1.0 | Critic 主导训练 | Critic 更新不足 |
| $c_2$（熵系数） | 0.01~0.05 | 策略过于随机 | 熵坍塌 |

### 5.3 Critic 收敛问题与解法

**问题**：训练初期 Critic 估计误差大，GAE 计算的优势函数不可靠，Actor 基于错误优势更新方向偏错。

**诊断**：监控 `explained_variance = 1 - Var(returns - values) / Var(returns)`，若远低于 0.5 则 Critic 未收敛。

**解决方案**（按优先级）：

```python
# 1. Critic 预热：前 N 步只训练 Critic
for step in range(warmup_steps):
    value_loss = compute_value_loss(critic, batch)
    optimizer_critic.step(value_loss)
    # Actor 不更新

# 2. Critic LR = Actor LR × 3~10
optimizer_critic = Adam(critic.parameters(), lr=actor_lr * 5)

# 3. 增大 Critic Loss 系数 c1
c1 = 1.0  # 默认 0.5，增大到 1.0

# 4. Value Clipping（防止 Critic 单步更新过大）
v_clipped = v_old + (v_new - v_old).clamp(-eps_v, eps_v)
value_loss = torch.max((v_new - ret)**2, (v_clipped - ret)**2).mean()
```

### 5.4 显存优化技术

#### 5.4.1 梯度检查点（Gradient Checkpointing）

以重计算换显存：前向传播时不保存中间激活值，反向传播时重新计算：

```python
model.gradient_checkpointing_enable()
# 显存减少约 50%，计算增加约 30%
```

#### 5.4.2 ZeRO（Zero Redundancy Optimizer）

将模型参数、梯度、优化器状态分散到多个 GPU：

```
ZeRO-1：分片优化器状态（节省 4x 显存）
ZeRO-2：分片优化器状态 + 梯度（节省 8x 显存）
ZeRO-3：分片优化器状态 + 梯度 + 参数（节省 64x 显存）
```

#### 5.4.3 Actor-Critic 参数共享

Actor 和 Critic 共享 Transformer backbone，只有最后一层不同：

```python
class ActorCritic(nn.Module):
    def __init__(self, backbone):
        self.backbone = backbone              # 共享
        self.lm_head = nn.Linear(d, vocab)   # Actor head
        self.value_head = nn.Linear(d, 1)    # Critic head

    def forward(self, x):
        h = self.backbone(x).last_hidden_state
        logits = self.lm_head(h)             # 生成 token
        values = self.value_head(h[:, -1])   # 状态价值
        return logits, values
```

**风险**：Actor 和 Critic 的优化目标可能冲突（Actor 最大化奖励，Critic 最小化价值估计误差），导致训练不稳定。实践中通常分离。

#### 5.4.4 Reference Model 的显存优化

Reference Model 只用于 log prob 计算，不需要存储梯度：

```python
# 方案1：使用 no_grad
with torch.no_grad():
    logprobs_ref = ref_model(input_ids).logits

# 方案2：量化到 INT8
from transformers import AutoModelForCausalLM
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name, load_in_8bit=True
)

# 方案3：CPU offload（适合显存极度紧张的场景）
ref_model = ref_model.to("cpu")
logprobs_ref = ref_model(input_ids.cpu()).logits.to("cuda")
```

---

## 第六章 PPO 的稳定性与调试

### 6.1 常见训练失败模式

| 失败模式 | 症状 | 根因 | 解法 |
|---|---|---|---|
| Critic 发散 | Value Loss 突然增大 | 学习率过大或梯度爆炸 | 降低 Critic LR，加梯度裁剪 |
| Actor 坍塌 | 生成长度极短（只输出 EOS） | KL 惩罚过大或优势估计偏负 | 降低 $\beta$，检查 GAE 计算 |
| Reward Hacking | RM 分数高但生成质量差 | $\beta$ 过小 | 增大 $\beta$，参见 [[RM（Reward Model） 深度介绍]] |
| 熵坍塌 | 策略退化为固定模板 | 探索不足，$c_2$ 过小 | 增大熵系数 $c_2$ |
| 梯度爆炸 | Loss 为 NaN | clip 范围过大或 LR 过高 | 梯度裁剪（max norm=1.0） |

### 6.2 训练监控指标清单

```python
# 每个 iteration 记录以下指标
monitor = {
    # 核心奖励
    "rm_score_mean": rm_scores.mean(),
    "rm_score_std":  rm_scores.std(),

    # KL 散度
    "kl_divergence": kl_per_token.sum(dim=-1).mean(),
    "kl_penalty":    (config.beta * kl_per_token.sum(dim=-1)).mean(),

    # 策略梯度
    "policy_loss":   policy_loss.item(),
    "ratio_mean":    ratio.mean().item(),
    "ratio_clipped": (ratio < 1-eps).float().mean() + (ratio > 1+eps).float().mean(),

    # Critic
    "value_loss":         value_loss.item(),
    "explained_variance": explained_variance(values, returns),

    # 探索
    "entropy": entropy.mean().item(),

    # 生成质量
    "response_length_mean": response_lengths.mean(),
    "response_length_std":  response_lengths.std(),
}
```

**关键阈值**：

| 指标 | 正常范围 | 异常信号 |
|---|---|---|
| KL 散度 | $< 4\beta$ | $> 10\beta$，可能 reward hack |
| ratio_clipped | 10~30% | $> 50\%$，更新太激进 |
| explained_variance | $> 0.5$ | $< 0.3$，Critic 未收敛 |
| 熵 | 缓慢下降 | 急剧下降（$<$ 初始值 50%） |

### 6.3 PPO 超参调优顺序

```
Step 1: 先固定 β，跑短时间训练，确认 KL 散度在合理范围
Step 2: 调 Critic LR（目标：explained_variance > 0.5）
Step 3: 调 Actor LR（目标：policy_loss 稳定下降）
Step 4: 调 ε（如果 ratio_clipped > 50%，减小 ε）
Step 5: 调 c2（如果熵下降过快，增大 c2）
Step 6: 根据 reward hack 情况调整 β
```

---

## 第七章 PPO 的局限与改进方向

### 7.1 PPO 的核心问题

| 问题 | 描述 | 改进方向 |
|---|---|---|
| 四模型显存 | Actor+Critic+Ref+RM，大模型无法承受 | GRPO（去Critic）、DAPO（去Ref+RM） |
| Critic 难以训练 | 语言任务奖励稀疏，Critic 估计困难 | GRPO（组相对估计替代Critic） |
| 超参敏感 | $\beta, \epsilon, \lambda, \gamma$ 需要精细调节 | DAPO（去掉 KL，简化超参） |
| On-policy 低效 | 每批数据只能复用 4 个 epoch | Off-policy 方法（DPO 系列） |
| 推理任务不适用 | 策略被 KL 约束，探索受限 | DAPO（$\beta=0$）、RLVR |

### 7.2 GRPO：去掉 Critic

用组内相对奖励替代 GAE，消除 Value 网络：

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$$

从 4 个模型降至 3 个，代价是优势估计精度下降。详见 [[RLHF（Post-Training）]] 第二章。

### 7.3 DPO：消除 RL 循环

DPO 将整个 PPO 流程（RM + PPO）压缩为一个监督学习目标。适合对话对齐，不适合需要探索的推理任务。

### 7.4 DAPO：推理任务优化

去掉 KL 约束（$\beta=0$），解耦 Clip 上下界，Dynamic Sampling 解决 Dead Zone，适合大规模推理训练。

---

## 第八章 面试题

### Q1：PPO-Clip 的 clip 机制解决了什么问题？为什么取 min？

**答**：

**问题**：PPO 使用重要性采样复用旧数据，当新旧策略差异过大时，重要性权重 $\rho_t$ 方差爆炸，梯度估计不可靠，一步更新可能毁掉策略。

**取 min 的含义**：

$$\mathcal{L} = \mathbb{E}_t\left[\min(\rho_t \hat{A}_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t)\right]$$

取 min 保证目标函数是**悲观估计**（conservative）：
- $\hat{A}_t > 0$：目标函数被限制在 $(1+\epsilon)\hat{A}_t$ 以下（不过激地增大好动作的概率）
- $\hat{A}_t < 0$：目标函数被限制在 $(1-\epsilon)\hat{A}_t$ 以上（不过激地惩罚坏动作的概率）

本质是一阶方法近似 TRPO 的信任域约束，避免了二阶优化的计算开销。

### Q2：GAE 中 $\lambda$ 参数的物理含义？为什么 RLHF 通常用 $\lambda=0.95$？

**答**：

$\lambda$ 控制 GAE 对不同时间步 TD 残差的指数衰减权重：

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots$$

- $\lambda=0$：只用一步 TD 残差，高偏差（依赖 Critic 准确性）低方差
- $\lambda=1$：蒙特卡洛，低偏差（不依赖 Critic）高方差
- $\lambda=0.95$：指数衰减，考虑约 20 步的 TD 残差

**RLHF 用 $\lambda=0.95$ 的原因**：
- 语言任务奖励极稀疏，Critic 在训练初期估计不准，$\lambda$ 不宜过小（避免高偏差）
- 序列较长（数百 token），$\lambda=1$ 方差过大
- $0.95$ 是 bias-variance 的经验最优点

### Q3：Critic 收敛慢导致 Actor 学歪，如何诊断和解决？

**诊断**：`explained_variance = 1 - Var(returns - values) / Var(returns)` 远低于 0.5。

**解决**（按优先级）：
1. Critic 预热（前 N 步只训 Critic）
2. Critic LR = Actor LR × 3~10
3. 增大 $c_1$（Critic Loss 系数）
4. Value Clipping 防止 Critic 更新过激
5. 换用 GRPO（彻底去掉 Critic）

### Q4：PPO 训练中如何降低四模型的显存压力？

**答**：

| 手段 | 效果 | 代价 |
|---|---|---|
| Actor-Critic 共享 backbone | 省 ~1× | 优化目标冲突，不稳定 |
| Reference Policy INT8 量化 | 省 ~50% Ref 显存 | KL 计算有误差 |
| Reference Policy CPU offload | 省全部 Ref GPU 显存 | 推理延迟增大 |
| 梯度检查点 | 省 ~50% 激活显存 | 重计算开销 +30% |
| ZeRO-3 多卡分片 | 线性扩展显存 | 通信开销 |
| 换用 GRPO | 去掉 Critic（4→3） | 优势估计精度下降 |
| 换用 DAPO + RLVR | 去掉 Critic+Ref+RM（4→1） | 只适用于可验证任务 |

### Q5：PPO 的 on-policy 特性是什么？与 DPO 的 off-policy 有什么本质区别？

**答**：

**PPO（on-policy）**：每次更新必须用当前策略（或接近当前策略）生成的数据。rollout 后只能复用 ~4 个 epoch，之后数据"过期"，需要重新采样。

**DPO（off-policy）**：直接在固定数据集上优化，数据可无限复用。

**本质区别**：

| 维度 | PPO（on-policy） | DPO（off-policy） |
|---|---|---|
| 数据新鲜度 | 高（当前策略分布） | 低（固定数据集） |
| 探索能力 | 强（持续采样新 response） | 无（固定数据） |
| 分布偏移 | 弱（数据始终来自近似当前策略） | 强（模型更新后与数据分布差距增大） |
| 样本效率 | 低（数据只用几次） | 高（数据可无限复用） |
| 推理任务适用性 | 强（可以探索新推理路径） | 弱（受固定数据限制） |

**工程结论**：对话对齐任务（偏好数据充足）优先 DPO；推理优化任务（需要探索新解法）必须用 on-policy 方法（PPO/GRPO/DAPO）。

### Q6：RLHF 中的 KL 散度为什么要 per-token 展开，而不是序列级？

**答**：

序列级 KL 在 PPO 实现中无法直接作为每步的奖励信号（只能在末尾给），而 per-token 展开后可以将 KL 惩罚分配到每一步：

$$\mathbb{KL}[\pi_\theta \| \pi_{ref}] = \sum_t \log\frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{ref}(y_t|x,y_{<t})}$$

per-token 展开的工程意义：
1. **每步都有奖励信号**：GAE 的 TD 残差计算需要每步奖励 $r_t$，per-token KL 作为每步的惩罚项，使奖励不再极度稀疏
2. **梯度信号更密集**：不再只依赖末尾的 RM 奖励传播梯度，提升训练稳定性
3. **实现简单**：$\log\pi_\theta - \log\pi_{ref}$ 在 rollout 阶段就可以计算，无额外开销
