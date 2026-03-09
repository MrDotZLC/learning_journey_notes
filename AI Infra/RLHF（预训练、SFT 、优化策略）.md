---

tags:

- RLHF
- 强化学习
- LLM对齐
- 策略优化
- 模型推理
- 面试 aliases:
- RLHF笔记
- RLHF面试题
- 人类反馈强化学习 created: 2025-03-09 status: complete

---

## RLHF 与策略优化技术全景

> **定义**：RLHF（Reinforcement Learning from Human Feedback）——用人类偏好信号通过强化学习对语言模型进行对齐训练的技术框架。

---

### ⚡ 速查卡

#### 方法选型

```
任务类型？
├─ 对话对齐（主观偏好）
│   ├─ 有成对数据           → DPO / SimPO
│   ├─ 只有单条标注           → KTO
│   └─ 最强效果              → PPO
└─ 推理优化（可验证奖励）
    ├─ 中等规模（≤7B）         → GRPO + RLVR
    ├─ 大规模（32B+）SOTA      → DAPO
    ├─ 理论严谨性优先          → Dr.GRPO
    └─ 超难题 / 低正确率场景   → VAPO
```

#### 显存需求

```
PPO:        [Actor] [Critic] [Ref] [RM]   ← 4 个模型
GRPO:       [Actor] [Ref]   [RM]          ← 3 个模型（去 Critic）
RLVR+GRPO:  [Actor] [Ref]                 ← 2 个模型（去 RM，规则验证）
DPO:        [Actor] [Ref]                 ← 2 个模型（离线）
DAPO:       [Actor]                       ← 1 个模型（去 KL，去 Ref）
SimPO:      [Actor]                       ← 1 个模型
```

#### 方法横向比较

|方法|Value模型|Clip|长度偏差修正|KL|适用|
|---|---|---|---|---|---|
|PPO|✓ 全程|对称|GAE|✓|对话对齐|
|GRPO|✗|对称|✗|✓|推理中等难度|
|DAPO|✗|解耦|Token-level|✗|推理大规模|
|Dr.GRPO|✗|对称|Per-token归一化|可选|推理理论修正|
|VAPO|✓ 轻量|解耦|GAE+组混合|✓|推理极难题|
|DPO|✗|无|✗|隐式|对话偏好|
|SimPO|✗|无|内置 length norm|✗|对话无参考|

#### 核心结论速查

|结论|一句话|
|---|---|
|RLHF 为何需要 KL 惩罚|RM 是有噪声近似，无约束策略会 reward hack|
|DPO 为何不需要 RM|最优策略有解析解，RM 被隐式编码进策略比值|
|GRPO 为何去掉 Critic|组内相对奖励作为基线，无需 Value 网络估计|
|DAPO 为何 $\beta=0$|推理任务用规则验证器，策略应大幅偏离初始|
|Dr.GRPO 为何去 std|std 归一化引入难度偏差和长度偏差，是系统性错误|
|DeepSeek-R1 为何不用神经 RM|大规模 RL 中神经 RM 容易被 reward hacking 攻击|

---

### 统一符号定义

|符号|含义|
|---|---|
|$\pi_\theta$|待优化策略（语言模型）|
|$\pi_{ref}$|参考策略（SFT 模型，冻结）|
|$r(x,y)$|奖励函数（RM 或规则）|
|$\mathcal{D}$|偏好数据集 $(x, y_w, y_l)$|
|$\beta$|KL 惩罚系数|
|$y_w$|preferred 回答（赢家）|
|$y_l$|rejected 回答（输家）|

**所有方法的统一核心优化目标：**

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D},, y \sim \pi_\theta} \left[ r(x,y) \right] - \beta \cdot \mathbb{KL}\left[\pi_\theta | \pi_{ref}\right]$$

---

## 第一章 RLHF 概述

### 1.1 核心问题：预训练目标与人类偏好的错位

|预训练目标|人类偏好目标|
|---|---|
|最大化 token 预测似然|有帮助、无害、诚实|
|对所有文本一视同仁|区分好 / 坏回答|
|可完全自动化|需要人类主观判断|

### 1.2 整体流程

```
[预训练 LLM]
     │
     ▼
[SFT]   监督微调，构造指令跟随能力与初始策略 π₀
     │
     ▼
[RM]    奖励模型训练，将人类偏好编码为标量信号
     │
     ▼
[PPO]   强化学习策略优化
     │
     ▼
[对齐后的 LLM]
```

### 1.3 已知问题全景

|问题|描述|主要改进方案|
|---|---|---|
|Reward Hacking|策略利用 RM 漏洞，高分低质|KL 惩罚、定期重训 RM、RLVR|
|显存开销|四模型同时加载|DPO / GRPO / DAPO|
|训练不稳定|PPO 对超参极敏感|GRPO / DAPO / Dr.GRPO|
|标注偏差|标注者偏好不一致|多标注者 + 不确定性建模|
|分布偏移|离线数据与生成分布不匹配|Online RLHF / Online DPO|
|稀疏奖励|长推理链信用分配困难|PRM / VAPO|

---

## 第二章 SFT（Supervised Fine-Tuning，监督微调）

### 2.1 目标

将预训练模型转化为**指令跟随模型**，为后续 RL 提供高质量初始策略 $\pi_0$。

### 2.2 数据构造

```
输入：(prompt, demonstration) 对
  - prompt       ：用户指令
  - demonstration：人工标注者撰写的高质量回答
```

> InstructGPT 使用约 **13K 条** SFT 数据——量少但质量极高。

### 2.3 训练目标

$$\mathcal{L}_{SFT} = -\sum_{t=1}^{T} \log P_\theta(y_t \mid x, y_{1:t-1})$$

### 2.4 局限性

- 标注者难以覆盖所有场景
- 标注质量参差不齐
- **无法表达偏好程度**——只能说"这个好"，无法说"这个比那个好多少"

→ 以上局限驱动了 RM 的引入。

---

## 第三章 RM（Reward Model，奖励模型）

### 3.1 核心思想

> **人类比较比生成容易得多。**

让标注者判断「A vs B 哪个更好」，而非直接撰写回答。

### 3.2 模型结构

取 SFT 模型，**移除最后的 LM head，替换为线性层输出标量**：

```
Input:  [prompt || response]
            │
    [Transformer Layers]   ← 复用 SFT 权重初始化
            │
    [Final hidden state of last token]
            │
    [Linear(d_model → 1)]
            │
Output: scalar reward  r ∈ ℝ
```

> 最后一个 token 的 hidden state 已 attend 到整个序列，是对完整 response 的压缩表示。

### 3.3 训练目标：Bradley-Terry 模型

**Bradley-Terry 模型**：成对比较概率模型，假设每个选项有潜在得分，比较结果服从 sigmoid 分布。

$$P(y_w \succ y_l \mid x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[ \log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)) \right]$$

**用 BT 模型的原因**：

- 人类标注天然是成对比较，与数据形式完全匹配
- 只需奖励的**相对大小**，不需要绝对量纲
- 可从排序数据中提取 $\binom{K}{2}$ 个偏好对，数据利用率高

|情形|$\sigma(\cdot)$|损失|
|---|---|---|
|$r(y_w) \gg r(y_l)$，已区分好坏|$\to 1$|$\to 0$ ✓|
|$r(y_w) \approx r(y_l)$，无法区分|$\to 0.5$|$-0.693$ ✗|

### 3.4 偏好数据收集

```python
for each prompt x:
    生成 K 个回答 {y_1, ..., y_K}   # K = 4~9
    标注者对所有对进行排序
    提取 C(K,2) 个偏好对 (y_w, y_l)
```

### 3.5 核心问题：Reward Hacking

$r_\theta \neq r^*_{human}$，RM 是人类偏好的有噪声近似。策略模型会找到 RM 的漏洞，产生高分但低质量输出——**这正是 KL 惩罚存在的根因**。

---

## 第四章 PPO（Proximal Policy Optimization）

### 4.1 RL 概念映射

|RL 术语|RLHF 中的对应|
|---|---|
|环境（Environment）|人类 / Reward Model|
|智能体（Agent）|语言模型 $\pi_\theta$|
|状态（State）|当前 prompt + 已生成 token|
|动作（Action）|生成下一个 token|
|奖励（Reward）|仅在序列末尾给出 $r(x, y)$|
|回合（Episode）|一次完整的 prompt → response|

> 语言生成是**稀疏奖励、长序列**的 RL 问题（奖励只在句子末尾）。

### 4.2 完整优化目标

$$\max_{\pi_\theta} \mathbb{E}_{x,, y \sim \pi_\theta} \left[ r_\phi(x, y) - \beta \cdot \mathbb{KL}\left[\pi_\theta(y|x) ,|, \pi_{ref}(y|x)\right] \right]$$

**KL 散度 per-token 展开**（实践中作为负奖励加到每一步）：

$$\mathbb{KL}[\pi_\theta | \pi_{ref}] = \sum_t \log \frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{ref}(y_t|x,y_{<t})}$$

**每步奖励构造**：

$$\tilde{r}_t = \begin{cases} r_\psi(x,y) - \beta\log\frac{\pi_\theta}{\pi_{ref}} & t = T \ -\beta\log\frac{\pi_\theta}{\pi_{ref}} & t < T \end{cases}$$

### 4.3 Policy Gradient 基础

**REINFORCE 梯度（对数技巧 log-derivative trick）**：

$$\nabla_\theta J(\theta) = \mathbb{E}\left[R(y) \cdot \nabla_\theta \log \pi_\theta(y)\right]$$

推导：

$$\nabla_\theta \mathbb{E}[R] = \sum_y R \nabla_\theta \pi_\theta = \sum_y R \cdot \pi_\theta \cdot \frac{\nabla_\theta \pi_\theta}{\pi_\theta} = \mathbb{E}\left[R \cdot \nabla_\theta \log \pi_\theta\right]$$

完整推导步骤：

1. 展开期望为积分：$\nabla_\theta \int R(y)\pi_\theta(y),dy$
2. 梯度进入积分（Leibniz rule）：$= \int R(y)\nabla_\theta\pi_\theta(y),dy$
3. 恒等变换：$\nabla_\theta\pi_\theta = \pi_\theta \cdot \nabla_\theta\log\pi_\theta$
4. 还原为期望：$= \mathbb{E}[R \cdot \nabla_\theta\log\pi_\theta]$

**带基线的 REINFORCE**（引入 $V(s)$ 降低方差，不影响期望）：

$$\nabla_\theta J = \mathbb{E}\left[(R - V(s)) \cdot \nabla_\theta \log \pi_\theta\right]$$

**TRPO**（2015）：加 KL 散度约束，保证单调提升，但需要二阶优化（Fisher 信息矩阵的逆），无法用于大模型。

### 4.4 PPO-Clip 机制

重要性采样比率：

$$\rho_t(\theta) = \frac{\pi_\theta(y_t \mid s_t)}{\pi_{\theta_{old}}(y_t \mid s_t)}$$

PPO-Clip 目标（$\epsilon = 0.2$）：

$$\mathcal{L}^{CLIP} = \mathbb{E}_t \left[ \min\left( \rho_t \hat{A}_t,; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

**clip 机制直觉**：

```
Â_t > 0（该动作好）：ratio 增大有利 → 但限制在 1+ε，防止过激更新
Â_t < 0（该动作差）：ratio 减小有利 → 但限制在 1-ε，防止过激惩罚
```

本质是一阶方法近似 TRPO 的信任域约束，同时保持实现简单。

### 4.5 优势函数：GAE（Generalized Advantage Estimation）

$$A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$$

$$\hat{A}_t^{GAE} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

|超参|$\lambda = 0$|$\lambda = 1$|
|---|---|---|
|偏差|高（纯 TD，依赖 $V$ 准确性）|低（蒙特卡洛，不依赖 $V$）|
|方差|低（只用一步奖励）|高（累积所有随机性）|

> 语言模型 RL 中通常 $\gamma \approx 1$，一句话内 token 在时间上接近，未来奖励不应大幅衰减。

### 4.6 四模型内存布局

```
┌──────────────────────────────────────────────────┐
│  Actor  (Policy)  π_θ       ←  被更新            │
│  Critic (Value)   V_φ       ←  被更新            │
│  Reference Policy π_ref     ←  冻结              │
│  Reward Model     r_ψ       ←  冻结              │
└──────────────────────────────────────────────────┘
※ 显存压力极大——这是 PPO 在大模型上难以扩展的核心原因
```

### 4.7 完整训练流程

```python
for iteration in range(num_iterations):

    # ── Phase 1: Rollout ─────────────────────────────
    for prompt x in sample_batch(dataset):
        y            = actor.generate(x)
        logprob_act  = actor.log_prob(x, y)
        logprob_ref  = ref_model.log_prob(x, y)
        reward       = reward_model.score(x, y)       # 序列级
        kl           = logprob_act - logprob_ref       # per-token
        r[:, -1]    += reward
        r           -= beta * kl
        values       = critic(x, y)

    # ── Phase 2: GAE ─────────────────────────────────
    gae = 0
    for t in reversed(range(T)):
        delta    = r[t] + gamma * values[t+1] - values[t]
        gae      = delta + gamma * lam * gae
        adv[t]   = gae
    adv     = normalize(adv)
    returns = adv + values

    # ── Phase 3: PPO update（通常 4 轮）──────────────
    for epoch in range(ppo_epochs):
        ratio       = exp(new_logprob - old_logprob)
        clip_ratio  = clamp(ratio, 1-eps, 1+eps)
        policy_loss = -min(ratio * adv, clip_ratio * adv).mean()
        value_loss  = mse(critic(x, y), returns)
        loss        = policy_loss + c1 * value_loss + c2 * entropy
        optimizer.step(loss)
```

### 4.8 关键超参数

|超参数|典型值|影响|
|---|---|---|
|$\beta$（KL系数）|0.01 ~ 0.1|过大→模型不动；过小→reward hack|
|$\epsilon$（clip）|0.2|更新步长上限|
|$\gamma$（折扣）|0.99 ~ 1.0|语言任务通常接近 1|
|$\lambda$（GAE）|0.95|bias-variance 权衡|
|PPO epochs|4|每批数据复用次数|

---

## 第五章 策略优化技术演进（2023–2025）

### 核心演进脉络

```
PPO 四模型显存过大
    └─► GRPO（2024）
            去掉 Critic，组相对估计替代 GAE
        └─► GRPO 四大缺陷：梯度消失 / 熵坍塌 / 长度偏差 / 难度偏差
            ├─► DAPO（2025.03）    工程修复
            │       Clip-Higher / 动态采样 / Token损失 / 超长过滤
            ├─► Dr.GRPO（2025.03） 理论修复
            │       去 std 归一化 / per-token 归一化
            └─► VAPO（2025.04）    混合修复
                    轻量 Value Model + Clip-Higher + 分离正负样本

PPO 需要显式 RM + RL 循环
    └─► DPO（2023）    将 RM 隐式编码进策略，消除 RL 循环
        ├─► IPO（2023）    修正 DPO 过拟合退化
        ├─► KTO（2024）    无需成对数据
        └─► SimPO（2024）  去掉参考模型
```

---

### 5.1 GRPO（Group Relative Policy Optimization，2024，DeepSeekMath）

#### 思想

对同一 prompt 采样 $G$ 个回答，用**组内相对奖励**作为基线，彻底消除 Value 网络：

$$\hat{A}_i = \frac{r_i - \text{mean}({r_j}_{j=1}^G)}{\text{std}({r_j}_{j=1}^G)}$$

#### 目标函数

$$\mathcal{L}^{GRPO} = -\frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \min\left(\rho_{i,t}\hat{A}_i,; \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_i\right) + \beta,\mathbb{KL}[\pi_\theta|\pi_{ref}]$$

#### 四大缺陷

|缺陷|根因|
|---|---|
|**梯度消失（Dead Zone）**|全对/全错 → $\text{std}=0$ → $\hat{A}=0$|
|**长度偏差**|Sample 级损失，长/短回答权重相同|
|**熵坍塌**|对称 clip 抑制低概率 token，探索停止|
|**截断干扰**|被截断序列参与损失，引入错误梯度|

---

### 5.2 DeepSeek-R1（2025.01）

#### R1-Zero vs R1

**R1-Zero**：直接在 Base Model 上用 GRPO，跳过 SFT。

- AIME 2024 pass@1：$15.6% \to 71.0%$，多数投票达 $86.7%$
- 问题：可读性差，语言混杂
- **涌现现象（Aha Moment）**：自发出现 self-verification、reflection、动态策略调整，无需显式教导

#### 四阶段流水线

```
阶段1: Cold Start SFT
    数千条高质量长CoT → 微调 Base Model
    目的：解决可读性和语言混杂

阶段2: 推理导向 RL
    GRPO + 数学/代码/逻辑（可验证奖励）+ 语言一致性奖励

阶段3: 拒绝采样 + SFT
    阶段2模型生成 → 筛选高质量推理链 + 通用能力数据 → 再次 SFT

阶段4: 全场景 RL
    GRPO + 推理任务 + 帮助性/无害性偏好数据 → DeepSeek-R1
```

#### 奖励函数（纯规则，无神经 RM）

- 准确性：答案是否正确 / 代码是否通过单元测试
- 格式：推理在 `<think>` 内，答案在 `<answer>` 内

> 不用神经 RM：大规模 RL 中神经 RM 容易被 reward hacking 攻击。

#### 实际超参（第一阶段 RL）

|超参|值|
|---|---|
|学习率|$3\times10^{-6}$|
|KL 系数|0.001|
|Clip ratio $\epsilon$|**10**（远大于 PPO 的 0.2）|
|每题采样数 $G$|16|
|最大序列长度|32768|
|Batch size|512|

---

### 5.3 DPO 系列（消除显式 RM）

#### 5.3.1 DPO（Direct Preference Optimization，2023，Stanford）

**理论推导**：带 KL 约束的 RL 最优解有解析形式：

$$\pi^*(y|x) = \frac{\pi_{ref}(y|x)\exp(r(x,y)/\beta)}{Z(x)}$$

反解 $r$，代入 Bradley-Terry，$Z(x)$ 自动消去，得：

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\underbrace{\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}}_{\text{赢家隐式奖励}} - \underbrace{\beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}}_{\text{输家隐式奖励}}\right)\right]$$

完整推导步骤（6步）：

1. 写出带 KL 约束的 RL 目标
2. 展开 KL，整理为负 KL 散度形式
3. KL $\geq 0$，等号成立时取最优解 $\pi^* \propto \pi_{ref}\exp(r/\beta)$
4. 反解 $r = \beta\log\frac{\pi_\theta}{\pi_{ref}} + \beta\log Z(x)$
5. 代入 Bradley-Terry，$\log Z(x)$ 在差值中消去
6. 最大化对数似然得 DPO 损失

**本质**：$\beta\log\frac{\pi_\theta}{\pi_{ref}}$ 即为隐式奖励，RM 被编码进策略。DPO 是 PPO 的**闭式近似**，不是独立发明的算法。

**梯度自调节**：已能区分好坏时（$\hat\sigma\to1$）梯度自动趋零；无法区分时（$\hat\sigma\to0.5$）梯度最强。

**主要问题**：离线无探索；分布偏移；$y_l$ 概率可能异常下降；不适合推理任务。

#### 5.3.2 IPO（2023）

$y_l$ 退化修复：将 $\log\sigma$ 替换为平方损失，约束奖励差趋向固定常数：

$$\mathcal{L}_{IPO} = \mathbb{E}\left[\left(\log \frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)} - \log \frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)} - \frac{1}{2\beta}\right)^2\right]$$

#### 5.3.3 KTO（2024）

基于前景理论（Prospect Theory），无需成对数据，可用单条 $(x, y, \text{label})$ 训练。人类对损失的感受比同等收益更强烈，设计非对称价值函数。

#### 5.3.4 SimPO（2024）

去掉参考模型，内置长度归一化，加 margin $\gamma$：

$$\mathcal{L}_{SimPO} = -\mathbb{E}\left[\log \sigma\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|x) - \gamma\right)\right]$$

**长度归一化的必要性**：累积 log 概率天然 $\propto$ 序列长度，不归一化时模型会学到"生成更长的 $y_w$"而非"生成更好的 $y_w$"，偏好数据中的长度分布会成为 artifact 被学习。

---

### 5.4 REINFORCE++（2025.01，Jian Hu）

**动机**：GRPO 的 question-level 归一化在小数据集上过拟合。

**核心改进**：

1. **Global Normalization**：跨 batch 全局标准化替代 per-question 局部化
2. **Mini-batch KL Loss**：KL 以损失项形式计入
3. **去 Clip**：依赖归一化控制步长

$$\hat{A}_i^{global} = \frac{r_i - \mu_{global}}{\sigma_{global}}, \qquad \mathcal{L}^{REINFORCE++} = -\mathbb{E}\left[\hat{A}_i^{global} \cdot \log\pi_\theta(y_i|x)\right] + \beta,\mathbb{KL}[\pi_\theta|\pi_{ref}]$$

---

### 5.5 DAPO（2025.03，ByteDance Seed）

**全称**：Decoupled Clip and Dynamic sAmpling Policy Optimization

> 在 Qwen2.5-32B 上达到 AIME 2024 **50分**，仅用 DeepSeek-R1 **50% 训练步数**。

#### 改进1：Clip-Higher（解耦 Clip 上下界）

**根因**：对称 clip 下界 $1-\epsilon$ 过度抑制低概率 token，导致熵坍塌。

$$\mathcal{L}^{DAPO} = \min\left(\rho_t \hat{A}_t,; \text{clip}(\rho_t, 1-\epsilon_{low}, 1+\epsilon_{high})\hat{A}_t\right)$$

典型值：$\epsilon_{low}=0.2$，$\epsilon_{high}=0.28$

#### 改进2：Dynamic Sampling（动态采样）

**根因**：全对/全错组无梯度（Dead Zone）。

```python
responses = sample(prompt, k=2*G)
valid = [r for r in responses if 0 < pass_rate(r) < 1]
batch = valid[:G]
```

#### 改进3：Token-Level Loss

从 sample 级改为全 batch token 级：

$$\mathcal{L} = -\frac{\sum_{i}\sum_{t} \ell_{i,t}}{\sum_{i} |y_i|}$$

#### 改进4：Overlong Filtering + Soft Punishment

- 被截断序列：mask 掉整个样本损失
- 正确但过长（$> L_{cache}$）：线性软惩罚

#### 去掉 KL（$\beta = 0$）

推理训练策略应大幅偏离初始策略；规则验证器提供准确奖励，KL 约束有害探索。

---

### 5.6 Dr.GRPO（2025.03，"GRPO Done Right"）

**核心发现**：GRPO 的组内标准化引入两类系统性偏差。

#### 偏差1：Question-Level Difficulty Bias（定量分析）

设题目真实正确率为 $p$，奖励 $r_i \in {0,1}$，则 $\mathbb{E}[\sigma_G] \approx \sqrt{p(1-p)}$。

归一化后有效梯度权重 $\propto 1/\sqrt{p(1-p)}$：

|难度|$p$|$\sigma_G$|有效权重|
|---|---|---|---|
|极简单|0.95|$\approx 0.22$|$\approx 4.5$|
|中等|0.5|$0.5$|$2.0$|
|极困难|0.05|$\approx 0.22$|$\approx 4.5$|

极端难度题权重比中等题高 **2.25 倍**，且 $p \to 0$ 或 $1$ 时权重趋向无穷——系统性梯度扭曲。

#### 偏差2：Response Length Bias

$\hat{A}_i$ 广播到所有 token：长度 $L$ 的序列梯度贡献 $\propto L$，模型被推向生成更长回答。

#### 修正

$$\hat{A}_i^{Dr.GRPO} = r_i - \text{mean}({r_j}) \quad \text{（去掉 std 归一化）}$$

$$\ell_{i,t} = \frac{1}{|y_i|}\min\left(\rho_{i,t}\hat{A}_i,;\text{clip}(\cdot)\right) \quad \text{（per-token 归一化）}$$

#### DAPO vs Dr.GRPO

||DAPO|Dr.GRPO|
|---|---|---|
|动机|工程优化|理论修正|
|Clip|解耦上下界|保持对称|
|长度偏差|Token-level loss|per-token 归一化|
|难度偏差|动态采样（间接）|去 std（直接）|
|KL|去掉|可选|

---

### 5.7 VAPO（2025.04，ByteDance + 阿里）

**全称**：Value-Augmented Policy Optimization

**适用场景**：正确率极低的极难题——GRPO/DAPO 在此场景失效（Dead Zone + Credit Assignment 缺失）。

#### 三大创新

**创新1：轻量级 Value Model（仅在正样本上训练）**

$$\mathcal{L}_{value} = \mathbb{E}_{y_w \sim \pi_\theta}\left[(V_\phi(s_t) - R_t)^2\right]$$

与 PPO 区别：只在正确回答上训练，避免 Critic 因负样本主导而难以收敛。

**创新2：Clip-Higher**（继承自 DAPO）

**创新3：混合优势估计**

$$\hat{A}_t^{VAPO} = \lambda_{GAE} \cdot A_t^{GAE}(V_\phi) + (1-\lambda_{GAE}) \cdot \hat{A}_i^{group}$$

有 Value 估计时用 GAE；否则退化为 GRPO 组相对估计。

---

## 第六章 前沿方向（2025）

### 6.1 推理 vs 对话的本质差异

|维度|对话对齐|推理优化|
|---|---|---|
|奖励来源|主观偏好（RM）|客观可验证（规则）|
|序列长度|短~中|极长（CoT 数千 token）|
|奖励稀疏性|中|极高（仅末尾对/错）|
|探索需求|低|极高|

### 6.2 RLVR（Reinforcement Learning with Verifiable Rewards）

用**可验证奖励**替代神经 RM，从根源消除 reward hacking：

```python
def reward(response, ground_truth):
    if not has_correct_format(response):
        return -0.5
    return 1.0 if extract_answer(response) == ground_truth else 0.0
```

|场景|验证器|
|---|---|
|数学|答案精确匹配|
|代码|单元测试通过率|
|逻辑|形式化验证器|

### 6.3 PRM（Process Reward Model，过程奖励模型）

||ORM（结果奖励）|PRM（过程奖励）|
|---|---|---|
|奖励时机|仅序列末尾|每步推理步骤|
|优点|简单，可自动化|密集信号，Credit Assignment 精准|
|缺点|信用分配困难|标注成本高，"步骤"定义困难|

当前趋势：用 LLM 自动生成 PRM 标注 + 结合 GRPO 提供 per-step 奖励。

### 6.4 Online DPO

标准 DPO 离线，分布偏移严重。Online DPO 持续采样更新，等效于将 DPO 变为 on-policy 算法：

```
loop:
    π_θ 采样 → RM 打分 → 构造 (y_w, y_l) → DPO 更新 π_θ
```

### 6.5 Test-Time Compute Scaling

与训练阶段策略优化**正交，可叠加使用**：

|方法|思路|
|---|---|
|Best-of-N|多次采样，取最高 RM 分|
|Beam Search + RM|RM 引导束搜索|
|MCTS|蒙特卡洛树搜索引导解码|

### 6.6 Self-Play（SPIN）

将当前模型作为"对手"生成 $y_l$，用 SFT 数据的 $y_w$ 作为赢家，反复自我博弈迭代。

---

## 第七章 面试题

### 一、基础概念类

#### Q1：RLHF 的整体流程是什么？为什么需要这三个阶段？

三阶段：SFT → RM 训练 → PPO 优化。

- **SFT**：预训练模型不能跟随指令，SFT 提供初始策略 $\pi_0$，同时为 RM 提供基础生成能力
- **RM**：人类直接生成高质量回答成本极高，但判断"哪个更好"成本低。RM 将成对偏好编码为可微分的标量信号
- **PPO**：有了可微分奖励信号，才能用梯度方法优化策略

三阶段缺一不可：跳过 SFT 直接 RL 会导致生成质量极差（cold start 问题）；跳过 RM 直接让人类打分无法扩展（实时人工反馈不现实）。

> **追问**：DeepSeek-R1-Zero 跳过了 SFT，为什么能成功？ 推理任务有可验证奖励（答案对/错），不依赖 RM 质量；Base Model 有足够语言能力，cold start 问题不致命。代价是输出可读性差、语言混杂——这是 R1 再加 Cold Start SFT 阶段的原因。

---

#### Q2：Reward Model 的训练目标是什么？为什么用 Bradley-Terry 模型？

$$P(y_w \succ y_l \mid x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

$$\mathcal{L}_{RM} = -\mathbb{E}\left[\log \sigma(r(y_w) - r(y_l))\right]$$

**用 BT 模型的原因**：

1. 人类标注天然是成对比较，BT 模型与数据形式完全匹配
2. 只需奖励的**相对大小**，不需要绝对量纲，避免奖励尺度校准问题
3. 可从排序数据中提取 $\binom{K}{2}$ 个偏好对，数据利用率高

> **追问**：RM 结构为什么用最后一个 token 的 hidden state？ 语言模型是自回归的，最后一个 token 已经 attend 到整个序列，是完整 response 的压缩表示。

---

#### Q3：PPO 中 KL 惩罚的作用是什么？$\beta$ 过大或过小会怎样？

KL 惩罚约束策略不能偏离参考策略太远：

$$\text{penalty} = \beta \cdot \mathbb{KL}[\pi_\theta | \pi_{ref}] = \beta \sum_t \log\frac{\pi_\theta(y_t)}{\pi_{ref}(y_t)}$$

**根本原因**：RM 是有噪声近似（$r_\theta \neq r^*$），无约束最大化 RM 分会找到 RM 漏洞，产生高分低质输出（reward hacking）。

|$\beta$ 过大|$\beta$ 过小|
|---|---|
|策略几乎不动，RL 阶段无效|策略 reward hack，输出退化|

---

#### Q4：PPO-Clip 的 clip 机制解决了什么问题？

**问题**：重要性采样复用旧数据时，新旧策略差异过大导致梯度估计方差爆炸，一步更新毁掉策略。

$$\mathcal{L}^{CLIP} = \mathbb{E}_t\left[\min\left(\rho_t \hat{A}_t,; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

取 min 的效果：

- $\hat{A}_t > 0$（好动作）：$\rho_t$ 最多增大到 $1+\epsilon$，不会过激增大概率
- $\hat{A}_t < 0$（坏动作）：$\rho_t$ 最少减小到 $1-\epsilon$，不会过激减小概率

本质是一阶方法近似 TRPO 的信任域约束。

---

#### Q5：GAE 是什么？$\lambda$ 参数如何权衡 bias 和 variance？

$$\hat{A}_t^{GAE} = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$\lambda$ 是对不同时间步 TD 残差的指数加权：

- $\lambda = 0$：纯 TD，高偏差（依赖 $V$ 准确性），低方差
- $\lambda = 1$：蒙特卡洛，低偏差，高方差（累积所有随机性）
- $\lambda = 0.95$（典型值）：两者权衡

---

### 二、算法对比类

#### Q6：GRPO 相比 PPO 的核心改进是什么？代价是什么？

**核心改进**：用组内相对奖励替代 Critic 基线估计，消除 Value 网络，显存从 4 个模型降到 3 个。

**代价**：

|维度|PPO|GRPO|
|---|---|---|
|优势估计精度|高（GAE，时序精细）|中（组内粗粒度）|
|难度分布|均匀|有偏（难度偏差）|
|长度处理|GAE 自然处理|有长度偏差|
|极难/极易题|正常学习|梯度消失（Dead Zone）|

---

#### Q7：DPO 和 PPO 的本质区别是什么？

|维度|PPO|DPO|
|---|---|---|
|是否需要显式 RM|✓|✗（隐式编码）|
|是否需要 RL 循环|✓（on-policy 采样）|✗（直接监督）|
|奖励形式|标量 $r(x,y)$|隐式 $\beta\log\frac{\pi_\theta}{\pi_{ref}}$|
|在线 vs 离线|on-policy|offline|
|训练稳定性|难（超参多）|易（普通 Adam）|
|推理任务|强（可在线探索）|弱（无探索）|

**理论联系**：DPO 是 PPO 的闭式近似——KL 约束 RL 的最优解 $\pi^* \propto \pi_{ref}\exp(r/\beta)$ 反代入 Bradley-Terry，消去显式 RM，直接得到 DPO 损失。

---

#### Q8：DAPO 的四个改进分别针对 GRPO 的哪个具体问题？

|GRPO 缺陷|DAPO 改进|机制|
|---|---|---|
|熵坍塌（探索停止）|Clip-Higher（解耦 $\epsilon_{low}/\epsilon_{high}$）|放大上界给低概率 token 更多提升空间|
|Dead Zone（全对/全错无梯度）|Dynamic Sampling|过采样后过滤 pass_rate=0 或 1 的组|
|长度偏差（短回答权重虚高）|Token-Level Loss|损失按 token 数归一化，而非 sample 数|
|截断序列引入错误梯度|Overlong Filtering + Soft Punishment|截断样本 mask 掉；接近上限时软惩罚|

额外：$\beta=0$，彻底去掉 KL——推理任务有可验证奖励，KL 约束有害探索。

---

#### Q9：Dr.GRPO 发现了 GRPO 的什么理论问题？

GRPO 的组内标准化 $\hat{A}_i = (r_i - \mu) / \sigma$ 引入两类系统性偏差：

**偏差1（难度偏差）**：简单/困难题 $\sigma \to 0$，梯度权重 $\propto 1/\sigma \to \infty$，极端难度题权重比中等题高 2.25 倍。

**偏差2（长度偏差）**：$\hat{A}_i$ 广播到所有 token，长序列梯度贡献 $\propto L$，模型被推向生成更长回答。

**修正**：去掉 $\sigma$ 归一化 + per-token 损失归一化：

$$\hat{A}_i = r_i - \mu, \qquad \ell_{i,t} = \frac{1}{|y_i|}\min(\rho_{i,t}\hat{A}_i, \text{clip}(\cdot))$$

---

#### Q10：VAPO 解决了什么场景下的问题？

VAPO 针对**正确率极低的极难题**，此时 GRPO/DAPO 失效：

1. 组内大多数全错 → $\mu \approx 0, \sigma \approx 0$ → Dead Zone，无梯度
2. trajectory-level 奖励无法区分哪些 token 贡献了正确答案（Credit Assignment 缺失）

解法：轻量级 Value Model（仅正样本训练）+ 混合优势估计：

$$\hat{A}^{VAPO} = \lambda \cdot A^{GAE}(V) + (1-\lambda) \cdot \hat{A}^{group}$$

与 PPO 区别：Value Model 只在正样本上训练，避免 Critic 因负样本主导而难以收敛。

---

### 三、工程实现类

#### Q11：PPO 训练需要几个模型？如何降低显存压力？

标准 PPO 需要 4 个模型：Actor、Critic、Reference Policy、Reward Model。

|手段|效果|代价|
|---|---|---|
|Critic 和 Actor 共享底层参数|省 ~1× 显存|优化目标冲突，不稳定|
|Reward Model offload（CPU 推理）|节省 RM 显存|推理延迟增大|
|Reference Policy 量化（INT8/INT4）|节省 ref 显存|KL 计算有误差|
|换用 GRPO|去掉 Critic（4→3）|优势估计精度下降|
|换用 DAPO/RLVR|去掉 RM 和 Ref（4→1）|只适用于可验证任务|
|换用 DPO|完全离线，无 RM，无 RL|失去 on-policy 探索能力|

---

#### Q12：GRPO 中 $G$ 的大小如何影响训练？

DeepSeek-R1 第一阶段使用 $G=16$。

|$G$|优势估计|显存/计算|Dead Zone 风险|
|---|---|---|---|
|小（2~4）|高方差，基线不稳|低|高|
|中（8~16）|较稳定|中|中|
|大（32+）|低方差，基线稳定|高|低|

DAPO 的 Dynamic Sampling 通过过采样 $2G$ 再过滤，等效于提高有效 $G$。

---

#### Q13：如何判断 RLHF 训练是否发生了 Reward Hacking？

```
训练曲线信号：
  - RM 分数持续上升，但人工评估分数下降或停滞
  - KL 散度快速增大，超出预期范围
  - 生成长度异常增大（堆砌废话拿高分）
  - 熵快速下降（模式固化）

输出特征：
  - 回答变得冗长、重复
  - 出现固定的高分模板
  - 对抗性 token 或异常符号出现
```

**缓解手段**：增大 $\beta$；定期重训 RM；换用 RLVR；监控 KL 并设置早停阈值。

---

#### Q14：DPO 训练时 $y_l$ 的概率为什么会异常下降？如何修复？

**根因**：DPO 梯度同时增大 $y_w$ 概率并减小 $y_l$ 概率。当 $y_l$ 本身概率已很低时，继续减小导致 $\log\pi_\theta(y_l) \to -\infty$，隐式奖励异常偏大，破坏奖励相对大小。

|修复方案|机制|
|---|---|
|IPO|平方损失约束奖励差趋向固定常数，防止无界退化|
|SFT 正则化项|叠加 $y_w$ 的 NLL 损失，防止 $y_w$ 概率也下降|
|参考模型定期更新|$\pi_{ref}$ 跟踪 $\pi_\theta$ 历史版本（类似 Online DPO）|
|控制 $\beta$|$\beta$ 过小时梯度过强，适当增大可缓解|

---

### 四、原理推导类

#### Q15：推导 DPO 损失函数（6步）

**Step 1**：带 KL 约束的 RL 目标：

$$\max_{\pi} \mathbb{E}_{y \sim \pi}[r(x,y)] - \beta \mathbb{KL}[\pi | \pi_{ref}]$$

**Step 2**：展开 KL，整理为负 KL 散度形式：

$$= -\mathbb{KL}\left[\pi(y|x) ,\Big|, \pi_{ref}(y|x)\exp\left(\frac{r(x,y)}{\beta}\right) / Z(x)\right] + \log Z(x)$$

**Step 3**：KL $\geq 0$，等号成立时取最优策略解析解：

$$\pi^*(y|x) = \frac{\pi_{ref}(y|x)\exp(r(x,y)/\beta)}{Z(x)}$$

**Step 4**：反解奖励函数（用 $\pi_\theta$ 替代 $\pi^*$）：

$$r(x,y) = \beta\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} + \beta\log Z(x)$$

**Step 5**：代入 Bradley-Terry，$\log Z(x)$ 在差值中自动消去：

$$P(y_w \succ y_l|x) = \sigma\left(\beta\log\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)}\right)$$

**Step 6**：最大化对数似然，得 DPO 损失：

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)}\right)\right]$$

---

#### Q16：GRPO 的组内标准差归一化为何引入难度偏差？定量分析。

设题目真实正确率为 $p$，组内奖励 $r_i \in {0,1}$：

$$\mathbb{E}[\sigma_G] \approx \sqrt{p(1-p)}$$

有效梯度权重 $\propto 1/\sqrt{p(1-p)}$，极简单（$p=0.95$）和极困难（$p=0.05$）题权重比中等题（$p=0.5$）高 **2.25 倍**，且 $p \to 0/1$ 时权重趋向无穷。

这是系统性梯度扭曲，不是随机噪声。Dr.GRPO 修正：用 $\hat{A}_i = r_i - \mu$（不除以 $\sigma$），梯度大小仅由奖励绝对值决定。

---

#### Q17：REINFORCE 的 log-derivative trick 推导。

$$\nabla_\theta \mathbb{E}[R(y)] = \nabla_\theta \int R(y)\pi_\theta(y),dy = \int R(y)\nabla_\theta\pi_\theta(y),dy$$

$$= \int R(y)\cdot\pi_\theta(y)\cdot\nabla_\theta\log\pi_\theta(y),dy = \mathbb{E}\left[R(y)\cdot\nabla_\theta\log\pi_\theta(y)\right]$$

**意义**：将对概率分布的梯度转化为对 log 概率的梯度，可用 MC 采样直接估计，无需对整个状态空间求和。

---

### 五、开放设计类

#### Q18：在自动驾驶感知模型上应用 RLHF，如何设计？

**核心判断**：感知任务有可验证客观指标（mAP、3D IoU、ADE/FDE），更适合 **RLVR** 而非人工偏好 RM。

```
奖励函数设计：
  r = w1 * detection_reward          # IoU(pred, gt) > threshold
    + w2 * tracking_consistency_reward  # MOTA / MOTP
    - w3 * false_positive_penalty
    - w4 * latency_penalty              # max(0, infer_time - budget_ms)
```

**策略选型**：

- 感知模型输出是确定性预测（非自回归），无法直接用语言模型 RLHF 框架
- 方案A：将 backbone 接自回归 head（VLM），用 GRPO 优化 VQA 形式的感知问题
- 方案B（更务实）：DPO——收集"好检测 vs 坏检测"成对数据，规避 PPO 工程复杂度

---

#### Q19：PPO 训练中 Critic 收敛慢导致 Actor 学歪，怎么解决？

**根因**：训练初期 Value 估计误差大，GAE 计算出的优势函数不可靠，Actor 基于错误优势函数更新方向偏错。

解决方案（按优先级）：

1. **Critic 预热**：前 N 步只更新 Critic，不更新 Actor，直到 Value Loss 低于阈值
2. **增大 Critic 学习率**：通常设为 Actor LR 的 3~10 倍
3. **增大 Critic Loss 系数**：$\text{loss} = \mathcal{L}^{CLIP} + c_1 \mathcal{L}^V + c_2 H$，增大 $c_1$
4. **换 GRPO**：彻底去掉 Critic——大模型场景最彻底的解法
5. **Value Clipping**：对 Value 预测也做 clip，防止 Critic 更新过激

---

#### Q20：SimPO 为什么要做长度归一化？不做会有什么问题？

**不做的问题**：累积 log 概率天然 $\propto$ 序列长度，模型会学到"生成更长的 $y_w$"而非"生成更好的 $y_w$"，偏好数据中的长度分布成为 artifact 被学习。

**SimPO 的修正**：序列平均 log 概率剔除长度影响，模型必须提高**每个 token 的平均概率**：

$$\frac{1}{|y|}\log\pi_\theta(y|x) = \frac{1}{|y|}\sum_t \log\pi_\theta(y_t|x,y_{<t})$$

**附带好处**：长度已显式处理，不再需要 $\pi_{ref}$ 做归一化基准，SimPO 因此可彻底去掉参考模型。

---

## 总结

### 核心技术演进逻辑

```
问题1: 预训练目标 ≠ 人类偏好
  └─► 解法: RLHF（SFT + RM + PPO）

问题2: PPO 四模型显存过大 + 训练不稳定
  ├─► 解法A: 消除 Critic    → GRPO
  └─► 解法B: 消除 RM + RL   → DPO 系列

问题3: GRPO 在推理任务的四个缺陷
  ├─► 工程修复 → DAPO
  ├─► 理论修复 → Dr.GRPO
  └─► 稀疏奖励修复 → VAPO

问题4: 神经 RM 的 reward hacking
  └─► 解法: RLVR（可验证规则奖励）

问题5: RL 探索成本高 + 离线 DPO 分布偏移
  └─► 解法: Online DPO / Test-Time Compute Scaling
```

### 各方法核心公式一览

|方法|核心公式|
|---|---|
|RM|$\mathcal{L} = -\mathbb{E}[\log\sigma(r(y_w) - r(y_l))]$|
|PPO|$\mathcal{L} = \mathbb{E}[\min(\rho\hat{A},;\text{clip}(\rho,1-\epsilon,1+\epsilon)\hat{A})]$|
|GAE|$\hat{A}_t = \sum_l (\gamma\lambda)^l \delta_{t+l},;\delta_t=r_t+\gamma V_{t+1}-V_t$|
|GRPO|$\hat{A}_i = (r_i - \mu_G)/\sigma_G$|
|DPO|$\mathcal{L} = -\mathbb{E}[\log\sigma(\beta\log\frac{\pi(y_w)}{\pi_{ref}(y_w)} - \beta\log\frac{\pi(y_l)}{\pi_{ref}(y_l)})]$|
|SimPO|$\mathcal{L} = -\mathbb{E}[\log\sigma(\frac{\beta}{|
|DAPO|$\text{clip}(\rho, 1-\epsilon_{low}, 1+\epsilon_{high})$，$\beta=0$，token-level loss|
|Dr.GRPO|$\hat{A}_i = r_i - \mu_G$（去 std），per-token 归一化|
|VAPO|$\hat{A}^{VAPO} = \lambda A^{GAE} + (1-\lambda)\hat{A}^{group}$|

### 面试高频考点清单

- [ ] RLHF 三阶段各自解决什么问题
- [ ] Bradley-Terry 模型推导，为什么选它
- [ ] PPO-Clip 机制与 TRPO 的关系
- [ ] GAE 的 bias-variance 权衡，$\lambda$ 的物理意义
- [ ] DPO 完整推导（KL 约束 RL → 最优解析解 → 消去 $Z(x)$）
- [ ] GRPO vs PPO 核心区别与代价
- [ ] GRPO 四大缺陷及 DAPO/Dr.GRPO 的对应修复
- [ ] Reward Hacking 的识别信号与缓解手段
- [ ] $y_l$ 概率退化问题根因与修复方案
- [ ] RLVR vs 神经 RM 的适用场景判断
