---
tags: [RLHF, 强化学习, LLM对齐, 策略优化]
aliases: [RLHF主文档, 人类反馈强化学习]
created: 2025-03-09
status: complete
---

## RLHF 主文档

> **定义**：RLHF（Reinforcement Learning from Human Feedback）——用人类偏好信号通过强化学习对语言模型进行对齐训练的技术框架。

子文档：[[SFT（Supervised Fine-Tuning）深度解析]] · [[RM（Reward Model） 深度介绍]] · [[PPO（Proximal Policy Optimization）深度解析]]

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

| 方法 | Value模型 | Clip | 长度偏差修正 | KL | 适用场景 |
|---|---|---|---|---|---|
| PPO | ✓ 全程 | 对称 | GAE | ✓ | 对话对齐 |
| GRPO | ✗ | 对称 | ✗ | ✓ | 推理中等难度 |
| DAPO | ✗ | 解耦 | Token-level | ✗ | 推理大规模 |
| Dr.GRPO | ✗ | 对称 | Per-token归一化 | 可选 | 推理理论修正 |
| VAPO | ✓ 轻量 | 解耦 | GAE+组混合 | ✓ | 推理极难题 |
| DPO | ✗ | 无 | ✗ | 隐式 | 对话偏好 |
| SimPO | ✗ | 无 | 内置 length norm | ✗ | 对话无参考 |

#### 核心结论速查

| 结论 | 一句话 |
|---|---|
| RLHF 为何需要 KL 惩罚 | RM 是有噪声近似，无约束策略会 reward hack |
| DPO 为何不需要 RM | 最优策略有解析解，RM 被隐式编码进策略比值 |
| GRPO 为何去掉 Critic | 组内相对奖励作为基线，无需 Value 网络估计 |
| DAPO 为何 $\beta=0$ | 推理任务用规则验证器，策略应大幅偏离初始 |
| Dr.GRPO 为何去 std | std 归一化引入难度偏差和长度偏差，是系统性错误 |
| DeepSeek-R1 为何不用神经 RM | 大规模 RL 中神经 RM 容易被 reward hacking 攻击 |

---

### 统一符号定义

| 符号 | 含义 |
|---|---|
| $\pi_\theta$ | 待优化策略（语言模型） |
| $\pi_{ref}$ | 参考策略（SFT 模型，冻结） |
| $r(x,y)$ | 奖励函数（RM 或规则） |
| $\mathcal{D}$ | 偏好数据集 $(x, y_w, y_l)$ |
| $\beta$ | KL 惩罚系数 |
| $y_w$ | preferred 回答（赢家） |
| $y_l$ | rejected 回答（输家） |

**统一核心优化目标（所有方法的出发点）：**

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta} \left[ r(x,y) \right] - \beta \cdot \mathbb{KL}\left[\pi_\theta \| \pi_{ref}\right]$$

---

## 第一章 RLHF 概述

### 1.1 与 LLM 的位置关系

RLHF 是 LLM **训练流程的后处理阶段**，不改变模型架构，只改变权重：

```
预训练（Pre-training）
    目标：next-token prediction，学习语言知识
    数据：万亿级无标注文本
    产出：Base Model（有能力，但不对齐）
         │
         ▼
RLHF Post-training
    ├─ SFT   → 指令跟随能力，构造初始策略 π₀  → 详见 [[SFT_深度笔记]]
    ├─ RM    → 学习人类偏好，提供奖励信号       → 详见 [[RM_深度笔记]]
    └─ PPO   → 强化学习策略优化                → 详见 [[PPO_深度笔记]]
         │
         ▼
Chat Model / Instruct Model（对齐后）
```

### 1.2 核心问题：预训练目标与人类偏好的错位

| 预训练目标 | 人类偏好目标 |
|---|---|
| 最大化 token 预测似然 | 有帮助、无害、诚实 |
| 对所有文本一视同仁 | 区分好 / 坏回答 |
| 可完全自动化 | 需要人类主观判断 |

Base Model 的典型问题：
- 对指令不响应，而是续写文本
- 生成有害、虚假、无用内容
- 无法维持对话格式和角色设定

### 1.3 RLHF 的三个阶段
#### 1.3.1 SFT
[SFT（Supervised Fine-Tuning）深度解析](SFT（Supervised%20Fine-Tuning）深度解析.md)
#### 1.3.2 RM
[RM（Reward Model） 深度介绍](RM（Reward%20Model）%20深度介绍.md)
#### 1.3.3 PPO
[PPO（Proximal Policy Optimization）深度解析](PPO（Proximal%20Policy%20Optimization）深度解析.md)

### 1.4 已知问题全景

| 问题 | 描述 | 主要改进方案 |
|---|---|---|
| Reward Hacking | 策略利用 RM 漏洞，高分低质 | KL 惩罚、定期重训 RM、RLVR |
| 显存开销 | 四模型同时加载 | DPO / GRPO / DAPO |
| 训练不稳定 | PPO 对超参极敏感 | GRPO / DAPO / Dr.GRPO |
| 标注偏差 | 标注者偏好不一致 | 多标注者 + 不确定性建模 |
| 分布偏移 | 离线数据与生成分布不匹配 | Online RLHF / Online DPO |
| 稀疏奖励 | 长推理链信用分配困难 | PRM / VAPO |

---

## 第二章 策略优化技术演进（2023–2025）

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

### 2.1 GRPO（Group Relative Policy Optimization，2024，DeepSeekMath）

对同一 prompt 采样 $G$ 个回答，用**组内相对奖励**作为基线，彻底消除 Value 网络：

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$$

$$\mathcal{L}^{GRPO} = -\frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \min\left(\rho_{i,t}\hat{A}_i,\; \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_i\right) + \beta\,\mathbb{KL}[\pi_\theta\|\pi_{ref}]$$

**四大缺陷**：

| 缺陷 | 根因 |
|---|---|
| 梯度消失（Dead Zone） | 全对/全错 → std=0 → $\hat{A}$=0 |
| 长度偏差 | Sample 级损失，长/短回答权重相同 |
| 熵坍塌 | 对称 clip 抑制低概率 token |
| 截断干扰 | 被截断序列参与损失 |

### 2.2 DeepSeek-R1（2025.01）

**R1-Zero**：直接在 Base Model 上用 GRPO，跳过 SFT。AIME 2024 pass@1：$15.6\% \to 71.0\%$，涌现出 self-verification、reflection 等行为。

**R1 四阶段流水线**：

```
阶段1: Cold Start SFT      → 数千条高质量长CoT，解决可读性问题
阶段2: 推理导向 RL          → GRPO + 可验证奖励 + 语言一致性奖励
阶段3: 拒绝采样 + SFT       → 筛选高质量推理链 + 通用能力数据
阶段4: 全场景 RL            → GRPO + 推理 + 帮助性/无害性偏好
```

奖励函数：纯规则（准确性 + 格式），无神经 RM。

### 2.3 DPO 系列（消除显式 RM）

**DPO（2023，Stanford）** 理论推导：KL 约束 RL 的最优解析解为

$$\pi^*(y|x) = \frac{\pi_{ref}(y|x)\exp(r/\beta)}{Z(x)}$$

反解 $r$ 代入 Bradley-Terry，$Z(x)$ 自动消去，得：

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)} - \beta \log \frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)}\right)\right]$$

| 变体 | 核心改进 |
|---|---|
| IPO（2023） | 平方损失替代 log σ，防止 $y_l$ 概率退化 |
| KTO（2024） | 基于前景理论，支持单条标注数据 |
| SimPO（2024） | 去掉参考模型，内置长度归一化，加 margin $\gamma$ |

### 2.4 REINFORCE++（2025.01）

全局归一化替代 per-question 局部归一化，去掉 Clip，Mini-batch KL Loss：

$$\hat{A}_i^{global} = \frac{r_i - \mu_{global}}{\sigma_{global}}$$

### 2.5 DAPO（2025.03，ByteDance Seed）

在 Qwen2.5-32B 上 AIME 2024 达 **50分**，仅用 DeepSeek-R1 **50% 训练步数**。

| 改进 | 根因 | 机制 |
|---|---|---|
| Clip-Higher | 熵坍塌 | $\epsilon_{low}=0.2$，$\epsilon_{high}=0.28$，解耦上下界 |
| Dynamic Sampling | Dead Zone | 过采样 2G，过滤 pass_rate=0 或 1 的组 |
| Token-Level Loss | 长度偏差 | $\mathcal{L} = -\frac{\sum_i\sum_t \ell_{i,t}}{\sum_i |y_i|}$ |
| Overlong Filter | 截断干扰 | 截断样本 mask；过长软惩罚 |
| $\beta=0$ | KL 有害探索 | 规则验证器已足够约束 |

### 2.6 Dr.GRPO（2025.03，"GRPO Done Right"）

**两类系统性偏差**：

- **难度偏差**：$\sigma \propto \sqrt{p(1-p)}$，极端难度题权重比中等题高 2.25 倍
- **长度偏差**：$\hat{A}_i$ 广播到所有 token，梯度贡献 $\propto L$

**修正**：$\hat{A}_i = r_i - \mu$（去 std），per-token 归一化损失。

### 2.7 VAPO（2025.04，ByteDance + 阿里）

针对正确率极低的极难题，引入轻量级 Value Model（仅正样本训练）+ 混合优势估计：

$$\hat{A}_t^{VAPO} = \lambda_{GAE} \cdot A_t^{GAE}(V_\phi) + (1-\lambda_{GAE}) \cdot \hat{A}_i^{group}$$

---

## 第三章 前沿方向（2025）

### 3.1 RLVR（Reinforcement Learning with Verifiable Rewards）

用规则验证器替代神经 RM，从根源消除 reward hacking：

| 场景 | 验证器 |
|---|---|
| 数学 | 答案精确匹配 |
| 代码 | 单元测试通过率 |
| 逻辑 | 形式化验证器 |

### 3.2 ORM vs PRM

| | ORM | PRM |
|---|---|---|
| 奖励时机 | 序列末尾 | 每步推理步骤 |
| 优点 | 简单，可自动化 | 密集信号，Credit Assignment 精准 |
| 缺点 | 信用分配困难 | 标注成本高 |

当前趋势：LLM 自动生成 PRM 标注（MCTS 或自洽性筛选）+ 结合 GRPO 提供 per-step 奖励。

### 3.3 Online DPO

持续采样更新，将 DPO 变为 on-policy：

```
loop: π_θ 采样 → RM 打分 → 构造 (y_w, y_l) → DPO 更新 π_θ
```

### 3.4 Test-Time Compute Scaling

[Test-Time Compute Scaling 介绍](Test-Time%20Compute%20Scaling%20介绍.md)

| 方法               | 思路            |
| ---------------- | ------------- |
| Best-of-N        | 多次采样，取最高 RM 分 |
| Beam Search + RM | RM 引导束搜索      |
| MCTS             | 蒙特卡洛树搜索引导解码   |

---

## 第四章 面试题（RLHF 整体）

### Q1：RLHF 三阶段各自解决什么问题？

- **SFT**：预训练模型无法跟随指令，SFT 提供初始策略 $\pi_0$ 和基础生成能力
- **RM**：人类直接生成高质量回答成本极高，RM 将成对偏好编码为可微分标量信号
- **PPO**：有了可微分奖励，才能用梯度方法持续优化

三阶段缺一不可。跳过 SFT 直接 RL 会有 cold start 问题；跳过 RM 直接让人打分无法扩展。

> R1-Zero 跳过 SFT 成功的原因：推理任务有可验证奖励，不依赖 RM 质量；代价是可读性差，这是 R1 再加 Cold Start SFT 的原因。

### Q2：DPO 和 PPO 的本质区别？

| 维度 | PPO | DPO |
|---|---|---|
| 显式 RM | ✓ | ✗（隐式编码） |
| RL 循环 | ✓（on-policy） | ✗（直接监督） |
| 训练稳定性 | 难 | 易 |
| 推理任务 | 强（可探索） | 弱（无探索） |

DPO 是 PPO 的**闭式近似**：KL 约束 RL 的最优解解析形式反代入 Bradley-Terry，消去显式 RM。

### Q3：如何判断训练是否发生了 Reward Hacking？

```
训练曲线：RM 分↑ 但人工评估分↓ / KL 散度快速增大 / 生成长度异常增大 / 熵快速下降
输出特征：冗长重复 / 固定高分模板 / 对抗性 token
```

缓解：增大 $\beta$；定期重训 RM；换用 RLVR；监控 KL 并设置早停阈值。

### Q4：GRPO 四大缺陷及对应修复？

| 缺陷 | DAPO 修复 | Dr.GRPO 修复 |
|---|---|---|
| Dead Zone | Dynamic Sampling | — |
| 长度偏差 | Token-Level Loss | per-token 归一化 |
| 熵坍塌 | Clip-Higher | — |
| 难度偏差 | Dynamic Sampling（间接） | 去 std 归一化（直接） |
| 截断干扰 | Overlong Filter | — |

---

## 总结

### 核心技术演进逻辑

```
问题1: 预训练目标 ≠ 人类偏好
  └─► RLHF（SFT + RM + PPO）

问题2: PPO 四模型显存 + 不稳定
  ├─► 消除 Critic    → GRPO
  └─► 消除 RM + RL   → DPO 系列

问题3: GRPO 四大缺陷
  ├─► 工程修复 → DAPO
  ├─► 理论修复 → Dr.GRPO
  └─► 稀疏奖励 → VAPO

问题4: 神经 RM 的 reward hacking
  └─► RLVR（可验证规则奖励）

问题5: 离线 DPO 分布偏移
  └─► Online DPO / Test-Time Compute Scaling
```

### 各方法核心公式

| 方法 | 核心公式 |
|---|---|
| RM | $\mathcal{L} = -\mathbb{E}[\log\sigma(r(y_w) - r(y_l))]$ |
| PPO | $\mathcal{L} = \mathbb{E}[\min(\rho\hat{A},\;\text{clip}(\rho,1-\epsilon,1+\epsilon)\hat{A})]$ |
| GAE | $\hat{A}_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$ |
| GRPO | $\hat{A}_i = (r_i - \mu_G)/\sigma_G$ |
| DPO | $\mathcal{L} = -\mathbb{E}[\log\sigma(\beta\log\frac{\pi(y_w)}{\pi_{ref}(y_w)} - \beta\log\frac{\pi(y_l)}{\pi_{ref}(y_l)})]$ |
| SimPO | $\mathcal{L} = -\mathbb{E}[\log\sigma(\frac{\beta}{|y_w|}\log\pi(y_w) - \frac{\beta}{|y_l|}\log\pi(y_l) - \gamma)]$ |
| DAPO | $\text{clip}(\rho,1-\epsilon_{low},1+\epsilon_{high})$，$\beta=0$，token-level loss |
| Dr.GRPO | $\hat{A}_i = r_i - \mu_G$（去 std），per-token 归一化 |
| VAPO | $\hat{A}^{VAPO} = \lambda A^{GAE} + (1-\lambda)\hat{A}^{group}$ |
