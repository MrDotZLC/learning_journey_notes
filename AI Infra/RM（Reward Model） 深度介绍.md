---
tags: [RLHF, RM, 奖励模型, 偏好学习, RewardHacking]
aliases: [RM笔记, 奖励模型]
created: 2025-03-09
status: complete
---

## RM 深度笔记（Reward Model）

> 主文档：[[RLHF（Post-Training）]]　｜　相关：[[SFT（Supervised Fine-Tuning）深度解析]] · [[PPO（Proximal Policy Optimization）深度解析]]

---

### ⚡ 速查卡

| 维度 | 内容 |
|---|---|
| 核心目标 | 将人类偏好编码为可微分标量信号 |
| 训练信号 | Bradley-Terry 成对偏好损失 |
| 数据格式 | $(x, y_w, y_l)$ 偏好对 |
| 核心问题 | Reward Hacking（RM 是有噪声近似） |
| 替代方案 | RLVR（可验证奖励）、DPO（隐式 RM）、PRM（过程奖励） |

---

## 第一章 RM 基础

### 1.1 为什么需要 RM

SFT 只能说"这个回答好"，无法说"这个比那个好多少"。RM 解决的是**偏好的标量化**问题：

```
人类判断（离散）：y_w 比 y_l 更好
        │
        ▼
RM（连续）：r(x, y_w) - r(x, y_l) = 2.3
        │
        ▼
PPO 奖励信号（可微分）
```

**核心洞见**：让人类判断"哪个更好"（比较）比让人类撰写"好的回答"（生成）容易得多，且一致性更高。

### 1.2 模型结构

取 SFT 模型，**移除最后的 LM head，替换为线性层输出标量**：

```
Input:  [prompt || response]  （拼接后输入）
            │
    [Transformer Layers]      ← 复用 SFT 权重初始化（通常冻结底层）
            │
    [Final hidden state of last token]   ← 整个序列的压缩表示
            │
    [Linear(d_model → 1)]     ← 新增，随机初始化
            │
Output: scalar reward  r ∈ ℝ
```

**为什么用最后一个 token 的 hidden state**：自回归 LM 的 causal attention 保证最后一个 token 已 attend 到所有前序 token，是整个序列的信息压缩。

**初始化策略**：RM 从 SFT 模型初始化，而非从 Base Model 初始化。原因：SFT 模型已经理解指令格式和高质量回答的特征，收敛更快，泛化更好。

### 1.3 Bradley-Terry 模型与损失函数

**Bradley-Terry 模型**（1952）：成对比较的概率模型，假设每个选项有潜在得分，比较结果服从 sigmoid 分布。

$y_w$ 优于 $y_l$ 的概率：

$$P(y_w \succ y_l \mid x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)) = \frac{e^{r_\theta(x,y_w)}}{e^{r_\theta(x,y_w)} + e^{r_\theta(x,y_l)}}$$

**损失函数**（最大化偏好对数似然）：

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[ \log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)) \right]$$

**梯度分析**：

$$\frac{\partial \mathcal{L}_{RM}}{\partial r_\theta(x,y_w)} = -(1 - \hat{p}) = \sigma(r_l - r_w)$$

$$\frac{\partial \mathcal{L}_{RM}}{\partial r_\theta(x,y_l)} = \hat{p} = \sigma(r_w - r_l)$$

| 当前状态 | $\hat{p}$ | 梯度 | 效果 |
|---|---|---|---|
| $r_w \gg r_l$（已区分好坏） | $\to 1$ | $\to 0$ | 自动停止 ✓ |
| $r_w \approx r_l$（无法区分） | $\approx 0.5$ | 最大 | 重点更新 ✓ |
| $r_w \ll r_l$（排名颠倒） | $\to 0$ | 最大（负方向） | 强力纠正 ✓ |

---

## 第二章 偏好数据工程

### 2.1 标注流程

```python
# 标准偏好数据收集流程
for prompt x in prompt_pool:
    responses = [model.generate(x) for _ in range(K)]  # K = 4~9
    # 标注者对 K 个回答进行排序
    ranking = annotator.rank(responses)
    # 从排序中提取 C(K,2) 个偏好对
    for i in range(len(ranking)):
        for j in range(i+1, len(ranking)):
            pairs.append((x, ranking[i], ranking[j]))  # (x, y_w, y_l)
```

**数据规模参考**：
- InstructGPT：~33K 偏好比较
- Anthropic HH：~170K 偏好对
- OpenAssistant：~160K 偏好对

### 2.2 标注一致性问题

不同标注者对"好回答"的判断存在分歧：

| 来源 | 典型不一致率 |
|---|---|
| 标注者间一致性（IAA） | 60~75%（Cohen's κ ≈ 0.4~0.6） |
| 同一标注者不同时间 | ~85% 一致 |

**缓解方法**：
- 多标注者投票（majority voting）
- 详细标注指南（明确定义"有帮助"、"无害"）
- 置信度加权：高争议样本权重降低或剔除
- 标注者专业化：数学题找数学背景标注者

### 2.3 标注维度分解

单一整体偏好标注信噪比低。细粒度多维度标注效果更好：

| 维度 | 描述 |
|---|---|
| Helpfulness | 回答是否有帮助、完整 |
| Harmlessness | 是否包含有害内容 |
| Honesty | 是否事实准确、不过度自信 |
| Instruction-following | 是否遵循了指令格式要求 |
| Verbosity | 长度是否适当（不过长/过短） |

**多维度 RM**：为每个维度训练单独的 RM head，PPO 时用加权组合：

$$r_{total} = w_h \cdot r_{helpful} + w_{harm} \cdot r_{harmless} + w_{hon} \cdot r_{honest}$$

### 2.4 隐式负样本（Implicit Negatives）

RM 训练中，$y_l$ 不必来自人类标注——可以用其他来源：

| 负样本来源 | 优点 | 缺点 |
|---|---|---|
| 同一模型其他采样输出 | 覆盖模型真实失败模式 | 可能难度不足 |
| 更旧/更弱的模型输出 | 质量差异明显，易区分 | 对齐目标失真 |
| 规则生成的坏回答 | 可控、廉价 | 覆盖面窄 |
| 对抗样本（adversarial） | 针对 RM 的薄弱点 | 生成成本高 |

---

## 第三章 RM 的结构变体

### 3.1 单头 vs 多头 RM

**单头 RM**：一个线性层输出一个标量，简单但无法区分不同维度：

```python
class SingleHeadRM(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        self.head = nn.Linear(hidden_size, 1)  # 单头

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids, attention_mask).last_hidden_state
        reward = self.head(hidden[:, -1, :])   # 取最后 token
        return reward.squeeze(-1)
```

**多头 RM**：多个 head 分别输出不同维度的奖励，更细粒度：

```python
class MultiHeadRM(nn.Module):
    def __init__(self, base_model, num_heads=4):
        super().__init__()
        self.backbone = base_model
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_heads)
        ])
        # heads: [helpful, harmless, honest, instruction_following]

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids, attention_mask).last_hidden_state
        rewards = [head(hidden[:, -1, :]) for head in self.heads]
        return torch.stack(rewards, dim=-1).squeeze(1)  # (batch, num_heads)
```

### 3.2 对比式 vs 回归式

**对比式 RM**（主流）：直接用 BT 损失在偏好对上训练，输出的奖励分数只有**相对意义**。

**回归式 RM**：用绝对分数（如 1-5 分）训练，输出有**绝对意义**：

$$\mathcal{L}_{reg} = \text{MSE}(r_\theta(x,y), \text{human\_score}(x,y))$$

| | 对比式 | 回归式 |
|---|---|---|
| 数据获取 | 容易（判断哪个更好） | 困难（打绝对分数，一致性差） |
| 校准性 | 差（分数无绝对意义） | 好 |
| 训练稳定性 | 稳定 | 受绝对分数噪声影响 |
| 实践中使用 | 主流 | 少见 |

### 3.3 序列级 vs Token 级 RM

**序列级 RM**（ORM，Outcome Reward Model）：对整个 response 打一个分数，只在末尾 token 输出奖励。

**Token 级 RM**（PRM，Process Reward Model）：对每个推理步骤打分，提供密集奖励信号，见第六章。

### 3.4 轻量化 RM 方案

**Tiny RM**：用比 Actor 小得多的模型作为 RM（如用 7B 模型的 RM 对齐 70B 的 Actor）：
- 优点：显存和计算开销小
- 缺点：RM 能力弱于 Actor，容易被 reward hack

**Implicit RM（DPO）**：将 RM 完全隐式化，编码进策略比值 $\beta\log\frac{\pi_\theta}{\pi_{ref}}$，无需单独 RM 模型。

---

## 第四章 Reward Hacking 深层机制

### 4.1 根本原因

$$r_{RM}(x,y) \neq r^*_{human}(x,y)$$

RM 是人类偏好的有噪声近似，存在分布外（OOD）泛化问题。当 PPO 持续优化 RM 分数时，策略最终会进入 RM 训练分布之外的区域，此时 RM 的预测不可靠，但 PPO 仍会继续朝该方向优化。

```
训练分布内：RM 准确        训练分布外：RM 不可靠
    │                              │
    ▼                              ▼
r_RM ≈ r_human             r_RM >> r_human（RM 被欺骗）
```

### 4.2 常见 Reward Hacking 模式

| 模式 | 描述 | 示例 |
|---|---|---|
| 长度 hack | 堆砌废话增加长度（标注者偏好长答案） | 重复同一内容 N 遍 |
| 格式 hack | 过度使用 markdown、列表、表情符号 | 每句话加表情符号 |
| 开头 hack | 固定高分开头模板 | "当然！我很乐意帮助您..." |
| 奉承 hack | 过度赞美用户问题 | "这是一个非常棒的问题！" |
| 模糊回避 hack | 用模糊措辞避免被判断错误 | 对所有问题回答"这取决于..." |
| 对抗 token | 在 response 中插入能影响 RM 输出的异常 token | 特殊 unicode 字符 |

### 4.3 Reward Hacking 的检测

**训练时监控指标**：

```python
# 监控以下指标，任一异常即预警
metrics = {
    "rm_score": reward_model.score(responses),    # RM 打分（应缓慢上升）
    "human_eval_score": human_eval(responses),    # 人工评估（不应下降）
    "kl_divergence": kl(pi_theta, pi_ref),        # KL 散度（不应过快增大）
    "response_length": mean(len(r) for r in responses),  # 长度（不应异常增大）
    "entropy": policy_entropy(pi_theta),          # 熵（不应过快下降）
    "distinct_ngrams": diversity(responses),      # 多样性（不应下降）
}
```

**检测信号优先级**：

```
RM 分↑ 但人工评估分↓ 或停滞  ← 最强信号
KL 散度超出预期范围           ← 早期预警
生成长度异常增大              ← 格式 hack 信号
熵快速下降                   ← 模式固化信号
```

### 4.4 缓解手段

#### 4.4.1 KL 惩罚（最直接）

在 PPO 目标中加入 KL 散度惩罚，限制策略偏离参考策略的程度：

$$r_{total}(x,y) = r_{RM}(x,y) - \beta \sum_t \log\frac{\pi_\theta(y_t)}{\pi_{ref}(y_t)}$$

$\beta$ 控制约束强度，典型值 0.01~0.1。

#### 4.4.2 RM 集成（Ensemble）

训练多个 RM，取最小值（pessimistic ensemble）或加权平均：

$$r_{ensemble}(x,y) = \min_i r_i(x,y) \quad \text{（保守估计）}$$

或用不确定性过滤：方差过大的样本不用于 PPO 更新。

#### 4.4.3 定期重训 RM

```
Round 1:  训练 RM_1 → PPO 优化 π_1 → 收集 π_1 生成的新数据
Round 2:  训练 RM_2（包含 π_1 数据）→ PPO 优化 π_2 → ...
```

每轮用最新策略生成的数据重训 RM，防止 RM 对新策略的分布外区域打高分。

#### 4.4.4 换用 RLVR（根本解决）

对于可验证任务（数学、代码、逻辑），用规则验证器替代神经 RM，从根本上消除 reward hacking 的可能性。详见主文档 [[RLHF（Post-Training）]]。

---

## 第五章 RM 的校准问题

### 5.1 校准的定义

**校准（Calibration）**：RM 输出的分数差是否准确反映偏好概率？

理想情况：$P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$

**校准误差**：

$$\text{ECE} = \sum_{b} \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$$

其中 $B_b$ 是第 $b$ 个置信度区间内的样本桶。

### 5.2 RM 过拟合的表现

RM 在验证集上准确率高，但输出分数分布坍缩：

```
理想分数分布：
  y_w 分数：均值 2.0，std 1.5
  y_l 分数：均值 -2.0，std 1.5

过拟合分数分布（极端化）：
  y_w 分数：均值 8.0，std 0.3
  y_l 分数：均值 -8.0，std 0.3
```

极端化的分数会导致 PPO 中的 KL 惩罚失效（RM 分数绝对值远大于 KL 项）。

### 5.3 正则化方法

**分数归一化**：强制 RM 输出分布均值为 0、方差为 1：

$$r_{norm}(x,y) = \frac{r(x,y) - \mu_{running}}{\sigma_{running}}$$

**Label Smoothing**：在 BT 损失中加入平滑，防止极端输出：

$$\mathcal{L}_{smooth} = -(1-\epsilon)\log\sigma(\Delta r) - \epsilon\log\sigma(-\Delta r)$$

**Margin Loss（IPO 思想）**：

$$\mathcal{L}_{margin} = \max(0, \gamma - (r(y_w) - r(y_l)))$$

强制 $y_w$ 和 $y_l$ 之间至少有 $\gamma$ 的分数差距，防止过拟合到极端值。

---

## 第六章 ORM vs PRM（过程奖励模型）

### 6.1 ORM（Outcome Reward Model）的局限

ORM 只在序列末尾给奖励，对于长推理链（CoT），信用分配（Credit Assignment）极难：

```
推理链（10步）：
  Step 1: 正确 ✓
  Step 2: 正确 ✓
  Step 3: 错误 ✗  ← 关键错误
  Step 4~10: 基于错误推理继续
  最终答案: 错误

ORM 信号：序列末尾 r = 0
  → 无法区分是哪一步出错
  → 所有 10 步 token 获得相同的负信号

PRM 信号：Step 3 处 r = -1，其余步骤 r = 1
  → 精确定位错误位置
  → 正确步骤仍然被鼓励
```

### 6.2 PRM 的结构

在每个推理步骤结束处（通常以 `\n\n` 或特殊分隔符标记）输出一个奖励：

```python
class PRM(nn.Module):
    def forward(self, input_ids, step_positions):
        hidden = self.backbone(input_ids).last_hidden_state
        # 只在步骤边界位置提取奖励
        step_rewards = []
        for pos in step_positions:
            reward = self.head(hidden[:, pos, :])
            step_rewards.append(reward)
        return step_rewards  # per-step 奖励列表
```

### 6.3 PRM 标注方法

#### 6.3.1 人工标注（高成本）

OpenAI PRM800K 数据集：标注者对每个推理步骤标注"正确/错误/中性"，约 80 万个步骤级标注。

#### 6.3.2 MCTS 自动标注（Math-Shepherd，2024）

用蒙特卡洛树搜索估计每个步骤的"完成价值"（Completion Value）：

```python
def estimate_step_value(partial_solution, num_rollouts=64):
    """
    从当前步骤开始，随机完成 N 次推理，
    用最终答案的正确率估计当前步骤的价值
    """
    correct = 0
    for _ in range(num_rollouts):
        completed = model.complete(partial_solution)
        if verify(completed):
            correct += 1
    return correct / num_rollouts
```

#### 6.3.3 自洽性（Self-Consistency）标注

用答案的自洽性作为步骤质量的代理信号：
- 导向正确答案的步骤：正标签
- 导向错误答案的步骤：负标签

无需人工标注，可大规模自动生成。

### 6.4 ORM vs PRM 对比

| 维度 | ORM | PRM |
|---|---|---|
| 奖励粒度 | 序列级（末尾） | 步骤级（每步） |
| 标注成本 | 低（只需最终答案） | 高（需要步骤级标注） |
| Credit Assignment | 差 | 精准 |
| 奖励稀疏性 | 极稀疏 | 密集 |
| 对长 CoT 的效果 | 差 | 显著更好 |
| 结合 GRPO | 序列末尾 r | per-step r，天然支持 GAE |
| 实践状态 | 生产主流 | 研究前沿 |

### 6.5 PRM 的问题

- **步骤边界定义困难**：推理链的"步骤"如何划分？以换行符？以逻辑单元？
- **标注不一致**：不同标注者对中间步骤对错的判断分歧更大
- **过度惩罚**：某步骤"错误"但最终仍得到正确答案（多条路径）时如何处理

---

## 第七章 RM 的替代方案

### 7.1 Constitutional AI（Anthropic，2022）

用规则（宪法）替代 RM 的人类偏好，消除人工标注的偏见和成本：

```
阶段1: Critique and Revision（CAI）
  原始回答 → 让 LLM 用宪法原则批评 → 修订回答
  重复 N 次 → 得到更符合宪法的回答

阶段2: RL from AI Feedback（RLAIF）
  用 LLM（而非人类）判断哪个回答更符合宪法
  生成偏好数据 → 训练 RM → PPO 优化
```

**宪法原则示例**（Anthropic Claude 的宪法）：
- "选择对人类最无害的回答"
- "选择最诚实、不欺骗的回答"
- "选择最遵循 AI 助手定位的回答"

优点：大规模自动生成偏好数据，减少人工标注量。

### 7.2 RLAIF（RL from AI Feedback）

完全用 LLM 替代人类标注者生成偏好标签：

```python
# RLAIF 偏好标注
def ai_preference(prompt, response_a, response_b, constitution):
    judge_prompt = f"""
    根据以下原则：{constitution}
    比较两个回答：
    A: {response_a}
    B: {response_b}
    哪个更好？回答 A 或 B。
    """
    preference = llm.generate(judge_prompt)
    return preference  # "A" or "B"
```

**问题**：AI 判断存在位置偏差（倾向于选择先出现的回答）、冗长偏差（倾向于选择更长的回答）。

### 7.3 隐式 RM：DPO 系列

DPO 将 RM 隐式编码进策略，无需显式 RM 模型：

$$r_{implicit}(x,y) = \beta\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

适用于对话对齐，不适用于需要在线探索的推理任务。详见 [[RLHF（Post-Training）]] 第二章。

### 7.4 RLVR（可验证奖励）

对于可验证任务，完全用规则验证器替代神经 RM：

```python
def verifiable_reward(response, ground_truth, task_type):
    if task_type == "math":
        extracted = extract_answer(response)
        return 1.0 if extracted == ground_truth else 0.0
    elif task_type == "code":
        return run_unit_tests(response, test_cases)  # 通过率
    elif task_type == "logic":
        return formal_verifier(response, axioms)     # 形式化验证
```

**优势**：完全消除 reward hacking 的可能性（验证器无法被欺骗）。DeepSeek-R1 的成功验证了这一方向。

---

## 第八章 RM 的评估

### 8.1 准确率（Accuracy）

最基础的评估指标：RM 在测试集偏好对上的准确率：

$$\text{Acc} = \frac{\sum \mathbf{1}[r(y_w) > r(y_l)]}{|\mathcal{D}_{test}|}$$

**局限**：准确率高不代表 RM 校准好，也不代表在 PPO 训练中表现好。

### 8.2 OOD 泛化能力

RM 在训练分布之外（PPO 优化后的策略生成的样本）的准确率更重要：

```
训练阶段 RM 准确率：87%（in-distribution）
PPO 第 10k 步 RM 准确率：71%（OOD，策略已偏移）
PPO 第 50k 步 RM 准确率：58%（严重退化）
```

**实践启示**：RM 的 OOD 准确率是判断是否需要重训的关键信号。

### 8.3 与人工评估的相关性

最终评估：RM 排序与人工评估排序的 Spearman 相关系数 / Kendall τ。

$$\tau = \frac{(\text{协调对数}) - (\text{不协调对数})}{\binom{n}{2}}$$

### 8.4 RM 评估基准

| 基准 | 评估内容 |
|---|---|
| RewardBench | 综合 RM 能力（对话、安全、推理） |
| MT-Bench | 多轮对话 RM 评估 |
| AlpacaEval | 与 GPT-4 输出的对比胜率 |
| MATH-Verify | 数学推理 ORM 评估 |

---

## 第九章 面试题

### Q1：Reward Model 的训练目标是什么？为什么用 Bradley-Terry 模型而不是直接回归？

**答**：

**训练目标**：最大化人类偏好对的对数似然：

$$\mathcal{L}_{RM} = -\mathbb{E}[\log\sigma(r(y_w) - r(y_l))]$$

**为什么用 BT 模型而非直接回归**：

直接回归需要绝对分数（如 1-5 分），存在以下问题：
1. **标注一致性差**：不同标注者对"4分"和"5分"的边界判断不一致
2. **量纲问题**：不同类型任务的绝对分数难以对齐
3. **数据获取难**：让人判断"哪个更好"远比让人打分数容易

BT 模型只需要**相对偏好**（哪个更好），与人类标注的自然形式完全匹配，且只需要奖励的相对大小，不需要绝对量纲。

### Q2：如何识别和缓解 Reward Hacking？

**识别**：RM 分数上升但人工评估分数停滞或下降；KL 散度快速增大；生成长度异常增大；回答风格固化（低熵）。

**缓解**（按效果强弱排序）：
1. **RLVR**（根本解决）：换用可验证奖励，消除神经 RM 漏洞
2. **增大 KL 惩罚系数 $\beta$**：限制策略偏离参考策略
3. **定期重训 RM**：用最新策略数据更新 RM，缩小分布差
4. **RM 集成**：多个 RM 取保守估计，减少单点失效
5. **监控并早停**：KL 散度超过阈值时停止 PPO 训练

### Q3：ORM 和 PRM 的核心区别是什么？PRM 的标注成本如何降低？

**核心区别**：ORM 只在序列末尾给奖励，PRM 在每个推理步骤给奖励。PRM 解决了长推理链的信用分配问题，但标注成本高。

**降低标注成本**：
- **MCTS 自动标注**（Math-Shepherd）：从当前步骤随机完成 N 次，用正确率作为步骤价值
- **自洽性标注**：导向正确答案的步骤标为正，导向错误答案的标为负，全自动
- **LLM 自动批判**（Constitutional AI 思想）：用 LLM 评估每步推理是否合理

### Q4：为什么 RM 要从 SFT 模型初始化，而不是从 Base Model 初始化？

**答**：

1. **格式理解**：SFT 模型已经学会了指令格式，能更好地理解 prompt-response 的语义关系
2. **质量感知**：SFT 训练过程中模型接触了大量高质量 response，形成了对"好回答"的隐式表示
3. **收敛速度**：从 SFT 初始化收敛更快（与目标分布更近），从 Base Model 初始化需要更多训练步数
4. **泛化性**：SFT 模型在多任务上训练，RM 的泛化能力更强

### Q5：多维度 RM（Helpful/Harmless/Honest）在 PPO 训练中如何使用？权重如何确定？

**答**：

多维度 RM 的组合奖励：

$$r_{total} = w_h \cdot r_{helpful} + w_{harm} \cdot r_{harmless} + w_{hon} \cdot r_{honest}$$

**权重确定方法**：
1. **人工设定**：根据业务需求手动设定（最常见）
2. **帕累托前沿搜索**：在 helpful-harmless 空间中找帕累托最优权重组合
3. **约束优化**：以 harmless 为约束（$r_{harm} \geq \text{threshold}$），最大化 helpful
4. **学习权重**：用元学习或 bandit 算法自动调整权重

**实践注意**：Harmlessness 通常作为硬约束而非软权重——低于安全阈值的 response 直接屏蔽，不进入 PPO 训练。
