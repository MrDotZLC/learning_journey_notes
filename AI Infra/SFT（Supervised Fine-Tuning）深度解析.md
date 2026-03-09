---
tags: [RLHF, SFT, 监督微调, 指令微调, 数据工程]
aliases: [SFT笔记, 监督微调]
created: 2025-03-09
status: complete
---

## SFT 深度笔记（Supervised Fine-Tuning）

> 主文档：[[RLHF（Post-Training）]]　｜　相关：[[RM（Reward Model） 深度介绍]] · [[PPO（Proximal Policy Optimization）深度解析]]

---

### ⚡ 速查卡

| 维度 | 内容 |
|---|---|
| 核心目标 | 将 Base Model 转化为指令跟随模型，构造初始策略 $\pi_0$ |
| 训练信号 | 交叉熵损失，监督学习 |
| 数据规模 | 通常 1K～100K 条，质量远比数量重要 |
| 关键超参 | 学习率、epoch 数、数据配比 |
| 核心瓶颈 | 数据质量、格式多样性、灾难性遗忘 |

---

## 第一章 SFT 基础

### 1.1 在 RLHF 流程中的定位

```
预训练 Base Model
    │  有语言能力，但不能跟随指令，不区分好坏回答
    ▼
SFT（监督微调）
    │  输入：(prompt, high-quality response) 对
    │  输出：初始策略 π₀，可以跟随指令
    ▼
RM 训练 / PPO 优化
    │  SFT 提供两个基础：
    │    1. π₀：PPO 的起点策略
    │    2. π_ref：KL 惩罚的参考策略（冻结）
    ▼
对齐后的 LLM
```

**SFT 的两个角色**（经常被忽视）：
- 作为 **初始策略 $\pi_0$**：PPO 从它开始优化
- 作为 **参考策略 $\pi_{ref}$**：冻结后用于计算 KL 散度，防止 reward hacking

### 1.2 训练目标

标准语言模型交叉熵损失，仅在 response 部分计算（prompt 部分 mask 掉）：

$$\mathcal{L}_{SFT} = -\sum_{t=1}^{T} \log P_\theta(y_t \mid x, y_{1:t-1})$$

| 符号 | 含义 |
|---|---|
| $x$ | prompt token 序列 |
| $y_t$ | 第 $t$ 个 response token |
| $T$ | response 总长度 |
| $\theta$ | 模型参数 |

**为什么只在 response 部分计算损失**：prompt 是已知条件，让模型"记住"prompt 无意义，甚至会干扰指令跟随能力的学习。

### 1.3 SFT 与预训练的本质区别

| 维度 | 预训练 | SFT |
|---|---|---|
| 数据 | 万亿 token，无标注 | 千～万条，人工标注 |
| 目标 | 学习语言知识和世界知识 | 学习指令跟随格式和风格 |
| 训练时长 | 数周～数月 | 数小时～数天 |
| 学习率 | 较大（从随机初始化） | 极小（微调，防止遗忘） |
| 核心挑战 | 计算资源、数据清洗 | 数据质量、灾难性遗忘 |

> **关键洞见**：SFT 不教模型新知识，只是"解锁"预训练中已经学到的知识——让模型以正确的格式输出它本来就知道的内容。

---

## 第二章 SFT 数据工程

### 2.1 数据格式：Instruction Tuning 模板

不同模型使用不同的 chat template，本质上是用特殊 token 标记角色边界：

**ChatML 格式**（OpenAI，GPT 系列）：

```
<|im_start|>system
你是一个有帮助的助手。
<|im_end|>
<|im_start|>user
解释什么是梯度下降。
<|im_end|>
<|im_start|>assistant
梯度下降是一种优化算法...
<|im_end|>
```

**Llama 3 格式**：

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
你是一个有帮助的助手。<|eot_id|>
<|start_header_id|>user<|end_header_id|>
解释什么是梯度下降。<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
梯度下降是一种优化算法...<|eot_id|>
```

**损失 mask 实现**：

```python
# 构造 labels，prompt 部分设为 -100（CrossEntropyLoss 忽略）
labels = input_ids.clone()
labels[:prompt_length] = -100   # mask prompt tokens
loss = cross_entropy(logits, labels, ignore_index=-100)
```

### 2.2 数据质量维度

SFT 数据质量远比数量重要。InstructGPT 仅用 **13K 条**高质量数据即超越 1750 亿参数的预训练模型在多数任务上的表现。

**质量评估五维度**：

| 维度 | 描述 | 常见问题 |
|---|---|---|
| 准确性 | response 内容是否事实正确 | 幻觉、错误知识 |
| 相关性 | response 是否真正回答了 prompt | 答非所问 |
| 完整性 | response 是否涵盖所有必要内容 | 截断、遗漏关键信息 |
| 格式 | 是否符合预期输出格式 | 格式混乱、特殊字符 |
| 安全性 | 是否包含有害内容 | 歧视、危险信息 |

### 2.3 数据来源与构造方法

#### 2.3.1 人工标注（Human Annotation）

- 标注者直接撰写高质量 response
- 成本最高，质量最可控
- InstructGPT、Claude 早期版本使用

**标注指南的重要性**：Anthropic 的标注者指南明确定义了"有帮助"、"无害"、"诚实"的操作化标准，是 SFT 数据质量的核心保障。

#### 2.3.2 Self-Instruct（2023，Wang et al.）

用现有 LLM 生成指令-回答对，人工验证后加入训练集：

```python
# Self-Instruct 流程
seed_tasks = load_human_written_seeds(n=175)

for iteration in range(N):
    # 1. 用种子任务作为示例，让 LLM 生成新指令
    new_instructions = llm.generate(
        prompt=f"参考以下示例，生成新的指令：{sample(seed_tasks, k=8)}"
    )
    # 2. 过滤：去重、过滤低质量、过滤不安全
    filtered = filter(new_instructions, rouge_threshold=0.7)
    # 3. 生成 response
    responses = llm.generate(filtered)
    # 4. 人工抽样验证后加入种子集
    seed_tasks.extend(validated(responses))
```

#### 2.3.3 Alpaca / Stanford Alpaca 方法

直接用 GPT-4 / Claude 批量生成高质量指令数据，成本低、速度快：

```python
# Alpaca 数据生成 prompt 模板
prompt = """
生成 20 条多样化的任务指令，覆盖不同类型：
- 开放式问答
- 代码生成
- 文本改写
- 逻辑推理
- 创意写作
每条指令需要独特、具体，并给出高质量回答。
"""
data = gpt4.generate(prompt, n=20)
```

#### 2.3.4 LIMA（Less Is More for Alignment，2023）

核心发现：**1000 条精心挑选的 SFT 数据**可以媲美数万条低质量数据。

质量筛选标准：
- 多样性：覆盖尽可能多的任务类型
- 风格一致：response 风格统一、清晰
- 无歧义：prompt 明确，response 直接

#### 2.3.5 数据飞轮（Data Flywheel）

```
部署模型 → 收集真实用户交互 → 人工筛选高质量对话
    ↑                                      │
    └──────── 加入 SFT 训练集 ←────────────┘
```

Claude、GPT 的持续迭代依赖这种数据飞轮：真实用户数据比合成数据更有价值，因为覆盖了标注者想不到的边缘场景。

### 2.4 数据配比（Data Mixture）

不同任务类型的数据比例对最终模型能力影响显著：

| 任务类型 | 比例参考 | 影响 |
|---|---|---|
| 通用指令跟随 | 40~60% | 基础对话能力 |
| 代码 | 15~25% | 编程能力 |
| 数学/推理 | 10~20% | 推理能力 |
| 安全/拒绝 | 5~10% | 拒绝有害请求 |
| 角色扮演/创意 | 5~10% | 创意生成 |

**配比失衡的典型问题**：
- 代码数据过多 → 通用对话能力下降
- 安全数据过多 → 过度拒绝，无用性增加（over-refusal）
- 数学数据过多 → 回答风格变得过于结构化

### 2.5 数据去重与清洗

```python
# 基于 Rouge-L 的去重（Alpaca 方法）
from rouge_score import rouge_scorer

def is_duplicate(new_inst, existing_insts, threshold=0.7):
    scorer = rouge_scorer.RougeScorer(['rougeL'])
    for existing in existing_insts:
        score = scorer.score(new_inst, existing)['rougeL'].fmeasure
        if score > threshold:
            return True
    return False

# 更高效的方法：MinHash LSH（大规模数据集）
from datasketch import MinHash, MinHashLSH
```

---

## 第三章 SFT 训练技术

### 3.1 参数高效微调（PEFT）

全参数微调对大模型成本极高。PEFT 方法在保持预训练权重基本不变的前提下，只更新少量参数。

#### 3.1.1 LoRA（Low-Rank Adaptation，2021）

**核心思想**：微调时的权重更新矩阵 $\Delta W$ 具有低秩结构，用两个低秩矩阵近似：

$$W' = W_0 + \Delta W = W_0 + BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d,k)$。

**初始化**：$A$ 随机高斯初始化，$B$ 初始化为 0，保证训练初始时 $\Delta W = 0$。

**前向传播**：

$$h = W_0 x + \frac{\alpha}{r} BAx$$

其中 $\alpha$ 是缩放超参（通常 $\alpha = r$ 或 $\alpha = 2r$）。

**参数量对比**（以 LLaMA-7B 为例）：

| 方法 | 可训练参数 | 显存占用 |
|---|---|---|
| 全参数微调 | 7B | ~112GB（fp16） |
| LoRA（r=16） | ~4M（0.06%） | ~18GB |
| QLoRA（r=16，4bit base） | ~4M | ~10GB |

```python
# HuggingFace PEFT 实现
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                    # 秩
    lora_alpha=32,           # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 应用到哪些层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%
```

#### 3.1.2 QLoRA（Quantized LoRA，2023）

在 LoRA 基础上将 Base Model 量化到 4bit，进一步降低显存：

```
Base Model（4bit NF4 量化，冻结）
    + LoRA 适配器（fp16，可训练）
    + 双重量化（量化常数本身再量化）
    + 分页优化器（防止显存 OOM）
```

关键创新：**NF4（NormalFloat4）量化**，基于正态分布设计量化区间，比均匀量化精度更高。

#### 3.1.3 其他 PEFT 方法对比

| 方法 | 思路 | 优点 | 缺点 |
|---|---|---|---|
| LoRA | 低秩矩阵分解 | 推理无额外延迟（可合并） | 秩选择需调参 |
| Prefix Tuning | 在 KV cache 前插入可训练前缀 | 不修改模型权重 | 推理时有额外 token |
| Adapter | 在 FFN 后插入小型 MLP | 模块化，可叠加 | 推理有额外延迟 |
| IA3 | 对 K/V/FFN 施加可学习缩放向量 | 参数量极少 | 表达能力有限 |

### 3.2 学习率策略

SFT 的学习率需极小，防止灾难性遗忘：

```python
# 典型 SFT 超参（LLaMA-7B 量级）
learning_rate = 2e-5          # 全参微调
learning_rate = 2e-4          # LoRA（适配器参数）
warmup_ratio = 0.03           # 3% 步数 warmup
lr_scheduler = "cosine"       # cosine decay
weight_decay = 0.0            # SFT 通常不加权重衰减
gradient_clipping = 1.0
```

**Cosine decay with warmup**：

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t - t_{warmup}}{T - t_{warmup}}\pi\right)\right)$$

### 3.3 序列打包（Sequence Packing）

将多条短样本拼接成一个长序列，提高 GPU 利用率：

```
不打包（低效）：
  [prompt1|response1|PAD|PAD|PAD]  ← 大量 padding 浪费计算
  [prompt2|response2|PAD|PAD|PAD]

打包（高效）：
  [prompt1|response1|prompt2|response2|prompt3...]  ← 无 padding
```

**注意**：打包时需用 attention mask 防止不同样本间的 attention 穿透。

```python
# Flash Attention 2 支持 varlen（variable length）模式
# 可以高效处理打包序列，自动处理跨序列 attention mask
from flash_attn import flash_attn_varlen_func
```

### 3.4 多轮对话训练

多轮对话中，只对 **assistant** 的 response 计算损失：

```python
conversation = [
    {"role": "user",      "content": "你好"},         # mask
    {"role": "assistant", "content": "你好！"},        # 计算损失
    {"role": "user",      "content": "介绍一下自己"},  # mask
    {"role": "assistant", "content": "我是..."},       # 计算损失
]

# 构造 labels
for turn in conversation:
    if turn["role"] == "assistant":
        labels[turn_start:turn_end] = input_ids[turn_start:turn_end]
    else:
        labels[turn_start:turn_end] = -100  # mask
```

---

## 第四章 灾难性遗忘（Catastrophic Forgetting）

### 4.1 问题描述

SFT 在提升指令跟随能力的同时，可能导致预训练阶段学到的通用能力下降：

```
预训练能力：代码生成、数学推理、多语言、知识问答
                    │
                    ▼ SFT（只用英文指令数据）
                    │
SFT 后：英文指令跟随↑，但代码↓、多语言↓
```

### 4.2 缓解方法

#### 4.2.1 在 SFT 数据中混入预训练数据

```python
sft_data_ratio = 0.9      # 90% SFT 数据
pretrain_data_ratio = 0.1  # 10% 预训练数据（维持通用能力）
```

#### 4.2.2 降低学习率

学习率越小，对预训练权重的扰动越小。LoRA 天然有这个优势（基础权重完全冻结）。

#### 4.2.3 EWC（Elastic Weight Consolidation）

对预训练中重要的参数施加 L2 正则化，防止大幅偏移：

$$\mathcal{L}_{EWC} = \mathcal{L}_{SFT} + \frac{\lambda}{2}\sum_i F_i(\theta_i - \theta^*_i)^2$$

其中 $F_i$ 是 Fisher 信息矩阵对角线元素，衡量参数 $\theta_i$ 对预训练任务的重要程度。

#### 4.2.4 数据多样性

SFT 数据覆盖的任务类型越多，灾难性遗忘越轻微。LIMA 的实验表明，1000 条多样化数据比 50000 条单一任务数据效果更好。

---

## 第五章 SFT 的局限性与 RM 的动机

### 5.1 SFT 无法表达偏好程度

SFT 的损失函数把所有 token 同等对待：

$$\mathcal{L}_{SFT} = -\sum_t \log P_\theta(y_t | \cdot)$$

对于同一个 prompt，以下两个 response 在 SFT 中权重相同：
- Response A（好）："梯度下降是通过计算损失函数对参数的梯度，沿负梯度方向更新参数的优化方法。"
- Response B（差）："梯度下降是一种方法。"

SFT 无法说"A 比 B 好多少"。

### 5.2 标注者撰写回答的成本与质量上限

- 标注者能力参差不齐
- 难以覆盖所有场景（长尾 prompt）
- 撰写高质量 response 比判断"哪个更好"难得多

→ 以上局限驱动了 RM 的引入，见 [[RM（Reward Model） 深度介绍]]。

### 5.3 Cold Start 问题

对于复杂推理任务，如果 SFT 数据质量不足（缺乏长 CoT 示例），即使后续 PPO/GRPO 训练也难以提升推理能力——模型需要有"雏形"才能通过 RL 强化。

DeepSeek-R1 的 Cold Start SFT 阶段（数千条人工标注长 CoT）正是为了给 GRPO 提供一个足够好的起点。

---

## 第六章 SFT 面试题

### Q1：SFT 中为什么只对 response 部分计算损失，而不对 prompt 计算？

**答**：prompt 是给定的条件，不是模型需要"生成"的内容。对 prompt 计算损失会：
1. 让模型学习"记住"prompt 的文本模式，而非学习"根据 prompt 生成好的 response"
2. 浪费计算资源（prompt 通常较长）
3. 引入噪声：用户 prompt 质量参差不齐，对其计算损失会干扰指令跟随能力的学习

### Q2：LoRA 的原理是什么？为什么有效？

**答**：LoRA 基于"微调时权重更新矩阵具有低秩结构"的假设。

$$W' = W_0 + \Delta W \approx W_0 + BA, \quad B \in \mathbb{R}^{d\times r}, A \in \mathbb{R}^{r\times k}, r \ll \min(d,k)$$

**为什么有效**：
- 实验观察：微调前后权重差的秩通常很低（内在维度假说）
- 预训练知识已编码在 $W_0$ 中，任务适配只需要低维调整
- 推理时可以将 $BA$ 合并进 $W_0$，无额外延迟

**超参选择**：$r$ 越大表达能力越强，但参数量越多。通常 $r=8\sim64$。

### Q3：如何缓解 SFT 的灾难性遗忘？

**答**（按工程优先级）：

1. **LoRA**：基础权重冻结，从根本上避免遗忘
2. **混入预训练数据**：SFT 数据中加 5~10% 预训练文本，维持通用能力
3. **降低学习率**：$\leq 2 \times 10^{-5}$（全参微调），减小对权重的扰动
4. **SFT 数据多样化**：覆盖更多任务类型，避免单一任务导致的能力失衡
5. **EWC**：对重要参数施加弹性约束（工程复杂，实践中少用）

### Q4：LIMA 的核心发现是什么？对 SFT 数据工程有什么启示？

**答**：LIMA（Less Is More for Alignment）发现，**1000 条高质量、多样化的 SFT 数据**可以媲美数万条低质量数据。

核心结论：
- 预训练已经学会了知识和能力，SFT 只是"对齐输出风格"
- 数据质量 >> 数据数量
- 多样性比规模更重要

**工程启示**：
- 投入预算应优先用于数据质量筛选，而非数据采集规模
- 每条数据都应该是"如果人类专家来回答这道题，最好的答案是什么"
- 用 Rouge-L 或 embedding 相似度定期去重，防止数据分布退化

### Q5：Self-Instruct 和直接用 GPT-4 生成数据有什么区别？各自适用什么场景？

**答**：

| | Self-Instruct | GPT-4/Claude 直接生成 |
|---|---|---|
| 种子数据 | 少量人工撰写（~175 条） | 可以从零开始 |
| 多样性 | 依赖 LLM 自我扩展，逐步多样化 | 可以直接指定任务分布 |
| 质量控制 | 需要多轮过滤 | 质量较高，但需验证 |
| 成本 | API 调用量大（迭代生成） | API 调用一次即可 |
| 适用场景 | 有少量高质量种子，需要扩展 | 从头构建 SFT 数据集 |

**实践中**：通常两者结合——先用 GPT-4 生成初始数据集，再用 Self-Instruct 方式用自己的模型迭代扩展（数据飞轮）。

### Q6：序列打包（Sequence Packing）有什么风险？如何避免？

**答**：主要风险是**跨序列 attention 穿透**——打包后多个样本拼在一起，如果 attention mask 处理不当，后一个样本的 token 会 attend 到前一个样本的内容，引入错误的上下文信息。

**避免方法**：
1. **位置编码重置**：每个新样本从位置 0 开始编码
2. **因果 mask 修改**：确保 attention 不跨越样本边界
3. **Flash Attention varlen 模式**：原生支持打包序列，自动处理边界

```python
# 构造打包序列的 attention mask
# cu_seqlens: 每个序列的累积长度
cu_seqlens = torch.tensor([0, len_seq1, len_seq1+len_seq2, ...])
# flash_attn_varlen_func 自动处理边界
output = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)
```
