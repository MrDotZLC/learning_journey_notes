## 1. 本质区别

当前业界通常将模型分为：

|类型|代表模型|
|---|---|
|标准 LLM（Non-Reasoning Model）|GPT-4o、Llama-3、Qwen3-Instruct|
|推理模型（Reasoning Model）|o1、o3、DeepSeek-R1、Qwen3-Reasoning|

两者最大的区别并不是模型结构。

绝大多数推理模型仍然是：

$$  
\text{Decoder-only Transformer}  
$$

架构上与普通 LLM 基本一致。

真正区别在于：

$$  
\boxed{  
\text{训练目标}  
+  
\text{推理策略}  
}  
$$

---

## 2. 标准 LLM 的工作方式

### 2.1 Next Token Prediction

标准 LLM 的目标是：给定历史 Token预测输出。

例如：问题是“中国的首都是”，模型直接输出“北京“。

整个过程：

```text
问题
 ↓
直接生成答案
```

没有显式推理过程。

### 2.2 特点

优点：
- 延迟低
- Token 数少
- 吞吐高

缺点：
- 数学能力有限
- 长链推理容易出错
- Agent 规划能力较弱

例如：问题“37 × 48”，标准模型经常直接猜“1776”，甚至“1736”也可能出现。

---

## 3. 推理模型的工作方式

### 3.1 显式生成推理过程

显式推理过程（Explicit Reasoning Trace）定义更宽泛：只要模型显式输出中间思考内容即可，CoT、ToT、自纠错、搜索轨迹等。

### 3.2 Chain of Thought

CoT：

Chain of Thought

形式：

```text
Step1
Step2
Step3
...
Answer
```

例如：

```text
已知：

A>B
B>C

问：

A 与 C 的关系？
```

---

## 4. 训练过程差异

### 4.1 标准 LLM

训练数据：

```text
Question
Answer
```

例如：

```text
法国首都？

巴黎
```

损失函数：

$$  
\mathcal L_{SFT}

-\sum_t \log P(x_t)  
$$

学习：

```text
问题
 ↓
答案
```

映射。

### 4.2 推理模型

训练数据：

```text
Question
Reasoning
Answer
```

例如：

```text
37×48

↓

37×40=1480

37×8=296

1480+296=1776
```

训练：

$$  
Q  
\rightarrow  
CoT  
\rightarrow  
A  
$$

---

## 5. RL 的作用

Reasoning Model 最大突破来自 RL。

典型流程：

```text
Pretrain
 ↓
SFT
 ↓
RL
```

例如 DeepSeek-R1：

训练奖励：

```text
答案正确
 +1

答案错误
 0
```

模型逐渐发现：

```text
多想几步
```

更容易获得奖励。

于是自发形成：

```text
Let me think...

Step1
Step2
Step3
```

本质：

RL 学会：

$$  
\boxed{  
用更多 Token 换更高正确率  
}  
$$

---

## 6. Test-Time Compute Scaling

这是两类模型最大的工程区别。

### 标准 LLM

固定计算量：

```text
Question
 ↓
Answer
```

例如：

```text
20 Tokens
```

结束。

计算量：

$$  
O(20)  
$$

### 推理模型

动态计算量：

```text
Question
 ↓
Think
 ↓
Think
 ↓
Think
 ↓
Answer
```

可能：

```text
500
```

Token。

也可能：

```text
5000
```

Token。

甚至：

```text
50000
```

Token。

即：

$$  
\boxed{  
\text{Accuracy}  
\propto  
\text{Thinking Tokens}  
}  
$$

这就是：

Test-Time Compute Scaling。

---

## 7. Scaling Law 的变化

### 标准模型

依赖：

$$  
\text{Performance}

f(  
\text{Parameters},  
\text{Training FLOPs}  
)  
$$

提升能力：

```text
7B → 70B → 700B
```

### 推理模型

增加新维度：

$$  
\text{Performance}

f(  
\text{Parameters},  
\text{Training FLOPs},  
\text{Inference FLOPs}  
)  
$$

即：

推理阶段继续扩展计算。

例如：

7B 推理模型可能超过：

```text
70B 标准模型
```

在数学任务上的表现。

---

## 8. 对推理优化的影响

### 标准 LLM

Token 分布：

```text
Prompt
↓↓↓↓↓↓↓↓
Answer
```

通常：

```text
Prompt > Answer
```

Prefill 占主导。

优化重点：
- FlashAttention
- Chunked Prefill
- Prefix Cache
    
### 推理模型

Token 分布：

```text
Prompt
↓↓

Reasoning
↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

Answer
↓↓
```

通常：

$$  
L_{CoT}  
\gg  
L_{Prompt}  
$$

甚至：

$$  
L_{CoT}

10L_{Prompt}  
$$

因此 Decode 成为瓶颈。

优化重点变成：
- Speculative Decoding
- Adaptive γ
- Medusa
- EAGLE
- KV Cache 管理
- PD Disaggregation
- Continuous Batching

---

## 9. 从推理引擎角度看

对于标准模型：

总时间：

$$  
T

T_{prefill}  
+  
T_{decode}  
$$

且：

$$  
T_{prefill}  
\approx  
T_{decode}  
\text{ 或更大}  
$$

对于推理模型：

$$  
L_{reason}  
\gg  
L_{prompt}  
$$

因此：

$$  
T_{decode}  
\gg  
T_{prefill}  
$$

最终：

$$  
T  
\approx  
T_{decode}  
\approx  
L_{reason}  
\times T_{step}  
$$

所以推理模型时代：

关注点从

```text
Attention优化
```

逐渐转向

```text
Decode吞吐优化
KV Cache优化
Speculative Decoding
```

## 10. 核心总结

|维度|标准 LLM|推理模型|
|---|---|---|
|目标|直接回答|先推理再回答|
|输出|Answer|CoT + Answer|
|训练数据|Q→A|Q→Reasoning→A|
|RL依赖|较弱|极强|
|Test-Time Scaling|无|核心能力|
|推理Token数|少|极多|
|主要瓶颈|Prefill + Decode|Decode|
|KV Cache压力|中等|极高|
|Speculative收益|一般|极大|
|推理优化重点|Attention/GEMM|Decode/KV/Speculative|

从推理优化工程师视角，可以将两者概括为：

$$  
\boxed{  
\text{标准LLM}=  
\text{一次前向得到答案}  
}  
$$

$$  
\boxed{  
\text{推理模型}=  
\text{利用大量 Decode Token 搜索答案}  
}  
$$

推理模型本质上把 **训练阶段的部分计算转移到了推理阶段（Test-Time Compute）**，用额外的 Decode FLOPs 换取更高的正确率。
