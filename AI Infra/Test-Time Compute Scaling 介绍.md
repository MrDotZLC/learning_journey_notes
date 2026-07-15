## 1. Test-Time Compute Scaling 概述

### 1.1 定义

**Test-Time Compute Scaling（测试时计算扩展，TTS）** 是指：

> 在模型部署推理阶段（Inference / Test Time），动态增加计算预算，使模型通过更多计算过程获得更高质量输出。

传统大模型 Scaling Law 主要关注：

$$  
\text{Performance}=f(\text{Parameters},\text{Data},\text{Training Compute})  
$$

即：

- 增大模型参数量；
- 增加训练数据；
- 增加训练 FLOPs。

Test-Time Compute Scaling 引入第三个扩展维度：

$$  
\text{Performance}=f(\text{Training Compute},\text{Model Size},\text{Test-Time Compute})  
$$

即：

> 模型训练完成后，不改变参数，通过增加推理阶段计算量提升能力。

OpenAI o1 系列、DeepSeek-R1 等 Reasoning Model 将这一方向推向主流。([ICLR 会议录](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b623663fd9b874366f3ce019fdfdd44-Abstract-Conference.html?utm_source=chatgpt.com "Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning"))

### 1.2 与传统推理的区别

普通 LLM 推理：

$$  
x_1,x_2,...,x_n  
\rightarrow  
\text{Transformer}  
\rightarrow  
y  
$$

模型直接生成答案。

例如：

```
用户：
证明费马大定理

LLM：
直接输出答案
```

计算量近似固定：

$$  
C_{test}\approx O(N_{token}\times P)  
$$

其中：

- $N_{token}$：生成 token 数；
- $P$：模型参数量。

Test-Time Scaling：

```
输入问题

      ↓

生成多个思考过程

      ↓

验证 / 搜索 / 修正

      ↓

选择最佳答案
```

计算量变为：

$$  
C_{test}
=
C_{generation}  
+  
C_{search}  
+  
C_{verification}  
$$

模型拥有：

- 更多思考 token；
- 更多候选路径；
- 更多自我验证；
- 更多搜索步骤。

---

## 2. 为什么需要 Test-Time Scaling

### 2.1 训练 Scaling 的收益降低

早期：

$$  
\text{模型规模}\uparrow  
\Rightarrow  
\text{能力}\uparrow  
$$

例如：

GPT-2 → GPT-3 → GPT-4。

但是继续扩大：

- 数据不足；
- GPU 成本指数增长；
- 参数收益递减。

因此出现新的问题：

> 能否把更多计算放到用户提问之后？

### 2.2 推理任务天然需要搜索

语言生成本质：

$$  
P(y|x)

\prod_{t=1}^{T}P(y_t|x,y_{<t})  
$$

普通生成：

每一步选择概率最高 token：

$$  
y_t=\arg\max P(y_t)  
$$

但是复杂任务：

例如：

- 数学证明；
- 代码 Debug；
- 复杂规划；

正确答案通常不是第一条路径。

需要：

$$  
\max_y P(y|x)  
$$

近似搜索。

#### 2.3 人类思考模式类似 Test-Time Scaling

人类解决难题：

```
快速思考

↓

发现错误

↓

重新推导

↓

验证

↓

输出
```

LLM 以前只有：

System 1：

> 快速生成

Reasoning Model 增加：

System 2：

> 慢速推理

### 3. Test-Time Scaling 的核心形式

目前主要有四类方法。

| 方法           | 核心思想   | 代表技术             |
| ------------ | ------ | ---------------- |
| Long CoT     | 增加思考长度 | o1               |
| Best-of-N    | 多次采样选择 | Self-consistency |
| Search       | 搜索推理空间 | MCTS             |
| Verification | 引入验证器  | PRM              |

### 4. Long Chain-of-Thought Scaling
#### 4.1 基本思想

增加模型内部推理 token：

普通：

```
问题
 ↓
答案
```

Long CoT：

```
问题

↓

Step1

↓

Step2

↓

Step3

↓

检查

↓

答案
```

计算：

$$  
C_{test}  
\propto  
N_{reasoning}  
$$

其中：

$N_{reasoning}$：

思考 token 数。

#### 4.2 示例

问题：

$$  
x^2+5x+6=0  
$$

普通：

```
x=-2,-3
```

Long CoT：

```
因式分解：

x²+5x+6

寻找两个数：

2+3=5

2×3=6

因此：

(x+2)(x+3)=0

得到：

x=-2,-3
```

增加中间计算，提高可靠性。

#### 4.3 缺点：Overthinking

并不是：

$$  
Thinking \uparrow  
\Rightarrow  
Accuracy\uparrow  
$$

存在：

$$  
Accuracy=f(C_{test})  
$$

通常：

```
          *
        *   *
      *       *
_____*_________*____

      Compute

```

先提升：

```
更多思考
↑
正确率
```

之后：

```
过度思考
↑
错误修改
```

部分研究发现过长 CoT 可能导致性能下降，存在 overthinking 现象。([arXiv](https://arxiv.org/abs/2506.04210?utm_source=chatgpt.com "Does Thinking More always Help? Understanding Test-Time Scaling in Reasoning Models"))

### 5. Best-of-N Sampling
#### 5.1 核心思想

不要让一个路径思考很久。而是生成多个答案。

例如：

$$  
N=8  
$$

生成：

```
Answer1
Answer2
Answer3
...
Answer8
```

然后选择：

$$  
y^*

\arg\max_y Score(y)  
$$

### 5.2 Self-Consistency

多个推理路径：

$$  
z_1,z_2,...,z_N  
$$

最终答案：

$$  
y_i=f(x,z_i)  
$$

投票：

$$  
y^*
=
\arg\max_y  
\sum_i  
I(y_i=y)  
$$

其中 $I$ 是指示函数。

#### 5.3 优点

并行度高：

```
GPU0:
 path1

GPU1:
 path2

GPU2:
 path3

...
```

适合推理集群。

### 6. Search-Based Scaling
#### 6.1 思想

把推理过程看成搜索树。

例如：

```
              Start

          /     |     \

        A       B       C

      / |       |
     A1 A2     B1

```

节点：

$$  
s_i  
$$

表示：

当前推理状态。

目标：

找到：

$$  
\max_s Reward(s)  
$$
#### 6.2 Monte Carlo Tree Search (MCTS)

流程：

```
Selection

↓

Expansion

↓

Simulation

↓

Backpropagation

```

类似 AlphaGo。

用于：

- 数学推理；
- Agent Planning；
- Code Agent。

### 7. Verification Scaling

核心变化：

过去：

```
模型生成答案
```

现在：

```
模型生成答案

↓

验证器判断

↓

重新搜索
```

结构：

```
Generator

      ↓

Candidate Answer

      ↓

Verifier

      ↓

Reward

```

数学形式：

$$  
y^*

\arg\max_y  
R(x,y)  
$$

其中，$R$ 为 Reward Model。

---

## 8. Test-Time Scaling 与 RL 的关系

现代 Reasoning Model 通常结合 RL 和 Test-Time Compute 。

### 8.1 Reinforcement Learning

训练：

$$  
\max_\theta  
E_{x,y}  
[R(x,y)]  
$$

模型学习：

什么时候：

- 多想；
- 少想；
- 检查；
- 放弃。

### 8.2 Test-Time Compute 是动态策略

普通模型是固定的，Reasoning Model 则是动态的，即：问题越难，投入更多计算。

例如：

简单问题：`2 + 2 = ?` 消耗 10 tokens

复杂问题：`证明黎曼猜想`  消耗 10000 tokens。

---

## 9. Test-Time Scaling 对推理部署的影响

对于推理系统，最大变化：

传统：

```
Batch Size ↑

吞吐 ↑
```

Reasoning：

```
Compute Budget ↑

Latency ↑
Accuracy ↑
```

出现新的优化目标：

$$  
\text{Utility}

\frac{Accuracy}{Latency\times Cost}  
$$

### 9.1 推理引擎需要支持动态计算

传统：

```
Request

↓

Generate 100 tokens

↓

Finish
```

未来：

```
Request

↓

Estimate difficulty

↓

Allocate budget

↓

Thinking

↓

Verify

↓

Answer
```

需要：

- Dynamic batching；
- KV Cache 管理；
- Early Exit；
- Speculative Decoding；
- Priority Scheduling。

## 10. Test-Time Scaling 与 GPU 推理优化关系
### 10.1 Decode 成本增加

Reasoning Model：

大量增加：

$$  
N_{decode}  
$$

KV Cache：

$$  
Memory

2L\times H\times N_{token}\times d  
$$

因此：

KV Cache 压力增加。

## 10.2 长思考降低吞吐

普通：

```
1 request
100 tokens
```

Reasoning：

```
1 request
10000 tokens
```

GPU 时间：

$$  
T  
\approx  
N_{token}  
\times  
T_{decode}  
$$

因此需要：

- PagedAttention；
- KV Cache Offload；
- Continuous Batching；
- Chunked Prefill。

---

## 11. 当前研究方向
### 11.1 Compute Optimal Scaling

目标：不是`越想越好`，而是 `给定 FLOPs，获得最大收益`。

优化：

$$  
\max_{C}  
Accuracy(C)  
$$

约束：

$$  
C\leq C_{budget}  
$$

### 11.2 Parallel Thinking

替代：

```
一个思考链无限增长
```

采用：

```
多个短思考

↓

投票
```

部分研究显示并行思考在一些场景下比单纯延长 CoT 更有效。([arXiv](https://arxiv.org/abs/2506.04210?utm_source=chatgpt.com "Does Thinking More always Help? Understanding Test-Time Scaling in Reasoning Models"))

## 12. 总结

|Scaling|优化对象|阶段|
|---|---|---|
|Parameter Scaling|参数量|Training|
|Data Scaling|数据量|Training|
|Training Compute Scaling|FLOPs|Training|
|**Test-Time Compute Scaling**|推理计算|Inference|

核心思想：

$$  
\boxed{  
\text{更强模型}  
\neq  
\text{更大参数}  
}  
$$

未来的大模型能力提升路线：

$$  
\boxed{  
\text{Training Scaling}  
+  
\text{Test-Time Scaling}  
+  
\text{Search/Verification}  
}  
$$

对于推理部署系统而言，Test-Time Scaling 的本质变化是：

> 从“固定成本生成模型”转变为“根据任务难度动态分配 GPU 计算预算的推理系统”。
