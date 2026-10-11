## AI Infra 训练平台全链路

> 机器人数据采集 → 数据管理 → 分布式训练 → 训练任务调度 → 实验管理 → 模型压缩 → 云端/端侧部署

## Linux
### 常用指令
ps、top等
### 网络基础
TCP/IP、TSL/SSL、RPC、IPC、Socket、ping、ip、netstat等
### 权限管理
文件的rwx、软链接 ln -s、chmod、chown、sudo
### GPU
nvidia-msi、ncu、nsys
### 性能分析
pref、top、iostat

## Docker
### 基础
OS级容器，保持环境一致，与主机共享kernel。
namespace：隔离资源
网络模式：host、bridge、none、overlay
卷挂载
layer：每次修改都会生成完整的副本layer，尽量多个操作合并为一行，layer可合并
### GPU
GPU分配：容器内安装该NVIDIA Container Toolkit 和 cuda runtime 包，会自动使用宿主机的驱动。
### 启动工具
多服务部署：使用docker-compose

## Kubernetes

Kubernetes 是一个**容器编排平台**，用于自动管理大量 Container 的部署、调度、扩缩容和故障恢复。
```
			kubectl
			   ↓
		   api server
		       ↓
		----------------
  Scheduler Controller etcd
		       ↓
		      Node 
```
```
			  Node
			   ↓
			kubectl
			   ↓
			  Pod
			   ↓
		   Container
```

1. kubectl 是 k8s 的控制面板和 Node 的代理，调用 api server 查看 Node、查看 pod 、创建查看容器、修改Deployment、滚动重启等。
2. etcd 是 Kubernetes 的分布式 Key-Value 存储，保存整个集群状态（Pod状态、Node信息、Service信息、ConfigMap、Secret）。
3. Scheduler 根据现有资源为 Container 分配 Node。
4. Controller 持续监听实际状态，并控制 pods 达到期望状态。
5. Pods 是 Kubernetes 最小调度单位，可包含多个 Container，只能修改 Pod 的配置。
   Pod 的状态有：Pending、Running、Succeeded、Failed、Unknown。
   重启策略有：Always（失败自动重启）、OnFailure（失败重启）、Never（不重启）

6. Deployment 管理无状态应用。提供副本管理、滚动升级、自动恢复。、
7. ReplicaSet 保证 Pod 数量。
8. StatefulSet 管理有状态实例。
9. Service 是为一组 Pod 提供稳定访问入口和四层负载均衡的抽象。通过 Label Selector 找到 Pod，用 Endpoints 记录后端，靠 kube-proxy 实现转发，用 DNS 提供名字。四种类型分别解决集群内、Node 端口、云负载均衡和外部域名映射的需求。

### 网络

1. Pod IP唯一
2. Pod可以直接通信
3. Node之间通信走 CNI
4. CNI 是实现 Pod 跨 Node 互通的插件
5. 高质量 GPU 通信主要依赖**Multus + SR-IOV/GPUDirect**。
   - Multus 是让 Pod 有多网卡，主网卡走CNI，众多辅网卡跑训练。
   - SR-IOV 是硬件虚拟化技术，把一张物理网卡虚拟成多个 VF，分配给多个 Pod 独占使用，绕过了宿主机的内核网络栈，直接从网卡硬件收发数据。
   - GPUDirect RDMA 允许网卡直接通过 PCIe 总线读写 GPU 的显存，完全跳过 CPU 和系统内存。
   - 支持NCCL。

### GPU

1. 在每个 Node 上安装 NVIDIA 驱动和 NVIDIA Device Plugin。
   然后在 Pod Yaml 中申请 GPU 资源
   Scheduler 为 Pod 分配 Node

2. 通常 Pod 不可共享同一张 GPU，有方案可实现。

## Slurm

> 一个开源的、高可扩展性的集群管理和作业调度系统，专门为 Linux 集群设计。接收用户提交的计算任务，并根据资源需求和优先级，合理地分配计算节点来运行这些任务。

Slurm 作为工作负载管理器，主要有三大功能
1. **资源分配**：在一段时间内，将计算节点（资源）的独占或非独占访问权分配给用户，让他们可以执行工作。
2. **任务管理**：提供一个框架，用于在分配到的节点上启动、执行和监控任务（通常是并行作业）。
3. **仲裁争用**：通过管理待处理作业的队列，来仲裁多个用户对有限资源的竞争。

|       | Slurm  | Kubernetes |
| ----- | ------ | ---------- |
| 定位    | HPC调度  | 云原生调度      |
| 对象    | 计算任务   | 服务/任务      |
| GPU训练 | 非常成熟   | 需要插件       |
| 应用    | 科研训练集群 | 企业平台       |

Slurm 和 k8s  可混合使用。Slurm用于分布式训练，k8s 用于数据预处理、实验管理、Notebook、推理服务、CI（持续集成）/CD（持续部署） 等云原生负载。

## 数据流水线
### 多模态数据时间同步
ROS 有该功能。
1. 时钟同步
   PTP硬件同步，软件精度不够，
2. 时间戳语义统一
3. 时间戳对齐
   - 硬件触发同步
   - 帧对齐：设基准数据帧时间戳 t_c，找[t_c - △t, t_c + △t]内的其他源数据，线性插值到 t_c
   - 高频对齐：设基准数据帧时间戳 t_c，找其他源数据的前后两帧，线性插值或最近帧+补偿
   - 统一到全局时间轴：重采样到固定频率，再按照 基准时间戳 线性插值或最近帧+补偿

### 数据版本管理
通用工具：LakeFS
具身智能常用：LanceDB 机器人数据湖仓（Episode/Trajectory  diff、质量检查、适配MCAP/ROS 等特殊格式等等）

### 模型版本管理
通用工具：MLFlow、Kubeflow

## MLOps 工具
### Kubeflow
> 构建在 Kubernetes 之上的机器学习（Machine Learning）工作流平台，用于管理模型训练、调优、部署和生命周期管理。

| 组件                 | 作用      |
| ------------------ | ------- |
| Kubeflow Notebook  | 提供开发环境  |
| Kubeflow Pipelines | 机器学习流水线 |
| Training Operator  | 分布式训练   |
| Katib              | 超参数搜索   |
| KServe             | 模型部署    |
| Central Dashboard  | 统一入口    |

1. Kubeflow Notebook 本质是 Kubernetes 中运行的 Jupyter Pod。
2. Kubeflow Pipeline 将流程定义成 DAG，每个节点对应 Kubernetes Pod，实现全流程的自动化执行。
3. Kubeflow Training Operator 就是 Kubernetes 上的“分布式训练调度器”，通过一个通用的控制器来管理不同框架的作业 Custom Resource (CR)，CR 是与 Node 同级别的自定义资源类型，可表示“训练任务”。
4. Katib 是解决自动化机器学习（AutoML） 问题的组件，主要功能：
   **超参数调优**：寻找模型训练的最佳参数组合
   **早停**：发现某组参数明显效果差时，提前终止训练，节省算力
   
5. KServe 是负责**模型推理服务部署**的组件，负责把模型部署成在线服务。封装了自动扩缩容、流量路由、健康检查、版本管理、GPU 调度等。
6. 大模型训练任务接入 Kubeflow 流程
```
Docker Image
 |
Training Dataset
 |
PyTorchJob(Custom Resource)
 |
GPU Scheduler
 |
NCCL Distributed Training
 |
Checkpoint Storage
 |
Evaluation
```

Kubeflow 是 Kubernetes 上的机器学习平台，主要解决 ML 生命周期管理问题。它通过 Notebook 提供开发环境，通过 Pipeline 编排数据处理、训练、评估流程，通过 Training Operator 管理分布式训练任务，例如 PyTorchJob，通过 KServe 完成模型部署。底层依赖 Kubernetes 的 Pod 调度、GPU Device Plugin、RBAC 和存储系统，实现训练和推理任务的自动化管理。

## ROS

ROS 是机器人领域最主流的机器人软件中间件（Robotics Middleware），是一套运行在 Linux 上的分布式机器人应用框架，提供通信机制、硬件抽象、工具链、调试工具和生态组件。

### ROS 1
Apollo 早期使用的就是 ROS 的通信RPC，通讯风格和 ROS 一模一样，都是模块 Node 解耦、消息驱动、Topic 异步发布/订阅 和 主动请求等特性。

### Action
此外，Action 是 ROS 处理耗时任务、有中间反馈、可取消的异步调用机制。Apollo 没有这个机制，它核心流程是单向流动的，用 topic 足够处理了。

### ROS 2
#### 删除 ROS Master，替换成 DDS
Master 是中心化组件，多机器人部署存在单点故障。使用 DDS 数据分布式服务实现：
	去中心化
	广播自动发现
	单播实时通信
	多机通信
	控制QoS
Apollo 中期自研通讯架构也是因为 Master 单点瓶颈，自行设计了一套类似 DDS 的机制，包括可靠性、历史数据保存、缓存策略、周期约束、消息生命周期。

### Launch 大量节点
统一管理节点生命周期、参数配置、环境变量和启动顺序。

### 节点动态配置
使用 yaml 动态修改 Node 节点。

### TF 坐标变换系统
统一管理机器人各部件之间坐标系关系的核心基础设施

## RAG（检索增强生成）
### 什么是 RAG？
**RAG（Retrieval-Augmented Generation，检索增强生成）** 是一种将外部知识库（检索）**与**大语言模型（生成）结合的技术架构。

它的核心痛点在于解决 LLM 的三大原罪：
1. **幻觉（Hallucination）：** 模型一本正经地胡说八道。
2. **时效性滞后（Knowledge Cutoff）：** 无法获取最新的信息或实时数据。
3. **私有数据不可见：** 企业内部的文档、规章制度等不在模型预训练集中。

### 基本工作流程（Pipeline）

标准的 RAG 系统主要分为三个阶段：

```
[原始文档] ➔ 分块(Chunking) ➔ 向量化(Embedding) ➔ 写入向量数据库
                                                        │
[用户 Query] ➔ 向量化(Embedding) ➔ 向量检索(Top-K) ─────┘
                                      │
                                [相关 Context] + [用户 Query]
                                      │
                                      ▼
                                [大模型 LLM] ➔ 生成最终回答

```

1. **数据准备与索引（Indexing）：**
* **读取与解析（Parse）：** 读取 PDF、Word、Markdown 等不同格式的文件。
* **切分（Chunking）：** 将长文本切分成合适大小的片段（Chunk）。
* **向量化（Embedding）：** 通过 Embedding 模型将文本片段转化为高维向量。
* **存储（Storage）：** 存入向量数据库（如 Milvus, Pinecone, Chroma, FAISS）。

2. **检索（Retrieval）：**
* 用户输入 Query，通过同样的 Embedding 模型转为向量。
* 在向量数据库中计算余弦相似度或欧氏距离，检索出相似度最高的前 $K$ 个文本块。

1. **生成（Generation）：**
* 将检索到的文本块作为 Context（上下文），拼接上用户的 Query 组成提示词（Prompt）。
* 将 Prompt 送入 LLM，生成带有依据的准确回答。

### 核心进阶技术（Advanced RAG）

**检索前优化（Pre-Retrieval）：**
* **Query Rewriting（查询重写）：** 用户输入的 Query 可能很模糊，用 LLM 将其扩展或重写为更具体的查询。
* **HyDE（Hypothetical Document Embeddings）：** 让 LLM 先生成一个假设性的回答，再用这个假设回答去检索真实的文档块（语义更接近）。
* **Multi-Query Expansion：** 将一个 Query 扩展成多个不同维度的同义 Query 并并行检索。

**检索中优化（In-Retrieval）：**
* **混合检索（Hybrid Search）：** **向量检索（语义相似） + 关键字检索（BM25/TF-IDF）**。这是工业界落地必用的策略，能兼顾语义理解与精准匹配（如专有名词、编号）。
* **切分策略优化：** 句级切分、段落切分、以及 **Parent-Child Chunking（父子分块）**（检索时用小 Chunk 计算相似度，喂给 LLM 时带上完整的大 Chunk 上下文）。

**检索后优化（Post-Retrieval）：**
* **重排序（Reranking）：** 使用交叉编码器（Cross-Encoder，如 BGE-Reranker）对初筛出来的 Top-K 结果进行更精准的重排序，大幅提升上下文质量。
* **上下文压缩/精简（Context Compression）：** 剔除检索结果中的噪音，只保留核心句子，避免超出 LLM 上下文窗口或引起“Lost in the Middle”问题。

### Q&A
#### Q1: 如何解决 RAG 中的“Lost in the Middle”（中间迷失）问题？

* **现象描述：** 大模型在处理长上下文时，往往更容易关注 Prompt 开头和结尾的信息，而忽略中间的信息。
* **回答要点：**
1. **限制 Context 数量：** 减少 Top-K 的值，提升重排序（Rerank）精准度。
2. **智能重排（Rerank & Reorder）：** 调整上下文顺序，将最相关的文档放在 Prompt 的最头部或最尾部。
3. **上下文压缩（Context Compression）：** 提取相关段落的核心摘要或关键句，只将高价值文本传入 LLM。

#### Q2: 为什么有了纯向量检索，还需要混合检索（Hybrid Search）？

* **现象描述：** 向量检索擅长理解“语义相似”，但对“精准匹配”极度不敏感。
* **回答要点：**
* 当用户搜索特定**产品型号（如 `iPhone 15 Pro Max`）**、**人名、错误码（如 `ERR_502`）** 或**缩写**时，向量 Embedding 容易将相似概念混淆。
* **混合检索（BM25 + Vector）**：利用 BM25 做精准字面匹配，利用向量做语义泛化，最后通过 **RRF（Reciprocal Rank Fusion）** 或加权评分融合两者结果，大幅提升召回率和准确率。

#### Q3: 如何合理选择与优化切分粒度（Chunk Size）？

* **现象描述：** Chunk 太小丢失上下文语义；Chunk 太大混入太多噪音且占用 LLM 窗口。
* **回答要点：**
1. **Chunking 策略分类：** 固定长度切分、按标点/段落语义切分、以及**语义切分（Semantic Chunking）**（利用 Embedding 变化率寻找语义转折点切分）。
2. **父子文档（Parent-Child / Small-to-Big Retrieval）：** 切分出非常小的子 Chunk 用于高精度的向量比对，找到对应子 Chunk 后，将包含它的完整父级段落送入 LLM，平衡检索精准度与上下文完整度。

#### Q4: 如何评估（Evaluation）RAG 系统的效果？

* **回答要点：** 面试中常提及 **RAG Triad（RAG 三元组评估指标）**，通常结合框架如 **Ragas** 或 **TruLens** 进行自动化评估：
1. **检索相关性（Context Relevance）：** 检索到的 Context 是否和 Query 紧密相关（量化检索效果）。
2. **忠实度/真实性（Faithfulness/Groundedness）：** 生成的回答是否严格基于检索出的 Context，有没有模型自带的幻觉（量化生成效果）。
3. **回答相关性（Answer Relevance）：** 生成的回答是否真正回答了用户提出的问题。

#### Q5: 微调（Fine-Tuning）与 RAG 该如何选择？

* **回答要点：**
* **RAG 适合：** 经常变动的动态数据、私有知识库检索、需要明确给出出处来源（溯源）、对时效性要求高的场景。成本低、更新快。
* **Fine-Tuning 适合：** 调整模型的**输出风格/格式**、特定领域的**语言表达惯例**、特定任务（如按照特定 JSON 格式输出）或者注入不易改变的内部专业领域思维方式。
* **结合方案（RAG + FT）：** 先微调一个特定领域能力更强、理解格式更好的小模型，再配合 RAG 检索最新知识库，是工业界非常流行的落地组合。

## SFT（Supervised Fine-Tuning，有监督微调）

### 一、 什么是 SFT？

SFT 指在**预训练模型（Base Model）** 的基础上，使用经过人工标定或高质量生成的**指令-回答对（Instruction-Response Pairs）** 进行有监督的训练，使模型学会**理解指令、遵循格式，并按照人类期望的方式进行对话和任务处理**。

#### 1. 核心作用

* **指令遵循（Instruction Following）：** 让模型从“接话茬”（预训练补全）变成“听懂指令并执行”。
* **任务对齐（Alignment）：** 让模型具备回答提问、摘要提取、代码生成、角色扮演等特定领域/任务能力。
* **格式规范：** 约束模型的输出格式（如输出 JSON、Markdown 或指定语调）。

#### 2. SFT 的训练目标（损失函数）

SFT 依然采用预训练时代的 **自回归交叉熵损失（Autoregressive Cross-Entropy Loss）**，但关键区别在于：**只对目标回答（Response）部分计算 Loss，而掩码（Mask）掉指令（Prompt）部分**。

具体数学表示为：

$$L = -\sum_{i=1}^{N} \log P(x_i \mid x_{<i}, \text{Prompt})$$

其中 $x_i$ 是 Response 中的 Token。

### 二、 SFT 的核心流程与关键细节

#### 1. 数据准备（最具决定性的环节）

* **数据结构：** 通常为 `<Prompt/Instruction, Input, Response>`。
* **数据质量 > 数据数量（Less is More）：** LIMA 论文证明，只需 1,000 条极其高质量的 SFT 数据，模型就能展现出优异的指令遵循能力。
* **多样性与分布：** 涵盖通用对话、逻辑推理、代码编写、多语言、角色扮演等多任务，避免过拟合于单一领域。

#### 2. 模型训练与参数更新方式

* **全量微调（Full Fine-Tuning）：** 更新模型的所有参数。效果最好，但显存开销极大。
* **高效参数微调（PEFT / LoRA / QLoRA）：** 冻结原模型绝大部分参数，仅训练低秩矩阵（LoRA）或极少量附加层。极大降低算力门槛。

### 三、 Q&A

#### Q1：SFT 中只对 Response 部分算 Loss，为什么？如何实现？

* **回答要点：**
1. **为什么：** Prompt 是人类给出的条件输入，模型的目标是**学习根据条件生成好回答**，而不是学习去预测人类的 Prompt。如果对 Prompt 算 Loss，模型会倾向于记住 Prompt 的分布，甚至导致指令泛化能力下降。
2. **如何实现：** 采用 **Mask 机制**。在构建训练数据序列（`Prompt + Response`）时，生成对应的 `labels` 序列。将对应 `Prompt` 位置的 `label` 全部设为 `-100`（PyTorch 默认忽略该 token 的 loss 计算），只保留 `Response` 位置的真实 Token ID。

#### Q2：SFT 训练时，多轮对话（Multi-turn Dialogue）如何高效构建训练样本？

* **回答要点：**
* **常规做法（低效）：** 把 $N$ 轮对话拆成 $N$ 个独立样本。第 1 轮做一次前向，第 1~2 轮做一次前向……会导致大量的重复计算（Prompt 部分重算）。
* **高效做法（拼接计算）：** 将整个对话拼接成一个长 Sequence，形如 `User1 -> Bot1 -> User2 -> Bot2`。
* **Loss 计算：** 仅把所有 `User` 输入处的 label 设置为 `-100`，所有 `Bot` 回答处的 label 设置为真实 Token ID。这样只需 **一次前向传播**，就能同时计算多轮回答的 Loss，大幅提升训练效率。

#### Q3：为什么 SFT 容易发生“灾难性遗忘”（Catastrophic Forgetting）？如何缓解？

* **回答要点：**
* **原因：** SFT 的数据量（通常几万到几十万条）远小于预训练（几万亿 Token）。模型在小数据集上高强度训练，权重被大幅拉向特定任务，破坏了预训练阶段建立的通用知识网络。
* **缓解策略：**
1. **混合预训练数据（Data Mixing）：** 在 SFT 数据中混入 5%~10% 的通用预训练文本（General Language Modeling Data）。
2. **使用轻量化微调（LoRA）：** 冻结底座模型权重，仅更新旁路参数，天然保护了主干模型的知识。
3. **控制学习率（Learning Rate）与 Epoch：** SFT 学习率通常设得极小（如 $10^{-5}$ 或更小），训练 1~3 个 Epoch 即可，切忌过拟合。

#### Q4：SFT 与预训练（Pre-training）、RLHF 有什么区别和联系？

* **回答要点：**
* **预训练（PT）：** 学习海量无标注文本，积累世界知识与语言建模能力（基座）。
* **SFT：** 学习“指令与遵循”，激发并约束模型的知识表达方式，让模型变得可用（有用性 Alignment）。
* **RLHF/DPO（强化学习）：** 进一步解决模型的安全性、偏好对齐以及“幻觉”问题，调整模型的回答风格偏好（符合人类价值倾向）。

#### Q5：SFT 数据的“质”和“量”应该如何权衡？

* **回答要点：**
* **质量远优于数量：** LIMA 论文展示了“Less Is More”原则。1,000 条筛选极其严格（逻辑严密、无幻觉、格式统一）的数据，效果远好于 10 万条噪声较大的爬虫/开源清洗数据。
* **筛选标准：** 准确性（无事实错误）、多样性（覆盖不同指令格式与任务）、安全性、清晰度（避免歧义）。
* **实操方案：** 先用大模型（如 GPT-4）进行初步数据清洗或合成（Self-Instruct），再通过专家人工校验抽样筛选。

#### Q6：训练过程中，如何防止模型出现“幻觉”或“胡说八道”？

* **回答要点：**
* SFT 阶段模型产生幻觉，很多时候是因为**训练数据中包含了模型预训练阶段未曾学过的知识**（强迫模型生成未知知识）。
* **解决方案：**
1. **清洗数据：** 移除超出模型能力上限的生僻/过度专业化数据。
2. **允许拒绝回答：** 在数据中加入“对不起，我无法回答该问题”或“根据已知信息无法得出结论”的拒答样本。
3. **结合 RAG（检索增强生成）：** 改变 Prompt 结构，教模型“根据给定的参考文本”来回答。

#### Q7：什么是 Packing 技术？为什么在 SFT 中要用 Packing？

* **回答要点：**
* **问题背景：** SFT 数据每条长度不一，传统做法是用 `<pad>` 补齐到固定长度（如 4096），但这会导致巨大的显存浪费和无效计算（Padding 部分算废代码）。
* **Packing 机制：** 把多条较短的 `Prompt + Response` 拼接在同一个 Sequence 里，直到填满最大序列长度（如 4096），中间用 `<eos>` 或特有 Token 隔开。
* **注意力隔离（Block Diagonal Mask）：** 拼接后需要修改 Attention Mask，确保不同样本之间互不可见，防止前面的回答关注到后面 unrelated 的问题。

#### Q8：简述 LoRA 的原理以及为什么它在 SFT 中表现优异？

* **回答要点：**
* **原理：** 假设模型权重的更新量 $\Delta W$ 具有较低的“固有秩”（Intrinsic Rank）。因此将 $\Delta W$ 分解为两个低秩矩阵的乘积：

$$W_{new} = W_0 + \Delta W = W_0 + B \cdot A$$

其中 $W_0 \in \mathbb{R}^{d \times k}$ 冻结，$A \in \mathbb{R}^{r \times k}$ 和 $B \in \mathbb{R}^{d \times r}$（$r \ll \min(d, k)$）参与训练。
* **优势：** 极大地节省显存与训练成本；推断时可以将 $B \cdot A$ 权重直接融入 $W_0$，**零推理延时**；同时防止全量参数被微调破坏导致过拟合。
