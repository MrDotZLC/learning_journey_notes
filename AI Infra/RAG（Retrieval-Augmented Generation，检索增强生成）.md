## 1. 动机与背景

大模型的知识来自训练语料，存在两个结构性缺陷：知识截止日期（训练后无法感知新信息）和幻觉（对不确定问题倾向于编造答案）。微调可以注入新知识，但代价高昂且更新周期长。RAG 提供了一条轻量路径：**推理时动态检索外部知识，作为上下文注入 Prompt**，无需修改模型权重。

---

## 2. 基本架构

RAG 系统由两条流水线组成：离线索引流水线和在线推理流水线。

### 2.1 离线索引（Indexing Pipeline）

```
原始文档（PDF、HTML、数据库等）
        ↓
   文档解析与清洗
        ↓
   分块（Chunking）
        ↓
   Embedding 模型编码 → 向量
        ↓
   存入向量数据库
```

**分块策略**是索引阶段最关键的设计决策。块太大会引入噪声，块太小会丢失上下文。常见策略：

|策略|适用场景|缺点|
|---|---|---|
|固定长度分块（Token 数）|通用场景|可能截断语义单元|
|句子/段落边界分块|结构化文本|块大小不均匀|
|递归分块（LangChain 默认）|通用|实现复杂|
|语义分块（Semantic Chunking）|高精度场景|依赖额外模型|
|父子分块（Parent-Child）|需要兼顾精度与上下文|存储开销翻倍|

**Embedding 模型**将文本块编码为稠密向量，常用模型包括 `text-embedding-ada-002`（OpenAI）、`bge` 系列（BAAI）、`e5` 系列（Microsoft）。选型标准：向量维度、多语言支持、检索任务的 MTEB 榜单得分。

### 2.2 在线推理（Retrieval & Generation Pipeline）

```
用户提问 q
    ↓
Embedding 模型编码 q → 查询向量
    ↓
向量数据库相似度搜索（Top-K）
    ↓
检索到 K 个文档块
    ↓
（可选）Reranking
    ↓
拼入 Prompt → LLM 生成回答
```

---

## 3. 向量检索

### 3.1 相似度度量

最常用的是**余弦相似度**：

$$ \text{sim}(q, d) = \frac{q \cdot d}{|q| \cdot |d|} $$

也可使用内积（Inner Product）或 L2 距离，取决于 Embedding 模型的训练目标。

### 3.2 近似最近邻（ANN）

精确最近邻搜索复杂度为 $O(N \cdot d)$，百万级向量库下不可接受。实际使用近似算法：

|算法|原理|代表实现|
|---|---|---|
|HNSW|层次化小世界图|FAISS、Qdrant、Weaviate|
|IVF|倒排文件索引 + 聚类|FAISS|
|ScaNN|各向异性量化|Google ScaNN|
|DiskANN|磁盘友好的图索引|Microsoft DiskANN|

HNSW 是当前工程实践中最常用的方案，查询复杂度接近 $O(\log N)$，召回率高。

### 3.3 向量数据库选型

|数据库|特点|
|---|---|
|FAISS|Facebook 出品，纯库，无服务层，适合嵌入式场景|
|Qdrant|Rust 实现，支持 Payload 过滤，云原生|
|Weaviate|支持混合检索（向量 + BM25），GraphQL 接口|
|Chroma|轻量，适合原型开发|
|Pinecone|全托管云服务|
|pgvector|PostgreSQL 插件，适合已有 PG 基础设施的场景|

---

## 4. Reranking

Top-K 向量检索召回的是**近似相关**的文档，精度不足以直接喂给 LLM。Reranking 对召回集做精排：

$$ \text{score}(q, d_i) = \text{CrossEncoder}(q, d_i) $$

Cross-Encoder 将查询和文档**拼接后**送入模型，计算精确相关性分数，精度显著高于双塔 Embedding 的内积近似。代价是计算量为 $O(K)$ 次前向推理，因此只对 Top-K（通常 $K \leq 100$）做精排，再取 Top-$k'$（$k' \leq 10$）送入 LLM。

常用 Reranker：`bge-reranker`（BAAI）、`Cohere Rerank`、`ms-marco-MiniLM` 系列。

---

## 5. Prompt 构造

检索到的文档块拼入 Prompt 的典型结构：

```
System: 你是一个问答助手。请仅根据以下参考资料回答问题，
        若资料中没有相关信息，请明确说明。

参考资料：
[1] {chunk_1}
[2] {chunk_2}
...
[K] {chunk_K}

User: {用户问题}
```

**上下文窗口限制**是工程上的硬约束。块数 $K$ 与块大小的乘积必须小于模型的上下文长度减去问题和系统 Prompt 的长度。

**Lost-in-the-Middle 问题**：研究表明 LLM 对 Prompt 中间位置的内容注意力较弱，最相关的文档应放在开头或结尾。

---

## 6. RAG 的主要变体

### 6.1 Naive RAG

最基础的形式：单轮检索 → 生成，即第 2 节描述的标准流程。问题在于检索质量直接决定生成质量，查询表述不佳时召回效果差。

### 6.2 Advanced RAG

在 Naive RAG 基础上引入查询优化和后处理：

**Pre-retrieval 优化：**

- **Query Rewriting**：用 LLM 将用户原始问题改写为更适合检索的形式
- **HyDE（Hypothetical Document Embeddings）**：先让 LLM 生成一个假设性答案，用该答案的向量做检索，而非用问题向量，缓解问题-文档语义鸿沟
- **Query Expansion**：将一个问题扩展为多个子问题分别检索

**Post-retrieval 优化：**

- Reranking（见第 4 节）
- 上下文压缩（Context Compression）：用小模型过滤掉检索块中与问题无关的句子，减少噪声

### 6.3 Modular RAG

将检索、生成、验证等模块解耦，支持迭代检索和自我修正：

- **Iterative Retrieval**：生成过程中多轮检索，每轮检索基于上一轮的中间输出
- **Recursive Retrieval**：先检索摘要，再根据摘要检索细节（父子分块的应用场景）
- **Self-RAG**：模型自主决定是否需要检索，生成时插入特殊 Token（`[Retrieve]`、`[Relevant]` 等）控制流程

### 6.4 Graph RAG

微软提出的方案，将知识库构建为知识图谱而非平铺的文档块。检索时沿图的边做社区发现和路径遍历，适合需要跨文档推理的复杂问题（如"A 和 B 之间有什么关联"）。代价是索引构建成本极高。

---

## 7. 混合检索（Hybrid Search）

纯向量检索对精确关键词匹配效果差（如专有名词、代码片段、ID）。混合检索将稠密检索与稀疏检索结合：

$$ \text{score}_{\text{hybrid}}(q, d) = \alpha \cdot \text{score}_{\text{dense}}(q, d) + (1 - \alpha) \cdot \text{score}_{\text{sparse}}(q, d) $$

稀疏检索通常使用 **BM25**：

$$ \text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)} $$

其中 $f(t, d)$ 为词频，$|d|$ 为文档长度，$k_1$ 和 $b$ 为超参数。

融合方式也可用 **RRF（Reciprocal Rank Fusion）** 代替线性加权，避免分数量纲不一致的问题：

$$ \text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}, \quad k = 60 $$

---

## 8. 评估

RAG 系统的评估分为检索侧和生成侧两部分。

**检索侧指标：**

|指标|含义|
|---|---|
|Recall@K|Top-K 中包含相关文档的比例|
|Precision@K|Top-K 中相关文档占比|
|MRR|第一个相关文档的排名倒数均值|
|NDCG|考虑排名位置的归一化折损累积增益|

**生成侧指标（以 RAGAS 框架为例）：**

|指标|含义|
|---|---|
|Faithfulness|回答是否忠实于检索到的上下文，不超出其范围|
|Answer Relevance|回答是否切题|
|Context Precision|检索上下文中相关内容的占比|
|Context Recall|相关信息是否被检索上下文覆盖|

---

## 9. RAG vs. Fine-tuning

|维度|RAG|Fine-tuning|
|---|---|---|
|知识更新|更新知识库即可，实时|需重新训练，周期长|
|幻觉控制|有据可查，可溯源|仍可能幻觉|
|私有知识注入|直接入库|需准备训练数据|
|推理成本|增加检索延迟|无额外延迟|
|适用场景|知识密集型、需溯源|风格迁移、格式规范、领域适配|

两者不互斥。常见的生产实践是：Fine-tuning 让模型学会领域的表达方式和输出格式，RAG 提供具体的事实性知识。

---

## 10. 常见面试题

**Q1：RAG 和 Fine-tuning 的区别是什么？分别适用于什么场景？**

Fine-tuning 将知识编码进模型权重，推理时无需外部依赖，适合风格迁移、格式规范、领域术语适配；缺点是知识更新需重新训练，成本高。RAG 在推理时动态检索外部知识库，知识更新只需更新库，可溯源，适合知识密集型、时效性强、需要引用来源的场景；缺点是增加检索延迟，且回答质量受检索质量约束。两者不互斥，生产中常结合使用。

---

**Q2：RAG 系统中，分块策略（Chunking）如何选择？块大小如何权衡？**

块太大：单块包含过多无关内容，检索时噪声大，且占用更多上下文窗口。块太小：单块缺乏足够上下文，模型难以基于碎片化信息生成完整回答。实践中常用父子分块（Parent-Child Chunking）：小块用于检索（精度高），检索命中后返回其父块（上下文完整）送给 LLM。块大小通常需要根据具体文档结构和 Embedding 模型的输入长度限制实验调优，没有通用最优值。

---

**Q3：向量检索召回率不高时，有哪些优化手段？**

- **查询侧**：Query Rewriting（改写问题）、HyDE（生成假设文档再检索）、Query Expansion（拆分为多个子问题）
- **索引侧**：优化分块策略、提升 Embedding 模型质量、引入混合检索（向量 + BM25）
- **召回后**：加入 Reranker 精排，提升 Top-$k'$ 的精度
- **数据侧**：检查知识库覆盖范围，补充缺失文档

---

**Q4：什么是 HyDE？解决了什么问题？**

HyDE（Hypothetical Document Embeddings）：先让 LLM 根据用户问题生成一个假设性答案（不要求准确），再用该假设答案的向量去检索知识库，而非直接用问题向量检索。解决的问题是**问题与文档之间的语义鸿沟**——用户问题的表述形式与知识库文档的表述形式往往差距较大，假设答案的语义空间与文档更接近，因此检索效果更好。

---

**Q5：Lost-in-the-Middle 问题是什么？如何缓解？**

研究发现 LLM 对 Prompt 中间位置的内容注意力显著弱于开头和结尾，导致即使相关文档被检索到，若被放在上下文中间也可能被忽略。缓解方法：将最相关的文档放在 Prompt 的开头或结尾；减少无关文档数量（Reranking + Context Compression）；使用支持更长上下文且注意力更均匀的模型。

---

**Q6：RAG 系统如何评估？**

分检索侧和生成侧两部分。检索侧用 Recall@K、Precision@K、MRR、NDCG 评估召回质量。生成侧常用 RAGAS 框架，核心四指标：Faithfulness（回答是否忠实于检索上下文）、Answer Relevance（回答是否切题）、Context Precision（上下文中相关内容占比）、Context Recall（相关信息是否被覆盖）。端到端评估也可用人工标注的问答对做精确匹配或 LLM-as-Judge。

---

**Q7：混合检索（Hybrid Search）是什么？为什么比纯向量检索效果更好？**

混合检索将稠密向量检索（Dense Retrieval）和稀疏检索（Sparse Retrieval，通常是 BM25）的结果融合。纯向量检索对精确关键词匹配效果差，例如专有名词、代码、型号、ID 等在语义空间中可能与问题距离较远；BM25 对精确词匹配有天然优势。两者互补：向量检索捕捉语义相似性，BM25 捕捉词汇精确匹配。融合方式常用线性加权或 RRF（Reciprocal Rank Fusion），RRF 不依赖分数量纲，更鲁棒。

---

**Q8：Reranker 和 Embedding 模型的区别是什么？为什么不直接用 Reranker 做检索？**

Embedding 模型（双塔结构）：查询和文档分别独立编码为向量，检索时做向量内积，支持离线预计算文档向量，检索复杂度低，适合大规模召回。Reranker（Cross-Encoder）：将查询和文档拼接后送入模型，精度更高，但每次推理都需要重新计算，无法离线预计算，复杂度为 $O(K)$ 次前向推理。若直接用 Reranker 对全库做精排，百万级文档下延迟不可接受。因此工程上分两阶段：Embedding 大规模召回 Top-K，Reranker 对 Top-K 精排。

---

**Q9：RAG 中如何处理多跳推理（Multi-hop Reasoning）问题？**

单轮检索无法处理需要跨多个文档推理的问题（如"A 的 CEO 和 B 的 CEO 是同一所大学毕业的吗"）。解决方案：迭代检索（Iterative Retrieval），每轮生成中间结论后再次检索；Graph RAG，将知识库构建为知识图谱，沿图边做路径遍历；Query Decomposition，将复杂问题拆解为多个子问题分别检索后合并。

---

**Q10：RAG 系统上线后效果不好，排查思路是什么？**

按流水线逐段排查：

1. **检索侧**：抽样问题，查看 Top-K 是否包含正确文档。若召回率低，检查分块策略、Embedding 模型、是否需要混合检索或 Query Rewriting。
2. **Reranking 侧**：查看精排后 Top-$k'$ 的质量，是否将相关文档排在前列。
3. **Prompt 侧**：检查上下文拼接格式，相关文档位置是否在开头/结尾，是否有无关噪声文档混入。
4. **生成侧**：若检索内容正确但回答仍有误，问题出在 LLM 的指令遵循能力或上下文利用能力，考虑换模型或调整 System Prompt。
5. **数据侧**：检查知识库覆盖范围，是否存在文档质量差、格式混乱、解析错误等问题。
