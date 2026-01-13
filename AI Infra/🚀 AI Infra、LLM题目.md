🎯 Part 1：AI Infra 20 题（含参考答案）
---------------------------------------------------------
1. 什么是 KV Cache？为什么能加速推理？
答案：
		Decoder-only 模型每生成一个 token 都要重新计算之前所有 token 的 K/V。
		KV Cache 保存历史 tokens 的 K/V，从第二 token 开始只算当前 query，时间复杂度从 O(n²) → O(n)。

2. vLLM 为什么快？核心优化是什么？
答案：
		PagedAttention：将 KV 存到类似“虚拟内存页”的块中，减少碎片。
		Continuous Batching：动态合并请求。
		Block-wise KV 管理：避免显存碎片。
		CUDA kernels 优化。

3. Continuous Batching（连续批处理）如何实现？
答案：
		将不同请求的 decoding step 对齐为“token step”
		统一在每一步对所有 active sequences 做 forward
		新请求随时插入，结束的随时移除
		要求 attention mask 能动态扩展
		收益：吞吐量显著提升。

4. Speculative Decoding 原理？
答案：
		用一个草稿模型（小模型）预测一批 token，再用大模型批量验证，减少大模型调用次数。
		吞吐↑ 延迟↓

5. Tensor 并行 vs Pipeline 并行区别？
答案：
		Tensor 并行（横向拆矩阵）：多卡同时计算一个层。
		Pipeline 并行（纵向拆网络）：顺序执行不同层。
		前者减少单层计算，后者减少显存占用。

6. KV Cache OOM 如何处理？（3 种方式）
答案：
		复用/回收 KV（LRU block reuse）
		限制最大并发请求数
		量化 / 分布式 KV 分片（tensor parallel）

7. Token 限速器怎么设计？
答案：
		基于漏桶 / 令牌桶，对流量按 tokens/s 限制：
		每个请求 token_count 累加
		调度器统一扣减 bucket
		不满足则排队 / 拒绝

8. 如何构建异步 batch queue？
答案：
		多请求通过 channel/queue 进入等待池
		定时器（如 2ms）触发 batch assemble
		构建 batch → 推送GPU → 返回序列化结果

9. RAG chunk 多大合适？为什么？
答案：
		通常 200–500 tokens。
		太小：语义不完整。
		太大：embedding cost ↑，召回变差，噪声多。

10. Hybrid Search 是什么？
答案：
		Dense（向量） + Sparse（BM25）融合；
		可通过 weighted sum 或 rank fusion。

11. 向量数据库为什么需要 HNSW？
答案：
		因为暴力搜索是 O(n)，HNSW 做分层图搜索，将复杂度降到 O(log n)。

12. Reranker 在 RAG 中作用是什么？
答案：
		把召回的 top-k 文档，按 query-document 的 matching 分数重新排序，提高相关性。

13. 为什么 LLM 需要 Streaming？
答案：
		降低延迟，改善 UX，同时让多 token 并行处理更自然。

14. 长上下文推理为什么慢？如何优化？
答案：
		Attention O(n²)；
		优化：RoPE scaling、KV cache 复用、分块 attention、长文模型（例如 Mamba、Gated-SSM）。

15. 量化 q4 和 q8 的差别是什么？
答案：
		q4：显存减半，损失略大
		q8：更稳定，但压缩不如 q4
		推理速度取决于 kernel 和算子融合。

16. LLM 服务架构核心组件？
答案：
		Gateway → Scheduler → Worker（模型） → KV Manager。

17. 如何做模型的负载均衡？
答案：
		加权轮询
		token-level 流控
		根据 KV 占用做动态调度（smart routing）

18. Prompt Cache 是什么？
答案：
		相同 prompt 复用前若干层 KV，减少重复计算。

19. 如何避免显存碎片？
答案：
		PagedAttention / block allocation / static shape kernels。

20. 如何写一个最小推理引擎？步骤？
答案：
		tokenizer
		attention
		sampling（top-k/top-p）
		KV cache
		decode 循环
		streaming 输出

---------------------------------------------------------
🎯 Part 2：LLM 理论 20 题（含答案）
---------------------------------------------------------
41. Transformer 为什么有效？
自注意力可以捕捉全局依赖，训练稳定，易做并行。

42. RoPE 的本质？
利用复数旋转编码位置，使 attention 可外推，并保留相对位置信息。

43. 多头的意义？
让模型捕捉多种特征子空间，提高表达能力。

44. 为什么参数越大，推理能力越强？
更大的 capacity + deeper patterns + implicit knowledge。

45. Pretrain → SFT → RLHF 逻辑？
Pretrain：基本世界知识
SFT：模仿行为
RLHF：奖励对齐（偏好）

46. 为什么要做混合精度训练？
减少显存 + 加速 + 保持精度（FP16/BF16）

47. LLM 训练为什么要大 batch？
更稳定的梯度估计，学习信号更强。

48. Attention O(n²) 的根本原因？
每个 token 对所有 token 计算相似度。

49. Top-k / Top-p 的区别？
Top-k：只保留概率最高 k 个
Top-p：保留概率和超过 p 的最小集合

50. 为什么 LLM 不适合严格数值推理？
LLM 本质是概率模型，非符号系统。

51. 为什么位置编码不能去掉？
否则模型无法区分 token 序列（无序的）。

52. 为什么 decoder-only 模型成为主流？
简单、统一架构、生成稳定、训练成本低。

53. 词表大小越大越好吗？
不一定；影响 embedding 参数、训练速度。

54. 为什么需要 gradient checkpointing？
节省显存，代价是计算变多。

55. MLP block 在 Transformer 中作用是什么？
在 attention 后做特征变换和非线性表达。

56. 为什么模型需要 layer norm？
稳定训练，防止梯度爆炸。

57. 为什么大模型能进行思维链推理？
复杂模式可以在海量训练数据中学习。

58. 为什么模型 increasing context 会让性能下降？
长距离 attention 更难学习 + 概率稀释。

59. 为什么微调后可能出现遗忘（catastrophic forgetting）？
参数被局部任务覆盖，破坏原有分布。

60. LLM “沉默 token” 是什么？有什么用？
特殊 token，模型难以输出，用于控制输出范围或 mask。

---------------------------------------------------------
🚀 Part 4：基于你“3 年 C++后端经验”的定制突击路线
---------------------------------------------------------

背景：
✔ 扎实工程能力
✔ 熟悉高并发、内存管理、多线程
✔ C++ → vLLM / TensorRT-LLM / Serving 方向天然优势
最适合的岗位：
✅ AI Infra / LLM 推理工程师
✅ Agent 平台 / 多工具调度后端
✅ AI 后端工程 / LLM 应用工程师