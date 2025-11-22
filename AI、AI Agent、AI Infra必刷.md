## **Day 1：LLM 推理底层必修**
目标：补齐理论  
重点：Attention、KV cache、RoPE、sampling。  
产出：写个 50 行的最小 GPT 推理代码（我可以给你模板）。

---

## **Day 2：GPU / vLLM / TensorRT 基础**
- 安装 vLLM
- 跑一遍 profiling
- 看 kernel 吞吐
重点学习：Continuous Batching。
---

## **Day 3：写一个“小号 Serving”**
用你熟悉的 C++/Python：
`请求队列 → batch 合并 → token streaming`

---

## **Day 4：RAG & Agent**

实现工具调用 + RAG 小管线。
---

## **Day 5：项目组合拳（重点）**

我会建议你构建：
### ⭐ **项目1：小型 LLM 服务框架（核心亮点工程）**
组件：
- Gateway
- Scheduler
- Worker（vLLM 多实例）
- Token 限速器
- KV 监控

这个项目面试有直接加成（AI Infra 岗必问）。

---

### ⭐ **项目2：多工具 Agent（偏产品）**
支持：
- 搜索
- 计算器
- RAG QA
- 文件解析
---