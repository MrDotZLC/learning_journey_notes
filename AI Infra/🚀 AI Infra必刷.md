# 第 1 阶段（1 周）：基础概念理解
学习材料：
- “什么是 Transformer”
- “自注意力如何工作”
- “什么是 KV Cache”
- “prefill vs decode”
- “显存带宽瓶颈是什么”

---

# 🎯 第 2 阶段（1–2 周）：CUDA 必备知识

只需要学：
- 线程模型（block / grid / warp）
- memory hierarchy
- 写简单 kernel（如 add、softmax）
- nsys 基本 profiling

不需要：
- 写复杂 kernel
- 训练模型
- 深入矩阵乘法优化

---

# 🎯 第 3 阶段（3–4 周）：LLM Serving（核心）

学：
- 动态批处理（Continuous Batching）
- KV Cache 分块与管理
- PagedAttention 原理
- 一个推理 loop 怎么写？
- 如何给多个用户同时生成 token？

你可以简单想成：

> 写一个高并发 C++ Server + GPU 调度器 + 显存管理器。

这是你的领域优势！

---

# 🎯 第 4 阶段（2–4 周）：做一个简历项目（必须）

## 最推荐的项目：

**“迷你版 vLLM：用 C++ + CUDA 实现一个 LLM 推理服务”**
功能：
- 请求队列
- prefill + decode
- KV Cache 分块
- 动态批处理
- gRPC API

这个项目做完你就可以投：
- AI Infra
- GPU 推理工程师
- 模型服务工程师
- LLM Serving 工程师