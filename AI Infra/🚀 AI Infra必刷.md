# 第 1 阶段（1 周）：基础概念理解
学习材料：
- “什么是 Transformer”[[🚀 Transformer解析]]
- “自注意力如何工作”[[注意力和自注意力（Attention vs Self-Attention）]]
- “什么是 KV Cache”[[KV Cache介绍]]
- “prefill vs decode”[[🚀 大模型推理流程]]第3章
- “显存带宽瓶颈是什么”[[🚀 大模型推理优化技术（显存带宽瓶颈优化）]]

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

- [x] 1. **AI Infra 最基础版本：从零讲什么是 Transformer？** ✅ 2025-12-10

- [ ] 2. **从零讲 GPU（连 warp 都不会也行）**

- [x] 3. **从零讲 KV Cache（最重要的概念）**

- [ ] 4. **从零讲 Continuous Batching（怎么并发跑多个用户）**

- [ ] 5. **迷你 vLLM 项目的代码结构 & 每个模块如何写**

- [ ] 6. **AI Infra 面试题（含详细答案、不需要背景）**