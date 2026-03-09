# 1. 为什么需要 KV Cache？
## 1.1 自回归推理的问题
LLM（如 GPT/Llama）是 **自回归生成**：  
第 t 步生成 token 需要用到 **全部 0…t−1 的 token**。
对于每一步，都要：
1. 重新算一次 Self-Attention
2. 用当前 query 去跟所有历史 key/value 做 attention

如果不做缓存，意味着：
- 第 1 步：算 1 个 token
- 第 2 步：算 2 个 token
- 第 3 步：算 3 个 token
- …
- 第 T 步：算 T 个 token

总复杂度：
$O(T^2)$

🔴 **T=4096** 时计算量巨大，推理根本快不起来。
# 2. KV Cache 核心思想（一句话）
> **第一次把所有历史 tokens 的 Key 和 Value 保存下来（按层缓存），后续生成步骤只需要计算当前 token 的 Key/Value，然后与缓存拼接即可。**
即：
$$K_{0:t} = [K_0, K_1, ..., K_{t}]$$
$$V_{0:t} = [V_0, V_1, ..., V_{t}]$$

注意：  
**Q 不能缓存**（因为每一步生成的 Q 不同）。
从 O(T²) → O(T)
# 3. KV Cache 的具体流程
## Step 1：用户输入 "Hello world"
模型执行 **prefill（预填）** 阶段：
对所有 token 计算 K、V，并缓存到每一层的 KV Cache。
## Step 2：开始生成第一个 token
只计算当前 token 的：
- Query（Q_t）
- Key（K_t）
- Value（V_t）
并且：
- K_cache = [历史Ks, K_t]
- V_cache = [历史Vs, V_t]
然后 attention 只需要：
$$Q_t \cdot K_{0:t}^T$$
这就避免重复算历史部分。
## Step 3：继续生成
重复过程，KVCache 逐步扩展：
```
Step t:
  Q_t
  K_t → append to KVCache
  V_t → append to KVCache
  Attention(Q_t, K_cache, V_cache)
```
KV Cache 越积越多，但每个 token 都只需一次 attention，而不是全序列 attention。
# 4. KV Cache 的内存结构
KV Cache 是 **每层** 单独维护的缓存：
假设：
- n 层 Transformer（如 32 / 48 / 80 层）
- h 个头（如 32、64）
- head_dim = 128
- sequence length = 4096

那么每层 KVCache 大小：
$$2 \times \text{seq\_len} \times h \times \text{head\_dim}$$
"2" 代表 Key + Value。
**不同层互不共享，独立缓存。**
## 实际内存消耗（举例）
以 Llama-7B、fp16 为例：
- hidden_dim = 4096
- head_dim = 128
- heads = 32

**单层 KV Cache** 大小 = 2 * 4096 * 32 * 128 * 2 bytes  
≈ **67MB**
如果 32 层：
32 * 67MB ≈ **~2.1GB**

KV Cache 是推理中非常大的显存消耗来源。
# 5. KV Cache 的性能收益
## Prefill 阶段
无变化，仍然是 full attention。
## Decode 阶段
每步只算 1 个 token 的 attention。
减少了：
- 90% 以上的注意力计算量
- 大量矩阵乘法
- GPU 计算占用

推理速度提升 **10-100 倍**。
因此：
> KV Cache 是 LLM 高速推理的根基。
# 6. KV Cache 的优化技术大全（全部）
以下是业界主流加速方案：
## 6.1 KV Cache 压缩（Caching Compression）
### ❶ FP16 → FP8 / FP4 / NF4
- Hugging Face / Nvidia Transformer Engine 已能用 FP8 KVCache。
- 甚至可以用 4 bit 存 KVCache（QLoRA 中的 NF4 技术）。
显存减少一半以上。
### ❷ 弱注意力 Token 丢弃（Token Eviction）
如 StreamingLLM、LM-Infinite：
- 对过去很远且 attention 权重极低的 token 执行裁剪（evict）。

能把上下文延长至 **无限**。
## 6.2 KV Cache 的存储优化
### ❶ Paged Attention（vLLM）
- 将 KVCache 存在 GPU 的"分页"内存中（类似操作系统 Page）。
- 支持动态扩展，不需要连续显存。
这是 vLLM 猜大火的关键。
### ❷ 完整重排（Reordering-free Attention）
新方法使得 **不用 reorder**，极大加速。
## 6.3 KV Cache 的访问优化
### ❶ Multi-Query Attention（MQA）
所有头共享 **一个 Key、一个 Value**：
- 大幅减少 KVCache 大小（降低头数倍数）
### ❷ Grouped-Query Attention（GQA）
介于 MQA 与 MHA 中间：
- 多个 Query 头共享同一组 KV 头
例如：
- 32 Q-heads
- 8 K/V-heads
KVCache缩小4倍。
大模型主流（GPT-4、Llama2/3、Mistral）都用 GQA。
## 6.4 序列切分带来的 KV 避免冗余计算
### ❶ Flash-Decoding
FlashAttention 的推理版：
- 更快的 Q⋅KT kernel
- 更高带宽利用
### ❷ Continuous batching（vLLM）
多个用户并行推理时共享 KVCache，有效提升吞吐量。
# 7. KV Cache 的局限性
- 显存占用极大（可达 4GB+）
- 随着生成长度增长速度线性增长
- 序列过长时访问变慢（bandwidth bound）
- Prefill 阶段仍然是 O(N²)，无法优化
- 多 GPU 下需要大量 KV 同步通信

因此出现了大量优化工作（GQA、分页、压缩、裁剪等）。
# 8. 总结
> **KV Cache = 推理阶段缓存所有历史 Key/Value。  
> 使得每次生成只需与缓存做一次 Attention，从 O(T²) → O(T)。**

带来的结果：
- 推理速度提升 10–100 倍
- 显存占用变大（但可压缩）
- 需要 paging / GQA / compression 等大量优化

KV Cache 是 LLM 推理的核心底层技术中最重要的一环。
