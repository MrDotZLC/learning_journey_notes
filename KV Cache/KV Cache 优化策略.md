## 1. KV Cache 基础回顾

### 1.1 内存占用量化

设模型层数 $L$，KV head 数 $n_{kv}$，head 维度 $d_h$，序列长度 $T$，数值精度占用 $b$ 字节：

$$ M_{\text{KV}} = 2 \cdot L \cdot n_{kv} \cdot d_h \cdot T \cdot b \quad \text{(bytes)} $$

以 LLaMA-3-70B（$L=80,\ n_{kv}=8,\ d_h=128,\ b=2$，BF16）在 $T=128\text{K}$ 下：

$$ M_{\text{KV}} = 2 \times 80 \times 8 \times 128 \times 128{,}000 \times 2 \approx 42 \text{ GB} $$

模型权重约 140 GB，KV Cache 已达权重量级的 **30%**；批次越大、序列越长，此比例持续攀升。

### 1.2 推理阶段与瓶颈定性

|阶段|计算特征|主要瓶颈|
|---|---|---|
|Prefill|所有 Prompt Token 并行 QKV 计算|计算密度（Compute-bound）|
|Decode|逐 token 生成，读取全量 KV Cache|内存带宽（Memory bandwidth-bound）|

Decode 阶段的 Arithmetic Intensity（AI）为：

$$ \text{AI} = \frac{2 \cdot T \cdot d}{2 \cdot (T + n_{\text{param}}) \cdot d \cdot b} \approx \frac{T}{T + n_{\text{param}}} \cdot \frac{1}{b} \ll \text{Ridge Point} $$

即 Decode 天然是 Memory-bound，降低 KV Cache 的字节数直接降低带宽压力并提升吞吐。

---

## 2. 优化策略分类体系

KV Cache 优化策略按作用层次分为三级：

```
KV Cache 优化
├── Token-Level（操作已有 Cache 内的 Token 粒度）
│   ├── 2.1 Token Eviction（驱逐）         → 文档 01
│   ├── 2.2 KV Quantization（量化）        → 文档 02
│   ├── 2.3 Low-Rank Decomposition（低秩） → 文档 03
│   └── 2.4 KV Merging（合并）             → 文档 03
│
├── System-Level（系统与存储管理）
│   ├── 3.1 PagedAttention / 内存管理      → 文档 04
│   ├── 3.2 Prefix Caching / RadixAttention→ 文档 04
│   └── 3.3 KV Offloading（分级卸载）      → 文档 04
│
└── Model-Level（架构级，需预训练支持）
    ├── 4.1 MQA / GQA                       → 文档 05
    ├── 4.2 MLA（Multi-head Latent Attention）→ 文档 05
    └── 4.3 Cross-Layer KV Sharing          → 文档 05
```

---

## 3. 正交性与组合潜力

各策略之间大多数情况下正交，可叠加：

|组合|效果|典型代表|
|---|---|---|
|Eviction + Quantization|条目数↓ + 每条目字节数↓|ALISA (INT8 + 稀疏)|
|Eviction + Low-Rank|结构压缩 + 维度压缩|LESS|
|Quantization + Low-Rank|表示压缩双重叠加|GEAR|
|PagedAttention + Quantization|碎片消除 + 位宽压缩|vLLM FP8|
|Prefix Cache + Eviction|命中则复用，未命中则驱逐冗余|SGLang + SnapKV|
|MLA + Quantization|潜在向量更小，量化误差更低|SnapMLA (SGLang-FluentLLM)|

CommonKV（2025）报告称叠加低秩、量化与驱逐后可达 **98%** 压缩率且性能无显著下降。

---

## 4. 各文档导航

|文档编号|主题|核心策略|
|---|---|---|
|**01**|Token Eviction 驱逐策略|StreamingLLM, H2O, SnapKV, PyramidKV, AdaKV, CAKE, NACL|
|**02**|KV Cache 量化|KIVI, KVQuant, KVTuner, KITTY, FP8/NVFP4, GEAR|
|**03**|低秩分解与 KV 合并|Palu, xKV, CommonKV, CaM, KVMerger, D2O|
|**04**|系统与存储优化|PagedAttention, Prefix Caching, RadixAttention, Offloading|
|**05**|模型架构级优化|MQA, GQA, MLA, CLA, Cross-Layer Sharing|

---

## 5. 性能对比参考（LLaMA 系模型，同等硬件）

> 【图示占位】：横坐标为 KV Cache 内存压缩比（1× = Full Cache），纵坐标为 LongBench 平均分，标注各策略的 Pareto 前沿，包括 H2O、SnapKV、PyramidKV、KIVI-2bit、KVQuant-3bit、MLA。

|策略类别|典型压缩比|精度损失|实现复杂度|
|---|---|---|---|
|Token Eviction (20% budget)|5×|中等|低|
|INT8 Quantization|2×|极小|低|
|INT4 Quantization|4×|小|中|
|INT2 Quantization|8×|中等|高|
|Low-Rank (Palu, r=50%)|2×|小|中|
|xKV Cross-Layer SVD|6–8×|小|中|
|GQA (g=8)|8× vs MHA|极小|需重训|
|MLA|~13× vs MHA|可超 MHA|需重训|
|叠加（Eviction+INT4+低秩）|20–50×|中等|高|
