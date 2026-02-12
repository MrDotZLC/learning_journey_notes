# ❌ 坑 1：Calibration 数据不匹配
问题：
- 用短文本校准长文本模型
- token 分布差异大
后果：
❌ 激活范围严重偏差  
❌ attention logits 爆炸

---
# ❌ 坑 2：Sequence length 不一致
Transformer 激活范围与：
seq_len↑→variance↑seq\_len ↑ → variance ↑seq_len↑→variance↑
校准用 32 token，部署 2k token：
❌ dynamic range 失效

---
# ❌ 坑 3：Softmax / LayerNorm 被 INT8
这些层：
- 对数值微扰极敏感
- clipping / rounding 放大
典型策略：
✅ 保留 FP16 / FP32

---
# ❌ 坑 4：Outlier Channel
LLM 常见：
`少数 channel 幅值极大`
影响：
❌ scale 被拉大  
❌ 有效 bit 利用率低
解决：
✅ SmoothQuant  
✅ AWQ / GPTQ  
✅ per-channel weight scaling

---
# ❌ 坑 5：Residual Add 放大量化噪声
残差结构：
x+f(x)x + f(x)x+f(x)
误差叠加：
❌ 噪声累积  
❌ 深层精度下降

---
# ❌ 坑 6：只看 MinMax
LLM 激活长尾：
❌ absmax 极不稳定
必须：
✅ KL / Percentile / MSE 校准

---
# 🎯 Transformer INT8 校准核心结论
---
## ✅ 真正难点
不是权重，而是：
👉 **Activation Range Explosion**
来源：
- LayerNorm
- Attention
- Residual
---
## ✅ 有效策略组合
|技术|作用|
|---|---|
|KL Calibration|稳定 threshold|
|Percentile|抑制 outlier|
|Per-channel Weight|降低权重误差|
|SmoothQuant|压缩激活范围|
|Mixed Precision|保护敏感层|

---