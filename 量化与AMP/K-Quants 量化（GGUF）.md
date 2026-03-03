## 1. 背景与问题定义
K-Quant 是 llama.cpp 引入的**非均匀、分组混合精度**量化方案，解决了朴素均匀量化（如 Q4_0）在权重分布非均匀时的精度损失问题。
**核心思路**：在一个 **super-block** 内，对不同 sub-block 使用不同的量化尺度（scale），尺度本身也被量化存储。

---

## 2. 均匀量化基础（对比参照）
给定浮点权重 $x \in \mathbb{R}$，均匀h量化到 $b$ bit：
$$ q = \text{round}\left(\frac{x}{s}\right) + z $$
$$ \hat{x} = s \cdot (q - z) $$
其中：
- $s$ = scale（尺度因子）
- $z$ = zero-point（零点偏移）
- $q \in [0, 2^b - 1]$
**量化误差**：$\epsilon = x - \hat{x}$，最大误差为 $\frac{s}{2}$。
**问题**：全局单一 $s$ 无法适应局部权重分布差异。

---

## 3. K-Quant 结构定义
### 3.1 Block 层次结构
```
Super-Block (e.g., 256 elements)
├── Sub-Block 0 (32 elements) → scale s_0
├── Sub-Block 1 (32 elements) → scale s_1
├── ...
└── Sub-Block 7 (32 elements) → scale s_7
```
以 **Q4_K** 为例（实际实现参数）：

|参数|值|
|---|---|
|Super-block size|256|
|Sub-block size|32|
|权重量化位宽|4 bit|
|Scale 量化位宽|6 bit|
|Min 量化位宽|6 bit|

### 3.2 存储布局
每个 super-block 存储：
- $d$：super-block 的 FP16 scale（量化 scales 的 scale）
- $d_{min}$：super-block 的 FP16 min-scale
- ${s_i}_{i=0}^{7}$：8 个 6-bit 量化后的 sub-block scales
- ${m_i}_{i=0}^{7}$：8 个 6-bit 量化后的 sub-block mins
- ${q_j}$：4-bit 量化权重

---

## 4. 量化过程推导
### 4.1 Sub-block Scale 计算
对第 $i$ 个 sub-block，权重集合 $\mathbf{x}_i = {x_{i,0}, \ldots, x_{i,31}}$：
$$ x_{max,i} = \max_j(x_{i,j}), \quad x_{min,i} = \min_j(x_{i,j}) $$
理想 scale 与 min：
$$ s_i^* = \frac{x_{max,i} - x_{min,i}}{2^b - 1}, \quad m_i^* = x_{min,i} $$
权重量化：
$$ q_{i,j} = \text{round}\left(\frac{x_{i,j} - m_i^*}{s_i^*}\right) $$
$$ q_{i,j} \in [0, 15] \quad (b=4) $$
### 4.2 Super-block 对 Scales 的二次量化
收集所有 sub-block 的 $s_i^*$，对其再做均匀量化到 6 bit：
$$ d = \frac{\max_i(s_i^*)}{2^6 - 1} = \frac{\max_i(s_i^*)}{63} $$
$$ \tilde{s}_i = \text{round}\left(\frac{s_i^*}{d}\right), \quad \tilde{s}_i \in [0, 63] $$
对 mins 同理：
$$ d_{min} = \frac{\max_i(m_i^*)}{63} $$
$$ \tilde{m}_i = \text{round}\left(\frac{m_i^*}{d_{min}}\right), \quad \tilde{m}_i \in [0, 63] $$
> $d$、$d_{min}$ 以 **FP16** 存储，$\tilde{s}_i$、$\tilde{m}_i$ 以 **6-bit integer** 存储。

---

## 5. 反量化过程推导（Dequantization）
### 5.1 还原 sub-block scale 与 min
$$ \hat{s}_i = d \cdot \tilde{s}_i $$
$$ \hat{m}_i = d_{min} \cdot \tilde{m}_i $$
### 5.2 还原权重
$$ \hat{x}_{i,j} = \hat{s}_i \cdot q_{i,j} + \hat{m}_i $$
展开后完整误差链：
$$ \hat{x}_{i,j} = \underbrace{(d \cdot \tilde{s}_i)}_{\hat{s}_i} \cdot q_{i,j} + \underbrace{(d_{min} \cdot \tilde{m}_i)}_{\hat{m}_i} $$
### 5.3 完整误差分析
$$ \epsilon_{total} = \underbrace{(s_i^* - \hat{s}_i) \cdot q_{i,j}}_{\text{scale量化误差}} + \underbrace{(m_i^* - \hat{m}_i)}_{\text{min量化误差}} + \underbrace{s_i^* \cdot (q_{i,j} - q_{i,j}^{true})}_{\text{权重舍入误差}} $$
其中：
- Scale 量化误差上界：$|s_i^* - \hat{s}_i| \leq \frac{d}{2}$
- Min 量化误差上界：$|m_i^* - \hat{m}_i| \leq \frac{d_{min}}{2}$

---

## 6. 位宽与压缩率计算
以 Q4_K（256 elements per super-block）为例：

|存储项|数量|位宽|总 bits|
|---|---|---|---|
|权重 $q$|256|4|1024|
|sub-block scales $\tilde{s}_i$|8|6|48|
|sub-block mins $\tilde{m}_i$|8|6|48|
|super-block $d$|1|16|16|
|super-block $d_{min}$|1|16|16|
|**合计**|||**1152**|

**等效位宽**：
$$ \text{bpw} = \frac{1152}{256} = 4.5 \text{ bits/weight} $$
对比 FP16（16 bpw），压缩率：
$$ \text{compression} = \frac{16}{4.5} \approx 3.56\times $$
---
## 7. K-Quant 系列对比

|类型|Super-block|Sub-block|权重位宽|Scale位宽|bpw|
|---|---|---|---|---|---|
|Q2_K|256|16|2|4|2.5625|
|Q3_K|256|16|3|6|3.4375|
|Q4_K|256|32|4|6|4.5|
|Q5_K|256|32|5|6|5.5|
|Q6_K|256|16|6|8|6.5625|

---

## 8. 伪代码实现
```python
# Q4_K 量化伪代码
def quantize_q4_k(weights: np.ndarray) -> Q4K_Block:
    """
    weights: shape (256,)，一个 super-block
    """
    SUPER_BLOCK = 256
    SUB_BLOCK   = 32
    N_SUB       = SUPER_BLOCK // SUB_BLOCK  # = 8
    B_WEIGHT    = 4   # 权重位宽
    B_SCALE     = 6   # scale 位宽
    s_star = np.zeros(N_SUB)  # 理想 sub-block scale
    m_star = np.zeros(N_SUB)  # 理想 sub-block min
    q_raw  = np.zeros((N_SUB, SUB_BLOCK), dtype=np.int32)
    # Step 1: 计算每个 sub-block 的 scale 和 min
    for i in range(N_SUB):
        blk = weights[i * SUB_BLOCK : (i+1) * SUB_BLOCK]
        x_max = blk.max()
        x_min = blk.min()
        s_star[i] = (x_max - x_min) / (2**B_WEIGHT - 1)  # = /15
        m_star[i] = x_min
        if s_star[i] > 0:
            q_raw[i] = np.round((blk - m_star[i]) / s_star[i]).clip(0, 15)
    # Step 2: 对 scales 和 mins 做二次量化（6-bit）
    d     = s_star.max() / (2**B_SCALE - 1)  # = /63, FP16存储
    d_min = m_star.max() / (2**B_SCALE - 1)  # = /63, FP16存储
    s_q = np.round(s_star / d    ).clip(0, 63).astype(np.uint8)  # 6-bit
    m_q = np.round(m_star / d_min).clip(0, 63).astype(np.uint8)  # 6-bit
    return Q4K_Block(d=d, d_min=d_min, scales=s_q, mins=m_q, quants=q_raw)
def dequantize_q4_k(block: Q4K_Block) -> np.ndarray:
    """反量化，还原 FP32 权重"""
    weights = np.zeros(256)
    for i in range(8):
        s_hat = block.d     * block.scales[i]  # 还原 sub-block scale
        m_hat = block.d_min * block.mins[i]    # 还原 sub-block min
        start = i * 32
        weights[start:start+32] = s_hat * block.quants[i] + m_hat
    return weights
```

---

## 9. 关键设计决策
**为何用分层量化而非直接存 FP16 scales？**

若直接存 8 个 FP16 scales：$8 \times 16 = 128$ bits overhead  
K-Quant 方案：$8 \times 6 \times 2 + 2 \times 16 = 128$ bits overhead

→ overhead 相同，但 K-Quant 通过 $d$ 的共享，使 scales 之间具有**相对精度保障**，减少 scales 的相对误差（absolute scale 误差被 $d$ 的 FP16 精度兜底）。
