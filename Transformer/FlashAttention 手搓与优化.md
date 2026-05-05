[FlashAttention 详细介绍](FlashAttention%20详细介绍.md)
[FlashAttention 算子解析](FlashAttention%20算子解析.md)

## 一、FlashAttention 核心思想

将 $K$、$V$ 分块，对每个 Q tile 用 **online softmax** 流式处理所有 KV tile，避免存储 $S[N,N]$：

$$ \text{HBM 访问量} = O(N \cdot d) \quad \text{（线性，无 } N^2 \text{ 项）} $$

online softmax 递推公式（处理第 $t$ 个 KV tile 时）：

$$ m^{(t)} = \max\left(m^{(t-1)},\ \max_j s^{(t)}_j\right) $$

$$ l^{(t)} = e^{m^{(t-1)} - m^{(t)}} \cdot l^{(t-1)} + \sum_j e^{s^{(t)}_j - m^{(t)}} $$

$$ O^{(t)} = e^{m^{(t-1)} - m^{(t)}} \cdot O^{(t-1)} + \sum_j e^{s^{(t)}_j - m^{(t)}} \cdot V_j $$

最终 $O = O^{(T)} / l^{(T)}$。

## 二、算子实现
### 2.1 flash_attn_v1_naive

