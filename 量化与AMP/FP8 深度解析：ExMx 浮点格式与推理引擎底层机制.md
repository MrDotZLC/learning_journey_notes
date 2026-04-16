## 1. FP8 格式设计背景

FP8（8-bit Floating Point）用于缓解大模型中的**带宽压力、缓存占用、Tensor Core 吞吐瓶颈**。

FP8 位宽拓扑约束：

$$ 1(\text{Sign}) + x(\text{Exponent}) + y(\text{Mantissa}) = 8 $$

设计折中：

|调整方向|影响|
|---|---|
|增大 $x$|扩展动态范围|
|增大 $y$|提升数值精度|

IEEE 754 在 8-bit 下表示效率有限 → 工业界采用 OFP8。

---

## 2. OCP OFP8 标准

现代 AI 加速器遵循：**OCP (Open Compute Project) – 8-bit Floating Point Specification (OFP8)**

硬件生态：

- NVIDIA Tensor Core（H100 / B200）
- AMD Matrix Core（MI300X）

偏离 IEEE 754 的工程动机：

|目标|解释|
|---|---|
|信息熵最大化|8-bit 极限受限|
|推理吞吐优先|异常路径代价高|
|硬件实现简化|Inf/NaN 逻辑昂贵|
|DNN 分布适配|非通用数值计算场景|

---

## 3. OFP8 格式族

### 3.1 E4M3FN（Finite Only）

结构：

- Exponent = 4 bits
- Mantissa = 3 bits
- Bias = 7
- 无 Infinity

设计哲学：舍弃 Inf，压缩 NaN，扩展有限数覆盖密度

典型用途：推理权重（Weights）、激活值（Activations）

优势：更高精度密度，推理数值分布适配性强

### 3.2 E5M2

结构：

- Exponent = 5 bits
- Mantissa = 2 bits
- Bias = 15
- 保留 Infinity

典型用途：训练梯度（Gradients）、宽动态范围张量

优势：抗溢出能力强；代价：精度降低

### 3.3 格式对比

|格式|$x$|$y$|Bias|最大有限值|机器精度 $\epsilon$|Inf|
|---|---|---|---|---|---|---|
|**E4M3FN**|4|3|7|±448|$2^{-3}=0.125$|❌|
|**E5M2**|5|2|15|±57344|$2^{-2}=0.25$|✅|

---

## 4. 数学基础：Bias 与解码模型 — 名词与符号注解

### 4.1 符号总表

第4章涉及的所有符号，按首次出现顺序列出：

| 符号                       | 类型     | 含义                                                                                         |
| ------------------------ | ------ | ------------------------------------------------------------------------------------------ |
| $x$                      | 格式参数   | Exponent 字段的 bit 数                                                                         |
| $y$                      | 格式参数   | Mantissa 字段的 bit 数                                                                         |
| $\text{bias}$            | 常数     | 指数偏置值，用于将无符号 $e_{\text{field}}$ 映射为有符号真实指数                                                 |
| $v$                      | 变量     | 浮点编码所代表的真实数值（decoded value）                                                                |
| $s$                      | bit 字段 | 符号位（Sign bit），$s \in {0,1}$，$0$ 为正，$1$ 为负                                                  |
| $e$ / $e_{\text{field}}$ | bit 字段 | 指数字段的无符号整数读数（$x$ bits，范围 $[0, 2^x - 1]$）                                                   |
| $E$                      | 变量     | 真实指数（Unbiased Exponent），$E = e_{\text{field}} - \text{bias}$                               |
| $b_i$                    | bit    | 尾数字段第 $i$ 个 bit，$b_1$ 为最高有效位（MSB），$b_y$ 为最低有效位（LSB）                                        |
| $m$                      | bit 字段 | 尾数字段整体（Mantissa / Significand field），共 $y$ bits                                            |
| $M$                      | 变量     | 有效数（Significand），含隐含位：Normal 时 $M = 1 + \sum b_i 2^{-i}$，Subnormal 时 $M = \sum b_i 2^{-i}$ |
| $E_{\text{sub}}$         | 常数     | Subnormal 数的有效指数，固定为 $1 - \text{bias}$                                                     |
| $\Delta_{\text{sub}}$    | 变量     | Subnormal 区间的 ULP（相邻两个可表示数之间的间距）                                                           |
| $v_{\text{sub\_max}}$    | 变量     | 最大正次正规数                                                                                    |
| $v_{\text{norm\_min}}$   | 变量     | 最小正规数（Normal 区间起点）                                                                         |

---

### 4.2 名词解释

#### 4.2.1 Bias（指数偏置）

**定义**：浮点格式将真实指数 $E$（可为负）编码为无符号整数 $e_{\text{field}}$，存储时加上偏置：

$$ e_{\text{field}} = E + \text{bias} \quad \Longleftrightarrow \quad E = e_{\text{field}} - \text{bias} $$

**动机**：硬件比较无符号整数比有符号整数更廉价。加入 bias 后，指数比较退化为普通无符号整数比较（浮点排序、max/min 等操作受益）。

**取值规则**：

$$ \text{bias} = 2^{x-1} - 1 $$

使 $e_{\text{field}}$ 的中点（$2^{x-1}-1$）对应真实指数 $E=0$，即 $2^0 = 1.0$。

E4M3FN（$x=4$）：$\text{bias}=7$，$e_{\text{field}}$ 范围 $[0,15]$，有效指数范围 $[-6, 8]$（去除特殊编码后）。

---

#### 4.2.2 Normal（正规数）

**定义**：$e_{\text{field}} \neq 0$ 且 $e_{\text{field}} \neq e_{\max\_reserved}$ 的浮点数。

**核心特征：隐含位（Implicit Leading 1）**

尾数字段只存储小数部分，整数位隐含为 `1`，不占存储空间：

$$ M = 1.\underbrace{b_1 b_2 \cdots b_y}_{\text{存储的 }y\text{ bits}} \quad = \quad 1 + \sum_{i=1}^{y} b_i \cdot 2^{-i} $$

**解码公式**：

$$ v = (-1)^s \cdot 2^{e_{\text{field}} - \text{bias}} \cdot \left(1 + \sum_{i=1}^{y} b_i \cdot 2^{-i}\right) $$

**示例**（E4M3FN，编码 `0 0110 100`）：

$$ s=0,\quad e_{\text{field}}=6,\quad m=100_2 $$

$$ E = 6-7 = -1,\quad M = 1 + 2^{-1} = 1.5,\quad v = 1.5 \times 2^{-1} = 0.75 $$

---

#### 4.2.3 Subnormal / Denormal（次正规数）

**触发条件**：

$$ e_{\text{field}} = 0, \quad m \neq 0 $$

**设计目的**：填补 0 与最小 Normal 之间的空洞，实现 Gradual Underflow（渐进式下溢），使数值趋近于 0 时精度平滑降低而非突然归零。

**隐含位变为 `0`**：此时不存在隐含 1，有效数为纯小数：

$$ M = 0.\underbrace{b_1 b_2 \cdots b_y}_{\text{存储的 }y\text{ bits}} \quad = \quad \sum_{i=1}^{y} b_i \cdot 2^{-i} $$

**有效指数固定为 $E_{\text{sub}} = 1 - \text{bias}$**（而非 $0 - \text{bias}$），原因见 §4.2.4。

**解码公式**：

$$ v = (-1)^s \cdot 2^{1-\text{bias}} \cdot \sum_{i=1}^{y} b_i \cdot 2^{-i} $$

---

#### 4.2.4 Gradual Underflow（渐进式下溢）与连续性

**问题背景**：若令 Subnormal 的有效指数为 $0 - \text{bias}$（与 $e_{\text{field}}=0$ 字面值一致），则：

$$ v_{\text{sub\_max}} = (1 - 2^{-y}) \cdot 2^{-\text{bias}} $$

$$ v_{\text{norm\_min}} = 1.0 \cdot 2^{1-\text{bias}} $$

二者之差：

$$ v_{\text{norm\_min}} - v_{\text{sub\_max}} = 2^{1-\text{bias}} - (1-2^{-y})\cdot 2^{-\text{bias}} = 2^{-\text{bias}}(1 + 2^{-y}) $$

这个间隙**远大于** Subnormal 区间的 ULP $\Delta_{\text{sub}} = 2^{-\text{bias}} \cdot 2^{-y}$，数轴出现断层。

**修正**：令 $E_{\text{sub}} = 1 - \text{bias}$，此时：

$$ v_{\text{norm\_min}} - v_{\text{sub\_max}} = 2^{1-\text{bias}} - (1-2^{-y})\cdot 2^{1-\text{bias}} = 2^{1-\text{bias}} \cdot 2^{-y} = \Delta_{\text{sub}} $$

间隙恰好等于一个 ULP，数轴连续。

---

#### 4.2.5 ULP（Unit in the Last Place）

**定义**：在给定指数区间内，相邻两个可表示浮点数之间的间距，即最低有效尾数 bit（LSB）对应的量级。

对于 Normal 数，指数为 $E$ 时：

$$ \text{ULP}_{\text{normal}}(E) = 2^{E} \cdot 2^{-y} = 2^{E-y} $$

对于 Subnormal 区间（指数固定为 $E_{\text{sub}} = 1 - \text{bias}$）：

$$ \Delta_{\text{sub}} = 2^{1-\text{bias}} \cdot 2^{-y} = 2^{1-\text{bias}-y} $$

**物理意义**：ULP 越小，该区间内的表示精度越高；ULP 越大，量化误差越大。

---

#### 4.2.6 机器精度 $\epsilon$（Machine Epsilon）

**定义**：1.0 处的 ULP，即满足 $1.0 + \epsilon \neq 1.0$ 的最小正数：

$$ \epsilon = 2^{-y} $$

代入：E4M3FN（$y=3$）$\epsilon = 0.125$；E5M2（$y=2$）$\epsilon = 0.25$。

**区别于最小可表示正数**（最小 Subnormal）：$\epsilon$ 描述精度分辨率，最小 Subnormal 描述动态范围下界，两者概念独立。## 5. 数值边界分析

### 5.1 特殊编码

#### 5.1.1 E4M3FN

$$ e_{\text{field}} = 1111_2, \quad m = 111_2 \quad \Rightarrow \quad \text{NaN（唯一 NaN 编码）} $$

无 Infinity 编码。

#### 5.1.2 E5M2

$$ e_{\text{field}} = 11111_2, \quad m = 00_2 \quad \Rightarrow \quad \pm\text{Inf} $$

$$ e_{\text{field}} = 11111_2, \quad m \neq 00_2 \quad \Rightarrow \quad \text{NaN} $$

### 5.2 最大有限值（E4M3FN）完整推导

最大有限编码（正值）：$s=0$，$e_{\text{field}}=1111_2=15$，$m=110_2$

> **注意**：$m=111_2$ 在 $e_{\text{field}}=1111$ 时为 NaN，故最大有限值的尾数取 $110_2$。

$$ E = e_{\text{field}} - \text{bias} = 15 - 7 = 8 $$

$$ M = 1 + 2^{-1} + 2^{-2} = 1 + 0.5 + 0.25 = 1.75 $$

$$ \text{Max}_{E4M3FN} = 1.75 \times 2^8 = 448 $$

工程阈值：

```cpp
constexpr float E4M3FN_MAX = 448.0f;
```

### 5.3 最大有限值（E5M2）完整推导

最大有限编码（正值）：$s=0$，$e_{\text{field}}=11110_2=30$，$m=11_2$

> **注意**：$e_{\text{field}}=11111$ 保留给 Inf/NaN。

$$ E = 30 - 15 = 15 $$

$$ M = 1 + 2^{-1} + 2^{-2} = 1.75 $$

$$ \text{Max}_{E5M2} = 1.75 \times 2^{15} = 57344 $$

---

## 6. 计算示例

### 6.1 E4M3FN 编码 → 数值解码

#### 6.1.1 正规数示例

**编码**：`0 0101 010`（$s=0$，$e_{\text{field}}=0101_2=5$，$m=010_2$）

$$ E = 5 - 7 = -2 $$

$$ M = 1 + 2^{-2} = 1.25 $$

$$ v = (+1) \cdot 2^{-2} \cdot 1.25 = \frac{1.25}{4} = 0.3125 $$

---

**编码**：`1 1000 011`（$s=1$，$e_{\text{field}}=1000_2=8$，$m=011_2$）

$$ E = 8 - 7 = 1 $$

$$ M = 1 + 2^{-2} + 2^{-3} = 1 + 0.25 + 0.125 = 1.375 $$

$$ v = (-1) \cdot 2^{1} \cdot 1.375 = -2.75 $$

#### 6.1.2 次正规数示例

**编码**：`0 0000 001`（$s=0$，$e_{\text{field}}=0$，$m=001_2$）

$$ E_{\text{sub}} = 1 - 7 = -6 $$

$$ v = (+1) \cdot 2^{-6} \cdot 2^{-3} = 2^{-9} \approx 0.001953 $$

此为 E4M3FN 可表示的**最小正次正规数**。

#### 6.1.3 最大有限值验证

**编码**：`0 1111 110`（$s=0$，$e_{\text{field}}=15$，$m=110_2$）

$$ E = 15 - 7 = 8, \quad M = 1.75 $$

$$ v = 1.75 \times 256 = 448 \quad \checkmark $$

**编码**：`0 1111 111`（$e_{\text{field}}=15$，$m=111_2$）→ NaN，不表示数值。

### 6.2 E5M2 编码 → 数值解码

#### 6.2.1 正规数示例

**编码**：`0 01111 01`（$s=0$，$e_{\text{field}}=01111_2=15$，$m=01_2$）

$$ E = 15 - 15 = 0 $$

$$ M = 1 + 2^{-2} = 1.25 $$

$$ v = 1.25 \times 2^0 = 1.25 $$

---

**编码**：`0 10001 11`（$s=0$，$e_{\text{field}}=10001_2=17$，$m=11_2$）

$$ E = 17 - 15 = 2 $$

$$ M = 1 + 2^{-1} + 2^{-2} = 1.75 $$

$$ v = 1.75 \times 4 = 7.0 $$

#### 6.2.2 最小正次正规数

**编码**：`0 00000 01`（$e_{\text{field}}=0$，$m=01_2$）

$$ E_{\text{sub}} = 1 - 15 = -14 $$

$$ v = 2^{-14} \cdot 2^{-2} = 2^{-16} \approx 1.526 \times 10^{-5} $$

### 6.3 FP32 → E4M3FN 量化步骤

以 $v_{\text{fp32}} = -6.5$ 为例，演示 Truncate 舍入路径。

**Step 1：FP32 二进制分解**

$$ -6.5 = -1.101_2 \times 2^2 $$

FP32：$s=1$，$e_{\text{FP32}}=129$（即 $E=2$），$m_{\text{FP32}} = 10100\ldots0_2$（23 bits）

**Step 2：计算目标 exponent field**

$$ e_{\text{new}} = E + \text{bias}_{E4M3FN} = 2 + 7 = 9 $$

范围检查：$1 \le 9 \le 14$（未溢出，未 underflow）✓

**Step 3：Truncate 尾数**

FP32 尾数高 3 bits：`101`

$$ m_{\text{E4M3FN}} = 101_2 $$

**Step 4：组装**

$$ \text{编码} = \underbrace{1}_{s} \underbrace{1001}_{e=9} \underbrace{101}_{m} = \texttt{0xCD} $$

**Step 5：验证解码**

$$ E = 9 - 7 = 2, \quad M = 1 + 2^{-1} + 2^{-3} = 1 + 0.5 + 0.125 = 1.625 $$

$$ v_{\text{decoded}} = -1.625 \times 2^2 = -6.5 \quad \checkmark $$

（本例无截断误差，因为 FP32 尾数高 3 bits 恰好精确表示了原值）

### 6.4 FP32 → E4M3FN：含截断误差示例

以 $v_{\text{fp32}} = 3.14159$ 为例。

**Step 1：FP32 二进制**

$$ 3.14159 \approx 1.10010010_2 \times 2^1 $$

$s=0$，$E=1$，FP32 尾数高 3 bits = `100`（即 $2^{-1}$）

**Step 2：目标 exponent field**

$$ e_{\text{new}} = 1 + 7 = 8 $$

**Step 3：Truncate 尾数**

FP32 尾数 23 bits = `10010010000111111011011...`，取高 3 bits：`100`

**Step 4：组装**

$$ \text{编码} = 0\underbrace{1000}_{e=8}\underbrace{100}_{m} $$

**Step 5：验证解码**

$$ M = 1 + 2^{-1} = 1.5, \quad v_{\text{decoded}} = 1.5 \times 2^1 = 3.0 $$

**截断误差**：

$$ \varepsilon = |3.14159 - 3.0| \approx 0.14159 $$

相对误差：

$$ \frac{\varepsilon}{|v|} \approx 4.5\% $$

> 此误差在推理量化中属于正常范围。使用 RNE（Round-to-Nearest-Even）可将该示例舍入至 `100` 对应的 3.0 或检查 `101` 对应的 3.5，取最近者：$|3.14159-3.0|=0.14159 < |3.14159-3.5|=0.35841$，故 RNE 亦选 3.0，结果相同。

---

## 7. 推理引擎数值工程策略

FP8 推理路径的策略选择是**吞吐 / 延迟 / 数值稳定性 / 硬件代价**的折中。

### 7.1 Underflow 策略

|策略|行为|
|---|---|
|Subnormal 保留|精度连续|
|**FTZ**|Flush-to-Zero|

**推理偏向 FTZ 的原因：**

|原因|工程解释|
|---|---|
|硬件代价|Subnormal 解码路径复杂|
|Pipeline 效率|次正规数触发慢路径|
|Tensor Core 吞吐|Denorm 降速明显|
|DNN 容错性|极小值贡献有限|
|时延确定性|避免 denorm penalty|

结论：FTZ 提升吞吐与延迟稳定性。

### 7.2 Overflow 策略

|策略|行为|
|---|---|
|Inf 保留|IEEE 语义|
|**Saturation**|Clamp → Max Normal|

**推理偏向 Saturation 的原因：**

|原因|工程解释|
|---|---|
|Inf 传播风险|Inf × 0 → NaN 污染|
|网络稳定性|Clamp 行为更可控|
|Clip 对齐|与量化阈值一致|
|异常隔离|防止 NaN 扩散|
|推理场景|无反向传播需求|

结论：Saturation 更稳定。

### 7.3 舍入模式

|模式|特征|
|---|---|
|RNE|统计误差最优|
|**Truncate**|实现简单|

Fallback / Reference kernel 常用 Truncate：逻辑简单，延迟低，易做 bit-level 验证。

---

## 8. Microscaling (MX) 与 Block Scaling

### 8.1 MX（Microscaling）定义

**MX = Microscaling Floating Point**，OCP 定义的**块级共享指数缩放机制**：

$$ X_i \approx \hat{X}_{\text{lowp},i} \cdot 2^{e_{\text{shared}}} $$

核心思想：张量按 Block 切分，Block 内共享指数 / scale，数据保持 FP8 / FP4。

### 8.2 MX 解决的问题

DNN 激活分布特征：Heavy-tailed，Outliers 主导 scale。

传统 Per-Tensor scaling 导致大量 underflow、精度塌缩。MX 改善：

|改善项|效果|
|---|---|
|动态范围利用率|↑|
|Underflow 概率|↓|
|Quantization 分辨率|↑|
|Clipping 压力|↓|

### 8.3 与传统 Scaling 对比

|方法|特征|
|---|---|
|Per-Tensor|全局 scale|
|Per-Channel|通道级|
|**MX / Block**|Block 共享指数|

优势：Outlier 隔离，更均匀精度分布，无额外 FP32 scale tensor。

### 8.4 硬件收益

|维度|改善|
|---|---|
|Tensor Core 解码|✔|
|带宽效率|✔|
|Cache locality|✔|
|Quantization stability|✔|

相关指令：`mma.sync` (MX variants)

---

## 9. Kernel 实现：FP32 → E4M3FN

```cpp
#include <bit>
#include <cstdint>
#include <cmath>

[[nodiscard]] constexpr uint8_t fp32_to_e4m3fn_trunc(float v) noexcept {
    const uint32_t bits = std::bit_cast<uint32_t>(v);
    const uint32_t sign = (bits >> 31) & 0x1;
    const int32_t  exp  = ((bits >> 23) & 0xFF) - 127;
    const uint32_t mant = bits & 0x7FFFFF;

    if (v == 0.0f) return sign << 7;
    if (std::isnan(v)) return (sign << 7) | 0x7F;  // Canonical NaN

    int32_t new_exp = exp + 7;  // rebias: FP32 bias 127 → E4M3FN bias 7

    // Overflow → Saturation (clamp to Max Normal = 448)
    // e_field=1111 且 m 高3bits >= 111 时为 NaN，最大有限值为 e=1111, m=110
    // FP32 mant 高3bits=111 对应 mant >= 0x700000
    if (new_exp > 15 || (new_exp == 15 && mant >= 0x700000))
        return (sign << 7) | 0x7E;  // 0 1111 110 = 448

    // Underflow → FTZ
    if (new_exp <= 0)
        return sign << 7;

    const auto res_exp  = static_cast<uint8_t>(new_exp);
    const auto res_mant = static_cast<uint8_t>(mant >> 20);  // 取 FP32 尾数高3bits
    return (sign << 7) | (res_exp << 3) | res_mant;
}
```

### 9.1 原始版本勘误

**原文 overflow 判断条件存在错误：**

```cpp
// ❌ 原文（错误）
if (new_exp > 15 || (new_exp == 15 && mant >= 0x600000))

// ✅ 修正后
if (new_exp > 15 || (new_exp == 15 && mant >= 0x700000))
```

**错误分析：**

|条件|`mant >> 20` 高3bits|E4M3FN 语义|应有行为|
|---|---|---|---|
|`mant >= 0x600000`|$\ge 110_2$|$m=110$ 对应有限值 448|❌ 错误 clamp|
|`mant >= 0x700000`|$= 111_2$|$m=111$ 对应 NaN|✅ 正确 clamp|

具体验证：FP32 值 $v = 1.101_2 \times 2^8 = 448.0$（即 E4M3FN 最大有限值）

- $e_{\text{new}} = 8 + 7 = 15$，$\text{mant}_{\text{FP32}} = 0x600000$（高3bits = $110_2$）
- 原文条件：$0x600000 \ge 0x600000$ → 触发 Saturation → 输出 `0x7E`，**解码仍为 448**，结果碰巧正确
- 但 $v = 1.1011_2 \times 2^8 = 460.0$ 时：$e_{\text{new}} = 15$，$\text{mant} = 0x680000$，原文条件触发 Saturation；修正版不触发，Truncate 后 $m = 110_2$，**正确输出 448**（而非 Saturate 本来也是 448，这里结果相同但语义不同）
- 关键区别在 $v = 1.1101_2 \times 2^8 = 472.0$：$\text{mant 高3bits} = 110_2$（因为 $1.110\ldots \times 2^8$，mant高3bits=110），原文条件：$0x700000 > 0x600000$ → 触发；修正版 `mant = 0x700000` → 也触发。此处实际无差异。

> 更精确的反例：令 FP32 mant 高3bits = $110_2$ 且低位非全零（如 $v \approx 448.1$）：原文在 $e_{\text{new}}=15$ 时因 $0x60xxxx \ge 0x600000$ 触发 Saturation 输出 448；修正版因 $0x60xxxx < 0x700000$ 不触发，Truncate 输出 $m=110$，解码同为 448。**两版本实际数值结果相同**，但原文边界条件语义错误（其本意是"尾数会导致 NaN"，而 $m=110$ 不是 NaN）。

**结论：原文条件虽在绝大多数情况下结果等价，但边界语义与注释不符，应修正为 `0x700000`。**

### 9.2 代码与策略映射

|代码逻辑|数值策略|
|---|---|
|`new_exp <= 0`|FTZ|
|overflow branch|Saturation|
|`mant >> 20`|Truncate（取高3bits）|
|`isnan(v)`|Canonical NaN（`0x7F` / `0xFF`）|

### 9.3 对应硬件指令

CUDA PTX：

```
cvt.rn.e4m3fn.f32    // RNE 舍入（硬件路径）
```

用途：Reference path，Fallback kernel，Bit-exact 验证。

---

## 10. 关键工程结论

|主题|结论|
|---|---|
|FP8 位宽|8-bit|
|Bias 公式|$2^{x-1}-1$|
|次正规指数|$1-\text{bias}$（保证数轴连续）|
|E4M3FN 最大有限值|448（$e=1111, m=110$）|
|E5M2 最大有限值|57344（$e=11110, m=11$）|
|E4M3FN NaN 编码|$e=1111, m=111$（唯一）|
|推理主格式|E4M3FN|
|梯度主格式|E5M2|
|Underflow 策略|FTZ 常见|
|Overflow 策略|Saturation 常见|
|Outlier 抑制|MX / Block Scaling|
|Overflow clamp 阈值（代码）|`mant >= 0x700000`（高3bits = $111_2$）|
