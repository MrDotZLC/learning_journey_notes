## 一、FP8 格式设计背景
FP8（8-bit Floating Point）用于缓解大模型中的 **带宽压力、缓存占用、Tensor Core 吞吐瓶颈**。
FP8 位宽拓扑约束：
$$  
1(\text{Sign}) + x(\text{Exponent}) + y(\text{Mantissa}) = 8  
$$
设计折中：

|调整方向|影响|
|---|---|
|增大 $x$|扩展动态范围|
|增大 $y$|提升数值精度|

IEEE 754 在 8-bit 下表示效率有限 → 工业界采用 OFP8。

---

## 二、OCP OFP8 标准
现代 AI 加速器遵循：
**OCP (Open Compute Project) – 8-bit Floating Point Specification (OFP8)**
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

## 三、OFP8 格式族
### 3.1 E4M3FN（Finite Only）
结构：
- Exponent = 4 bits
- Mantissa = 3 bits
- Bias = 7
- 无 Infinity
设计哲学：
- 舍弃 Inf
- 压缩 NaN
- 扩展有限数覆盖密度
典型用途：
- 推理权重（Weights）
- 激活值（Activations）
优势：
- 更高精度密度
- 推理数值分布适配性强
### 3.2 E5M2
结构：
- Exponent = 5 bits
- Mantissa = 2 bits
- Bias = 15
- 保留 Infinity
典型用途：
- 训练梯度（Gradients）
- 宽动态范围张量
优势：
- 抗溢出能力强
代价：
- 精度降低
### 3.3 格式对比

|格式|x|y|Bias|最大有限值|机器精度 $\epsilon$|Inf|
|---|---|---|---|---|---|---|
|**E4M3FN**|4|3|7|±448|$2^{-3}=0.125$|❌|
|**E5M2**|5|2|15|±57344|$2^{-2}=0.25$|✅|

---

## 四、数学基础：Bias 与解码模型
### 4.1 Bias 推导
$$  
bias = 2^{x-1} - 1  
$$
### 4.2 正规数（Normal）
$$  
v = (-1)^s \cdot 2^{e-bias} \cdot  
\left(1 + \sum_{i=1}^{y} b_i 2^{-i}\right)  
$$
隐含位：`1.`
### 4.3 次正规数（Subnormal）
触发条件：
$$  
e = 0,\quad m \neq 0  
$$
指数规则：
$$  
E_{sub} = 1 - bias  
$$
解码：
$$  
v = (-1)^s \cdot 2^{1-bias} \cdot  
\left(\sum_{i=1}^{y} b_i 2^{-i}\right)  
$$
隐含位：`0.`
### 4.4 连续性（Gradual Underflow）
若指数取 $0-bias$：
- 最大 Subnormal 与最小 Normal 存在断层
采用：
$$  
E_{sub} = 1-bias  
$$
→ 数轴连续

---

## 五、数值边界分析
### 5.1 特殊编码
#### E4M3FN
$$  
e = 1111_2,\quad m = 111_2 \rightarrow NaN  
$$
- 无 Infinity
### 5.2 最大有限值（E4M3FN）
边界构造：
$$  
e_{max}=15,\quad E=15-7=8  
$$
$$  
m_{max}=110_2  
$$
$$  
M = 1 + (2^{-1}+2^{-2}) = 1.75  
$$
$$  
Max = 1.75 \cdot 2^8 = 448  
$$
工程阈值：
```
clip_threshold = 448.0f
```

---

## 六、推理引擎数值工程策略
FP8 推理路径的策略选择是：
> **吞吐 / 延迟 / 数值稳定性 / 硬件代价** 的折中
### 6.1 Underflow 策略

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
### 6.2 Overflow 策略

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
### 6.3 舍入模式

|模式|特征|
|---|---|
|RNE|统计误差最优|
|**Truncate**|实现简单|

Fallback / Reference kernel 常用 Truncate：
- 逻辑简单
- 延迟低
- 易做 bit-level 验证

---

## 七、Microscaling (MX) 与 Block Scaling
### 7.1 MX（Microscaling）定义
**MX = Microscaling Floating Point**
OCP 定义的 **块级共享指数缩放机制**：
$$  
X_i \approx \hat{X}_{lowp,i} \cdot 2^{e_{shared}}  
$$
核心思想：
- 张量按 Block 切分
- Block 内共享指数 / scale
- 数据保持 FP8 / FP4
### 7.2 MX 解决的问题
DNN 激活分布特征：
- Heavy-tailed
- Outliers 主导 scale
传统 scaling 导致：
- 大量 underflow
- 精度塌缩
MX 改善：

|改善项|效果|
|---|---|
|动态范围利用率|↑|
|Underflow 概率|↓|
|Quantization 分辨率|↑|
|Clipping 压力|↓|

### 7.3 与传统 Scaling 对比

|方法|特征|
|---|---|
|Per-Tensor|全局 scale|
|Per-Channel|通道级|
|**MX / Block**|Block 共享指数|

优势：
- Outlier 隔离
- 更均匀精度分布
- 无额外 FP32 scale tensor
### 7.4 硬件收益

|维度|改善|
|---|---|
|Tensor Core 解码|✔|
|带宽效率|✔|
|Cache locality|✔|
|Quantization stability|✔|

相关指令：
```
mma.sync (MX variants)
```

---

## 八、Kernel 实现：FP32 → E4M3FN
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
    if (std::isnan(v)) return (sign << 7) | 0x7F;
    int32_t new_exp = exp + 7;
    if (new_exp > 15 || (new_exp == 15 && mant >= 0x600000))
        return (sign << 7) | 0x7E;
    if (new_exp <= 0)
        return sign << 7;   // FTZ
    const auto res_exp  = static_cast<uint8_t>(new_exp);
    const auto res_mant = static_cast<uint8_t>(mant >> 20);
    return (sign << 7) | (res_exp << 3) | res_mant;
}
```

---

### 8.1 代码与策略映射

|代码逻辑|数值策略|
|---|---|
|new_exp <= 0|FTZ|
|overflow branch|Saturation|
|mant >> 20|Truncate|
|isnan(v)|Canonical NaN|

### 8.2 对应硬件指令
CUDA PTX：
```
cvt.rn.e4m3fn.f32
```
用途：
- Reference path
- Fallback kernel
- Bit-exact 验证

---

## 九、关键工程结论

| 主题           | 结论                 |
| ------------ | ------------------ |
| FP8 位宽       | 8-bit              |
| Bias 公式      | $2^{x-1}-1$        |
| 次正规指数        | $1-bias$           |
| E4M3FN 最大值   | 448                |
| 推理主格式        | E4M3FN             |
| 梯度主格式        | E5M2               |
| Underflow 策略 | FTZ 常见             |
| Overflow 策略  | Saturation 常见      |
| Outlier 抑制   | MX / Block Scaling |
