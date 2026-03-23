TensorRT 的 **INT8 Calibration（8-bit 量化校准）** 本质是**在离线阶段估计各 Tensor 的动态区间（dynamic range）**，从而确定 scale，**在截断误差（clipping error）与量化误差（rounding error）之间取得最优平衡**，使 FP32/FP16 网络安全映射到 INT8 推理。
## 一、为什么需要 Calibration？
 INT8 量化形式：
$$x_{int8} = \text{round}\left(\frac{x}{s}\right), \quad x_{int8} \in [-127,127]$$
 scale 来自 Tensor 的**动态区间**：
$$x \in [-T, T], \quad s = \frac{T}{127}$$  
 关键问题：
 👉 **动态区间 (T) 如何确定？**
 - 权重（weights）：静态 → 可直接统计
 - 激活（activations）：输入相关 → 必须通过 Calibration 估计

 ---

## 二、动态区间与误差的关系
 动态区间选择本质是误差权衡：
### ① 动态区间过大（如真实最大值）
$$T = \max(|x|)  $$
 优点：
	✅ 无 clipping（截断误差≈0）
 代价：
	❌ scale 变大  
	❌ 量化步长变粗  
	❌ **量化误差（rounding error）增大**
 量化误差上界：
$$|e_{quant}| \le s/2  $$
### ② 动态区间较小（主动 clipping）
$$T < \max(|x|)$$  
 超出范围的值：
$$x_{clip} = \text{clip}(x, -T, T)$$  
 产生：
$$e_{clip} = x - x_{clip}$$  
 特性：
 - 仅影响 (|x| > T)
 - 信息丢失（不可逆）
 但带来：
✅ scale 变小  
✅ rounding error 显著降低
### 🎯 Calibration 的数学目标
	$$T^* = \arg\min_T \left(E[e_{clip}^2] + E[e_{quant}^2]\right) $$ 👉 最小化总量化误差

---

## 三、TensorRT Calibration 在做什么？
 TensorRT 在校准阶段：
 1️⃣ 用校准数据运行 FP32 推理  
 2️⃣ 统计 activation 分布（histogram）  
 3️⃣ 估计最优动态区间 (T)  
 4️⃣ 计算 dynamic range  
 5️⃣ 推导 scale：
$$s = T / 127  $$
 ---
## 四、TensorRT 支持的 Calibration 算法
### 1️⃣ MinMax Calibrator（Legacy）
 **方法：**
$$T = \max(|x|)$$  
 特点：
	✅ 简单直接  
	✅ 无 clipping
 缺点：
	❌ 对 outlier 极度敏感  
	❌ scale 往往偏大 → rounding error 增加
 适用：
 - 激活分布平稳
 - 长尾弱
### 2️⃣ Entropy Calibrator（KL Divergence）
 TensorRT 默认推荐方法。
 **核心思想：**
👉 选择 clipping threshold (T)，使量化后分布与原始分布差异最小。
 步骤：
	1️⃣ 构建 activation histogram  
	2️⃣ 尝试不同 threshold (T)  
	3️⃣ 截断得到 clipped 分布  
	4️⃣ 映射为 INT8 分布  
	5️⃣ 计算：
	$$D_{KL} ​(P∣∣Q) = ∑ P(i) log \frac {Q(i)} {P(i)}$$
		P(i)：原始 FP32 histogram
		Q(i)：量化后 histogram
	6️⃣ 选取最小 KL 的 (T)
	 特点：
		偏向保留高概率区域
		容忍低概率大幅值 outlier 被截断
		更稳定
	 优点：
		✅ 抗 outlier  
		✅ 自动平衡 clipping vs rounding  
		✅ 通常精度最佳
	 代价：
		❌ 计算复杂度较高
	 适合：
		分类 / softmax 前层
		概率结构重要的网络
### 3️⃣ Percentile Calibrator
 **方法：**
$$T = P_{p\%}(|x|)$$
 例如 99.9%。
 优点：
	 ✅ 抑制异常值影响  
	 ✅ 简单稳定
 缺点：
	 ❌ p 需经验设定
### 4️⃣ MSE Threshold Calibrator（自定义 / 非官方）
 **方法：**
$$T^* = \arg\min_T E[(x - \hat{x})^2]$$
- $\hat{x} = \text{round}(\text{clip}(x,-T,T)/s) \cdot s$
- 本质上通过 **最小化总平方误差** 选择最优动态区间
 **特点：**
- 偏向保留幅值较大的区域
- 对概率分布尾部敏感
- 不直接优化分布信息

**优点：**
- ✅ 对数值误差最小
- ✅ 对某些 FC / value 层效果好
**缺点：**
- ❌ 对 outlier 容忍度低
- ❌ 不考虑概率结构信息
- ❌ TensorRT 默认不提供，需要自定义实现
**适合：**
- 回归或数值敏感层
- 自定义 PTQ / Transformer 校准实验

---

### 📌 本质差异总结

| 算法            | 动态区间策略 / 特性                    |
| ------------- | ------------------------------ |
| MinMax        | 零 clipping，rounding error 可能大  |
| Percentile    | 显式忽略 outlier                   |
| Entropy (KL)  | 优化信息损失，保留高概率区域，低概率 outlier 可截断 |
| MSE Threshold | 最小化平方误差，保留大值幅度，outlier 敏感      |

---

## 五、Calibration Workflow（执行流程）
### Step 1️⃣：实现 IInt8Calibrator
 用户提供：
 - `getBatchSize()`
 - `getBatch()`
 - cache 读写接口
 功能：
 👉 向 TensorRT 提供**代表性输入数据**
### Step 2️⃣：运行校准推理
 TensorRT：
 - 执行 FP32 inference
 - 收集 activation histogram（bin 数量经验值为2048）
### Step 3️⃣：估计动态区间
 对每个 Tensor：
$$[-T, T]$$  
 依据所选算法：
 - MinMax
 - KL Divergence
 - Percentile
### Step 4️⃣：计算 scale
$$s = T / 127$$  
 写入 Tensor dynamic range。

 ---

### Step 5️⃣：生成 Calibration Cache
 缓存：
```
tensor_name → dynamic_range
```
 优点：
 ✅ engine 重建无需重复校准  
 ✅ 加速部署流程

 ---

## 六、为什么 Calibration 能提升精度？
 若直接用真实最大值：
 👉 rounding error 往往主导
 Calibration 通过：
 ✅ 适度 clipping  
 ✅ 减小 scale  
 ✅ 提升有效分辨率
 通常降低总体误差：
$$E[e_{clip}^2] + E[e_{quant}^2]$$  
 ---
## 七、INT8 精度下降常见根因
 与动态区间/误差直接相关：
 - 激活长尾分布
 - 极端 outlier
 - 小数值密集（分辨率不足）
 - Softmax / LayerNorm / GELU 敏感
 TensorRT 通常策略：
 👉 对敏感层保留 FP16 / FP32（混合精度）

---

## 八、一句话精确总结
 👉 **TensorRT INT8 Calibration = 基于代表性数据估计各 Tensor 的最优动态区间，通过平衡截断误差与量化误差确定 scale。**
 
