# 0. 背景与系统瓶颈 (Background & System Bottlenecks)
## **0.1 内在维度假说 (Intrinsic Dimension)**
Aghajanyan 等人指出，预训练大语言模型（LLM）处于严重的过参数化状态。在特定下游任务微调时，参数更新的“内在维度”极低，即梯度主要分布在低秩子空间中。
核心推论：
$$W \in \mathbb{R}^{d \times k}  
\quad \Rightarrow \quad  
\Delta W \approx AB,; r \ll \min(d,k) $$ 
无需全量更新权重矩阵，仅需学习低秩增量 (\Delta W) 即可逼近全参数微调（Full Finetuning, FFT）效果。
## **0.2 显存墙 (Memory Wall) 瓶颈**
FFT 显存占用来源：
- 模型权重 (Weights)
- 梯度 (Gradients)
- 优化器状态 (Optimizer States)
- 前向激活值 (Activations)
FP16 + AdamW：

|组件|显存|
|---|---|
|权重|(P \times 2) bytes|
|梯度|(P \times 2)|
|Adam m|(P \times 4)|
|Adam v|(P \times 4)|
$$\mathcal{M}_{total} \approx P \times 12 \text{ bytes}$$
7B 示例：
$$7 \times 10^9 \times 12 \approx 84\text{ GB (不含 Activations)}$$
核心矛盾：显存带宽与容量增长速度远滞后于模型规模扩张。

---
# 1. LoRA (Low-Rank Adaptation) 系统架构
## **1.1 核心定义与前向计算**
冻结预训练权重：
$$W_0 \in \mathbb{R}^{d \times k}$$
低秩增量参数化：
$$\Delta W = sAB$$
$$A \in \mathbb{R}^{d \times r},;  
B \in \mathbb{R}^{r \times k},;  
r \ll \min(d,k)  $$
缩放因子：
$$s = \frac{\alpha}{r}$$
前向传播：
$$Y = X(W_0 + sAB)$$
展开：
$$Y = XW_0 + sXAB$$
初始化策略：
$$A \sim \mathcal{N}(0, \sigma^2),\quad B = 0$$
确保训练起始：
$$\Delta W_{t=0} = 0$$

---
## **1.2 严密梯度链式推导 (Rigorous Gradient Derivation)**
### **1.2.0 问题设定**
输入：
$$X \in \mathbb{R}^{n \times d}$$
输出：
$$Y \in \mathbb{R}^{n \times k}$$
损失函数：
$$\mathcal{L}(Y)$$
定义损失对输出梯度：
$$G = \nabla_Y \mathcal{L}  
\in \mathbb{R}^{n \times k}  $$
LoRA 路径：
$$Y_{lora} = sXAB$$
### **1.2.1 显式微分展开**
对输出矩阵微分：
$$dY = s,d(XAB)$$
由于 (X) 为常量：
$$dY = sX,d(AB)$$
乘积微分法则：
$$d(AB) = (dA)B + A(dB)$$
代入：
$$dY = sX[(dA)B + A(dB)]$$
拆分：
$$dY = sX(dA)B + sXA(dB)$$
### **1.2.2 损失函数微分**
标量损失微分定义：
$$d\mathcal{L}  
= \langle G, dY \rangle  
= \text{Tr}(G^T dY)$$  
代入 (dY)：
$$d\mathcal{L}  
= s\text{Tr}(G^T X(dA)B) 
- s\text{Tr}(G^T XA(dB))  $$
### **1.2.3 对 A 的梯度推导**
考虑第一项：
$$\text{Tr}(G^T X(dA)B)$$
利用迹循环不变性：
$$= \text{Tr}(B G^T X(dA))$$
重排：
$$= \text{Tr}((X^T G B^T)^T dA)$$
根据矩阵微分定义：
$$d\mathcal{L}  
= \text{Tr}\left(  
\left(\frac{\partial \mathcal{L}}{\partial A}\right)^T dA  
\right)  $$
得到：
$$\boxed{  
\frac{\partial \mathcal{L}}{\partial A}  
= s X^T G B^T  
}  $$
维度校验：
$$(d \times n)(n \times k)(k \times r)  
= d \times r  $$
### **1.2.4 对 B 的梯度推导**
考虑第二项：
$$\text{Tr}(G^T XA(dB))$$
迹循环：
$$= \text{Tr}((XA)^T G dB)$$
$$= \text{Tr}((A^T X^T G)^T dB)$$
得到：
$$\boxed{  
\frac{\partial \mathcal{L}}{\partial B}  
= s A^T X^T G  
} $$ 
维度校验：
$$(r \times d)(d \times n)(n \times k)  
= r \times k $$ 
### **1.2.5 梯度路径解释**
梯度传播链：
[  
\mathcal{L}  
\rightarrow Y  
\rightarrow XAB  
\rightarrow A,B  
]
关键观察：
- 梯度不流向 (W_0)（冻结）
- 梯度被限制于低秩结构：
[  
\nabla_W \mathcal{L}  
\approx s X^T G  
\Rightarrow \text{rank constrained}  
]
---
## **1.3 LoRA 低秩有效性解释**
---
### **1.3.1 一阶泰勒近似**
[  
f(W_0 + \Delta W)  
\approx  
f(W_0) + J_{W_0} \Delta W  
]
若梯度主导方向低维：
$$\Delta W \approx AB$$
---
### **1.3.2 谱角度解释**
经验现象：
$$\sigma_1 \gg \sigma_2 \gg \dots$$
更新矩阵奇异值快速衰减 ⇒ 低秩近似损失小。
---
# 2. QLoRA (Quantized LoRA) 与存储压缩
---
## **2.1 机制定义**
QLoRA 将底座模型 (W_0) 压缩至 4-bit（NF4），显存需求降低约 75%，LoRA 路径用于补偿任务适配与量化误差。
---
## **2.2 量化路径数学展开**
底座权重：
$$W_0 \xrightarrow{\text{Quant}} W_q$$
计算时：
$$\hat{W}_0 = \text{Dequant}(W_q)$$
最终权重：
$$W = \hat{W}_0 + sAB$$
---
## **2.3 量化误差分解**
定义误差：
$$E = W_0 - \hat{W}_0$$
模型实际权重：
$$W = W_0 - E + sAB$$
若：
$$sAB \approx E + \Delta W^*$$
⇒ LoRA 同时补偿：
- 任务适配
- 量化误差
---
## **2.4 双重量化 (Double Quantization, DQ)**
scale 数量：
$$\frac{P}{B_1}$$
scale 再量化：
$$\frac{P}{B_1 B_2}$$
额外 bits：
# [  
\text{Overhead}
\frac{8}{B_1}  
+  
\frac{32}{B_1 B_2}  
]
典型：
[  
B_1=64,; B_2=256  
\Rightarrow 0.127 \text{ bits/param}  
]
⚠️ 表示 **scale 存储额外开销**
---
## **2.5 分页优化器 (Paged Optimizers)**
机制：
- 基于 CUDA Unified Memory
- Adam 状态页级换出 CPU RAM
延迟隐藏：
$$\text{cudaMemPrefetchAsync}$$
---
## **2.6 量化误差补偿**
[  
W_{\text{final}} =  
\text{Dequant}(W_{NF4}) + sAB  
]
LoRA 路径学习量化残差结构。
---
# 3. bitsandbytes (bnb) 底层技术
---
## **3.1 NF4 最优性推导逻辑**
目标：
$$\min_q \mathbb{E}[(w - q(w))^2]$$
若：
$$w \sim \mathcal{N}(0,1)$$
最优标量量化边界：
$$P(w \in \text{bin}_i) = \text{constant}$$
⇒ 使用分位数
---
## **3.2 分位数量化表达**
$$q_i = \Phi^{-1}(p_i)$$
$$p_i = \frac{i + 0.5}{16}$$
（概念表达，实际码本经数值优化）
---
## **3.3 LLM.int8() 与异常值处理**
$$|x| > \alpha,\quad \alpha \approx 6$$
|维度|计算|
|---|---|
|Outlier|FP16|
|Normal|INT8|
---
# 4. CUDA 工程实现 (C++ / Cutlass 视角)
---
## **4.1 反量化数学表达**
$$w_{fp16} = \text{LUT}[code] \times scale$$
Kernel 流程：
[  
\text{Load} \rightarrow  
\text{Unpack} \rightarrow  
\text{LUT} \rightarrow  
\text{Scale} \rightarrow  
\text{MMA}  
]
---
## **4.2 计算-访存重叠目标**
理想：
$$T_{dequant} \subseteq T_{mma}$$
实现手段：
- shared LUT
- vectorized load
- warp pipeline
---
## **4.3 显存足迹计算公式 (VRAM Footprint)**
[  
\mathcal{M}  
\approx  
\underbrace{\Phi \cdot 0.5 (1+\epsilon)}_{\text{NF4+DQ}}  
+  
\underbrace{\sum 2r(d_l+k_l)\cdot 4}_{\text{Adapters}}  
+  
\mathcal{A}(bs, seq)  
]
$$\epsilon \approx 0.03$$
---
# 5. 总结与逻辑闭环
---
### **数学层面**
- 内在维度 → 低秩结构
- LoRA 梯度 → 显式矩阵微分
- NF4 → 高斯分位数量化
---
### **系统层面**
- 4-bit + DQ 降低存储
- Paged Optimizer 降低 VRAM 峰值
---
### **工程层面**
- Fused Dequant + W4A16 GEMM
- Latency hiding 为核心优化目标
---
如需下一步深化，可扩展：
✔ LoRA 与 Hessian 近似  
✔ Rank 与泛化误差  
✔ TensorRT / vLLM kernel 映射