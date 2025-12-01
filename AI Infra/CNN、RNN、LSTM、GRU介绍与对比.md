
## **1️⃣** CNN（**Convolutional Neural Network**，卷积神经网络）

### **基本概念**
CNN解决了图像处理中存在的**参数爆炸**和**无空间结构**的问题，特点是：
- 利用“局部相关性” → 卷积核只看小区域
- 参数共享 → 同一个卷积核对整幅图扫一遍
- 层级表示 → 低层纹理，高层语义

### **卷积（Convolutional）**
#### 卷积层/卷积核
卷积核（filter）是一个 **K×K 的小窗口**，在图像上滑动，每个位置做内积。
例如 同一个 3×3 卷积核，在每个区域重复计算：
$$y = \sum W_{ij} \cdot X_{ij}$$
🔥 卷积层特点：
👉 **权重共享**（整个图像只用一个核）  
👉 **局部连接**（只连接 K×K 范围）  
👉 输出是 Feature Map（特征图）
#### 参数量
假设一个卷积核：  
大小 K×K，输入通道 Cin，输出通道 Cout
参数量：
$$(K \times K \times C_{in}) \times C_{out} + C_{out}​$$
例如 ResNet 常用：
3×3 卷积，输入 128 通道，输出 256：
(3×3×128)×256 = 294,912
远远小于 MLP 的数百万甚至上亿参数。
#### 超参数
1. Kernel size（卷积核大小）
   常用：
	- 3×3（最经典）
	- 5×5、7×7（感受野大一点）
	- 1×1（降维/升维，也是网络的非线性组合）
2. Stride（步幅）
		步幅是卷积核滑动的步长。
		stride=2 → 图像尺寸减半  
		stride=1 → 保持相近
3. Padding（填充）
   让卷积核不用越界，也能控制输出大小。
		valid padding（不填充）→ 尺寸缩小
		same padding（左右补 0）→ 尺寸保持不变（TensorFlow 风格）

### **池化层（Pooling）**
目的：  
✔ 降低 feature map 的尺寸  
✔ 提供一定平移不变性  
✔ 减少计算量
### 常用：
- Max Pooling（最大池化）
- Average Pooling（均值池化）
通常 kernel=2, stride=2（尺寸减半）

### **CNN 的三大核心思想**
#### ✔ 1. 局部感受野（Local Receptive Field）
每一层的小卷积核看图像的一小块  
→ 捕获局部模式（边缘、角点、纹理）
#### ✔ 2. 权重共享（Shared Weights）
同一个卷积核在整个图像扫  
→ 参数量大幅减少  
→ 对平移不变性更友好（物体在图像不同位置仍能被识别）
#### ✔ 3. 多通道（Channel）
输入 RGB 三个通道 feature map  
卷积核大小为 K×K×C（例如 C=3）
每个 filter 会跨通道求和，捕获更复杂模式。

### **CNN 的层级表示（Hierarchical Features）**
CNN 学到的特征是逐层抽象的：
- 第一层：边缘（Gabor-like）
- 第二层：纹理
- 中层：局部结构
- 高层：物体的一部分
- 最后层：完整语义，如“狗脸”、“车轮”

这是 CNN 为什么强大的关键。



## 2️⃣ RNN（Recurrent Neural Network，循环神经网络）
### **基本概念**
RNN 是处理序列数据（如文本、时间序列、语音）的基础网络。它的特点是：
- 有 **隐藏状态 hth_tht​**，能够记录序列前面的信息。
- 每个时间步的输出依赖于 **当前输入 xtx_txt​** 和 **前一隐藏状态 ht−1h_{t-1}ht−1​**。

公式：
$ht=tanh⁡(Whht−1+Wxxt+b)h_t = \tanh(W_h h_{t-1} + W_x x_t + b)ht​=tanh(Wh​ht−1​+Wx​xt​+b)$
### **优点**
- 能处理任意长度的序列。
- 模型结构简单。

### **缺点**
- **梯度消失/爆炸**：当序列很长时，早期信息难以传递到后面。
- **记忆能力弱**：难以捕捉长距离依赖。

---

## 3️⃣ LSTM（Long Short-Term Memory，长短期记忆网络）
### **基本概念**
LSTM 是对 RNN 的改进，专门解决长距离依赖问题。  
核心思想：引入 **门控机制（Gate）** 来控制信息流。
#### **LSTM结构**
1. **遗忘门 ftf_tft​**：决定丢弃多少过去的记忆
    $ft=σ(Wf[ht−1,xt]+bf)f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)ft​=σ(Wf​[ht−1​,xt​]+bf​)$
2. **输入门 iti_tit​**：决定当前信息写入多少到记忆
    $it=σ(Wi[ht−1,xt]+bi)i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)it​=σ(Wi​[ht−1​,xt​]+bi​)$

3. **候选记忆 C~t\tilde{C}_tC~t​**：
    $C~t=tanh⁡(Wc[ht−1,xt]+bc)\tilde{C}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)C~t​=tanh(Wc​[ht−1​,xt​]+bc​)$
4. **记忆更新 CtC_tCt​**：
    $Ct=ft∗Ct−1+it∗C~tC_t = f_t * C_{t-1} + i_t * \tilde{C}_tCt​=ft​∗Ct−1​+it​∗C~t​$

5. **输出门 oto_tot​**：控制最终输出
    $ht=ot∗tanh⁡(Ct)h_t = o_t * \tanh(C_t)ht​=ot​∗tanh(Ct​)$
### **优点**
- 可以捕捉 **长距离依赖**。
- 门控机制缓解梯度消失问题。
### **缺点**
- 参数多，计算量大。
- 训练比普通 RNN 慢。
    

---

## 4️⃣ GRU（Gated Recurrent Unit，门控循环单元）
### **基本概念**
GRU 是 LSTM 的简化版本，把 LSTM 的三个门合并成 **两个门**（更新门 + 重置门），结构更简单。
#### **GRU结构**
1. **更新门 ztz_tzt​**：控制记忆更新
    $zt=σ(Wzxt+Uzht−1)z_t = \sigma(W_z x_t + U_z h_{t-1})zt​=σ(Wz​xt​+Uz​ht−1​)$
2. **重置门 rtr_trt​**：控制新信息结合旧记忆的程度
    $rt=σ(Wrxt+Urht−1)r_t = \sigma(W_r x_t + U_r h_{t-1})rt​=σ(Wr​xt​+Ur​ht−1​)$
3. **候选隐藏状态 h~t\tilde{h}_th~t​**：
    $h~t=tanh⁡(Whxt+Uh(rt∗ht−1))\tilde{h}_t = \tanh(W_h x_t + U_h (r_t * h_{t-1}))h~t​=tanh(Wh​xt​+Uh​(rt​∗ht−1​))$
4. **隐藏状态更新**：
    $ht=(1−zt)∗ht−1+zt∗h~th_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_tht​=(1−zt​)∗ht−1​+zt​∗h~t​$

### **优点**
- 参数比 LSTM 少，计算更快。
- 能捕捉长距离依赖。
- 结构更简单，训练更容易收敛。
### **缺点**
- 灵活性略低于 LSTM（少了一个门控制输出）。
- 在一些任务上 LSTM 表现可能更好。

---

## 5️⃣ 对比总结

|特性|RNN|LSTM|GRU|
|---|---|---|---|
|门控机制|❌|✅ 遗忘门、输入门、输出门|✅ 更新门、重置门|
|长期依赖能力|弱|强|较强|
|参数量|少|多|中|
|训练速度|快|慢|较快|
|适用场景|短序列或简单任务|长序列、复杂依赖|中长序列、速度要求高|

---
✅ **总结理解**
- **RNN**：基础，但难以记住长序列信息。
- **LSTM**：通过三个门，能灵活控制信息流，适合长序列。
- **GRU**：LSTM简化版，保留主要门控功能，速度快，效果接近LSTM。