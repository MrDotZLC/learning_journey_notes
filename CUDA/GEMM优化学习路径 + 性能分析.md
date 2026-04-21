## 一、硬件数据
### 1.1 架构：Turing（TU116）
### 1.2 SM 数量：24
### 1.3 CUDA Cores：1536（= 24 × 64）
### 1.4 Boost Clock：约 1770 MHz（不同厂商略有浮动）
### 1.5 显存：GDDR6，192-bit
### 1.6 算力峰值：1536×2×1.77×109=5.44 TFLOPS
### 1.7 理论显存带宽：288 GB/s
### 1.8 Ridge Point（屋脊点）：18.9FLOP/Byte
$$I_{\text{ridge}} = \frac{\text{算力峰值}}{\text{带宽峰值}} = \frac{5.44 \times 10^{12}}{288 \times 10^9} \approx 18.9 \ \text{FLOP/Byte}$$
### 1.9 GEMM 的算术强度：
GEMM $C = A \times B$ ，矩阵尺寸$M \times N \times K$：

$$\text{浮点运算量} = 2MNK \ \text{FLOP}$$
$$\text{数据搬运量（理论最小）} = (MK + KN + MN) \times \text{sizeof(float)} \ \text{Bytes}$$

当 M = N = K时：

$$I = \frac{2M^3}{3M^2 \times 4} = \frac{M}{6}$$

|M=N=K|算术强度|瓶颈|
|---|---|---|
|64|10.7 FLOP/Byte|带宽瓶颈|
|128|21.3 FLOP/Byte|算力瓶颈|
|512|85.3 FLOP/Byte|算力瓶颈|
|1024|170.7 FLOP/Byte|算力瓶颈|
|4096|682.7 FLOP/Byte|算力瓶颈|

## 二、SGEMM
### 2.1 sgemm_v1_naive
Naive kernel 每次从 Global Memory 读数据，无复用。

实际访存量（M=N=K=1024）：

$$ \text{每个线程读 K 次 A、K 次 B} \Rightarrow \text{总读取} = M \times N \times 2K \times 4 \ \text{Bytes} $$

$$ = 1024 \times 1024 \times 2048 \times 4 \approx 8 \ \text{GB} $$

在 288 GB/s 下，**仅访存就需要约 27.8ms**，而算力上限对应的时间是：

$$ t_{\text{compute}} = \frac{2 \times 1024^3}{5.44 \times 10^{12}} \approx 0.39 \ \text{ms} $$

两者相差 70 倍——Naive kernel 的瓶颈完全在 Global Memory 访问，计算单元绝大多数时间在等数据。这是 Shared Memory Tiling（v2）存在的根本原因。

```

```
