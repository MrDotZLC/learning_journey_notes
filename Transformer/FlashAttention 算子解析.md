## FlashAttention 算子解析
本文旨在学习FlashAttention，学习路径来自[比飞鸟贵重的多_HKL](https://www.bilibili.com/video/BV1FM9XYoEQ5?spm_id_from=333.1387.collection.video_card.click)。
手搓 flash attention 算子的代码库[flash-attention-minimal](https://github.com/MrDotZLC/flash-attention-minimal)。
研究LLaMA.cpp中的flash attn 需要打开对应的开关。

### 一、手搓 flash attention 算子
#### 1.1 寻找切入点
思考：如何快速debug一个算子？调试完整的flash-attention太笨重。

github找一个[flash-attention demo](https://github.com/tspeterkim/flash-attention-minimal)，发现其通过pytorch运行单个forward前向运算，本身是通过python调用C++代码访问cuda核函数。
考虑有CUDA C++的debug经验，用libtorch代替pytorch，直接用C++调用核函数，用CMake进行debug。

同理，不仅可以快速调试flash-attention的算子，也能够把LLaMA.cpp等框架的算子拿来debug和学习。

通过python调用CUDA C++的方式，也能够用cuda-gdb进行debug，在此不做深入探讨。

#### 1.2 配置debug环境
下载解压 libtorch ，并在 CMakeLists.txt 中设置 CMAKE_PREFIX_PATH ，不要忘记打开-G。
![](assets/Pasted%20image%2020260127011246.png)
具体代码参考[CMakeLists.txt](https://github.com/MrDotZLC/flash-attention-minimal/blob/main/CMakeLists.txt)。
#### 1.3 运行一个demo
在 CMakeLists.txt 的 add_executable 中，把cu文件编译并链接到入口函数。
![](assets/Pasted%20image%2020260127011355.png)
注意：在调用核函数后，最好执行一次`cudaDeviceSynchronize();`进行设备同步，原因在于Kernel异步执行，会导致cuda-gdb attach 不到Kernel内的断点。
![](assets/Pasted%20image%2020260127020055.png)
#### 1.4 验证online softmax的正确性
假设有数组x=[1...n]，用三种softmax
- naive softmax（2次load，1次store）
  一次循环计算出归一化因子，一次循环计算每个元素的softmax。
  $$\alpha_i = \frac{e^{x_i}}{\sum_{k=1}^n e^{x_k}}, \quad 1 \le i \le n$$
  ![](assets/Pasted%20image%2020260128053222.png)
- safe softmax（3次load，1次store）
  为防止 $x_i$ 过大，导致的 exp 为0。
  在naive的基础上，多遍历一次求全局最大值。
  $$\alpha_i = \frac{e^{x_i-m}}{\sum_k e^{x_k-m}}, \quad m=max(x), \quad 1 \le i \le n$$
  ![](assets/Pasted%20image%2020260128053240.png)
- online softmax（2次load，1次store）
  利用指数运算原理，在一次循环中在线计算出全局最大值和归一化因子（公式分母），减少循环次数。推导参考[FlashAttention 详细介绍](FlashAttention%20详细介绍.md)的第四章内容。
  1. 全局最大值 $m_{t-1}$
    $$m_t = \max(m_{t-1}, m_t^{(block)})$$
  2. 全局归一化因子 $l_{t-1}$ 
    $$l_t = l_{t-1} e^{m_{t-1} - m_t} + \sum_i e^{x_i^{t}-m_t}$$
  $$\alpha_i = \frac{e^{x_i-m}} {l_t}, \quad m=max(x), \quad 1 \le i \le n$$
  ![](assets/Pasted%20image%2020260128034738.png)
```
#include <vector>
#include <iostream>
#include <cmath>

std::vector<float> naive_softmax(const std::vector<float> src) {
    std::vector<float> dst(src.size());
    float sum = 0.f;
    for (int i = 0; i < src.size(); i++) {
        sum += std::exp(src[i]);
    }
    for (int i = 0; i < src.size(); i++) {
        dst[i] = std::exp(src[i]) / sum;
    }
    return dst;
}

std::vector<float> safe_softmax(const std::vector<float> src) {
    std::vector<float> dst(src.size());
    float sum = 0.f, mx = -99999.f;
    for (float f : src) {
        mx = std::max(f, mx);
    }
    for (float f : src) {
        sum += std::exp(f - mx);
    }
    for (int i = 0; i < src.size(); i++) {
        dst[i] = std::exp(src[i] - mx) / sum;
    }
    return dst;
}

std::vector<float> online_softmax(const std::vector<float> src) {
    std::vector<float> dst(src.size());
    float sum = 0.f, mx = -99999.f, pre_mx = 0.0f;
    for (float f : src) {
        pre_mx = mx;
        mx = std::max(f, mx);
        sum = sum * std::exp(pre_mx - mx) + std::exp(f - mx);
    }
    for (int i = 0; i < src.size(); i++) {
        dst[i] = std::exp(src[i] - mx) / sum;
    }
    return dst;
}

int main() {
    std::vector<float> src = {1.2f, 2.5f, 4.61f, 10.85f, 48.12f};
    std::vector<float> dst = naive_softmax(src);
    for (float f : dst) {
        std::cout << f << " ";
    }
    std::cout << std::endl;

    std::vector<float> dst1 = safe_softmax(src);
    for (float f : dst) {
        std::cout << f << " ";
    }
    std::cout << std::endl;

    std::vector<float> dst2 = online_softmax(src);
    for (float f : dst) {
        std::cout << f << " ";
    }
    std::cout << std::endl;

    return 0;
}

// 输出：
// 0.0493478 0.181072 1.49352 765.966 1.17588e+19 
// 0.0493478 0.181072 1.49352 765.966 1.17588e+19 
// 0.0493478 0.181072 1.49352 765.966 1.17588e+19
```

#### 1.5 online softmax 与 value 的点积优化
online softmax 仍然需要两次循环（2次load和1次store），如果能在一次循环中完成所有操作，则只需要一次存一次取。
因为要计算最大值和求和，online softmax 计算中已经将遍历次数优化至2次。但计算注意力权重和（softmax 点乘 V）的过程中，可以直接1次循环直接计算得到权重和。
1. 计算更新最大值和归一化因子：
  - 全局最大值 $m_{t-1}$
    $$m_t = \max(m_{t-1}, m_t^{(block)})$$
  - 全局归一化因子 $l_{t-1}$
    $$l_t = l_{t-1} e^{m_{t-1} - m_t} + \sum_i e^{x_i^{t}-m_t}$$
3. 和对应 block 的 value 做点积，求得注意力加权和 $O_t$ 。

$$\begin{aligned}
O_t = \frac{\sum_{(x,V)\in S_{ \lt t}} e^{x - m_{t-1}} e^{m_{t-1} - m} V + \sum_{(x,V)\in S^{(t)}} e^{x - m_t} V} {l_t} \\
= \frac{ l_{t-1} e^{m_{t-1}-m_t} O_{t-1} + \sum_i e^{x_i^t - m_t} V }{l_t}   
\end{aligned}$$
4. 处理完所有数据时，$O=O_t$ 。
```
float online_softmax_dot_product(const std::vector<float> src, const std::vector<float> value) {
    float dst = 0.f, l = 0.f, mx = -99999.f, pre_mx = -99999.f;
    for (float f : src) {
        mx = std::max(f, pre_mx);
        l = l * std::exp(pre_mx - mx) + std::exp(f - mx);
        pre_mx = mx;
    }
    for (int i = 0; i < src.size(); i++) {
        dst += std::exp(src[i] - mx) / l * value[i];
    }
    return dst;
}

float online_softmax_dot_product_perfect(const std::vector<float> src, const std::vector<float> value) {
    float dst = 0.f, l = 0.f, pre_l = 0.f, mx = -99999.f, pre_mx = -99999.f;
    for (int i = 0; i < src.size(); i++) {
        mx = std::max(src[i], pre_mx);
        l = pre_l * std::exp(pre_mx - mx) + std::exp(src[i] - mx);
        dst = (dst * std::exp(pre_mx - mx) * pre_l + std::exp(src[i] - mx) * value[i]) / l;
        pre_mx = mx;
        pre_l = l;
    }
    return dst;
}

int main() {
    std::vector<float> src = {1.2f, 2.5f, 4.61f, 10.85f, 4.12f};

    std::vector<float> value = {3.1f, 6.42f, 5.161f, 4.85f, 7.12f};
    float dst3 = online_softmax_dot_product(src, value);
    std::cout << dst3 << std::endl;

    float dst4 = online_softmax_dot_product_perfect(src, value);
    std::cout << dst4 << std::endl;

    return 0;
}

// 输出
// 4.85356
// 4.85356
```
## 1.6 手搓 flash attn 的注意力加权和算子
```
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void forward_kernel(const float *Q, const float *K,
                               const float *V, const int N,
                               const int d, const int Tc,
                               const int Tr, const int Bc,
                               const int Br,
                               const float softmax_scale, float *l,
                               float *m, float *O) {
    int thread_num_per_block = blockDim.x * blockDim.y;
    int tid = threadIdx.x;
    int bid = gridDim.x * blockIdx.y + blockIdx.x;

    const float *Q_start = Q + bid * d * N;
    const float *K_start = K + bid * d * N;
    const float *V_start = V + bid * d * N;
    float *O_start = O + bid * d * N;

    float *m_start = m + bid * N;
    float *l_start = l + bid * N;

    extern __shared__ float smem[];
    int offset = 0;
    float *Qs = smem + offset;
    offset += Br * d;
    float *Ks = smem + offset;
    offset += d * Bc;
    float *Vs = smem + offset;
    offset += d * Bc;
    float *Ss = smem + offset;

    // 遍历 K/V tiles
    for (int tc = 0; tc < Tc; tc++) {
        /* ---- 1. 载入 K/V tile 到 shared memory ---- */
        for (int idx = tid; idx < d * Bc;
             idx += thread_num_per_block) {
            Ks[idx] = K_start[tc * Bc * d + idx];
            Vs[idx] = V_start[tc * Bc * d + idx];

            // 初始化m_start、l_start、O_start
            m_start[idx] = -INFINITY;
            l_start[idx] = 0.f;
            O_start[idx] = 0.f;
        }
        __syncthreads();

        for (int tr = 0; tr < Tr; tr++) {
            int q_row = tr * Br + tid;
            if (q_row >= N) continue;
            /* ---- 2. 载入 Q row ---- */
            for (int idx = tid; idx < d * Br;
                 idx += thread_num_per_block) {
                Qs[idx] = Q_start[q_row * d + idx];
            }
            __syncthreads();

            /* ---- 3. 计算 S = QK^T + scale ---- */
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float acc = 0.f;
                for (int x = 0; x < d; x++) {
                    acc += Qs[tid * d + x] * Ks[y * d + x];
                }
                acc *= softmax_scale;
                Ss[tid * Bc + y] = acc;
                row_m = max(row_m, acc);
            }

            /* ---- 4. softmax（减 max）---- */
            float row_l = 0.f;
            for (int y = 0; y < Bc; y++) {
                float v = __expf(Ss[tid * Bc + y] - row_m);
                Ss[tid * Bc + y] = v;
                row_l += v;
            }

            /* ---- 5. online softmax 合并 ---- */
            float row_m_pre = m_start[q_row];
            float row_l_pre = l_start[q_row];

            float row_m_new = max(row_m_pre, row_m);
            float row_l_new =
                (__expf(row_m_pre - row_m_new) * row_l_pre) +
                (__expf(row_m - row_m_new) * row_l);
            row_l_new = max(row_l_new, 1e-6f); // 防止除零

            /* ---- 6. 计算 PV 并更新 O ---- */
            for (int x = 0; x < d; x++) {
                float pv = 0.f;
                for (int y = 0; y < Bc; y++) {
                    pv += Ss[Bc * tid + y] * Vs[y * d + x];
                }
                int o_idx = q_row * d + x;
                float o_prev = O_start[o_idx];
                O_start[o_idx] =
                    (row_l_pre * __expf(row_m_pre - row_m_new) *
                         o_prev +
                     __expf(row_m - row_m_new) * pv) /
                    row_l_new;
            }

            /* ---- 7. 写回 m / l ---- */
            m[q_row] = row_m_new;
            l[q_row] = row_l_new;
        }
        __syncthreads();
    }
}
```

# 二、LLaMA.cpp 中的 flash attn 算子
## 2.1 example.cpp 中 打开 flash attn 开关
![](assets/Pasted%20image%2020260129031254.png)
![](assets/Pasted%20image%2020260129031406.png)
