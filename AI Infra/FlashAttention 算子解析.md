本文旨在学习FlashAttention，学习路径来自[比飞鸟贵重的多_HKL](https://www.bilibili.com/video/BV1FM9XYoEQ5?spm_id_from=333.1387.collection.video_card.click)，代码库可参考[flash-attention-minimal](https://github.com/MrDotZLC/flash-attention-minimal)。
# 一、寻找切入点
思考：如何快速debug一个算子？调试完整的flash-attention太笨重。

github找一个[flash-attention demo](https://github.com/tspeterkim/flash-attention-minimal)，发现其通过pytorch运行单个forward前向运算，本身是通过python调用C++代码访问cuda核函数。
考虑有CUDA C++的debug经验，用libtorch代替pytorch，直接用C++调用核函数，用CMake进行debug。

同理，不仅可以快速调试flash-attention的算子，也能够把LLaMA.cpp等框架的算子拿来debug和学习。

通过python调用CUDA C++的方式，也能够用cuda-gdb进行debug，在此不做深入探讨。

# 二、配置debug环境
下载解压 libtorch ，并在 CMakeLists.txt 中设置 CMAKE_PREFIX_PATH ，不要忘记打开-G。
![](Pasted%20image%2020260127011246.png)
具体代码参考[CMakeLists.txt](https://github.com/MrDotZLC/flash-attention-minimal/blob/main/CMakeLists.txt)。
# 三、运行一个demo
在 CMakeLists.txt 的 add_executable 中，把cu文件编译并链接到入口函数。
![](Pasted%20image%2020260127011355.png)
注意：在调用核函数后，最好执行一次`cudaDeviceSynchronize();`进行设备同步，原因在于Kernel异步执行，会导致cuda-gdb attach 不到Kernel内的断点。
![](Pasted%20image%2020260127020055.png)
# 四、验证online softmax的正确性
假设有数组x=[1...n]，用三种softmax
- native softmax
  $$\alpha_i = \frac{e^{x_i}}{\sum_{k=1}^n e^{x_k}}, \quad 1 \le i \le n$$
  ![](Pasted%20image%2020260128053222.png)
- safe softmax
  为防止 $x_i$ 过大，导致的 exp 为0
  $$\alpha_i = \frac{e^{x_i-m}}{\sum_k e^{x_k-m}}, \quad m=max(x), \quad 1 \le i \le n$$
  ![](Pasted%20image%2020260128053240.png)
  ```
  
  ```
- online softmax
  对x进行分块，处理一块 $x_t$ 的同时，维护下述三个变量，用于结果计算。推导参考[FlashAttention 详细介绍](FlashAttention%20详细介绍.md)的第四章内容。
  1. 全局最大值 $m_{t-1}$
  2. 全局归一化因子 $l_{t-1}​$
  3. 当前累积输出 $O_{t-1}​$
  $$\begin{aligned}
O_t = \frac{\sum_{x\in S_{ \lt t}} e^{x - m_{t-1}} e^{m_{t-1} - m} + \sum_{x\in S^{(t)}} e^{x - m_t}} {l_t} \\
= \frac{ l_{t-1} e^{m_{t-1}-m_t} O_{t-1} + \sum_i e^{x_i^t - m_t}}{l_t}   
\end{aligned}$$
  ![](Pasted%20image%2020260128034738.png)
```
#include <vector>
#include <iostream>
#include <cmath>

std::vector<float> native_softmax(const std::vector<float> src) {
    std::vector<float> dst(src.size());
    float sum = 0.f;
    for (float f : src) {
        sum += f;
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
    std::vector<float> dst = native_softmax(src);
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

# 五、online softmax 与 value 的点积优化
online softmax 仍然需要两次循环和两次存取，如果能在一次循环中完成所有操作，则只需要一次存取。

