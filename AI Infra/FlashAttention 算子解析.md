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
- base softmax
  $$\alpha_i = \frac{e^{x_i}}{\sum_{k=1}^n e^{x_k}}, \quad 1 \le i \le n$$
- safe softmax
  为防止 $x_i$ 过大，导致的 exp 为0
  $$\alpha_i = \frac{e^{x_i-m}}{\sum_k e^{x_k-m}}, \quad m=max(x), \quad 1 \le i \le n$$
- online softmax
  对x进行分块
  全局最大值 $m_{t-1}$
  全局归一化因子 $l_{t-1}​$
  当前累积输出 $O_{t-1}​$

  $$\begin{aligned}
O_t = \frac{\sum_{(x,V)\in S_{ \lt t}} e^{x - m_{t-1}} e^{m_{t-1} - m} V + \sum_{(x,V)\in S^{(t)}} e^{x - m_t} V} {l_t} \\
= \frac{ l_{t-1} e^{m_{t-1}-m_t} O_{t-1} + \sum_i e^{x_i^t - m_t} V }{l_t}   
\end{aligned}$$

  ![](Pasted%20image%2020260128034738.png)

