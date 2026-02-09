**[代码](https://github.com/MrDotZLC/cuda_practice/blob/main/6_cutlass_study/v2_turing_tensorop_gemm.cu)**
# 一、GEMM 分层架构

![](assets/Pasted%20image%2020260208153210.png)
下图是CUDA Core的分层架构
![](assets/Pasted%20image%2020260208164406.png)
下图是使用Tensor Core的分层架构
![](assets/Pasted%20image%2020260208164443.png)
![](assets/Pasted%20image%2020260208165004.png)

---
# 二、Device层
进入Kernel前的准备：计算layout、内存分配、tile形状。
![](assets/Pasted%20image%2020260208215245.png)
![](assets/Pasted%20image%2020260208215023.png)

---
# 三、Kernel 层
Kernel 内执行mma计算前的准备：将矩阵ABC进行拆解tile迭代器
![](assets/Pasted%20image%2020260208224821.png)

---
# 四、Threadblock 层
执行 Block MMA：
1. **预处理 prologue**：
   - 数据从 GMem 搬到 Reg，再搬到 SMem。（GPU指令集原因，普通的 cuda core 写法要求指令操作数中必须有一个是寄存器）
   - 为 Double Buffer 流水线对A、B迭代器进行状态更新
2. 同步 gmem_wait
3. 迭代 gemm_iters：
   - 外层迭代 block tile
   - 内层迭代 warp tile
![](assets/Pasted%20image%2020260208231256.png)
![](assets/Pasted%20image%2020260208231523.png)
---
# 五、Warp & Thread 层
执行 Warp MMA：
![](assets/Pasted%20image%2020260209154553.png)
