环境：[LLaMA.cpp 概述与环境准备](Learning/LLaMA.cpp框架+InternLM2模型/LLaMA.cpp%20概述与环境准备.md)
本文主要讲述 LLaMA.cpp 将 InternLM2 1.8b 模型 F16 精度权重文件转成架构所需 gguf Q8-0 精度文件过程中的**量化**步骤。
源权重文件和目标文件的精度根据自身所需可调整，本文是F16转成Q8-0。

# 一、python转换程序中的量化操作
## 1.1 文件入口
python文件位于LLaMA.cpp目录下，名为convert_xxx_to_gguf.py。
![](Learning/LLaMA.cpp框架+InternLM2模型/Pasted%20image%2020260122050026.png)
InternLM2 1.8B模型的权重文件使用convert_hf_to_gguf.py。
## 1.2 函数入口
函数入口：
![](Learning/LLaMA.cpp框架+InternLM2模型/Pasted%20image%2020260122212637.png)
![](Learning/LLaMA.cpp框架+InternLM2模型/Pasted%20image%2020260122230936.png)
## 1.3 写入张量数据
![](Learning/LLaMA.cpp框架+InternLM2模型/Pasted%20image%2020260123022727.png)

