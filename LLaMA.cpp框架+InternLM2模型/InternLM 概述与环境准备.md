[InternLM](https://github.com/InternLM/InternLM?tab=readme-ov-file) 是商汤在Github、Hugging Face上开源的一系列语言大模型。受设备限制，笔者这里选取最小的[internlm2-1_8b](https://huggingface.co/internlm/internlm2-1_8b)模型进行学习，F16仅需3.78G显存。
![Pasted image 20260109195802](assets/Pasted%20image%2020260109195802.png)
1. 通过Transformers库加载模型到本地，并执行推理。
2. 基于llama.cpp框架运行调试internlm2-1_8b模型，精度按设备情况选择F16/F32转成gguf文件（详见llama.cpp-convert_hf_to_gguf.py）。
以下环境需根据设备情况进行调整，故不提供详细版本号，自行AI/搜索。
# 环境准备
1. Ubuntu系统（或windows下的wsl2）
2. vscode及相关插件环境（python、c++、cmake等）
3. cuda环境（driver、cuda toolkit等）
4. python3环境（python3等）
5. python三方库：参考[InternLM](https://github.com/InternLM/InternLM?tab=readme-ov-file) github中的requirements.txt
	1. 注意与cuda环境适配
	2. 建议用虚拟python环境与llama.cpp的python环境隔离
6. llama.cpp调试运行环境[LLaMA.cpp 概述与环境准备](LLaMA.cpp%20%E6%A6%82%E8%BF%B0%E4%B8%8E%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87.md)
7. 开启cuda调试：llama.cpp>ggml-src-ggml-cuda-CMakeList.txt
![Pasted image 20260109193326](assets/Pasted%20image%2020260109193326.png)
8. 配置debug调试：.vscode下新增/修改launch.json
![Pasted image 20260109200136](assets/Pasted%20image%2020260109200136.png)

