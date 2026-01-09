[InternLM](https://github.com/InternLM/InternLM?tab=readme-ov-file) 是商汤在Github、Hugging Face上开源的一系列语言大模型。受设备限制，笔者这里选取最小的[internlm2-1_8b](https://huggingface.co/internlm/internlm2-1_8b)模型进行学习，精度按自身情况选择F16/F32。
通过Transformers库加载模型到本地，并执行推理。
基于llama.cpp框架运行调试internlm2-1_8b模型。
# 环境准备
1. Ubuntu系统（或windows下的wsl2）
2. cuda环境（driver、cuda toolkit等）
3. python3环境（python3等）
4. python三方库：参考[InternLM](https://github.com/InternLM/InternLM?tab=readme-ov-file) github中的requirements.txt
	1. 注意与cuda环境适配
	2. 建议用虚拟python环境与llama.cpp的python环境隔离
5. llama.cpp调试运行环境
6. 开启cuda调试：llama.cpp>ggml-src-ggml-cuda-CMakeList.txt
![[Pasted image 20260109193326.png]]



