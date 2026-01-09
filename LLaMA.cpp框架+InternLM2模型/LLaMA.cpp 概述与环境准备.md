[LLaMA.cpp](https://github.com/ggml-org/llama.cpp/tree/master)是一个用C/C++实现的开源推理框架，目标是在**本地、低资源环境**（尤其是 CPU）上高效运行大型语言模型（LLM）。
llama.cpp的学习路线：
		配置llama.cpp的运行调试环境 ->
		使用example/simple/simple.cpp运行一个推理模型 ->
		利用debug逐步阅读例子的运行流程

# 环境准备
## 1. Ubuntu系统（或windows下的wsl2）
## 2. cuda环境
## 3. vscode环境
1. 安装相关插件：python、c++、cmake等
2. cmake配置：Use C Make Presets设为never
   ![[Pasted image 20260109203805.png]]
## 4. python环境：
建议使用虚拟python3环境与InternLM的python3环境区别开，防止包冲突。
直接pip install -r requirments.txt
## 5. llama.cpp 项目代码调整：
1. 删除预调试文件：CMakePresets.json
2. cuda开关：llama.cpp-ggml-CMakeList.txt中GGML_CUDA设为ON
   ![[Pasted image 20260109203703.png]]
3. simple.cpp代码调整：
   ![[Pasted image 20260109210319.png]]
## 6. 模型获取
1. 这里选择[internlm2-1_8b](https://huggingface.co/internlm/internlm2-1_8b)模型
2. 加载模型到本地
   - 需要【科学】上网，模型3.78GB。
   ![[Pasted image 20260109210741.png]]
   - 模型缓存位置：
     
1. 