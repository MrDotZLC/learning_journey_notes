LLaMA.cpp是一个用C/C++实现的开源推理框架，目标是在**本地、低资源环境**（尤其是 CPU）上高效运行大型语言模型（LLM）。
llama.cpp的学习路线：
		配置llama.cpp的运行调试环境 ->
		使用example/simple/simple.cpp运行一个推理模型 ->
		利用debug逐步阅读例子的运行流程

# 环境准备
1. Ubuntu系统（或windows下的wsl2）
2. cuda环境
3. vscode环境
	1. 安装相关插件：python、c++、cmake等
	2. cmake配置：Use C Make Presets设为never
	   ![[Pasted image 20260109203805.png]]
	3. 
	4. 

4. 项目代码调整：
	1. 删除预调试文件：CMakePresets.json
	2. cuda开关：llama.cpp-ggml-CMakeList.txt中GGML_CUDA设为ON
	   ![[Pasted image 20260109203703.png]]
	3. simple.cpp参数调整：
	   
	4. 
5. 