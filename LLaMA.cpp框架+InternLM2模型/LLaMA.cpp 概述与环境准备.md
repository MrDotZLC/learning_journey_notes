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
   ![Pasted image 20260109203805](Pasted%20image%2020260109203805.png)
3. debug配置：.vscode/launch.json
```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "CUDA C++: Launch",
            "type": "cuda-gdb",
            "request": "launch",
            "program": "/your/absolute/path/llama.cpp/build/bin/llama-simple"
        },
        {
            "name": "CUDA C++: Attach",
            "type": "cuda-gdb",
            "request": "attach"
        },
        {
            "name": "Debug Transformers & PyTorch (debugpy)",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",

            // 关键：允许进入第三方库
            "justMyCode": false,

            // 强烈建议显式指定解释器（虚拟环境）
            "python": "${workspaceFolder}/../internlm_venv/bin/python",

            "args": [
                "/your/absolute/path/.cache/huggingface/hub/models--internlm--internlm2-1_8b/snapshots/d753f1de0510e2551779f30e6a147fdbadb4a6ee",
                "--outtype", "q8_0" // debug uint8量化
            ],

            // PyTorch / Transformers 常用环境
            "env": {
                "PYTHONPATH": "${workspaceFolder}",
                "CUDA_VISIBLE_DEVICES": "0"
            },

            // 支持 DataLoader / DDP 子进程
            "subProcess": true
        }
    ]
}
```
## 4. python环境：
建议使用虚拟python3环境与InternLM的python3环境区别开，防止包冲突。
直接pip install -r requirments.txt
## 5. llama.cpp 项目代码调整：
1. 删除预调试文件：CMakePresets.json
2. cuda开关：llama.cpp-ggml-CMakeList.txt中GGML_CUDA设为ON
   ![Pasted image 20260109203703](Pasted%20image%2020260109203703.png)
3. simple.cpp代码调整：
   ![Pasted image 20260109210319](Pasted%20image%2020260109210319.png)
## 6. 模型获取
1. 这里选择[internlm2-1_8b](https://huggingface.co/internlm/internlm2-1_8b)模型
2. 加载模型到本地
   需要【科学】上网，模型3.78GB。
     ![Pasted image 20260109210741](Pasted%20image%2020260109210741.png)
   模型缓存位置：
	- Hugging Face缓存目录：
       windows：C:\Users\\<你的用户名>\\.cache\huggingface\\hub\models--internlm--internlm2-1_8b\snapshots\\<模型编码>\
       ubuntu：/home/<你的用户名>/.cache\huggingface\\hub\models--internlm--internlm2-1_8b\snapshots\\<模型编码>\
	- ubuntu下，快照文件都是软链，源文件存放在:
      /home/<你的用户名>/.cache\huggingface\\hub\models--internlm--internlm2-1_8b\blob\
	   ![Pasted image 20260109214806](Pasted%20image%2020260109214806.png)
3. pytorch_modle.bin转gguf文件
```
   python convert_hf_to_gguf.py <gguf文件路径> --outtype <精度>
   # 精度："f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"
   # 不考虑量化，精度一般选 f16/f32 ，默认 f16
   # 选择f32，gguf 文件会比源文件 pytorch_modle.bin 大1倍
```
4. 将gguf文件**复制**到build/bin中，防止误删

