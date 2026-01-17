 InternLM和llama.cpp的职责划分
InternLM 负责“学会如何思考和生成文本”，llama.cpp 负责“高效、可控地把这种能力跑起来”。
InternLM是通过transformer和pytorch可以直接进行训练和推理，生成了**权重 bin 文件**。
LLaMA.cpp 将**权重 bin 文件**转成其可识别的**权重 gguf 文件**，并加载其中包含的**模型结构、权重**，自己实现了所有结构并进行推理（不做训练），来进一步提高推理效率。

综上，学习思路是：先在InternLM python实现中，学习模型结构；再到LLama.cpp中学习每个算子的具体实现。（两边都需要 debug 调试，进入代码库中学习）

| 组件        | 本质角色           | 是否包含模型参数 | 是否参与训练 |
| --------- | -------------- | -------- | ------ |
| InternLM  | 语言模型架构 + 预训练权重 | 是        | 是      |
| llama.cpp | 推理框架 / Runtime | 否（仅加载）   | 否      |

[LLaMA](LLaMA.cpp%20%E6%A6%82%E8%BF%B0%E4%B8%8E%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87.md)

[LLaMA](LLaMA.cpp%20%E5%B7%A5%E7%A8%8B%E6%BA%90%E7%A0%81%E7%BB%93%E6%9E%84%E8%A7%A3%E6%9E%90%EF%BC%88%E5%BE%85%E6%95%B4%E7%90%86%EF%BC%89.md)

[InternLM 概述与环境准备](InternLM%20%E6%A6%82%E8%BF%B0%E4%B8%8E%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87.md)

[InternLM2 + LLaMA.cpp 源码算子解析](Learning/LLaMA.cpp框架+InternLM2模型/InternLM2%20+%20LLaMA.cpp%20源码算子解析.md)