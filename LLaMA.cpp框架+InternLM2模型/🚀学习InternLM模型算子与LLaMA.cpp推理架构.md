 InternLM和llama.cpp的职责划分
InternLM 负责“学会如何思考和生成文本”，llama.cpp 负责“高效、可控地把这种能力跑起来”。

| 组件        | 本质角色           | 是否包含模型参数 | 是否参与训练 |
| --------- | -------------- | -------- | ------ |
| InternLM  | 语言模型架构 + 预训练权重 | 是        | 是      |
| llama.cpp | 推理框架 / Runtime | 否（仅加载）   | 否      |

[[LLaMA.cpp 概述与环境准备]]

[[LLaMA.cpp 源码结构解析]]

[[InternLM 概述与环境准备]]

[[InternLM2 源码算子解析]]