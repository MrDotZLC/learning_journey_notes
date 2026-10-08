## 雷点
1. 别说"实现了 TensorRT-LLM / vLLM"。准确说法：**把其中若干核心机制手写了一遍**（分页 KV、

   自研 PagedAttention、设备侧采样、双引擎），且**明确不做**调度器 / 多卡 / 服务化。

2. 别说"TRT 没有 attention / RoPE / KV cache"。TRT 10.15 **有** `IAttention`、

   `IRotaryEmbeddingLayer`、`IKVCacheUpdateLayer`（KV cache **只有 `kLINEAR`**）。

   我们自研的理由是**分页布局**与**并行切分的控制面**，以及 prefill 手搭这条明确取舍

   （`docs/dev/REQ-004-gpt2-native/phase2_development_plan.md` 的 D1）。

3. 别说"支持任意 HuggingFace 模型 / LLaMA / Qwen"。注册表里只有 `gpt2` 与 `resnet18`；

   算子件齐 ≠ 能跑（LLaMA 缺 builder，SentencePiece **至今零用例、零调用方、未验证**）。

4. 别说"支持 INT4 / FP8 / KV cache 量化"。PagedAttention 只接受 FP32 / FP16。
5. 别说"做了 ONNX 子图替换"。那两条路只做到**子图识别与连接级判据**，替换是未排期的开放项。
6. 别说 `temperature != 1.0` 是"暂不支持"。是**显式拒绝**（静默忽略会让"调参无效"看起来像"模型就这样"）。
7. FP16 端到端**已经修好并真机验证**（2026-10-08），但讲的时候要带上"这是定点绕过 + 换 TRT 版本要重跑"

   这句边界；另外别说"batch 支持是没验证的"——**批量已经真机全绿**（REQ-016 已交付），

   但 **packed 路径的性能收益不成立**（见 §2F-44），这两句都要说准。

8. 不要引用别的项目的性能数字，也不要引用自己没有实测过的数字。

---

## Part 1｜应聘者视角：从部署到推理输出的全流程（我做了什么 / 怎么实现的 / 亮点在哪）

### 1.0 一句话定位 + 三档自述

> 一个**面向 Turing `sm_75` 的极简多模态 TensorRT 推理框架**（C++17 + CUDA）：

> 用同一套构建 / 引擎 / 运行时分层同时承载 CV（ResNet18）与 LLM（GPT-2），

> 并把 TensorRT-LLM 的核心机制——配置驱动建图、Paged KV Cache、自研 PagedAttention、

> 设备侧采样、prefill/decode 双引擎——**从零实现了一遍**，每条结论都保留实测出处。

**30 秒版**

> 我用 C++ / CUDA 手写了一个极简的 TensorRT 推理引擎，把 TensorRT-LLM 的核心机制实现了一遍：

> 配置驱动建图、分页 KV Cache、自研 PagedAttention（含长上下文的上下文维切分）、采样器、prefill/decode 双引擎，CV 与 LLM 跑在同一套框架里。

> GPT-2 FP32 的贪心输出与 HuggingFace 逐 token 一致，logits 相对偏差 `9.19e-07`；

> 长上下文下每步 decode 的延迟增长斜率降了 **88%**。整个开发过程按"先量、再优化、再验证"走，

> 60 多条排查记录每条都带命令与证据。

**2 分钟版**（按"背景 → 做法 → 验证 → 性能 → 收尾"五段）

1. **背景**：仓库里原本是两个各写各的 ONNX 示例（ResNet18 / GPT-2），我把它们合并成一个框架，并给自己定了一条约束：**只依赖 TensorRT + CUDA，关键 kernel 自己写**。目标是搞清楚机制，不是替换 TensorRT-LLM。
2. **做法**：`config.json` 描述结构 → `ModelRegistry` 分发到具体 `IModelBuilder`；

   权重走 Safetensors（自己写 dtype 转换）；同一个 builder 出三种切面  （`kSingle` / `kPrefill` / `kDecode`）；KV Cache 分页；decode 注意力用 `IPluginV3` 手写；

   采样器（greedy / top-k / top-p）全部设备侧，**解码循环内零 H2D/D2H**；另留一条 ONNX 路径做交叉验证。

3. **验证**：与 HuggingFace / PyTorch 对拍——GPT-2 FP32 贪心 **8/8 token 全中**， logits 相对偏差 `9.19e-07`；ResNet18 FP32 对 torchvision `9.5e-06`； INT8 用"FP32 余量子集一致率"判，实测 **12/12**。
4. **性能**：先建立可复现的测量口径与**判别下限**，再定位瓶颈——长上下文下 attention 占每步约 80%，机制是每层只有 `12 heads × 1 batch = 12` 个 block、而机器有 24 个 SM（一半闲置），且每块串行扫完整段上下文；改成上下文维切分（stage-1 局部 `m/l/acc` + stage-2 保序归约）后，斜率降 **88.85%**（kernel 级）/ **88.20%**（端到端），结果与单趟实现完全一致。
5. **收尾**：引擎构建指纹缓存、插件与图版本的手工代次契约、按 `REQ-NNN` 组织的 20 个开发条目，每个条目一套 artifact（需求 / 分析 / 设计 / 评审 / 基准 / 用例 / 总结 / 面试笔记）。

**5 分钟版**：2 分钟版 + 下面两条"反直觉的实测结论"（这是最能体现工程判断力的部分）

- **packed 混合批不如 padding**：我做了"把两相装进一个 packed 张量、只算真实 token"的路径，但放大到真实 GPT-2（prompt 192+64）实测**比 padding 慢 11.79%**（28.441 → 31.793 ms，远超本机 0.6 ms 的判别下限，三次可复现）。再做判别实验（把长度比从 1.5× 拉到 1.92×），劣势从 +11.2% 收到 +5.88% ⇒ 说明它**不是天生更贵，而是固定成本还没被省下的 token 抵消**。

  结论写进 `benchmark.md` 的 `Negative`，Gate-B 裁决"保留 opt-in、不继续投入"，并把 kernel 级定位另立成 REQ-020。

- **LLM 的 weight-only INT8 收益不成立**：prefill 中/长档慢 +10.2% / +4.1~4.4%，decode 无显著差异，体积两引擎合计只省 6.67%（且 decode 引擎反而大 7.7%）。根因是**这个形态不产生 INT8 计算**——48 个 Linear 折回 FP32、0 层带 Int8，多出来的 DQ 每步都要算。我原来的推导（"decode 访存受限 ⇒ 省带宽"）被实测否掉。
- **FP16 的 NaN 不只是"修好了"，而是"定位到了算子和数值机理"**：TRT 把 MLP 的 `c_fc` 的 MatMul+bias+GELU 融进一个 FP16 的 Turing GEMM epilogue，而那个 epilogue 的 tanh 在 FP16 下 `exp` 会溢出（自变量 `2v ≈ 120` ≫ `ln(65504) ≈ 11.09`）⇒ `inf/inf` = NaN。

  证据是**逐层一一对应**（该融合内核所在的层 ⟺ 该层残差 NaN），修法是 FP16 建图下把 GELU

  展开 + tanh 自变量限幅 ±5。详见 §0.3。

### 1.1 总图

```text

  【离线 · Python】                【进程启动 · C++】                    【请求 · C++/CUDA】

  

  HF 权重 ──convert──▶ config.json + model.safetensors ─┐

  （或 ONNX 图 ──convert──▶ config.json + *.onnx ───────┤

                                                       ▼

                                        EngineBuilder ──build──▶ .engine + .fingerprint

                                                       │  （指纹不一致/缺失才重建）

                                                       ▼

                              Engine（IRuntime → ICudaEngine → IExecutionContext）

                                                       │

                                      LLMRunner(Config, prefill_engine, decode_engine, tokenizer)

                                                       │  （构造期：精度约定校验 + KV 块池 + 缓冲）

  文本 ──BpeTokenizer──▶ ids ──▶ Prefill ──▶ 分页写 KV ──▶ Decode 循环 ──▶ tokens ──▶ 文本

                                  (S6)         (S6/S7)        (S7/S8/S9)      (S10)

```

### 1.2 逐段：S1~S10

#### S1 离线产图 —— 接口约定：HF / ONNX 权重 → `config.json` + `model.safetensors`（+ 量化清单）

**实现**：`tools/convert/hf_to_mini_trt_llm.py` 把 HF checkpoint 转成 `config.json`（`model_type` / `architecture` / `hyper_params` / `weight_map`）+ `model.safetensors`；

ResNet18 走 `onnx_to_mini_trt_llm.py`（**BN 在导出时就折进 Conv**）。INT8 另由 `quantize_*.py` 产出 `quant_int8.json` 清单 + int8 权重文件。

**取舍与代价**：`weight_map` 是"TRT 侧逻辑名 → 权重源 key"的映射，**改名只改 JSON、不动建图代码**；

代价是多一层间接（查表）与一份必须与权重同步维护的文件。量化走**显式 Q/DQ 图**而不是

`IInt8Calibrator`：`BuilderFlag::kINT8` 自 TRT 10.12 起废弃，而显式 Q/DQ 在图上看得见、能逐层自证。

**归属**：手写——转换脚本、`weight_map` 设计、dtype 转换（FP16 / BF16 / FP64 **统一先还原 FP32

再降精度**，BF16 位截断不成立）、转换结果按张量名缓存（保证 TRT 的裸指针在建图期稳定）。

复用——safetensors 文件格式与解析。
#### S2 引擎构建与缓存 —— 接口约定：`model_dir` + `Config` → `<engine>` + `<engine>.fingerprint`

**实现**：`EngineBuilder::BuildFromConfig(model_dir, engine_path, stage)` → `SetupBuilder`（workspace / builder flags）→ 模型 builder 的 `Build(network, weights, config, BuildOptions)`→ `AddLlmOptimizationProfiles` / `AddCvOptimizationProfile` → `buildSerializedNetwork`→ `SerializeAndSave`（引擎与指纹**一起落盘**）。ONNX 路径 `BuildFromOnnx` 走同一套 profile 语义。

**取舍与代价**：① `BuildStage{kSingle, kPrefill, kDecode}` 把"建哪个切面"显式化——prefill 图与decode 图的 I/O 约定（I/O contract）不同（前者导出每层 K/V、attention 手搭；后者走 PagedAttention 插件、吃每层 cache），**不是同一张图的两个 profile**；② `BuildOptions` 携带 `weight_dtype` / `quant` / `export_diagnostics`/ `packed_mixed`，因为 **builder 必须显式知道目标 dtype 与切面**，TRT 的 builder flag 只表达"整网倾向"；

③ 缓存复用**只由指纹决定**。代价：profile 区间或任何建图数值一改就要重建（分钟级/份）。

**归属**：手写——profile 规则、指纹 sidecar、全部建图逻辑与护栏（算子组合、命名、精度决策）；

复用——TRT 的构建与 tactic 选择、TRT 层算子（MatMul / Slice / Softmax / Gather / Conv…）。

**`BuildOptions` 逐项**（`include/mini_trt_llm/core/imodel_builder.hpp` —— 它是**传给 `IModelBuilder::Build`的"本次构建语义"**，不是模型描述、也不是 profile 配置）：

| 字段                            | 默认                                                           | 语义                                                                                                                                                    | 谁在读                                                                                  | 为什么有它 / 代价                                                                                             |
| ----------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| `stage`                       | `kSingle`                                                    | 建哪个切面：`kSingle`（一个引擎挂 prefill+decode 两组 profile，Phase 1.5 的旧行为）/ `kPrefill`（整段前向 + 导出每层 K/V）/ `kDecode`（单 token + PagedAttention 插件）                  | GPT-2 builder 推 `is_decode` / `export_kv`；**ResNet18 builder 只接受 `kSingle`**，其它值直接报错 | 切面是**构建期**参数：模型描述（`ModelConfig`）说不清、profile 配置里也没有它的位置                                                 |
| `weight_dtype`                | 结构体默认 `kFLOAT`，但生产路径**总被 `Config::precision` 覆盖**（工程默认 FP16） | 权重**常量层**建成的精度                                                                                                                                        | 两个 builder 都用它建常量；GPT-2 还用它决定 **GELU 走哪条路**（FP16 才展开 + 限幅）                           | TRT 的 builder flag 只表达"整网倾向"，回答不了"这个常量该是什么类型"                                                          |
| `quant`                       | `nullptr`                                                    | 量化清单指针：非空时，清单点名的权重走"int8 常量 + DQ"，其余走原路径                                                                                                              | 仅 GPT-2 builder（存成 `quant_`）                                                         | **生命周期只覆盖本次 Build**：调用方要在同一作用域持有 `QuantSpec`，并保证它在 `buildSerializedNetwork` 之前不析构；`nullptr` 时建图逐位回到改动前 |
| `export_diagnostics`          | `false`                                                      | 是否把排查用的中途张量挂成网络输出                                                                                                                                     | GPT-2 builder，且**只对 prefill 生效**（`export_diagnostics && export_kv`）                  | 它是 **I/O 约定的开关**而非调试开关：打开后每个消费方都要多分配并绑定，否则 enqueue 被拒（`TS-019`）                                        |
| `diagnostic_probe_point_mask` | `kDiagnosticProbeAllPoints` = `0xFF`                         | 探针**切点位图**：位序 = `kDiagnosticProbePointNames`（`block_in` / `attn_ctx` / `attn_out` / `attn_res` / `ln2_out` / `mlp_fc` / `mlp_gelu` / `mlp_res`，共 8 个） | GPT-2 builder（按**名字**匹配；名单外的切点名**响亮失败**，不静默丢掉探针）                                     | REQ-018 二分定位时加的仪器：**挂不挂探针本身就是另一张图**，所以要能选子集、并且进引擎指纹                                                    |
| `diagnostic_probe_max_layer`  | `-1`（= 全部层）                                                  | 只挂前 N+1 层的探针                                                                                                                                          | 同上                                                                                   | 同上（二分时逐层收窄）；`-1` 是"全部层"的哨兵                                                                             |
| `packed_mixed`                | `false`                                                      | prefill 是否建 S9 的 **packed 混合批图**（一个张量装两相、K/V 输出 token-major `[T,NH,D]`、去掉 `padding_bias`）                                                             | GPT-2 builder；**decode 阶段传 `true` 会被拒绝**（"only supported for the prefill stage"）     | 它改的是**图的 I/O 约定**，所以必须与 `kPackedPrefillGraphVersion` 配套使用                                              |

**`EngineBuilder::Config` 与 `BuildOptions` 的分工**（常被问"这两个配置为什么不合并"）：

前者是**构建器**的配置（精度 / workspace / 两组 profile 区间 / 开关 / 量化清单路径 / 探针子集），决定"建什么形状的引擎"与"缓存键是什么"；其中 **profile 区间与 workspace 只进 TRT 的 `IBuilderConfig`、不进 `BuildOptions`**（模型 builder 不需要知道）。后者是**传给模型 builder 的本次构建语义**。

两者的传递是 `builder.cpp` 里一次**显式映射**：`stage`（入参）→ `weight_dtype = ToTrtDataType(config_.precision)`→ 诊断/打包的**四个开关**（`export_diagnostics` / 两个探针字段 / `packed_mixed`）直传 →

`quant` 由 `quant_manifest` 解析而来。

不合并的理由：`ModelConfig` 描述**模型是什么**（层数 / hidden / `weight_map`），

而这两类是"**这次怎么建**"——混在一起会让同一个模型的不同切面拥有两份可能不一致的描述。

**★ 会被追问**

- **`BuildOptions` 里为什么要这些开关？和 `EngineBuilder::Config` 怎么分工？** 见上面两张表；

  一句话：**Config 管"建什么形状 + 缓存键"，BuildOptions 管"这次建哪个切面、用什么精度、开什么仪器"**。

- **为什么要双引擎？** 见"取舍"第 ①。继续追：代价？→ 两份引擎体积与构建时间；`kSingle` 保留给  Phase 1.5 的旧行为与 ONNX 路径。坑：说成"为了性能"（主因是 I/O 约定不同）。

- **指纹包含什么？为什么不用编译时间？** `EngineFingerprintInputs` = stage + precision + source_kind +  源文件身份（size+mtime）+ 全部建图数值参数（profile 区间、workspace、`model.n_positions`）+ 开关（`export_diagnostics` / `detailed_profiling` / 诊断探针版本与子集）+ TRT 版本 + CUDA runtime 版本 +  **手工 `graph_version`** → 64-bit FNV-1a → `<engine>.fingerprint`。不用编译时间：那会让任何一次

- **CV 的 profile 为什么是 1/8/16？** TRT 针对 `kOPT` 形状挑最快 kernel；历史工程就用 1/8/16，

  改回 `opt=1` 会让 batch ≥ 2 走非最优 kernel、性能结论失真（`REQ-007` 的 PH4-CV-PROFILE）。

#### S3 引擎装载 —— 接口约定：`.engine` 路径 → 可用的 `Engine`

**实现**：`core/engine.cpp` —— `IRuntime::deserializeCudaEngine` → `createExecutionContext`，外面包一层 `SetInputShape` / `SetTensorAddress` / `SetOptimizationProfile` / `Enqueue`，外加两个符号查询接口 `GetProfileDims` / `GetProfileDim`。

**取舍与代价**：这一段是 TRT API 的薄包装，唯一自研的护栏是**精度与形状一律向引擎查询、不按 `Config` 假定**。代价：多几次 `getTensorDataType` / `getProfileShape` 调用（构建期与构造期，不在热路径上）。`GetProfileDims` 的存在理由：运行时需要"这一步最多能送多少 token"这类上界，而它是**建图期定的、Config 里没有的**。

**归属**：手写——绑定 / 形状设置 / 符号查询 / 缓冲精度决策；复用——TRT Runtime 与 `IExecutionContext`。

#### S4 Runner 构造 —— 接口约定：`Config` + 两个引擎 + tokenizer → `LLMRunner`（`ok()` 为真）

**实现**：构造函数不做计算，只做**约定校验（contract check）+ 资源准备**：

`decode.key_cache_0` 的声明精度 == `Config::is_half`；
**逐层** `k_layerN` / `v_layerN` 与第 0 层一致；
prefill 与 decode 的 K/V 精度一致；
packed 模式下从侧车读 `n_positions` 并做两条自检、`chunk_limit` 过五条交叉校验（① ≥ 1；② ≤ `n_positions`；③ `L × block_tables.dim0.max ≤ input_ids.dim1.max`；④ ≤ 插件上限 1024；⑤ profile 查询必须成功）。任一条不过 → `valid_ = false`，**拒绝启动并打印实际值与上界**。

资源侧：`PagedKVCache` 按 `num_layers × num_blocks × block_size × num_kv_heads × head_size` 一次分配（`BlockAllocator` = `vector<bool>` + `std::list<int>` 空闲表），各缓冲按容量水位备好。

**取舍与代价**：把"内存越界"提前成**一条可读的启动错误**；代价是构造期多几次查询、

以及调用方必须显式提供那些"引擎查不到、Runner 又推不出来"的几何量。

**归属**：手写——全部校验逻辑、KV 块池与块分配器、缓冲水位管理；复用——无（这一段是纯自研）。

**★ 会被追问**

- **为什么逐层校验 K/V 精度，不只是第 0 层？** 启动闸此前只查 `key_cache_0` / `k_layer0`，

  其余层是**假定**相同；若某层不同，那层缓冲会按错误宽度读写，与 S3 那条完全同型且**不报错**。

  附带收益：`getTensorDataType` 对**不存在的名字**按 TRT 约定返回 `kFLOAT`，所以张量名写错也会响亮失败。

- **`num_blocks` 和 `max_blocks_per_seq` 是一回事吗？** 不是：前者是**物理块池**，后者是  `block_tables` 的**宽度**（`ceil(n_positions / block_size)`）。物理池可以更大：插件用运行期算出的  偏移在连续缓冲里寻址，TRT 不校验缓冲容量，所以"池 ≥ 宽度"就不会越界；构造期有反向检查。

- **分页布局长什么样？"两套长度"指什么？** 布局 `[num_layers, num_blocks, block_size, num_kv_heads, head_size]`，  每层切出一段 4-D 交给插件。两套长度：host 侧 `sequences_[id].length`（记账）与设备侧 `context_lens[b]`  （**引擎与 kernel 实际读的那个**）；decode 每步在设备端推进后者。

#### S5 请求入口 —— 接口约定：文本 → ids → `GenerateRequest`；失败 = 整批拒绝

**实现**：`BpeTokenizer`（byte-level BPE）把文本切成 ids；`GenerateBatch` 在入口逐条校验：

批内**等长**、**同一采样策略**、**同一 seed**、`seq_id` 唯一、`top_k ≤ 64`；任一条不满足 →**整批拒绝**（不做部分成功）。返回值语义：**只含新生成的 token（不含 prompt）**，空 vector 表示失败。

**取舍与代价**：`BpeTokenizer::Load` 的入参是**目录**（要 `vocab.json` + `merges.txt` 两个文件）——

基类只有一个路径参数，为它专门放宽了约定；预切分里的 ` ?` **只吃字面空格 U+0020**，

因为写成"任何空白"会把 `"a\tb"` 切成 `["a","\tb"]`（HF 是 `["a","\t","b"]`）。

**失败与"生成 0 个"必须可区分**，所以"至少 1 个 token"才算成功。

**归属**：手写——byte-level BPE、入口校验、`seq_id` 分配；复用——SentencePiece（已嵌入但**未验证**）。

**★ 会被追问**

- **为什么 tokenizer 不放进 `LLMRunner` 的推理路径？** Runner 只做 token-id 级生成（收 id、还 id），

  tokenizer 只在构造时校验；把分词混进来会让"数值对不对"和"分词对不对"两个问题纠缠在一起。

- **批内为什么要求"同一种采样策略"？** S1 的限制：三种策略各自是"整批一个分支"，

  但 `top_k` / `top_p` 的**值**可以逐序列不同（D3 的实质部分）。伪命题要能答：不是"不支持逐行参数"。

#### S6 Prefill —— 接口约定：`[B,S]` + 每层 K/V 输出 → 分页 cache 的 prefix K/V + 第 1 个 token

**实现**：`BindPrefill` 绑四类张量：`input_ids[B,S]`、`position_ids[B,S]`、**每一层的 K/V 输出**（`k_layerN` / `v_layerN`，只有 `kPrefill` 切面才挂）、padding_bias[B,1,1,S]`（批内长度不齐时按每行真实长度填加性 `-1e4`）。一次 `enqueueV3` 后，`PagedKVCache::WritePrefillKV` 按 block table 把 `[B, NH, S, D]` 散写进 `[num_blocks, block_size, NH, D]`（模板按**源精度 × 目标精度四种组合** 显式分发），`UploadMetadata` 把块表与语境长度同步到设备；最后一次采样拿到第 1 个 token。

**取舍与代价**：prefill 的 attention **手搭**（`MatMul → scale → causal mask → Softmax → MatMul`），而不是用 TRT 的 `IAttention(causal=true)`。三条理由：可按算子对拍定位差异、把同一份 K/V 张量复用给KV Cache、Turing 上融合 kernel 的可用性需真机确认。代价：两条路各自建引擎、代码量更大。

掩码走**图上常量**（`causal_mask_`，`n_positions × n_positions`），而不是每次前向现算。

**"为了能写进 KV Cache"这条要讲准**：手搭的直接收益是**能在图上把每层 K/V 挂成网络输出**

（`k_layerN` / `v_layerN`），再由 `WritePrefillKV` 按**我们自己的分页块表**散写。TRT 那条现成的路是`IAttention` + `IKVCacheUpdateLayer`，而 `KVCacheMode` **只有一个取值 `kLINEAR`**（`NvInfer.h` 可核）——用现成的就等于接受线性 K/V 布局、放弃分页。

**归属**：手写——整段前向的算子组合、K/V 导出、分页写入 kernel、mask 构造；复用——TRT 的算子层。

**★ 会被追问**

- **prefill 的 attention 为什么手搭？** 见"取舍"；继续追：那要不要换成 `IAttention`？→ 属 `REQ-019` 的  **S3（子图替换）**，前置是 PF-7 的跨构建对照，而且 **PF-7 只决定 S3**，不决定整条。

- **为什么需要行映射？** 源张量的行序与块表的行序**不总是同一个集合**：调度下 context 段只装

  "本步新入批的行"，而块表里还有正在 generation 的行；少了映射就会拿源缓冲里上一轮的残留行覆盖**别的序列自己的** prompt K/V（静默算错）。反向也查过（`TS-054`）：decode 追加的行序等于批内顺序，所以走构造期备好的**恒等表**，免得每步拷一次。

- **K/V 为什么是每层一对独立张量？** 这是 `TS-015` 的教训：早期所有层共用一张 cache 张量 → 每层都读第 0 段 cache，2 层差 `1.2e-3`、**12 层完全失真**。代价是 I/O 数量随层数线性增长 （也是"这套做法撑不住 70B"的直接原因）。

#### S7 Decode 单步 —— 接口约定：`[B,1]` + 每层 cache → `logits[B,V]`（**循环内零同步**）

**实现**：每步 = `BindDecode`（token / 位置 / 每层 cache 指针）→ `LaunchFillPositionIds`（用**设备端** `context_lens` 填 `position_ids`）→ PagedAttention 插件（块表寻址 + online softmax，长上下文自动切 split-K）→ `AppendDecodeStep`（一次写全层 K/V，**只在末尾推进一次**长度）→ 采样（结果直接留在显存，下一步当 `input_ids`）。

**取舍与代价**：三处配合才做到"循环内零 H2D/D2H"；代价是**无法在循环内早停 EOS**（要早停必须每步知道设备上的 token 值）——多算几步、语义仍正确，这是**有意取舍**。

split-K 的代价：短上下文多一次归并（实测 +1.877%，观测项），它来自"宿主判定不可行"，不是顺手能优化掉的东西。

**诚实边界**：没做 block 共享 / copy-on-write / prefix 复用，所以**别把它讲成"解决内存碎片"**（那是 vLLM 论文的口径；本项目既没做那几件事，也没有对应实测）。另外分页 cache 是**Phase 1/2 就落地的**（`REQ-001` / `REQ-002` / `REQ-004`），那时 `LLMRunner` 还是 `batch = 1`——"请求长度不一"不是它当初的动机，"按需分配 + 可增补退出"才是。

**归属**：手写——插件本体与全部 CUDA kernel、写入/推进 kernel、位置填充 kernel、分片规则；复用——TRT 插件框架。

**★ 会被追问**
- **★ 零同步具体怎么做到的？** 三处：位置由设备端 `context_lens` 填、长度在设备端推进、  采样结果直写显存并当下一步输入。继续追：EOS 怎么办？→ 循环外一次性 D2H 后截断；  真要早停需要设备侧 stop flag + 条件图（G2-4）。坑：说"没有同步"（循环外有一次，且**必须**有）。

- **长度为什么必须"整步只推进一次"？** `AppendDecodeKV`（只写、不推进）+ `AppendDecodeStep`  （一次写全层、只推进一次）：长度是 **per-token 量**，挂在"每层一次"的接口上会被推进 `n_layer` 倍  ——第 3 个新 token 起就发散（`TS-016`）。

- **split-K 的 stage-1 / stage-2 各做什么？为什么不能原子累加？** stage-1 出局部 `m/l/acc`  （online softmax 中间量）；stage-2 用 max-trick 归约。不能用原子累加是因为**不保序 → 结果不可复现**，  而判据正是"生成结果与单趟实现**完全一致**"。

- **分片数为什么是 8 / 目标片长为什么是 128？** `kPagedAttentionMaxSplits = 8`、  `kPagedAttentionTargetChunk = 128`，来源写在头文件里：改动前每层只有 `12 heads × batch(1) = 12`  个 block，而本机 **24 个 SM**——一半闲置，且每块要串行扫完整段上下文（976 次迭代）。取 8 让每层 block  数变成 `12 × 8 = 96`（24 SM 的 4 波）；再加会被"每位置一次 BlockReduceSum"+ 归并开销吃掉收益。

  128 让 ctx≈20 落在 1 片、ctx≈976 用满 8 片。这些数字是**先量再定**的（`REQ-014`）。

- **空片为什么要写哨兵？** 宿主拿不到"当前上下文多长"（`context_lens` 在设备上、又不许 D2H），

  所以 stage-1 的 grid **恒为上限**，超出有效片数的 block 必须立即返回、不读不写，并写

  `m = -inf, l = 0`；否则归并会读未初始化显存。Merge 侧还有 `l <= 0` 兜底防 `exp(-inf − -inf)`。

- **`getWorkspaceSize()` 为什么按 `desc.max` 报？** 动态轴在 `desc.dims` 里是 `-1`，按它算会偏小、  kernel 照写就是越界写——而这只在真机暴露。紧跟着的一问：**布局谁定？** →  `plugins/paged_attention_split.hpp` 是**唯一实现**（`__host__ __device__`），`getWorkspaceSize()`  报多少字节、kernel 往哪写走同一组函数；各写一份最容易在"改片数 / 改 head_size"时静默错位。

- **插件反序列化后还会执行 `configurePlugin` 吗？** 不会。而 head 配置是从输入形状推导的，  所以运行期入口必须以**当前形状**为准重新自洽，否则正常形状会被误判成"改了 head 配置"（`TS-008`）。

  相关设计：插件只序列化 `rotary_dim` / `base` 这类属性，`num_heads` / `num_kv_heads` / `head_size`  全从形状推导，避免第二份可能与形状失配的状态。

#### S8 采样 —— 接口约定：`logits[B,V]` + per-row k/p/seed → `token_ids[B]`（设备）

**实现**：分派规则 `top_p < 1` → top-p；否则 `top_k == 1` → greedy；其余 → top-k。greedy 不做 softmax（单调性等价于 argmax），两级归约、并列取小下标；top-k 快速路径 = "一行一个 warp + k 轮 warp 归并"（上限 `kTopKFastMaxK = 64`）；top-p = CUB 分段排序 + "块和 → 子块和 → 元素"三级定位。

Philox 自己实现（省 `curand` 依赖 + 固定 `(seed, offset)` 可复现）。

**取舍与代价**：`SamplerArgs` 里**所有指针都必须是设备地址**，`seeds` / `offsets` 让随机流只由`(请求 seed, 该行自己的步号)` 决定；workspace 由调用方预先算好（**不在循环里 `cudaMalloc`**）。

top-p 的阈值不逐项归一化，而是把阈值乘上总和（`u · Σ`）——数学等价、省一次遍历。

**归属**：手写——三个采样 kernel、Philox、前缀定位逻辑；复用——CUB 的分段排序。

**★ 会被追问**

- **为什么参数全是设备指针？** 因为 decode 循环必须全程驻留显存。显然的两个是 `logits` / `token_ids`；

  **容易被当成 host 数组的**是 `seeds` / `offsets` / `top_k` / `top_p` / `eos_hit`——它们由  `LLMRunner` 上传到常驻设备缓冲后再交给采样器。传 host 数组的表现是真机 `illegal access`，  或"读到垃圾随机流"这种**不报错的错**。

- **随机流为什么不能把行号算进去？** 判据是"**同一请求无论落在批内哪一行、批次怎么组成，token 序列都逐位相同**"，  所以非空 `seeds` 时走 `Uniform01(seeds[row], offset, 0)`；`offsets` 是 per-row 的**随机步号** （= 该行自己的已生成计数）。怎么证明？→ `BatchEqualsSequentialUnderScheduling` 与  `DeterminismWithArrivalSteps`（后者缺陷根因就是"逐行参数只认第 0 行"，`TS-057`）。

- **greedy 需要做 softmax 吗？** 不需要（softmax 单调）；并列取小下标以对齐 `torch.argmax`。
- **top-k 快速路径更快吗？** **不**——正确性与旧实现逐 token 相同

  （`SamplerKernelTest.TopKFastMatchesLegacyTokens`），但基线实测**慢 6~9 倍**，已撤出生产、

- **`eos_hit` 是干什么的？** 采样时顺带写"该行采到 EOS"的**设备侧标记**，调度器每步只回读

  `batch_size` 字节就能决定谁退出（不必把 token 拷回主机自己比 EOS）。

#### S9 循环与调度 —— 接口约定：多个 `SchedulerRequest`（带 `arrival_step`）→ `GenerateResult[]`

**实现**：每轮五步——`retire`（完成/EOS 退出、行号压实、归还块）→ `admit`（按 `arrival_step` 与块池预算准入）→ context 段（**只装本步新入批的行**）→ generation 段（只装本步在跑的活跃行）→ 结果落位 + 交回 finish flag。两条 prefill 路径：默认 `kPaddedTwoPhase`（右填充 + `padding_bias`）与 opt-in `kPackedMixed`（一个 packed 张量装两相、`cu_seqlens_ctx` 分段）；长 prompt 另有 chunked prefill（每步最多 `max_prefill_seq_len` 个 token，位置取**绝对位置**）。

**取舍与代价**：三条不变量——**活跃行号 = cache 行号**；**写回必须带行映射**；

**随机流 = `(请求 seed, 该行自己的已生成计数)`**。验证两层：结果侧"批量 == 逐条单跑逐位相同"，

形状侧 `SchedulerStats` 自证"分块 / 混合步真的发生了"。代价：每步重建块表/语境长度镜像并 H2D；

packed 路径省 token 但**实测在长度不齐下更慢**（见下）。

**因果链（连续批 → 不等长 → 两条 prefill 路径）**：连续批把"**同一步里既有 context 行、又有generation 行，且各序列长度不等**"变成常态。默认的 padding 路径用**右填充 + `padding_bias`** 处理长度不齐——**连续批靠它就跑通了**（S3 的两段式），代价是**按最长计费**（填充位置的算力白算）；

packed 路径是"不想为填充付算力、并把两相并成一次调用"的**另一种解法**（S4，`cu_seqlens_ctx` 分段），代价是每步打包与元数据重建。口径要点：**packed 不是连续批的前提**，它是 opt-in 的另一条路（默认仍是 padding），而且 P7 实测它在长度不齐下**更慢**。

**归属**：手写——调度循环、行映射、预算算法、packed 打包与写回、`SchedulerStats` 观测口；

复用——无。

**★ 会被追问**

- **为什么"批量 == 逐条单跑逐位相同"是硬判据？** 它同时管住三件事：KV 落点、位置编码、随机流。

  只要任一与"批位置"耦合，结果就不再与单跑一致。

- **长度不齐怎么处理？** 默认右填充 + `padding_bias`（加性 `-1e4` 填在注意力分数上，

  `[B,1,1,S]` 的 FP32 常量输入）；opt-in packed 只算真实 token。两条路径**调用 API 同一套** （AC8 明写"不要求跨路径逐位相同"）。继续追：为什么不改图实现掩码？→ 图不变、代价只是 S 维上界。

  **坑：说"packed 是为了实现连续批"**——连续批在 S3 的 padding 路径上就已经跑通了，

  packed 是"不想为填充付算力"的 opt-in 解法，而且实测更慢（见 F-6）。

- **调度五步是什么？** 见"实现"。其中 context 段"只装新入批的行"是**硬前提**：整批跑 context 会把在跑序列的 K/V 覆盖掉（静默算错）。

- **packed 图的 I/O 约定和 padding 图差在哪？** 多 6 个输入、K/V 输出从 `[1,NH,T,D]` 换成  **token-major** 的 `[T,NH,D]`、去掉 `padding_bias` ⇒ 两套图**各自一个图版本**、复用错引擎是静默错。

  `cu_seqlens_ctx` 的行维上界是 `max_prefill_batch + 1`（长度是 `B_ctx + 1`；用 `max_prefill_batch`  会让"整批都是 context 行"的首步 `setInputShape` 直接失败——5→6 那次 bump 就是它）。

- **★ packed 为什么反而更慢？** 先把结论说准：**放大到真实 workload（GPT-2，prompt 192+64）慢 11.79%**  （28.441 → 31.793 ms，绝对差 3.352 ms ≫ 判别下限 0.6 ms，三次可复现）；判别实验（192+8，  浪费比 1.5×→1.92×）把劣势从 **+11.2% 收到 +5.88%** ⇒ **不是天生更贵，而是固定成本还没被省下的 token  抵消**，趋势指向交叉点；**具体来源未定位**——这正是 `REQ-020` 要做的 kernel 级 profiling。

  继续追：默认值改了吗？→ **没有**，默认仍是 padding；`requirement.md` 里写的"打包为默认"在实测面前不成立。坑：把它讲成"优化失败"——它的价值是"可复现 + 被判别实验拆成两项"。

- **`arrival_step` 为什么由调用方给？** 调度必须**可复现**，否则"批量 == 逐条单跑"的逐位对拍没法做。
- **packed attention 和 paged attention 什么关系？** `PackedAttentionPlugin` 在一个 packed 张量里装两相、按段分派两条 kernel：context 段是 **chunked 分页因果注意力**（K/V = 分页缓存前缀 ++ 本 chunk）generation 段每行 1 个 token；段边界由 `cu_seqlens_ctx`（设备值）表达，不引入 D2H。

  硬约束：context kernel 按 `kPackedAttentionMaxContextSeqLen = 1024` 分配 `__shared__ s_scores[]`。

#### S10 输出与观测 —— 接口约定：设备 tokens → host `int64` → 文本；过程可证

**实现**：循环外一次 `cudaMemcpyAsync` + 同步 → 主机侧按各行的 `max_new_tokens` 截断 EOS → `int32 → int64` → 可选 detokenize。观测侧：`SchedulerStats`（`steps` / `max_active` / `context_rows` / `generation_rows` 是**跨路径**口径，`prefill_calls` / `decode_calls` **只属于 S3 两段式**）、诊断探针（默认关）、引擎指纹 sidecar。

**取舍与代价**：EOS 截断放在循环之后（见 S7）；诊断探针默认关是因为它改 I/O 约定。

CV 路径同构：`CVRunner` 的输入约定是 float32 / NCHW / `[0,255]`、归一化在 Runner 内、维度与 batch 范围**一律向引擎查询**（输入用 `getProfileShape`、输出用 `getTensorShape`）、`pixels_per_channel` 显式传入（不从总长度反推——`batch = 1` 时错法与对法恰好等价，`TS-023`）。

**归属**：手写——结果切分 / EOS 截断 / CV 前处理与 Runner；复用——无。

**★ 会被追问**

- **`SchedulerStats` 能当性能指标吗？** 不能，它是**只读观测口**（不参与计算）。它的存在理由：

  "采到 EOS 的序列下一步就退出""context 段只装本步新入批的行"这两条判据在公开接口上本来不可观测。

- **失败语义是怎么设计的？** `Generate` / `GenerateBatch` 失败返回**空 vector**（成功至少 1 个 token）；

  `GenerateBatch` 任一条不满足即**整批拒绝**、不做部分成功；`CVRunner::Infer` 失败返回空 vector、

  `Benchmark` 失败返回零值统计；都不抛异常（异常留给构造期与底层库错误）。

### 1.3 跨阶段的四组追问（不属于任何单一段落）

#### X1 定位与取舍

**为什么不用 TensorRT-LLM / vLLM？** 三条：① 目标是搞清楚机制，调库学不到；② 这台机器**没有 Tensor Core**，靠调库吃低精度吞吐本来走不通，力气只能放在访存与并行上；③ 生产替代本来就不成立（没有调度器 / 多卡 / 服务化）。另外我要的**分页布局**与**并行切分控制面**，TRT 里对应的是 `IKVCacheUpdateLayer`（只有`kLINEAR`）——不是不能做，是不给我要的形状。

**"多模态"体现在哪？** 框架统一承载 CV 与 LLM 两条链路（同一套 `Engine` 封装、同一套 profile 规则、`BaseTokenizer` 抽象 + 模型注册表）；真正的多模态融合是**未触发的开放项**。坑：夸成"能跑多模态模型"。

**这个项目最大的不足是什么？** 不装：① 绝对性能与生态远不如 TRT-LLM；② LLM 侧只支持 GPT-2 （LoRA / beam search / CUDA Graph / 投机解码都没做）；③ `SentencePieceTokenizer` 零用例、零调用方、未验证；

**和 TRT-LLM 的能力对照**（一句话版）：

已实现——分页 KV / block manager、generation 阶段注意力（MHA/GQA/MQA + 多 block 归约）、fused RMSNorm / RoPE、prefill·decode 分离、采样内核、构建指纹、最小连续批（含打包与分块）；

未做——TP/PP/EP、CUDA Graph、投机解码、beam search、LoRA、INT4 / FP8 / KV cache 量化、服务化。

#### X2 性能方法论（这块最容易拉开差距）

- **"提升 19%"为什么不能写进结论？** 因为绝对差只有 0.17 ms，**低于本机判别下限 0.6 ms**，  测量分辨率不够（本机一个平凡 `greedy` kernel 单发就要 34~156 µs）。纪律：**先声明本次口径的判别  下限（±400~600 µs）再下结论**，低于下限只能写"无显著差异"，**不得按百分比自动接受**  （`PROGRESS.md` §5.13b、`TS-037` / `TS-038`）。

- **怎么把噪声压到这个量级？** 同 session、同二进制、逐轮**交替（ABBA）**、报中位数与四分位；  跨 session 的数字不可比。

- **为什么"改动前后"必须用同一次构建的引擎？** TRT 的 tactic 是 **timing-based**：同一网络重建后引擎体积实测从 `54,196,084` 变到 `52,357,812`。拿两份不同构建的引擎比，比的是 tactic 抽签。

- **split-K 的收益怎么量的？** 同一二进制里 A/B（`SetPagedAttentionNumSplitsOverride` 取负 = 强制单趟），  按**上下文长度扫描**看斜率：降 **88.85%**（kernel）/ **88.20%**（端到端），`max_rel 1.5e-06`，  `ctx = 1024` 每层每步 **0.919 → 0.126 ms**。为什么用斜率而不是单点？单点被固定开销主导。

- **为什么说"负结果也有用"？** 因为它可复现、且被判别实验拆成了两项（固定成本 + 工作量差），指向一个可检验的假设；不可复现的"提升"才是没用的。

#### X3 工程流程与协作（含"用 Codex 怎么开发"）

- **开发流程是什么？** 按 P0–P9 阶段链走（Requirement → Analysis → Design → Review → 实现 → Test →  基准 → 收口 → 面试笔记），每个条目一套 artifact 落 `docs/dev/REQ-NNN-*/`；Gate-A（设计是否被接受）  与 Gate-B（性能结论是否成立）是两个独立关口。入口 `AGENTS.md` §5，流程权威是技能  `trt-inference-engineering`。

- **用 AI 写这个项目怎么保证质量？** 三条自律：① **计划对账**（动手前先确认：有没有计划文档 /

  是否一致 / 偏差怎么处理 / 反向矛盾也要查 / 来源缺失就停）；② **批准目标 ≠ 批准手段**，

  删除或覆盖要单独确认；③ **不许把未验证写成通过**。最硬的例子：`TS-051` 的静态对账逐行读代码

  抓出 6 处问题；"探针改名后忘了进指纹"这类问题被写成规则（`TS-040`）。

#### X4 诚实边界（问到"你还会什么 / 为什么不做 X"时用）

- **没有 Tensor Core 的影响**：低精度吞吐那条收益曲线本来拿不到，所以优化重心放在**访存与并行**  （split-K、分页布局、设备侧采样、减少 H2D/D2H）；这也是"FP16 修复的目标是**默认精度可用**、而不是更快"的原因。

### 1.4 亮点排序（按"面试加分"）

| 亮点                                   | 为什么算亮点（**判据**）                                                                                  | 在哪段   |
| ------------------------------------ | ----------------------------------------------------------------------------------------------- | ----- |
| PagedAttention + 长上下文 split-K        | 有实测收益（斜率降 **88.85% / 88.20%**）**且**与单趟实现逐位一致；难点在约束：宿主拿不到上下文长度、归约必须保序、workspace 必须按 `desc.max` 报 | S7、X2 |
| 自回归循环内零 H2D/D2H                      | 不是"少拷贝几次"，而是三处机制配合，并且**主动交代代价**（不能循环内早停 EOS）                                                    | S7    |
| 最小连续批 + 性能方法论（两个 Negative）           | 结论是**负的但可复现**、被判别实验拆成两项；体现"先声明判别下限再下结论"                                                         | S9、X2 |
### 1.5 只背这 10 组数字

| #   | 数字                                                                                                   | 出处                                             |
| --- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| 1   | GPT-2 FP32 贪心 **8/8 token 与 HF 一致**；FP16 修复后同样 **8/8**                                               | `PROGRESS.md` §3.0a / §5.11                    |
| 2   | GPT-2 prefill logits：`max_abs 9.92e-05`、**相对 `9.19e-07`**、cosine 1.0                                 | 同上                                             |
| 3   | ResNet18 FP32 对 torchvision `9.5e-06`；原生 vs ONNX `1.07e-06`                                          | `PROGRESS.md` §3.0c / §3.0d                    |
| 4   | ResNet18 INT8：FP32 余量子集一致率 **12/12**（整体 38.3%）                                                       | `PROGRESS.md` §3.0j、`REQ-011`                  |
| 5   | split-K：斜率降 **88.85%**（kernel）/ **88.20%**（端到端）；`max_rel 1.5e-06`；ctx=1024 每层每步 **0.919 → 0.126 ms** | `PROGRESS.md` §3.0i、`REQ-014`                  |
| 6   | Top-P 采样：配对净收益 **12.6 / 22.4 / 15.7 / 17.6×**                                                        | `REQ-012`、`PROGRESS.md` §3.0h                  |
| 7   | packed 在长度不齐下**慢 11.79%**（28.441 → 31.793 ms）；判别实验收到 **+5.88%**                                      | `REQ-016/benchmark.md`（`Negative`）             |
| 8   | LLM 的 INT8：prefill 中/长档 **+10.2% / +4.1~4.4%**，decode 无显著差异，体积两引擎合计 **-6.67%**（decode 引擎反而 +7.7%）    | `REQ-017/benchmark.md`（`Negative`）             |
| 9   | 规模：`src`+`include` ≈ 13.9k 行、**351** 条 ctest、**61** 条排查记录                                            | §0.2（实测）                                       |
| 10  | 本机性能判别下限 **±400~600 µs**；FP16 的 `exp` 上界 `ln(65504) ≈ 11.09`                                         | `PROGRESS.md` §5.13b；`TROUBLESHOOTING.md` 18.3 |

固定项：TensorRT 10.15.1、CUDA 12.6.85、`sm_75`、C++17。

## 2. 代码地图（被问"在哪实现的"时用）

| 模块 | 文件 | 关键符号 / 常量 |

|---|---|---|

| 引擎封装 | `core/engine.cpp` | `Engine`、`SetOptimizationProfile`、`GetProfileDims`、`Path()` |

| 建图入口 | `core/builder.cpp` | `EngineBuilder`、`BuildFromConfig`、`BuildFromOnnx`、`kEngineGraphVersion`、`kPackedPrefillGraphVersion`、`kDiagnosticProbeVersion` |

| 缓存指纹 | `core/engine_cache.cpp` | `EngineFingerprintInputs`、`ComputeEngineFingerprint`、`ReadEngineSidecarField` |

| 模型注册 / 配置 | `core/model_registry.cpp`、`model_config.cpp` | `ModelRegistry`、`ModelConfig::Load` |

| GPT-2 建图 | `core/gpt2_model_builder.cpp` | LN `setComputePrecision`、prefill 手搭 attention、`AddTanhGeluSubgraph`、int8 常量 + DQ |

| ResNet18 建图 | `core/resnet18_model_builder.cpp` | 折 BN 后的 Conv、Q/DQ |

| LLM 运行时 | `core/llm_runner.cpp` | `GenerateBatch`、`RunScheduler`、`RunPackedMixedStep`、`ReserveBuffers`、`SampleBatch`、`SchedulerStats` |

| 设备侧小工具 | `core/llm_runner_kernel.cu` | `LaunchFillPositionIds` |

| KV Cache | `kv_cache/paged_kv_cache.cpp`、`block_allocator.cpp` | `AllocateSequence`、`WritePrefillKV`、`AppendDecodeKV`、`AppendDecodeStep`、`UploadMetadata` |

| KV 写入 kernel | `kv_cache/paged_kv_cache_kernels.cu` | `WriteKVKernel<SrcT,DstT>`、`AdvanceContextLensKernel` |

| PagedAttention | `plugins/paged_attention_plugin.cu` | `PagedAttentionDecodeKernel`、`...SplitKernel`、`...MergeKernel` |

| split 纯逻辑 | `plugins/paged_attention_split.hpp` | `PagedAttentionResolveSplits`、`kPagedAttentionMaxSplits = 8`、`kPagedAttentionTargetChunk = 128` |

| PackedAttention | `plugins/packed_attention_plugin.cu` | `MiniTrtLlmPackedAttention`、`kPackedAttentionMaxContextSeqLen = 1024` |

| RoPE / RMSNorm | `plugins/rope_*`、`rmsnorm_*` | half-split 约定、`rotary_dim` / `base` |

| 采样器 | `sampler/sampler_kernels.cu`、`nucleus_cutoff.hpp` | `GreedyKernel`、`FastTopKSampleKernel`、`TopPParallelSampleKernel`、`FindCrossingSegment` |

| CV 运行时 | `core/cv_runner.cpp` | `NormalizePixelsToNchw`、`CVRunner::Infer` |

| 权重 / I/O | `utils/safetensors_loader.cpp`、`weight_loader.cpp`、`json.hpp` | dtype 转换与缓存、递归下降 JSON |

| 测试基建 | `tests/test_gpu_guard.hpp`、`tools/check_skips.py`、`tools/check_docs_index.py` | `MINI_TRT_SKIP_IF_NO_CUDA`、`MINI_TRT_REQUIRE_GPU/ASSETS` |

---

## 3. 一分钟速查（面试中快速回忆）

| 问 | 一句话答 |

|---|---|

| 为什么不直接用 TRT-LLM | 目标是搞清楚机制；本机无 Tensor Core、调库吃不到低精度收益；生产替代不成立 |

| TRT 有没有 attention / KV cache 设施 | 有（`IAttention` / `IKVCacheUpdateLayer`，KV 仅 `kLINEAR`）；我们要的是分页布局与并行控制面 |

| 为什么双引擎 | prefill 手搭子图 + 导出每层 K/V，decode 走分页插件，**I/O 约定不同** |

| 引擎什么时候重建 | 指纹不一致或缺失就重建（含 profile 区间 / 源文件身份 / 开关 / 版本 / 手工图版本） |

| 指纹看不见什么 | 建图代码与插件源码——所以改图必须手工 bump 图版本 |

| 循环里真的没有同步拷贝吗 | 是：位置由设备端填、长度在设备端推进、采样结果留在显存 |

| EOS 为什么不早停 | 与"循环内零同步"冲突；代价是多算几步，语义仍正确 |

| 精度为什么不能按配置假定 | 弱类型下由 TRT 决定；假定会让 FP16 越界写而不报错 |

| 为什么每层一对 cache | 共用一张时每层都读第 0 段，12 层完全失真 |

| 长度为什么整步只推进一次 | 挂在 per-layer 接口上会被推进 `n_layer` 倍 |

| split-K 为什么不能原子加 | 不保序 → 不可复现；判据是"与单趟实现完全一致" |

| 分片数为什么是 8 | 12 blocks vs 24 SM 一半闲置；8 让每层 96 blocks，再加被归并开销吃掉 |

| 空片怎么办 | 写哨兵 `m = -inf, l = 0`，且 grid 恒为上限 |

| 采样为什么全设备指针 | 循环内零 H2D/D2H；传 host 数组会非法访存或读垃圾随机流 |

| 随机流为什么不带行号 | 判据是"同一请求在任何批位置逐位相同" |

| top-k 快速路径更快吗 | 不，慢 6~9 倍，已撤出生产，只留 A/B 入口 |

| 连续批的硬判据 | 批量 == 逐条单跑，逐位相同（管住 KV 落点、位置、随机流） |

| 长度不齐怎么办 | 默认右填充 + `padding_bias`；opt-in packed 只算真实 token |

| packed 更省吗 | 实测**慢 11.79%**（真实 GPT-2 192+64）；判别实验显示是固定成本尚未抵消 |

| `chunk_limit` 从哪来 | 显式声明的 `max_prefill_seq_len`（五条交叉校验），不从 profile 反推 |

| `n_positions` 从哪来 | 引擎 sidecar 读回；缺则拒绝启动 |

| INT8 判据为什么不用绝对误差 | 这批图 FP32 自身摇摆；改用余量子集一致率 ≥ 90%（实测 12/12） |

| 怎么证明在跑 INT8 | `IEngineInspector` + `kDETAILED` 读逐层 `Format/Datatype: Int8` |

| LLM 的 INT8 有收益吗 | 没有：不产生 INT8 计算（0 层带 Int8），prefill 慢 10.2% / 4.1% |

| FP16 的 NaN 怎么回事 | `c_fc` 的 GEMM+GELU 融合 epilogue 在 FP16 下 `exp` 溢出 ⇒ `inf/inf`；判据是逐层一一对应 |

| FP16 怎么修的 | FP16 建图下 GELU 展开 + tanh 自变量限幅 ±5；两次失败尝试说明"精度提示 ≠ 结构约束" |

| 性能结论凭什么可信 | 同 session、同二进制、ABBA、先声明判别下限；低于下限只写"无显著差异" |

| 为什么必须同一次构建比 | TRT 的 tactic 是 timing-based，同网络重建体积都会变 |

| 遇到真的过不去的用例 | 三种动作：继续查 / 证明期望值错（要独立依据）/ 保持红；禁止放宽阈值 |

| 沙箱能验什么 | 编译、host 逻辑与约定、脚本护栏；GPU 用例显式跳过，真机跳过即失败 |

| 最大的不足 | 绝对性能与生态、模型覆盖窄、SentencePiece 未验证、还剩 1 条按设计红 |

---

## 附录 A｜出处索引（面试前抽查用）

| 你想证明的 | 去哪看 |

|---|---|

| 现状 / 基线 / 版本 / 产物尺寸 | `PROGRESS.md`（唯一出处） |

| 某条结论怎么查出来的 | `TROUBLESHOOTING.md` 的 `TS-001`~`TS-061`（FP16 的定位是 `TS-018` 的 18.3） |

| 没做什么 / 触发条件 | `future_iterations.md`（`OI-*`） |

| 某个 feature 的完整 artifact | `docs/dev/REQ-NNN-*/`（入口 `STATE.md`） |

| 条目编号与状态快照 | `docs/dev/INDEX.md` §1 |

| 权限与开工纪律 | `AGENTS.md` §0 / §5 |

**最常被追问的 8 个 TS**：`TS-002`（`supportsFormatCombination` 越界读）、`TS-004`（部分旋转未写尾部）、

`TS-015`（多层共用 cache）、`TS-016`（按层推进长度）、`TS-018`（FP16：非法访存 → NaN → 18.3 定位修复）、

`TS-019`（诊断输出打破 I/O 约定）、`TS-042`（同一 kernel 两套量法差 2.5 倍）、`TS-046`（量 A 裁 B）。

## 附录 B｜与 `interview_summary.md` 的已知差异（**那份需要更新**）

| 项 | `interview_summary.md` 写的 | 现状 |

|---|---|---|

| 用例数 / 红 | 268 条 | **351 条 / 1 红**（2026-10-08 第 8 轮） |

| 代码规模 | 13.8k 行（2026-10-05） | **13.9k 行**（2026-10-07 实测） |

| 排查记录 | 56 条 | **61** 条 |

| 批量的状态 | "已落码未验证、默认单请求" | **REQ-016 已交付**（真机全绿）；packed 结论 = `Negative` |

| LLM 量化 | "LLM 量化未做" | **REQ-017 已交付**（weight-only INT8，opt-in，收益不成立） |

| FP16 | 已同步（2026-10-08） | 一致（`TROUBLESHOOTING.md` + 18.3） |

> 两份都属**材料类**、不参与 SSOT；数字冲突时以 `PROGRESS.md` 为准。
