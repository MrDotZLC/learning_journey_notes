## 1. 基础概念
### 1.1 Warp
- **定义**：GPU 中最小调度单元，由 32 条线程组成
- **逻辑执行**：Warp 内线程共享指令执行流（SIMT 模式）
- **物理执行**：每条线程占用一个 **lane**（通道），物理并行
- **特点**：
    - Warp 内线程共享 PC 和 SIMT Stack 上下文
    - Divergence 发生时通过 SIMT Stack 管理不同路径
### 1.2 Warp Scheduler (WS)
- **位置**：SM 内的硬件单元
- **功能**：
    1. 从活跃 warp 中选择 warp
    2. 发射 warp 指令到执行单元（ALU/FP/MEM）
    3. 管理 warp 发射顺序
- **数量**：
    - Ampere/Ada：4
    - Hopper：6
- **发射能力**：每个 scheduler 每周期可发射 1 个 warp 指令
### 1.3 Program Counter 相关变量
GPU divergence 中涉及 **三个 PC**：

|变量|全称|存储位置|管理者|功能|
|---|---|---|---|---|
|**PC**|Program Counter|Warp 内寄存器（硬件）|Warp Scheduler / SM 控制逻辑|指向当前 warp 要执行的指令地址|
|**NPC**|Next Program Counter|Warp 内寄存器（硬件）|Warp Scheduler / SM 控制逻辑|指示下一条顺序指令地址，支持分支跳转与顺序执行|
|**RPC**|Reconvergence Program Counter|SIMT Stack Entry|SIMT Stack 管理逻辑|divergence 合并点指令地址，warp 执行完当前分支后跳转到此汇合点|

**管理说明**：
- **PC / NPC**：由 warp 硬件寄存器存储，warp scheduler 读取并更新
- **RPC**：存储在 SIMT Stack entry 中，由 warp scheduler 在 pop stack 时读取，确保正确回到汇合点
- **更新逻辑**：
    - Divergence 发生时，当前路径 PC 指向 active path
    - NPC 指向下一条顺序指令
    - RPC 保存未执行路径的汇合点
### 1.4 SIMT Stack
- **功能**：管理 warp divergence
- **Entry 内容**：
    - PC：未执行路径的起始指令
    - Active Mask：未执行路径的活跃线程 mask
    - RPC：路径汇合点
- **操作**：
    - Warp 遇到分支 → active mask 拆分 → push 未执行路径 entry
    - Warp 执行当前路径 → 完成后 pop stack → 跳转 RPC
### 1.5 Active Threads / Lane
- **Active Threads**：当前 warp 内执行指令的线程
- **Lane**：warp 内线程物理执行通道
- Divergence 时：
    - Active threads 执行指令
    - Inactive lane idle（线程等待）
- Lane 利用率 = active threads / 32
## 2. Divergence处理流程（详细）
假设 warp 内有 `if (cond) { A } else { B }`：
### Step 0：初始状态
- PC → `if (cond)`
- NPC → `if(cond)+1`
- Active Mask = 全部 1（32 threads）
- SIMT Stack = 空
### Step 1：条件计算
- 每条线程计算 cond → 0/1
- Active Mask 拆分：
    - A 路径线程 mask = cond=1
    - B 路径线程 mask = cond=0
### Step 2：更新 SIMT Stack
- 为未执行路径（B） push entry：
    - PC = B 起始地址
    - Active Mask = B 线程 mask
    - RPC = 汇合点 addr_join
### Step 3：更新 Warp PC 和 Active Mask
- Warp PC → 当前执行路径 A
- Warp NPC → 下一条顺序指令
- Active Mask → A 线程
- Scheduler 发射 warp 指令
- Inactive lane idle
### Step 4：执行路径
- 仅 active threads 执行 A
- 线程局部变量仅更新 active threads
- 非 active threads 保留状态
### Step 5：完成当前路径 → Reconvergence
- Pop SIMT Stack → 得到 B entry
- Warp PC → entry.PC（B 起始地址）
- Active Mask → B 线程
- Scheduler 发射 warp 执行 B
### Step 6：完成所有路径 → 汇合
- B 执行完后：
    - Warp PC → RPC（addr_join）
    - Active Mask = 全 warp 或后续指令决定
- Divergence 结束
## 3. Ampere/Ada vs Hopper 架构差异（Divergence）

|特性|Ampere/Ada|Hopper|
|---|---|---|
|Warp Scheduler|4|6|
|Divergence处理|每 warp 保持独立路径|Dynamic Warp Recomposition (DWR) 重组 active threads|
|Active Mask|warp 内分支线程独立|合并同路径线程形成 full warp|
|SIMT Stack|多 entry → 多 warp 发射|entry 少，lane 利用率高|
|发射次数|多次|少次 → 发射开销降低|
|Lane 利用率|< 100%，idle lane 多|≈ 100%|

## 4. Hopper 详细优化
### 4.1 Dynamic Warp Recomposition (DWR)
- 将多个 warp 的相同路径 active threads 重新组合成 **full warp**
- 新 warp 的 Active Mask = 合并后线程
- Idle lane 减少 → ALU 利用率接近 100%
- 示例：

| Warp | A   | B   |
| ---- | --- | --- |
| W0   | 8   | 24  |
| W1   | 10  | 22  |
| W2   | 6   | 26  |
| W3   | 8   | 24  |

- 路径 A：32 threads → 1 warp 发射
- 路径 B：96 threads → 3 warp 发射
- 发射次数 = 4（相比 Ampere/Ada 的 8 次）
### 4.2 SIMT Stack 优化
- 合并相同路径线程 entry → stack push/pop 次数减少
- RPC 保持一致，保证 divergence 汇合点正确
- stack depth 降低 → stack 操作延迟减少
```
// Ampere/Ada（优化前）：
// 每个 warp divergence 独立 entry
Entry 1: { mask = warp1_active_threads, pc = path_start, joinPC = path_end }
Entry 2: { mask = warp2_active_threads, pc = path_start, joinPC = path_end }
Entry 3: { mask = warp3_active_threads, pc = path_start, joinPC = path_end }
...

// Hopper（优化后，DWR 合并）:
// 多 warp 相同路径线程合并为 full warp
Entry 1: { mask = merged_active_threads_pathA, pc = pathA_start, joinPC = path_end }
Entry 2: { mask = merged_active_threads_pathB, pc = pathB_start, joinPC = path_end }
...

```

### 4.3 Warp Scheduler 优化
- Hopper SM 拥有 6 个 scheduler
- 可以同时发射更多 warp
- 配合 DWR 提高 warp 发射并行能力
### 4.4 总体效果
- Lane 利用率 ≈ 100%
- Divergence 执行总时间降低
- 发射开销显著减少
## 5. 分支处理示例：4 warp

|Warp|路径 A|路径 B|
|---|---|---|
|W0|8|24|
|W1|10|22|
|W2|6|26|
|W3|8|24|

- 总线程：
    - A = 32
    - B = 96
### 时间开销
#### Ampere/Ada
- 路径 A：4 warp 发射 + 执行 = 8 单位
- 路径 B：4 warp 发射 + 执行 = 8 单位
- 总时间 = 16 单位
#### Hopper
- 路径 A：1 warp 发射 + 执行 = 5 单位
- 路径 B：3 warp 发射 + 执行 = 7 单位
- 总时间 = 12 单位

> Hopper 节省发射开销，并提高 lane 利用率
## 6. 核心结论
1. Divergence 核心变量：
    - **PC**：当前执行指令
    - **NPC**：下一条顺序指令
    - **RPC**：汇合点
    - **Active Mask**：当前执行线程
    - **SIMT Stack**：管理未执行路径和 reconvergence
2. Ampere/Ada：
    - 多 warp 独立执行
    - lane idle → 发射开销高
3. Hopper：
    - DWR 重组 active threads → lane 利用率高
    - 发射次数少 → 总执行时间短
4. 线程变量仅在 active mask 为 1 时更新
5. Reconvergence PC 保证 warp divergence 后正确汇合
6. PC/NPC 由 warp 内寄存器存储并由 warp scheduler 管理；RPC 存储在 SIMT Stack entry，由 stack 管理
