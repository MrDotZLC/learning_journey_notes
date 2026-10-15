## 一、先亮底牌：硬件约束表

|资源|实测值|对 Kernel 的直接含义|
|---|---|---|
|SM 数|24|Grid 至少约 24 个 Block 才能覆盖全部 SM；当 N=1024、Grid=32 时，分两波执行，第二波只有 8 个 Block，理想化波次利用率为 67%|
|线程/SM|1024（32 Warps）|达到线程数上限时无法再驻留更多线程；Block=1024 线程时，一个 Block 就可能占满 SM 的线程容量，降低隐藏延迟的能力|
|寄存器/SM|65536 个 32-bit 寄存器|若 2048 个线程全部驻留，则平均预算为 32 regs/thread；若只有 1024 个线程驻留，则平均预算为 64 regs/thread。实际 Occupancy 还受寄存器分配粒度等因素影响|
|Shared Memory/SM|64 KB（Block 默认上限 48 KB）|限制 Tile 大小和每个 SM 可同时驻留的 Block 数；超过默认单 Block 限制时，需要检查是否支持并配置动态 Shared Memory 上限|
|Shared Memory 带宽|128 B/cycle/SM，32 Banks × 4 B|理想情况下，一个 Warp 的 32 个线程各访问一个不同 Bank、每线程读取 4 B，可在一个 Bank 服务周期内完成 128 B 访问；Bank Conflict 会降低有效吞吐|
|FP32 FFMA|64 lane/cycle/SM|理论吞吐约为 2 Warp-FFMA/cycle；按 1.59 GHz 计算约 4.88 TFLOPS，按 1.77 GHz 计算约 5.44 TFLOPS|
|MUFU（exp2）|16 lane/cycle/SM|理论吞吐约为 FP32 FFMA 的 1/4；但 `exp` 的实际成本还取决于指令映射、延迟及其他流水线行为|
|SHFL / MIO|32 lane/cycle/SM|Shuffle 吞吐约为 FFMA 的一半；Warp Reduction 等操作可能受到 Shuffle 吞吐和依赖链限制|
|L2 Cache|1.5 MB|当 N=1024、d=64，且 Q/K/V 均为 FP16 时，三者总计约 384 KiB，可整体驻留 L2；实际命中率仍受其他访问和 Cache 竞争影响|
|DRAM 带宽|288 GB/s|按 4.88 TFLOPS 计算，Roofline 拐点约为 \(4.88\times10^{12}/288\times10^9\approx16.9\) FLOP/B，即约 17 FLOP/B；高于该算术强度时才可能进入计算受限区|

  关键的一条：Turing 的 SM 是 64 FP32 + 64 INT32，不是 Ampere 的 128 FP32。所以 FP32 峰值只有 4.88 TFLOPS，这让"指令效
  率"成为唯一的战场——v2 的 0.22 TFLOPS 只吃掉了 4.5% 的 FP32 峰值，说明瓶颈根本不在 FMA 上。

## 二、硬件 → 参数：映射表

| 参数                                    | 决定它的硬件约束                                                 | 反推算式                                                       |
| ------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------- |
| `block = 1024`                        | 线程/SM = 1024，`Br` 个 Warp                                 | \(B_r \le 1024/32=32\)，取满                                  |
| `Br = 32`                             | 同上，每行分配一个 Warp                                           | \(32\text{ warps}\rightarrow32\text{ 行}\)                  |
| `Bc = 32`                             | Shared Memory/SM = 64 KB，默认单 Block 上限 48 KB              | \((B_r+2B_c)D\times2+B_rB_c\times4\le\text{预算}\)           |
| 每 Lane 2 个 `o` 寄存器                    | 寄存器/SM = 65536 个 32-bit 寄存器，1024 线程时平均预算为 64 regs/thread | 每个线程只能分配有限寄存器；若每 Lane 持有完整的 `o[64]`，可能导致 Register Spilling |
| `smem_S` 使用 Shared Memory 而非寄存器       | 寄存器预算有限                                                  | 用约 4 KB Shared Memory 存储中间状态，以减少寄存器压力                      |
| 每 Lane 覆盖 `d = lane_id, lane_id + 32` | `warpSize = 32`                                          | \(D/\text{warpSize}=64/32=2\)                              |
| 两次 `__syncthreads()`                  | 跨 Warp 的 Shared Memory 生产者—消费者同步                         | 第一次通常必要；第二次只有在确认不存在跨 Warp 依赖且访问范围安全时，才可考虑降为 `__syncwarp()` |
| 不使用 MMA 的 SIMT 点积                     | SHFL 吞吐约为 FP32 FFMA 的一半                                  | 需要核算 Warp Reduction 的 Shuffle 指令数量、依赖链及其相对 FFMA 的吞吐开销      |

## 三、三条完整的反推链（面试时按这个顺序讲）

  ① block / Br：不是"我选了 32"，是硬件把我推到了 32。

  Turing 线程/SM 上限 = 1024
  → 一个 block 最多 32 warps
  → 一行 Q 一个 warp（为了让 softmax 规约留在 warp 内）
  → Br = 32，block = 1024

  但反推到这里必须再走一步，这才是加分的地方：block=1024 在 Turing 上等于把整个 SM 交给一个 block（1024 就是上限），于是
  regs/thread 硬顶 64、没有第二个 block 兜底延迟、barrier 抖动无人填。所以硬件的正确指向其实是 4~8 warps/block（128~256
  线程）：此时 regs/thread 额度升到 256（65536/256），float o[D/32] 这种累加器阵完全放得下，同时每 SM 能驻留 2~4 个
  block 互相填 stall（maxBlocksPerSM=16，够用）。当初把 o 砍成 o0/o1 是为了迁就 1024 线程的寄存器天花板，而真正的解法是
  缩小 block。

  ② Bc：由 64 KB smem 反推，不是拍脑袋的 32。

  smem/SM = 64 KB，单 block 默认上限 48 KB
  预留 2 个 block/SM 的余量 → 每 block ≤ 32 KB
  D=64: (Br+2Bc)·64·2 + Br·Bc·4
    Br=Bc=32 → 12288 + 4096 = 16 KB  ✓ 余量充足
    Br=Bc=64 → 24576 + 16384 = 40 KB  ✗ 逼近 48 KB，每 SM 只能 1 个 block

  所以在这张卡上 Bc 的上界是"48 KB 减掉 Q tile 和 S 缓存"算出来的，Bc=64 就已经把 double buffering 的空间吃光了。同时 Bc
  还受 smem 带宽约束：LDS 吞吐 128 B/cycle/SM，只有 FFMA 吞吐的一半，每 lane 只用 2 B 去载入 half，会让 LDS 先于 FFMA 饱
  和——这就是向量化 half2/float4 的硬依据，不是"习惯写法"。

  ③ 每 lane 该管几个维度：从 MUFU 和 SHFL 的吞吐反推。

  有效计算量：每 (行, tile) 4096 MAC = 8192 FLOP
  FFMA 管：   4096/64 = 64 cycle/SM 的 FFMA 时间
  shuffle 管：每 score 5 次 shfl + 1 次广播 = 6 条 MIO 指令
              32 score × 6 = 192 条 warp-指令 ÷ 1 条/cycle = 192 cycle
  exp 管：    实际执行 32×32 = 1024 次 expf（有效只需 32 次）
              1024 ÷ 16 lane/cycle = 64 cycle，还只是"有效"口径，
              算上前置的缩放/转换指令，softmax 段实际指令数 ≈ 380 条/lane

  三条一摆就清楚了：FFMA 需要的周期最少（64），MIO 要 192，softmax 段要 380 条指令。所以这个 kernel 的瓶颈完全不在算术
  上，而在 shuffle 链和 32× 冗余的 exp 上。反推结论有两条，而且方向一致：

  • 让每个 lane 管更多维度：D/32 从 2 提到 4 或 8，shuffle 开销不变而有效 FFMA 翻倍，MIO:FFMA 比直接减半；
  • 消灭 exp 的 32× 冗余：Bc == 32 == warpSize 时让 lane j 持 S_j，exp 从 1024 次降到 32 次——按 MUFU 只有 FFMA 1/4 吞吐
    算，这一项就值回好几个百分点。

  ④ 顺带用硬件反推"为什么 v2 能赢 v1"。

  v1（block-per-row）：grid 级 K/V 读放大 = N²D
    N=1024 → 每 block 读 256 KB × 1024 block = 256 MB
    256 MB ÷ 288 GB/s ≈ 0.89 ms  ← 实测 2.38 ms，访存主导
  v2（KV tiling）：32 个 block × 256 KB = 8 MB，全部命中 1.5 MB L2
    → 有效 DRAM 流量 ≈ 384 KB，0.0013 ms  ← 完全不是瓶颈

  一句话讲清机理：v1 是访存受限，v2 把问题从"搬数据"转成了"发指令"，所以 v2 的加速比随 N 增长（1024 上 1.97×），而后续优
  化的方向也就顺理成章地从访存转到了指令效率和并发度。另外 attention 的算术强度是 N/2 = 512 FLOP/B，远高于拐点 17，所
  以"别去优化带宽、去优化指令"这句话是有硬件依据的。

## 四、还能用硬件反推一处你现在没考虑到的浪费

  grid = 32（N=1024, Br=32）落在 24 个 SM 上：波次利用率 = 32/(2×24) = 67%，第二波只有 8 个 SM 在干活。从"铺满 24 SM 且
  波次整齐"反推，应该缩小 Br 把 grid 做大：Br=16 → grid=64 → 64/(3×24=72) = 89%，且 512 线程/block 时每 SM 能驻留 2 个
  block。这里有个明确的 trade-off 要主动说出来：Br 缩小会让 Bc（当前与 Br 耦合）跟着缩小、单 tile 的工作量变少、摊销变差
  ——所以正确做法不是二选一，而是先解耦 Br/Bc/warp 数，再按"波次整除 + regs/smem 预算"两端夹逼。这个回答能同时展示"我知道
  硬件在限制什么"和"我知道我的实现哪里被限制住了"。

## 五、面试话术模板

  被问"你这个 32 是怎么定的"，不要答"经验值"，按这个结构答：

  > 先看硬件：sm_75 的 SM 是 1024 线程、65536 寄存器、64 KB 共享内存、64 个 FP32 lane。上限那一头，block 最多 32 warps，
  > 我又用"一行 Q 一个 warp"把 softmax 规约关进 warp 内，所以 Br 顶到 32。下限那一头，smem 公式 (Br+2Bc)·D·2 + Br·Bc·4
  > 在 48 KB/block 的额度下，Bc=64 就已经要 40 KB，所以 Bc 只能取 32。中间还有个隐性的寄存器约束：1024 线程把 regs/
  > thread 压到 64，所以我只敢给输出留两个寄存器，这也是 head_dim 被写死成 64 的原因。如果重做，我会把 block 降到 4~8
  > warps：寄存器额度从 64 升到 256，累加器阵和第二个 block 的并发度都能拿到，那时再按 smem 和波次整除重新解 Br/Bc。