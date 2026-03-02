Step 1 → nsys/ncu 性能分析（Layer Fusion 可视化 + Tensor Core 利用率）
Step 2 → 多 Batch 性能曲线（latency vs throughput 权衡分析）
Step 3 → Workspace 调优实验
Step 4 → INT8 Calibration 用真实数据替换随机数据


# nsys/ncu 性能分析
![](assets/Pasted%20image%2020260302073817.png)
![](assets/Pasted%20image%2020260302073824.png)
![](assets/Pasted%20image%2020260302073829.png)

汇总三个阶段的优化结果：
## 优化前后对比

|阶段|优化前|优化后|改善|
|---|---|---|---|
|H2D|1.49 ms|0.036 ms|**41x**|
|Infer|1.16 ms|~1.1 ms|持平|
|D2H|22.306 ms|0.014 ms|**1593x**|

---

## 当前单次推理时间分解

```
H2D   : 0.036 ms  (3%)
Infer : ~1.1  ms  (90%+)
D2H   : 0.014 ms  (1%)
```

**瓶颈已经回归 GPU 计算本身**，内存传输开销基本消除。这是正确的状态。

---

## 阶段总结：Step 1 完成

你已经掌握的内容：

- NVTX 标注 + nsys 采集推理 timeline
- 识别 Pageable Memory 导致的 H2D/D2H 瓶颈
- Pinned Memory 优化输入输出缓冲区
- 用 SQLite 查询量化优化效果

# 多 Batch 性能曲线

### Throughput（img/s）

|Batch|FP32|FP16|INT8|
|---|---|---|---|
|1|583|813|770|
|2|861|992|1190|
|4|868|1383|1591|
|8|956|1597|1942|
|16|1014|1655|**2070**|

### 关键结论

**① Throughput 随 batch 增大持续提升，但边际收益递减**

```
FP32: batch 1→8 吞吐提升 1.64x，batch 8→16 仅提升 1.06x
INT8: batch 1→8 吞吐提升 2.52x，batch 8→16 仅提升 1.07x
```

说明 batch=8 附近 GPU 已接近饱和，batch=16 收益有限。

**② INT8 在 batch≥2 时超过 FP16，batch=1 时反而略慢**

- batch=1 时 kernel launch overhead 占比高，INT8 优势不明显
- batch≥4 时 INT8 Tensor Core 优势充分发挥

**③ FP16 加速比（vs FP32）**

```
batch=1:  1.40x
batch=8:  1.67x
batch=16: 1.63x
```

**④ INT8 加速比（vs FP32）**

```
batch=1:  1.32x
batch=8:  2.03x
batch=16: 2.04x  ← GTX 1660 Ti INT8 Tensor Core 上限
```

**⑤ Latency vs Throughput 权衡**

|场景|推荐配置|
|---|---|
|实时推理（延迟优先）|INT8 batch=1，latency=1.30ms|
|离线批处理（吞吐优先）|INT8 batch=16，2070 img/s|
|精度敏感|FP16 batch=16，1655 img/s|
# Workspace 调优实验


# INT8 Calibration 用真实数据替换随机数据