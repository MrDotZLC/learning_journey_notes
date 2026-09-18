## AI Infra 训练平台全链路

> 机器人数据采集 → 数据管理 → 分布式训练 → 训练任务调度 → 实验管理 → 模型压缩 → 云端/端侧部署

## Linux
### 常用指令
ps、top等
### 网络基础
TCP/IP、TSL/SSL、RPC、IPC、Socket、ping、ip、netstat等
### 权限管理
文件的rwx、软链接 ln -s、chmod、chown、sudo
### GPU
nvidia-msi、ncu、nsys
### 性能分析
pref、top、iostat

## Docker
### 基础
OS级容器，保持环境一致，与主机共享kernel。
namespace：隔离资源
网络模式：host、bridge、none、overlay
卷挂载
layer：每次修改都会生成完整的副本layer，尽量多个操作合并为一行，layer可合并
### GPU
GPU分配：容器内安装该NVIDIA Container Toolkit 和 cuda runtime 包，会自动使用宿主机的驱动。
### 启动工具
多服务部署：使用docker-compose

## Kubernetes

Kubernetes 是一个**容器编排平台**，用于自动管理大量 Container 的部署、调度、扩缩容和故障恢复。
```
			kubectl
			   ↓
		   api server
		       ↓
		----------------
  Scheduler Controller etcd
		       ↓
		      Node 
```
```
			  Node
			   ↓
			kubectl
			   ↓
			  Pod
			   ↓
		   Container
```

1. kubectl 是 k8s 的控制面板和 Node 的代理，调用 api server 查看 Node、查看 pod 、创建查看容器、修改Deployment、滚动重启等。
2. etcd 是 Kubernetes 的分布式 Key-Value 存储，保存整个集群状态（Pod状态、Node信息、Service信息、ConfigMap、Secret）。
3. Scheduler 根据现有资源为 Container 分配 Node。
4. Controller 持续监听实际状态，并控制 pods 达到期望状态。
5. Pods 是 Kubernetes 最小调度单位，可包含多个 Container，只能修改 Pod 的配置。
   Pod 的状态有：Pending、Running、Succeeded、Failed、Unknown。
   重启策略有：Always（失败自动重启）、OnFailure（失败重启）、Never（不重启）

6. Deployment 管理无状态应用。提供副本管理、滚动升级、自动恢复。、
7. ReplicaSet 保证 Pod 数量。
8. StatefulSet 管理有状态实例。
9. Service 是为一组 Pod 提供稳定访问入口和四层负载均衡的抽象。通过 Label Selector 找到 Pod，用 Endpoints 记录后端，靠 kube-proxy 实现转发，用 DNS 提供名字。四种类型分别解决集群内、Node 端口、云负载均衡和外部域名映射的需求。

### 网络

1. Pod IP唯一
2. Pod可以直接通信
3. Node之间通信走 CNI
4. CNI 是实现 Pod 跨 Node 互通的插件
5. 高质量 GPU 通信主要依赖**Multus + SR-IOV/GPUDirect**。
   - Multus 是让 Pod 有多网卡，主网卡走CNI，众多辅网卡跑训练。
   - SR-IOV 是硬件虚拟化技术，把一张物理网卡虚拟成多个 VF，分配给多个 Pod 独占使用，绕过了宿主机的内核网络栈，直接从网卡硬件收发数据。
   - GPUDirect RDMA 允许网卡直接通过 PCIe 总线读写 GPU 的显存，完全跳过 CPU 和系统内存。
   - 支持NCCL。

### GPU

1. 在每个 Node 上安装 NVIDIA 驱动和 NVIDIA Device Plugin。
   然后在 Pod Yaml 中申请 GPU 资源
   Scheduler 为 Pod 分配 Node

2. 通常 Pod 不可共享同一张 GPU，有方案可实现。

## Slurm

> 一个开源的、高可扩展性的集群管理和作业调度系统，专门为 Linux 集群设计。接收用户提交的计算任务，并根据资源需求和优先级，合理地分配计算节点来运行这些任务。

Slurm 作为工作负载管理器，主要有三大功能
1. **资源分配**：在一段时间内，将计算节点（资源）的独占或非独占访问权分配给用户，让他们可以执行工作。
2. **任务管理**：提供一个框架，用于在分配到的节点上启动、执行和监控任务（通常是并行作业）。
3. **仲裁争用**：通过管理待处理作业的队列，来仲裁多个用户对有限资源的竞争。

|       | Slurm  | Kubernetes |
| ----- | ------ | ---------- |
| 定位    | HPC调度  | 云原生调度      |
| 对象    | 计算任务   | 服务/任务      |
| GPU训练 | 非常成熟   | 需要插件       |
| 应用    | 科研训练集群 | 企业平台       |

Slurm 和 k8s  可混合使用。Slurm用于分布式训练，k8s 用于数据预处理、实验管理、Notebook、推理服务、CI（持续集成）/CD（持续部署） 等云原生负载。

## 数据流水线
### 多模态数据时间同步
ROS 有该功能。
1. 时钟同步
   PTP硬件同步，软件精度不够，
2. 时间戳语义统一
3. 时间戳对齐
   - 硬件触发同步
   - 帧对齐：设基准数据帧时间戳 t_c，找[t_c - △t, t_c + △t]内的其他源数据，线性插值到 t_c
   - 高频对齐：设基准数据帧时间戳 t_c，找其他源数据的前后两帧，线性插值或最近帧+补偿
   - 统一到全局时间轴：重采样到固定频率，再按照 基准时间戳 线性插值或最近帧+补偿

### 数据版本管理
