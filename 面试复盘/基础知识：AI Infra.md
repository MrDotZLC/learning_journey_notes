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
通用工具：LakeFS
具身智能常用：LanceDB 机器人数据湖仓（Episode/Trajectory  diff、质量检查、适配MCAP/ROS 等特殊格式等等）

### 模型版本管理
通用工具：MLFlow、Kubeflow

## MLOps 工具
### Kubeflow
> 构建在 Kubernetes 之上的机器学习（Machine Learning）工作流平台，用于管理模型训练、调优、部署和生命周期管理。

| 组件                 | 作用      |
| ------------------ | ------- |
| Kubeflow Notebook  | 提供开发环境  |
| Kubeflow Pipelines | 机器学习流水线 |
| Training Operator  | 分布式训练   |
| Katib              | 超参数搜索   |
| KServe             | 模型部署    |
| Central Dashboard  | 统一入口    |

1. Kubeflow Notebook 本质是 Kubernetes 中运行的 Jupyter Pod。
2. Kubeflow Pipeline 将流程定义成 DAG，每个节点对应 Kubernetes Pod，实现全流程的自动化执行。
3. Kubeflow Training Operator 就是 Kubernetes 上的“分布式训练调度器”，通过一个通用的控制器来管理不同框架的作业 Custom Resource (CR)，CR 是与 Node 同级别的自定义资源类型，可表示“训练任务”。
4. Katib 是解决自动化机器学习（AutoML） 问题的组件，主要功能：
   **超参数调优**：寻找模型训练的最佳参数组合
   **早停**：发现某组参数明显效果差时，提前终止训练，节省算力
   
5. KServe 是负责**模型推理服务部署**的组件，负责把模型部署成在线服务。封装了自动扩缩容、流量路由、健康检查、版本管理、GPU 调度等。
6. 大模型训练任务接入 Kubeflow 流程
```
Docker Image
 |
Training Dataset
 |
PyTorchJob(Custom Resource)
 |
GPU Scheduler
 |
NCCL Distributed Training
 |
Checkpoint Storage
 |
Evaluation
```

Kubeflow 是 Kubernetes 上的机器学习平台，主要解决 ML 生命周期管理问题。它通过 Notebook 提供开发环境，通过 Pipeline 编排数据处理、训练、评估流程，通过 Training Operator 管理分布式训练任务，例如 PyTorchJob，通过 KServe 完成模型部署。底层依赖 Kubernetes 的 Pod 调度、GPU Device Plugin、RBAC 和存储系统，实现训练和推理任务的自动化管理。

## ROS

ROS 是机器人领域最主流的机器人软件中间件（Robotics Middleware），是一套运行在 Linux 上的分布式机器人应用框架，提供通信机制、硬件抽象、工具链、调试工具和生态组件。

### ROS 1
Apollo 早期使用的就是 ROS 的通信RPC，通讯风格和 ROS 一模一样，都是模块 Node 解耦、消息驱动、Topic 异步发布/订阅 和 主动请求等特性。

### Action
此外，Action 是 ROS 处理耗时任务、有中间反馈、可取消的异步调用机制。Apollo 没有这个机制，它核心流程是单向流动的，用 topic 足够处理了。

### ROS 2
#### 删除 ROS Master，替换成 DDS
Master 是中心化组件，多机器人部署存在单点故障。使用 DDS 数据分布式服务实现：
	去中心化
	广播自动发现
	单播实时通信
	多机通信
	控制QoS
Apollo 中期自研通讯架构也是因为 Master 单点瓶颈，自行设计了一套类似 DDS 的机制，包括可靠性、历史数据保存、缓存策略、周期约束、消息生命周期。

### Launch 大量节点
统一管理节点生命周期、参数配置、环境变量和启动顺序。

### 节点动态配置
使用 yaml 动态修改 Node 节点。

### TF 坐标变换系统
统一管理机器人各部件之间坐标系关系的核心基础设施
