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
### 基础
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

5. Pods 是 Kubernetes 最小调度单位，可包含多个 Container。
