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
		
```

