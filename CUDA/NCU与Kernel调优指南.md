# 〇、NCU
 Nsight Compute，NVIDIA研发的GPU核函数级别的性能分析工具。Nsight Compute图形界面工具，缩写为NCU。
1. 运行环境：windows10 + wsl2 + vscode + cmake
2. 启动指令：ncu-ui
3. 修改cmake配置：cu文件所在子CMakeList.txt中，追加
`target_compile_options(${EXE_NAME} PRIVATE -lineinfo)`
![[img_ncu_修改cmake配置.png]]
4. Start Activity 连接对应平台及可执行文件
![[Pasted image 20260102231109.png]]
 ![[Pasted image 20260102231307.png]]
5. 
 
注：在终端执行 ncu-ui 时，可能会出现报错，尽量全部解决掉（正常执行命令后，终端无输出），保证ncu-ui正确启动，避免功能无法使用。
![[Pasted image 20260102230033.png]]


# 一、合并访存

# 二、、Occupancy Calculator

# 三、Warp调度

# 四、Roofline Model

# 五、Shared Memory 与 Bank Conflict