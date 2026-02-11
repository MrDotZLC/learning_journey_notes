1. **确认系统环境**
   系统：Ubuntu24.04(wsl2)
   cuda：12.6
   vscode：
   CMake：3.28.3
2. **下载TensorRT Debian仓库包**
   从 [TensorRT 下载页面](https://developer.nvidia.com/tensorrt)，下载适用于你的 CUDA 版本和作系统的 Debian 仓库包。
   ![](assets/Pasted%20image%2020260210163102.png)
3. **安装仓库包**
	```
   sudo dpkg -i nv-tensorrt-local-repo-ubuntu2404-10.15.1-cuda-12.9_1.0-1_amd64.deb
	```
4. **复制钥匙圈**
	```
   sudo cp /var/nv-tensorrt-local-repo-ubuntu2404-10.15.1-cuda-12.9/nv-tensorrt-local-ABCD58F4-keyring.gpg /usr/share/keyrings/
	```
5. **更新包索引**
	```
sudo apt-get update
	```
6. **安装TensorRT包**
	```
   sudo apt-get install tensorrt
	```
7. **安装python包**
   - 系统Python环境
     ```
	    python3 -m pip install numpy
		sudo apt-get install python3-libnvinfer-dev
     ```
   - 虚拟Python环境
	 1. 去 NVIDIA TensorRT 下载页，下载tar包
        ![](assets/Pasted%20image%2020260210163126.png)
     2. 进入虚拟环境
        ```
        source trt-env/bin/activate
        ```
     3. 解压安装
     ```
	     python -m pip install numpy
	     python -m pip install tensorrt-*-cp3x-none-win_amd64.whl
	     python -m pip install tensorrt_lean-*-cp3x-none-win_amd64.whl
	     python -m pip install tensorrt_dispatch-*-cp3x-none-win_amd64.whl
     ```
     ![](assets/Pasted%20image%2020260210171351.png)
     4. （可选）安装 ONNX Graph Surgeon
        ```
        python -m pip install numpy onnx onnx-graphsurgeon
        ```

2. **验证**
	 1. 验证 TensorRT 软件包   
	    ```
		# 验证 TensorRT 软件包
		dpkg-query -W tensorrt
	    ```
	    ![](assets/Pasted%20image%2020260211144406.png)
	2.  下载 TensorRT OSS
		```
		git clone -b main https://github.com/nvidia/TensorRT TensorRT
		cd TensorRT
		git submodule update --init --recursive
		```
	3. 修改根目录 CMakeLists.txt
	   TensorRT 10.15.1 GA 默认CUDA=13.1、GPU_ARCHS=110。使用vscode的CMake插件，有三种方式去build：
		  1. 在 .vsocde/settings.json 中配置
		   2. 使用 CMakePresets.json 配置
		   3. 修改 CMakeLists.txt 配置
	   这里使用第三种：修改 CMakeLists.txt
	   ```
	    # CUDA targets
		set(DEFAULT_CUDA_VERSION 12.6.85)
		set_ifndef(CUDA_VERSION ${DEFAULT_CUDA_VERSION})
		message(STATUS "CUDA version set to ${CUDA_VERSION}")
		
		# GPU targets
		set(DEFAULT_GPU_ARCHS 75)
		set_ifndef(GPU_ARCHS ${DEFAULT_GPU_ARCHS})
		message(STATUS "CUDA version set to ${GPU_ARCHS}")
	   ```
	   ![](assets/Pasted%20image%2020260211165029.png)
	
	4. 构建 protobuf 和 sample_onnx_mnist
	   sample_onnx_mnist 依赖 protobuf，需要先构建 protobuf。
	   ctrl+shift+p -> Set Build Target -> third_party.protobuf -> build
	   ctrl+shift+p -> Set Build Target -> sample_onnx_mnist -> build
	   
	5. 
	   
	   
	