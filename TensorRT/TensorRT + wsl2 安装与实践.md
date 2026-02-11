1. **确认系统环境**
   系统：Ubuntu24.04(wsl2)
   cuda版本：12.6
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
			   1. 