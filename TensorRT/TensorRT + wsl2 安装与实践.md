1. **确认系统环境**
   系统：Ubuntu24.04(wsl2)
   cuda版本：12.6
2. **下载TensorRT Debian仓库包**
   从 [TensorRT 下载页面](https://developer.nvidia.com/tensorrt)，下载适用于你的 CUDA 版本和作系统的 Debian 仓库包。
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
     