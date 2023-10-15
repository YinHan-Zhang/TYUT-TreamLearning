## 0x0 什么是PIFuHD ?

[PIFuHD](https://shunsukesaito.github.io/PIFuHD/)：用于高分辨率 3D 人体数字化的多级像素对齐隐式函数，基于Shunsuke Saito, Tomas, Simon, Jason Saragih, Hanbyul Joo, University of Southern California, Facebook Reality Labs, Facebook AI Research等研究人员及企业在2020年四月发表在[arxiv上的论文](https://arxiv.org/pdf/2004.00452.pdf)，使用Python打造，可以通过深度学习计算的方式将平面2D人体照片转换为3D空间模型。

## 0x1 环境准备

### 硬件配置：
* 显卡：RTX 4060 Laptop
* CPU：Intel i9-13900HX
* 内存：32GB
### 软件配置：
* 操作系统：Ubuntu 22.04.3 LTS
* 显卡驱动：NVIDIA-Linux-x86_64-525.125.06
* CUDA版本：11.8
* cudnn版本：8.9.5
* python版本：3.10
* pytorch版本：1.12.0
* 编译器版本：gcc 9.4.0

## 0x2 环境准备
### 安装Ubuntu22.04
在[Ubuntu官网](https://cn.ubuntu.com/download/desktop)下载安装镜像，使用UltraISO等软件制作启动U盘，在BIOS中设置U盘启动，进入Ubuntu安装界面，选择安装Ubuntu，一路默认安装即可，注意勾选“安装专有的硬件驱动”。由于虚拟机无法使用CUDA，所以请**不要**使用虚拟机安装！！！
### 安装显卡驱动
安装完成Ubuntu后，打开终端，输入命令：
```shell
sudo apt update
sudo apt upgrade
ubuntu-drivers list
```
此时系统会列出适用于当前设备的显卡驱动，比如
```
nvidia-driver-535-server-open, (kernel modules provided by linux-modules-nvidia-535-server-open-generic-hwe-22.04)
nvidia-driver-525-server, (kernel modules provided by linux-modules-nvidia-525-server-generic-hwe-22.04)
nvidia-driver-535, (kernel modules provided by linux-modules-nvidia-535-generic-hwe-22.04)
nvidia-driver-535-server, (kernel modules provided by linux-modules-nvidia-535-server-generic-hwe-22.04)
nvidia-driver-525, (kernel modules provided by linux-modules-nvidia-525-generic-hwe-22.04)
nvidia-driver-525-open, (kernel modules provided by linux-modules-nvidia-525-open-generic-hwe-22.04)
nvidia-driver-535-open, (kernel modules provided by linux-modules-nvidia-535-open-generic-hwe-22.04)

//通常会列出多个版本的驱动，建议选择不带任何后缀的版本
```
选择合适的版本运行，比如
```shell
sudo apt install nvidia-driver-535
//注意将驱动名称替换为上一条指令输出的结果
```
进行安装
### 安装CUDA
**注意！注意！注意！** ，由于需要用到 `openpose`，而`openpose`年久未修，建议最高cuda版本为**11.8**，高于此版本可能会出现bug。

首先，在终端中运行`nvidia-smi`，查看显卡支持的最高cuda版本
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| N/A   45C    P8     3W / 115W |    420MiB /  8188MiB |      1%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1848      G   /usr/lib/xorg/Xorg                187MiB |
|    0   N/A  N/A      2010      G   /usr/bin/gnome-shell               93MiB |
|    0   N/A  N/A     43583      G   ...RendererForSitePerProcess      136MiB |
+-----------------------------------------------------------------------------+

```

可以看到`CUDA Version: 12.0`，就是说当前显卡支持的最大cuda版本是12.0,考虑到前文提到的兼容性问题，我们选择安装`cuda11.8`。

进入[CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)，选择合适的版本号，此处以`11.8.0`为例。

点击`CUDA Toolkit 11.8.0`，跳转安装页面，依次选择`Linux -> x86_64 -> Ubuntu-22.04 -> runfile(local)`然后分别运行下方显示的两条指令。
```shell
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

//不同版本安装指令可能不同，以页面具体显示为准
```
在安装过程中，会询问你是否接受协议，输入accept，然后弹出安装选项，需要**取消勾选**显卡驱动。
```
[]Driver
```
安装完成后，需要配置环境变量，打开~/.bashrc文件，添加如下内容：
```
export PATH=$PATH:/usr/local/cuda-11.8/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64

//其他版本CUDA请自行将11.8改成自己的版本
```
最后重载:
```shell
source ~/.bashrc
```

### 安装cuDNN
在[官网](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local)先登录NVIDIA，然后点击`Download cuDNN`并勾选`I Agree To the Terms of the cuDNN Software License Agreement`，这里只显示11.x和12.x，如果需要历史版本就在文末进入存档版本。这里下载11.x的 v8.9.5版本，下载`Local Installer for Ubuntu22.04 x86_64 (Deb)`安装包。

接着安装zliblg
```shell
sudo apt-get install zlib1g
```
安装cudnn包
```shell
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.5.29_1.0-1_amd64.deb
//不同版本记得修改版本号
```
安装结束后会提示安装key，直接复制运行弹出的命令即可，以下命令仅为示例，以自己终端输出的内容为准。
```
sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.5.29/cudnn-local-72322D7F-keyring.gpg /usr/share/keyrings/
```
更新库
```
sudo apt-get update
```
查看libcudnn8版本，下列命令均需要根据这个版本号修改
```
apt-cache policy libcudnn8
```
得到输出结果，包含以下内容
```
...
候选： 8.9.5.29-1+cuda11.8
...
```
记下版本号并在下面命令中替换。
```
sudo apt-get install libcudnn8=8.9.5.29-1+cuda11.8
sudo apt-get install libcudnn8-dev=8.9.5.29-1+cuda11.8
sudo apt-get install libcudnn8-samples=8.9.5.29-1+cuda11.8
```
### 测试
运行：
```
sudo apt-get install libfreeimage3 libfreeimage-dev
cp -r /usr/src/cudnn_samples_v8/ $HOME
cd $HOME/cudnn_samples_v8/mnistCUDNN
make clean && make
./mnistCUDNN
```

得到输出结果
```
Test passed!
```
即为安装成功。

## 0x2 部署PIFuHD代码

### 拉取源码
方法一：使用官方源码
```
git clone https://github.com/facebookresearch/pifuhd.git
```
然后将所有文件中的`np.int`替换为`int`,所有`np.bool`替换为`bool`。

方法二：使用我修改好的的源码
```
git clone https://github.com/SeimoDev/pifuhd.git
```
### 安装依赖
```
sudo apt install python3-pip python-is-python3
pip install -r requirements.txt 
sudo apt-get install freeglut3-dev
```
### 下载预训练模型
```
sh ./scripts/download_trained_model.sh
```
### 测试
```
sh ./scripts/demo.sh
```
得到输出结果,如果没有报错且正常输出模型展示页面，即为安装成功。

## 配置openpose
### 获取源代码
```
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
cd openpose/
git submodule update --init --recursive --remote
```
### 编译源代码
```
sudo apt-get install cmake-qt-gui
mkdir build
cd build
cmake-gui ..
```
![IniT.png](https://i.imgs.ovh/2023/09/19/IniT.png)

将`Where is the source code`设置为Openpose源码目录
将`Where to build binaries`设置为build目录
点击`Configure`
![Ilxp.png](https://i.imgs.ovh/2023/09/19/Ilxp.png)

选择`Unix Makefiles`
然后点击`Finish`，关闭后勾选`BUILD_PYTHON`，取消勾选`USE_CUDNN`然后点击`Generate`。
待下方输出
```
Configuring done
Generating done
```
关闭cmake-gui窗口，执行
```
make -j`nproc`
```
如果不出意外，这样会自动完成openpose的编译。
### 下载预训练模型
由于官方模型下载服务器Shutdown了，这里提供一个存档，可以自行下载放到models文件夹下

[下载Models.zip](https://zmqpk-my.sharepoint.com/:u:/g/personal/seimo_seimo_cn/EZlfdWAmNBRPniMFG0tbsEcB_RzqcP0ldRsJq0hTNHqfdQ?e=gHzf5J)

### 测试

运行以下命令
```
./build/examples/openpose/openpose.bin --video examples/media/video.avi
```
如果成功输出食品并且带有人体骨架即为成功

## 开始使用
恭喜你，截至这里，前期工作已经完成，下面开始正式使用

切换到PIFuHD项目目录，将需要渲染的图片放到`sample_images`文件夹，然后运行如下命令：
```
python apps/batch_openpose.py -d openpose/ -i sample_images -o sample_images
//如果你的openpose项目目录不在pifuhd/openpose下，请自行修改
python -m apps.simple_test
python apps/clean_mesh.py -f results/pifuhd_final/recon
```

到此，如果你没有遇到报错，那么恭喜你，你已经成功运行了PIFuHD，并将人体图片转换为了三维模型。

你可以在`results/pifuhd_final/recon`下找到生成的obj文件，拖入需要的建模程序即可使用。

也可以运行下面这个命令将其渲染为视频展示
```
python -m apps.render_turntable -f results/pifuhd_final/recon -ww 512 -hh 512
```

## 关于部分报错的处理：

- 在进行cmake编译时如遇到报错缺少库文件，请自行搜索相关库安装补全，一般报错时会有提示。
- 如果在最后生成模型时报错`CUDA out of Memory`,请降低图片分辨率后在次尝试。实测8GB显存可以渲染96*108分辨率的图片。