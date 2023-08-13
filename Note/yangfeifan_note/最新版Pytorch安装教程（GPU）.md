# 最新版Pytorch安装教程(GPU)

## 0.准备

* 本教程使用conda安装，需要提前安装配置conda环境
* 操作系统：windows11

## 1.CUDA和CUDNN下载

### (1)查看CUDA对应版本

* win+R 打开cmd输入

```
nvidia-smi
```

![001](https://s2.loli.net/2023/08/13/dLKOkVsQy3ci4qg.png)

![002](https://s2.loli.net/2023/08/13/XGHklvTQ4jqFme2.png)

* 查看自己的cuda版本（我当前的版本为12.0）
* **注意**：这里并不是表明您已经安装了cudatookit，仅仅代表，您当前驱动可以支持的**cuda最高版本**

### (2)去NVIDIA官网下载对应版本的CUDA和CUDNN

CUDA下载：https://developer.nvidia.com/cuda-toolkit-archive
CUDNN下载：https://developer.nvidia.com/rdp/cudnn-download

从官网下载对应版本的安装程序 [CUDA Toolkit Archive | NVIDIA Developer](https://link.zhihu.com/?target=https%3A//developer.nvidia.com/cuda-toolkit-archive)

**注意：我这里下载的是11.8**

* 下载完成后，安装到自己的指定的文件夹，一路next

* CUDA验证安装，输入：

```
nvcc -V
```

![003](https://s2.loli.net/2023/08/13/skHO3fYohS5nDZa.png)

* 下载对应的CUDNN（需要先注册账号）

![](https://s2.loli.net/2023/08/13/dH39bnpzBkqDxZ5.png)

* 下载完成后将压缩包解压至CUDA安装路径下重命名cudnn

![007](https://s2.loli.net/2023/08/13/nur6QO8HsFCVjUe.png)

* 运行bandwidthTest.exe (验证安装)
* **注意我的安装目录为C盘，大家按照自己的安装目录寻找对应CUDA文件夹**

![008](https://s2.loli.net/2023/08/13/Y5QEWCNAcMzIKJk.png)

![](https://s2.loli.net/2023/08/13/ZnJ6fdaqS4rXNzE.png)

* 输出结果PASS说明安装成功

## 2.安装pytorch

进入官网，查看以往版本 ：[Previous PyTorch Versions | PyTorch](https://link.zhihu.com/?target=https%3A//pytorch.org/get-started/previous-versions/)，根据版本选择合适的下载命令

* 我这里用的CUDA11.8版本

![009](https://s2.loli.net/2023/08/13/ToXxaLOUmEt3Def.png)

* 先不着急在命令行下载，为了方便管理，我们先建立一个虚拟环境，这里我建立了一个名为“pytorch39”的虚拟环境

```
conda create –n pytorch39 python=3.9
```

![010](https://s2.loli.net/2023/08/13/ujUtbq8OKiyBSHp.png)

* 查看我们有的虚拟环境，并激活pytorch39

```
conda env list
conda activate pytorch39
```

![011](https://s2.loli.net/2023/08/13/3sVWqKrl7RcQSj5.png)

* 下载pytorch

* **注意！！！**

  这里用conda安装时有大坑，我安装完运行torch.cuda.is_available()返回的一直是false，找了很久发现是清华源的问题。清华源中没有cuda版本的torch，所以一直下载的cpu版本，如果要用conda下载记得修改镜像地址。而且conda下载可能会由于torch文件太大而中断。

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

* 我最后使用pip安装的

```
pip3 install numpy --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu118
```

安装成功！

![](https://s2.loli.net/2023/08/14/O1ix5PpNoZAGLhI.png)