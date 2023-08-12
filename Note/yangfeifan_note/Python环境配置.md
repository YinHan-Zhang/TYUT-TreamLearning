# Miniconda+Jupyter 小白设置指南(Windows)

## **前言**

* 关于 Conda 和 Jupyter 的说明

Conda 是一种环境管理软件，尤其是 Python2 和 Python3 不兼容的情况下，**Conda 可以建立多个相互隔离的虚拟环境**，让不同的程序在不用的 Python 版本以及不同版本的包下运行。

[Jupyter Notebook](https://link.zhihu.com/?target=https%3A//jupyter.org/) 是基于网页的用于交互计算的应用程序。其可被应用于全过程计算：开发、文档编写、运行代码和展示结果。简而言之可以**直接编写代码**，**运行代码**，**运行结果**也会直接显示在代码块下方。

另外，强烈建议下载 VSCode，VSCode 自带的 Juputer 插件和 Python 插件可以完美兼容 Miniconda 和 Jupyter，使用起来比专门打开 Jupyter 网页端更为顺滑。

## 0.准备

首先你需要有一台电脑 然后你需要保证这个电脑是干净的：没有额外安装 Python ，没有安装miniconda/conda这些环境管理软件（建议小白从零开始，大佬请忽略）

## 1.Miniconda 安装配置

### 1.1下载 Miniconda

Miniconda 下载地址：[https://docs.conda.io/en/latest/miniconda.html](https://gitee.com/link?target=https%3A%2F%2Fdocs.conda.io%2Fen%2Flatest%2Fminiconda.html)

最新版 Miniconda For Windows 下载链接：

[https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe](https://gitee.com/link?target=https%3A%2F%2Frepo.anaconda.com%2Fminiconda%2FMiniconda3-latest-Windows-x86_64.exe)

### 1.2Windows 下安装配置

#### (1)开始安装

- Just Me (recommended) (无需管理员权限)
- Clear the package cache upon completion

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_01.png)

#### (2)点击安装程序，选择路径

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_02.png)

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_03.png)

* **环境变量选择(PATH)：**在这一步不建议勾选第二个环境变量选项，建议在之后手动添加环境变量以防止环境变量污染的问题。
* 一路next安装完成
* 检验是否安装成功，在开始菜单出，左击 Anaconda Prompt

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_04.png)

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_05.png)

* 打开后，左边有 (base)，即安装成功

#### (3) 环境变量配置

* 这里我使用的是win11（win10系统的同学可以自己找一下高级选项，或者询问度娘）
* **windows图标右键**-->**系统** -->**高级系统设置**

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_06.png)

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_07.png)

* 上方选项卡“高级” --> “环境变量”按钮

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_08.png)

* 对“Path”进行设置

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_09.png)

* 新建对应环境变量
* **注意：我的miniconda安装路径为“`D:\MiniConda`”，设置时修改为自己的安装路径**

```
D:\MiniConda
D:\MiniConda\Scripts
D:\MiniConda\Library\bin
```

#### (4)conda 换源(镜像服务器)

由于 miniconda 下载文件/依赖库等默认的采用国外的服务器，下载速度很慢，一般改为国内的清华源/阿里源等方式解决。我这里安装的是[清华源](https://link.zhihu.com/?target=https%3A//mirrors.tuna.tsinghua.edu.cn/help/anaconda/)。

* “Win”+R 打开“运行”，输入cmd进入cmd终端，依次输入以下两行(每输入一行后回车)

```text
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

* 第一行代码添加清华源    第二行代码生成".condarc"文件
* 打开`C:\Users\"用户名"\.condarc`找到该文件

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_11.png)

* 将第一步中清华源下载界面的内容复制进去然后保存

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_12.png)

* 在PowerShell中输入以下代码清除索引缓存，就可以使用了

  ```
  conda clean -i
  ```

#### (5)conda安装成功验证

* 安装成功后，输入查看conda环境/版本相关信息

```
conda info
conda --version
```

* 若出现以下界面，说明安装成功

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_13.png)

#### (6)pip换源

* pip 是Python安装第三方包的管理工具，该工具提供了对Python 包的查找、下载、安装、卸载的功能。我们安装完conda环境pip就是最新的了

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_14.png)

* 复制文本后，在 **Anaconda Powershell Prompt** 中粘贴运行创建pip.ini文件

```
# 设置 PyPI 镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

* 这里我用的清华源，复制以下配置项到pip.ini文件中即可

```
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = https://pypi.tuna.tsinghua.edu.cn
```

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_18.png)

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_17.png)

#### (7)创建与激活conda环境

* 打开**Anaconda Powershell Prompt**输入以下命令
* 第一次激活 conda 虚拟环境需要初始化

```
conda init
```

* 创建conda环境
* 其中 ***-n*** 代表创建的环境名称，这里是 ***Datawhale***，并指定 ***Python 版本为 3.11***

```
conda create -n Datawhale python=3.11
```

* 激活conda环境

```
conda activate Datawhale
```

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_15.png)

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_16.png)

* 如果需要删除某个 Conda 环境：

```
conda deactivate # 退出该环境
conda remove -n Datawhale --all # 删除整个环境
```

## 2.安装Jupyter

**下面提供两种安装Jupyter的方法（推荐conda安装）**

### 2.1conda安装Jupyter

 	打开**Anaconda Powershell Prompt**输入以下命令

* 进入虚拟环境（这里用的刚刚的Datawhale环境）

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_22.png)

* 安装Jupyter

```
conda install jupyter notebook
```

* 安装成功以后菜单会出现notebook

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_24.png)

* 点击运行即可

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_23.png)

* 运行成功会自动跳转到浏览器

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_25.png)

### 2.2pip安装Jupyter

* 命令行运行代码，第二行出现的就是刚刚配置的清华源

  ```
  pip install jupyter
  ```

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_19.png)

* 在指定路径输入会跳转到浏览器

  ```
  jupyter-notebook
  ```

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_20.png)

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_21.png)

* 结束学习时关掉命令行窗口或者使用：

  ```
  Ctrl + C # 关闭 Jupyter Notebook 服务
  ```

### 2.3Jupter默认空间设置

运行 Jupyter 后会出现一堆文件夹，这是你的`\Home`目录里面的目录文件，如果不希望今后编写的 Jupyter Notebook 文件都直接放在该目录下，就需要修改 Jupyter Notebook 的存放路径。

打开文件，查找`c.NotebookApp.notebook_dir`，删除注释符号，并将目标文件夹路径复制黏贴到后面的`''`中。保存文件。

* 获取配置文件路径

```
jupyter notebook --generate-config
```

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_26.png)

* 打开文件，CTRL + F 找到notebook_dir，删除注释符号，并将目标文件夹路径复制黏贴到后面的`''`中。保存文件。

![](C:\Users\yangf\Desktop\DW_Learning\TYUT-TreamLearning\Note\yangfeifan_note\picture\pic_27.png)

***python相关包举例***

- 科学计算相关

- - numpy
  - pandas
  - scipy
  - scikit-learn

- 绘图相关

- - matplotlib
  - seaborn

- Jupyter Notebook相关

- - jupyter
  - notebook
  - ipykernel
  - nb_conda_kernels

- 其他

- - beautifulsoup4
  - reportlab

## 3.vscode安装

