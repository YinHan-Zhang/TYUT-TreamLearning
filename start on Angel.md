# week 1

# 从零开始Angel编译

``踩了很多坑，记录一下，希望能帮助到大家一点吧``  
**大家有遇到类似的问题，可以附在回答里面，把问题集中一下，群策群力解决问题快一点,谢谢大家啦!**



## 1. 下载VMware
网上教程很多，我参考了这个教程，挺详细的[VMware安装教程](https://blog.csdn.net/weixin_45912291/article/details/108894737)

* 需要密钥 ，要么自己购买，要么自己去找免费的（可能会花费一点时间去找，很多事失效的）

## 2. 下载Ubuntu系统
参考教程：[Ubuntu系统安装](https://blog.csdn.net/weixin_45912291/article/details/108901106)  

**注意一下几点**：
```
1. 用户名、计算机名、文件夹名 全部英文
2. 尽量不要装输入法，就用纯英文环境
3. 尽量不要装汉化包 （不要问为什么，因为以后在你意想不到的地方会发生意想不到的错误）
4. 
```

## 3. 配置Ubuntu系统
* 更换镜像源
 参考教程：[更换阿里镜像源](https://blog.csdn.net/weixin_45877759/article/details/107017960)
 * 小问题
通过虚拟机安装可能会出现和Windows之间不能通过复制粘贴传输文字或文件，并且Ubuntu显示的界面不能铺满显示窗口的现象，可以通过下面两个命令解决上述问题：
```shell
sudo apt-get autoremove open-vm-tools

sudo apt-get install open-vm-tools-desktop
```

[系统设置博客](https://blog.csdn.net/weixin_44750512/article/details/108310906?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-108310906-blog-108901106.pc_relevant_multi_platform_whitelistv1&spm=1001.2101.3001.4242.1&utm_relevant_index=3)


## docker安装
### 常规安装教程
* 参考安装教程

[安装教程1](https://blog.csdn.net/u010381752/article/details/114086343?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165896695716781432960296%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165896695716781432960296&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-114086343-null-null.142%5Ev35%5Epc_rank_34,185%5Ev2%5Econtrol&utm_term=ubuntu20.04%E5%AE%89%E8%A3%85docker%E6%95%99%E7%A8%8B&spm=1018.2226.3001.4187)    
[安装教程2](https://blog.csdn.net/lhaobin/article/details/119217255)  

### 小白傻瓜教程
[无脑敲命令](https://github.com/JeromeYHJ/start-on-Angel)

* 我的ubuntu安装的是20.04哦 ，教程里面是16.04 把版本号改成20.04，就行了！

`好的，必须安装的东西，就差不多了，还有一些开发工具什么的，自行安装就好了，下面就开始配置Angel啦！`

## git安装与拉取
[git教程](https://blog.csdn.net/qq_51212018/article/details/111053474?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165898231916781667830457%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165898231916781667830457&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-111053474-null-null.142%5Ev35%5Epc_rank_34,185%5Ev2%5Econtrol&utm_term=ubuntu20.04%E5%AE%89%E8%A3%85git&spm=1018.2226.3001.4187)


## 报错

*  ./ configure 报错  ： 未安装gcc编译器 [安装gcc教程](https://blog.csdn.net/dotdotyy/article/details/120107716?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165898003416781818716400%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165898003416781818716400&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-120107716-null-null.142%5Ev35%5Epc_rank_34,185%5Ev2%5Econtrol&utm_term=ubuntu%E5%AE%89%E8%A3%85gcc&spm=1018.2226.3001.4187)
* 


## docker脚本编译
`把angel项目下载到本地`
```python
进入angel项目目录
执行 sh ./docker-build.sh
```
...等待就行了 ， 我等了1h58mins


## 编译成功截图
![在这里插入图片描述](https://img-blog.csdnimg.cn/08547d3508e747caa6fd794baa6d3763.png)
* 输出压缩包!
![在这里插入图片描述](https://img-blog.csdnimg.cn/845fbb47eaad4749a8cc7d38f23700e5.png)
* 解压发布包
```shell
cd ./dist
unzip angel-3.2.0-bin.zip
```
 `解压完成后：`
 
![在这里插入图片描述](https://img-blog.csdnimg.cn/44fb68b9f9744e05b2863d6a7929c21e.png)
## 大佬详解
[手动编译过程](https://github.com/JeromeYHJ/start-on-Angel)


## idea 下载
[下载教程](https://blog.csdn.net/qq_52732885/article/details/123347228?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165899723516781683956953%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=165899723516781683956953&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~pc_rank_34-2-123347228-null-null.142%5Ev35%5Epc_rank_34&utm_term=ubuntu20.04%E4%B8%AD%E5%AE%89%E8%A3%85Intellj&spm=1018.2226.3001.4187)


* 使用 idea 打开angel项目后 ，idea会自动加载依赖 ，等它自动加载完哈...慢慢等待

### 依赖下载报错

[解决方案](https://blog.csdn.net/qq_36416905/article/details/109579106)

* 参考链接里面提供了三种解决方案，各取所需

**我目前还没完全解决这个问题**

### ubuntu卡死

* 运行example的时候会卡死 ， 不要暴力关机重启，参考这个blog
[解决卡死问题](https://blog.csdn.net/jiesunliu3215/article/details/108572153)


### 分配内存空间

[分配内存](https://blog.csdn.net/Sun_study/article/details/125108945?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~default-2-125108945-blog-119840433.pc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~default-2-125108945-blog-119840433.pc_relevant_multi_platform_whitelistv3&utm_relevant_index=5)

####三个输入框解释：
* 第一个框从当前硬盘空间中索取的空间
* 第二个框当前硬盘的新空间
* 第三个框留取的剩余空间（自动计算，不用动）

