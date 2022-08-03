# week 1

# 从零开始Angel编译

``踩了很多坑，记录一下，希望能帮助到大家一点吧``  
**大家有遇到类似的问题，可以附在回答里面，把问题集中一下，群策群力解决问题快一点,谢谢大家啦!**



# 从零开始Angel编译
@[TOC]

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


## 编译warning记录

![在这里插入图片描述](https://img-blog.csdnimg.cn/078b5c44091542e0a5eba5c17ff2e56a.png)
  
  	很不幸，编译失败了！

![在这里插入图片描述](https://img-blog.csdnimg.cn/07f485aa45314b2e927921f635758913.png)  
  
  	分析一下报错原因：maven中有jar包没有下载下来
  
  **思路一： 把maven下载到的本地库中的plugin文件全删了，然后用maven reimport 一下**
  	
[解决尝试1](https://blog.csdn.net/WGH100817/article/details/101722633?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165934781316782350858841%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=165934781316782350858841&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~pc_rank_v36-1-101722633-null-null.142%5Ev37%5Epc_rank_v36&utm_term=Plugin%20org.apache.maven.plugins:maven-jar-plugin:2.6%20or%20one%20of%20its%20dependencies%20could%20not%20be%20resolved:%20Failed%20to%20read%20artifact%20descriptor%20for%20org.apache.maven.plugins:maven-jar-plugin:jar:2.6:%20Could%20n&spm=1018.2226.3001.4187)

**思路二：你的maven版本和项目依赖不匹配，降低一下你的maven版本**
>去下载低版本的maven，然后在idea 的setting里面重新导入低版本的maven，同时删掉本地库中的plugin文件，重新下载

**思路三：你的idea版本和maven版本不匹配**
>首先，你的idea会自带集成的maven，和idea是匹配的，但是这个集成的maven是某个时候坑死你，自带的seeting.xml是隐藏文件，不容易找到，对新手很不友好，所以自己去下载一个对应的maven吧！

## 3. 配置Ubuntu系统
* 更换镜像源
 参考教程：[更换阿里镜像源](https://blog.csdn.net/weixin_45877759/article/details/107017960)
 * 小问题
通过虚拟机安装可能会出现和Windows之间不能通过复制粘贴传输文字或文件，并且Ubuntu显示的界面不能铺满显示窗口的现象，可以通过下面两个命令解决上述问题：
```shell
sudo apt-get autoremove open-vm-tools

sudo apt-get install open-vm-tools-desktop
```

下面是新安装ubuntu系统的一些常见配置：  

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
[git安装及配置教程](https://blog.csdn.net/qq_51212018/article/details/111053474?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165898231916781667830457%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165898231916781667830457&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-111053474-null-null.142%5Ev35%5Epc_rank_34,185%5Ev2%5Econtrol&utm_term=ubuntu20.04%E5%AE%89%E8%A3%85git&spm=1018.2226.3001.4187)


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
我下载的是idea2021版本的，注意后面的包的配置和idea版本对应

[下载教程](https://blog.csdn.net/qq_52732885/article/details/123347228?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165899723516781683956953%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=165899723516781683956953&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~pc_rank_34-2-123347228-null-null.142%5Ev35%5Epc_rank_34&utm_term=ubuntu20.04%E4%B8%AD%E5%AE%89%E8%A3%85Intellj&spm=1018.2226.3001.4187)


* 使用 idea 打开angel项目后 ，idea会自动加载依赖 ，等它自动加载完哈...慢慢等待

### 依赖下载报错

[解决方案](https://blog.csdn.net/qq_36416905/article/details/109579106)

* 参考链接里面提供了三种解决方案，各取所需

### ubuntu卡死

* 运行example的时候会卡死  

[解决卡死问题](https://blog.csdn.net/jiesunliu3215/article/details/108572153)


### 分配内存空间
	内存不够，那就分配多一点内存给虚拟机
[分配内存的教程](https://blog.csdn.net/Sun_study/article/details/125108945?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~default-2-125108945-blog-119840433.pc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~default-2-125108945-blog-119840433.pc_relevant_multi_platform_whitelistv3&utm_relevant_index=5)
#### 教程中三个输入框解释：

* 第一个框从当前硬盘空间中索取的空间
* 第二个框当前硬盘的新空间
* 第三个框留取的剩余空间（自动计算，不用动）

## maven配置-问题解决

#### maven版本与idea不匹配

[maven下载配置教程](https://blog.csdn.net/qq_31635851/article/details/123338562?ops_request_misc=&request_id=&biz_id=102&utm_term=ubuntu20.04%20maven%E5%AE%89%E8%A3%85&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-123338562.142%5Ev35%5Epc_rank_34&spm=1018.2226.3001.4187)

`下面是总结的几个可能报错的原因和步骤：`
#### 1、IntelliJ IDEA 和maven版本不兼容导致

更换到匹配版本

2021版本最多支持到maven3.8.1

#### 2、删除之前的插件依赖

```shell
cd C:\Users\你的账户名\.m2\repository\org\apache\maven\plugins
rm -r *
```
删除该路径下的文件（里面可能是高版本下载的插件，还是会冲突）

如下图：刷新会重新下载

![在这里插入图片描述](https://img-blog.csdnimg.cn/40705d02166a4b2c983dc56feeef152f.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/a17cdab6f69e4789902afcf570314acd.png)

* 问题解决！
![在这里插入图片描述](https://img-blog.csdnimg.cn/87f9f32ea57840de8666a82ba1d39293.png)

## 编译报错

![在这里插入图片描述](https://img-blog.csdnimg.cn/45c87893744140afaf8d536edc16de87.png)
`观察一下报错，发现又是maven包没有下载下来，仔细看报错信息，数据包请求超时，下载不下来，所以问题的根源是网络问题`

 1. 你挂了vpn，取消掉vpn重新下载
 2. 你的网络情况不容乐观，建议换一个快一点的网络

* 关掉vpn，换一个快一点的WIFI ， SUCCESS！

![在这里插入图片描述](https://img-blog.csdnimg.cn/7b8ac24e7c314941868900ef03d47fad.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/6824ca80382a4167a7907e25e421104b.png)
> 编译倒是表面上成功了，哈哈哈哈哈哈哈哈不怀好意的笑了

 ## 运行example报错
* 包没有下载下来

![在这里插入图片描述](https://img-blog.csdnimg.cn/0c5518a1018444e2bd3315676b606f31.png)





* maven插件没下载下来
![78f27f32994cce0e7b9565662d09dca](https://img-blog.csdnimg.cn/img_convert/d8cb6295c1bdbd5e883a1b9fbf4fdcde.jpeg)

* idea自带的maven找不到setting.xml文件

![b3de37807ffffccb53eff98e380035b](https://img-blog.csdnimg.cn/img_convert/81596c42dc206249f99ace354b6920ca.jpeg)


- 尝试解决1：  

![f7c3be3445b5b029415b6fae301d8b8](https://img-blog.csdnimg.cn/img_convert/04f71d437de82de9022a5feda4c7601d.jpeg)
>唉，没解决....

尝试解决2：查看路径  
> 都是idea默认的路径

![image](https://img-blog.csdnimg.cn/img_convert/c9dbe83ee756acbe6dad235daf75c930.png)

- 尝试解决3：自己配置maven3.8.6
> 重新加载依赖，然后编译，重新生成发布包，然后再运行example，还是报错。。。。


![在这里插入图片描述](https://img-blog.csdnimg.cn/85805785da814a48bdacfdaef7afc120.png)

`缺失spark包，编译有问题，回去重新编译吧！`
> 之前不是编译过了么，还要重新编译？
> 对，因为idea2021版本和maven版本不匹配 ，怎么解决这个问题，降低maven版本，我配了个maven3.6.3，重新下载依赖，然后编译，成功了...

`编译成功了，不代表demo运行成功，报错找不到类。。。`

![在这里插入图片描述](https://img-blog.csdnimg.cn/35290ccd64244191a631182315f3b981.png)


![在这里插入图片描述](https://img-blog.csdnimg.cn/4b6cf970dbae42b7a5f404a7c129b689.png)
- 可能思路
![在这里插入图片描述](https://img-blog.csdnimg.cn/05bbf55b664748559310beea10568ad3.png)

## 运行成功的奥秘
- 为什么找不到类的原因，因为压根没加入路径  


![在这里插入图片描述](https://img-blog.csdnimg.cn/917bd86a8a47480ca46ceada14812e21.png)


![在这里插入图片描述](https://img-blog.csdnimg.cn/9fbfa9187adb4b02b136aa02a55e21d7.png)
* 运行成功

![在这里插入图片描述](https://img-blog.csdnimg.cn/08cd7359117b421896ed5765ef8e6f30.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/8003cc0191824bda9c4bddee6da7c703.png)


* CCExample报错退出

![在这里插入图片描述](https://img-blog.csdnimg.cn/5b5a364a35ff4f70ac87f6b7a4573827.png)  


### 解决方法
`重写start函数`  

![在这里插入图片描述](https://img-blog.csdnimg.cn/5f5e90108e7e4a99b79bdda876614d52.png)



- 运行成功

![在这里插入图片描述](https://img-blog.csdnimg.cn/b39ebda5974f42958e3dd99391d6f667.png)

# 总结
`首先，我在windows环境下折腾了几天，心态崩了，因为windows环境下有太多的意想不到的惊喜，最后放弃舒适的Windows，来使用Linux环境，配置Ubuntu系统。这是一个过程吧，Linux学了很久了也没用，正好复习回忆命令了，熟悉未来的开发环境。`

## 非常感谢Rachel导师对我遇到的问题的耐心解答和指导，受益良多
> 这个过程太过于艰难，但走完这个过程，来回的琢磨环境问题，maven，idea，scala，spark，hadoop,docker 这些东西本来我不熟悉，但是经过数天的磨练，底层环境配置问题摸了个门儿清，下次配置部署项目肯定轻车熟路了！

**希望以上我遇到的报错，能够帮助到大家，让大家少走弯路，节省时间精力！**
