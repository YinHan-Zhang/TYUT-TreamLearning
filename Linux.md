常用终端命令：



## 操作命令：

### ls

ls：查看文件夹下的内容	

ls -a:查看多有文件包括隐藏文件；以.开头的就是隐藏文件

ls -l :以列表的形式显示所有的文件信息     

 ls -l -h :以人性化的方式输出

ls -d :只输出目录本身

-n : 列出UID和GID，和群组名称

-t :按时间排序

【--color=auto】

快捷键：ctrl+shift+‘+’方大窗口字体，ctrl+‘—’缩小

### 通配符

*：代表任意个字符     

 ？：戴白哦一个字符   

[]：表示可以匹配字符组中的任一一个     

[abc]：匹配a,b,c中的任意一个

[a-f]：匹配从a到f范围内的任意一个

举例： ls 1*.txt:查询以1开头的文件  ls 1?1.txt  ls 

#### 命名规则

* 除了/字符，所有字符都合法
* 符号字符最好不用
* “.”开头的文件时隐藏的

### 文件夹命令

pwd：查看当前所在的文件夹

cd[ 目录名]:切换文件夹

#### touch[acdmt] [文件名]：

> touch可以创建新的文件名

mtime:modifytime  文件内容变更时间

ctime: status time 权限属性变更时间

atime: access time	文件内容被读取的时间

如果文件不存在，新建文件 	 **gedit 文件名 文本文档的形式编辑**

-a: access time 

-c： 仅修改文件时间

-d：后面可接修订日期  ，也可用 --date=“time”

-m：仅修改时间，格式[YYYYMMDDHHmm]



#### mkdir/rm

 mkdir[目录名]：创建目录                                                              

rm[文件]：删除指定的文件名  

-f：就是强制删除   

-i ：有一次机会  回答y真删   

-r：直接删掉整个文件夹

clear：清屏幕          

  **在敲文件名或目录名时，敲出前几个字母+一个tab键，系统自动补全命令**

cd + “-” : 切换到上一次使用cd命令前的目录 

cd + ..   ：就是回到上级目录，一个.表示当前目录，两个.表示上一级目录



tree：将文件夹以树状图的形式展现出来        **当前目录路径  ./**

```
tree -L 3/yinhan/ 显示yinhan目录下三级目录树

```

### 文件类型

正规文件(regular file): 纯文本、二进制、数据格式文件

目录（directory）

链接文件(link)

设备文件（device）

数据接口文件（socket2）



* .sh ：脚本或批处理文件（script），因为是用shell写的
* *Z，.tar,tar.gz,.zip
* .html,.php





### 文件命令

拷贝文件命令：

cp 源文件路径  新文件路径      

-a：复制目录时使用（保留文件原有属性）

 -f：已经存在的目标文件而不提示（强制） 

 -v：显示拷贝过程

-p: 连同文件的属性（权限，用户，时间）

-i ：交互式复制，覆盖时有提醒用户    

  -r：如果是一个目录的文件，将递归复制该目录下的子目录和文件

例：cp  ./a.txt    a/a.txt   从当前目录拷贝a.txt到a目录下   cp  a.txt  b/a   拷贝到b目录下的a文件夹

​					 **不写文件路径，默认当前文件路径**

---

移动命令：mv 原位置路径  目标路径

​	mv  1.txt   c /1.txt      

**-f  **:强制

**-i  和前面一样**    

 移动的时候不需要 -r

​	重命名：mv 1.txat  1.txt   将1.txat改名为1.txt





### 账户

文件拥有者：

>  多个人共同使用主机来进行工作，考虑每个人的隐私权以及每个人喜好的工作环境。
>
> 只有把文件权限设置成文件的拥有者，才能进行查看。

群组

> 群组是用户的集合
>
> 每个组都分配了一个唯一的组ID号（GID）

标准组：创建文件要指定它的组

私有组：私有组只有用户自己，防止新文件分配到公共组里面



例子：王大毛家和张小猪家

> 王大毛家就是一个群组，三兄弟就是三个使用者，三个使用者在同一个群组里面。
>
> 三个使用者虽然在同一个群组内，但是我们可以设置权限，好让某些使用者个人的信息不被群组的拥有者查询，以保护个人隐私。
>
> 张小猪家又是另外一个群组。张小猪无法进行王家，除法张小猪和王大毛认识。

三种身份：

* User
* Group
* Others
* 万能天神：root

超级管理员：

> 在系统上的任务是对，普通用户和整个系统进行管理。

* 所有的用户名和UID被保存在/etc/passwd  这个文件夹中
* /etc/shadow :保存加密后的用户口令，所有用户可查看
* /etc/group：存群组属性
* /etc/gshadow： 用户组口令、组管理员信息等，只有root可查看



### 管理账户

```
useradd [选项] 用户名
usermod
userdel
groupadd
groupmod
groupdel 
```



设置密码时效：

```

```



#### 输出的权限显示

##### 第一栏：

十位字母代表文件权限：``-type --- --- ---``

* 第一个字符：代表文件类型 

​	【d】: 目录

​	【-】：文件

​	【I】：链接文件

​	【b】:设备文件里面的周边设备（随机存取文件）

​	【c】:设备文件里面的外界设备（鼠标、键盘）

* 接下来3个字符一组

  第一组：拥有者权限 ； 第二组：同群组使用者权限 ； 第三组：其他使用者权限

  rwx：可读可写可执行 ，没有某个权限用【-】代替

例题：``drwxr-xr--``

> * 文件拥有者rwx可进行任何操作
> * 同组使用者权限：rx ，可以进入目录进行工作，但不能修改
> * 其他人权限：r ，虽然有可读权限，但是没有x权限无法进入此目录

##### 第二栏：

> 表示有多少文件名链接到此节点（i-node）

每个文件的权限和属性都会记录到文件系统的i-node中，但是目录树却使用文件名来记录，因此每一个文件名就是一个i-node。也就是说，一个i-node会链接多个文件名和对应的文件的属性

##### 第三栏:

> 表示文件的（目录）的拥有者

##### 第四栏：

> 文件的所属群组

##### 第五栏：

> 文件容量大小，默认Bytes

##### 第六栏：

> 文件的创建日期或者最近的修改日期   ls -l -- full-time

##### 第七栏：

> 这个文件的文件名，文件名前多一个.表示隐藏文件   





### 文件权限

* chgrp:改变文件所属群组
* chown: 改变文件拥有者
* chmod: 改变文件的权限

#### chgrp

> change group，要改变的群组必须在/etc/group里面存在

```
chgrp [-R] 新群组名 被更改文件名

-R：进行递归变更，该目录下所有文件都变更
```



#### chown:

数字类型改变文件权限：

> 文件的基本权限就只有9个，owner/group/others 三种身份和对应的 read/write/execute
>
> r: 可以查询目录下的文件
>
> w: 对目录和目录下的文件，可增删改查
>
> x： 能否进入目录进行工作
>
> 
>
> 九个权限3个3个一组[rwx] [rwx] [rwx]
>
> 权限对应的分数：
>
> r:4  	w:w	x:1
>
> 例如：[-rwxrwx---]的分数  
>
> ```
> owner：rwx = 4+2+1 =7
> group: rwx = 4+2+1 =7
> others: --- = 0
> 
> 
> 
> ls -al :查看文件权限
> chmod 777(改变后的权限) filename
> ```
>
> 如果文件不希望被别人看见 ， 权限分数：740
>
> vim编辑一个shell文件处理后，权限一般是-rw-rw-r-- :664
>
> 变成可执行文件且不让他人修改：-rwxr-xr-x :755

#### umask

umask： 数字形式查看权限
umask -S: 文字形式查看权限

umaks 【三位数字组合】 ：当前权限减去umask指定权限

> umask 002 : 777 - 002 = 775  drwxrwxr-x 

符号类型：

> u: owner
>
> g:group
>
> o: other
>
> a:all 所有权限
>
> chmod u=rwx,g=rx,o=rwx
>
> ---
>
> +:加上权限
>
> -：减去权限
>
> chmod a+w filename

![1646875396466](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1646875396466.png)



![1646880671044](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1646880671044.png)



```
chattr +i 文件名  ：添加i属性

chattr -i 文件名	：删除i属性
```



lsattr :显示文件隐藏属性

```
-a  显示隐藏文件属性
-d	仅显示目录属性
-R	显示子目录属性
```

#### 特殊权限

SUID： set user ID

* 仅对二进制文件有效
* 当前用户对该文件需要有x权限
* 本权限只在程序执行过程中有效
* 当前用户暂时获得文件拥有者权限

``-rws-xr-x``     s出现在user权限组位置



SGID: 同理 ，  s出现在group权限组位置

``-rwx--s--x``



SBIT:  

* 只对目录有效
* 使用者要具有w,x权限
* 使用者在该目录下创建文件目录，只有root才能删除



如何设置特殊权限：

* 4：SUID
* 2：SGID
* 1：SBIT

在原来的9位权限码之前再加一位,只加一位！！！ 421的组合

```
-rwxr-xr-x(755)  改为 -rws r-x r-x  (4755)   chmod 4755 filename

符号：
chmod 0+s/g+s/u+s filename
```



#### 权限重点：

进入目录权限：

>  使用者要进入目录，必须要有x权限 
>
> 有x权限可使用cd等变换工作目录命令
>
> 想ls查阅文件名，还得有r权限

读取文件权限：

> 比如cat,less,more 等等
>
> 目录权限必须要有x，文件权限必须有r

修改文件权限：

> nano和vim编译器
>
> 目录权限：x
>
> 文件权限：r和x

创建目录权限：

> w和x

进入目录并执行该目录下的文件：

> 目录权限：x
>
> 文件权限：x



### 日期

1. 日期日历指令：cal  显示日历     -3：显示系统前一个月，当前月，下一个月的日历       -y：一年的日历

   ​							-j ：显示是当前年的第几天    cal＋年份   就是显示哪一年的日历

   显示当前时间：date   格式化输出   “+%Y年%m月%d日%H时%M分%S秒”  %F就是年月日，%T直接显示时钟

2. 历史指定：history：查看历史指令   ！历史指令编号 -->执行历史命令

   history +数字 ：就是显示最近的多少条命令
   
3. mtime: 最后修改时间（ls -lt）

    ctime: 状态改变时间(ls -lc)  
    
    atime:最后访问时间(ls -la)

   

### 文件内容查看



file [选项] 文件名   查看文件的类型

stat[选项] 文件名   查看文件的所有信息



#### 查阅非文本文件

od [-t TYPE] 文件名

TYPE参数选项

- a : 默认字符输出
- c ：使用ASCII字符输出
- d[size] : 十进制输出 ，size为每个整数的大小
- f[size]: 浮点数
- o[size]: 八进位
- x[size]:  十六进位





#### cat

cat/tac：文件查看或合并文件内容

-n:  查看文件并给每一行一个编号     

-b:  就不给空行编号   -s：遇到两个以上的空白只显示1行

-sb：不给空行编号而且遇到两个以上的空白只显示1行

连接文件（就是一下打开多个文件查看）：cat  -sb 1.txt 2.txt

* cat与head联用

  ```
  cat -n 文件名 |head -n 20 |tail -n 10
  ```

#### nl

nl: 添加行号打印

-b a : 带空行

-b t : 空行不带序号

-n ：显示位置  ln/rn/rz 左右（z加0）



#### more

more:分页查看文件内容  

> 有时信息过长无法显示在一屏上时，会出现快速滚屏，使得用户无法看清文件，
>
> 此时可以使用more命令，每次只显示一页，
>
> * 按下空格键可以向下翻一页
> * b：往回翻页
> * :f  立即显示文件名和当前所在行数
> * 按下q健退出
> * /字串  ：向下搜寻这个字串
> * 按下h键可以获取帮助

‘+’ 数字：从第几行开始查看，例more +5：从第五行开始    

**-p：先清屏再显示文件内容**   

-s：当两个以上的空行，换成一行空白

Ctrl +F  向下滚动一屏    +B返回上一屏   



#### less

* 空格：向下翻一页
* pagedown / pageup 按键直接翻页
* /字串 ：向下搜寻
* ?字串 ：向上搜寻



#### sort  [文件名]

以行为单位对文本文件进行排序



#### wc

wc [文件名]

-c, --bytes            print the byte counts
  -m, --chars            print the character counts
  -l, --lines            print the newline counts
      --files0-from=文件	从指定文件读取以NUL 终止的名称，如果该文件被
					指定为"-"则从标准输入读文件名
  -L, --max-line-length	显示最长行的长度
  -w, --words			显示单词计数
      --help		显示此帮助信息并退出
      --version		显示版本信息并退出



## 目录配置

按照FHS规则来配置目录 FHS（FileSystem Hierarchy Standard）

![Screenshot_20220317_091850](D:%5CHuawei%20Share%5CHuawei%20Share%5CScreenshot_20220317_091850.jpg)

* / :根目录，和开机系统有关
* /usr : 与软件安装/执行有关
* /var : 与系统运行过程相关

#### /

| 目录   | 放置的文件                                   |
| ------ | -------------------------------------------- |
| /bin   | 放置系统常用可执行文件                       |
| /boot  | 放置开机所需文件                             |
| /dev   | 设备和周边设备                               |
| /etc   | 系统配置文件，如密码，服务的起始档           |
| /lib   | 开机使用的函数库                             |
| /media | 可移动设备                                   |
| /mnt   | 挂载的额外设备                               |
| /opt   | 放置第三方软件                               |
| /run   | 放置开机后产生的信息                         |
| /sbin  | 设置系统环境的的命令                         |
| /srv   | 网络服务产生的数据                           |
| /tmp   | 一般使用者或正在运行的程序暂时放置文件的地方 |
| /usr   | Unix SoftWare Resource                       |
| /var   |                                              |
| /home  | 使用者的主文件夹                             |
| /lib   | 存放二进制函数                               |
| /lost  | 系统发生错误，遗失片段放在这里               |
| /proc  | 虚拟文件系统，记录内存                       |
| /sys   | 虚拟文件系统，记录系统硬件信息               |





#### /var

> 系统运行后，变动的文件产生的信息，会逐渐占用硬盘的目录

| /var/cache/ | 应用程序本身运行过程中产生的一些暂存盘     |
| ----------- | ------------------------------------------ |
| /var/lib/   | 程序本身运行的过程中，需要使用到的文件目录 |
| /var/lock/  | 两个程序同时使用一个程序，会上锁           |
| /var/log/   | 放置登录文件的目录！  /var/log/messages    |
| /var/mail/  |                                            |
| /var/run/   | 程序或服务启动后，放置他们的pid            |
| /var/spool/ | 放置排队等待其他程序使用的数据             |
|             |                                            |



## 磁盘与文件管理

磁盘的组成：

* 圆形的盘片：主要记录数据
* 机械手臂：读写盘片上的数据
* 主轴马达，转动盘片，让机械手臂在盘片上读写数据

磁盘基本概念：

磁盘有一圈一圈的磁道组成

磁针一圈一圈的去读

平均寻道时间：指磁头从得到指令到寻找到数据所在磁道的时间，描述硬盘读取数据的能力

数据传输率： 从硬盘缓存向外输出数据的速度

高速缓存：缓存是数据的临时寄存器，主要用来缓解速度差和实现数据的预存取



## 文件系统

常见文件系统：

> Windows：FAT(FAT16), NTFS
>
> Linux：Ext2 , windows不支持ext2

一个分区就是只能够被格式化为一个文件系统，所以说一个filesystem就是一个分区，但是寻找一个分区可以被格式化为多个文件系统（LVM）



接口类型：



SAS：



文件系统的区块：

* superblock:记录filesystem的整体信息，包括inode/block的总量、使用量、剩余量、以及文件系统的格式与相关信息等
* inode:记录文件的属性，一个文件占用一个inode，同时记录此文件的数据所在的block号码，比如权限、属性放在inode里面
* block: 记录文件的内容，若文件太大，会占用多个block

索引式文件系统：

![Screenshot_20220324_081453](D:%5CHuawei%20Share%5CHuawei%20Share%5CScreenshot_20220324_081453.jpg)

这种数据存取方法称为：索引式文件系统



FAT文件系统：

> 常用的U盘（闪存盘）的文件系统为FAT格式，FAT这种文件格式没有inode存在，所以没有办法一次性把所有block取出来，只能一个一个按照存储的号码记录来读（类似于指针）





![Screenshot_20220324_082705](D:%5CHuawei%20Share%5CHuawei%20Share%5CScreenshot_20220324_082705.jpg)

### 静态分区

安装系统的时候系统管理员要当前某个分区需要的容量，还要预见该分区以后可能需要的容量的最大值

某个分区空间耗尽时：

* 使用符号链接 ---C语言（goto）破坏了Linux的文件系统的标准结构
* 使用调整分区大小的工具 -- 必须停机一段时间进行调整
* 备份整个系统、清除硬重新分区、然后恢复数据到分区





### 文件系统的操作

分区工具：

fdisk <硬盘设备名> 、parted 【】<设备>

#### 磁盘与目录

容量：

df: 列出文件系统的整体磁盘使用量

du：评估文件系统的磁盘使用量

![Screenshot_20220324_084120](D:%5CHuawei%20Share%5CHuawei%20Share%5CScreenshot_20220324_084120.jpg)



![Screenshot_20220324_084346](D:%5CHuawei%20Share%5CHuawei%20Share%5CScreenshot_20220324_084346.jpg)



### 分区、格式化、检验、挂载

* 对磁盘进行分区，以创建可用的partion
* 对该partion进行格式化format，以创建系统可用的filesystem
* 对创建好的filesystem进行检验
* 如果需要创建挂载点（就是目录），并将它挂载上来

lsblk ：列出系统上所有的磁盘列表

* ​	MAJ / MIN : 主要/次要设备代码
* RM: 是否为可卸载设备
* RO: 是否为只读设备
* MOUTPOINT: 挂载点

```
lsblk -ip /目录名   列出该目录下所有完整设备名
```

blkid :列出设备的UUID等参数



#### gdisk 和fdisk 进行分区

![Screenshot_20220324_092358](D:%5CHuawei%20Share%5CHuawei%20Share%5CScreenshot_20220324_092358.jpg)

先lsblk和 blkid找到磁盘 ， 再parted  dev/xxx  print 找到内部分区表的类型 ，再用gdisk和fdisk进行分区

记得退出用q别用w





#### 磁盘格式化（创建文件系统）：

Centos默认使用xfs ， make filesystem 

![Screenshot_20220324_092939](D:%5CHuawei%20Share%5CHuawei%20Share%5CScreenshot_20220324_092939.jpg)

mkfs.ext4



![Screenshot_20220324_093344](D:%5CHuawei%20Share%5CHuawei%20Share%5CScreenshot_20220324_093344.jpg)



#### 文件系统的检验

> 系统运行时发生死机，会发生文件的错乱，还可以挽救

检查与修正文件系统：

```
xds_repair [-fnd] 设备名称

fsck.ext4 [-pf] 设备名称
```



#### 系统的挂载与卸载

> 挂载点就是目录，目录时进入磁盘分区（文件系统）的入口

* 单一文件系统不应该被挂载在不同的挂载点中
* 单一文件系统不应该重复挂载到多个文件系统
* 要作为挂载点的目录，理论上应该是空目录，如果不是空，挂载后原目录下的大小就暂时隐藏起来了，等新挂在的挂载点删除之后，就会显示出来了

挂载命令mount ：

![Screenshot_20220324_094327](D:%5CHuawei%20Share%5CHuawei%20Share%5CScreenshot_20220324_094327.jpg)



卸载：umount  设备文件名或者挂载点

xfs_admin 修改xfs文件系统：

![Screenshot_20220324_094946](D:%5CHuawei%20Share%5CHuawei%20Share%5CScreenshot_20220324_094946.jpg)





#### parted命令

> 可以一行命令完成新建分区

![Screenshot_20220324_101743](D:%5CHuawei%20Share%5CHuawei%20Share%5CScreenshot_20220324_101743.jpg)

```
parted /dev/vda unit mb print  同一start和end的容量单位为mb
```



## LVM

>  LVM是逻辑卷管理的简称，是Linux环境下对磁盘分区进行管理的一种机制。它是建立在硬盘或分区上的一个逻辑层，为文件系统布局，从而提
>
> 高磁盘分区管理的灵活性。

物理卷：在LVM系统最底层

每一个物理卷都会被划分为基本的单元，称为PE ，具有唯一编号的PE是可以被LVM寻址的最小存储单元

* PE的大小可根据实际情况指定，默认为4MB
* PE一旦确定就不能被改变，一个卷组中所有的物理卷的PE的大小必须一致

创建物理卷： pvcreate <磁盘或者分区名>

创建卷组：vgcreate <卷组名> <物理卷设备名>
创建逻辑卷：lvcreate <逻辑卷名> <卷组名>









## 常用命令：

1. 数据流、管道：重定向    例子：echo“ xxxxx”  > helloworld.txt就是将屏幕上的输出重新定向到txt文本文件中

   管道呢就是将上一个命令的输出用到下一个命令的输入：**指令1|指令2   而且指令1必须有输出**

   例子：ls -h \ | more 

2. 建立链接文件：ln    分为软链接（快捷方式）硬链接（多个名字)

   软链接：ln -s  源文件名 链接文件名        硬链接：ln 源文件 链接文件
   
   删除就是  rm  软链接名   但是硬链接删了还能打开源文件
   
3. 文件搜索   grep：在文件内部内容搜索 

   grep [-选项]  ‘搜索内容串’  文件名     例：grep  ‘worker’   test.txt

   -i :不管大小写   -n ：输出内容在哪一行   -v ：取反（不包含你输入的内容）

   ‘搜索内容字符串可以用正则表达式’

4. find：计算机内文件搜索 

   find [路径] 【选项】【名称】

   * 时间参数：

     -[m/a/c]time n : [+/-]n为时间 ， 向前向后

     -newer file :file是文件名，找出比file这个文件更新的文件

   * 使用者或群组参数

     -uid n:  用户id搜索

     -gid n:  群组id搜索
   
     -user/group 名字：找用户拥有的文件
   
     -no[user/group] : 找没有归属者的文件
   
   * 文件权限及名称有关参数
   
     -name 文件名：按照文件名搜寻
   
     -size [+-] 文件大小: 按照文件大小搜寻
   
     -type 文件类型 ：按照文件类型搜寻 
   
     类型：
   
     * 正规文件（f）,设备文件（b,c），目录（d）,链接文件（l），socket（s）
   
     -perm mode :按照权限查找，mode是权限值（如0755）  mode前面啥都不加找权限刚刚好的，加- 权限大于mode的，加/ 找小于mode权限的文件
   
   * 命令的最后可加额外的动作
   
     * -print ：打印
     * -exec ls -l {} \;    :前面find找到的内容会被放到{}里面 ls -l是我进行的操作 ，\;是结束符号
   
   ```
   
   find ./ -name  文件名    find ./ -name ‘*.sh’   搜索以.sh为后缀的文件
   
   find ./ -size +30M  找大于30M的文件  -30M 小于30M的文件 
   
   find ./ -size +15M -size -30M  找15M到30M之间的文件
   ```
   
   
   
   
   
   
   
   


### 打包和压缩

归档和压缩：就是把多个文件打包成一个包

归档管理：

tar [参数]  打包文件名  文件1 文件2    目录归档：tar [参数]   打包文件名   目录

​	-c ：生成档案文件，创建打包文件   -v：列出归档解档的详细过程，显示进度

   -f ：指定档案文件名称  f后面一定是.tar文件，使用必须放选项最后      -x :解开档案文件

归档压缩 tar -zcvf   1.txt  2.txt    解压（多个z）解档：tar -（z）xvf

```
gzip 压缩或解压文件或目录，压缩文件后缀为gz
bzip2 压缩或解压文件或目录，压缩文件后缀为bz2
xz 压缩或解压文件或目录，压缩文件后缀为xz
```



### 系统命令

su：切换用户  passwd：修改用户密码   exit：退出用户登录   who：查看登录用户

sudo：临时提升权限   su root：切换超级管理员

关机重启： 

reboot 重启   

shutdown -r now 立即重启    

shutdown -h now 立刻关机

shutdown -h  20：20 设置时间  

 shutdown -h +10 系统再过10分钟关机   

shutdown -c 取消关机

### 查看系统信息

| 命令          | 信息                   |
| ------------- | ---------------------- |
| LABG=C lscpu  | 显示CPU信息            |
| lsblk         | 显示块设备             |
| lsblk -f      | 显示文件系统           |
| yum repolist  | 查看已启用的软件更新源 |
| timedatectl   | 查看日期和时间         |
| homenamectl   | 显示主机名             |
| ip addr show  | 显示网络接口参数       |
| ip route show | 显示路由信息           |
|               |                        |



### Vim编辑器

> 就是一个文本编辑器，不能对字体、格式、段落进行编排

| 方式                 | 说明                                         |
| -------------------- | -------------------------------------------- |
| vi                   | 默认                                         |
| vi filename          | 新建文件，光标置于行首                       |
| vi n filename        | 打开文件,光标置于第n行行首                   |
| vi  filename         | 打开文件，光标置于行尾                       |
| vi /pattern filename | 打开文件，光标置于第一个与pattern匹配的串处  |
| vi -r filename       | 打开上次用vi编辑时发生系统崩溃的文件，并恢复 |



a,i,o键：进入输入模式   

```
Esc：命令模式   ：进入末行模式   w保存 wq保存并退出

 大写I在行首插入，大写A行尾插入，大写O下一行插入空白行,jj自动取开头，JJ自动末尾

 H：开头 L：末尾 M：中间
```

| 命令 | 说明                           |
| ---- | ------------------------------ |
| i    | 当前位置插入文本               |
| I    | 在行首插入                     |
| a    | 光标当前所在位置之后追加新文本 |
| A    | 行尾插入                       |
| o    | 光标下面新开一行，光标置于行首 |
| O    | 光标上面插入一行并置于行首     |
|      |                                |

#### 光标定位

| 命令 | 说明             |
| ---- | ---------------- |
| n+   | 光标下移n行      |
| n-   | 光标上移n行      |
| 0    | 所在行的行首     |
| $    | 所在行的行尾     |
| ^    | 所在行第一个字符 |
|      |                  |



#### 复制粘贴

| 命令 | 说明                                       |
| ---- | ------------------------------------------ |
| yy   | 当前行复制到缓冲区                         |
| nyy  | 往下n行复制到缓冲区                        |
| yG   | 将当前位置到最后一行的所有内容复制到缓冲区 |
| y1G  | 将当前位置到第一行的所有内容复制到缓冲区   |
| y$   | 光标到当前行末尾                           |
| y0   | 光标到当前行开头                           |
| p    | 粘贴                                       |



#### 字符串搜索

| 命令  | 说明                |
| ----- | ------------------- |
| /str  | 往右移动移动到有str |
| ？str | 往左                |
| n     | 相同方向            |
| N     | 相反方向            |

#### 撤销和重复

| u    | 恢复之前的状态           |
| ---- | ------------------------ |
| .    | 再次执行上一次完成的命令 |







## Shell

> 每一个命令都返回一个0值，命令出错返回一个非0值







## 网络配置

网络接口类型：

| 接口类型           | 名称  | 说明                                                 |
| ------------------ | ----- | ---------------------------------------------------- |
| 以太网接口         | ethX  | 最常用的网络接口                                     |
| 无线网络接口       | whanX | 无线局域网络接口                                     |
| 光纤分布式数据接口 | fddiX | FDDI接口设备昂贵，通常用于核心网或高速网络中         |
| 点对点协议接口     | pppX  | 用于Modem和ADSL拨号网络，或基于PPTP协议的VPN         |
| 本地回环接口       | lo    | 用于支持UNIX Domain Socket 技术的进程相互通信（IPC） |

其中名称中的X就是编号，用来区分鉴别网络接口设备名



网络设备名：

| 前缀名      | 设备                                                    |
| ----------- | ------------------------------------------------------- |
| en          | 以太网设备                                              |
| wl          | 无线局域网设备                                          |
| ww          | 无线广域网设备                                          |
| o           | 主板板载设备                                            |
| s           | 热插拔插槽上的设备                                      |
| p           | PCI总线或USB设备                                        |
| eno16777736 | 板载的以太网设备（设备编号为16777736）                  |
| enp0s8      | PCI接口的以太网设备（PCI总线地址为0，插槽编号为8）      |
| wlp12s0     | PCI接口的无线以太网设备（PCI总线地址为12，插槽编号为0） |



### 配置网络的方法

临时性网络配置：

* 使用ip命名配置
  * ip , idconfig , route , sysctl -w
* 立即生效，但重启后失效

持久性网络配置：

* 使用nmtui/nmcli 配置工具修改网络配置文件
* 使用文本编辑器nano/vim,直接修改网络配置文件



#### ip命令

1.显示网络参数

```
ip addr show / ip a s  显示全部接口的IP地址
ip addr show eno16777736【设备编号】

ip -s link show [设备编号]/ ip -s l s  显示全部接口的传输统计信息
ip route show /ip r s		显示路由信息

ip neighbor show /ip n s
```



2.更改网络

```
ip addr [addr/del] 【ip地址】 dev 【设备名】
#例：/24是子网掩码
ip addr add/del 192.168.140/24 dev eth1

```

​      

3.设置静态路由

```
ip route [add/del] default [地址] via [网关地址] dev [流出设备接口]

ip route add 192.0.2.1 via 10.0.0.1 dev eth0
```





### 手动修改网络配置

> 映射，A->B 一一对应起来

| 配置文件名                            | 功能                                                   |
| ------------------------------------- | ------------------------------------------------------ |
| etc/sysconfig/network-scripts/ifcfg-* | 网络接口配置文件                                       |
| etc/sysconfig/network-scripts/route-* | 网络接口路由配置文件                                   |
| /etc/hostname                         | 本地主机名配置文件                                     |
| /etc/networks                         | 主机名映射为ip地址的解析功能                           |
| /etc/hosts                            | 完成域名与网络地址（网络id）的映射，把这俩一一对应起来 |
| /etc/host.conf                        | 配置域名服务端的控制文件                               |
| /etc/resolv.conf                      | 配置域名服务客户端的配置文件，用于指定域名服务器的位置 |
| /etc/protocols                        | 获知Linux支持的协议和各个协议支持的协议号              |
| /etc/services                         | 获知Linux支持的网络服务及其端口号                      |



网络接口的静态路由配置文件：

```
192.168.2.0/24 via 172.16.10.88    关联之后，用172.16.10.88打开192.168.2.0
```

​      

本地域名解析配置文件：

> 用域名关联ip地址，soho是别名

```
192.168.1.200  soho.mylabs.me  soho  操作后可用 soho打开192.168.1.200
```



远程域名解析器：

```
nameserver:DNS服务器，最多指定3个
nameserver:
nameserver:
domain:指定默认的搜索域
search: 指定默认的搜索域

```



​      

配置域名解析顺序：

```
order hosts,bind
先查找/etc/hosts文件进行域名解析，然后使用/etc/resolv.conf文件指定域名服务
```

​      

### nmcli配置工具





## 网络配置工具

```
ipconfig: 检测网络接口配置
route：检测路由配置
ping：检测网络联通性
ss：查看套接字信息
lsof：查看指定IP和端口进程的运行情况
traceroute：检测到目的主机经过的路由器
dig：测试DNS服务器能否正常工作
```



### 网络测试工具

1.ping命令

```
ping www.sina.con.cn
```

2.ss命令

```
ss -a
ss -l
显示TCP Socket
ss -t
ss -ta
ss -tl
```

​      

3.lsof命令

```
lsof -i:22 
lsof -i@192.168.0.200 察看指定IP使用的端口
```

4.traceroute命令

```
tarceroute www.sina.com.cn
```

5.dig命令

```
dig ls-al.me
dig @202.106.196.115 g.cn  向指定的DNS服务器查询g.cn的IP地址
```



### Internet 网络客户

```
telnet
ftp/lftp/ncftp
smbclient：类似ftp
wget：下载文件、镜像WEB
rsync：远程文件同步
links/w3m/lynx：浏览器
ssh/scp/sftp :远程登录
```

1.lftp命令：

```
lftp [-p port] [-u user] [-e cmd] [site]

-p: 指定连接的端口
-u: 指定的用户名或口令进行FTP身份验证
-e: cmd执行子命令
site: 指定主机名、URL 
```

>  lftp是一个交互式下使用lftp，有非常多子命令可用 ，远程文件目录操作，上传/下载，站点镜像



2.wget命令

> wget是Linux下常用的下载和镜像工具



3.links/w3m 浏览器

> Linux下常用的字符界面浏览器



4.rsync工具

> 远程数据同步工具





### OpenSSH客户

1.ssh命令

```
ssh [-p/v/q] [-l login_name][hostname|[username@]hostname][command]

ssh -l osmond 192.168.0.100  登录 
```



2.scp 命令

```
基于SSH协议在本地主机和远程主机之间复制文件的客户端

scp [-r/p/c] [user@host:remotefile] locfile
scp [-r/p/c] localfile [user@host:remotefile] 
```



3.sftp命令

> 加密传输

```
sftp [user@]host 

sftp osmand@192.168.0.101
```

​      

### RPM包管理

> 更新，安装，卸载，升级包的管理





### YUM更新系统

> 包和包之间往往存在关联性，升级了这个又升级那个
>
> YUM的优点：
>
> * 自动解决包的依赖性问题
> * 便于管理大量系统的更新问题
> * 可以同时配置多个仓库
> * 保持RPM数据库的一致性
> * 有比较详细的日志



### 软件详解

Linux上面的软件几乎都是经过GPL的授权，所以每个软件均提供源码

* 开放源码：就是程序码，写给人类看的程序语言
* 编译器：将程序码转译为机器看得懂的语言
* 可执行文件：经过编译器变成二进制文件后，机器看的懂所以可以执行的文件
* 函数库：类似副程序的角色，可以被调用来执行一段功能函数

Linux软件使用流程：

1. 从官方网站下载程序原始码
2. configure侦测程序，找到所需的函数库、编译器、其他所需资料
3. 由configure主动建立Makefile
4. 系统既有的gcc编译器、函数库、其他依赖软件
5. 按照Makefile的定义、呼叫原始码、编译器来编译
6. 最终软件

Tarball的软件：

原始程序码：就是写满程序码的纯文本

所谓的Tarball文件就是将软件的所有源代码先以tar打包，然后再以压缩技术来压缩

近来bzip2和xz的压缩率较佳，所以bzip2和xz会取代gzip

软件构成：

* 原始程序码
* 侦测程序文件（configure、config等文件名）
* 本软件的简易说明与安装说明（install、readme）



如何安装与升级软件？

升级

* 直接以源代码通过编译来安装与升级
* 直接以编译好的binary program来安装与升级

安装

1. 将Tarball由厂商的网页下载下来
2. 将Tarball解开，会产生很多源代码文件
3. 开始以gcc进行源代码的编译
4. 然后以gcc进行函数库、主、副程序的链接，以形成主要的binary file
5. 将上述的binary file 以及相关的配置文件安装至主机上



#### 如何编辑程序码？

```
gcc hello.c

gcc编译后会产生a.out文件，也就是编译后的二进制文件，a.out是默认文件名

./a.out 运行编译好的二进制文件
```

​      

主程序和副程序的编译：

> 在主程序里面可能会调用另外一个程序

```
比如在thanks.c 里面调用thanks_2.c

此时要进行程序的编译与链接
gcc -o thank thanks.c thanks_2.c

输出的二进制文件名为thank
运行：./thanks
```

   ​    

gcc命令的参数

```
仅将源代码编译成为目标文件

在编译时产生最优化的执行速度
gcc -O -c ... ...  -O:产生最优化参数

编译二进制文件时，将链接的函数库和相关路径填入
gcc sin.c -lm -L/lib -L/lib64  

gcc -O hello hello.c -Wall
-Wall ：产生更详细的编译过程信息，包括warning
```

​      

调用外部函数库：

 函数库放置的地方默认在 /lib，/lib64

例如从库中调用sin.c函数：

```
gcc sin.c -lm -L/lib -L/lib64  

-l:加入某个函数库
m：则是libm.so这个函数库
-L：我要的libm.so这个函数库到 /lib或/lib64里面去找
```

头文件放置目录：

```
例如 #include<stdio.h>  放置在/usr/include/stdio.h 下

如何不使用系统默认读取，命令这么写
gcc sin.c -lm -I/usr/include/stdio.h
```



make命令：

> 如果你要链接多个文件，gcc执行命令要写很长
>
> 而且你如果更改了其中一个文件，这时又要重新编译

```
vim makefile
main: main.o  ... 一系列链接文件
		gcc -o ...

然后执行make
```

* 简化编译时所需要下达的指令
* 若在编译完成后，修改某个源代码文件，则make仅会针对被修改的文件进行编译，其他的不变
* 最后可以按照相依性来更新update可执行文件



makefile语法：

```
主程序名 ：目标文件1 文件2
<tab> 	gcc -o 想创建的可执行文件 目标文件1 文件2

现在有main、和clean两个程序
编译执行：
make main
make clean main
```

避免makefile里面数据太多出错：使用shell script 

```
vim makefile
LIBS = -lm
OBJS = main.0 haha.o
CFLAGS = -Wall
main: ${OBJS}
	gcc -o main ${OBJS} ${LIBS}
clean:
	rm -f main ${OBJS}

```



所以Tarball完整执行步骤：

1. 取原始文件：将文件放在/usr/local/src目录下解压缩
2. 取步骤流程：去查看INSTALL.README
3. 相依属性软件流程
4. 创建makefile :makefile :以自动侦测程序侦测作业环境
5. 编译：make命令编译，makefile里面的参数配置文件
6. 安装：make install 



安装建议：

* 最好将tarball的原始数据解压缩到/usr/local/src当中
* 安装时，最好安装到 /usr/local默认路径下
* 考虑到反安装步骤，最好可以将每个软件单独安装在/usr/local下面
* 为安装到单独目录的软件之 man page 加入man path搜索
  * 在  etc/man_db.conf 内40-50行
  * 写入 MANPATH_MAP/ usr/local/software/bin/usr/local/software/man



利用patch更新源代码：

> 基本语法：patch -p数字 < 更新的文件名
>
> -p数字，就是更新文件里面的第几行信息

更新完记得重新编译