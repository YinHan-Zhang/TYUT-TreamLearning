# Git 学习笔记
[TOC]
## 前置概念:工作区与暂存区
工作区(Working Directory)
电脑中的本地Git仓库就是工作区,本地git仓库就是一个本地目录被指定为git仓库，如何指定，接下来会说

暂存区(Repository)
工作区有一个隐藏的==目录.git==，这个目录是Git的版本库

Git的版本库中存了很多东西，最重要的是暂存区(stage)，还有Git自动创建的第一个分支master,以及指向master的指针叫HEAD

git add实际是把文件修改添加到暂存区
git commit 实际是把暂存区的内容提交到当前分支
![](../..\blog_picture\Git.png)
## 设置git用户名与邮箱：
Command(命令):
==命令都在git bash 中输入==
```python
git config --global user.name ""
git config --global user.email ""
```
参数解释:
使用了--global 这个参数，表示这台机器上所有的Git仓库都会使用这个配置

Response(响应)：
无响应

## 建立git仓库:
前置步骤：
在电脑上选择一个文件夹，并且cd(转换路径)到该文件夹下

Command:
```python
git init
```
命令执行后效果:
init后，目录下会多一个.git目录，用来跟踪管理版本库

## 将文件提交到仓库
前置步骤:
将文件放到仓库目录下

Command:
```python
git add 文件名
```
命令执行后效果:
将文件==添加==到git仓库的暂存区，什么是暂存区之后会说

Command:
```python
git commit -m "message"
```
命令执行后效果:
将文件==提交==到仓库

参数解释:
git -m后面输入的是本次提交的说明，可以输入任何内容

Response:
![](../..\blog_picture\commit.png)
==此处的[dev 8fe6b56]是本次提交的编号，sss是本次提交添加的message，下面一行是本次提交与原来版本的区别==

git添加文件分为add、commit两步的原因:因为commit可以一次提交很多文件，所以可以多次add之后再提交

## 修改文件
前置步骤:
在本地对文件进行修改

Command:
git add 
git commit

辅助Command:
git status

修改文件后使用git status:
Response:
Changes not staged for commit

修改文件且git add 后使用git status:
Response：
Changes to be committed
修改文件且git add、commit后使用:
nothing to commit, working tree clean

辅助Command:
git diff
修改后add前使用:
Response：
与原文件的区别
add后使用:
无响应

## 查看提交历史
Command:
git log

参数解释：
--pretty=oneline可以简化输出信息

Response:
返回提交历史

## 回退版本
Git中用HEAD表示当前版本，上一个版本就是HEAD^,上上一个就是HEAD^^,当然往前太多的话可以用HEAD~n

Command:
git reset --hard HEAD^
or
git reset --hard 版本号(可以只写前几位，Git会自己去找)

参数解释:


回退版本后，再使用git log就无法看到比回退后的版本新的版本了，但可以使用Git reflog查看

Command:
git reflog

## 管理修改
Git比其他版本控制系统优秀的地方在于,它管理的是修改而不是文件

git add每次提交到暂存区的是一次修改，所以如果进行了两次修改，既可以先进行两次修改然后add，也可以进行一次修改，进行一次add

## 撤销修改
Command:
git checkout -- 文件名
效果:
撤销工作区的修改

Command:
git reset HEAD 文件名
效果:撤销版本库中暂存区的修改

Command:
git reset --hard HEAD^
or
git reset --hard 版本号
效果:
撤销已经commit的修改

## 删除文件
前置操作:在工作区中将文件删除
Command:
git rm
git commit

恢复文件，用版本库中的版本替换工作区中的版本
Command:
git checkout -- 文件名

## 远程仓库
Git是版本控制系统，同一个Git仓库可以分布到不同的机器上。

别的机器可以克隆其他机器的版本库。

github需要ssh是因为确保是你本人推送的，github上的仓库内容是任何人都可以看到的，但只有你自己可以修改

### 添加远程库
首先在Github上新建新仓库

可以从这个仓库中clone出新的仓库

也可以将本地的仓库与之关联，之后将本地的内容推送到Github


前置操作：转到要关联的本地仓库下
Command:
git remote add origin git@github.com:Github用户名/github仓库名
参数解释:
origin表示这个命令是远程连接仓库，origin是默认的名字，也可以换其他的名字，@github.com:Github用户名/github仓库名
是要远程连接的仓库所在的用户名与仓库名

然后进行推送
Command:
git push -u origin master
参数解释:
-u,Git不但会把本地的mster分支内容推送到远程新的master分支，并且会把本地得到master分支和远程的master分支联系起来，在以后的推送或者拉取时就可以简化命令

之后，只需要
Command:
git push origin master


分布式版本系统最大的好处之一就是在本地工作时完全不需要考虑远程库的存在


### 解除远程库与本地库的关系
如果需要删除远程库
Command:
git remote rm <name>
Command:
git remote -v 
效果:
查看远程库信息

==这里的删除其实只是解除了本地与远程的绑定关系，并非物理上删除了远程库==

==真正意义上的删除远程库，需要去Github上删除==

### 从远程库clone
Command:
git clone git@github.com:账户名/仓库名

这个地址还可以使用https但是https有些慢而且每次推送都要输入口令

虽然Git支持多种协议，但是ssh协议是最快的

## 分支管理
分支就是同一个项目的不同走向，但可以共存

### 分支管理策略
实际开发中，我们应该按照几个基本原则进行分支管理。

master分支应该非常稳定，用来发布新版本，平时在另一条测试分支上进行工作。

团队中的每个人都可以从测试分支上拉取，推送。稳定版本发布在master上即可。

![](../..\blog_picture\strategy.png)

### 分支合并类型
合并分支时，如果可能，Git会用Fast forward模式，但这种模式下，删除分支后，会丢掉分支信息。

使用--no-ff，合并后的历史有分支，能看出来曾经做过合并，而fast-forward看不出来曾经做过合并

### 创建与合并分支
基本原理：
其实HEAD指向的并不是提交，而是当前分支，当前分支若为master，则如下图所示：
![](../..\blog_picture\HEAD.png)

创建新分支后，可以将HEAD指向新的分支:
![](../..\blog_picture\dev.png)
Command:
git branch <name>
git checkout <name>
git branch
git checkout -b <name>
git switch -c <name>
git switch <name>
效果:
第一条创建新分支
第二条切换到指定分支
第三条列出所有分支，当前分支前加一个*号
第四条：第一条与第二条的结合
第五条：相当于第四条
第六条：切换到已有的分支

合并分支，就是将指定分支的指针指向要合并到的分支:
![](../..\blog_picture\merge.png)
Command：
git merge <name>
效果:
将指定分支切换到当前分支上

合并后，可以删除指定合并的分支：
![](../..\blog_picture\del.png)

Command:
git branch -d <name>

### 解决冲突
当我们在两个分支上都进行了修改，Git就无法快速合并了，而是遇到了冲突。
![](../..\blog_picture\allchanged.png)

此时合并遇到冲突，需要解决冲突后才能合并。

查看文件，可以发现
<<<<<<<，=======，>>>>>>>标记出不同分支的内容，此时修改后就解决了冲突，可以合并了

git log --graph --pretty=oneline --abbrev-commit
也可以查看合并的分支情况

### Bug分支
软件开发中，有了bug就要修复，每个bug可以单独创建一个临时分支来修复，修复后，合并分支。

在创建bug分支前，可以先将现场进行储藏起来，即把还没有提交的文件先保存起来，先去修复bug
Command:
git stash

之后，就可以切换到要修复的分支，然后在这个分支上新建分支，进行修改。修改完成后，去继续完成暂时储存的工作。

Command:
git stash list
效果:
可以查看储存的工作
Command:
git stash pop
git stash apply
git stash drop
效果:
第一条可以恢复工作状态并且删除储存，
第二条只可以恢复工作状态，可以指定恢复某一条
第三条可以删除储存的工作状态

如果我们想要把相同的操作复制到另一个分支上，可以复制提交
前置操作：切换到要粘贴的分支
Command:
git cherry-pick id号
效果:
将提交复制到这个分支

### Feature分支
每添加一个新功能，最好新建一个feature分支，在上面开发，完成后，合并，最后删除分支。

如果这个分支最后不需要合并了，并且需要删除，此时使用Git branch -d <name>，会删除失败，因为还没有合并就删除会丢失修改，此时需要使用以下命令
Command:
git branch -D <name>

## 多人协作
Command:
git remote
git remote -v
git push origin 分支名
git clone 地址
git checkout -b 本地分支名 origin/分支名
git pull
git branch --set-upstream-to <branch-name> <origin/branch-name> 
效果:
第一条查看远程库的状态
第二条查看远程库状态的具体信息
第三条将本地的指定分支推送到远程库的对应分支
第四条从远程库克隆仓库
第五条关联远程库的其他分支，在本地仓库中创建对应分支
第六条从远程库拉取最新的状态
第七条设置本地分支与远程分支的对应关系

## rebase
将原本的分支变基，变成直线简化分支图

## 标签管理
git的标签就像是版本库的一个快照

commit的编号太复杂，不便于使用，所以使用tag
Command:
git tag <tag-name>
git tag <tag-name> <commit-id>
git tag -a <tag-name> -m "description" <commit-id>
git show <tag-name>
git tag
效果:
第一条将当前分支的当前提交设置标签
第二条将指定的提交设置标签
第三条将指定的提交设置标签并加上描述
第四条展示标签对应的提交信息
第五条查看所有的标签

## 操作标签
Command:
git tag -d <name>
git push origin <tag-name>
git push origin --tags
git push origin :refs/tags/<tag-name>
效果：
第一条删除本地标签
第二条推送标签
第三条推送所有标签
第四条从远程删除标签，需要注意的是首先需要从本地删除后，再从远程删除

## 使用Github
Github既是一个免费的远程仓库，又是一个开源项目，放到Github完全没有问题。

点击Fork就可以在自己的账号下克隆一个仓库，然后从自己的仓库下clone

如果从作者的项目clone，那么将没有push权限

如果希望官方接收你的修改，那么就需要在Github上发起pull request

## 自定义Git
Command:
git config --global color.ui true

## Note

1.==切勿随意修改仓库内的文件==

2.
所有的版本控制系统只能跟踪文本文件的改动，对于TXT这些文件，可以知道每次的改动，比如第几行多了什么，但是对于图片、视频这些二进制文件，只能知道变化了多少大小，但不知道具体改变了什么

比如说word就是二进制文件，git便无法跟踪word文件的改动

3.建议使用utf-8编码，被所有平台所支持

4.使用windows需要注意:
不要使用windows自带的记事本编辑任何文本文件，建议使用VScode代替

