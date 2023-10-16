# learn-python-the-smart-way-v2

## - Chapter1

### 1.Comment

两种注释方法：

- 单行注释，使用 `#` 开头
- 多行注释，使用 `'''` 或 `"""` 包裹起来

例如：

```python
# 单行注释
print("Datawhale") # for the learner，和学习者一起成长 

'''
多行注释
Python is powerful... and fast;
plays well with others;
runs everywhere;
is friendly & easy to learn;
is Open.
'''
print("人生苦短，我用 Python")
```

### 2. Console Input & Output

小白要问了：**console是什么？**

**“console”是指计算机中用于与用户进行交互的控制台窗口或命令行界面。**它是一个文本输入输出的终端界面，允许用户输入命令并查看程序输出、错误消息等。

在不同的操作系统中，控制台可能有不同的名称和实现方式。以下是一些常见的控制台名称：

- 在 Windows 系统中，控制台通常称为“命令提示符”或“命令行窗口”。你可以通过在开始菜单中搜索“命令提示符”来找到它。

- 在 macOS 系统中，控制台通常称为“终端”或“终端窗口”。你可以在“应用程序/实用工具”文件夹中找到它。

- 在类 Unix 或 Linux 系统中，控制台通常称为“终端”或“终端窗口”，你可以使用类似于 macOS 的方法来找到它。

在控制台中，你可以执行各种命令来操作计算机和运行程序。控制台是开发人员和高级用户经常使用的工具，因为它提供了直接访问操作系统功能和管理计算机的能力，以及执行各种任务的灵活性。

#### input

例如：红框框里就是我们的console了，我用的vscode+Jupyter环境，所以控制台在这里（跟骆老师的不太一样哦）

![image-20231016195628150](https://s2.loli.net/2023/10/16/JGT3wW5n9UiK8q7.png)

输入我的名字后就会在控制台输出相应的字符串

![image-20231016195643213](https://s2.loli.net/2023/10/16/PEwOpQjAoekKByh.png)

**注意！返回的格式是字符串**

下边这种情况就是错误的，它试图将一个字符串除以2

![image-20231016200258043](https://s2.loli.net/2023/10/16/4tFCB5lXcdNGMI8.png)

这种情况我们做类型转换就可以了

<img src="https://s2.loli.net/2023/10/16/LrSuFtd2PC3VkHQ.png" alt="image-20231016200609005" style="zoom:67%;" />

当我们想要在一行输入多个值时，需要用到`split()`函数

<img src="https://s2.loli.net/2023/10/16/eIVCTfDUHSBkpjR.png" alt="image-20231016201224953" style="zoom: 50%;" />

输出效果如下

<img src="https://s2.loli.net/2023/10/16/OkcfJlUHGbWyaIN.png" alt="image-20231016201237332" style="zoom: 67%;" />



#### output

输出大家都很熟悉了，也是我们会的第一行代码`print()`

* `print()` 的作用是将填入的内容显示在 Console 中，默认每次输入后会换行（等价于按了一次回车，或者 `\n`），控制结尾的参数是 `end`

* `print()` 一次也可以输出多个内容，默认以空格分隔，控制分隔的参数是 `sep`

<img src="https://s2.loli.net/2023/10/16/KYREoSaeyOFWCVl.png" alt="image-20231016201929981" style="zoom: 67%;" />

* 一次输出多行

```python
print("""
Python is powerful... and fast;
plays well with others;
runs everywhere;
is friendly & easy to learn;
is Open.
""")
```

* 这是一个使用 ipywidgets 模块的 Python 代码示例。它通过创建一个交互式小部件，让用户输入两个数字，并计算它们的和。

![image-20231016210622153](https://s2.loli.net/2023/10/16/5wQyazglpCERBnm.png)

### 3.Error

- 语法错误 Syntax Errors，不符合语法规范，代码根本**没有开始运行**
- “运行时”错误 Runtime Errors，代码在**运行过程中出错**，也就是常说的“崩溃”（Crash）
- 逻辑错误 Logical Errors，代码能够运行，且运行过程中没有出错，但是**不是想要的结果**

### 4.Summary

\- 写注释是个好习惯

\- 调整输入输出的参数来控制其呈现效果

\- 大部分错误类型可以归为：语法错误、运行时错误和逻辑错误

\- Python 的库能让很多操作变方便

### 5.Homework

必做题：

- hello, world
- f-string
- Draw Datawhale

提高题：

- The Zen of Python
- Math Floor



这里对提高题进行解读

#### - The Zen of Python

##### **题目描述**

还记得我们在 Chap 0 安装 Installation 提到的作业，“Python 之禅”吗？我们现在来试试来导入它吧。

**输入格式**

无

**输出格式**

输出 Python 之禅

##### 代码如下

```python
import this
```

![image-20231016214132228](https://s2.loli.net/2023/10/16/jFl1Cpxvk6ie8S9.png)

#### - Math Floor

##### 题目描述

在数学中，有一种函数叫做“向下取整”函数，或者说“向下舍入”、“向零取舍”，即 **取不大于 x 的最大整数** ，与“四舍五入”不同，下取整是直接取按照数轴上最接近要求值的左边值，即不大于要求值的最大的那个整数值。

现在给你一个浮点数，你可以 **暂时把它理解成是一个小数** ，要求输出这个浮点数向下取整后的值

关于“向下取整”函数的更多信息可见：[Floor Function](https://mathworld.wolfram.com/FloorFunction.html#:~:text=The floor function %2C also called the greatest,by K. E. Iverson (Graham et al. 1994).)

**输入格式**

一个浮点数（小数）

**输出格式**

输出它的向下取整后的值

**输入样例-1**

```bash
2.5
```

**输出样例-1**

```bash
2
```

**输入样例-2**

```bash
123.456
```

**输出样例-2**

```bash
123
```

##### 代码如下

```python
import ast
import math
a = ast.literal_eval(input())
# 现在程序中有一个名叫 “a” 的浮点数

# 在这行注释下面，编写代码，输出你的答案
print(math.floor(a))
```

![image-20231016215843044](https://s2.loli.net/2023/10/16/YPWyuNqlKb6X2V1.png)