# learn-python-the-smart-way-v2

[TOC]

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



## - Chapter2

> Chapter 2 数据类型和操作 Data Types and Operators

### 1.常用内置类型 Builtin Types

骆师傅在这节给我们讲了基本数据类型：

- 整数 Integer（int）
- 浮点数 Float
- 布尔值 Boolean（bool）
- 类型 Type（是的，“类型”也是种类型！）

![image-20231017225728467](https://s2.loli.net/2023/10/17/yhka93KrNjvCTeJ.png)

其他常见的类型有：

- 字符串 String（str）
- 列表 List
- 元组 Tuple
- 集合 Set
- 字典 Dictionary（dict，或者你可以叫它 `映射 map`）
- 复数 Complex Number（complex)
- 函数 Function
- 模块 Module

```python
print(type("2.2"))       # str (string or text)
print(type([1,2,3]))     # list
print(type((1,2,3)))     # tuple
print(type({1,2}))       # set
print(type({1:42}))      # dict (dictionary or map)
print(type(2+3j))        # complex  (complex number)
print(type(f))           # function
print(type(math))        # module
```

### 2.常用内置常数 Builtin Constants

Python的内置常数是指在Python语言中已经预先定义好的一些常数值，可以直接在代码中使用。以下是一些常用的内置常数：

1. True：表示布尔类型的真值，值为True。
2. False：表示布尔类型的假值，值为False。
3. None：表示一个空值或者空对象。
4. Ellipsis：表示省略号，用于切片操作时表示省略部分。
5. NotImplemented：表示某个方法或操作尚未实现。

另外，还有一些和数学相关的内置常数和函数：

1. pi：表示圆周率 π，约等于 3.14159。
2. e：表示自然对数的底数 e，约等于 2.71828。
3. inf：表示正无穷大，用于表示数值中的正无穷大。
4. nan：表示“Not a Number”，用于表示数值中的非法数值或未定义的结果。
5. abs(x)：返回 x 的绝对值。
6. min(x1, x2, …)：返回参数中的最小值。
7. max(x1, x2, …)：返回参数中的最大值。
8. round(x, n)：将 x 四舍五入到小数点后 n 位。

### 3.常用内置运算符 Builtin Operators

- 算术：`+`, `-`, `*`, `@`, `/`, `//`, `**`, `%`, `-` (一元算符), `+` (一元算符)
- 关系：`<`, `<=`, `>=`, `>`, `==`, `!=`
- 赋值： `+=`, `-=`, `*=`, `/=`, `//=`, `**=`, `%=`
- 逻辑：`and`, `or`, `not`

这里骆师傅重点讲了一下**整除和取模运算**

#### 整除 Integer Division (//)

`/` 指的是**浮点数**除法，它的结果是一个浮点数

`//` 指的是**整除**除法，它的计算结果是整数，舍弃余数

![image-20231017231206554](https://s2.loli.net/2023/10/17/7WjOL6yBlhcbvXE.png)

#### 模运算或余数运算符 (%)

`%` 代表模运算（取余），结果为商的余数

公式如下：
$$
a \mod b \iff a - (a \mid b) \times b
$$
这里要注意对0取模会出错

![image-20231017232333426](https://s2.loli.net/2023/10/17/lXYcprMU6sC2xng.png)

### 4.类型影响语义 Types Affect Semantics

运算符的运作方式会受到**运算数据的类型**的影响

```python
print(3 * 2)  # 输出 6，将整数3与整数2相乘

print(3 * "p2s")  # 输出 p2sp2sp2s，将字符串 "p2s" 重复3次

print(3 + 2)  # 输出 5，将整数3与整数2相加

print("Data" + "whale")  # 输出 Datawhale，将字符串 "Data" 与字符串 "whale" 连接

print(3 + "p2s")  # 报错，无法将整数和字符串相加
```

### 5.短路求值 Short-Circuit Evaluation

逻辑运算参照表

| X     | Y     | X and Y | X or Y | not X | not Y |
| :---- | :---- | :------ | :----- | :---- | :---- |
| True  | True  | True    | True   | False | False |
| True  | False | False   | True   | False | True  |
| False | False | False   | False  | True  | True  |
| False | True  | False   | True   | True  | False |

```python
# 我们先来定义一些函数
def yes():
    return True

def no():
    return False

def crash():
    return 1/0 # 会崩溃！

# 执行下列函数
print(no() and crash()) # 成功运行！

print(crash() and no()) # 崩溃了！

print (yes() and crash()) # 因为上一行崩溃了，所以这行不会被运行，就是运行也会因为短路求值崩溃

```

在第一个输出语句中，由于逻辑运算符 `and` 的短路求值特性，当左侧的表达式 `no()` 返回 `False` 时，右侧的 `crash()` 不会被执行，因此不会发生异常，最终结果为 `False`。

在第二个输出语句中，表达式 `crash()` 由于存在除零操作而导致异常抛出，程序崩溃，代码无法继续向下执行，因此该语句会导致异常。

在第三个输出语句中，由于前面的 `crash()` 引发异常，导致短路求值，不会执行后面的 `yes()` 函数，因此这一行的代码不会被运行。

```python
# 当我们换成or
print(yes() or crash()) # 成功运行

# print(crash() or yes()) # 崩溃了

print(no() or crash())  # 因为上一行崩溃了，所以这行不会被运行，就是运行也会因为短路求值崩溃
```



类似的，当我们换个例子

![image-20231017235553016](https://s2.loli.net/2023/10/17/M2bnRlQD5WmiKsz.png)

这里由于逻辑运算符 `and` 的短路求值特性，第二行只调用了一个函数便输出了`False`。

### 6.type() vs isinstance()

骆师傅在这里讲`type()` 和 `isinstance()`函数，引出了两个观点

- `isinstance()` 比 `type()` 更具有 `稳健性（Robustness）`，具体可以体现在前者包含了**复数**

  ```python
  import numbers
  def isNumber(x):
      return isinstance(x, numbers.Number) # 可以应对任何类型的数字
  ```

- 这种做法更加符合 `面向对象编程` 中 `继承（inheritance）` 的思想

### 7.总结 Summary

- Python 的类型系统很丰富，可以**使用 `type()` 查看对应的类型**
- 常数类型的**值是不可修改的**
- 除法操作**默认是浮点数除法**，整除操作需要使用 `//`
- 运算符之间有运算优先级，运算符作用于不同对象之间的效果是不同的
- 在进行逻辑判断时，会使用**短路求值**

### 8.作业 Homework

必做题：

- Is Number
- Egg Cartons
- Number of Pool Balls

提高题：

- Number of Pool Ball Rows
- Get Kth Digit

这里对提高题进行解读

#### - Number of Pool Ball Rows

##### **题目描述**

本题是 [`Number of Pool Ball Rows`](https://hydro.ac/d/datawhale_p2s/p/P1013) 的相反操作。台球按行排列，其中第一行包含 1 个台球，每行最多可比前一行多包含 1 个台球。例如，3 行最多包含 6 个台球 （1+2+3）。输入一个 `int` 整数 `n`，代表台球总数，要求编写程序，输出 `n` 代表台球的总行数

**输入格式**

一个 `int` 整数 n，代表台球总数。

**输出格式**

一个 `int` 整数，代表台球的总行数.

**输入样例**

```bash
6
```

**输出样例**

```bash
3
```

**解题思路**

跟第三题相反，这题要求通过总台球数n，求出台球的总行数。难点在于如何不用循环，只用算术方法求得行数

在这里我第一次没有AC，原因是没有考虑到4个球的时候应该是摆了3行的，所以应该向上取整，使用`ceil`函数

##### 代码如下

```python
import math
import ast
n = ast.literal_eval(input())
# 现在程序中有一个整数，n

total_rows = math.ceil((math.sqrt(8 * n + 1) - 1) / 2)
print(total_rows)
```

#### - Get Kth Digit

##### **题目描述**

我们会输入 2 个非负的 `int` 整数 `n` 和 `k` 。你需要编写程序，返回整数 `n` 从右开始数的第 `k` 个数字（下标从 `0` 开始）

**输入格式**

2 个非负的 `int` 整数 `n` 和 `k`，以空格分隔。

**输出格式**

1 个 `int` 整型数

**输入样例**

```bash
789,1
```

**输出样例**

```bash
8
```

**解题思路**

本题的难点在于，不允许使用字符串索引（如 `n[k]`）的情况下返回下标为k的整数n

我们可以将数字k看作权值，例如0就是个位，1就是十位，那么问题就可以转化为求整数n的某一位数字了

##### 代码如下

```python
import math
import ast
n, k = ast.literal_eval(input())
# 现在程序中有两个整数，n, k

kth_digit = math.floor(n//10**k)%10
print(kth_digit)
```

这次的题目虽然比较基础，但是不用循环和数组完成也有一定的难度，对小白来说还是很有挑战性的。

## - Chapter3

> Chapter 3 变量与函数 Variables and Functions

### 1.变量 Variables

* A variable is a named value that references or stores a piece of data.
* 变量是一个**标签**，它所指代的是一段数据，使用 `=` 来对这段数据的区域进行**赋值**

变量命名规则：

- **必须以字母或下划线（`_`）开头**
- 命名可由字母、数字和下划线组成
- **大小写敏感**
- 尽量**避免使用保留字**命名

保留字：

```python
import keyword
keyword.kwlist
```

> ['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda',
>
> ...
>
>  'return', 'try', 'while', 'with', 'yield']

多变量赋值：

```python
# 
a = b = c = 2
print(f"a={a}, b={b}, c={c}") # a=2, b=2, c=2
# 解包
a, b, c = 1, 2, 6
print(f"a={a}, b={b}, c={c}") # a=1, b=2, c=6
```

### 2.函数 Functions

> A function is a procedure (a sequence of statements) stored under a name that can be used repeatedly by calling the name.

- 函数是一个名字，代表一串代码序列（流程、过程）
- 函数由两个部分组成：**header**和**body**
  - `header` 用于定义函数接口（函数 **名称** 与 **参数**）
  - `body` 包含函数所需要执行的操作

#### - header

`header` 用于定义函数的**名称**和**参数**

- 当函数被**调用**时，参数将会作为变量被提供给函数的 `body` 部分
- 可以提供多个参数（用逗号 `,` 分隔），也可以不提供参数（0 个）
- `header` 以冒号（`:`）结尾，代表后面会跟着 `body` 部分

#### - body

`body` 包含函数执行的语句（`statement`）

- 语句需要**缩进**（由 Code Style Guide 决定）
- 当语句**不再缩进，函数部分结束**
- 一般会使用 `return` 语句，来让函数返回其结果，但不是必须的

下面是一些函数的例子：

```python
# 调用示例函数 double() 会返回一个值（2 * x）
def double(x):
    print("我在一个名叫 “double” 函数里！")
    return 2 * x

print(double(2)) # 会输出 4

# 无参数
def g():
    return 42

print(g()) # 返回 42

# 三个参数
def f(x, y, z):
    return x + y + z

print(f(1, 3, 2)) # 返回 6
```

值得一提的是，python函数支持多返回值，例如：

```python
# 1.使用元组返回多个值
def get_name_and_age():
    name = "Alice"
    age = 25
    return name, age

result = get_name_and_age()  # 返回一个元组 ('Alice', 25)

name, age = get_name_and_age()  # 使用解包将返回值分配给多个变量
print(name)  # 输出 'Alice'
print(age)  # 输出 25

# 2.使用解包将返回的多个值分配给多个变量
def get_coordinates():
    x = 10
    y = 20
    z = 30
    return x, y, z

x, y, z = get_coordinates()  # 使用解包将返回值分配给多个变量
print(x, y, z)  # 输出 10 20 30
```

### 3.语句与表达式 Statements and Expressions

> An expression is a data value or an operation that evaluates to a value.

对于表达式

- 它本身是**值**
- 它的**计算结果是值**

> Statements, by contrast, do not evaluate to a value, and we can't print them. Usually they perform some action, though.

对于语句

- 它不是值
- 它不能打印
- 但它能**执行一些操作**

```python
# Python 只能 print 值和表达式，如果你能用 print() 输出它，那它就是表达式

print((2 < 3) and (9 > 0)) # True
```

### 4.变量作用域 Variable Scope

- 每个变量都有属于自己的作用范围
- 超出作用范围后，变量不可见

今天的课程中，骆老师重点讲了局部变量和全局变量的区别，并且通过可视化带大家理解函数调用过程

例如在讲全局变量时：

```python
g = 100

def f(x):
    return x + g

print(f(5)) # 105
print(f(6)) # 106
print(g)    # 100
```

![image-20231019214556007](https://s2.loli.net/2023/10/19/nUD4f3CWNOI1pRq.png)

```python
g = 100

def f(x):
    # 如果我们想要修改 g 的值，我们必须声明它是全局变量
    # 否则 Python 会假设它是局部变量
    global g
    g += 1
    return x + g

print(f(5)) # 106
print(f(6)) # 108
print(g)    # 102
```

![image-20231019221821484](https://s2.loli.net/2023/10/19/pxY65GviHmUzqAl.png)

### 5.函数的一些技巧/注意事项 Trick

* 返回语句 Return Statements，一旦返回，函数**立即结束！**

```python
def isPositive(x):
    print("Hello!")   # 会运行
    return (x > 0)
    print("Goodbye!") # 不会运行

print(isPositive(5))  # 输出 “Hello!” 然后返回 True
```

* 没有返回语句的时候，函数会返回 `None`

```python
def f(x):
    result = x + 42

print(f(5)) # None
```

* Helper Functions

```python
def onesDigit(n):
    return n%10

def largerOnesDigit(x, y):
    return max(onesDigit(x), onesDigit(y))

print(largerOnesDigit(134, 672)) # 4
print(largerOnesDigit(132, 674)) # 依然是 4
```

### 6.总结 Summary

- 变量只是个标签，物理设备上有啥才是重点
- 函数定义：`def`、header、body、缩进、`return`
- 函数是有作用域的，类似双层玻璃，里面可以看见外面，外面不能看见里面
- Helper Function 有时候会很有用
- 补充资料：
  - [递归&分治](https://oi-wiki.org/basic/divide-and-conquer/)
  - [Teach Yourself Computer Science](https://teachyourselfcs.com/)

### 7.作业 Homework

必做题：

- Square Root
- Square
- Odd number

提高题：

- Range
- Circles Intersect

这里对提高题进行解读

#### - Range

##### **题目描述**

我们会输入两个数字`x1`和`x2`，你需要编写程序，求解其上界与下界并返回相应的值。

**输入格式**

输入两个`int`整型数，用逗号分隔。

**输出格式**

分别输出下界和上界，中间以空格隔开，具体见输出样例。

**输入样例**

```bash
19,12
```

**输出样例**

```bash
12 19
```

##### **代码如下**

```python
import ast

x1, x2 = ast.literal_eval(input())
# 现在程序中有变量x1和x2

# 在这行注释下面，编写代码，输出你的答案
print(min(x1, x2), max(x1, x2))
```

#### - Circles Intersect

##### **题目描述**

我们会输入 6 个数字 `x1`，`y1`，`x2`，`y2`，`r1`，`r2` 它们代表两个圆，圆心分别为 `(x1, y1)` 和 `(x2, y2)` ，半径分别为 `r1` 和 `r2`。你需要编写程序，判断两个圆是否相交，若相交则返回 `True` ，否则返回 `False`。（相交指两个圆在一个或多个点接触或重叠）。

**输入格式**

六个数字，数字类型为整数或者浮点数，以逗号分隔。

**输出格式**

`True` 或者 `False`，判断两个圆是否相交。

**输入样例**

```bash
0,0,2,3,0,2
```

**输出样例**

```bash
True
```

##### **代码如下**

```python
import ast

x1, y1, r1, x2, y2, r2 = ast.literal_eval(input())
# 现在程序中有六个变量，x1, y1, r1, x2, y2, r2

# 在这行注释下面，编写代码，输出你的答案
distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
print((distance <= (r1 + r2)) and (distance >= abs(r1 - r2)))
```

今天的题目都较为简单，主要是让大家学会使用常用的内置函数解决一些问题。



## - Chapter4

> Chapter 4 条件 Conditionals

### 1.if 语句 

![img](https://s2.loli.net/2023/10/21/gtxylrDK5dBWseQ.jpg)

<center>if语句流程</center>

今天骆老师讲了很多种`abs()`的实现方法，通过实现绝对值函数带着大家理解逻辑判断语句。

> Python 内置了一个函数叫 `abs()` 用于绝对值计算，所以我们将函数命名成 `abs1`、`abs2`……

```python
def abs1(n):
    if n < 0:
        n = -n
    return n
# 写在一行    
def abs2(n):
    if n < 0: n = -n 
    return n
# 多个返回语句
def abs3(n):
    if n < 0:
        return -n
    return n
# 使用布尔表达式
def abs4(n):
    return (n < 0)*(-n) + (n>=0)*(n) # 不建议这样写
```

### 2.if-else 语句

一个例子：

```python
def f(x):
    print("A", end="")
    if x == 0: 
        print("B", end="")
        print("C", end="")
    else:
        print("D", end="")
        if x == 1:
            print("E", end="")
        else:
            print("F", end="")
    print("G")
```

同样的，骆老师为了让大家更直观的理解判断的流程，用可视化的方式让大家看到了代码是如何执行的。

![image-20231022001214976](https://s2.loli.net/2023/10/22/KCBN91rtXpAwGu3.png)

if-else实现方法：

```python
# if-else
def abs5(n):
    if n >= 0:
        return n
    else:
        return -n
# 符号标记
def abs6(n):
    if n >= 0:
        sign = +1
    else:
        sign = -1
    return sign * n
```

### 3.实践小任务

两个都是很经典的题目

任务一：实现一个函数，输入一元二次函数的各项系数，返回其解的个数。

> $$
> 提示：一元二次方程 ax^2+bx+c=0\ (a \neq 0) 的根与根的判别式 有如下关系：
> $$
>
> $$
> \Delta = b^2 - 4 ac
> $$
>
> - 当 Δ>0时，方程有两个不相等的实数根；
> - 当 Δ=0时，方程有两个相等的实数根；
> - 当 Δ<0 时，方程无实数根。

![image-20231022002833772](https://s2.loli.net/2023/10/22/u3ZoAnORtzChMlv.png)

任务二：实现传说中的“学生分数登记管理系统”

![image-20231022002944062](https://s2.loli.net/2023/10/22/OkZaw5doW3neQmK.png)

### 4.if-else 推导式

* 推导式的使用需要满足的条件：可以使得代码更简洁，更容易理解

```python
def abs7(n):
    return n if (n >= 0) else -n

print("abs7(5) =", abs7(5), "and abs7(-5) =", abs7(-5))
```

### 5.match-case 语句

这里骆老师给了大家两个示例带大家理解match-case的使用场景，上边的学生分数登记管理系统就可以使用新特性解决

![image-20231022003720834](https://s2.loli.net/2023/10/22/AeuO6hwaE5f3cVn.png)

[菜鸟教程示例](https://www.runoob.com/python3/python3-conditional-statements.html#:~:text=2 和 3-,match...case,-Python 3.10 增加)

[HTTP 响应状态码](https://developer.mozilla.org/zh-CN/docs/Web/HTTP/Status)

**一个小的随堂练习**

**背景：**小 ϵ 是一名大四学生，他的学校毕业要求是通过大学英语六级考试，你能写个程序看看他能不能毕业嘛？

**输入格式**：1-2 个整数，以空格分隔，第一个数字代表 CET 4 成绩，第二个数字代表 CET 6 成绩，如果四级未通过则没有六级成绩。

**输出格式：**1 个字符串，`Yes` 代表能够毕业，`No` 代表不能毕业。

**输入示例：**

> 500 430

**输出示例：**

> Yes

**代码如下：**

```python
# if-else方法
def isGraduate(cet4, cet6):
    if cet4 >= 425 and cet6 >= 425:
        return "Yes"
    else:
        return "No"

cet4, cet6 = input().split()
cet4 = float(cet4)
cet6 = float(cet6)

isGraduate(cet4, cet6)
```

### 6.清晰的代码风格 Clarity and style

* 一个好的代码风格对于程序员来说非常重要，要让代码易于理解，简洁明确

例如：

```python
# 1.否定条件（带有else部分）

# 可能不清晰的:
b = True
if not b:
    print('no')
else:
    print('yes')
    
# 清晰的:
b = True
if b:
    print('yes')
else:
    print('no')
    
# 2.空白的if部分

# 不清晰的:
b = False
if b:
    pass
else:
    print('no')
    
# 清晰的:
b = False
if not b:
    print('no')
    
# 3.用嵌套的 if 而不是 and 来判断

# 混乱的:
b1 = True
b2 = True
if b1:
    if b2:
        print('both!')
# 清晰的:
b1 = True
b2 = True
if b1 and b2:
    print('both!')

# 4.使用 if 而不是 else 来控制

# 又混乱又有产生 bug 的风险:
x = 10
if x < 5:
    print('small')
if (x >= 5) and (x < 10):
    print('medium')
if (x >= 10) and (x < 15):
    print('large')
if x >= 15:
    print('extra large')
    
# 更好的做法:
x = 10
if x < 5:
    print('small')
elif x < 10:
    print('medium')
elif x < 15:
    print('large')
else:
    print('extra large')
```

### 7.总结 Summary

- Conditionals Make Decisions.
- if-else 结构构成了 Python 分支控制，if 还能嵌套使用。
- 合理的编写风格会让代码更易读，还能尽可能避免引入 bug。

### 8.作业 Homework

必做题：

- Output Letter Grade by Score
- getInRange
- Is Point Inside Square

提高题：

- Check Leap Year
- Days in Month

这里对提高题进行解读

#### - Check Leap Year

**题目描述**

输入一个年份，判断这一年是否是闰年，如果是输出 `True`，否则输出 `False`。

**输入格式**

输入一个正整数 `x`，表示年份。

**输出格式**

输出一行。如果输入的年份是闰年则输出 `True`，否则输出 `False`。

**样例输入-1**

```bash
1944
```

**样例输出-1**

```bash
True
```

**样例输入-2**

```bash
1900
```

**样例输出-2**

```bash
False
```

**解题思路**

闰年：

- **四年一闰百年不闰**：即如果year能够被4整除，但是不能被100整除，则year是闰年。
- **每四百年再一闰**：如果year能够被400整除，则year是闰年。

代码如下：

```python
x = int(input())
# 现在程序中有一个变量x

# 在这行注释下面，编写代码，输出你的答案


def is_leap_year(year):
    if year % 4 == 0 and year % 100 != 0:
        return True
    elif year % 400 == 0:
        return True
    else:
        return False


print(is_leap_year(x))
```

#### - Days in Month

**题目描述**

输入年份和月份，输出这一年的这一月有多少天。需要考虑闰年。

**输入格式**

输入两个正整数，分别表示年份 *y* 和月数 *m*，以`,`隔开。

**输出格式**

输出一行一个正整数，表示这个月有多少天。

**样例输入-1**

```bash
1926,8
```

**样例输出-1**

```bash
31
```

**样例输入-2**

```bash
2000,2
```

**样例输出-2**

```bash
29
```

**解题思路**

* 先考虑特殊情况，2月在闰年有29天，平年28天，所以先判断是不是闰年
* 其次考虑其他月份，1, 3, 5, 7, 8, 10, 12月有31天，其余月份30天

**代码如下**

```python
import ast

y, m = ast.literal_eval(input())
# 现在程序中有两个变量y, m

# 在这行注释下面，编写代码，输出你的答案

# 根据年份和月份判断天数
if m == 2:
    if y % 4 == 0 and y % 100 != 0 or y % 400 == 0:
        days_in_month = 29
    else:
        days_in_month = 28
elif m in [1, 3, 5, 7, 8, 10, 12]:
    days_in_month = 31
else:
    days_in_month = 30

print(days_in_month)
```

## - Chapter5

> Chapter 5 循环 Loop

### 1.for 循环

**for 循环的特点**：基于提供的范围，**重复执行特定次数**的操作

for循环：for循环用于遍历一个可迭代对象（如列表、元组、字符串等），按顺序执行其中的语句块。语法如下：

```python
for 变量 in 可迭代对象:
    执行的语句块
```

例子：

```python
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)
```

输出：

```
apple
banana
cherry
```

![image-20231024182520429](https://s2.loli.net/2023/10/24/w8Jv6QfcFVLzbrS.png)

这里扩展了range函数，并且强调了左闭右开的特性

### 2.for 循环嵌套

这里通过for循环嵌套输出二维数组

![image-20231024213629252](https://s2.loli.net/2023/10/24/qWEHesiZvKNXV3l.png)

### 3.while 循环

while循环：while循环根据一个条件表达式的真假来循环执行一段代码。只要条件为真，循环就会一直进行。语法如下：

```
while 条件表达式:
    执行的语句块
```

例子：

```python
count = 1
while count <= 5:
    print(count)
    count += 1
```

输出：

```
1
2
3
4
5
```

* 当你**不知道循环什么时候停下来**的时候，可以试试 **`while`**

例如：找出一个数最左边的那一位的数值（`12345` 的 `1` ）

```python
# 我不知道它什么时候停下来

def leftmostDigit(n):
    n = abs(n)
    while n >= 10:
        n = n//10
    return n 
    
leftmostDigit(46535248) # 4
```

### 4.break 与 continue 语句

```python
for n in range(200):
    if n % 3 == 0:
        continue # 跳过这次循环
    elif n == 8:
        break # 跳出当前整个循环
    else:
        pass # 啥也不做，占位符（不会被运行）
    print(n, end=" ")

# 1 2 4 5 7 
```

### 5.假·死循环

与环境交互后，在特定条件下终止的循环

```python
# 不需要看懂这些代码，关注演示的过程

def readUntilDone():
    linesEntered = 0
    while True:
        response = input("输入一个字符串（输入 done 则退出）: ")
        if response == "done":
            break
        print("你输入了: ", response)
        linesEntered += 1
    print("Bye!")
    return linesEntered
    
linesEntered = readUntilDone()
print("你输入了", linesEntered, "行 (不包括 'done').")
```

![image-20231024232227096](https://s2.loli.net/2023/10/24/jZYTM3KsRi7PnOg.png)

### 6.循环的效率

关于循环的效率，骆老师通过判断质数的不同方法带大家体验代码的执行效率

**isPrime 判断一个数是不是质数**

```python
# 不是最快的写法，但最容易理解

def isPrime(n):
    if n < 2:
        return False
    for factor in range(2,n):
        if n % factor == 0:
            return False
    return True

# 快了一点

def fasterIsPrime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    maxFactor = round(n**0.5)
    for factor in range(3, maxFactor+1, 2):
        if n % factor == 0:
            return False
    return True

# 验证他它俩结果是一样的
# assert是Python中的断言语句，用于检查程序的某个条件是否为真。

for n in range(100):
    assert(isPrime(n) == fasterIsPrime(n))
print("两种解法的结果一致")
```

![image-20231025000622974](https://s2.loli.net/2023/10/25/MANt18G7fxk645d.png)

很明显，代码效率有了很大提升

### 7.总结 Summary

- For 循环用于指定范围的重复操作。
- `range()` 可以生成一个数字范围。
- 在不知道循环什么时间停止的时候，应该试试 While 循环。
- 循环同样也是可以嵌套的。
- 巧妙地使用 `break` 和 `continue` 语句。
- 合理的剪枝，缩小搜索范围/循环范围，可以大幅提高程序运行效率。

### 8.作业 Homework

必做题：

- N*M Matrix
- ReverseNumber
- hasConsecutiveDigits

提高题：

- nthPalindromicPrime
- carrylessAdd

这里对提高题进行解读

#### - nthPalindromicPrime

**题目描述**  

**质数**（Prime number），又称**素数**，指在大于1的自然数中，除了1和该数自身外，无法被其他自然数整除的数（也可定义为只有1与该数本身两个因数的数）。**回文数**或**迴文数**是指一个像14641这样“对称”的数。同时满足**质数**和**回文数**的条件的数被称为回文素数。在本题中我们会输入一个整型数`n`， 你需要编写程序，来返回第n个回文素数的值。前十个回文素数如下：2, 3, 5, 7, 11, 101, 131, 151, 181, 191。

**输入格式**

一个整型数`n`。

**输出格式**

一个整型数。

**输入样例**

```none
0
```

**输出样例**

```none
2
```

**提示说明**

- 本题中 `n`的范围为：[0 , 21)
- 供参考的**回文素数表**如下：2, 3, 5, 7, 11, 101, 131, 151, 181, 191, 313, 353, 373, 383, 727, 757, 787, 797, 919, 929, 10301, 10501, 10601, 11311；
- 本题可以先使用`is_prime(num)`函数来判断一个数是否为质数，然后使用`is_palindrome(num)`函数比较、判断是否为回文数。

**代码如下**

```python
import ast

n = ast.literal_eval(input())
# 现在程序中有一个变量n
# 在这行注释下面，编写代码，输出你的答案


def is_prime(num):
    """判断一个数是否是质数"""
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True


def is_palindrome(num):
    """判断一个数是否是回文数"""
    reversed_num = 0
    original_num = num
    while num > 0:
        curr_digit = num % 10
        reversed_num = reversed_num * 10 + curr_digit
        num = num // 10
    return reversed_num == original_num


def get_nth_palindrome_prime(n):
    """获取第n个回文素数的值"""
    count = 0
    num = 2

    # 不断递增num，直到找到第n个回文素数为止
    while count < n:
        # 判断num是否是回文素数
        if is_prime(num) and is_palindrome(num):
            count += 1
            if count == n:
                return num
        num += 1


print(get_nth_palindrome_prime(n+1))
```

#### - carrylessAdd

**题目描述**

众所周知，我们常见的加法规则是类似与 8 + 7 = 15这种，但是现在我们需要设计一种全新的加法运算规则：忽略进位的加法计算。例如输入18和27，答案会是35，而非正常的45。输入两个正整数`x1`和`x2`，返回此方法下计算后的结果。

**输入格式**

两个整型数`x1`和`x2`，用逗号隔开。

**输出格式**

一个整型数。

**输入样例**

```none
785,376
```

**输出样例**

```none
51
```

**解题思路**

当我们需要对两个整数进行无进位相加时，可以分别取出它们的个位进行相加，然后将相加结果乘以 10 的相应次方累加起来，以得到最终的结果。

**代码如下**

```python
import ast

x1, x2 = ast.literal_eval(input())
# 现在程序中有两个变量x1, x2
# 在这行注释下面，编写代码，输出你的答案

def carrylessAdd(x1, x2):
    # 设定一个初始值 result 和一个乘数 multiplier
    result = 0
    multiplier = 1
    
    while x1 > 0 or x2 > 0:
        sum_bits = (x1 % 10 + x2 % 10) % 10
        result += sum_bits * multiplier
        multiplier *= 10
        x1 //= 10
        x2 //= 10
        
    return result

print(carrylessAdd(x1, x2))
```

## - Chapter6

> Chapter 6 字符串 Strings

### 1.字符串文字

#### 四种引号

在这里骆老师介绍了不同引号的作用。 

引号的作用就是将文字包裹起来，告诉 Python "这是个字符串！"

单引号 `'` 和双引号 `"` 是最常见的两种字符串引号

```python
print('单引号')    # 单引号
print("双引号")	# 双引号
```

三个引号的情况不太常见，但是它在一些场合有特定的作用（如函数文档 doc-strings）

```python
print('''三个单引号''') # 三个单引号
print("""三个双引号""") # 三个双引号
```

**我们为什么需要两种不同的引号？**

```python
# 为了写出这样的句子
print("聪明办法学 Python 第二版的课程简称是 'P2S'")
# 聪明办法学 Python 第二版的课程简称是 'P2S'
```

如果我们偏要只用一种引号呢？

```python
# 这会导致语法错误，Python 无法正确判断一个字符串的终止位置
print("聪明办法学 Python 第二版的课程简称是 "P2S"")
  Cell In [4], line 2
    print("聪明办法学 Python 第二版的课程简称是 "P2S"")
                                   ^
SyntaxError: invalid syntax
```

#### 字符串中的换行符号

前面有反斜杠 `\` 的字符，叫做**转义序列**，反斜杠 `\` 叫做转义字符

 `\n` 代表**换行**，尽管它看起来像两个字符，但是 Python 依然把它视为一个特殊的字符

![image-20231027162324021](https://s2.loli.net/2023/10/27/6Oqe7h4GFb3WAmd.png)

 `\n` 还可以用来排除后面的换行。

![image-20231027162548606](https://s2.loli.net/2023/10/27/9spuLQTqkfGvAmE.png)

**其他的转义序列**

```python
print("双引号：\"")  # 双引号："

print("反斜线：\\")  # 反斜线：\

print("换\n行")  
换
行

print("这个是\t制\t表\t符\n也叫\t跳\t格\t键")
这个是	制	表	符
也叫	跳	格	键
```

**转义序列只作为一个字符存在**

```python
s = "D\\a\"t\ta"
print("s =", s)
print("\ns 的长度为：", len(s))
```

```
s = D\a"t	a

s 的长度为： 7
```

#### repr() vs. print()

`repr()` 和 `print()` 都是 Python 的内置函数，但它们的作用略有差别。

`repr()` 的作用是返回一个对象的字符串表示形式，它通常被用于开发和调试，以及在可能需要重新创建该对象时将对象的表示形式输出到控制台或日志文件中。

`print()` 的作用是将对象输出到控制台或其他输出流中，它通常被用于向用户显示输出。它会将对象转换成一个字符串并在控制台输出，但该字符串会使用特定的分隔符和结尾标志。例如：

```python
s1 = "Data\tWhale"
s2 = "Data        Whale"

print("s1:", s1) # s1: Data        Whale
print("s2:", s2) # s2: Data        Whale

s1 == s2	# 输出False
```

为什么不一样呢？我们输出一下试试

```
print(repr(s1))
print(repr(s2))
```

![image-20231027163652400](../../../../AppData/Roaming/Typora/typora-user-images/image-20231027163652400.png)

**在这里引出了一个 python2 安全性的问题**

如果遇到恶意代码时，我们可以使用repr来输出隐藏的代码

```python
hack_text = "密码应当大于 8 个字符，小于 16 个字符，包含大写字母、小写字母、数字和特殊符号\t\t\t\t\t\t\t\t\t\t\t\t\t"
```

![image-20231027164222126](https://s2.loli.net/2023/10/27/Zqbw9unWyaPDfQi.png)

多行字符串作为注释

```python
"""
Python 本身是没有多行注释的，
但是你可以用多行字符串实现同样的操作，
还记得我们之前学过的“表达式“吗？
它的原理就是 Python 会运行它，
但是马上扔掉！（垃圾回收机制）
"""
print("Amazing!")
```

### 2.一些字符串常量

![image-20231027164939608](https://s2.loli.net/2023/10/27/xFAcMNLDB2ilYT6.png)

### 3.一些字符串的运算

#### 基本运算

* 字符串的加减

```python
print("abc" + "def")
print("abc" * 3)

abcdef
abcabcabc
```

```bash
print("abc" + 3)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In [31], line 1
----> 1 print("abc" + 3)

TypeError: can only concatenate str (not "int") to str
```

* `in` 运算（超级好用！）

```python
print("ring" in "strings") # True
print("wow" in "amazing!") # False
print("Yes" in "yes!") # False
print("" in "No way!") # True
print("聪明" in "聪明办法学 Python") # True

# 输出
True
False
False
True
True
```

#### 字符串索引和切片

* 单个字符索引  索引可以让我们在特定位置找到一个字符

```python
s = "Datawhale"
print(s)
print(s[0])
print(s[1])
print(s[2])
print(s[3])

# 输出
Datawhale
D
a
t
a
```

```python
len(s)  # 9
print(s[len(s)-1])  # e
```

* 用切片来获取字符串的一部分

```python
print(s[0:4])
print(s[4:9])

# 输出
Data
whale
```

* 切片的默认参数

```python
print(s[:4])
print(s[4:])
print(s[:])

# 输出
Data
whale
Datawhale
```

* 切片的第三个参数 `step`

```python
print(s[:9:3])
print(s[1:4:2])

# 输出
Daa
aa
```

* 翻转字符串

```python
# 实在是太优雅辣
def reverseString(s):
    return s[::-1]

print(reverseString(s))

# 输出
elahwataD
```

### 4.字符串的循环

用索引的 for 循环

```python
for i in range(len(s)):
    print(i, s[i])
```

其实也可以不用索引（超级好用的 `in`）

```python
for c in s:
    print(c)
```

![image-20231027215650607](https://s2.loli.net/2023/10/27/qaL2TYbcFPRZiw4.png)

也可以使用 `enumerate()` 获得元素的序号

```python
for idx, c in enumerate(s):
    print(idx, c)
```

`zip(a, b)` 可以在一次循环中，分别从 `a` 和 `b` 里同时取出一个元素

```python
for a, b in zip(s, reverseString(s)):
    print(a, b)
```

![image-20231027221350347](https://s2.loli.net/2023/10/27/XW1o9vCxUIOekdH.png)

**例子：回文判断**

如果一个句子正着读、反着读都是一样的，那它就叫做“回文”。在这里骆老师用四种不同的方法实现了回文判断。

```python
def isPalindrome1(s):
    return (s == reverseString(s))
```

```python
def isPalindrome2(s):
    for i in range(len(s)):
        if (s[i] != s[len(s)-1-i]):
            return False
    return True
```

```python
def isPalindrome3(s):
    for i in range(len(s)):
        if (s[i] != s[-1-i]):
            return False
    return True
```

```python
def isPalindrome4(s):
    while (len(s) > 1):
        if (s[0] != s[-1]):
            return False
        s = s[1:-1]
    return True
```

### 5.一些跟字符串相关的内置函数

* `str()` 和 `len()`

```python
name = input("输入你的名字: ")
print("Hi, " + name + ", 你的名字有 " + str(len(name)) + " 个字！")
```

* `chr()` 和 `ord()`

```python
print(ord("A"))  # 65
print(chr(65))   # A
print(chr(ord("A") + ord(" ")))  # a
```

* `eval()`

`eval()` 是 Python 的一个内置函数，它用于将字符串作为 Python 表达式来执行。

```python
def 电脑当场爆炸():
    from rich.progress import (
        Progress, 
        TextColumn, 
        BarColumn, 
        TimeRemainingColumn)
    import time
    from rich.markdown import Markdown
    from rich import print as rprint
    from rich.panel import Panel


    with Progress(TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeRemainingColumn()) as progress:
        epoch_tqdm = progress.add_task(description="爆炸倒计时！", total=100)
        for ep in range(100):
            time.sleep(0.1)
            progress.advance(epoch_tqdm, advance=1)

    rprint(Panel.fit("[red]Boom! R.I.P"))
    
s = "电脑当场爆炸()"
eval(s) # 如果这是一串让电脑爆炸的恶意代码，那会发生什么
```

![image-20231027233008815](https://s2.loli.net/2023/10/27/1jygMXxckLSU4o8.png)

为了避免不安全的情况出现，我们推荐使用以下方法

![image-20231027233052656](https://s2.loli.net/2023/10/27/OYt4IQCgHkeWFwz.png)

### 6.一些字符串方法

```python
def p(test):
    print("True     " if test else "False    ", end="")
def printRow(s):
    print(" " + s + "  ", end="")
    p(s.isalnum())
    p(s.isalpha())
    p(s.isdigit())
    p(s.islower())
    p(s.isspace())
    p(s.isupper())
    print()
def printTable():
    print("  s   isalnum  isalpha  isdigit  islower  isspace  isupper")
    for s in "ABCD,ABcd,abcd,ab12,1234,    ,AB?!".split(","):
        printRow(s)
printTable()
```

![image-20231027233405648](https://s2.loli.net/2023/10/27/CQoFhumgDTWkcSG.png)

其他的一些方法：

![image-20231027233638969](https://s2.loli.net/2023/10/27/vRSW6e7lX35Tcwq.png)

![image-20231027233654123](https://s2.loli.net/2023/10/27/vjQoD6beZtF2R3Y.png)

### 7.格式化字符串

* 用 `f-string` 格式化字符串

```
x = 42
y = 99

print(f'你知道 {x} + {y} 是 {x+y} 吗？')
```

Python 还有其他方法去格式化字符串：

- `%` 操作
- `format()` 方法

参考资料：

- [Class Notes: String Formatting - CMU 15-112](http://www.cs.cmu.edu/~112-f22/notes/notes-string-formatting.html)
- [Python 字符串 - 菜鸟教程](https://www.runoob.com/python3/python3-string.html#:~:text=\n \n-,Python 字符串格式化,-Python 支持格式化)
- [Python format 格式化函数 - 菜鸟教程](https://www.runoob.com/python/att-string-format.html)

### 8.注意事项

* 字符串是不可变的

![image-20231027234004767](https://s2.loli.net/2023/10/27/mJ1EirjXKf8Nbyc.png)

* 字符串是不可变的，所以它的别名也是不可变的

![image-20231027234020812](https://s2.loli.net/2023/10/27/HS8Q1hnpFebL2tB.png)

### 9.基础文件操作

**`Open()` 函数**

Python `open()` 函数用于打开一个文件，并返回文件对象，在对文件进行处理过程都需要使用到这个函数。

`open(file, mode)` 函数主要有 `file` 和 `mode` 两个参数，其中 `file` 为需要读写文件的路径。`mode` 为读取文件时的模式，常用的模式有以下几个：

- `r`：以字符串的形式读取文件。
- `rb`：以二进制的形式读取文件。
- `w`：写入文件。
- `a`：追加写入文件。

不同模式下返回的文件对象功能也会不同。![image-20231027234148648](https://s2.loli.net/2023/10/27/C47JEVjHhu6xcnG.png)

**文件对象**

`open` 函数会返回一个 文件对象。在进行文件操作前，我们首先需要了解文件对象提供了哪些常用的方法

- `close( )`: 关闭文件
- 在r与rb模式下：
  - `read()`: 读取整个文件
  - `readline()`: 读取文件的一行
  - `readlines()`: 读取文件的所有行
- 在w与a模式下：
  - `write()`:写入文本内容或字节到文件中
  - `writelines()`:写入文本所有行

**需要注意的是，在使用文件操作完成后，需要调用 `close()` 方法来关闭文件，以释放系统资源**  

这时候，**with()函数**就显得非常好用了

> 我不想写 close() 啦！

祭出我们经典的python之禅

![image-20231028004155551](https://s2.loli.net/2023/10/28/RH61GzgQN3cn7TX.png)

这时候我们可以用**with函数**来操作文件了

![image-20231028004316759](https://s2.loli.net/2023/10/28/6ZSqDais5f3ePV2.png)

### 10.总结 

- 单引号与双引号要适时出现，多行文本用三引号。
- 字符串中可以包含转义序列。
- `repr()` 能够显示出更多的信息。
- 字符串本身包含许多内置方法，`in` 是一个特别好用的玩意。
- 字符串是不可变的常量。
- 文件操作推荐使用 `with open("xxx") as yyy`，这样就不用写 `f.close()` 啦。

### 11.作业

必做题：

- Find You
- isPalindrome
- New String

提高题：

- Upper and Lower
- Rotate sentence please

#### - Rotate sentence please

**题目描述**

输入一串句子(不含标点符号)，单词与单词之间以空格隔开，要求编写程序，将单词顺序进行翻转，但不翻转每个单词的内容。

**输入格式**

一串非空字符串，包含数个单词，中间以空格隔开，不含标点符号。

**输出格式**

翻转后的句子。

**输入样例**

```bash
This works but is confusing
```

**输出样例**

```bash
confusing is but works This
```

**解题思路**

要将单词顺序进行翻转，但不翻转每个单词的内容，我们可以联想到用之前的`reverse()`方法，但是不同的是，这次需要保留单词的含义，从空格处分开

**代码如下**

```python
str1 = input()
# 现在程序中有一个变量，名为str1

# 在这行注释下面，编写代码，输出你的答案

words = str1.split(" ")
words.reverse()
result = " ".join(words)

print(result)
```

我们这里可以联想到后进先出的栈（stack）来完成题目，使用栈的方法通常会比使用列表的 `reverse()` 方法更高效。原因是使用列表的 `reverse()` 方法需要先将整个列表反转，然后再使用 `join()` 方法进行连接，而这涉及到更多的内存操作。使用栈的方法只需要将单词逐个入栈，然后再按照出栈顺序进行连接。这个过程不需要额外的内存操作和列表反转。因此，从效率上来说，使用栈的方法更高效。

```python
str1 = input()
# 现在程序中有一个变量，名为str1

# 在这行注释下面，编写代码，输出你的答案

words = str1.split()
stack = []

for w in words:
    stack.append(w)

result = ""
while stack:
    result += stack.pop() + " "

print(result.strip())
```

------

到此为止，**《聪明办法学 Python 第二版》基础部分的全部学习内容**就算学完啦，期待Datawhale聪明办法学 Python 团队对进阶版的更新！

## - 结课竞赛

### A. 成绩

**题目描述**

小鲸鱼最近学习了 Datawhale 的聪明办法学 Python 线下课程，这门课程的总成绩计算方法是：

总成绩 = 作业成绩 ×20%+×20%+ 小测成绩 ×30%+×30%+ 期末考试成绩 ×50%×50%

小鲸鱼想知道，这门课程自己最终能得到多少分。

**输入格式**

三个非负整数 A*,*B*,*C，分别表示小鲸鱼的作业成绩、小测成绩和期末考试成绩。相邻两个数之间用一个空格隔开，三项成绩满分都是100 分。

**输出格式**

一个整数，即小鲸鱼这门课程的总成绩，满分也是100分。

**输入数据 1**

```input1
100 100 80
```

**输出数据 1**

```output1
90
```

**输入数据 2**

```input2
60 90 80
```

**输出数据 2**

```output2
79
```

**代码如下**：

```python
A, B, C = map(int, input().split(" "))

total_grade = int(A * 0.2 + B * 0.3 + C * 0.5)
print(total_grade)
```

------

### B. 小鲸鱼的游泳时间

**题目描述**

亚运会开始了，小鲸鱼在拼命练习游泳准备参加游泳比赛，可怜的小鲸鱼并不知道鱼类是不能参加人类的奥运会的。

这一天，小鲸鱼给自己的游泳时间做了精确的计时（本题中的计时都按 24 小时制计算），它发现自己从a时b分一直游泳到当天的 c时 d分，请你帮小鲸鱼计算一下，它这天一共游了多少时间呢？

小鲸鱼游的好辛苦呀，你可不要算错了哦。

**输入格式**

一行内输入四个整数，以空格隔开，分别表示题目中的 a*,*b*,*c*,*d。

**输出格式**

一行内输出两个整数*e* 和 *f*，用空格间隔，依次表示小鲸鱼这天一共游了多少小时多少分钟。其中表示分钟的整数 *f* 应该小于 6060。

对于全部测试数据0≤*a*,*c*≤24，0≤*b*,*d*≤60，且结束时间一定晚于开始时间

**输入数据 1**

```input1
12 50 19 10
```

**输出数据 1**

```output1
6 20
```

**代码如下**：

```python
a, b, c, d = map(int, input().split(" "))

total_minutes_start = a * 60 + b
total_minutes_end = c * 60 + d

swim_time = total_minutes_end - total_minutes_start

e = swim_time // 60
f = swim_time % 60

print(e, f)
```

------

### C. 不高兴的小鲸鱼

**题目描述**

小鲸鱼上初中了。鲸鱼妈妈认为小鲸鱼应该更加用功学习，所以小鲸鱼除了上学之外，还要参加鲸鱼妈妈为它报名的各科复习班。另外每周鲸鱼妈妈还会送它去学习朗诵、舞蹈和钢琴。但是小鲸鱼如果一天上课超过八个小时就会不高兴，而且上得越久就会越不高兴。假设小鲸鱼不会因为其它事不高兴，并且它的不高兴不会持续到第二天。请你帮忙检查一下小鲸鱼下周的日程安排，看看下周它会不会不高兴；如果会的话，哪天最不高兴。

**输入格式**

输入包括 7行数据，分别表示周一到周日的日程安排。每行包括两个小于10 的非负整数，用空格隔开，分别表示小鲸鱼在学校上课的时间和鲸鱼妈妈安排它上课的时间。

**输出格式**

一个数字。如果不会不高兴则输出 0，如果会则输出最不高兴的是周几（用1,2,3,4,5,6,7 分别表示周一，周二，周三，周四，周五，周六，周日）。如果有两天或两天以上不高兴的程度相当，则输出时间最靠前的一天。

**输入数据 1**

```input1
5 3
6 2
7 2
5 3
5 4
0 4
0 6
```

**输出数据 1**

```output1
3
```

**代码如下**：

```python
schedules = []
for i in range(7):
    school_time, other_time = map(int, input().split())
    schedules.append((school_time, other_time))

max_unhappy_score = 0
most_unhappy_day = 0

for index, (school_time, other_time) in enumerate(schedules):
    total_time = school_time + other_time
    if total_time > 8 and total_time > max_unhappy_score:
        most_unhappy_day = index + 1
        max_unhappy_score = total_time

print(most_unhappy_day)
```

------

### D. 小鲸鱼的 Lucky Word

**题目描述**

小鲸鱼的词汇量很小，所以每次做英语选择题的时候都很头疼。但是它找到了一种方法，经试验证明，用这种方法去选择选项的时候选对的几率非常大！

这种方法的具体描述如下：假设maxn 是单词中出现次数最多的字母的出现次数，minn 是单词中出现次数最少的字母的出现次数，如果 maxn−minn 是一个质数，那么小鲸鱼就认为这是个 Lucky Word，这样的单词很可能就是正确的答案。

**输入格式**

一个单词，其中只可能出现小写字母，并且长度小于 100

**输出格式**

共两行，第一行是一个字符串，假设输入的单词被验证是 `Lucky Word`，那么输出字符串 `Lucky Word`，否则输出 `No Answer`；

第二行是一个整数，如果输入单词被验证是 `Lucky Word`，输出 maxn−minn 的值，否则输出 00。

**输入数据 1**

```input1
error
```

**输出数据 1**

```output1
Lucky Word
2
```

**输入数据 2**

```input2
olympic
```

**输出数据 2**

```output2
No Answer
0
```

**代码如下**：

```python
import math

word = input()
char_cnt = {}

for char in word:
    char_cnt[char] = char_cnt.get(char, 0) + 1

maxn = max(char_cnt.values())
minn = min(char_cnt.values())
diff = maxn - minn


def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(math.sqrt(num) + 1)):
        if num % i == 0:
            return False
    return True


if is_prime(diff):
    print("Lucky Word")
    print(diff)
else:
    print("No Answer")
    print(0)
```

### E. 哥德巴赫猜想

**题目描述**

输入一个偶数 N*，验证 4∼*N 所有偶数是否符合哥德巴赫猜想：任一大于 22 的偶数都可写成两个质数之和。如果一个数不止一种分法，则输出第一个加数相比其他分法最小的方案。例如 10，10=3+7=5+5，则10=5+5 是错误答案。

**输入格式**

第一行输入一个正偶数N

**输出格式**

输出 N-2 / 2行。对于第 *i* 行：

首先先输出正偶数2*i*+2，然后输出等号，再输出加和为2*i*+2 且第一个加数最小的两个质数，以加号隔开。

**输入数据 1**

```input1
10
```

**输出数据 1**

```output1
4=2+2
6=3+3
8=3+5
10=3+7
```

**代码如下**：

```python

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


N = int(input())

# 验证 4 ~ N 中的所有偶数
for num in range(4, N + 1, 2):
    # 我们只需要遍历从 2 ~ num/2 的整数
    for i in range(2, num // 2 + 1):
        if is_prime(i) and is_prime(num - i):
            print(f"{num}={i}+{num-i}")
            break
```

**完结撒花 ~~**

## - Chapter7

> todo...

