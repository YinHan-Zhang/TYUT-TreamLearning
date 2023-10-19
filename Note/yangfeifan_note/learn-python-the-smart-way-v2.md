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
print(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 <= (r1 + r2))
```

今天的题目都较为简单，主要是让大家学会使用常用的内置函数解决一些问题。



## - Chapter4

> Chapter 4 条件 Conditionalss



