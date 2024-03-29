# Python



## python变量：

1.创建变量x后，对x重复创建不同类型的变量x，之前创建的变量x会被覆盖掉

2.python变量不需要显示声明变量类型，但是python会根据赋值或运算自动推断变量类型

3.python允许多个变量引用同一个值。这是python基于值的内存管理方式，作用：减少内存空间的占用，提高内存利用率。

4.python具有自动管理内存的功能，会跟踪所有的值。自动删除不再使用或者引用次数为0的值。

5.python中变量不直接储存值，而是储存引用，也就是值在内存中的地址

```python
x = 8
y = x  此时x和y的引用地址是一样的

x = 7 修改x的值，x的地址会发生变化，但是y的地址没有变还是原来的地址

#修改变量值这个操作，并不是修改的值，而是修改了变量指向的内存地址
 

```





## 内置对象

| 对象类型                   | 示例                                            | 说明                                                 |
| -------------------------- | :---------------------------------------------- | ---------------------------------------------------- |
| 数字   int，float，complex | 12，1.3e5 ，3+4j                                | 支持复数运算                                         |
| 字符串  str                | ‘...’ ,  “...” , ‘’‘ ... ’‘’                    | 单双三引号作为界定符,可以互相嵌套，r开头为原始字符串 |
| 列表  list                 | [1,2,‘str’]                                     | 元素放在方括号中，以逗号为分割                       |
| 字典 dict                  | {“key”：value}                                  | 元素以键值对的方式储存，以逗号分割                   |
| 元组 tuple                 | （1，2） （1，）                                | 元素放在括号中，只有单个元素，逗号不能丢             |
| 集合 set，fronzenset       | {1,‘a’,...}                                     | 大括号储存，元素不能重复，set可变，fronzenset不可变  |
| 可迭代对象                 | 生成器、range、zip、enumernate、map、filter对象 |                                                      |

### 变量命名注意点

1.字符串和元组属于不可变序列，不能使用通过下标的方式修改元素值

2.变量名必须以字母、下划线、或汉字开头

3.变量名不能友空格以及标点符号

4.不能使用关键字作为变量名

5.不建议使用系统内置的模块名、类型名、函数名作为变量名

6.变量名要区分大小写  例：oneStudent、one_student

### 运算符注意点

* ‘ + ’ 运算符除了算术加法 ，还可用于列表、元组、字符串的连接，但不支持不同类型对象的相加或连接

  ​		``‘A’+1``这个表达式是错的，不是用ASCII码计算，就是类型不同不能加

* ’ * ‘ 除了乘法运算，可以对列表、字符串、元组的内容进行重复，**浅复制！！！**

* 除法：

  ’ / ‘  ：结果为实数   		’ // ‘ ：向下取整数    			’ % ‘ ：计算余数

* in : 测试一个对象是否为另一个对象的元素  ，返回的是布尔值

* is : 同一性测试运算符 ，测试两个对象是否是同一个 ，返回布尔值

* and 和 or ：具有惰性求值的特点，**只计算必须计算的表达式**

  ```
  3>5 and 1>10  只计算3>5不计算 1>10 ,3>5是假，表达式肯定为假
  3<5 or a>5	  不计算a>5,3<5为真，已经可以判断了
  
  当and两边的值同时为真 ，最后一个计算的值 就是整个表达式的值
  3 and 5  >>5
  
  ```

  

* python不支持 ++和-- 运算符

* 逗号，返回一个元组

* 列表的sort（）的返回值是None

### 小总结

1.python中一切都是对象

2.python采用的是基于值的内存管理方式，当多个对象被赋予相同值时，该值在内存中只有一个副本

3.del命令可以删除一个变量，也可以删除列表，字典，集合等可变序列中的部分元素

4.字符串转列表的时候，列表里的元素是字符串的每一个字符，每一个！！！

### 常用内置函数

####  abs（）：取绝对值

#### all（x）：只要可迭代对象x里面有一个是True，就返回True ，也就是x里面有一个满足条件的元素就返回True

#### dir（x）：返回对象或模块的成员列表

#### enumerate（x）：一个个枚举x，（0，x[0]），（1，x[1]）...

#### filter( func , seq) :  使用func函数 里制定的规则对seq序列里的元素进行过滤

#### isinstance（x，类型）：判断x是不是某种类型

#### range（start ,end,step）: 区间左闭右开，元素为整数

三种用法：range(end),	range(start,end),	range(start,end,step)

start默认为0 ，step默认为1 ，惰性求值的range对象。

#### reversed（x）：逆序x，返回一个对象

#### round（x，保留几位小数）：对x四舍五入，默认返回整数

#### zip(x,y):讲x和y里的元素以元组的形式一一配对组合，返回的个数取决于x，y中元素少的那个，多余的舍弃。当然不仅限于x，y两个，可以推广到多个

#### all(): 测试序列对象中是否所有的元素都等价于True ， 0就是False

#### any():测试序列对象中是否存在一个元素都等价于True

## 列表

**列表的元素索引从0开始**

| 方法                | 说明                                  |
| ------------------- | ------------------------------------- |
| append(x)           | 元素x添加到列表尾部,单个添加          |
| extend(x)           | 把x里面所有逐一拿出来，再注意添加进去 |
| insert(index,x)     | 插入到index的位置                     |
| remove(x)           | 移除首次出现的x                       |
| pop(x)              | 删除掉位置为x的元素，默认删除最后一个 |
| clear()             | 删除全部元素，但保留对象              |
| index(x)            | 查找元素x第一次出现的位置             |
| count(x)            | 统计x在列表中出现的次数               |
| reverse(x)          | x原地逆置，不可恢复                   |
| sort(key=,reverse=) | 对列表进行原地排序，不可恢复          |
| copy()              | 浅复制，不开辟新的内存                |

删除重复值时，如何遍历列表，使用remove方法，可能会删不干净。

**因为如果有相邻连续重复的值，python会自动收缩列表元素，索引就变了，所以删不干净**

解决办法：

1.采用切片遍历：``for i in x[:]``

2.更建议从后往前删除 ``for i in x[::-1]``



### 切片

**再说一遍，索引从0开始,左闭右开**

切片易错点：

* 切片截取列表中的任何部分，返回的是一个新的列表

* 切片操作不会因为下标越界而抛出异常，而是在列表尾部进行截断或返回一个空列表

* 切片返回的是浅复制，是指返回一个新的列表，并且把原列表中所有的元素都复制到新列表中。

  **如果列表中没有可变对象，那么他俩互不影响，但如果列表中有可变对象，包括列表本身的复制，一旦对其中的一个进行增删操作，另外一个也会受到影响。**

```
list[start:end:step]

#快速逆置  [::-1]
#取奇数位置元素 [::2]
#取偶数就只能自己设置循环，挨个遍历去了
```

### 列表排序

* sort(key=,reversed=) :  

  默认升序排列 ， key用来设置规则一般使用lambda匿名函数来实现，reversed表示是否逆序

* sort() 和sorted() : sort是原地排序，不开辟新的内存空间储存排序后的列表，不可逆操作

  而sorted是开辟新的空间来储存排序后的新列表，不对原列表进行修改

---



sort()是列表的方法

sorted()是内置函数：直接返回列表



1.reverse作为参数是列表中控制升降序的属性，原地操作

2.reverse()作为列表的方法，控制列表逆序，原地操作

3.reversed()作为内置函数，返回一个逆序排列后的可迭代对象，用使用list（）方法输出



#### 列表推导式的花式操作真的惊艳到我了！！！

```python
#1.同时控制索引和值
for i,v in enumerate(x):
    x[i]=v.strip()
    
= list(map(str.strip(),x))
#2.平铺
lis = [ i for x in y for i in x ]
#3.过滤
lis = [ i for i in x if ...]

```



#### 生成器表达式

> 什么是生成器表达式？ 关键字 generator
>
> 和列表推导式类似，只是生成器使用的是**括号**
>
> 区别：生成器表达式生成的是一个生成器对象

可以使用next或_ _next _ _ () 方法 获取下一个元素，或者直接for循环遍历

**重点：生成器对象中的每一个元素都只能使用一次，而且只能从前到后访问**

好处是什么呢？

生成对象之后，我可以随意的设置把生成器对象变成哪一种

## 元组

* 只能使用del命令删除整个元组对象，不能只删除元组中的部分元素，因为元组属于不可变序列
* 元组没有append(),insert(),remove(),pop()这些修改元素的方法
* 元组支持切片操作，但仅仅是访问哦
* 元组不可修改，但是元组里面如果有可变序列，可以通过切片访问该可变序列然后进行修改
* 

## 字典

字典中的键必须是不可变数据，不能使用列表、集合、字典作为字典的键，包含列表、集合、字典的元组也不行！

* dict(zip(a,b)) 快速创建字典

字典中常用方法：

| 方法              | 描述                                       |
| ----------------- | ------------------------------------------ |
| get(‘键’，默认值) | 由键获取值                                 |
| keys()            | 获取所有的键                               |
| values()          | 获取所有的值                               |
| items()           | 获取所有的键值对                           |
| update()          | 将一个字典的所有键值对加入到另一个字典     |
| del               | 删除字典指定的键                           |
| clear()           | 删除所有的元素                             |
| pop()             | 删除并返回指定键的元素，只返回值           |
| popitem()         | 删除并返回字典中的一个元素，两个值，键和值 |





## 集合

> 集合是无序可变对象，和字典一样使用大括号，但集合元素不允许重复

set（）函数创建集合

常用方法：

| 方法     | 描述                                       |
| -------- | ------------------------------------------ |
| add()    | 增加元素                                   |
| pop()    | 弹出并删除一个元素，默认最后一个，位置索引 |
| remove() | 删除元素，元素索引                         |
| clear()  | 清空元素                                   |
| del      | 删除整个集合                               |

* 集合不支持使用下标访问其中的元素
* 列表不可以作为集合的元素
* 集合可以作为列表的元素
* 元组可以作为集合的元素
* 集合可以作为元组的元素
* 字典不可以作为集合的元素
* 集合不可以作为字典的键
* 集合可以作为字典的值

# 字符串

编码：

1.UTF-8编码： 以1个字节表示英语字符，3个字节表示常见汉字

2.GB2313编码：1字节表示英语，2字节表示汉字 ， GBK是GB2313的扩展



## 常见方法：

| 方法          | 描述                                                         |
| ------------- | ------------------------------------------------------------ |
| find(x)       | 查找x在字符串中首次出现的位置                                |
| rfind(x)      | 查找最后一次出现的位置                                       |
| index(x)      | 查找x的首次出现位置                                          |
| rindex()      | 最后一次出现的位置                                           |
| count()       | 出现次数                                                     |
| split()       | 从左开始分割，分割符是内容=直接删除掉                        |
| rsplit()      | 从右开始分割，遇到非分割符停止                               |
| partition()   | 分割成三部分，分隔前的，要分隔的，分隔后的                   |
| rpartition()  | 从后面开始分隔                                               |
| join          | 将可迭代对象中多个字符串之间进行连接，可设置连接符           |
| lower()       | 字符串全部转换为小写                                         |
| upper()       | 字符串全部转换为大写                                         |
| capitial()    | 字符串的第一个字符大写                                       |
| title()       | 单词的首字母大写                                             |
| swapcase()    | 将字符串的大小写进行互换                                     |
| replace()     | （要替换上的字符串，待替换的字符串，替换次数）               |
| strip()       | 删除字符串两端的空格，如果加上指定分割字符，即为删除         |
| rstrip()      | 删除字符串左端的空格                                         |
| lstrip()      | 删除字符串右端的空格                                         |
| eval()        | 把字符串转换为表达式，并求值                                 |
| startswith(x) | （x，开始检测的位置，结束检测的位置）字符串是否以x开头       |
| endswith(x)   | 字符串是否以x结尾                                            |
| isalnum()     | 是否为数字或字母                                             |
| isalpha()     | 是否为字母                                                   |
| isdigit()     | 是否为数字                                                   |
| isspace()     | 是否为空白字符                                               |
| isupper()     | 是否为大写字母                                               |
| islower()     | 是否为小写字母                                               |
| center()      | 居中（参数可填指定宽度，指定填充字符），如果超过了原有字符串长度，用指定字符填充 |
| ljust()       | 左对齐，补充数字长度，填充至默认长度                         |
| rjust()       | 右对齐                                                       |

### random模块

| 方法      | 描述                                             |
| --------- | ------------------------------------------------ |
| choice()  | 从序列中任意选择一个                             |
| randint() | （开始范围，终止范围）生成随机整数               |
| shuffle() | 列表原地随机乱序                                 |
| sample()  | （列表，个数）从列表中选取指定个数的不重复的元素 |



# 正则表达式

| 字符匹配规则 | 描述                                                  |
| ------------ | ----------------------------------------------------- |
| \d           | 任意数字                                              |
| \D           | 代表不是数字的                                        |
| \w           | 代表字母，数字，下划线，也就是a-z，A-Z，0-9           |
| \W           | 不是字母数字下划线的                                  |
| \n           | 换行符                                                |
| \r           | 回车符                                                |
| \f           | 换页符                                                |
| \t           | 制表符                                                |
| \s           | 代表任何空白字符，空格、换行、换页符                  |
| \S           | 非任何空白字符                                        |
| \A           | 字符串的开头                                          |
| \Z           | 字符串的结尾                                          |
| \b           | 匹配单词头或者单词尾                                  |
| \B           | 匹配非单词头非单词尾                                  |
| .            | 匹配除换行符以外任意的字符                            |
| *            | 匹配0或无数次                                         |
| ?            | 0次或一次 ， 如果跟在 * ，+ ，{} 后面表示非贪心模式   |
| +            | 一次或多次                                            |
| \|           | a\|b , 匹配模式a和b，二选一                           |
| ^            | 匹配字符串的开头                                      |
| $            | 匹配字符串的结尾                                      |
| \ + 一个数字 | 数字代表子模式序号 ，“ (.) \1” 匹配两个连续相同的字符 |
| {m,n}        | 匹配最少m次，最多n次                                  |
| [ ]          | 匹配位于[ ...] 里的任意字符                           |



1. 字符匹配：

   1. \d:代表任意数字  \D：代表不是数字的（大写的字母一般是和小写的唱反调）

   2. \w：代表字母，数字，下划线，也就是a-z，A-Z，0-9，—。\W：不是字母数字下划线的

   3. \n：代表一个换行

   4. \r：代表一个回车      \f：代表一个换页       \t：代表一个tab   

   5. \s：代表所有的空白字符（包括换行回车换页tab)          \S：不是空白的字符

   6. \A:代表字符串的开始     \Z：代表字符串的结束

   7. **^:匹配字符串的开始位置     &：匹配字符串的结束位置**    

   8. \b:匹配一个单词的边界          \B：匹配非单词边界

   9. |：匹配两边的表达式 +（）                   **要想匹配的不多不少刚刚好，就必须用^和&来规定开始结束**

   10. 星号：匹配前面的字符串零次或多次 例如：zo* 能匹配z、zo、zoo

   11. +：匹配前面的字符串一次或多次      例如：zo+能匹配zo、zoo但不能匹配z

   12. ？：匹配前面的字符串零次或一次    例如：do（es）？可以匹配do或者does

   13. [n]：n是一个非负整数，匹配确定的n次   例如o[2]必须匹配到2个o，bob就不能匹配到

   14. [n，]：就是至少匹配n次，可以多不能少   [n,m]：最少匹配n次，最多匹配m次

   15.    .   :  匹配 \r  \n 之外的任何的单个字符

   16. [...]：表示一个范围内的字符 例如[a-z]就是a-z间任意一个字符    **[^]唱反调**

   17. {n}：匹配在{n}前面的东西

   18.  用点(.)去匹配任意字符的时候，忘记了点(.)不能匹配换行符的事实 

        `re.compile()` 函数接受一个标志参数叫 `re.DOTALL` ，在这里非常有用。 它可以让正则表达式中的点(.)匹配包括换行符在内的任意字符。

        比如 comment = re.compile(r'/\*(.*?)\*/', re.DOTALL)

     

2. ()括号可以作为一个分组 ，括号括起来的内容可以作为中间变量记录下来，要想记录下来并使用把后面的那一部分也括起来然后写  （\1）数字就是第几个括号

   括号一多容易混，所以用信息加以定义 (？p=< 定义名字>)   例子：(？p=< key1>)

#### 子模式

1.  子模式

    | 语法         | 功能描述                                                     |
    | ------------ | ------------------------------------------------------------ |
    | （?P<命名>） | 为子模式命名                                                 |
    | （？i）      | 后面跟的就是匹配标志，比如（？i）n\b                         |
    | （？=...）   | 用在正则表达式之后，满足后面的规则则匹配                     |
    | （？！x）    | 用在正则表达式之后，不出现x则匹配                            |
    | （? <=...）  | 用在正则表达式前面，要匹配的字符串前面出现括号内标志则匹配   |
    | （?<! ...）  | 用在正则表达式前面，要匹配的字符串前面没有出现括号内标志则匹配 |

  

### re模块主要函数

| 函数                                             | 功能说明                                   |
| ------------------------------------------------ | ------------------------------------------ |
| compile(pattern)                                 | 创建正则表达式对象                         |
| search(pattern,待搜索字符串)                     | 从字符串中搜索，返回Match对象或None        |
| match(pattern,待搜索字符串)                      | 从字符串的开始出匹配，返回Match对象或None  |
| findall(pattern,待搜索字符串)                    | 返回所有匹配项匹配的列表                   |
| split(pattern,待搜索字符串)                      | 根据匹配项分隔字符                         |
| sub(pattern，替换字符串，待搜索字符串，替换次数) | 根据匹配规则替换字符                       |
| escape()                                         | 将字符串中所有的特殊正则表达式字符进行转义 |

#### 特定搜索：

例：以某个字母开头

以“  \ ” 开头的元字符实现字符串的特定搜索

``(‘\\b a.+? \\b’)``    \b 本来就是匹配字符串开头，再加一个 \  表示特定搜索

``(' \b (?i) a \b')``  利用子模式的匹配标志也可以匹配以a开头的单词 

#### flags标志

| flag        | 含义                                    |
| ----------- | --------------------------------------- |
| re.S        | 使 ‘ . ’ 匹配任意字符，就是包括了换行符 |
| re.I        | 忽略大小写                              |
| re.M / re.L | 多行匹配模式，忽略换行符                |
| re.U        | 匹配Unicode字符                         |
| re.X        | 忽略模式中的空格                        |



### Match对象

match对象的方法

| 方法名      | 描述                           |
| ----------- | ------------------------------ |
| group()     | 返回1个或者多个匹配内容        |
| groups()    | 匹配所有匹配到的内容的元组     |
| groupdict() | 匹配所有匹配到的内容的字典     |
| start()     | 返回子模式的起始位置           |
| end()       | 返回子模式的终止位置           |
| span()      | 返回子模式的起始位置和终止位置 |

### 注意点：

* replace ()  ，maketrans() , translate() 和正则表达式的方法对内容进行修改替换，返回的都是新字符串，并不对原字符做修改





# 函数

定义细节：
1.函数形参不需要声明其类型，也不需要指定返回值类型

2.函数不接收参数，也要保留一对空的括号。

3.括号后面的冒号必不可少

4.python允许嵌套定义

## 参数类型

* 默认值参数：如果没有对形参传值，则使用默认值参数
* 关键参数：通过参数名字传递值 ``(a,b,c=5)``，可以不用按顺序传参
* 传参时，序列解包 。 一个*号，传递列表、元组等序列  ；两个 * 号 传递字典，键作为参数名，值作为参数的值
* 

## 变量作用域

变量分为三类： 局部变量、nonlocal变量和全局变量

局部变量：在函数内部定义，只在函数内部起作用，函数允许结束，变量自动删除

全局变量：能够同时作用于函数内外的变量，通过global定义



### 注意点：

1.一个变量在函数外定义，如果在函数内要修改这个变量的值，并将值反映到函数之外，需要用global关键声明使用全局变量

2.在函数内部使用global关键字 将一个变量声明为全局变量，如果在函数外没有定义该变量，调用该函数后会自动增加

3.使用默认值参数，必须保证默认值参数出现在函数参数列表的最后。



# 类

>  以变量形式表示的对象属性称为   -> 数据成员或者成员属性
>
> 以函数的形式表示的对象行为称为 -> 成员方法
>
> 成员属性和成员方法统称为类的成员

* 类的实例方法都必须带有一个self参数，而且必须是第一个。self的含义就是表示对象自己

* 访问实例属性时，需要以self为前缀。

## 类成员与实例成员

数据成员，也就是广义上的属性。

#### 实例属性

实例属性一般是指在 构造函数--init--()里面定义的。

实例属性属于对象，只能通过对象名访问。

#### 类属性

在类的所有方法之外定义的数据成员

类属性属于类，可以通过类名和对象名访问

## 私有成员与公有成员

1. _xxx : 保护成员 ，不能使用from  ... import * 导入
2. _ _ xxx _ _: 系统内部定义的特殊成员
3. _ _ xxx: 类的私有成员 ，在类的外部，访问： “对象名._类名 _ _xxx”

**注意：在交互模式下，一个下划线表示解释器最后一次显示的内容或最后一次正确语句执行的结果**





## 方法

> 类中的方法分为四大类：公有方法、私有方法、静态方法、类方法

公有方法通过对象名之间调用

私有方法不能通过方法名调用，只能在对象的方法中通过self调用，或者 “对象名._类名 _ _xxx”来访问



如果通过类名来调用对象的公有方法，需要为该方法的self参数传递一个对象名，来确定访问哪一个对象的数据成员。



静态方法和类方法：都不能直接访问属于对象的成员，只能访问属于类的成员，一般使用cls作为第一个参数



用@classmethod定义类方法

用@staticmethod定义静态方法



## 属性

通过@property来定义属性，可以设置属性状态 ：可读、可修改、可删除

```python
@property
def value(self):
    return self.__value

value = property(__get,__set,__del) #设置可读、可删除、可修改
```

可以动态增加新成员 ：``t.v = 5``直接赋值就好了

``del t.v``  直接 删除对象



## 继承机制

两种调用基类的方法：

1.内置函数 super（基类名，self）.方法名         **self不能少哦**

2.基类名.方法名



# 文件

## 文件打开模式

| 文件打开模式 | 说明       |
| ------------ | ---------- |
| r            | 只读模式   |
| w            | 只写模式   |
| a            | 追加模式   |
| b            | 二进制模式 |
| +            | 读写模式   |
| rb+          | 二进制写   |
| wb+          | 二进制读   |
| a+           | 追加读写   |

## 文件对象的属性：

| 文件属性 | 说明                             |
| -------- | -------------------------------- |
| closed   | 判断文件是否关闭，文件关闭为True |
| mode     | 返回文件的打开模式               |
| name     | 返回文件的名字                   |

## 文件对象常用方法

| 文件对象方法 | 说明                                               |
| ------------ | -------------------------------------------------- |
| close()      | 缓存区内容写入文件，然后关闭文件                   |
| read()       | 从文件中读取内容，可指定读取字节大小，默认读取全部 |
| readline()   | 从文件中读取一行                                   |
| readlines()  | 把文件中每行文本作为一个字符串存入列表中，返回列表 |
| seek()       | 移动文件指针                                       |
| tell()       | 返回文件指针的当前位置                             |
| write()      | 把s的内容写入文件                                  |
| writelines() | 把列表中的字符串逐个写入文本文件                   |

## pickle模块

存入文件：pickle.dump(内容，文件名)

读取文件：pickle.load(文件名)

上下文模式打开文件 ：with open(‘文件路径’，’打开模式‘)  as 文件别名 ：



## OS模块

### 操作文件函数

| 函数               | 功能说明                             |
| ------------------ | ------------------------------------ |
| remove(path)       | 删除文件                             |
| rename(path)       | 重命名                               |
| stat(path)         | 返回文件的属性                       |
| listdir(path)      | 返回该目录下所有的子文件名           |
| basename(path)     | 返回最后一个分隔符后面的部分--扩展名 |
| dirname(path)      | 返回最后一个分隔符前面的部分         |
| exists(filename)   | 文件是否存在                         |
| getatime(filename) | 文件最后访问时间--append             |
| getmtime(filename) | 文件最后修改时间 -- modify           |
| getctime(filename) | 文件创建时间 --create                |
| getsize(path)      | 返回文件大小                         |
| isdir(path)        | 是否是文件夹                         |
| isfile(path)       | 是否是文件                           |
| join(path)         | 连接path                             |
| split(path)        | 对路径进行分隔，返回元组             |
| splitext(path)     | 从路径名中分隔文件的扩展名，返回元组 |

### OS.path模块

| 成员                      | 功能说明                                               |
| ------------------------- | ------------------------------------------------------ |
| mkdir(path)               | 创建目录                                               |
| makedirs(path1,path2,...) | 创建多个目录                                           |
| rmdir()                   | 删除目录                                               |
| removedirs(path1,...)     | 删除多个目录                                           |
| listdir(path)             | 返回指定文件夹下文件和目录信息                         |
| getcwd(path)              | 获取当前工作目录                                       |
| chdir(path)               | 把path设置为当前工作目录                               |
| walk(top,topdown)         | 遍历目录树，返回（所有路径名，所有目录列表，文件列表） |







