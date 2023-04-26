# Pandas

## pandas的两种数据结构

### DataFrame

二维数据

相当于整个Excel表

### Series

一维数据

相当于每一列

### 创建Series

```python
import pandas as pd

s = pd.Series([1,2,3], index=list('abc'), dtype='int64', name='num')   # 方式一

d = {'a':1, 'b':2, 'c':3}                                              # 方式二
s = pd.Series(d)

s = pd.Series(3.0, index=['a','b','c'])                                # 方式三
```

### 创建DataFrame

#### 1、通过二维的list-like创建

```python
list_2d = [[1,2,3],
           [4,5,6]]
df = pd.DataFrame(list_2d, index=[1,2], columns=['A', 'B', 'C'])
```

#### 2、通过字典创建

```python
d = {'A':[1,4], 'B':[2,5], 'C':[3,6]}     # 字典的键是列索引
df = pd.DataFrame(d, index=['x','y'])
```

#### 3、通过读取Excel表

```python
df = pd.read_excel('census.xlsx')
```

## pandas.read_excel()参数

```python
pandas.read_excel(
	io,               # 路径
	sheet_name=0,     # sheet
	header=0,         # 列索引(表头)
	index_col=None,   # 行索引
	usecols=None,     # 使用的列
	skiprows=None     # 跳过行
)
```

### 1、io:路径/StringIO/URL

### 2、sheet_name:选择子表

默认：0     选第0个

int：选第几个sheet，从0开始

str：sheet名称

list：[0, 'sheetname']     返回字典

None：全部sheet           返回字典

### 3、header:指定作为列索引的行

header= None    

header=0              

header=1             

### 4、index_col:指定作为行索引的列

header=[0,1]   index_col=0   

### 5、usecols:指定只使用哪些列

None:全部(默认)

str : 'A,C'     'A,C:E'                           

int-list:[0,2]

str-list:['AAA','CCC']     # 建议使用

函数:lambda x:x=='CCC' 	    

### 6、skiprows:跳过行

skiprows = n               跳过n行

skiprows = [0,2]         跳过索引为0、2的两行

### 7、names:要使用的列名列表

如果文件不包含标题行，则应显示传递   header = None

```python
pd.read_excel('census.xlsx',
              header=None,
              names=['first','second','third','fourth','fifth'])
```

### 8、dtype:字典{'列名':'类型'} 设置列的类型



int8 / int16 / int32 / int64(默认)         整型

float16 / float32 / float64(默认)          浮点型

str / string                                               字符串

bool                                                          布尔

category                                                  分类

datetime64[ns]                                       时间戳（纳秒）

period[Y/M/D]                                         时间周期（年/月/日）

object                                                        python对象混合类型



```python
df.dtypes    # DataFrame的所有列的默认类型
```







# read_excel()参数

## 1、parse_dates：解析成日期格式

```python
pd.read_excel('test.xlsx',
             parse_dates=True)
```

True :尝试解析index，如果没有成功保持原有格式

[0,1]或['a', 'b'] :尝试解析指定列作为一个单独的日期列

[[0, 1, 2]] :结合多列解析为单个日期列

{'日期' : [0, 1, 2]} :结合多列解析为单个日期列，结果的列名改为'日期'

## 2、date_parser: 解析日期格式的函数

```python
# 主要解析带有中文的日期
df = pd.read_excel(
    'date_parser.xlsx',
	parse_dates = [0],
	date_parser = lambda x:pd.to_datetime(
    	x, format = '%Y年%m月%d日'
    )
)
```

## 3、na_values: 缺失值替换为NaN

缺失值： NaN      Not a Number(float类型)

na_values: scalar、str、list-like、dict、default None

dict:解析指定列

scalar、str、list-like:解析全局 

## 4、converters: 对同列值函数运算

```python
pd.read_excel(
	'test.xlsx',
	converters = {
        'a':lambda x:x+1,
        '货号':lambda x:x.strip()
    })
```

## 5、true_values   false_values

true_values: 传入list 默认None  视为True的值

false_values:传入list 默认None  视为False的值

```python
# 只适用于字符串
df = pd.read_excel(
	'true_false_values.xlsx',
	true_values=['a','b'],            # 值为a、b的单元格内容变为True
	false_values=['c','d']            # 值为c、d的单元格内容变为False
)          
```

```python
# 将数字转化为bool类型
df = pd.read_excel(
	'true_false_values.xlsx',
	dtype={'列3':bool}                # 0变为False 非0变为True
)
```

## 6、squeeze:如果数据只有一列，是否返回一个Series

bool类型   默认False

squeeze = False       # 默认     返回DataFrame

squeeze = True        #               返回Series

## 7、mangle_dupe_cols:是否重命名重复列名

mangle_dupe_cols: bool类型   默认False

mangle_dupe_cols = True     # 重命名重复列名

mangle_dupe_cols = False    # 如有重复列名抛出异常

## 8、nrows：解析的行数

nrows: int类型 默认为None（不包括表头行）

## 9、thousands:千位分隔符

thousands: str类型 默认None

千位分隔符，用于将字符串列解析为数字

thousands = None    2,123

thousands = ','           2123

## 10、convert_float:是否将float转换为int

 convert_float：bool类型，默认True 

 convert_float = False     # 列为float类型

 convert_float = True     # 列为int类型

## 详情：https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html?highlight=read_excel

# 写Excel文件

## to_excel()

```python
DataFrame.to_excel(
	self,
	excel_writer,           # 文件路径/ExcelWriter
	sheet_name='Sheet1'     # sheet名字
	index=True,             # 是否输出index
	float_format=None,      # 浮点数输出格式 如
	na_rep='',              # 缺失值输出的表示形式
)
```

## ExcelWriter

```python
class pandas.ExcelWriter(
	path,                      # xls/xlsx 文件路径
	datetime_format=None       # datetime输出格式 默认:'YYYY-MM-DD HH:MM:SS'
)
```

to_excel()函数第一个参数使用ExcelWriter类的两个作用：

1、设置datetime输出格式

2、输出多个sheet

具体用法

```python
with pd.ExcelWriter(
	'demo.xlsx',
	datetime_format='YYYY-MM-DD') as writer:
    df1.to_excel(writer, sheet_name='f1')
    df2.to_excel(writer, sheet_name='f2')
```

# 读/写csv文件

```python
# 写
df.to_csv(
	path_or_buf,          # 路径
	sep=',',              # 分隔符
	encoding='utf-8'      # utf8/gbk （90%情况utf8,少数情况gbk）
)

# 读
df.to_csv(
	path_or_buf,          # 路径
	sep=',',              # 分隔符
	encoding='utf-8'      # utf8/gbk （90%情况utf8,少数情况gbk）
)

```





# pandas选择数据

## 1、dict-like

### **Series**

s['B']

s[1]

s[['B','C']]

s[[True,False,True]]

s[s>1]

### **DataFrame**

df['B']

df[['B','C']]

df[[True,False,True]]

df[df['B']>2]

df[ (df['B']>2 ) & (df['B']<6) ]       选中行(且)

df[ (df['B']==2 ) | (df['B']==8) ]     选中行(或)

df[ ~(df['B']==2 )]                           选中行(非)

## 2、list-like

### Series

s[0:2]        前俩行

s[0:-1]      第一行到倒数第二行

s[::-1]       最后一行到第一行(反转)

s[::2]         步距2

s['A','B']      A行到B行

### DataFrame

df[0:2]      前两行

df['X','Y']   X行到Y行

## 3、.点选

### Series

Series.index_name

### DataFrame

DataFrame.columns_name

不推荐使用（1、代码可读性不好   2、有可能与方法或属性冲突）

## 4、loc

**df.loc[row_indexer,column_indexer]**

eg:

df.loc['X']

df.loc['X','B']

df.loc[['X','Y'],['B','D']]

df.loc[[True,False,True],[True,True,True,False]]

df.loc[['X':'Y'],['B':'D']]

df.groupby['day','time'].sum().loc[lambda df: df['total_bill']>100 ]

## 5、iloc

**df.iloc[row_indexer,column_indexer]**   (索引 0 1 2   0 1 2)

df.iloc[0]

df.iloc[0,1]

df.iloc[[0,1],[1,3]]

df.iloc[[True,False],[True,False]]

df.iloc[0:2,1:3]

选择单个值时 .loc==.at       .iloc==.iat

# 加减乘除

1、对空值的处理

number +-*/ NaN == NaN

NaN +-*/ number == NaN

将缺失值按0处理

df['A'].fillna(0) + df['B'].fillna(0)

df['A'].add(df['B'],fill_value=0)

2、对除数为0的处理

1 / 0 == inf

-1 / 0 == -inf

```python
pandas.options.mode.use_inf_as_na = True
```

1 / 0 == NaN

-1 / 0 == NaN

3、MultiIndex

```python
df_total.columns = pd.MultiIndex.from_product(
	[
        ['总'],
        df_total.columns    # 复杂索引
    ])
df.join(df_total)           # 拼接
```

# 字符串方法

## 1、strip lstrip rstrip 

去除空白

## 2、split

默认空白拆分

## 3、replace

替换

## 4、contains

```python
'ab'.__contains__('a')    # 返回True
```

## 5、count

计数

## 6、index find in

查找

index find 返回索引位置  in返回bool类型

查找不到

index 异常 find返回-1 in返回False

## 7、join

联合

## 8、startwith endwith

前缀 后缀   返回bool类型

## 9、lower  upper

转化为小写  大写

## 10、Series.str

df = pd.read_excel('tongji.xlsx').convert_dtypes     # 将每一列改为最有可能的类型

Series.str.strip()     # 去两边空白

Series.astype('str').str.strip()

Series.str.split()     # 分隔



# NumPy

## 前置知识点

1、内置数据结构

字符串、列表、元组、集合和字典

2、序列的索引和切片访问的相关操作

3、列表、集合和字典的推导式

4、lambda表达式

5、函数式编程的基础函数

filter()

map()

reduce()

## Numpy简介

### **1、Numpy简介**

1、Numpy是一个开源的Python数据分析和科学计算库

2、Numpy是后续学习Pandas（数据分析）、Scipy（科学计算）和Matplotlib（绘图库）的基础

3、Numpy官网：https://numpy.org/

### **2、Numpy的特点**

1、Numpy底层是使用C语言实现的，速度快。

2、Numpy提供数据结构（数组）比Python内置数据结构访问效率更高

3、支持大量高维度数组与矩阵运算

4、提供大量的数学函数库

## 数组对象

### 特点

1、用于存放同类型元素的集合

2、每个元素在内存中都有相同存储大小的区域

### 创建

numpy.array(object)     object类型可以是列表或者元组

### 数据类型

1、整型

2、浮点型

3、复数

4、字符串

5、布尔型

```python
import numpy as np

a = np.array([1,3,5,7,9], dtype=np.int64)  # 指定数据类型
a = np.array([1,3,5,7,9], dtype='i8')      # 指定类型代码

print(a, a.dtype)  # 输出数组a和其类型

```

字节序

< ：小端字节序    内存地址从低到高      > : 大端字节序    内存地址从高到低

## 创建一维数组

### 1、正常创建

```python
a = np.array([1,3,5,7,9])
```

### 2、arange()函数

```python
np.arange(start, stop, step, dtype)
```

start : 开始值，默认值为0，包含开始值

stop : 结束值，不包含结束值

step : 步长，默认值为1，该值可以为负数

dtype : 数组元素类型

### 3、linspace()函数（创建等差数组）

```python
np.linspace(start, stop, num, endpoint, retstep, dtype)
```

num : 设置生成的元素个数

endpoint : 设置是否包含结束值，False是不包含，True是包含，默认值是True

retstop : 设置是否返回步长(即公差)，False是不返回，默认值是False，True是返回，当值为True时返回值是二元组，包括数组和步长

### 4、logspace()函数（创建等比数组）

```python
np.logspace(start, stop, num, endpoint, base, dtype)
```

start : 开始值，值为 base**start

stop : 结束值，值为 base**stop

base : 底数

## 创建二维数组

数组的轴：二维数组有两个轴，轴索引分别为0和1        0 -> 行     1 -> 列

数组转置：b = a.T

### 1、正常创建

```python
a = np.array([1,2,3],
            [4,5,6])
```

### 2、ones函数

根据指定的形状和数据类型生成全为1的数组

```python
np.ones(shape, dtype=None)
```

shape : 数组的形状     列表 [行数,列数]

### 3、zeros()函数

根据指定的形状和数据类型生成全为0的数组

```python
np.zeros(shape, dtype=None)
```

shape : 数组的形状     列表 [行数,列数]

### 4、full()函数

根据指定的形状和数据类型生成数组，并用指定数据填充

```python
np.full(shape, fill_value, dtype=None)

```

fill_value : 指定填充的数据

### 5、identity()函数

创建单位矩阵

```python
np.identity(n, dtype=None)

```

n : 数组的形状

## 访问

### 1、索引访问

#### 1、一维数组索引访问

```python
正索引  0   1   2   3   4   5
值     1   2   3   4   5   6
负索引  0  -5  -4  -3  -2  -1 

```

#### 2、二维数组索引访问

```python
ndarray[所在0轴索引][所在1轴索引]
ndarray[所在0轴索引,所在1轴索引]

```

### 2、切片访问

#### 1、一维数组切片访问

```python
ndarray[start:end]              # 切片包括start，不包括end
ndarray[start:end:step]

```

#### 2、二维数组切片访问

```python
ndarray[所在0轴切片,所在1轴切片]
# 第一个第二个均为切片，返回二维数组；第一个或第二个为切片，另外一个为标量，返回一维数组

```

### 3、布尔索引

#### 简介

1、布尔索引必须与要索引的数组形状相同，否则会引发IndexError错误。

2、布尔索引返回的新数组是原数组的副本，与原数组不共享相同的数据空间，**即新数组的修改不会影响原数组，即深层复制**。（切片：浅层复制）

#### 具体使用

```python
import numpy as np
a = np.array([1,2,3,4])
b = np.array([True,False,True,False])
print(a[b])
输出结果：[1 3]

a1 = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])
b1 = np.array([[True,False,True],
               [True,False,True],
               [True,False,True]])
print(a1[b1])
输出结果：[1 3 4 6 7 9]     # 仍为一维数组

```

### 4、花式索引

#### 定义

1、索引为整数列表

2、索引为一维整数数组

3、索引为二维整数数组

#### 注意

1、花式索引返回的新数组与花式索引数组形状相同

2、花式索引返回的新数组与布尔索引类似，属于深层复制

3、二维数组上每一个轴的索引数组形状相同

## 数组的操作

### 连接数组

#### 1、concatenate()函数

该函数沿指定的轴连接多个数组

```python
np.concatenate((a1,a2,...),axis)

```

a1,a2是要连接的数组。注意，除指定轴外，其他轴元素个数必须相同。

axis是沿指定轴的索引，默认为0轴。

#### 2、vstack()函数

沿垂直堆叠多个数组，相当于concatenate()函数axis=0的情况

```python
np.vstack((a1,a2))

```

注意：1轴上元素个数必须相同

#### 3、hstack()函数

​	沿水平堆叠多个数组，相当于concatenate()函数axis=1的情况

```python
np.hstack((a1,a2))

```

注意：0轴上元素个数相同

#### 具体使用

```python
import numpy as np

a = np.array([[1,2],
              [3,4]])
b = np.array([[5,6]])
ab1 = np.concatenate((a,b.T),axis=1)  # concatenate函数
ab2 = np.vstack((a,b))                # vstack函数
ab3 = np.hstack((a,b.T))              # hstack函数

```

### 分割数组

#### 1、split()函数

该函数沿指定的轴分割多个数组

```python
np.split(ary,indices_or_sections,axis)

```

ary : 是要被分割的数组

indices_or_sections : 是一个整数或数组，如果是整数就用该数平均分割；如果是数组，则为沿指定轴的切片操作

axis : 指定轴的分割方向，默认为0轴

#### 2、vsplit()函数

该函数沿垂直方向分割数组，相当于split()函数axis=0的情况

```python
np.vsplit(ary,indices_or_sections,axis)

```

#### 3、hsplit()函数

该函数沿水平方向分割数组，相当于split()函数axis=1的情况

```python
np.hsplit(ary,indices_or_sections,axis)

```

#### 具体使用

```python
import numpy as np

# %%  一维数组 整数
a = np.arange(9)
b = np.split(a,3)
print(b)
# %%  一维数组 数组
a1 = np.arange(9)
sections1 = np.array([4,7])
b1 = np.split(a1,sections1)
print(b1)
# %%  二维数组 整数
a2 = np.array([[1,2,3,4],
              [5,6,7,8],
              [9,10,11,12],
              [13,14,15,16]])
b2 = np.split(a2,4)
b3 = np.vsplit(a2,4)
print(b2)
print(b3)
# %% 二维数组 数组
a2 = np.array([[1,2,3,4],
              [5,6,7,8],
              [9,10,11,12],
              [13,14,15,16]])
sec = np.split(a2,[1,2])
print(sec)

```

### 数组算术运算

数组对象可以使用Python原生的算术运算符，数组对应元素执行对应的算术运算

如：+  -  *  /  //  %  **

###  数组广播

#### 1、定义

数组与标量或者不同形状的数组进行算术运算时，就会发生数组广播

#### 2、数组与标量

数组与标量进行算术运算，相当于先将标量广播成相同形状的数组，然后再进行算术运算。

#### 3、数组与数组

数组与不同形状的数组进行算术运算时，会发生广播，需要遵循以下广播原则

(1)先比较形状，再比较维度，最后比较对应轴长度

(2)如果两个数组维度不相等，会在维度较低数组的形状左侧填充1，直到维度与高维数组相等

(3)如果两个数组维度相等时，要么对应轴的长度相同，要么其中一个轴长度为1，则兼容的数组可以广播，长度为1的轴会被扩展。

#### 4、具体代码详解

```python
import numpy as np

# %% 数组与标量
a1 = np.array([1,3,5])
print(a1+2)              # [1,3,5] + [2,2,2]

# %% 数组与数组 实例一
a2 = np.array([1,2])     # [[1,2],[1,2]]
b2 = np.array([[3,4],
               [5,6]])
print(a2 + b2)

# %% 0 实例二
a2 = np.array([[1,2]])   # (1,2)   [[1,2],[1,2]]
b2 = np.array([[3], 
               [4]])     # (2,1)   [[3,3],[4,4]] 
print(a2 + b2)

# %% 数组与数组 实例三
a3 = np.array([[1,2]])   # (1,2)   [[1,2],[1,2]]
b3 = np.array([[3,4,5],     
               [6,7,8]]) # (2,3)
print(a3 + b3)       
# 无法运算 报错

```

## 数组的常用函数

### 1、随机数函数

#### 1、rand()函数

该函数返回 [0.0,1.0) 的随机**浮点数**，即大于等于0.0，且小于1.0的随机浮点数

```python
np.random.rand(d0,d1,d2,...)

```

d0,d1,d2,...,dn : 表示数组的形状

#### 2、randint()函数

该函数返回 [low,high)的随机整数，如果high省略，则返回 [0, low)的随机整数

```python
np.random.randint(low, high, size, dtype)

```

size ：表示数组的形状

#### 3、normal()函数

该函数返回正态分布随机数

```python
np.random.normal(loc, scale, size)

```

loc ：表示平均值

scale : 表示标准差

#### 4、randn()函数

该函数返回标准正态分布随机数，即平均数为0，标准差为1的正态分布随机数

```python
np.random.randn(d0, d1, ..., dn)

```

### 2、排序函数

#### **1、sort()函数**

按照轴对数组进行排序，即轴排序(一行一列的排序)

```python
np.sort(a, axis=-1, kind='quicksort', order=None)

```

a : 表示要排序的数组

axis : 表示排序的轴索引，默认是-1，表示最后一个轴

kind : 表示排序类型。quicksort : 快速排序,为默认值，速度最快；mergesort : 归并排序；heapsort : 堆排序

order : 表示排序字段

#### 2、argsort()函数(一行/一列的排序，返回原元素在所在原行/列的索引)

按照轴对数组进行排序索引，即轴排序索引

```python
np.argsort(a, axis=-1, kind='quicksort', order=None)

```

### 聚合函数

#### 1、求和

##### 1、sum()函数

```python
np.sum(a, axis=None)

```

##### 2、nansum函数(忽略NaN)

```python
np.nansum(a, axis=None)

```

##### 3、sum()方法

```python
np.ndarray.sum(axis=None)

```

#### 2、求最大值

##### 1、amax()函数

```python
np.amax(a, axis=None)

```

##### 2、nanmax()函数(忽略NaN)

```python
np.nanmax(a, axis=None)

```

##### 3、max()方法

```python
np.ndarray.max(axis=None)

```

#### 3、求最小值

##### 1、amin()函数

```python
np.amin(a, axis=None)

```

##### 2、nanmin()函数

```python
np.nanmin(a, axis=None)

```

##### 3、min()方法

```python
np.ndarray.min(axis=None)

```

#### 4、求平均值

##### 1、mean()函数

```python
np.mean(a, axis=None)

```

##### 2、nanmean()函数(忽略NaN)

```python
np.nanmean(a, axis=None)

```

##### 3、mean()方法

```python
np.ndarray.mean(axis=None)

```

#### 5、求加权平均值

##### average()函数

```python
np.average(a, axis=None,weights=None)

```

weights：权重

## 数组的保存和读取

### 数组的保存

#### 1、save()函数

该函数可以将一个数组保存至后缀名为‘‘ .npy ’’的二进制文件中

```python
np.save(file,arr,allow_pickle=True,fix_imports=True)

```

file : 表示文件名/文件路径

arr : 表示要存储的数组

allow_pickle为布尔值，表示是否允许使用pickle来保存数组对象

fix_imports为布尔值，表示是否允许在Python2中读取Python3保存的数据

#### 2、savez()函数

该函数可以将多个数组保存到未压缩的后缀名为‘‘ .npz ’’的二进制文件中

```python
np.savez(file)

```

#### 3、savez_compressed()函数

该函数可以将多个数组保存到压缩的后缀名为‘‘ .npz ’’的二进制文件中

```python
np.savez_compressed(file)

```

### 数组的读取

#### load()函数

读取‘‘ .npy ’’和‘‘ .npz ’’文件中的数组

```python
np.load(file, mmap_mode, allow_pickle, fix_imports)

```

mmap_mode表示内存的映射模式，即在读取较大的Numpy数组时的模式，默认情况下是None





