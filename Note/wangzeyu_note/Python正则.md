# Python正则表达式模块：re

![lcp19tuq.png](https://www.9998k.cn/usr/uploads/2023/01/2825772936.png)

### 匹配方法

**compile**: 封装正则表达式

**search**: 从字符串中寻找，返回re.Match object **对象**

**match**: 从字符串开始处匹配，返回re.Match object **对象**

**findall**: 以列表形式返回所有匹配项，返回匹配结果的 **列表**

**finditer**: 

**fullmatch**: 

**sub**: 

**subn**: 

**split**: 

### re.Match object 对象方法

找不到会返回**None**，以下均**无效**，需在使用前判断是否**非None**

**group**: 获取子模式(组)的匹配项

+ group(): 匹配的是整体
+ group(0): 同上 
+ group(n): 匹配第n个小括号内容**(从1开始，没有括号或超范围会报错)**

**groups**: 返回所有小括号匹配到内容组成的**元组**

+ groups()

### 括号用法

re模块中的小括号与正则语法中的小括号有所不同

+ **()**: 分组，用于group匹配
+ **(?:)**: 相当于正则语法中的小括号

*未完待续...*

括号的使用