# Scala

## 变量定义

var：定义完可变

val：定义完不可变



## 函数

定义函数最常用的是作为某个对象的成员

### 头等函数

函数字面量：（x:Int , y:Int）=> x+y 

箭头是指明的意思



```scala
var increase = (x:Int) => x+1

val numbers = List(-11,-10,-5,0,5,10)

numbers.foreach(
    (x:Int)=> println(x) )

numbers.filter(
	(x:Int)=> x>0 )
```

* 短格式：去除参数类型

```
numbers.filter(x => x>0)
```

* 占位符语法：节省参数

```
numbers.filter( _>0 )
```

* 部分应用函数(占位符和函数之间的空格必须有！！！)

```
numbers.foreach(println _ )  ==  numbers.foreach(x=>println)
```



### 运用

* 乘法表

```scala
def makeRowSeq(row: Int) = 
	for(col <- 1 to 10) yield {
        val prod = (row * col ).toString
        val padding = " " * (4 - prod.length)
        padding + prod
    }


```



## 方法

避免使用返回语句，尤其是多条返回语句，

	1. 代之以把每个方法当作是创建返回值的表达式，最后一行语句就是返回值
 	2. 把大的方法分解为多个小的方法

3. scala很容易编写多个显示的return语句



## Singleton对象

> scala不能定义静态成员，而是代之以定义单例对象（object），除了用object关键字替换class以外，单例对象的定义与类定义一致



scala类中不能定义静态变量，所以单例对象就是拿来定义静态变量的



对象与类的差异：

1. 单例对象不能带参数，类可以。因为单例对象不是用new关键字实例化的，没机会传参
2. 

![1660989958500](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1660989958500.png)



## For表达式

### 枚举

```
for(i<- 0 to args.length-1)
	println(i)
```

> 这种枚举很少见，因为集合对象本来就可以被枚举，为什么还需要索引值嘞？而且索引还容易溢出

### 过滤

```
for(i <- args if args.getName.endWith(".scala")) 
	println(i)
	
for(
file<- files
if file.isFile
if files.getName.endWith(".scala")
) println(file)
```

* 过滤器还可以不断的添加，加if语句就行



### 过滤间变量绑定

> 在for循环里面，有一些变量需要复用，可以在for循环中定义一个变量来复用

```
for(i <- args if args.getName.endWith(".scala")) 
	println(i)
	
for(
file<- files
if file.isFile

line <- fileLines(file) #嵌套循环

trimmed = line.trim   #流间变量

if trimmed.matches(pattern)

) println(file + ":" + trimmed)
```

* 流间变量可以在for循环多个地方进行使用



### 推导式

> 一个集合去创造另一个集合  

`for  {子句}  yield {循环体} `



```scala
val newSet = 
	for(
    	file<- files
    	if file.isFile

        line <- fileLines(file) #嵌套循环

        trimmed = line.trim   #流间变量

        if trimmed.matches(pattern)

        ) yield trimmed.length
```

### 函数闭包

* 减少代码重复

```scala
object FileMatcher {
    private def filesHere = (new java.io.Files(".")).listFiles
    
    def filesEnding(query: String) = 
    	for(file <- filesHere; if file.getName.endWith(query))
    		yield file
}
```

* 检查类型

```scala
def filesEnding(query: String，
				matcher:(string,string) => Boolean ) = {
    	for(file <- filesHere; if file.getName.endWith(query))
    		yield file
    		}
```

* 多功能简化

```scala
def filesEnding(query:String) = 
	filesMatching(query,_.endWith(_))
def filesContaining(query:String) = 
	filesMatching(query,_.contains(_))
```



## 变量范围

在scala中所有的变量都存在有效范围，注意局部变量！！！

* 一定要注意你的花括号
* 如果无法避免，改变程序思路 ， 简短函数代码长度





## 语言特性

1. 用if语句替换continue 
2. 用布尔变量替换 break
3. **用递归代替循环**





## 富包装器

```
0 max/min 5

-2.7 abs/round

4 to 6

"bob" capitallize
"robert" drop 2
```

## 列表

List不提供append方法，因为每添加一个耗时太长，可以这样解决：

* 1.先把它们的前缀加进去然后调用reverse
* 2.用ListBUffer ，然后toList



#### 用法合集：

```
list.count(s=>s.length==4)

list.forall(s=>s.endWith("l"))

list.foreach(s => println(S))

list.map(s => s+"y" )  列表内每一个元素加y

list.sort( (s,t) =>
	s.CharAt(0) < t.CharAt(0) )
```





# GraphEmbedding



## 问题定义

给定一个离散型随机变量的概率分布规律 P(X=i)=pi,i∈1,...N ，希望设计一个方法能够从该概率分布中进行采样使得采样结果尽可能服从概率分布 P











# Angel组件解读



## PSModel

PSModel用于工作线程来操作PSServer上的分布式模型（矩阵）分区

![1661005450578](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1661005450578.png)

## Context

生成和保留上下文

![1661005644131](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1661005644131.png)



## 模板

![1661006563459](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1661006563459.png)