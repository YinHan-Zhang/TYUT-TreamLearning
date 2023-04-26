

## 基本数据类型

基本数据类型转换：

数字字符串转换为其他基本数据类型 

```
数据类型名.parse数据类型名
```



### 包装类

> java语言里面，基本数据类型不是面向对象的，实际处理有些不便，因此java类库里面提供了包装类
>
> 除了，int和char对应的类特殊一点，Integer和Character ，其他的都是对应基本数据类型大写

```
int n=200;
Integer in = new Integer(n);

String s = "987";
int n = Integer.parseInt(s);
```



### 变量

> 1.常用数据类型变量
>
> 2.引用类型变量



final变量：

> 使用修饰词final变量只能被修饰一次，程序执行过程中不会改变其值

```
public static final String start="程序开始运行！"
```



### 键盘输入：

1.BufferReader 和InputStreamReader

> 输入的不管是文字还是数字，都视作字符串

```java
public static void main(String[] args) throws IOException{
    String str;
    BufferReader buff = new BuffReader(new InputStreamReader(System.in))
    str = buff.readline()
}
等效于：
{
	String str;
	InputStreamReader inp;
	inp = new InputstreamReader(System.in);
	BufferReader buff;
	str = buff.readline(); //buff.read()读单个字符
}
```

2.Scanner

> Scnner对象可直接解析键盘输入的数据
>
> 1. ``reader.nextInt/nextFloat ...next数据类型()``
>
> 2. hasNextInt() 语法：判断键盘输入的类型，返回bool值
> 3. next()/nextLine() : 从键盘上读取一行文本，返回String ,next必须遇到有效数据之后空格回车才结束，nextLine直接结束
> 4. 

```
import java.util.Scanner;
...

Scanner sc = new Scanner(System.in);
int score = sc.nextInt();
...
sc.close();
```

## 数组

```
数据类型[] 数组名  这个括号放前后无所谓，里面不用写数字，java数组声明时不分配空间

数组名 = new 数据类型[个数]
```

常用方法：

| 方法                                  | 描述                           |
| ------------------------------------- | ------------------------------ |
| binarySearch(数组，值)                | 查找值在数组中首次出现的下标   |
| sort(数组)                            | 原地升序排序                   |
| sort(数组,开始排序下标，结束排序下标) | 从开始下标到结束下标排序，原地 |
| copyof(原数组,新数组)                 | 复制数组                       |
| equals(数组1，数组2)                  | 判断数组是否相等               |

局部数据类型推断：var

* var只能用于声明局部变量，不能用于声明方法的参数类型，返回值类型或类的实例变量
* 使用var每次只能声明一个变量
* 不能使用null作为初始值



数组初始化：

> var 不能用来初始化数组

```
 int a[] = {0,1,2,3,4};  
 声明数组不能指定其长度 ， 例如：int a[5];
 
```

java的二维数组，不一定时m*n的数组，可以是不规则的，

```
多维数组初始化也不能指定长度
int [][]x = new int [3][]

x[0] = new int [3]
x[1] = new int [2]

x.length; 返回数组长度

初始化：
int [][]a = {
	{...},
	{...},
	}
	

```

## 字符串

```
String str = new String("hello")
String str = "hello"
```

| 方法                                     | 描述                           |
| ---------------------------------------- | ------------------------------ |
| length()                                 |                                |
| equals(其他字符串对象)                   |                                |
| equalsIgnoreCase(其他字符串对象)         |                                |
| substring（开始位置，结束位置）          |                                |
| charAt(位置)                             |                                |
| getChars(开始位置，结束位置，存放的数组) |                                |
| indexOf(字符)                            |                                |
| lastIndexOf(ch)                          |                                |
| startsWith(前缀)                         |                                |
| endsWith(后缀)                           |                                |
| compareTo(其他字符串对象)                |                                |
| compareToIgnoreCase(其他字符串对象)      |                                |
| replace(字符，替换字符)                  |                                |
| matches(正则表达式)                      | 判断字符串是否与正则表达式匹配 |
| replaceFirst()                           |                                |
| replaceAll(正则表达式，替换字符)         |                                |
| trim()                                   | 去掉首尾空格                   |
| split(正则表达式，分割次数)              |                                |
| toLowerCase() / toUpperCase()            |                                |



StringBuffer 类型字符串

> 可修改的字符串

| 方法                          | -    |
| ----------------------------- | ---- |
| capacity()                    |      |
| ensureCapacity()              |      |
| append()                      |      |
| insert(插入位置，字符串)      |      |
| delete(开始，结束)            |      |
| deleteCharAt(位置)            |      |
| reverse()                     |      |
| substring(开始位置，结束位置) |      |
|                               |      |

### 正则表达式

![IMG_20220307_174536](D:%5CHuawei%20Share%5CHuawei%20Share%5CIMG_20220307_174536.jpg)





#### partern类

| 方法       | 描述                                     |
| ---------- | ---------------------------------------- |
| compile()  |                                          |
| matcher()  | 创建Matcher对象，可以调用Matcher类的方法 |
| pattern()  |                                          |
| toString() |                                          |
| split()    |                                          |

> 正则表达式必须先compile编译成pattern类型的对象，再调用Pattern对象的matcher方法创建Matcher对象，然后调用Matcher类的各种方法执行各种操作

#### Matcher类

| 方法                   | 描述                                         |
| ---------------------- | -------------------------------------------- |
| matches()              |                                              |
| find(开始位置)         | 判断是否匹配，找到一个，下一个从当前位置继续 |
| lookingAt()            | 只有匹配到的字符串在最前面才会返回true       |
| group()                | 返回匹配到的子字符串                         |
| start()                |                                              |
| end()                  |                                              |
| replaceAll(替换字符串) | 用替换字符串替换掉匹配到的字符串             |
| replaceFirst()         | 替换第一个                                   |





## 类与对象


关键字：

```
public
protected
private
缺省 : 同一个包中的类可以访问，即不添加任何的前缀
final :最终修饰符，该方法不能被覆盖，继承
static : 属于类的方法，不用创建对象即可调用

修饰符独有：
----
transient:过渡修饰符，该变量是一个系统保留，暂无特别作用的临时性变量
volatile:易失修饰符，该变量可以同时被几个线程修改和控制

成员方法独有：
----
abstract :抽象类，没有函数体，等着被继承
synchronized：同步修饰符，在多线程程序中，该修饰符用于对同步资源加锁，防止其他资源访问。
native：本地修饰符
```

构造方法的特殊点：

* 从类的一个构造方法中调用另一个构造方法

  > 要使用this语句来调用
  >
  > * this语句必须写在第一行的位置

  ```
  public Cylinder()
  {
  	this(2.5,5,"红色");
  	System.out.println("无参构造方法")
  }
  
  public Cylinder(double r,int h,String str)
  {
  	radius=r;
  	herght=h;
  	color=r;
  	System.out.println("有参构造方法调用")；
  }
  
  ```

公有构造方法和私有构造方法

> 构造方法一般都是公共的，这是因为它们在创建对象时，是在类的外部被系统自动调用的，如果构造方法被声明为private，则无法在该构造方法所在的类的外部被调用，但是在类的内部可以被调用的。
>
> * 想防止用户创建类的对象，就可以将构造方法声明为private，，这种情况下类名只能访问其静态成员

```
private Cylinder()
{
	System.out.println("私有无参构造")；
}

public Cylinder(double r...)
{
	this();
	radius=r;
}
```

对象数组：

```
Person[] per = new Person[3];

Person[0] = new Person("张三"，20)
Person[1] = new ...
```

对象数组还可以作为参数进行调用：

```
public static int minAge(Person[] p)
{
	for(int i=0;i<p.length;i++)
	{
		Person[i].age=age;
		Person[i].name=name;
	}
}
```

JAVA语言的垃圾回收：

* 清除无用的对象和引用，即对象或引用被使用的引用计数器为0时，自动销毁回收内存



### 继承

* extends关键字，可以把父类的非私有成员继承给子类。
* java不支持多继承，只能继承一个父类
* 可以跟多个接口，用逗号隔开

```
class Student extends Persons
{
	...
}
```

* 调用父类中的方法

```
class Person
{
	...
}

class Student extends Persons
{
	public Student(...)
	{
		super(name,age,dep);
	}
}
```



注意点：

* 构造方法不能继承，可以重载
* 系统默认的构造函数，数值类型初始化为0，Boolean初始化为false，引用类型初始化为null
* 构造方法不能使用方法名调用，必须通过new运算符调用，即实例化对象自动调用构造方法

#### 子类中访问父类的成员

> 在子类中使用super不但可以访问父类的构造方法，还可以访问父类的成员变量和成员方法，但super不能访问在子类中添加的成员
>
> ```
> super.变量名
> super.方法名
> ```

super的使用：

* super（）调用构造函数，要放在第一行

this的使用：

* 类方法中不能出现this，类方法就是使用static修饰的方法
* static修饰的成员中不能使用this



子类对象可以向上转换为父类对象,反过来不行

```
SuperClass s = new SubClass();
```



#### 覆盖

> 与重载类似，在子类中定义与父类方法具有一样的方法名、参数列表、返回值类型
>
> * 覆盖的方法不能被继承
> * final和static方法不能被覆盖，final还不能被继承
> * 可扩大父类方法的权限，不能缩小



#### final关键字

final修饰的类：为最终类，不能被继承

final修饰的成员变量：不可以被修改

fianl修饰的成员变量：不允许修改

fianl修饰的成员方法：不能被子类重新定义（覆盖重写）





#### Object类

| 方法       | 说明                                                     |
| ---------- | -------------------------------------------------------- |
| equals()   | 判断两个对象变量是否指向同一个对象                       |
| toString() | 将对象转换为字符串                                       |
| getClass() | 返回运行时对象所属的类                                   |
| clone()    | 返回该对象的一个副本                                     |
| instanceof | 判断指定对象是否时某类的示例    对象变量 instanceof 类名 |
|            |                                                          |

== 和 equals的异同：

== ：用于比较两个变量本身的值，即两个对象在内存中的首地址

equals：比较两个字符串中所包含的内容是否相同

相同：都是来比较两个对象变量是否指向同一个对象

#### 抽象类

```
abstract class Shape{
	protected String name;
	public Shape (String xm){
		name xm;
		System.out.print(..)
	}
	abstract public double getArea();  抽象方法
	abstract public double getArea();
}
```



### 接口

> 接口interface，接口具有数据成员、抽象方法、默认方法和静态方法
>
> 和抽象类的区别：
>
> （1）接口的数据成员都是静态的而且必须初始化，即数据成员必须是静态常量，一定要有初值
>
> （2）接口中除了声明的静态方法外，还可以定义私有方法、静态方法和默认方法，但是不能定义一般方法
>
> * 接口与抽象类一样不能用new运算符直接的创建对象，相反，要利用接口的特性来创建一个类，然后创建对象
> * 接口里面除了私有方法，其余都是public
> * 抽象方法默认使用public 

利用接口创建新的类，叫做接口的实现 。

接口类的实现类似于继承，但是关键字由extends换成了inplementation

* 接口其实就是抽象父类
  * 接口变量就是 用接口名创建的父类对象变量
  * 接口回调就是创建的接口父类对象去调用子类重载的方法，这个过程交接口回调

```java
interface Face1
{
    static final double PI=3.14;
    abstract double area();
}

interface Face2
{
    abstract void setColor(String c);
}

interface Face3 extends Face1,Face2
{
  	abstract void volume();
}

public class Cylinder implements Face3
{
    private double radius;
    private int height;
    private String color;
    public Cylinder(double r,int h)
    {
        radius =r;
        height =h;
    }
    public double area()
    {
        return PI*radius*radius;
    }
    ...
    依次实现定义的抽象类
}
```

注意点：

* 接口可以多继承，类只能单继承
* 接口的方法，必须全部是抽象的

### 定义一下各种类方法

* 静态方法不能被继承，也不能被类实现
* 静态方法和默认方法可以有方法体
* 默认方法只能通过对象调用，方法名不可以调用
* 多继承接口重名： 接口名.super.默认方法名()

```java
interface Face{
    final static double PI=3.14;  	//定义常量
    public default double area(int r) 	//定义默认方法
    {
        return PI*r*r;
    }
    public abstract volume(int r ,double h); 	//定义抽象方法
    public static String show()			//定义静态方法
    {
		Sysstem.out.print("...");
    }
}

public class App() impletments Face
{
    public double volume(int r,double h)
    {
        return area(r)*h;   
        //等价于 return Face.super.area(r)*h;
    }
}
```



#### 枚举类型

> 自带序号的列表对象

| 方法         | 说明                                       |
| ------------ | ------------------------------------------ |
| name         | 返回枚举变量的名称                         |
| ordinal      | 返回枚举成员的序号                         |
| toString     | 返回枚举变量的名称                         |
| values       | 返回枚举类型的数组，每个元素包含序号和名称 |
| valueof(str) | 返回str的枚举成员                          |
|              |                                            |

```
enum Direction
{EAST,SOUTH,WEST,NORTH}

Direction dir = Direction.EAST;
Direction dir1 = Direction.valueOf("North");
System.out.print(dir);
for(Direction:d,Direction.values())
	System.out.print(d.ordinal()+ d.name());

枚举类型还可以定义成员变量和成员方法，像类一样

enum Direction{
	EAST("东"),SOUTH("南"),WEST("西"),NORTH("北");
	private String name;
	private Direction(String name)
	{
		this.name=name;
	}
	public String toSting()
	{
		return name;
	}
}

public static void main()
{
	Direction dir = Enum.valueOf(Direction.class,"NORTH"); 给dir赋上NORTH的值
	System.out.println(dir);
	for(Direction d:Direction.values)
		System.out.println(d.name()+d.ordianl()+d.toString());
}
```





### 注解

> 注解也就是元数据，描述数据的数据，用于告诉编译器要做什么事情，在程序中可对任何元素进行注解
>
> 注解一般放在变量的前面，对变量进行解释说明，然后通过反射机制对注解进行访问

基本注解名：

@ Deprecation：该注解用于表示某个程序元素、类、方法已过时，不建议使用

@Override：该注解只用于**方法**，用来限定必须覆盖父类中的方法

@SuppressWarning:该注解用于抑制警告信息的出现

#### 自定义注解：

```
@interface 注解名{
	数据类型 成员变量名() [defalut 初始值]；
}

@interface Author{
	String name();
	int itAge default 0;
}

@Author(myname="刘洋")

```





### 反射类：

> 即类的多态，该对象在编译时和运行时类型不同，但程序又需要在运行时查看类和对象的动态信息
>
> 反射机制就解决了程序运行中获取动态消息的问题。



### 内部类：

>  把一个类放在另外一个类里面，里面的类称为内部类，外面的类：称为外部类。
>
> 内部类本质上时外部类的一个成员，因此内部类可以直接访问外部类的私有成员
>
> 特性：
>
> * 内部类和外部类不能重名
>
> * 实例化的时候，内部类名前面还得加外部类  
>
>   * 内部类初始化
>
>     ```
>     声明： OuterClass.InnerClass
>     Outer out = new Outer()
>     
>     内部类初始化：OuterClass.InnerClass inner = out.new InnerClass(形参列表)
>     静态内部类初始化：OuterClass.InnerClass inner =  new OuterClass.InnerClass(args)
>     ```
>
>     
>
> * 内部类可以访问外部类的成员和变量，也可以访问内部类的成员和变量
>
> * 静态内部类属于外部类本身，而不属于外部类的对象，所以只能访问静态成员，不能访问实例成员（对象）
>
> 外部类和内部类的变量区分：
>
> ```java
> class Outer()
> {
> 	int a=1;
> 	class inner()
> 	{
> 		int a=2;
> 	}
> 	
> }
> 
> 从里到外依次搜索： 内部类局部变量-》内部属性-》外部变量——》外部属性
> 
> 访问内部类属性：this.a
> 访问内部类变量：a
> 外部类访问：Outer.this.a
> 
> 其实就是类作用域的变量区分
> ```
>
> 匿名内部类：
>
> > 在函数/类  内部使用一个（）来定义类，没有名字，定义在括号里面，括号就创建匿名对象
>
> ```
> public static void func()
> {
> 	(new Inner()
>     {
>         void setName(String n)
>         {
>             name =n;
>             System.out.print(name);
>         }
>     }).setName("zyh")
> }
> 
> 传参传对象的时候现场重新定义类对象
> 
> class mytype
> {
> 	public void OutShape(IShape s){
> 	...
> 	}
> }
> 
> public static void main(...)
> {
> 	mytype a =new mytype();
> 	a.outShape( new IShape(){
> 	@override
> 	public void shape(){
> 	System.out.print("内部类重新定义")
> 	}
> 	} )
> }
> 
> ----------------------------------------------------------------------------------
> lambda表达式必须式函数式接口
> 
> @FunctionalInterface #注解声明是函数式接口
> public interface Runable(void run();)
> 
> lambda表达式： (形参列表)->{代码块}
> 
> 形参类型，可用var进行类型推断
> 如果代码块只有一条语句，花括号可以省略，且这条语句的值会作为返回值，即省略return
> 
> interface IntFun
> {
> 	double dis(int n);
> }
> 
> main()
> {
> 	使用匿名内部类
> 	IntFun fun = (new IntFun(){
> 	public double dis(int i){return 2*i;}
> 	}
> 	);
> 	等价于(使用lambda表达式)：
> 	IntFun fun = (var i)-> 2*i;
> 	
> 	double m = fun.dis(3);
> }
> ```
>
> lambda表达式的作用在于对某个函数的参数进行快速处理然后返回

### 小结

1. 注解即，元数据，所谓的元数据，就是用来描述数据的数据

2. 格式：@注解名  

   1. 按照作用分三类：

      基本注解、元数据注解、自定义注解

3. 匿名内部类不能同时继承一个类又实现一个接口，也不能实现多个接口

4. 函数式接口就是指 只包含一个抽象方法的接口

5. Lambda表达式只适用于函数式接口

6. 





### 类的总结



1. 通过extends关键字，可以将父类的非私有成员（成员变量和成员方法）继承给子类
2. 在执行子类的构造方法之前，如果子类没有使用super语句指定调用特定父类的构造方法，编译器在执行子类的构造方法之前先会执行父类的默认构造函数，其目的是帮助继承自父类的成员做初始化
3. 用this来区分同名的局部变量和成员变量
4. this（）和super（）方法必须放在构造方法的第一行，this（）可以调用本类其他的构造函数，所以this和super在构造函数内不能共存
5. final关键字可以避免子类覆盖
6. final关键字如果放在成员变量的前面，那么这个成员变量就变成的了常量，因为不能再进行修改了
7. java所以的类都继承自Object类
8. 抽象类的方法分为两种：一般方法和以abstract定义的抽象方法（没有定义方法体）
9. 抽象类和接口的区别：
   1. 接口的数据成员都是静态的且必须初始化
   2. 抽象类的数据成员可以为一般的成员变量
   3. 抽象类可以声明一般方法
   4. 接口中**不能**声明一般方法，但是可以声明私有方法、默认方法、静态方法
10. java类不能多继承，只能单继承，接口可以多继承
11. **接口回调**： 声明接口类型的变量，并用它来访问类所实现该接口的方法，这种访问方式称为接口回调
12. 枚举是一种特殊的类，是引用类型
    1. 枚举成员也称为枚举实例/对象



# 异常处理机制

语法错、语义错、逻辑错

错分两类：错误和异常

错误：程序执行过程中所遇到的硬件或操作系统错误。

异常：在硬件和操纵系统正常时，程序遇到的运行错。

* try抛出异常后后面的代码不再运行
* catch块中有System.exit(0) 则不执行finally语句 ，有return语句则会执行完finally语句再结束

## 抛出异常

```
方法体内：
	throw 异常类对象;
方法头：
	定义完方法后，加上 throws 异常类列表
```

打开文件自动读取文件内容，然后自动关闭

```
try(Scannner in = new Scanner(Path.get("./file.txt"));)
{
	while(in.hasText())
		in.nextline();
}
```

自定义异常类：

* 自定义异常类通常是Throwable类型的子类，java推荐以Exception为父类
* 可以为异常类定义属性和方法
* 这些属性和方法可以体现该类所对应的错误信息
  * 习惯上在自定义类中加入两个构造方法：
    * 无参构造方法
    * 含有字符串类型参数

异常处理不外乎两种方式：

1. 在try-catch块内部使用throw语句抛出异常
2. 方法头使用thows Exception 抛出异常，交给上一层调用结构去处理





# 多线程

## 一 基本概念







#### 线程间共享资源：

> Thread和Runable接口都可以创建多线程，但它们的区别就在于对数据的共享上
>
> Runable可以轻松实现数据共享，只要在一个公共的（父）对象上创建多个线程就可以

```
ThreadSale t = new ThreadSale(); 	//父对象
Thread t1 = new Thread(t,"第一售票窗口");
Thread t2 = new Thread(t,"第二售票窗口");
Thread t3 = new Thread(t,"第三售票窗口");
```

#### 线程同步

> 线程同步可以用互斥的控制方式来解决，互斥意味着一个线程访问共享资源时，其他线程不允许访问
>
> synchronized关键字来标识同步的资源

```
同步语句：
synchronized(线程对象)
{
	临界代码段
}

同步方法：
public 返回类型 方法名()
{
	方法体
}
```



jion方法：即插入线程，且插入后必须等插入线程运行结束后才能运行其他的线程



##  二 等待/唤醒机制

为了解决循环等待消耗CPU以及信息及时性问题，Java中提供了等待唤醒机制。通俗来讲就是由主动变为被动， 当条件成立时，主动通知对应的线程，而不是让线程本身来询问。

2.1 基本概念#

等待/唤醒机制，又叫等待通知(笔者更喜欢叫唤醒而非通知)，是指线程A调用了对象O的wait()方法进入了等待状态，而另一个线程调用了O的notify()或者notifyAll()方法，线程A收到通知后从对象O的wait()方法返回，进而执行后续操作。

上诉过程是通过对象O，使得线程A和线程B之间进行通信, 在线程中调用了对象O的wait()方法后线程久进入了阻塞状态，而在其他线程中对象O调用notify()或notifyAll方法时，则会唤醒对应的阻塞线程。

2.2 基本API#

等待/唤醒机制的相关方法是任意Java对象具备的，因为这些方法被定义在所有Java对象的超类Object中。

notify： 通知一个在对象上等待的线程，使其从wait()方法返回，而返回的前提是该线程获取到对象的锁

notifyAll: 通知所有等待在该对象上的线程

wait: 调用此方法的线程进入阻塞等待状态，只有等待另外线程的通知或者被中断才会返回，调用wait方法会释放对象的锁

wait(long) : 等待超过一段时间没有被唤醒就超时自动返回，单位是毫秒。



## 线程小结：

1. 多任务和多线程的区别： 
   1. 多任务针对操作系统而言，可以同时运行多个应用程序
   2. 多线程是针对一个程序而言的，表示一个程序内部运行多个线程
2. run（）方法给出线程要执行的任务。必须把线程要执行的代码放在run函数内部
3. 每一个线程有5种状态：新建状态、就绪状态、运行状态、阻塞状态、就绪状态和消亡状态
4. 线程休眠两种方法：
   1. wait（）方法，可用notify（）方法解除阻塞
   2. sleep方法，单位1/1000s，等待休眠时间后自动解除阻塞





# 泛型

泛型定义：

```
泛型类：【修饰符】 class 类名 <T>

泛型接口：【public】interface 接口名 <T>

泛型方法：【public】 [static] <T> 返回值类型 方法名（T参数）

实例化：

类名<类型实参列表>对象名= new 类名<类型实参列表>([构造方法参数列表])
```

* 泛型实例化的过程中，实际类型必须是引用类型，即必须是类类型，不能用int、double、char来替换类型参数T
* 一个static方法，无法访问泛型类的类型参数，所以如果static方法想使用泛型的能力，必须使其成为泛型方法

* 静态环境里面不能使用泛型类的参数T
* 异常类也不能是泛型

限制泛型的可用类型

```
class ClassName <T extend anyclass> 
无论是类还是接口，泛型限制时都必须使用extends关键字
```

泛型通配符

```
泛型类名 <? extends T> obj = null;
```



泛型数组

```
public class Test<T>
{
	private T[] array;
	void setT(T[] array)
	{
		this.array=array;
	}
	T getT()
	{ return array;}
}

Test <String> a = new Test<>();
String[] array = {".."...};
a.setT(array);

for(int i=0;i<a.getT.length();i++)
	System.out.println(a.getT()[i])

```



```
Arrays.asList("你好", "简单教程", "简单编程").forEach(System.out::println);


Map<String, Integer> items = new HashMap<>();
        items.put("A", 10);
        items.put("B", 20);
        items.put("C", 30);
        items.put("D", 40);
        items.put("E", 50);
        items.put("F", 60);

items.forEach((k,v)->System.out.println("Item : " + k + " Count : " + v));

        items.forEach((k,v)->{
            System.out.println("Item : " + k + " Count : " + v);
            if("E".equals(k)){
                System.out.println("Hello E");
            }
        });
        
import java.util.Arrays;

public class ForEachTester
{
    public static void main(String[] args)
    {
        Arrays.asList("你好","简单教程","简单编程")
            .stream()
            .filter(s->s.contains("简单"))
            .forEach(System.out::println);
            //等价于.forEach((s)->System.out.print(s+" "))
    }
}


import java.util.Arrays;

public class ForEachTester
{
    public static void main(String[] args)
    {
        Arrays.asList("你好","简单教程","简单编程")
            .stream()
            .filter(s->s.contains("简单"))
            .forEach(System.out::println);
    }
}
```







# 第三章 基本语法元素

## 标识符

> 标识符由字母、下划线和美元$字符组成的字符串
>
> **开头不能是数字**

1. 关键字不能作为标识符
2. 布尔常量true和false不能作为标识符
3. null不能作为标识符
4. “-”符号也不能开头

## 注释

1.单行注释： 以“ // ” 开头，后面都是注释

2.多行注释：以“ /* ”开头，以“ */ ”结尾  ，中间的都算注释。可以多行，也可以单行。

3.文档注释：“ /* * .....* * / ” ,属于多行注释，用于生成程序的帮助文档。

## 分隔符

分为空白符、注释、普通分隔符

空白符：

```
制表符 \t
换页符 \f
回车	\r
换行	\n
```

普通分隔符：

```
() , {}  [] ; . :		这些都是
```

## 数据类型分类：

数据类型分为：基本数据类型 和 引用数据类型

基本数据类型：

> 数值类型：
>
> ​	整形：byte（1字节）、short（2字节）、int（4字节）、long（8字节）
>
> ​	浮点型：float（4字节）、double(8字节)
>
> 字符型：char（2字节）
>
> 布尔型：boolean

引用数据类型：

> 类：Class
>
> 接口：Interface
>
> 数组：Array

注意点：

```
定义小数时，默认double类型
double n = 1.23;
float n=1.23报错   应改为 float n=1.23f ; 浮点数可带后缀，标识类型
```



## 运算符的优先级

```
1. () [] .
2. ++ -- ! ~   	注意：从右往左算
3. * / %
4. + -
5. << >> 位运算
6. < > <= >= instanceof 
7. == !=
8. &  (按位的与或非，先于逻辑与或非)
9. ^
10. |
11. ?:
12. = += -= *= /= %= 等等  从右往左
```

