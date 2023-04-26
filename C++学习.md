# 							C++ 学习

计算数组长度：sizeof(a)/sizeof(int)   或者sizeof(a)/sizeof(a[0])

###  基础概念：

C++程序结构由 注释、编译预处理命令、程序主体组成 。

以‘#’ 开头的行，叫做 编译命令预处理行 ，``#include ``称为**文件包含预处理命令**。

> 开始的开始
>
> 1）定义变量，意味着给变量分配内存空间，用来存放变量值。
>
> ​	变量有且只能定义一次，但是可以声明无数次
>
> 2）一个函数必须要在声明后才能使用（被调用），**函数定义包含函数声明**
>
> 3）



namespace 命名空间 

```
using namespace std;
```

在命名空间std中，所有标准库的定义都在命名空间std中。

但是有个副作用，当使用标准库中的一个名字时必须显示说明使用来自std空间中的名字。

```
while(std::cin>>value)
当while的条件时无限输入时，需要输入一个文件结束符来终端while循环
windows系统中文件结束符：Ctrl+Z
UNIX和MAC 系统是 Ctrl+D
```



### 怎么写头文件：

```C
#ifndef 头文件名
#define 头文件名

... 不写主函数

#endif
    
导入"xxx.h" 头文件是没有主函数的
导入"xxx.cpp"头文件 可以直接调用头文件里面的函数
```

头文件里写什么呢？

1. 类型声明
2. 函数声明和定义
3. 内联函数定义
4. 全局变量声明、常量定义
5. 数据声明
6. 结构体定义

### C++格式化输入输出：

* 在内存中，字符数据以ASCII码储存。字符数据可以和整形数据可以相互赋值，只要注意其表示的合理范围即可

  ```c
  int a ='b';
  char c = 97;
  ```



* 字符和字符串是不一样的，**字符串总是以‘\0’结尾**，**内存单元多一个**

* 单引号表示字符，双引号表示字符串，内存单元不一样，**‘0’和”0“不一样**

  ```c
  字符串不能赋给字符变量
  char c = "abc" //错的
  ```

  **字符串实际上字符指针类型！！！**



## 文件操作：

文件流： 

1. ofstream：写文件
2. ifstream：读文件
3. fstream：读写

写文件步骤：

1. 包含头文件：		#include <fstream>
2. 创建流对象：        ofstream ofs;
3. 打开文件：ofs.open(“文件路径”， 打开方式)
4. 写数据：ofs<<“写入的数据”
5. 关闭文件：ofs . close();



| 头文件   | 类型                                           |
| -------- | ---------------------------------------------- |
| iostream | istream,wistream 从流中读取数据                |
|          | ostream,wostream 向流中写入数据                |
|          | iostream、wiostream 读写流                     |
| fstream  | 同理                                           |
| sstream  | istringstream/ostringstream 在string中读写数据 |
|          | stringstream 读写                              |



文件打开方式：

```
ios ::in	只读
ios ::out	只写
ios ::ate	从文件尾开始
ios ::app	追加写文件，从尾部开始
ios ::trunc	先删除再创建
ios ::binary	二进制方式打开

```

指定文件模式：

* 只对ofstream或者fstream对象设定out模式
* 只对ifstream或者fstream对象设定in模式
* 只有当out被设定时，才可以设定trunc模式
* 在app模式下，即使没用显示指定out模式，文件总是以输出的方式被打开

读文件：

```c++
ifstream ifs;
ifs.open("文件名.txt", ios::in)
if(!ifs.is_open())
{
    cout<<"文件打开失败"<<endl;
    return;
}
#方法1
char buf[1024]={0}
while (ifs>>buf)	#读到结束
{
    cout<<buf<<endl;
}
#方法2
char buf[1024]={0};
while (ifs.getline(buf , sizeof(buf)))		#getline(要保存的数组，最大长度)
{
    cout<<buf<<endl;
}

#方法3
string buf;
while (geline(ifs, buf))	#getline（流对象，储存的字符串）
{
    cout<<buf<<endl;
}
```



#### 管理缓冲区：

* 程序正常结束，作为main函数的return操作的一部分，缓冲刷新被执行
* 缓冲区满的时候需要刷新缓冲，而后新的数据才能继续写入缓冲区
* 我们可以操纵如endl 来显示刷新缓冲区
* 在每个输出操作之后，我们可以操纵符unitbuf设置流的内部状态，来清空缓冲区。默认情况下，对cerr是设置unitbuf的
* 一个输出流可能关联到另一个流。当读写被关联的流时，关联到的流的缓冲区会被刷新。例如cerr和cin都关联到cout，因此读cin或写cerr都会导致cout的缓冲区被刷新

#### 刷新缓冲区

```
cout<<"hi!"<<endl; 输出hi和一个换行，然后刷新缓冲区

cout<<"hi!"<<flush; 输出hi，然后刷新缓冲区，不附加任何额外字符

cout<<"hi!"<<ends; 输出hi和一个空字符，然后刷新缓冲区

如果想在每次输出操作后都刷新缓冲区，我们可以使用unitbuf操作符，它告诉流在接下来的每次写操纵之后都进行一次flush操作。
cout<< unitbuf ; 

cout<<nounitbuf ; 回到正常的缓冲方式
```

tie方法可以关联流： 

```
cin.tie(&cout)
cin.tie(&cerr)
```





### 文件读写：

fstream（读写）的操作：

```
fstream fstrm (s,mode); 打开名为s的文件，以mode的形式打开

fstrm.open(s)
fstrm.close(s)
fstrm.is_open(s)
```

如果我们创建的流对象后面提供了一个文件名，则open函数会被自动调用

```
ofstream 自定义名字("文件路径.txt")

名字<<'.......\n';

判断是否打开 名字.fail()==-1 表示失败  ==1 成功

ifstream 自定义名字("文件路径.txt")

名字>>变量>> ...
```



#### string流

> 用来处理文本数据的







#### I/O流的书写格式：

```c
cout<<" ...";
cout<<"..."
    <<"..."
    <<endl;

cin>>i>>j>>z;
```

使用cin进行输入的时候，cin用空白符和行结束符将各个值分开。

如果在cin/cout/cerr前面加w，表示宽字符的输入输出： wcout / wcin / wcerr

#### getline（）

getline(字符数组，本次读取的最大字符个数，分隔字符作为读取结束的标志)

字符数组：用于放置读取的文本

```
cin.getline(str,sizeof(str),'X');
```

#### get()

cin.get()读取单个字符

#### put()

cout.put () 单个输出 



#### 控制符：

头文件：#include<iomanip>

| 控制符                       | 功能-                   |
| ---------------------------- | ----------------------- |
| setprecision（小数保留位数） | 控制小数的精度，默认6位 |
| setw(空格数)                 | 控制输出宽度            |
| setfill(填充的字符替换空格)  | 设置填充字符            |
| left                         | 左对齐                  |
| right                        | 右对齐                  |

#### 详解函数细节

* setfill( ‘*’ )  setfill里面时字符，不能用双引号

* fixed与setprecision（）合用，表示以指数表示的小数位数

  ``cout<<fixed<<setprecision(n)<<amount<<endl;``

* setw（n）：设置值的输出宽度，如果一个值的字符数比setw设定的字符数要少，就显示空白

  * 但是 setw（n）**只影响下一个的输出**

* 输出控制符连用

  ```C++
  #include<iomanip>
  cout<< setfill('*')
      << setw(3)
      << left
      << 2 <<endl;
  cout<< setfill(' ')<<endl;
  ```



左值与右值：

> 左值：能出现在赋值表达式左边的表达式 。 左值表达式具有存放数据的空间，并且存放是允许的。
>
> ``int a = 6;``  a就是左值
>
> 常量不能是左值，因此C++规定常量的数值一旦确定是不能更改的。
>
> 右值：只能出现在赋值表达式右端。左值表达式也可以作为右值表达式
>
> 



自增自减运算 ++  -- ：

>  先运算还是先赋值，搞清楚就好 。
>
> 如果出现多个自增自减运算，取前面两个作为自增自减运算 
>
> 太多个连在一起，怎么区分？
>
> 1.连在一起的最多三个，a+++ a =（a++）+ （a）
>
> 2.太多了，就用空格  a+++++a = (a++)+(++a)  即``a++ + ++a``



语句二义性：

> if和else的歧义，if嵌套，后面的else到底和谁匹配？
>
> C++规定：else语句和（自下往上的）第一个位匹配的if匹配



## C++基础

### 变量

整数类型

- 整数类型分为**数值整数类型**和**字符类型**。

- 数值整数用来表示**没有小数部分**的数字。
  - 数值整数可以按照**占用内存**大小分为`short`、`int`、`long`以及`long long`这四种，占用内存越大的类型能表示的数值范围就更大。
  - 同时又可以按照是否表示负值分为**有符号**版本和**无符号**版本  
- 字符类型则专门用来存储计算机中的**基本符号**：英文字母、数字以及标点等。

- - `short`类型至少占据**2**个字节，即16位；一般占用2字节；
  - `int`在现代系统中一般占用**4**个字节，即32位；类型长度大于等于`short`类型；
  - `long`类型长度至少占据**4**个字节，且大于等于`int`类型；一般占用4个字节；
  - `long long`类型长度至少占据**8**个字节，且大于等于`long`类型；一般占用8个字节。

Tips： 为了避免不同的计算机系统的设计差异，C++提供了一个标准来保证各种类型的**最小长度**



 实型：分为三种

float ：4位  小数后保留后6位

double：8位

long double：16位



- 计算机内存采用**二进制**的存储方式，每一个位可以表示0与1两种状态，因此占据n*n* 位的内存块，可以表示 2^n2*n* 个不同的数字。
  - 比如一个 8 位的内存块，可以表示 2 的 8 次方个不同的组合，也就是说能表示 256 个不同的整数；
  - 32 位的`int`类型，可以表示 2^{32}=4294967296232=4294967296 个不同的整数。  
- 每个类型数据可以分别指定**有符号版本**和**无符号版本**，用来明确该类型是否需要表示负值。
  - 比如`unsigned int`就表示无符号的`int`类型，只能表示正值；
  - `signed int`就表示有符号的`int`类型，可以表示负值。
  - 在不指定有无符号时，都默认是有符号版本。  
- 如果是**无符号**版本，那么一个8位的内存块可以一一对应到**0～255**之间的整数；
- 如果是**有符号**版本，那么就会考虑负数，这个8位的内存块可以表示**一128～127**之间的整数。



- 在三种浮点类型中，更大的内存空间可以表示更多的有效位数：
  - `float`类型通常占用**4**个字节，有效位数为6位
  - `double`类型占用的空间是`float`类型的两倍，即**8**个字节，有效位数为15位
  - `long double`类型一般占用**16**个字节的空间

 ### 运算规则

操作数为字符或短整形时，系统自动转换成整形。操作数为实型时，系统自动转换成双精度型。当两数操作数类型不同时，将精度低（或表示范围小的）的操作数的数据类型变换到与另一操作数类型相同再进行运算。



### 函数

注意的点：

* 程序总是从主函数开始启动。

* C++不允许在一个函数的内部再定义一个函数

* 函数原型不必包含参数名字，只需要包含参数类型，函数声明就是函数原形

  ``int Area(int,int)``

* 全局变量定义之前的所有函数定义不知道该变量

* 局部变量没有默认初始化，如果局部变量不被显示初始化，**值就是随机的**

* 静态局部变量和全局变量一样，如果不初始化，自动赋值为0

* 函数的静态局部变量，只创建一次，再次调用不会重新分配空间，它始终驻留在全局数据区，直到程序运行结束

* 

#### 内联函数：

> 在写程序的时候，频繁的调用一个1-5行的小函数，浪费空间效率低
>
> 直接写原代码，可读性又差

内联函数的意义：编译器看见inline后，就为该函数创建一段代码，以后碰见这个函数就用一段代码来替换

**注意点：**

* 内联函数里面不能有复杂的控制语句，例如switch和while
* 内联函数适用于1-5行的小函数
* **内联函数必须先声明后调用**

```c++
#include<iostream>
using namespace std;

inline int isnumber(char);

main()
{
    char c;
    while((c= cin.get())!='\n')
    {
        if(isnumber(c))
        {
            cout<<"you input a digit"<<endl;
        }
        else
            cout<<"you input not a digit"<<endl;
    }
    system("pause");
}

inline int isnumber(char ch)
{
    return (ch>='0'&&ch<='9')?1:0;
}
```



#### 重载函数：

如何确定调用哪一个函数？ **严格的形参类型定义！**

> 1.严格匹配
>
> ​		int : 0,char,short int 都是严格匹配
>
> ​		double: float属于严格匹配
>
> 2.内部类型转换
>
> 3.用户定义的转换
>
> 

* 重载函数 仅仅是返回类型不同，无法区分

* 怎么区分？ 

  1.返回类型

  2.参数个数 

  3.参数类型 
  
  4.参数顺序

默认函数参数的定义顺序： 

​		**默认参数应该从右至左逐个定义，调用函数时，只能从左匹配参数**



### 静态

#### 外部储存类型extern

> 假如一个程序由10个源文件构成，每一个源文件都必须访问一个全局变量或者一个函数，在这种情况下，其中的9个文件必须把变量声明为extern，剩下那一个就正常定义。

定义的文件下：

```
void fn1(int,int);
int n;
```

其他的文件下：

```
extern void fn1();
extern int n;
```

**带extern的变量只是变量声明，不是变量定义**

不能使用这样的语句：``extern int a=3;``



#### 静态全局变量

在单文件构成的程序中，全局变量和全局静态变量是没有区别的。

在多文件中，是有区别的。

**全局静态变量使得该变量成为由定义该变量的文件独享的变量，其他文件不能访问。**

* 不必担心另外的源文件使用它的名字，该名字在源文件中唯一
* 源文件的全局变量不能被其他的源文件所用，不能被其他的源文件修改



#### 静态函数：

> 函数的声明和定义在默认情况下在整个程序中是外部的。
>
> 但有时候，你需要某个函数只在一个源文件中有效，不能被其他的源文件所用，这时在函数前面加static关键字。

```
staic void fun1();
```



### 作用域

C++作用域分为  局部作用域（块作用域）、函数作用域、函数原型作用域、文件作用域，类作用域



块作用域：

> 当变量出现在由一对大括号括起来的一段程序块内时，该标识符的作用域从声明定义的地方开始，
>
> 到块结束，也就是大括号结束。

**语句是一个程序单位**

例子：

```c++
if(...)
{
	int i ; //i的作用域从此开始
	i=5;
	cout<<i;
} 	//i的作用域结束了

...
```



#### 函数作用域

冒号： 作为唯一具有函数作用域的标识符。

冒号的声明使得该标识符在一个函数内的任何位置均可以被使用。



#### 函数原型作用域：

> 在函数声明的语句中的参数作用在该作用域内。
>
> 开始于函数原型的左括号，结束于函数原型的右括号

```
void Aera(double width,double length);

width,length作用于 函数原型作用域，在这条语句之外，width和length是未声明的。
```



#### 文件作用域

> 文件作用域是在所有函数定义之外说明的，其作用域从说明点开始，一直延申到源文件结束。

声明点很关键，只有声明过后，才能对变量进行操作。



静态全局变量、静态函数、内联函数的作用域全部都是文件作用域。









### 数组

注意点：

* 在编译时，数组的下标必须是确定的。不能用变量来描述数组中定义的个数，必须用常量

  ``int size =50;int array[size]; ``这是错的！！！

* 判断字符数组的终止条件：针对于用户输入

  ``for(int i=0;Array[i];i++)``   不用 ``Array[i]!='\0'``,因为不知道用户输入了多少个，不用判断到最后

* 数组初始化时的个数如果少于 定义的个数，后面自动补0

* 数组定义省略行号  int a[] [4] 根据数组的大小自动计算行列



数组分配堆内存：

1.malloc

```c
int Size=10;
int *ap = (int *)malloc(Size*sizeof(int))
ap就是一个数组了，ap[i]
```

2.new

```
int Size=10;
int *ap = new int[Size]; 
```



### 字符串

>  字符串的定义：a = “hello”   是双引号而不是单引号
>
> 字符串就两种表示方法：1.字符数组  2.指针



1.字符串的地址和数组类似，字符串的变量名代表了字符串的首地址

``	char *p="hello" ``  指针指的是整个字符串

``	*(p+1)``=“ello”

2.char a[] = “hello” ,用数组下标去控制



#### 字符串的常用函数：

memset：（起始地址，初值，大小）

```c
int a[10];
memset(a,0,10*sizeof(int));
```

strcpy:

字符数组的复制：注意：strcpy只能对以‘\0’结尾的字符数组进行操作

```
char a1[10];
char a2[10];
strcpy(a1,"hello")
strcpy(a1,a2);
```

memcpy：按规定的字节数进行复制

```
mencpy(a1,a2,10*sizeof(char))
```



strncpy(str , str1 ,sizeof()  )

也是复制,可以按照大小来复制



strcmp(str1,str2):

> str1 == str2 :返回0
>
> str1>str2 :返回正值
>
> str1<str2 :返回负值

### 指针：

程序中定义变量时，系统会为不同的数据类型的数据对象分配对应的内存空间来保存数据。分配的内存空间以字节为储存单位，**每个字节都有唯一的编号，这个编号叫做地址。**

**变量的地址是指该变量所占内存空间的第一个字节的地址，数据则是存放在内存空间的内容。**

#### 指针的定义：

专门存放地址的变量叫做指针变量。指针变量存储的是地址，通过它能查找内存单元。

语法格式：

数据类型  *  指针变量名				“ * ” ：这个符号是指针变量的定义符，指明这是指针变量

```c++
int *a , *b  =   int *a,b		//二者是等价的~！！！
```

**指针变量的值是内存单元的地址，因此储存空间的大小都是4个字节**

**一个指针变量只能储存同一数据类型的变量**

不允许出现同名的指针和变量，编译会出现二义性！！！

#### 指针变量的初始化：

数据类型  *  指针变量名 = 地址表达式；

指针变量名 = 地址表达式			**#指针变量储存的内存单元的地址，不赋值就是系统随机分配的地址**

**指针与数组：**

```c++
int a[80];
int *p = a;
//  &a[i]   a+i  	&p[i]	p+i
// 这四种表示方式全部是表示  第i个元素的地址

a[i]	p[i]	*(a+i)	*(p+i)  互相等价
    
```

指针数组：以指针变量为元素的数组。

指针数组的一个常用的用法就是 **用来储存若干行长度不等的字符串**

```c++
char * questions[] =
{
 "where are you from？",
    "what is your name?",
    "is he ?"
}
 cout<<question[1]<<endl  就可以直接输出了
```

数组指针：指向数组的指针。

```c++
int a[3][3] ,(*p)[3];	// (* 指针名)[下标表达式];
p=a
  //(*p)[3]表示p是指向一个长度为3的数组
  //*(*(p+i)+j)	表示a[i][j]
```

指针与字符串：

```C
char array[8] = "xiaoxin" ; //   "\0"要占一个位置！！！
char *p="xiaoxin";

1.array[2] = 'b';  //字符数组可以对数组元素进行修改，但是字符指针引用的字串常量不可以进行修改
2.p[2]='b';
```

常量指针：指针所指向的值是常量

```c++
const int a =78;

const int *b=&a;  //指针指向的常量不能被修改，但是指针值可以改

*b= 79 			// error!
b = &79 		//TRUE
```

指针常量：指针变量本身是常量，不可改变，指针指向的值可以改变

**指针常量必须初始化**

```
char * const p = "adsa"  
*p="dasd"

p="sada" 这是不行的!
```

常量指针的用法：

````c++
void mystrcpy(char* dest, const char *source)
{
    while(*dest++ = *source++);
}
int main()
{
    char a[20]="how are you"
    char b[20];
    mystrcpy(b,a);
}
````

变量字符串a传给mystrcpy后变成一个常量，不允许任何修改。

但是在主函数里a是可以被修改的，所以我们要传到函数里，不然a发生该改变，把内容赋值给b。



指针只能进行加法和减法的运算



#### 指针函数：

指向函数的指针叫指针函数。

**！！！指针函数的返回值不能是局部变量的地址**



```
int *getInt()
{
	int value=20;
	return &value;  //这里是错的！value是局部变量，函数结束即销毁，指针函数不能返回局部变量的地址
}

应该这样定义：static int value=20
```

#### 函数指针

```c++
int (*func)(char a , char b)

//例子
    //定义函数
    int func1(char x,char y)
    int *func2(char x,char y)
    int func3(int a)
    //定义函数指针
    int *fp1(char a,char b)
    int *fp2(int s)
    
    //通过函数指针来调用函数 , 调用的函数的参数类型要和函数指针一致
    fp1 = func1;
	fp2 = func3; //这里不能传参！！！
	
	fp1(a,b); //通过函数指针传参调用
	fp2(5) ;
```





#### 引用：

引用就是别名，是已存在的变量或者对象的别名。创建引用时必须进行初始化！

**数据类型 & 引用名 = 变量（对象）名 ；** 

1. 数据类型必须与被引用的类型一致

2. 变量名或者对象名已经被声明或定义

3. &叫做引用运算符，只有在创建引用的时候叫做引用运算符，其余时候叫做取地址符

4. 引用不是值，不会占用内存空间 ，因此引用只有声明没有定义

5. **一但引用被初始化，将与其引用的变量或者对象永远绑定，不得更改**

6. 引用不能赋值给常量  ，int &r=1 (不会报错，但不要这样)，用常引用 const int &r=1

7. 修改引用的值会连带更改原变量的值，但是地址还是一样的

8. ```
   int * &rp = p； //rp既是p的引用，也是指针类型  ：相当于 * （&rp）
   ```
   
9. 不允许引用数组

**引用的返回值：**

>  为什么要使用 函数的引用？
>
> 很简单，我们要返回变量的值到主函数去，而且是修改过的。和引用传参一个意义。
>
> 引用传参，可以直接修改变量的值，引用返回值，接收函数中有用的临时变量

**记住一点：引用作为函数返回值，返回值绝对不可以是局部变量**

```C
int t; //t声明为全局变量

int s1 =square(15);
int &s2 =square(20);

int &square(int i )
{
	t = i*i;
	return t
}
/*
s1可以接收 square返回的值 ，s2当然也可以，但是s2不能接收非引用函数的返回值
返回值必须是左值，不能是个表示式 
*/
return t+1  //这是错的！！，非常量的引用必须是左值
    这两个接收，前提都是t为全局变量，不是局部变量
    用s1接收square函数的值，没有问题
    用s2接收square函数的值，也可以，但是s2不能接收局部作用域的返回值
```

**常引用：**

const 数据类型 引用名 = 变量（对象）名

**一旦创建了常引用，就不能通过常引用来更改变量（对象）的值！！！**

#### 右值引用

> 必须绑定到右值的引用，我们通过&&而不是&，来获得右值引用
>
> 性质：只能绑定到一个将要销毁的对象
>
> 右值引用只能绑定临时对象
>
> * 所引用的对象必须销毁
> * 该对象没有其他用户
>
> 意味着，使用右值引用的代码可以自由的接管所引用对象的资源



通过move函数来返回右值

```
int &&rr3 = std::move(rr1);  rr1赋值然后销毁它
```



#### 动态指针：

new 和delete：

```C++
cout<<"please input a number of array:"<<endl;
int size;
cin>>size; //io流的输入方法

int * array = new int[size]  //分配size个整形空间（size*4个字节的堆内存）

int *ap = (int *)malloc(size *sizeof(int))
```

#### 智能指针

> 头文件 memory

| shared_ptr | 允许多个指针指向同一个对象 |
| ---------- | -------------------------- |
| unique_ptr | 独占所指对象               |
| weak_ptr   | 指向shared_ptr所管理的对象 |

智能指针定义时，要指定类型

##### shared_ptr

```C++
share_ptr<T> sp;
shared_ptr<string> p1;
shared_ptr<list<int>> p2;

p.get()
swap(p,q)
p.unique()
p.use_count()
```

* 最安全的分配和使用动态内存的方法时make_shared函数

```
shared_ptr<int> p3 = make_shared<int> 42
shared_ptr<string> p4 = make_shared<string>(10,'9')
auto p6 = make_shared<vector<string>>()

```

shared_ptr的析构函数会自动递减它所指向的对象的引用计数。如果引用计数变为0，shared_ptr的析构函数就会销毁对象，并释放占用的内存

动态内存的使用是多个对象共享一个数据，当一个对象销毁后，所有的对象就都销毁了。现在，我们希望某个对象销毁后，我们不能单方面地销毁底层数据，Blob类就可以满足

```
Blob<string> b1;
{
	Blob<string> b2={"a","an","the"};
	b1=b2;
}
//花括号结束，b2销毁，但b2中的元素还在
```

与普通指针对比：

```C++
//普通指针：
auto v = new vector<int>();
int i ;
while(cin>>i)
	v->push_back(i);
for(auto &i : *v)
	cout<<i<<endl;
delete v;
//智能指针：
auto v = make_shared< vector<int>>();
int i ;
while(cin>>i)
	v->push_back(i);
for(auto &i : *v)
	cout<<i<<endl;
```

智能指针和new混用：

```C++
shared_ptr<int> p2(new int(42))

shared_ptr<int> clone(int p)
{
	return shared_ptr<int>(new int(p));
}
```

##### unique_ptr

> 它只能指向一个对象，不支持普通的拷贝或者赋值操作
>
> 初始化必须直接用new指定对象

```
unique_ptr<int> u(new string("hahaha"));
```

```C++
u.reset() //释放u指向的对象
u.reset(q) //重新指向
u.release() //释放对u的控制权
    
unique_ptr<string> p2(p1.release()) //p1的控制器移交给p2
p2.reset(p1.release())  //一样效果
```

##### 智能指针数组

```c++
unique_ptr<int[]> up(new int[10])
up.release()  //自动调用delete数组
for(size_type i=0;i!=10;i++)
	up[i]=i;

shared_ptr不支持自动管理数组，必须自定义删除器
shared_ptr<int> sp(new int[10] ,[] (int *p){delete []p;});

for(size_type i=0;i!=10;i++)
	*(sp.get()+i)=i;	//用get先获取内置指针再赋值
```

##### weak_ptr

> 一种不控制所指向的对象生存期的智能指针，它指向一个share_ptr管理的对象。将一个weak_ptr绑定到一个share_ptr，不会改变它的引用计数
>
> * 必须要用shared_ptr来初始化
> * 必须调用lock访问对象

```
weak_ptr<T> w
weak_ptr<T> w(sp) sp为shared_ptr
w=sp

w.reset()
w.use_count()
w.expired() 若use_count为0返回true，若use_count不为0返回false
w.lock()	expired为true,返回空shared_ptr ,false返回shard_ptr指向的对象

```

例子：

```
auto p = make_shared<int>(42);
weak_ptr<int> w(sp);
```

有什么用呢？

> 检查shared_ptr是否存在

```
auto ret = wptr.lock();
if(!ret)
	throw runtime_error("unbound");
if(i>=ret->size())
	throw out_of_range(msg);
return ret;
```



##### allocator类

> 它帮助我们将内存分配和对象构造分离开来，它会根据给定的对象类型来确定适合的内存大小和对齐位置。它也是一个模板，所以要指定类型



```
allocator<string> alloc;
auto const p = alloc.allocate(n); //分配n个未初始化的string
```

| 算法                | -                                    |
| ------------------- | ------------------------------------ |
| a.allocate(n)       | 分配并保存n个类型为T的对象           |
| a.deallocate(p,n)   | 从p指针开始释放n个对象               |
| a.construct(p,args) | p为指针，args传递给类型为T的构造函数 |
| a.destroy(p)        | 析构p指向的对象                      |

```
auto q=p //q指向p最后构造的位置
alloc.construct(q++);	// 空
alloc.construct(q++,10,'c'); 
alloc.construct(q++,"hi");

while(q!=p)
	alloc.delete(--q) ; //删除新构造的字符串
```

未初始化内存中创建对象

伴随算法：

| uninitialized_copy(b,e,b2)   | b,e迭代器范围，b2是内存 |
| ---------------------------- | ----------------------- |
| uninitialized_copy_n(b,n,b2) | 从b开始，拷贝n个        |
| uninitialized_fill(b,e,t)    | b,e迭代器范围，t为值    |
| uninitialized_fill_n(b,n,t)  |                         |

```
// 分配vi两倍的动态内存
auto p =alloc.allocate(vi.size()*2) 
//拷贝vi中的内存，从p开始
auto q=uninitialized_copy(vi.begin(),vi.end(),p);
//将剩余元素初始化为42
uninitialized_fill_n(q,vi.size(),42);

返回的是一个位置的迭代器
```





### 结构体：

结构体声明：

```c++
struct student （结构体名）
{
    数据类型 成员;
    ...
}（变量名）;   

声明结构体变量：
struct 结构体名
{
  ...  
} 结构体变量名1 ，结构体变量名2 ， *ptr = &结构体变量名1 （设置指针指向变量1）；
    
结构体变量赋值：
    student zhangsan = {2002,18,'zhangsan'，95，'中山大道108号'}
 
struct student
{
    ...
}zhangsan ， zhangsan = {2002,18,'zhangsan'，}

void teststruct( student &s)
{
    ...
}

```

结构体指针的使用事项：

> 想用结构体指针去创建对象，必须用typedef重新命名；

```
typedef struct student
{
    int num;
}*stu;

main()
{
    stu s;
    s = new student;
    s->num=10;
    cout<<s->num<<endl;
    delete(s);
    system("pause");
}

//结构体变量指针
struct student
{
    int num;
}s1,*stu=&s1;

main()
{
	s1.num=10;
    cout<<s1.num<<endl;
    stu->num=5;
    cout<<stu->num<<endl;
}
 	
```



##### 结构体数组：

```c
student s[10]={
    {...},
    {...},
    ...
}

student *pA = {&s[0],&s[1]...,&s[9]}  //还可以通过指针来调用

void getKey(student &s[])
{
    for(int i=0;i<10;i++)
    {
        cin>>name>>...
        s[i]={name,...}
    }
}

student s[10];
getKey(s);

student getKey()
{
    student tmp;
    tmp.name = ..
    //成员赋值
    return tmp;
}

student s1 = getKey() //用对象来接收就好了

```



##### 结构体嵌套：

```c++
struct Edu
{
    char major[20];
    char degree[20];
    double gpa;
};

struct Stu
{
    Edu school;
    char id[20];
    int graduate;
};
Stu ss;

//但是访问的时候，要注意要使用多个成员操作符

ss.school.major

```

##### 结构体传参：

```
void Print(student s)
void Print(student &s)  //这两种传参效果没有区别，只是传引用的效率会更高


Print(student[i])
```



##### 访问结构体成员：

结构体变量与其他数据类型的变量类似，都可以传给函数进行处理，也可以作为函数的处理结果返回。

作为参数传给函数处理时有内容(值)传递、地址传递和引用传递三种方式。

引用和地址传递，改变的是地址，会影响main（）函数中的值，而内容传递在函数结束后内存被释放掉了，不影响main（）函数的值

1. 使用指向运算符 -> ：

   结构体指针变量名 -> 结构体变量成员名

2. 使用成员运算符 “.” ：

   （* 结构体指针变量名）.结构体变量成员名

   (*ptr).num 或者 ptr -> num
   
   

### 类与对象

> 类：是一个抽象的概念，是一类具有特征的物体的集合。类是没有分配内存空间的
>
> 对象：是实实在在存在的物体，创建时自动分配内存空间的

类的定义：

```c++
class 类名
{
    pubilc:
    <公有数据成员和公有成员函数>;
    protected:
    <保护数据成员和保护成员函数>;
    private:
    <私有数据成员和私有成员函数>;
};
```

数据成员如果没有放在 这三种关键字中，则默认为private类型



成员函数：

```c++
class A
{
    public：
        void fun1();
    private:
    	int fun2();
    protected:
    	bool fun3(int i);
    
};
成员函数的全名 
    类名：：函数名
    A：：fun（int）
    
    ’ ：：‘ 这个符号是作用域区分符，跟在类后面时表示 函数或者数据属于这个类
```

注意区别一下类内的函数和类外的函数，虽然同名但是不一样

在类外定义类的成员函数：

```C
/* 
    返回值类型  类名 ：：成员函数名（形参列表）
    {
        函数体；
    }
*/
void Tdate::SetDate(int y,int m,int d)
{
    year = y;
    month = m;
    day = d;
}
...
    
```

使用头文件将类和函数分开：

```c
//Tdate.h
class Tdate
{
    public:
    ...
    private:
    ...
};

//Tdate.cpp
# include"Tdate.h"
void Tdate::SetDate(int y, int m, int d)
{
    year =y ;month =m; day = d;
}
```

### 构造函数：

构造函数就是专门用于初始化对象的成员函数，就是用来对象的数据成员赋初值 **类似与  def _ _init_ _  _ ()函数**

**创建对象时，构造函数自动调用**

同理：类外定义构造函数与成员函数类外定义一样

​		**构造函数名与类名完全相同，没有返回值，也没有任何的返回类型，一个类中可以有多个构造函数（重载）**

类初始化时两种方式给数据成员赋值：

```C
//使用函数体赋值
class X
{
    int a,b ;
    public:
    X(int i , int j)
    {
        a = i;
        b = j;
    }
};

//使用冒号+括号赋值
class X
{
    int a,b ;
    public:
    X(int i,int j):a(i),b(j)	//这种方式必须用括号，不能用=赋值
    {}
};
```

#### 输入流中读入

```
Sale_data(istream& is=cin ) {read(is,*this);};
```

#### 委托构造函数

> 意思就是，这个构造函数委托第一个定义的默认构造函数区初始化值

```
Sales_data(const std::string &s, unsigned n, double p) :bookNo(s), units_sold(n), revenue(n*p) {cout<<"默认构造函数"<<endl;}
	Sales_data() : Sales_data("", 0, 0.0f) {cout<<"构造函数1"<<endl;}
	Sales_data(const std::string &s) : Sales_data(s, 0, 0.0f) {cout<<"构造函数2"<<endl;}
	Sales_data(std::istream &is):Sales_data()
    {
        cout<<"构造函数3"<<endl;
        read(is,*this);
    }
```

#### 隐式类类型转换

> 在C++中，explicit关键字用来修饰类的构造函数，被修饰的构造函数的类，不能发生相应的隐式类型转换，只能以显示的方式进行类型转换。
>
> explicit使用注意事项:
>
> * explicit 关键字只能用于类内部的构造函数声明上。
>
>   explicit 关键字作用于单个参数的构造函数。
>
> * 在C++中，explicit关键字用来修饰类的构造函数，被修饰的构造函数的类，不能发生相应的隐式类型转换

```
string null_book = "9-9999-999-9"

item.combine(string("9-9999-999-9"))
item.combine(Scal_data("9-9999-999-9"))
```

#### consrexpr字面值常量类

> constexpr函数的参数和返回值都是字面值类型。
>
> * 数据成员都必须是字面值类型
> * 类必须至少有一个constexpr构造函数
> * 该类必须使用constexpr的默认析构函数
>
> 一般来说，constexpr的函数体都是空的

```
class Debug
{
    private:
    bool hw;
    bool io;
    bool other;

    public:
    constexpr Debug(bool b=true):hw(b),io(b),other(b){}
    constexpr Debug(bool b,bool i,bool o):hw(h),io(i),other(o){}
    constexpr bool any(){return hw||io||other;}
    void set_io(bool b) {io=b;}
    void set_hw(bool b) {hw=b;}
    void set_other(bool b) {other=b;}
};

constexpr Debug io_sub (false,true,false); 	调式io错误
if(io_sub.any())	{cerr<<"error message"<<endl;}
```

* constexpr构造函数必须初始化所有的数据成员
* conexpr函数必须有返回语句



#### 冒号语法

> 常量，引用变量和其他类的对象，必须放在构造函数的函数体执行之前完成初始化

```
class A
{
	public:
	A(参数):i(10),reI(i)
	{
		...
	}
	private:
	const int i;
	int &reI ;
}
```

#### 冒号语法初始化基类对象 

构造一个对象时，同时调用其他类的构造函数

> 必须使用冒号语法，因为如果放在自己的构造函数里去构造其他的对象，很不幸，自己的构造函数结束，其他的对象就会析构。

```c++
class StudentID
{
	public:
	StudentID(int id=0)
	{
		value =id ;
	}
	private:
	int value;
}

class Student
{
	public:
	Student(char * pname ="no name"):stu_id(10)
	或者
	Student(char *pname="no name",int ssID):stu_id(ssID) 传一个ssID参数上来，再赋给id
	{
		strncpy(pname , name, sizeof(name));
		name[sizeof(name)-1]='\0' 防止名字过长而崩溃
	}
	private:
	char name[20]; 
	studentID stu_id;  //类定义中声明对象,然后就可以在Studnet类中通过对象调用StudentID的成员函数
}
```

注意：在类定义中，类定义时不会调用构造函数的，不允许初始化，因为类定义不分配空间！



> 对冒号语法和基类对象初始化的深入理解
>
> 类成员的初始化，是一个机制。
>
> * 构造已分配空间的的对象成员，而不是重新创建一个对象。 也就是重新对已经构造的成员重新赋值

```c++
class Car 
{
    public:
        Car(){
            cout<<"构造Car类ing"<<endl;
        };        
        Car(int j){sizec=j;};
        ~Car(){};
       
        void display()
        {
            cout<<"这里是Car类的size="<<sizec<<endl;
        };
        
    protected:
        int sizec;
    
};

class Boat 
{
    public:
        Boat(int j):car(100)
        {
        	size=j;
        };
        ~Boat(){};
        
        void display(Boat &bt,Car&c);
        friend void display2(Boat &bt,Car&c);  //这两个函数是普通函数和友元函数的区别
        
        
    protected:
        int size;
        Car car;  //这里声明的car也是Boat类的数据成员，在成员函数里面可以直接使用
        
};

void Boat :: display(Boat &bt,Car& c)
{
    c.display(); 	//这里输出的是1
    car.display();  //输出的是100
}
```

> 这种初始化方法是定义里面直接声明一个基类的对象，再使用冒号语法初始化
>
> 作用域的范围也是Boat类的作用域
>
> 访问语法：
>
> 这样做有个好处，你可以在Boat类的内部直接使用这个对象去调用基类的成员函数，但是不能访问数据成员哦！！

```
class Boat : protected Car
{
    public:
        Boat(int j):Car(100)
        {
        	size=j;
        };
        ~Boat(){};
        void display(Boat &bt,Car&c);
      
    protected:
        int size;
        
        
};

void Boat :: display(Boat &bt,Car& c)
{
    c.display();
    cout<<bt.sizec<<endl;
     cout<<Boat::sizec<<endl;
```

> 这种对Car的初始化的方法：
>
> 意义：在Boat的类作用域里面，sizec的值已经被重新附上了100，仅仅是在Boat作用域
>
> 在Car的作用域内，sizec的值还是我给的初值1
>
> 访问数据的语法：
>
> 1.如果不是继承的关系，``类名：：数据成员名；``
>
> 2.继承关系，子类对象直接访问

#### 冒号语法调用基类构造函数初始化

```
class Boat :public Car,public Plane,public train
{
    public:
        Boat(int j,int h):Plane(3),train(5){}
```

这种形式的初始化就是调用了基类的构造函数，你给基类的数据成员赋了初值

* 但是如果你调用了基类的成员函数，且这个基类的成员函数里面有输出基类的数据成员，这时候这个基类的数据成员的值，要么是0要么是随机值

```
 //trian类里面的run成员函数 ，hour的值是0
 void run()
        {
            cout<<"the run time is "<<hour<<endl;
        }
```

如何解决这个问题呢？

这时候就不能使用初始化列表的形式了！！！改成函数体内赋值的形式赋值

```
Boat(int j,int h)
        {
            size=j;
            k =h;
            time = j;
            hour = h;
        }
```

* 这样赋值以后，基类的数据成员就有值了，但是前提条件是公有继承和保护继承才可以，私有继承不可以访问



那么问题又来了，如果我们使用初始化成员的形式初始化基类对象能不能获得数据成员的值呢？

​			可以！

```c++
//基类Circular中的getArea成员函数
double getAera()
        {
            return area;
        }

//子类Column的构造函数
Column(double a, double b):Circular(a)
        {
            h = b;
            cubage = getAera()*h;
            cout<<"得到圆的面积："<<getAera()<<endl;
        }
```

一旦在初始化列表里面初始化了基类的值，那么基类里面数据成员就被赋了初值，再调用getAera（）就可以获得

* 初始化之后，其他的成员函数也可以调用并获得值了

```
//子类Column的Aera函数
int Area()
        {
            return getAera();
        }
```



### 动态对象调用构造函数的顺序

 当用基类指针指向子类从堆中分配的对象时，如下形式 A*p = new B 

**先调用父类的构造函数，在调用自己的构造函数**

![1640743291820](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1640743291820.png)

```
当调用delete p；p=NULL；销毁对象时，是调用父类A的析构函数还是调用子类B的构造函数呢？
如下图所示：答案是会调用父类的构造函数。因为没有使用虚函数，不具有多态性，默认调用父类的成员

这样问题就来了，子类不是从父类继承来的那些独有的成员变量的内存将得不到释放，将会造成内存泄露，这种情况应该如何避免内存泄露呢？

这就引入了一个新的知识点：虚析构函数。

虚析构函数就是在父类的析构函数前加上virtual关键字，这种特性将会被继承下去，也即子类的析构函数也为虚析构函数，在下面的例子中做如下改变：

virtual ~Person(){}//将父类的析构函数变为虚析构函数

virtual ~Soldier(){}//子类的析构函数继承了这种特性，也变成了虚析构函数，即便子类不写virtual，子类构造函数也是虚析构函数

虚析构函数的使用场合：当存在继承关系，用父类的指针指向从堆中分配的子类的对象时，然后又想用父类的指针去释放掉内存，就会用到虚析构函数，用了虚析构函数后，再调用delete Person时，就会先调用子类的析构函数，再调用父类的构造函数了

```



#### 构造对象的顺序：

1. 局部和静态对象，以声明的顺序构造

2. **静态对象只被构造一次**：

   在文件作用域中，静态对象和全局对象一样，在main函数开始之前全部构造完毕

   在块作用域中（函数作用域中），在首次进入该静态函数时进行构造

3. **成员以其在类定义中的顺序构造，而不是按构造函数说明中冒号赋值的顺序进行构造**

#### 重载构造函数：

当被创建对象初始值个数或类型不同时，需要调用参数列表不同的构造函数来进行初始化

```C++
class T
{
    public:
    Tdate();
    Tdate(int d)
    {
        day =d
    }
    Tdate(int m,int d)
    {
       month = m;
       day =y ;
    }
    Tdate(int y,int m, intd)
    {
        ...
    }
};

int main()
{
    Tdate aday;
    Tdate bday(10);
    Tdate cday(2,12);  //根据参数的不同，反映对象的不同，然后再去调用对应的构造函数
}
```

**也就是说，对象的类型不同需要不同的构造函数进行初始化**



***默认构造函数**是指参数列表为空或所有参数都有默认值的构造函数，参数列表和函数体都为空！！*！

### 拷贝构造函数：

用对象取创造另一个对象。

**拷贝构造函数的形参必须是对象的引用**

当对象作为函数的形参的时候，会自动调用拷贝构造函数。



**默认拷贝构造函数：啥也没有**



浅拷贝：这样确实能创造出来对象，但是这两个对象公用一个资源，修改其中的一个另一个也会改变。

深拷贝：这种情况要单独写一个重载构造函数，给创造的对象分配资源

```c++
//浅复制
Person :: Person(char *pN)
{
	pName = new char[strlen(pN+1)];
	if(pName!=0) 
		strcpy(pName,pN)
}

int main()
{
	Person p("randy");
	Person p2=p;  			
}

共用一个资源，改动其中一个对象的数据成员，另外一个对象的数据成员也会跟着变化

//深复制
Person :: Person(char *pN)
{
	pName = new char[strlen(pN+1)];
	if(pName!=0) 
		strcpy(pName,pN)
}

Person :: Person(Person &p)
{
	pName = new char[strlen(p.pName)+1];   拷贝构造函数从堆里拿内存,就是深复制
	if(pName!=0)
		strcpy(pName,p.pName)
}
```



就是新建立的对象时已经存在的对象复制过来的

Tdate t1;

Tdate t2(t1)   **-->这是就是复制构造函数**，就这一行代码

有对应的成员函数： 复制的对象执行这个成员函数

```c++
Tdate: : Tdate(Tdate &t)
{
    ...
}
```

调用完复制构造函数

> 三种情况调用系统调用复制构造函数：
>
> 1. 复制新对象
> 2. 类的对象作为函数的参数，实参向形参传值的时候，系统调用复制构造函数
> 3. 函数的返回值是类的对象，函数执行结束时，系统调用复制构造函数来初始化来保存返回值的无名对象

* 解释一下2中的对象作为函数形参：

> 进入函数时，调用对象的构造函数
>
> 然后执行函数体
>
> 退出函数时，对象析构（这一步和临时对象差不多）

* 解释一下3中的函数返回的对象：

​	当一个函数返回一个对象时，要创建一个临时对象以存放返回的对象。

* 在含有函数返回值的表达式结束之后，这个临时对象自动析构

```
Student fn()  //这是一个返回值为Student类型的函数
{
	Student ms("randy")
	return ms   //返回一个Student的对象，临时对象
}

int main()
{
	Student s;
	s = fn();	 s用来接收ms这个临时对象，然而这个ms对象在完成这条语句之后就会被析构。
}
```

​	![1640744297748](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1640744297748.png)

这三种传参方式：

> test1()：会调用拷贝构造函数，结束还会调用析构函数 
>
> test2()：不会调用拷贝构造函数，不会调用析构函数
>
> test3()：不会调用拷贝构造函数，不会调用析构函数

#### =default

使用默认构造函数

#### =delete

>  删除函数是这样的一种函数，我们虽然声明了它们，但不能以任何的形式使用它们，就是被删除了
>
> 注意：析构函数不能被删除，因为析构函数不能被使用，就不能销毁对象了

#### 堆对象：

对对象的作用域是整个程序的周期，不会像局部对象那样，函数结束就释放空间了，直到遇到delete才会析构。

在堆对象上分配数组，只能调用默认的构造函数，不能调用其他任何构造函数

```
Tdate * p;
p = new Tdate;
---
Tdate *p = (Tdate *)malloc(sizeof Tdate)
```

什么时候使用堆对象？

1.直到运行的时候才知道需要多少对象空间

2.不知道对象的生存期到底有多长

3.知道运行时才知道一个对象需要多少内存空间

#### 拷贝控制示例

> 两个类，Message和Folder，分别表示电子邮件消息和消息目录。每一个Message对象可以出现多个Folder对象中，但是Message的内容只有一个副本。这样如果一条Message的内容被改变，则我们从它所在的任何Folder文件夹来浏览此Message时，都会看到改变后的内容。

在C++中，explicit关键字用来修饰类的构造函数，被修饰的构造函数的类，不能发生相应的隐式类型转换，只能以显示的方式进行类型转换。

explicit使用注意事项:

* explicit 关键字只能用于类内部的构造函数声明上。

   explicit 关键字作用于单个参数的构造函数。

* 在C++中，explicit关键字用来修饰类的构造函数，被修饰的构造函数的类，不能发生相应的隐式类型转换

例子：

```C++
#include<iostream>
#include<utility>
#include<memory>
#include<string>
using namespace std;

class StrVec
{
    public:
        StrVec():elements(nullptr),first_free(nullptr),cap(nullptr){}
        StrVec(const StrVec&);
        StrVec &operator = (const StrVec&);
        ~StrVec();
        void push_back(const string&);
        size_t size() const {return first_free - elements;}
        size_t capacity() const {return cap - elements;}
        string *begin() const {return elements;}
        string *end() const {return first_free;}
    private:
        allocator<string> alloc; //分配元素
        void chk_n_alloc()
            {if(size()==capacity()) reallocate();}
        pair<string*,string*> alloc_n_copy(const string*,const string*);
        void free();    //销毁元素并释放内存
        void reallocate();  //获得更多的内存并拷贝已有元素
        string *elements;   //指向数组的第一个元素的指针
        string *first_free; //指向数组的第一个空闲位置的指针
        string *cap;        // 指向数组的尾指针
};

void StrVec::push_back(const string& s)
{
    chk_n_alloc();  //确保有空间容纳新元素
    alloc.construct(first_free++,s);    //使用construct构造新元素分配空间
}

pair<string*,string*>
StrVec::alloc_n_copy(const string *b,const string *e)
{
    //分配空间并保存给定元素的范围
    auto data = alloc.allocate(e-b);
    //初始化并返回一个pair
    return {data,uninitialized_copy(b,e,data)};
}

void StrVec::free()
{
    //不能传递给deallocate一个空指针。如果elements为0，函数什么也不做
    if(elements)
    {
        for(auto p=first_free ;p!=elements;)    //删除元素
            alloc.destroy(--p);
        alloc.deallocate(elements,cap-elements);    //释放内存空间
    }

    // for_each(elements, first_free, [this](std::string &rhs){ alloc.destroy(&rhs); });
}

StrVec::StrVec(const StrVec &s)
{
    //调用alloc_n_copy分配空间以容纳与s一样多的元素,返回值是初始化的pair
    auto newdata = alloc_n_copy(s.begin(),s.end());
    elements = newdata.first;
    first_free = cap =newdata.second;
}

StrVec :: ~StrVec(){free();}

StrVec &StrVec::operator=(const StrVec &rhs)
{
    auto data = alloc_n_copy(rhs.begin(),rhs.end());
    free();
    elements = data.first;
    first_free = cap = data.second;
    return *this;
}

void StrVec::reallocate()
{
    //我们将分配当前大小两倍的内存空间
    auto newcapacity = size() ? 2*size():1 ;
    //分配新内存
    auto newdata = alloc.allocate(newcapacity);

    auto dest = newdata ;   //指向新数组的下一个空闲位置
    auto elem = elements;   //指向旧数组中下一个元素
    for(size_t i=0 ; i!= size() ; ++i)
        alloc.construct(dest++,std::move(*elem++)); //遍历新数组，依次添加旧数组中的元素
    free();
    elements = newdata;
    first_free = dest;
    cap = elements+newcapacity;
}





```

#### 移动构造函数

对象移动：

> 移动构造函数通常是将资源从给定对象移动，而不是拷贝到正在创建的对象，假设每个string都有一个指向char数组的指针，可以假定string的移动构造函数进行了指针的拷贝，而不是为字符分配内存空间然后拷贝字符。
>
> 拷贝，资源就会有两份，一份自己，一份副本
>
> 移动，只有一份，原来的被销毁
>
> ```
> hp = hp2 ;	拷贝
> hp = std::move(hp2) ; 移动
> ```
>
> 移动复制运算符：
>
> ```
> StrVec &StrVec::operator=(StrVec &&rhs) noexcept
> {
> 	if(this != &rhs)
> 	{
> 		free();
> 		elements = rhs.elements;
> 		s.eleemnt = s.cap = s.first_free = nullptr;
> 	}
> }
> ```
>
> 

noexcept:

```
StrVec::StrVec (StrVec &&s) noexcept: //noexcept就是不抛出任何异常
	elements(s.element),first_free(s.first_free),cap(s.cap)
	{
		s.eleemnt = s.cap = s.first_free = nullptr;
	}
```



移动迭代器：

> 之前reallocate成员函数使用for循环调用construct 从旧内存拷贝到新内存
>
> 还可以使用uninitialiezd_copy来构造新内存，指定首尾迭代器就ok，返回指向尾部的迭代器
>
> make_move_iterator:移动迭代器

```
auto newcapacity = size()? 2*size() :1 
auto first = alloc.allocate(newcapacity);

auto last = uninitialized_copy(make_move_iterator(begin()),make_move_iterator(end()),first);
free();
elements = first;
first_free = last;
cap = elements+newcapacity;
```



### 析构函数：

析构函数没有参数和返回值，一个类中只允许定义一个析构函数，不能重载！

对象按照生存期分为三类：

1. 局部对象：定义在函数体内的一般对象g。局部对象在栈中建立，当程序跳出该对象的函数体时，

   该对象被撤销，其内存空间被释放。

2. 静态对象：**定义前面有static 的对象，如 static Tdate a**；

   程序第一次执行定义静态对象时，系统自动调用构造函数创建对象有且执行一次，当整个程序结束后，对象被撤销，内存空间被释放。

3. 全局对象：定义在函数和程序之外的对象，在main（）函数之前执行创建对象，程序结束撤销。

**当对象被撤销时，对象的数据成员所占的内存空间需要被释放，这些工作由专门的成员函数----析构函数来完成**

对象建立时自动调用构造函数，对象被撤销时自动调用析构函数

```C
//体内定义
~Tdate()
{
	cout<<"destructing the object "<<endl;
}

//体外定义
Tdate::~Tdate()
{
    ...
}
```



析构函数与构造函数进行对比：

1. 析构函数的名字时在构造函数名字前面加一个“ ~ ”符号，析构函数没有数据类型，没有返回值，也没有参数
2. 析构函数不能重载，一个类中只能定义一个析构函数
3. 析构函数可以被系统调用，也可以被程序调用



### 对象：

创建对象：类名 对象名（参数列表）

#### 对象的初始化：

```C++
class A
{
    ...
};

A a;	//局部对象
A *a = new A ; //堆对象
```



#### 对象指针：

对象占用一段连续的内存空间，也可以使用一个指针来指向存放对象的内存地址，这个指针就是对象指针

```c++
/*
	类名 *对象指针名；
*/
Tdate *tpoint;     			//对象指针可以拿来访问该对象的成员
//	tpoint -> 数据成员名;
// 	tpoint ->成员函数名（参数列表）;
Tdate a(2021,8,6);
Tdate * tpoint = &a;
tpoint ->print();   //访问成员函数，相当于a.print()
```

**对象指针除了用来调用成员函数，还可以作为成员函数的形参。**

**在主函数调用成员函数，将对象的地址传给形参，这样可以通过形参对象指针的改变影响对象实参，达到信息双向传递的目的**

```c++
class Salary
{
    public:
    Salary(int ,int, int);
}
void addBonus(Salary *sp)
{
    ...
};

main()
{
    Salary s1(2021,8,6)
    addbonus(&s1)		//地址传参
}

```

#### 对象数组：

```c++
main()
{
    Tdate a[3]=
    {
        Tdate(2021,17),
        Tdate(2021,18),
        Tdate(2021,19);
    }
    for(int i = 0 ; i<3 ; i++)
    {
        a[i].print();		//对象数组调用成员函数
    }
    //堆数组
    Tdate * arr = new Tdare[2]= {
        Tdate(1,1),
    	Tdate(2,2),
    	}
}
```

**数组内每个对象不一样要重载构造函数**

其实这个重载主要体现在参数的个数不同上，一个参数的对象，两个参数的对象，三个参数的对象

#### 动态对象：

C++程序的内存区域分为4个区域：全局数据区域、代码区、栈区和堆区

堆区：自由存储区，自由建立和撤销,放动态对象等动态

全局数据区：全局变量、静态数据和常量放在这

栈区：局部变量、函数参数、返回数据和返回地址等存放地

代码区：函数代码编译后存放地

**语法格式： 对象指针 = new 类名 （初值表）**

```c++
Tdate *sp;
sp = new Tdate(2021);
sp.print();

delete sp;   // 删除动态指针指向的对象，系统先调用析构函数然后释放堆区的空间
```

动态对象也不能随便的消失，语法格式：delete 对象指针

```c++
main()
{
    Tdate *sp;
    sp = new Tdate[3];
    sp[0].setDate(2021,8,6)
    sp[1].setDate(2021,8,5)
    sp[2].setDate(2021,8,4)
        for (int i=0 ,i<3,i++)
        {
            sp[i].print();
        }
    delete sp;
}
```

#### this 指针：

```c++
Tdate a;
void Tdate::print()
{
	...
}
a.print()
//在print函数中需要引用调用它的对象，C++就会使用隐含的this指针，this指针时系统预定义的特殊指针，专门指向当前对象，表示当前对象的地址
    
cout<<this->month<<this->day<<this->day<<endl
    //相当于当前对象的指针
```

#### 静态成员：

> 为什么要用静态数据成员？
>
> 我们希望类的对象在类的范围内进行共享数据，而使用全局变量会破坏类的封装性

类描述的是所有对象的共同属性和行为，**共同属性**表示为数据成员，**共同行为**表示的是成员函数。

**静态成员就是被类中所有成员所共享，而不被某个对象独享**

静态数据成员：被类中所有对象所共享的数据成员。

静态数据成员：只能在类内定义，类外初始化 

非静态数据成员：只能在类内定义类内初始化

静态成员函数：被类中所有对象所共享的成员函数。

**静态成员函数里面，不能访问任何数据成员，因为静态成员函数为所有对象共享，不知道数据给谁。**





运用场合：

1.用来保存流动变化的对象个数

2.作为一个标志，指示一个特定的动作是否发生

```c++
//类内定义
static 数据类型  静态数据成员名；
    
//类外给初值
数据类型	类名	：：静态数据成员名==初始值；
    
//静态成员函数不能直接去访问非静态数据成员！！！
static void display(Student &s)
{
    cout<<s.name<<endl		//访问非静态成员必须将对象作为参数传上来，然后去访问
}
```

##### 注意点

* 静态数据成员确实是在程序一开始运行就得存在
* 放在类内定义，放在public下面就是公共静态成员，protected下面就是保护静态成员
* 静态数据成员 不能在任何函数内分配空间和初始化。
* 静态成员函数与任何当前的对象都没有联系，因为静态成员函数没有this指针



#### 成员的调用

1.对象调用：``对象.成员``

2.类名调用：``类名::成员``



#### 友元函数：

类的成员函数频繁的调用和参数传递，增加了时间开销降低了程序的运行效率。

一种方法：把所有的函数都放在public中，这将导致所有的非成员函数都随意访问类的成员，从而破坏了类的封装性和隐藏性。

**另一种方法：寻求能使类外的函数的普通函数直接访问类中的私有成员和保护成员的机制**

**友元只是类的朋友而表示类的成员，它可以访问私有成员和保护成员在内的所有成员**

> 具体是怎么访问的呢？
>
> 首先普通函数可以把对象作为形参传进去，通过对象来访问成员函数，但有且只能访问成员函数
>
> 要想访问数据成员，只能通过调用对象的成员函数内部访问；要想修改数据成员只能通过成员函数的返回值来修改
>
> * 那么声明了友元函数之后呢？
>
> 函数传上来的对象形参，可以直接访问所有的数据成员，对象.数据成员

```c++
friend 函数返回类型  函数名（形参列表）
    //友元可以在类内任意位置进行声明，在public和private区声明的意义完全相同。
class Tdate
{	
    friend class Ttime  //友类，表示 Tdate中可以访问Ttime中的私有成员
    public:
    	...
    private:
    	...
    
    friend void display(Tdate t);
	friend void Ttime :: add(Tdate &x , Tdate &y);  在Ttime的add函数中也能访问Tdate的成员
}
   //友元的意义就是在当前类中调用其他类的其他函数或者普通函数

void display(Tdate &t)
{
    ...
}

void Ttime :: add(Tdate &x , Tdate &y)
{
    ...
}
```

##### 注意点：

* 友元声明必须在类的内部
* 友元不具有传递性和对称性，没有双向声明就不能双向访问

##### 友类

在一个类中声明 另一个类是友类，这样每一个成员函数，传上对象形参后都可以直接访问数据成员

```
class Teachar
{
	friend class student ;
	public:
		void visit(student &t)
		{
			t.studentNumber;
			t.studentName;	//直接访问student数据成员，不用再一个一个声明为友元函数了
		}
}
	
```





#### 常对象与常成员函数：

* 常对象一旦被建立，它的数据成员在对象的整个生存内不能被改变，不能被修改

* 常对象只能访问，不能修改数据成员。

* 常成员函数只能访问常数据成员

* 常对象只能调用常成员函数

```c++
//常对象	
类名	const 	对象名 ;

Tdate const t(2021,8,6);
//	C++规定只有常对象才能访问常成员函数

返回类型 成员函数名（形参列表）	const
Tdate print() const;


class Tdate
{
    ...
    public:
    void print();			//这里是重载函数		
    void print() const;     //声明常函数
    ...
}
void Tdate::print()
{
    ...
}

void Tdate::print() const	//初始化常函数
{
    ...
}

const Tdate a(2021,,8,6)		//定义常对象，只有常对象才能调用常成员函数
a.print()						//调用


```



**！！！！如果没有重载函数，一般对象也可以调用常函数**

```c++
/*常数据成员
    数据类型	const	数据成员名；
*/
class	A
{
    public：
        A(int i)
        const int a;
    	static const int b;
};
const int A::b = 10		//静态常数据成员赋值方法 ， 一旦赋值就不能改变了哦！

    
A::A(int i):a(i),b(a)	//常数据成员赋值，只能通过初始化列表
{}

```



### 继承与派生：

```c++
class 派生类（子类）名：继承方式1 基类名 ，继承方式2 基类名，。。。
{
    pubilc:
    
    protected:
    
    private:
    
}
```

注意点：

 * 如果不注明继承方式系统继承方式默认为私有继承
 * 有两类成员函数，父类的和子类的，如果子类的成员函数与父类同名将覆盖父类的

**公有继承**：

​	子类的对象只能访问基类的公有成员

​	子类的成员函数可以访问父类的公有成员和保护成员。

**私有继承**：父类的私有成员仍然是私有的，父类的公有成员和保护成员作为子类的私有成员，而且不能被子类的子类访问。所以私有继承后，**不能再往下传递继承了！！！**



* 私有继承的对象不能直接调用父类里面的**成员函数或者数据成员**，需要在自己定义的类成员函数里去调用

  

**保护继承**：父类的所以共有成员和保护成员作为子类的保护成员，能够被它的子类的成员函数或者友元访问，但不能被类外的对象访问，父类的私有成员仍然为私有。



**类内的成员函数可以访问公有成员和保护成员、类外的对象只能访问公有成员**



#### 派生类的构造：

子类的对象由父类的子对象和子类的对象共同构成。分别由各自的构造函数初始化。

语法格式：

```c++
/*
	子类名（参数列表）：基类名1（参数列表1），...，基类名n（参数列表n），子对象1（子对象参数列表1）...
	{
		函数体；
	}；
*/
```

子类

调用顺序：

1. 父类的构造函数
2. 父类的子对象的构造函数，子对象按照父类中说明先后的顺序，与传参列表顺序无关
3. 子类自身的构造函数

#### 多继承：

​					解决顺序问题就解决了一切！！！  按照继承顺序来执行就好！！！

1. 按照各父类说明的顺序由左向右调用父类的构造函数
2. 按照子对象在子类中定义的顺序调用构造子对象函数
3. 子类自身的构造函数

```c++
class 派生类名：继承方式1 父类1 ，继承方式2 父类2 ...
{
    函数体
}；
```

继承的父类多了，会出现二义性！  所以要这样调用：

**对象名 . 类名 : : 成员函数名**



#### 虚拟继承与虚基类：

子类继承多个父类时，产生了多个不同途径来的继承来的同名成员在内存中有不同复制的数据不一致的问题，为了保证内存中只有父类成员的一份复制。

class 	子类名	：virtual	继承方式	父类名

**虚基类仅对后面的一个父类作用！**

main()函数创建对象，叫做最远派生类对象

调用顺序：

1. 虚基类的构造函数
2. 中间基类的构造函数
3. 子类自身的构造函数
4. 中间父类对虚基类的构造函数会被系统忽略，如果继承列表既有虚基类又有非虚基类，虚基类的构造函数优先调用！



### 访问权限

#### 无继承的情况下：

private:  

* 该类的成员函数
* 类的友元函数
* 其他任何都不能访问，包括该类的对象

protected：

	* 该类中的成员函数
	* 子类的成员函数
	* 该类的友元函数
	* 不能被类的对象访问

public：

	* 类中的成员函数
	* 子类的成员函数
	* 该类的友元函数
	* 类的对象



#### 继承的情况下：

private属性是不能被继承的，protected和private继承 降低了访问权限



protected数据成员，可以被子类对象的成员函数访问



### 类的成员函数

类的成员函数是一个定义的过程，不是访问的过程

它可以访问本类中所有的数据成员，所有的成员函数



### 对于继承的理解

#### 公有继承：

把基类的公有成员和保护成员当作自己的公有成员和保护成员

继承完之后，

​	在子类中的成员函数就可以调用基类所有的公有和保护成员，子类的对象只能访问子类和父类的所有公有成员函数、子类和父类的公有数据成员

#### 保护继承：

把基类的公有成员和 保护成员都当作自己的 保护成员 

继承完之后，子类的成员函数能访问父类的公有成员和保护成员 ，子类的对象只能访问基类和子类的所有公有和保护成员函数，有且只能访问自己的公有数据成员



#### 私有继承：

把基类的公有和保护成员当作自己的私有成员

子类只有成员函数才能取访问基类的成员 ，子类的对象什么都不能访问基类的，只能访问自己的



那基类的私有成员呢？

对不起，基类所有的私有成员，子类都不能直接访问，除非通过基类的公有成员函数里面有私有成员去访问



### 多态性与虚函数

多态性：指不同对象收到相同消息时，将产生不同的动作。

消息：指对类成员函数的调用

动作：指对成员函数的实现

联编：程序编译时或运行时，操作对象的过程叫做联编

静态联编：在编译、连接阶段，系统根据类型匹配等特征确定程序中的操作调用和执行该操作代码的关系的过程叫做静态联编，也叫先期联编

动态联编：程序在编译阶段不能确定将要调用的所有函数，某些函数的调用只有在程序运行才能确定，系统要求联编工作在程序运行时进行，这种工作叫做动态联编，也叫后期联编。

> 说人话就是调用带对象形参的函数的时候，会自动识别传的是哪个对象，然后对应的执行函数体

* 要想使某个成员函数具有多态性，必须用virtual关键字来标识



多态就希望你去继承，你去重载，这样才能发挥动态联编的功能

> 派生类的对象可以赋值给基类的对象
>
> 派生类的对象可以初始化基类的引用
>
> 派生类的对象不可以直接访问基类的成员
>
> 派生类的对象的地址可以赋给指向基类的指针，利用虚函数多态性，可以通过基类的指针调用子类的成员函数，如果虚函数或者不符合虚函数规则，基类的指针只能调用基类的成员函数



#### 多态虚函数的要求：

不满足要求，用virtual标识了也没用

1.函数名字相同

2.参数的数据类型相同

3.返回类型相同

### 运算符重载：

概念：对已有的运算符重新进行定义，赋予其另一种功能，以适应不同的数据类型

五种运算符不能重载：

. 		.*		::		?:		sizeof



意义：内置的数据类型，比如整形等，编译器自动如何相加，但是自定义类型不知道



语法： 返回类型 operator 运算符号（参数说明）；

> 用在类中有两种形式：
>
> 1.类中使用友元，类外来使用对象
>
> 2.直接定义为类的成员函数，方便调用，可以少传一个参数，因为有默认的this指针

```c++
class Person
{
    public:
    	int a;
    	int b;
}
Person	p1: p1.a=10;
Person 	p2: p2.b=10;

Person	p3= p1+p2;	//调用 operator+ 函数
//编译器不知道怎么去运算

//运算符重载就是一个调用函数进行运算

两种方法：1.成员函数重载	2.全局函数重载

Person operator+ (Person &p1, Person &p2)
{
    Person temp;
    temp.a= p1.a +p2.a
    temp.b= p1.b+p2.b
    return temp;
}
//调用：
Person p3= p1+p2   //简化调用，这个’+‘相当于调用operator这个函数
Person p3= p1.operator+(p2)  //本质调用
```

例子：体会一下重载和不重载之间的区别

```c++
#include<iostream>
using namespace std;

class RMB
{
    public:
        RMB(double d)
        {
            yuan=d;
            jf = (d-yuan)/100;
        };
        RMB interest (double rate);
        RMB add (RMB d);
        void display()
        {
            cout<<(yuan+jf/100)<<endl;
        };

        RMB operator + (RMB d)   //两个对象之间的运算，自己和传上来的对象d
        {
            return RMB( this->yuan + d.yuan +(this->jf + d.jf)/100);  
            //两个对象的元和元相加，jf和jf相加
        }

        RMB operator * (double rate) //对象和常数之间的运算
        {
            return RMB( (yuan+jf/100) * rate ); 
            // r对象带有两个数据成员，yuan和jf ，在这定义运算
        }
    private:
        unsigned int yuan;
        unsigned int jf ;

};

RMB RMB::interest(double rate)
{
    return RMB((yuan+jf/100)*rate);
}

RMB RMB::add (RMB d)
{
    return (yuan+ d.yuan+jf/100+d.jf/100);
}

RMB expense1(RMB r,double rate)
{
    RMB interest = r.interest(rate);
    return r.add(interest);
}

RMB expense2(RMB r,double rate)
{
    RMB interest = r * rate;  //算利息    
    return r + interest;    //连本带利息
}

main()
{
    RMB r=1000.0;
    double yrate=0.035;
    expense2(r,yrate).display();
    expense1(r,yrate).display();
}
```

运算符重载的返回方式：

##### 值返回和引用返回：

值返回，其实就是复制,把临时对象的值进行复制

引用返回，直接对参数进行修改

例如：

```
operator +() 以值返回
operator ++() 以引用返回
```

* 两个对象相加，不改变其中的一个对象，但它必须生成一个结果对象来存放加法的结果，并将该结果对象以值的方式返回给调用者。

  ```C++
  RMB operator + (RMB &s1,RMB &s2):
  {
      int jf = s1.jf +s2.jf;
      int yuan = s1.yuan + s2.yuan;
      RMB result(yuan,jf);
      return result ;      	//result是一个临时对象，将生成的结果值返回后，自动析构了
  }
  
  ```

//另外一种写法

```
 RMB operator + (RMB &s1,RMB &s2):
  {
      RMB temp;
      temp.yuan = s1.yuan+s2.yuan;
      temp.jf = s1.jf+s2.jf; 
      return temp;
  }
```


  ```
  
  operator++（）的返回值要求是左值，所以它不能以值返回。
  
  **因为 自增运算符要对传上来的参数运算，且要改变参数对象的值，所以要以引用的方式返回，保证修改的是形参对象的值。**

除此之外，加减乘除同理可得



运算符的左右两侧进行运算的变量，其实就是重载运算符函数的两个参数



有个问题，重载运算符之后，cout输出会出现问题  左移运算符编译器无法识别

​```c++
左移运算符的函数原型：
ostream operator <<(ostream& , 类名& )
  ```

#### 重载输出运算符

> 与iostream兼容的重载运算符必须是非成员函数

```c++
//只能全局变量定义，类内调用的时候记得用   友元  ！！！
ostream& operator << (ostream& cout ,MyInteger &myint (重载函数名和变量))
{
    cout<<myint.Num<<endl;
    return cout ; 
}

MyInteger & operator ++(MyIneger& myint)
{
    Num++;
    return *this		//注意返回值，类型要对,*this返回的是自身
}

void test()
{
    MyInteger myint;
    cout<< ++myint <<endl
}

```

```
ostream &operator<<(ostream& os ,const Sale_data &item)
{
	os<< item.isbn() ;
	return os;
}
//调用
cout<<s<<endl;
```



#### 输入重载

```
std::istream& operator>>(std::istream &is, Sales_data &item)
{
	double price = 0.0;
	is >> item.bookNo >> item.units_sold >> price;
	if (is)
		item.revenue = price * item.units_sold;
	else
		item = Sales_data();
	return is;
}

//调用
Sale_data s;
cin>>s;
```

#### 函数调用运算符

> 必须是成员函数，类内部还可以重载
>
> 

```C++
struct absInt{
	int opertaor() (int val) const {
	return val <0 ? -val:val;
	}
};

int i =-42;
absInt absobj;
int ui = absobj(i);

class GetInput
{
    public:
        GetInput(istream &i = cin):is(i){}
        string operator()() const
        {
            string str;
            getline(is,str);
            return is ? str:string();
        }
    private:
        istream &is;
    
};

class Replace
{
    public:
    Replace(int val):value(val){}
    bool operator()(int elem){
        return elem == value; 
    }
    private:
    int value;
};
```

#### 与lambda表达式等效

```C++
stable_sort(words.begin(),words.end(),
	[](const string&s1,const string&s2){return s1.size()<s2.size();});
	
class ShorterString
{
	public:
		bool operator()(const string&s1,const string&s2)
		{
			return s1.size()<s2.size();
		}
};

stable_sort(words.begin(),words.end(),ShorterString())
//第三个实参就是新建ShorterString对象，每次比较两个string就会调用这一对象，此对象将调用函数体，若s1的大小小于s2就返回true
```

#### 标准库定义的函数对象

> 标准库里面有定义的运算符的类。functional头文件

```C++
plus<T>					
minus<T>				
multiplies<T>
divides<T>
modules<T>
negate<T>

equal_to<T>
not_equal_to<T>
greater<T>
greater_equal<T>
less<T>
less_equal<T>

logical_and<T>
logical_or<T>
logical_not<T>
```

#### 例子：

```C++
plus<string>令string的加法运算符作用于string对象
moudules对应%

plus<int> intadd;
int sum = intadd(10,20)		//sum=30

//调用greater进行降序排列    
sort(svec.begin(),svec.end(),greater<string>());

vector<string*> nameTable ;
sort(nameTable.begin(),nameTable.end(),
    [](string *a,string *b) {return a<b ;});

sort(nameTable.begin(),nameTable.end(),less<string *>);
```

(a) 统计大于1024的值有多少个。 
(b) 找到第一个不等于pooh的字符串。
(c)将所有的值乘以2。

```
std::count_if(ivec.cbegin(), ivec.cend(), std::bind(std::greater<int>(), _1, 1024));

std::find_if(svec.cbegin(), svec.cend(), 
	std::bind(std::not_equal_to<std::string>(), _1, "pooh"));
	
std::transform(ivec.begin(), ivec.end(), ivec.begin(), 
	std::bind(std::multiplies<int>(), _1, 2));

占位符 _1 表示遍历中容器中的对象
```

 使用标准库函数对象判断一个给定的int值是否能被 int 容器中的所有元素整除 

```
auto data = { 2, 3, 4, 5 };
int input;
std::cin >> input;
std::modulus<int> mod;
auto predicator = [&](int i) { return 0 == mod(input, i); };
auto is_divisible = std::any_of(data.begin(), data.end(), predicator);
std::cout << (is_divisible ? "Yes!" : "No!") << std::endl;
```

#### 可调用对象与function：

> int(int,int) 接受两个int、返回一个int

```C++
//普通函数
int add(int i,int j){return i+j;};
// lambda
auto mod=[](int i,int j){ return i%j ;};
//函数对象
struct divede{
    int operator() (int denominator ,int divisor)
    {return denominator/divisor;}
}
```

建立一个从运算符到函数指针的映射关系

```C++
map<string ,int(*)(int,int)> binops;

binops.insert({"+",add});	
//add是一个函数指针,但是我们不能将mod和divede加入binops，mod和divede右自己的类的类型与int可能不匹配

所以要用function这个新的类型来解决上述问题
function<T> f;
function<T> f(obj);
f						f作为条件
f(args)					args是参数列表

function<int(int,int>)

//在这里声明一个function类型，它可以接受两个int，返回一个int的可调用对象。
    
function<int(int,int)> f1=add;
function<int(int,int)> f2=divide();
function<int(int,int)> f3=[](int i,int j){return i*j;};

//这样调用f对象
cout<<f1(4,2)<<endl; 
cout<<f2(4,2)<<endl;
cout<<f3(4,2)<<endl;

/*使用这个function类型我们可以重新定义map
1.列举了可调用对象与二元运算符对应关系表格
2.所有可调用对象都必须接受两个int，返回一个int
3.其中的元素可以是函数指针，函数对象或者lambda
*/
map<string,function<int(int,int)>> binops;

map<string,function<int(int,int)>> binops ={
    {"+",add},
    {"-",minus<int>},
    {"/",divide()},
    {"*",[](int i,int j){return i*j;}},
    {"%",mod},
}

//调用
binops["+"](10,5);  //先pair里面索引出函数对象，然后调用
```

#### 重载函数与function

```C++
int add(int i,int j){return i+j;};
Sale_data add(const Sale_data&,const Sale_data&);

binops.insert({"+",add}) //不知道要调用哪一个add
    
//解决方法
int (*fp)(int,int) =add  //函数指针
binops.insert({"+",fp});

binops.insert({"+",[](int a,int b){return add(a,b)})
```

#### 计算器完整代码

```C++
#include <iostream>
#include <string>
#include <map> 
#include <functional> 

int add(int i, int j) { return i + j; }
auto mod = [](int i, int j) { return i % j; };
struct Div { int operator ()(int i, int j) const { return i / j; } };

auto binops = std::map<std::string, std::function<int(int, int)>>
{
	{ "+", add },                               // function pointer 
	{ "-", std::minus<int>() },                 // library functor 
	{ "/", Div() },                             // user-defined functor 
	{ "*", [](int i, int j) { return i*j; } },  // unnamed lambda 
	{ "%", mod }                                // named lambda object 
};


int main()
{
	while (std::cout << "Pls enter as: num operator num :\n", true)
	{
		int lhs, rhs; std::string op;
		std::cin >> lhs >> op >> rhs;
		std::cout << binops[op](lhs, rhs) << std::endl;
	}

	return 0;
}
```





### 虚函数：

虚函数就是拿来实现多态的动态联编

virtual	：让函数的地址取决于创建的对象，在运行的时候才能决定调用谁

1. 虚函数就是：在父类中使用virtual关键字修饰的成员函数，它在程序运行时才能确定调用，时动态联编的基础

2. **虚函数从属于对象**，虚函数不能时静态成员函数，也不能是友元函数。

3. 虚函数只适用于有继承关系的类对象，不能是普通函数。

4. 虚函数一般用指针或者引用调用

5. 虚函数必须是共有继承，必须是成员函数

6. 要实现多态子类，子类要重写父类的函数


 override 的含义是重写基类中相同名称的虚函数，final 是阻止它的派生类重写当前虚函数 

```c++
class A
{
publics:
	virtual void dospeak()
    
}
class B:public A
{
public:
    void dospeak()
    
}
void dospeak(A &a )  	// 父类的引用在接收子类的对象	A  &a = b
{
    A.dospeak();
}

main()
{
    B b;
    A a;
    dospeak(b);			//创建的对象是b，就去调用B类的dospeak函数
    dospeak(a);
    A *p;
    p=&b;
    p->dospeak();   //基类指针用 ->调用成员函数
    
    A &r = b 		//引用直接使用 . 运算符
    r.dospeak();
}
```



#### 纯虚函数：

纯虚函数就是在父类中没有具体实现（没有函数体）的函数，父类的构造函数和析构函数不能调用纯虚函数

```
virtual 函数返回类型 函数名（参数列表）=0

子类中重写函数可以有函数体
```

纯虚函数是一个没有定义函数的语句的虚函数，值一定为0；

作为子类必须要给父类的纯虚函数提供一个相应的函数，即重载。

#### 多态使用纯虚函数的目的：

>  纯虚函数在基类中为子类保留一个位置，以便子类去重载覆盖定义
>
> 如果基类中没有使用纯虚函数，子类不能重载

#### 抽象类：

如果一个类中至少有一个纯虚函数，这个类叫做抽象类。

**抽象类无法实例化，无法创建抽象类的对象，但是抽象类的指针和引用可以访问子类的成员**

继承的子类必须重新定义纯虚函数，不然也会被视作抽象类

### 容器与继承

> 当我们使用容器存放继承体系的对象时，必须采取间接储存，**因为不允许在容器中保存不同类型的元素**，所以我们不能把具有继承关系的多种类型的对象直接存放到容器中

用智能指针存放具有继承关系的对象，就不用担心对象的转换和派生类被忽略了

``` 
vector<shared_ptr<Quote>> basket;
basket.push_back(make_shared<Queto>("0-13132-11e23",50))
basket.push_back(make_shared<Bulk_Queto>("0-13132-11e23",50,10,0.25))
cout<<basket.back()->net_price(15)<<endl;
```

智能指针也能发生转换，派生类指针转换为基类指针

### 模板：

函数模板:  不同数据类型都能用

> 若一个程序的功能是对某种特定的数据类型进行处理，把这个特定的数据类型变成参数，这样就可以传入不同类型的数据类型而实现相同的功能，这就模板的作用

```c++
template<class T>	//T就是数据类型参数<class T,class S> 也可以规定两种数据类型
void function(T &a, T &b)
{
    int temp=a;
    a=b;
    b=temp;
}

void test01()
{
	int a=10;
	float b=20.50;
	function(a,b);		自动类型推导
	function<int>(a,b);	指定类型 
    //function<int,float>(a,b);
}

推导出T的数据类型必须一致，自动类型推导是不会发生类型转换
```

普通函数和函数模板：

1. 普通函数和函数模板同时存在，优先调用普通函数
2. 可以通过空模板参数列表  强制调用  函数模板    f**uction<>(a,b)  加一个尖括号强制调用**
3. 函数模板可以发生函数重载
4. 如果函数模板可以产生更好的匹配，优先调用函数模板

区别：

函数模板是模板，而模板函数是在函数模板的基础之上，用一条语句传参然后重载，生成一个重载函数，该重载函数叫模板函数。

类模板:

```c++
template <类型形式参数表> class 类模板名
{
    public：
        
}


template <class NameT,class AgeT> //数据类型，NameT是一种数据类型，AgeT也是一种数据类型
class	Person
{
    public:
    Person(NameT name,AgeT age)
    {
        this->m_Name = name;
        this->m_Age = age;
    }
    NameT m_Name;
    AgeT m_Age;
}

类的成员函数模板
template <class T> 
返回类型 class 类名 <参数的数据类型> ::(形参列表)
{
    成员函数定义体
}



main()
{
    类的名字 <参数数据类型> 对象名（参数）
    Person<string,int> p1('zhangsan', 99)
}
```

### 异常处理

> try：将那些有可能产生错误的语句写在这里面
>
> catch: 放置出现异常后的语句
>
> throw: 抛出异常

打开文件的例子：

```c++
# include<fstream/iostream/cstdlib>
using namespace std;
int fun(int *argc , int **argv)
{
    ifstream file(argv[1]);
    char text[100];
    try
    {
        if(file.fail())
            throw argv[1];
        
	}
    catch(char *c)		//char * c 是拿来接收报错信息的
    {
		cout<<" error in opening file!"<< c <<endl;
        exit(1);
    }
    while(!file.eof())
    {
        file.getline(text,sizeof(text));
        cout<<text<<endl;
	}
}
```

catch语句只能容纳一个形参：

​	可以是 字符指针，获取报错信息

​	也可以是 数据类型 int  / double  x

**这种捕获是利用数据类型匹配实现的**

如果没有匹配到，就会调用默认异常处理程序abort（）

####  常用异常类

定义在头文件<stdexcept>

| 异常类            | 描述                                 |
| ----------------- | ------------------------------------ |
| exception         | 最常见的问题                         |
| runtime_error     | 只有在运行时才能检测出的问题         |
| range_error       | 生成的结果超出了有意义的值域范围     |
| overflow_error    | 计算上溢出                           |
| underflow_error   | 计算下溢出                           |
| logic_error       | 程序逻辑错误                         |
| domain_error      | 参数对应的结果值不存在               |
| invalid_arguement | 无效参数                             |
| length_error      | 试图创建一个超出该类型最大长度的对象 |
| out_of_range      | 使用一个超出有效范围的值             |



如果您想让 catch 块能够处理 try 块抛出的任何类型的异常，则必须在异常声明的括号内使用省略号 ...，如下所示：

```
try 
{   
	// 保护代码 
}catch(...) 
{  
	// 能处理任何异常的代码 
}
```





# STL：

standard Template  Library:	标准模板库

分为：容器（container）算法（algorithm）迭代器（iterator）

容器和算法之间通过迭代器无缝连接，STL几乎所有代码采用模板类或者模板函数

1. 容器：各种数据结构，如vector,list,deque,set,map
2. 算法：sort,find,copy,for_each
3. 迭代器：容器和算法间的胶合器
4. 仿函数：行为类似函数，可作为算法的某种策略
5. 适配器：一种用于修饰容器或者仿函数或迭代器的读写
6. 空间适配器：负责空间的配置和管理



#### 范围for语句

用来快速遍历容器或其他序列中所有的元素

```
for( auto &r :v)
{
	r=r*2
}
```



## 容器：

序列式容器：强调值的排序，序列式容器的每个元素均有固定位置

关联式容器：二叉树结构，各元素之间没有严格的物理上的顺序关系



顺序容器：

| 容器         | 解释                                                 |
| ------------ | ---------------------------------------------------- |
| vector       | 可变大小数组，在尾部之外的位置插入或删除元素可能很慢 |
| deque        | 双端队列，头尾插入删除速度很快                       |
| list         | 双向链表，双向顺序访问，链表任何位置进行插入和删除   |
| forward_list | 单向链表，单向顺序访问，任何位置插入和删除           |
| array        | 固定大小数组，不能添加和删除元素                     |
| string       | 与vector相似的容器，但专门用于保存字符串             |

### 容器操作：

| 容器操作             | 描述               |
| -------------------- | ------------------ |
| iterator             | 迭代器类型         |
| const_iterator       | 只读迭代器         |
| size_type            | 最大容器大小       |
| difference_type      | 两迭代器之间的距离 |
| value_type           | 元素类型           |
| a.swap() / swap(a,b) | 交换               |
| c.size()             | 大小               |
| c.max_size()         | 最大数目           |
| c.empty()            | 是否为空           |
| c.insert(args)       | 插入               |
| c.emplace(inits)     | 使用inits构造一个c |
| c.erase(args)        | 删除               |
| c.clear()            | 清空               |
| c.begin() / end()    | 迭代器             |
| cbegin() / cend()    | 只读               |

赋值：

```
list<string> a ={"...","..."}
vector<int> a(10,-1)
array<int,10> a1={...}
```

assign方法：（不适用于关联容器和array）

| seq.assign(b,e) | 将seq中的元素替换为迭代器b,e之间的元素，b,e不能指向seq中的元素 |
| --------------- | ------------------------------------------------------------ |
| seq.assign(il)  | 将seq中的元素替换为初始化列表il中的元素                      |
| seq.assign(n,t) | seq中的元素替换为n个值为t的元素                              |

容器操作注意点：

> forward_list 不支持 push_back 和 emplace_back ,pop_back
>
> vector 和string 不支持 push_front 和 emplace_front，pop_front

| 访问容器      |                                |
| ------------- | ------------------------------ |
| c.back()      | 尾元素的引用                   |
| c.front()     | 首元素的引用                   |
| c[n]          | 索引                           |
| c.at(n)       | 返回下标为n的引用              |
| c.pop_back()  | 删除最后一个                   |
| c.pop_front() | 删除第一个                     |
| c.resize(n)   | 保留前n个，多余丢弃            |
| c.resize(n,t) | 保留前n个，新添加元素初始化为t |
|               |                                |

容器的容量：

| shrink_to_fit() | 将capacity减小为size相同大小              |
| --------------- | ----------------------------------------- |
| capacity()      | 不重新分配内存空间，c最多可以保存多少元素 |
| reverse()       | 至少能容纳n个元素                         |
| size()          | 容器已经保存的数目                        |





### String：

> 可变长字符串

头文件：#include<string>

``using std::string ;``

初始化方式：

```
string s1 = "haiya"
string s2(s1) 等价于 string s2=s1

使用等于号：拷贝初始化
使用括号：直接初始化

对象与对象的拷贝，是通过一个临时对象来实现的
```

#### string的操作

| 操作            | 解释                                          |
| --------------- | --------------------------------------------- |
| os<<s           | 将s写入os中，返回os                           |
| is>>s           | 从is中读取字符串给s，字符串以空白分隔，返回is |
| getline（is,s） | 从is中读取一行赋给s，返回is                   |
| s.empty()       | 是否为空，返回布尔类型                        |
| s.size()        | 返回字符串的长度                              |
| s[n]            | 返回第n个字符 ，位置n从0记起                  |
| s1+s2           | 连接s1,s2                                     |
| s1=s2           | s2覆盖掉s1                                    |
| s1==s2          | 判断字符是否完全一样                          |
| s1!=s2          | 对字母的大小写非常敏感                        |
| <,> 等比较符号  | 按找字典中字符的顺序比较大小，对大小写敏感    |

* 当使用string对象相加的时候，至少保证+号一侧是string对象
* getline（cin,s）:会保留空白符，且遇到换行符停止



头文件#include<cctype>包含字符处理函数，例如：isalnum(c) , isalpha(c) ...



访问string中的每一个字符，两种方式 ：1.下标 2.迭代器

1.下标

```C++
for(char index=0 ; index!=s.size();++index)
        s[index] = 'X';
    
    // cout<<s<<endl;

    for(decltype(s.size()) index=0 ; index!=s.size();++index)
    {
        if (ispunct(s[index]))
        {
            for(decltype(s.size()) i=index ; i<s.size();i++)
                s[i]=s[i+1];
        }
    }
    cout<<s<<endl;
```



#### 按空格分割字符串

1. [使用 `std::string::find` 和 `std::string::substr` 函数在 C++ 中按空格分割字符串](https://www.delftstack.com/zh/howto/cpp/cpp-split-string-by-space/#%E4%BD%BF%E7%94%A8-stdstringfind-%E5%92%8C-stdstringsubstr-%E5%87%BD%E6%95%B0%E5%9C%A8-c%2b%2b-%E4%B8%AD%E6%8C%89%E7%A9%BA%E6%A0%BC%E5%88%86%E5%89%B2%E5%AD%97%E7%AC%A6%E4%B8%B2)
2. [使用 `std::istringstream` 和 `std::copy` 在 C++ 中按空格分割字符串](https://www.delftstack.com/zh/howto/cpp/cpp-split-string-by-space/#%E4%BD%BF%E7%94%A8-stdistringstream-%E5%92%8C-stdcopy-%E5%9C%A8-c%2b%2b-%E4%B8%AD%E6%8C%89%E7%A9%BA%E6%A0%BC%E5%88%86%E5%89%B2%E5%AD%97%E7%AC%A6%E4%B8%B2)
3. [使用 `std::getline` 和 `erase-remove` 习惯用法在 C++ 中按空格分割字符串](https://www.delftstack.com/zh/howto/cpp/cpp-split-string-by-space/#%E4%BD%BF%E7%94%A8-stdgetline-%E5%92%8C-erase-remove-%E4%B9%A0%E6%83%AF%E7%94%A8%E6%B3%95%E5%9C%A8-c%2b%2b-%E4%B8%AD%E6%8C%89%E7%A9%BA%E6%A0%BC%E5%88%86%E5%89%B2%E5%AD%97%E7%AC%A6%E4%B8%B2)

1.

 `find` 和 `substr` 是 `std::string` 内置函数，可用于通过字符串值或单个字符指定的任何定界符来分割字符串。

find 函数使用一个 string 参数，并返回给定子字符串开始的位置；

否则，如果找不到，则返回 `string::npos`。

因此，我们在 `while` 循环中进行迭代，直到 `find` 函数返回 `npos` 为止。同时，可以使用 `substr` 方法访问分隔符之前的字符串部分，在这种情况下，分隔符是单个空格字符，并存储到 `vector` 中供以后使用。之后，我们调用 `erase` 函数删除第一个序列，包括定界符，这时可能会进行新的迭代以重复操作。 

```
int main(){
    string text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                  "Sed laoreet sem leo, in posuere orci elementum.";
    string space_delimiter = " ";
    vector<string> words{};

    size_t pos = 0;
    while ((pos = text.find(space_delimiter)) != string::npos) {
        words.push_back(text.substr(0, pos));
        text.erase(0, pos + space_delimiter.length());
    }
    for (const auto &str : words) {
        cout << str << endl;
    }

    return EXIT_SUCCESS;
}
```

2.

 另外，我们可以使用 `istringstream` 类重新实现代码，该类为基于 `string` 的流提供输入/输出操作。一旦我们使用需要拆分的`字符串`值初始化了 `istringstream` 对象，就可以调用 `std::copy` 算法，将每个以空格分隔的字符串值输出到 `cout` 流。

请注意，此方法仅支持空格分隔符拆分，因为这是 `istringstream` 类的实现所提供的。 

```
int main(){
    string text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                  "Sed laoreet sem leo, in posuere orci elementum.";
    vector<string> words{};

    istringstream iss(text);
    copy(std::istream_iterator<string>(iss),
         std::istream_iterator<string>(),
         std::ostream_iterator<string>(cout, "\n"));

    return EXIT_SUCCESS;
}
```

3.

 先前解决方案的一个缺点是与解析的单词一起存储的标点符号。可以使用擦除删除惯用语解决此问题，该惯用语实质上是给定范围内的条件删除操作。

在这种情况下，我们对 `std::getline` 检索到的每个单词都调用此方法，以修剪其中的所有标点符号。注意，`ispunct` 函数对象作为第三个参数传递给 `remove_if` 算法，以检查标点符号。 

```C++
#include<iostream>
#include<string>
#include<vector>
#include<cctype>
#include<sstream>
#include <algorithm>
#include <iterator>
int main(){
    string text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                  "Sed laoreet sem leo, in posuere orci elementum.";
    char space_char = ' ';
    vector<string> words{};

    stringstream sstream(text);
    string word;
    while (std::getline(sstream, word, space_char)){
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
        words.push_back(word);
    }

    for (const auto &str : words) {
        cout << str << endl;
    }

    return EXIT_SUCCESS;
}
```

### 注意

```
getline(cin,text); 如果想输入带空格的字符串，必须用getline
cin>>text 遇到空格就终止了！
```



string是C++风格的字符串，而string本质是一个类

string和char *的区别：

char * 是一个指针，string是一个类，类内部封装了char *，这个字符串是一个char *的容器

```c++
string s1;

const char * str= 'hello world';
string s2(str);

string s3(s2);

string s4(10,'a')  #10个a
```

参数设置：

| args     | 解释                                  |
| -------- | ------------------------------------- |
| c,pos    | 从pos位置开始查找字符c                |
| s2,pos   | 从pos位置开始查找字符串s2             |
| cp,pos   | cp为指针，从pos开始，查找cp指向的字符 |
| cp,pos,n | 从pos开始，查找cp指向的数组前n个字符  |
|          |                                       |
|          |                                       |





###### **string的赋值操作**：

```c++
string str1;
str1= 'hello world';

string str2;
str2 = str1		#复制str1
    
string str3;
str3.assign('hello C++')
```

###### **string拼接：**

```c++
string str1= 'i '
str1 +='love playing game'
    
string str2='i'
string str3='love overwatch's
str3.append('love overwatch')
str3.append(str2,4)
str3.append(str2,2,4)	#按位置截取s，从2开始截取4个位置
```

###### **string查找**

```c++
string str1='abcdef'
int pos= str1.find('de')    #pos=3
#找不到字符串pos会返回-1 
    
pos= str1.rfind('de')
    
find和rfind的区别：
    rfind从右往左查找，find从左往右查找
    
str.find_first_of()    
str.find_last_of()
str.find_first_not_of()
str.find_last_not_of()
```

###### **string替换**

```c++
string str=‘abcdefg’
    str1.replace(1,3,'1111')	#从位置1到位置3替换，不是按位置替换，是把你整个给的字符串整个放进去
```

substr操作：它是原始string的一部分或者全部的拷贝，可以传递给substr一个可选的开始位置和计数值

```
string s("hello world")
string s2 = s.substr(0,5)
```



###### string比较：

```c++
string str1='xhello'
string str2='hello'
    str1.compare(str2)==0:相等
    str1.compare(str2)>0：大
    str1.compare(str2)<0：小
```

###### string存取：

```c++
string str='hello'
    1.[]访问
    for(int i= 0;i<sizeof(str) ; i++)
        cout<<str[i]<<endl;
	2.at方法
        for(int i = 0; i< str.size();i++)
        {
            cout<<str.at(i)<<endl;
        }
str[i]= '修改内容'
str.at(i)= '修改内容'
```

###### string的插入和删除：

```c++
string str='hello'			字符串是数组，位置编号从0开始
    str.insert(1,'hey')		在1位置插入hey
    str.insert(1,3,'hey')	#1到3位置插入hey
    
    str.erase(1，3)	#从1位置到3位置，删除
    str.append('o')
```

###### string子串：

```c++
string str='abcdefg'
    string subStr=str.substr(1,3)	#从位置1到位置3进行截取
```





### Vector：

> vector称为单端数组，与普通数组不同，vector可以动态扩展

**动态扩展不是后续添加，而是找一个更大的空间，把原来的数据整个放进去**

---

实际上vector是一个类模板，同时也是一个容器，能容纳绝大多数对象作为其元素，但是引用不是对象，所以不存在引用的vector。

> 容器包含容器：
>
> vector<vectot <T>  >   一定要再右侧加一个空格 

初始化：

```C++
无参构造：vector<T> v1;

通过区间的方式构造：vector <T> v2(v1.begin(), v1.end())

拷贝构造：vector <T> v3(v2)  	复制v2
    等价于 vector <T> v3=v2
    
vector<T> v4 (n,val)   重复构造

  赋值构造
    vector <T> v5 = {a,b,c,...}  等价于 vector <T> v6{a,b,c,...}
```

容器内元素为内置类型：初始值为0

```
vector <T> v1(10) :容器v1有10个元素，每个元素都为0
vector <T> v2{10} :容器v2有1个元素，值为10
同理 
(10,1) 10个1
{10,1} 一个10
```

* 不能以下标的形式添加元素

**不能用赋值语句直接初始化vector ** 

遍历输出：

```C++
for(auto i : num)  #对于num中的每一个元素
        cout<<"输出："<< i <<endl;
```



###### **vector赋值**：

```c++
v1.push_back(i)
    两种方法：
    v2=v1;
	v2.assign(v1.begin(), v1,end())
    v3.assign(10,100)				#10个100

```

###### vector容量和大小：

```c++
v1.empty():判断v1是否为空
    ca=v1.capacity()	获取容量
    v1.resize(15)		重新定义v1的容量，如果多了用0来填充空的位置
```

###### vector插入和删除

```
push_back(' ')	尾部插入
pop_back();		删除最后一个元素

v.insert(v.begin(), 100)	参数(迭代器，位置，插入元素)
insert(v.begin(), 2 , 1000)

v.erase(v.begin())
v.erase(v.begin,v.end) = v.clear()
```

###### vector数据存取：

```
at(int index)  	返回容器索引index所指的值
operater[];		返回索引index所指的数据
front()			返回容器的第一个数据元素
back()			返回容器的最后一个数据元素

```

互换容器：

swap（v）

v1.swap(v2)	#v1和v2容器中的元素呼唤

容器：vector	算法：for_each	

```c++
# include<iostream>
using namespace std;
#include<vector>
#include<algorithm>
void Test01()
{
    #创建了一个vector容器，数组
    vector<int> v;
    
    #向容器中插入数据,插入了10，20，30数据
    v.push_back(10);
    v.push_back(20);
    v.push_back(30);
    
    #通过迭代器访问容器中的数据
    vector<int>::iterator itBegin = v.begin();	起始迭代器，指向容器的第一个元素
    vector<int>::iterator itEnd = v.end();		结束迭代器，指向容器中的最后一个元素的下一个位置
        
    #遍历1
        while(itBegin != itEnd)
        {
            cout<< *itBegin<<endl;
            itBegin++;
        }
    #遍历2
    for(vector ::iterator:: it = itBegin; it != v.end(); it++)
    {
        cout<<*it<<endl;
    }
    #遍历3
    void myPrint(int val)
    {
        cout<<val<<endl;
    }
    for_each(v.begin(),v.end(), myPrint)  (起始迭代器，结束迭代器，调用函数名)
}
```



##### vector容器存放自定义类型数据：

```c++
class Person
{
    public:
    Person(string name, int age)
    {
        this->m_name = name;
        this->m_age = age;
    }
    string m_name;
    int m_age;
}
void test01()
{
    vector<Person> v;
    
    Person p1('aaa',10)
    Person p2('bbb',20)
    v.push_back(p1);
    v.push_back(p2);
    
    for(vector<Person>::iterator it =v.Begin(); it != v.end(); it++ )
    {
        cout<<"姓名："<<(*it).m_name<<endl;	# *it 就是指针，数据就是<Person>中的
    }
    	
}
#测试自定义类型指针,<>里面传的是指针
void test02()
{
    vector<Person*> v;
    Person p1('aaa',10)
    Person p2('bbb',20)
    v.push_back(&p1);
    v.push_back(&p2);
    
    for(vector<Person*>::iterator it =v.Begin(); it != v.end(); it++ )
    {
        cout<<"姓名："<<(*it)->.m_name<<endl;	
        # *it 就是指针，数据就是<Person*>中的
    }
}
```

##### Vector容器嵌套容器：

```c++
# include<vector>
void test02()
{
    vector< vector<int> > v;
    
    #创建小容器
    vector <int> v1;
    vector <int> v2;
    ...
    
    #向小容器中添加数据
    for(int i=0; i<3; i++)
    {
        v1.push_back(i+1)
        v2.push_back(i+2)
            ...
	}
    
    #把小容器添加到大容器
    v.push_back(v1);
    v.push_back(v2);
    
    #通过大容器，遍历所有数据
    for(vector<vector<int>>::iterator it = v.begin();it != v.end(); it++)
    {
        for(vector<int>::iterator vit= ( *it).begin() ; vit != (*it).end(); vit++) 
        {
            cout<<*vit<<endl;
        }
        cout<<endl;
    }
    
}
```



### Deque容器：

与vector的区别：

deque是双端数组，vector对于头部的插入效率低，数据量越大，效率越低，而deque对头部的插入删除比vector快，但访问元素的速度vector比deque快

deque内部有一个中控器

**插入、赋值、存储的操作全部与vector一样**

###### 构造函数：

```c++
deque<int> d1;
for(int i =0; i<10; i++)
{
    d1.push_back(i)
    Print(d1);
}

deque<int> d2(d1.begin(),d2.end())
 Print(d2);    

void Print(const deque<int> &d)
{
    for(deque <int> ::const_iterator it = d.begin() ; it != d.end(); it++)
    {
        //容器内数据不可修改
        cout<<*it<<endl;
    }
}
```

### Stack：

```c++
//构造函数：
stack<int> s;
//入栈
s.push(10);
s.push(20);

//查看栈顶并出栈
while (!s.empty())
{
    cout<<"栈顶元素为"<<s.top()<<endl;
    s.pop();
    
    s.size()；//栈的大小
}
```

### Queue：

```c++
//构造
queue<int> que;

//常用方法：
que.push(10);	//添加元素
que.pop();		//	从对头删除第一个	
que.back();		//	返回最后一个元素	查队尾：que.back().m_name
que.front;		//返回第一个元素		查对头：que.front().m_name

que.empty()
que.size()
```

### List：

```c++
//反转和排序
list<int> l1;
l1.reverse()
l1.sort();
l1.sort(函数名)；	//用自定义的函数定义排序规则
//所有不支持随机访问的迭代器，不可以用标准算法
 sort(l1.begin(),l1.end());	这是错的！！！
    
```

### Set/multiset：

所有元素插入时，会被自动排序。

**set不允许插入重复的元素，multiset不允许插入重复的操作**

```c++
set<int> s1;
//用insert插入
s1.insert(10)
    
empty() size()	swap()	都一样
    
//查找和统计
    find(key);		//若存在返回元素的迭代器，不存在返回set.end()
	count(key);		//统计key的个数

set<int>::iterator:: pos = s1.find(10);
//用迭代器来接收！
if(pos != s1.end())
{
    cout<<*pos<<endl;
}
else:
{
    cout<<"未找到"<<endl;
}

int num= s1.count(10);
cout<<num<<endl;
```

paie对组：

```c++
//
pair<string,int> p(string'Tom',20) 	//
    cout<<'姓名'<<p.frist<<'年龄'<<p.second<<endl;
pair<string,int> make_pair(string'Tom',20) 
```

**改变set的排序规则：**默认是从小到大

```c++
class MyCompare
{
    public:
    	bool  operator()(int v1, int v2)
            return v1 > v2;
}

//在插入之前进行声明排序规则
set<int,MyCompare>s2;
s2.insert(10);

//遍历
for(set<int,MyCompare>::iterator it =s2.begin();it!=s2.end();it++)
{
    cout<<*it<<endl;
}
```

```c++
#include<string>
#include<set>
class Person
{
    public:
        Person(string name, int age)
        {
            this->m_name= name;
            this->m_age= age;
        }
}
class Compare
{
    public:
    bool operator()(const &s1 , const &s2 )
    {
        //按照降序排列
        return s1.age > s2.age;
    }
}
void test()
{
    set<Person>s;
    Person s1('xiaoming',18);
    Person s1('xiaozhang',19); 
    
    set<Person,Compare> s;
    
    s.insert(s1);
    s.inset(s2);
    
    for(set<Person>::iterator it = s.begin(); it!= s.end(); it++)
    {
        
    }
}
```

### map/multimap:

简介：

1. map所有元素都是pair
2. pair的第一个元素是key值，起到索引的作用，第二个元素是value值（实际存储的值)
3. **所有元素都会根据元素的键值自动排序**

1. 优点：
   可以根据key值快速找到value值

2. map和mulitimap的区别：

   map不允许又重复的key值元素，value值可以

   multimap允许

```c++
构造：
map<T1,T2> mp;  	#注意有两个模板参数
    
empty(),size(),clear(),swap(),
erase(begin,end)：删除区间{begin，end}之间的元素，返回下一个迭代器的元素哦
    map<int,int> ::iterator position= m.find(10)
erase(key),
erase(position)删除迭代器中的所指的元素，返回下一个元素的迭代器

void test()
{
	map<int,int> m;
	
	m.insert(pair<int,int>(1,10))		#类似于python字典 1对应10
	PrintMap(m);
    
    map<int,int>m2;
    m2=m1;
}

void PrintMap()
{
	for (map<int,int> :: operator it =m.begin();it!=m.end();it++)
	{
		cout<<"key="<<(*it).first<<"value="<<(*it).end()<<ednl;	#指针可用first、second、end
	}
}
```

### 关联容器

> 两个主要关联容器：map和set

| 容器               | 描述                       |
| ------------------ | -------------------------- |
| map                | 关联数组，保存关键字的容器 |
| set                | 只保留关键字               |
| multimap           | 关键字可重复出现的map      |
| multiset           | 关键字可重复出现           |
| unordered_map      | 哈希的map                  |
| unordered_set      | 哈希的set                  |
| unordered_multimap | 哈希关键字可重复出现的map  |
| unordered_multiset | 哈希关键字可重复出现的set  |
|                    |                            |

| 类型        | -                                   |
| ----------- | ----------------------------------- |
| key_type    | 关键字类型                          |
| value_type  | 对于map，pair<key_type,mapped_type> |
| mapped_type | 值类型                              |



使用map统计单词数：

```
map<string ,size_t> word_count ;  //关键字是string类型，值是size_t类型
    string word;
    while(cin>>word)
        ++word_count[word];
    for(const auto &w : word_count)
        cout<<w.first<<"occurs" << w.second 
        <<((w.second >1) ? "times":"time")<<endl;
```

> 遍历map得到pair类型的对象，包含first和second两个数据类型
>
> map类型:  {key,value}



```C++
map<string,size_t> word_count ; //空容器

set<string> exclude = {{"Joyce","James"},
                   {"Austen","Jame"},
                       {"Dickens","Charles"},
}
```

容器运用：

```C++
 vector<int> ivec;
    for(vector<int>::size_type i=0 ;i!=10 ;++i)
    {
        ivec.push_back(i);
        ivec.push_back(i);
    }
    //iset 不重复 ，miset 重复
    set<int> iset(ivec.cbegin(),ivec.cend());
    multiset<int> miset(ivec.cbegin(),ivec.cend());
    cout<<ivec.size()<<" "<< iset.size()<<" "<<miset.size()<<endl;
```

map输入添加：

```
map<string, vector<string>> m;
for (string ln; cout << "Last name:\n", cin >> ln && ln != "@q";)
    for (string cn; cout << "|-Children's names:\n", cin >> cn && cn != "@q";)
    	m[ln].push_back(cn);
```



#### pair类型

> 在头文件utility中

```C++
pair<string,string> anon;
pair<string,size_t> word_count;
pair<string,vector<int>> line;

初始化：
 pair<string,string> authors{"James","Joyce"};
访问成员：
authors.first
authors.second
```

构造pair:

三种方法：

```
vec.push_back(std::make_pair(str, i));
vec.push_back({ str, i });
vec.push_back(std::pair<string, int>(str, i)); 
```



```C++
 pair<string,int>
     process(vector<string> &v)
 {
     if(!v.empty())
         return{v.back(),v.back().size()};  
     	//return make_pair(v.back(),v.back().size())
     else
         return pair<string,int>();
 }
```

#### 遍历关联容器

```
auto map_it = word_count.begin() 
cout<< map_it.first() << map_it.second<<endl;

//set容器只能读
set<int> iset ={0,1,2,3,4,5};
set<int>::iterator set_it = iset.begin();
if(set_it!=iset.end())
{
	cout<<*set_it<<endl;
}

//遍历
auto map_it = word_count.begin() 
while(map_it!=word_count.cend())
{
	cout<< map_it->first() << map_it->second<<endl;
	++map_it;
}

```

#### 关联容器算法

##### 添加

```
vector<int> ivec = (2,4,6,8);
set<int> set2;
set2.insert(ivec.cbegin(),ivec.cend()); //迭代器添加
set2.inert({1,3,5,7,9})		//手动添加

//map添加，四种方式,类型必须是pair
word_count.insert({word,1})
word_count.insert(make_pair(word,1))
word_count.insert(pair<string,size_t>(word,1))
word_count.insert(map(<string,size_t>::value_type(word,1)))
```

##### 删除

```
c.erase(k)
c.erase(p) //p是迭代器，指向一个元素
c.erase(b,e) //删除一个范围
```

##### 下标

> 支持下标操作
>
> c[k] :索引关键字得到值
>
> c.at(k) 

```
map<string,size_t> word_count;
word_count["Anna"]=1;

```

#### 访问

```C++
c.find(k)  
c.count(k)	//关键字等于k的数量
c.lower_bound(k)	//返回一个迭代器，指向第一个不小于k的元素
c.upper_bound(k)	//返回一个迭代器，指向第一个大于k的元素
c.equal_range(k)	//返回迭代器pair,表示关键字等于k的元素的范围
```

查找范围：

```
for(auto beg=authors.lower_bound(search_item),
			end=authors.upper_bound(search_item),
			beg != end;++beg)
	cout<<beg->second<<endl;
	d
for(auto pos=authors.equal_range(search_item);  
	pos.first != pos.second ; ++pos.first)
	cout<<pos.first->second<<endl;
//equal_range返回一个迭代器，first指向begin,second返回end
```

##### 文本转换

```C++
map<string,string> buildMap(ifstream &map_file)
{
    map<string,string> trans_map; //转换规则
    string key ;    //要转换的单词
    string value;   //替换后的内容
    //读取第一个单词存入key中，行中剩余内容存入value
    while(map_file>> key&&getline(map_file,value))
    {
        if(value.size()>1)//检查是否有转换规则
            trans_map[key] = value.substr(1);   //跳过前导空格
        else
            throw runtime_error("no rule for "+ key);
    }
    return trans_map;
}
const string &
transform(const string &s ,const map<string,string> &m)
{
    //如果单词在转换规则map
    auto map_it = m.find(s);;
    // 如果找到了有匹配规则，使用替换规则
    if(map_it != m.cend())
        return map_it->second;
    else
        return s;   //否则返回原string
}
void word_transform(ifstream &map_file,ifstream &input)
{
    auto trans_map=buildMap(map_file);
    string text;
    while(getline(input,text))
    {
        istringstream stream(text);
        string word ;
        bool firstword = true ;  //是否打印空格
        while(stream >>word)
        {
            if(firstword)
                firstword=false;
            else
                cout<<" " ;
            cout<<transform(word,trans_map);  //transform返回它的第一个参数或其转换之后的形式

        }
        cout<<endl; //完成一行的转换
    }
}

ifstream fp1("test2.txt"),fp2("test3.txt");
    word_transform(fp1,fp2);

```



### 仿函数：

函数对象在使用时，像普通函数一样调用

```c++
class Myadd()
{
	publics:
	Myprint()
	{
	  count=0; 记录调用次数
	}
	doprint();
	int operator()(int v1,int v2)
	{
		return v1+v2;
	}
	int count;		//定义仿函数状态
}
Myadd::doprint(Myadd &mp（引用对象名）,string test(参数名))
{
	cout<<test<<endl;
}
void test()
{
	Myadd myadd();				#创建函数对象
	cout<<myadd(10,10)<<endl;
	Myprint(myadd,'hello C++!')
}


```

内建仿函数：#include <functional>

1.算数仿函数：

```c++
template<class T> T plus(加减乘除)<T>
    negate（取反）：一元 	plus：二元
    
void test()
{
    plus<int> p ;		#内建对象
    cout<<p(20,20)<<endl;
}
```

2.关系仿函数：

```
template<class T> bool greater<T>
	sort(v.begin(),v.end(),greater<int>)
```



## 迭代器：

提供一种方法，使之能够依序访问某个容器所含的各个容器所含的各个元素，而又无需暴露该容器的内部表达方式。每个容器都有自己的专属迭代器，迭代器类似于指针，**初学阶段可以把迭代器理解为指针**

 

```C++
string s("some string")
if(s.begin()!=s.end())
{
    auto it = s.begin()
    *it = toupper(*it)
}
```

迭代器元素要解引用：``(*it).empty()``   如果不解引用，对it使用empty会报错

为了简化解引用，使用指针的箭头运算符 ``it->empty()``

* cbegin()和cend()是只读，就是常量限定读取



迭代器支持 +  - 等逻辑运算符，表示位置

用迭代器遍历：

```
for(auto it = text.cbegin();it!=text.cend();++it)
	cout<<*it<<endl;
```

### 额外的迭代器

| 类型                        |                |
| --------------------------- | -------------- |
| 插入迭代器 insert iterator  | 向容器插入元素 |
| 流迭代器  stream iterator   | 遍历关联的IO流 |
| 反向迭代器 reverse iterator | 向前移动       |
| 移动迭代器 move iterator    | 移动元素       |
|                             |                |
|                             |                |

#### 插入迭代器

> 插入操作：it = t

* back_inserter: 创建使用push_back的迭代器

* front_inserter:创建使用push_front的迭代器

* inserter: 指定位置插入的迭代器

```
*it = val 等效于 it = c.insert(it,val);	++it;

list<int> lst = {1,2,3,4,5};
list<int> lst2 ,lst3 ;
copy(lst.cbegin(),lst.cend(),front_inserter(lst2));
copy(lst.cbegin(),lst.cend(),inserter(lst3,lst3.begin()));
```

插入迭代器练习：

```C++
#include <iostream>
#include <algorithm>
#include <vector>
#include <list>
#include <iterator>

int main()
{
	std::vector<int> vec{ 1, 1, 3, 3, 5, 5, 7, 7, 9 };
	std::list<int> lst;

	std::unique_copy(vec.begin(), vec.end(), back_inserter(lst));
	for (auto i : lst)
		std::cout << i << " ";
	std::cout << std::endl;
	return 0;
}
```

使用三种迭代器插入：

```C++
#include <iostream>
#include <algorithm>
#include <vector>
#include <list>
#include <iterator>

using std::list; using std::copy; using std::cout; using std::endl;

template<typename Sequence>
void print(Sequence const& seq)
{
	for (const auto& i : seq)
		std::cout << i << " ";
	std::cout << std::endl;
}

int main()
{
	std::vector<int> vec{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };

	// inserter
	list<int> lst1;
	copy(vec.cbegin(), vec.cend(), inserter(lst1, lst1.begin()));
	print(lst1);

	// back_inserter
	list<int> lit2;
	copy(vec.cbegin(), vec.cend(), back_inserter(lit2));
	print(lit2);

	// front_inserter
	list<int> lst3;
	copy(vec.cbegin(), vec.cend(), front_inserter(lst3));
	print(lst3);

	return 0;
}
```



#### 流迭代器

> 创建流迭代器，使用istream_iterator 读取输入流

```C++
istream_iterator<int> int_it(cin)  //从cin读取int
istream_iterator<int> int_eof;		//尾后迭代器
ifstream in('afile');		
istream_iterator<string> str_it(in); //从"afiel"读取字符串

//读取数据，存入一个vector中
istream_iterator<int> in_iter(cin)  //从cin读取int
istream_iterator<int> eof;		//尾后迭代器
while(in_iter!=eof)    
{
    vec.push_back(*in_iter++);
}

istream_iterator<int> in_iter(cin),eof;
vector<int> vec(in_iter,eof) //根据迭代器范围构造容器
```

输出流操作：

```C++
istream_iterator<T> in(is); //in从输入流is中读取类型为T的值
istream_iterator<T> end;	// 尾后位置

in1 == in2
in1 != in2 
*in 		//读取值
++in,in++ 	//向后读取值
    
//累加实例
istream_iterator<int> in(cin),eof ;
cout<< accumulate(in,eof,0)<<endl;
```

输入流操作：

```C++
ostream_iterator<T> out(os); 
ostream_iterator<T> out(os,d); //d是每个值后面写入的，通常是“ ”或者“\n”

//流输出
ostream_iterator<int> out_iter(cout," ");
for(auto e:vec)
    *out_iter++ = e;

//用copy打印vec中的元素
copy(vec.begin(),vec.end(),out_iter);
cout<<endl;

```

使用流迭代器从txt中读取文本：

```
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iterator>

using std::string;

int main()
{
	std::ifstream ifs("H:/code/C++/Cpp_Primer_Answers/data/books.txt");
	std::istream_iterator<string> in(ifs), eof;
	std::vector<string> vec;
	std::copy(in, eof, back_inserter(vec));

	std::copy(vec.cbegin(), vec.cend(), std::ostream_iterator<string>(std::cout, "\n"));
	return 0;
}
```

判断奇数偶数，读取写入文件：

```C++
int num_test(char *a1,char *a2,char *a3)
{
	// if (argc != 4) return -1;
    cout<<a1<<a2<<a3<<endl;
	std::ifstream ifs(a1);
	std::ofstream ofs_odd(a2), ofs_even(a3);

	std::istream_iterator<int> in(ifs), in_eof;
	std::ostream_iterator<int> out_odd(ofs_odd, " "), out_even(ofs_even, "\n");

	std::for_each(in, in_eof, [&out_odd, &out_even](const int i)
	{
		*(i & 0x1 ? out_odd : out_even)++ = i;
	});

	return 0;
}
```



#### 反向迭代器

> ++it :向前迭代   --it：向后迭代
>
> 头部迭代器：crend() 
>
> 尾部迭代器：crbegin()

```
vector<int> vec={0,1,2,3,4,5,6,7,8,9}
for(auto r_iter=vec.crbegin();r_iter!=vec.crend();++r_iter)
	cout<<*r_iter<<endl;
```

例子：

```
//正常排序
sort(vec.begin(),vec.end())
//逆序
sort(vec.rbegin(),vec.rend())
```

练习：



## 算法：

有限的步骤，解决逻辑或数学上的问题，叫做算法。

质变算法：运算过程中会改变区间元素的内容，如拷贝，替换，删除等

非质变算法：运算过程中不会改变区间的元素，例如查找，计数，遍历，求极值



##### 求和

> 前两个参数是范围,第三个参数是初值

```
#对vec中的元素求和,求和初值为0
int sum = accumulate(vec.cbegin(),vec.cend(),0)
#连接字符串
string  sum = accumulate(v.cbegin(),v.cend(),string(""))
```

##### 写容器的算法

> 向目的位置写入元素

```
fill(vec.begin(),vec.end(),0)

fill(vec.begin(),vec.begin()+vec.size()/2,0)

#dest指向一个元素,从而dest开始的序列至少包含n个元素
fill_n(dest,n,val)
```

##### 插入迭代器:

> 向容器中添加元素的迭代器

```
vector<int> vec ;
auto it = back_inserter(vec); #通过it赋值添加到vec中
*it = 42;  #添加42到容器中

fill_n(back_inserter(vec),10,0)

```

##### 拷贝算法

```
int a1[] = {0,1,2,3,4,5,6,7,8,9}
int a2[sizeof(a1)/sizeof(*a1)];

auto ret = copy(begin(a1),end(a1),a2)
```

##### 替换算法

```C++
replace(ilst.cbegin(),ilst.cend(),0,42) #将容器中0全部替换为42
replace_copy(ilst.cbegin(),ilst.cend(),back_inserter(ivec),0,42)
```

##### 重排容器元素

> sort算法

```
void elimDups(vector<string> &words)
{
	#按照字典序排序
	sort(words.begin(),words.end());
	#unique使得单词只出现一次,不重复单词排列在范围的前部,返回指向不重复区域之后一个位置的迭代器
	auto end_unique = unique(words.begin(),words.end());
	#删除,不重复范围之后的
	words.erase(end_unique,words.end())
}
```

##### 逆序算法

```
reverse(begin,end);
reverse_copy(begin,end,dest) //逆序拷贝

remove_if(v1.begin(),v1.end(),[](int i){return i%2;});

//用插入迭代器拷贝到v2中
remove_if(v1.begin(),v1.end(),back_inserter(v2),[](int i){return i%2;});
```

##### 合并算法

> 适用于list 和forward_list

```
lst.merge(lst2)    //合并两个列表，且必须有序
lst.merge(lst2,comp) //元素从lst2中删除
```



##### 遍历算法：

```c++
#include<algorithm>
for_each:遍历容器
    
    vector<int>v;
    for_each(v.begin(),v.end(),print函数名)		#普通函数（不加括号）
    for_each(v.begin(),v.end(),print01())			#仿函数
    class print01
    {
        public:
        void operator()(int val)
        {
            cout<<val<<endl;
        }
    }
    
transform：从一个容器搬运到另一个容器
transform(iterator begin1,iterator end1, iterator begin2, _func);
begin1,end1:是原容器的开始结束迭代器	begin2：是目标容器的开始迭代器
    
    
   void test()
{
   vector<int>v;
    for(int i=0;i<10;i++)
    {
        v.push_back(i);
    }
    vector<int> vTarget;
    
    tranfrom(v.begin(),v.end(),vTarget.begin(),print01());	#仿函数
    
}
```

普通遍历:

```
for (auto it = v.cbegin(); it != pivot; ++it)
		std::cout << *it << " ";

#队列遍历
template<typename Sequence>
inline std::ostream& println(Sequence const& seq)
{
	for (auto const& elem : seq) std::cout << elem << " ";
	std::cout << std::endl;

	return std::cout;
}
```



##### 自定义算法

> 根据自定义的排序规则,来执行算法 ,例如:
>
> 排序算法:字典序排序sort,按长度大小排序(自定义),相同长度按字典序排序(stable_sort),分割排列(partition)
>
> 自定义规则:返回布尔值就行

```python
bool isShorter(const string &s1,const string &s2)
{
    return s1.size()< s2.size() ;
}



void sort_by_length()
{
    sort(words.begin(),words.end(),isShorter);
    for(auto const &i : words)
        cout<<i<<endl;
}

eliDumps(words);
stable_sort(words.begin(),words.end(),isShortter); #相同长度按字典序排序


#长度大于5的排在前面
bool predicate(const std::string &s)
{
	return s.size() >= 5;
}

int main()
{
	auto v = std::vector<std::string>{ "a", "as", "aasss", "aaaaassaa", "aaaaaabba", "aaa" };
	auto pivot = std::partition(v.begin(), v.end(), predicate);

	for (auto it = v.cbegin(); it != pivot; ++it)
		std::cout << *it << " ";
	std::cout << std::endl;

	return 0;
}
```

查找大于sz长度的单词，并打印满足条件的单词：

```C++
void biggies(vector<string> &words,vector<string>::size_type sz)
{
    elimDumps(words);
    stable_sort(words.begin(),words.end(),
    [sz](const string &a,const string &b)
    {return a.size()<b.size();});

    auto wc = find_if(words.begin(),words.end(),
        [sz](const string &a)
        {return a.size()>=sz;});   //wc是迭代器，位置是第一个长度大于sz的
    
    auto count = words.end()- wc ; //根据迭代器位置统计个数
    
    cout<< count << " " << make_plural(count,"word","s")
    <<" of length "<<sz<<" or longer"<<endl;

    for_each(wc,words.end(),    // 输出长度满足的单词
        [](const string &s){ cout<<s<<endl;});
    cout<<endl;
} 
```



##### 查找算法：

> find():查找元素
>     find(iterator begin,itertor end,value);

```c++

#内置数据类型   
vector<int>::  iterator it=find(v.begin(),v.end(),10)
    if (it==v.end())
    {
        cout<<"find it!"<<endl;
    }
	else:
		cout<<"None!"<<endl;
#自定义数据类型
class Person(string name,int age)
{
    public:
    Person()
    {
         this->name=name;
    	this->age=age;
    }
    string name;
    int age;
    
    bool operator==(const Person &p)
    {
        if(this->name==p.name&&this->age===p.age)
        {
            return Ture
        }
        else
        {
            return False
        }
    }
}
vector<Person>::  iterator it=find(v.begin(),v.end(),对象名(例p1))
    
    if (it==v.end())
    {
        cout<<"find it!"<<endl;
    }
	else:
		cout<<"None!"<<endl;	

void test02()
{
    vector <Person>v;
    Person p1('aaa',10);
    v.push_back(p1);
}


find_if：按条件查找元素，返回一个迭代器，指向第一个不满足条件的元素，若不存在这样的元素就返回end()
    
class TEST()
{
    public:
	bool operator()(int val)
    {
        return val>5
    }
}
void test01()
{
    vector <int>v;
	for (int i=0;i<10;i++)
    {
        v.push_back(i);
    }
     vector<int>::iterator it=find_if(v.begin(),v.end(),TEST())
    if (it==v.end())
    {
        cout<<"find it!"<<endl;
    }
	else:
		cout<<"None!"<<endl;
}


void test自定义()
{
    vector<int>
    vector<int>::iterator it=find_if(iterator begin,iterator end,TEST())
    if (it==v.end())
    {
        cout<<"find it!"<<endl;
    }
	else:
		cout<<"None!"<<endl;
    
}
   
class TEST
    {
        public:
        bool operator()(int val)||bool operator()(Person &p)
            return val>5;||p.name>5
    }

    adjacent_find()：查找相邻重复元素  
    binary_search()：二分查找法    
    count()：统计各个元素个数
    count_if()：按条件统计元素个数
```

##### lambda表达式

> auto f  = [ ] { return 0; }  捕获参数列表和函数体

接受两个参数，返回和

```
auto f = [](int i,int j) {return i+j;};
cout<<f(1,2)<<endl;
```



```
int i =1
auto f=[i](int j) {return i+j;};
cout<<f(2)<<endl;
```

可以捕获值或者是引用

显示捕获，直接指定变量

隐式捕获：由编译器推断



```
[] : 空列表
[names]：逗号间隔的变量名
[=]：值捕获
[&]：引用捕获
可混合使用
```

##### bind函数

参数绑定

```
bool check_size(const string&s,string::size_type sz)
{
	return s.size() >= sz ;
}

auto check6 = bind(check_size,_1,6)  #check_size中的第一个参数string&s绑定到_1上，6绑定到sz上

auto wc = find_if(words.begin(),words.end(),bind(check_size,_1,sz))

```

用bind重排参数顺序

```
for_each(words.begin(),words.end(),[&os,c](const string &s){os<<s<<c});

#代替
ostream &print(ostream &os,const string &s ,char c)
{
	return os<<s<<c;
}
```

使用bind的单词计数：

```
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>

using std::string;
using namespace std::placeholders;

bool isLesserThanOrEqualTo6(const string &s, string::size_type sz)
{
	return s.size() <= sz;
}

int main()
{
	std::vector<string> authors{ "1234567", "1234", "1234567890", "1234567", "12345" };
	std::cout << count_if(authors.cbegin(), authors.cend(), bind(isLesserThanOrEqualTo6, _1, 6)) << std::endl;
}
```





#### 文本查询程序

任务：

> 当程序读取文件时，它必须记住单词出现的每一行。因此，程序需要自己读取文件，并将每一行分解为独立的单词
>
> 输出：
>
> 1.能提取到每个单词出现的行号
>
> 2.行号必须升序且无重复
>
> 3.能打印给定行号中的文本

如何实现？

> * 使用vector<string>来保存整个输入文件，输入文件中的每行保存为vector中的一个元素。当需要打印一行时，可以用行号作为下标来提取行文本。
> * 我们使用一个istringstream 来将每行分解为下标来提取行文本
> * 我们使用一个set来保存每个单词在输入文本中出现行号，这保证了每行只出现一次且行号升序
> * 使用一个map来将每个单词与它出现的行号set关联起来，这样就可以提取任意单词的set



```C++
#include<iostream>
#include<vector>
#include<map>
#include<set>
#include<istream>
#include<fstream>
#include<string>
#include<memory>
#include<ostream>
#include<sstream>
using namespace std;

class QueryResult;
class TextQuery
{
    public:
        using line_no = vector<string>::size_type;
        TextQuery(ifstream&);
        QueryResult query(const string& ) const;
    private:
        shared_ptr<vector<string>> file;
        map<string,
            shared_ptr<set<line_no>>> wm;

};

TextQuery::TextQuery(ifstream &is):file(new vector<string>)
{
    string text;
    while(getline(is,text))
    {
        file->push_back(text);  //保存当前行的文本
        int n = file->size() -1 ; //当前的行号
        istringstream line(text); //文本分解为单词
        string word ;
        while(line>>word)
        {
            auto &lines=wm[word]; // 如果单词不在wm中就以之为下标添加

            if(!lines)      //我们第一次遇到这个单词时，此指针为空，就添加一个新的set
                lines.reset(new set<line_no>);
            lines->insert(n);   //把行号插入set中
        }
    }
};
class QueryResult
{
    friend ostream& print(ostream&,const QueryResult&);
    public:
    using line_no = vector<string>::size_type;
    QueryResult(string s,shared_ptr<set<line_no>>p, shared_ptr<vector<string>> f):  //查询结果的结构：单词-》行号-》行文本
    sought(s),lines(p),file(f){}

    private:
        string sought; //查询单词
        shared_ptr<set<line_no>> lines; //出现的行号
        shared_ptr<vector<string>> file; //输入文件

};

QueryResult TextQuery::query(const string &sought ) const
{
    static shared_ptr<set<line_no>> nodata(new set<line_no>); // 没查询到，分配一个新的set
    auto loc=wm.find(sought);
    if(loc == wm.end())
        return QueryResult(sought,nodata,file);
    else
        return QueryResult(sought,loc->second,file);
}

ostream &print(ostream &os ,const QueryResult &qr)
{
    os<<qr.sought<<" occurs "<<qr.lines->size()<<endl;
    for(auto num :*qr.lines)
        os<<"\t line" << num+1<<":" << *(qr.file->begin()+num)<<endl;
    return os;
}


void runQueries(ifstream &infile)
{
    TextQuery tq(infile);
    while(true)
    {
        cout<<"enter word to look for ,or q to quit:";
        string s;
        if(!(cin>>s)||s=="q") break;  //遇到文件尾或者输入q退出

        print(cout,tq.query(s))<<endl;
    }
}

int main()
{
    ifstream fp("test2.txt");
    runQueries(fp);
    system("pause");
    return 0;
}
```

