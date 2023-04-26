#                            C语言笔记

### ***首先先记-------------- 重要符号***

###### **·输出格式**：

int、long int 用%d输出		float和double用%f输出		

字符串用%s输出	

​	另外%和d/f之间可加附加字符：		

+l：表示长整型整数		

+m（数字）:表示输出数据的最小宽度		

+n（数字）：表示输出n位小数或者从字符串的左边截取n个字符   

`*尤其在printf函数里面可以指定数据的宽度和小数位数	example：printf("%6.1d",a);控制精度*`

###### **字符输入/输出函数**：**getchar()和putchar()**

语法：putchar(参数) ，参数为字符常量、字符变量和整型变量。**函数功能为输出该函数的ASCII值**，就是将数据值按照ASCII表依次输出，ABC

getchar函数没有参数而且只能接收一个字符，通常用于把输入的字符赋给一个字符变量变成赋值语句

example：

```c
char ch；
ch=getchar();
```

###### **字符串输入/输出函数：getc()和putc**			

参数stdin-键盘输入 	参数stdout-显示屏显示

putc()函数就是输出***一个字符***到指定的文件当中，函数作用：输出一个字符到指定位置的文件中。      语法：

```c
int putc(int ch,FILE *fp)
```

语法：getc()函数作用就是从fp所指向的文件中读取一个字符（我一般不用）

**字符串输出函数puts**——就是单纯输出字符串

```c
puts("hello,wlord!");			//就可以直接输出啦！
```

#### 解决一个问号：三目运算符

表达式a?表达式b:表达式c  （运算优先级低zhzhii）

 其执行步骤如下:

 (1)计算表达式a的值。
		 (2)如果表达式a的值为1为真,则执行表达式b。	
		 (3)如果表达式a的值为0为假,则执行表达式c。

#### 清空缓存区：

fflush(stdin);     将scanf输入缓存区的数据全部清除

### 常见疑惑

##### c语言的三个标准C89、C99、C11

![1606724723048](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1606724723048.png)

![1606725085191](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1606725085191.png)

##### c语言运行的内幕--编辑器和编译器。

- 常见问题：段错误、吐核、栈溢出。

  ​	**代码编辑器**：一个好的编辑器可以节省开发时间，提高工作效率，它们都能提供非常方便易用的开发环境。你可以用它们来编写代码，查看源文件和文档等，简化你的工作。以下是一些常用的代码编辑器，每个不同的编辑器都有不尽相同的目标用户群体。

  > NodePad++ :Notepad++ 是一款免费但优秀的代码编辑器，运行在 Windows 环境下，可以支持多种编程语言。 Vim :
  > Vim 是从 vi 发展出来的一个文本编辑器，在程序员中被广泛使用，运行在Linux环境下。 GNU Emacs : Emacs
  > 是一个轻便、可扩展、免费的编辑器，它比其它的编辑器要更强大，是一个整合环境，或可称它为集成开发环境。它可以处理文字，图像，高亮语法，将代码更直观地展现给开发者。

  **编译器**：C语言代码由固定的词汇按照固定的格式组织起来，简单直观，程序员容易识别和理解，但是对于CPU，C语言代码就是天书，根本不认识，CPU只认识几百个二进制形式的指令。**这就需要一个工具，将C语言代码转换成CPU能够识别的**二进制指令，**也就是将代码加工成 .exe 程序**；这个工具是一个特殊的软件，叫做编译器（Compiler）。

  编译器能够识别代码中的词汇、句子以及各种特定的格式，并将他们转换成计算机能够识别的二进制形式，这个过程称为编译

  ###### **吐核**

  --linus编译环境中执行编译环境好的文件时会出

  文件（数组开的太大了）太大了，多次吐核过于占用内存，会影响机器的性能

  ![1606725855310](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1606725855310.png)

  ##### 段错误

   [C语言再学习 -- 段错误（核心已转储）_不积跬步，无以至千里-CSDN博客_c语言段错误](https://blog.csdn.net/qq_29350001/article/details/53780697?ops_request_misc=%7B%22request%5Fid%22%3A%22160672611719721940265699%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=160672611719721940265699&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-53780697.pc_first_rank_v2_rank_v28&utm_term=段错误&spm=1018.2118.3001.4449) 

  - 访问不存在的内存地址
  - 访问系统保护的内存地址
  - 空指针废弃
  - 堆栈溢出
  - 内存越界

  1. **栈溢出**

     ![1606726588705](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1606726588705.png)

  

- **编码：ASCII、GBK、Unicode。为什么会出现乱码？**

  ​	额。。。。文件类型的问题需要将文件进行转换

- **缓冲区**

  ​	 缓冲区，它是内存空间的一部分。也就是说，在内存空间中预留了一定的存储空间，这些存储空间用来缓冲输入或输出的数据，这部分预留的空间就叫做缓冲区，显然缓冲区是具有一定大小的。
  ​    缓冲区根据其对应的是输入设备还是输出设备，分为输入缓冲区和 缓冲区就是一块内存区，它用在输入输出设备和CPU之间，用来存储数据。它使得低速的输入输出设备和高速的CPU能够协调工作，避免低速的输入输出设备占用CPU，解放出CPU，使其能够高效率工作。 输出缓冲区。 

- 函数式编程思想及多文件编程

  

  #### 										输出的高级用法

  ​	puts()：*只*能输出字符串，并且输出结束后会自动换行。

  ​	putchar()：只能输出单个字符。  

  ​	printf()：可以输出各种类型的数据。 

   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200220110120467.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTc0Mzc5OQ==,size_16,color_FFFFFF,t_70) 

   例子：**%-9d中，d表示以十进制输出，9表示最少占9个字符的宽度，宽度不足以空格补齐，-表示左对齐。**综合起来，%-9d表示以十进制输出，左对齐，宽度最小为9个字符。大家可以亲自试试%9d的输出效果 。

  ### 什么是集成开发环境

  实际开发中，除了编译器是必须的工具，我们往往还需要很多其他辅助软件，例如：

  编辑器：用来编写代码，并且给代码着色，以方便阅读；

  代码提示器：输入部分代码，即可提示全部代码，加速代码的编写过程；

  调试器：观察程序的每一个运行步骤，发现程序的逻辑错误；

  项目管理工具：对程序涉及到的所有资源进行管理，包括源文件、图片、视频、第三方库等；

  漂亮的界面：各种按钮、面板、菜单、窗口等控件整齐排布，操作更方便。

  这些工具通常被打包在一起，统一发布和安装，例如 Visual Studio、Dev C++、Xcode、Visual C++ 6.0、C-Free、Code::Blocks 等，它们统称为集成开发环境（IDE，Integrated Development Environment）。

  


### 数组的查找功能！

```c
#include <stdio.h>
int getIndex(int arr[5],int value)
{
    int i;
    int index;
    for(i=0;i<5;i++)
    {
       if(arr[i]==value)	//这个i就是查找数组里哪一个数值符合把它找出来
        {
            index=i;	//返回数组下标
            break;
        }  
       index=-1;
    }
    return index;	//返回参数
}
```

###### 字符串函数：

（1）strlen（s1）获取字符串的长度，汉字和字母的长度是不一样的。

   (2)strcmp（s1，s2）将s1和s2转换成ASCLL值然后再进行比较，若返回的结果为0表示s1和s2相等，返回值为1表示s1比s2大，返回值为-1表示s1比s2小。

​	(3)strcpy(s1，s2)将字符串s1，s2进行拷贝

（4）strncmp(s1,s2,n)  s1比较的第一个字符串，s2比较的第二个字符串，n要比较的最大字符串数。该函数的返回值：返回值<0，s1小于s2，返回值>0s1大于s2，返回值=0，s1等于s2

### 函数的调用及参数的作用：

​		·自定义函数：【数据类型】函数名（【参数】）；

​		·调用：函数名（【参数】）；

**参数分为形参和实参两种** ：  形参，就是在定义函数名和函数体使用的参数，目的就是用来接收该函数传入的参数	实参，在调用函数时使用的参数。实参可以是常量·变量·表达式·函数

 ![mark](http://myphoto.mtianyan.cn/blog/180604/gdj94a4jCD.png?imageslim) 

![IMG_20201129_221611](D:\Huawei Share\OneHop\IMG_20201129_221611.jpg)

**前方高能注意---错误点：** 

·无论实参是何种类型的量，在进行函数调用时，它们都必须具有**确定的值**，以便把这些值传送给形参。因此应预先用赋值等办法使实参获得确定值。 比如我在写学生管理系统是用的index预先赋给index值再传给形参！

·在参数传递时，实参和形参在数量上、类型上和顺序上必须严格一致，否则会发生类型不匹配的错误，**那个int age 惨痛的记忆**还记得么！！！！

###### 参数的传递方式：值传递和地址传递

- 值传递：在main函数中将值传递给形参，让被调函数进行计算，最后将值返回main函数
- 地址传递：用指针方式定义形参--形参指针变量，可以修改main函数中的输出值

### 函数的返回值：

返回值的意思就是函数被调用之后，执行体中的程序段所取得的并返回给主函数的值

这个例子就是把返回值返回主函数

![1606553762076](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1606553762076.png)



### 递归函数：

###### 函数嵌套调用：

在c语言中函数是独立，并列的模块。通过调用与被调用的方式相关联。虽然在函数内部不能嵌套定义函数但是可以嵌套调用其他函数

也就是说 不止是main函数，自定义的函数也可以互相调用

###### 		递归调用：分为直接递归调用和间接递归调用

直接递归就是在同一个函数中调用自己，但是会无休止的进行

```c
//递归调用实现n的阶乘
# include<stdio.h>
long fact(int n);
long fact(int n)
{
 long t;
    if(n==0||n==1)
        t=1;
    else
        t=n*fact(n-1);    //一个调用直接搞定
    return (t);             
}
```



###### 		一个函数在它的函数体内调用它自身

递归函数将反复调用自身，每调用一次就进入新的一层

![1606554085353](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1606554085353.png)

这就是递归！解决阶乘问题哦！！！！

![1606554445536](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1606554445536.png)





## 文件流程

###### （1）头文件类型：

​			<1>stdio.h是编译器自带的头文件。引用头文件相当于复制头文件的内容。

​			<2>#include<file>引用用户有文件，在当前目录中搜索名为file的文件。

​			<3>math.h 用数学运算的库函数的头文件。

###### （2）typedef

是为用户自定义类型取一个新名字    typeof  原类型名  新类型名

例如   typeof  int   t  ；typeof  printf   f

就代表下面的代码 t直接代替int，f直接代替printf

经常在结构体里出现

```c
typeof struct student
{
    long num;
    char name[10];
}STUDENT;
STUDENT stu1,stu2   ==  strcut student stu1,stu2;
```

![1606445650124](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1606445650124.png)

###### （3）打开文件

```c
文件指针名=fopen("file name","使用文件方式")；
FILE *fp
    fp=fopen("D:\\books.txt","r");//打开D盘根目录下books文件

```

![1606446011673](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1606446011673.png)

*其实，r=read、w等于write、a=append追加、t=txet文本文件、+就是可读入和写*入

！！！在打开一个文件时，如果出错，fopen函数将返回一个空指针值NULL。这个信息可以用来判断是否打开了文件，并做出相应处理。

```c
if(fp=fopen("d:\\students.txt","r")==NULL)
{
    printf("\打开文件失败 d:\\students.txt file！");
    exit(1);//这是退出程序的意思
}
```



###### 关闭文件：

```c
fclose（文件指针）
fclose(FILE *FP);
```

###### 写入文件：

```c
#include <stdio.h>
 
int main()
{						//fputs()把字符串s写到fp所指输出流里
   FILE *fp = NULL;		//fprintf把字符串写到函数里
 
   fp = fopen("/tmp/test.txt", "w+");	//打开文件
   fprintf(fp, "This is testing for fprintf...\n");
   fputs("This is testing for fputs...\n", fp);
   fclose(fp);
```

###### 读取文件：

```
int *fgets(FILE *fp)；
```

fgets()就是从fp所指向的输入文件中读取一个字符。

int fscanf（）在读取文件中字符串遇到第一个空格和换行符是会停止读取。

## 读写文件函数

###### fputs函数

![1606786546781](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1606786546781.png)

###### fgets函数

![1606786646048](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1606786646048.png)

###### fwrite函数

![1606786762563](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1606786762563.png)

###### fread函数

![1606786728202](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1606786728202.png)

###### fscanf函数

![1606786699183](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1606786699183.png)

###### fprintf函数

![1606786474762](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1606786474762.png)

###### fseek函数

![1606786578723](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1606786578723.png)

- ![1606786624923](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1606786624923.png)

  ###### rewind函数

  ![1606786443974](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1606786443974.png)

  ###### 例题 


![1606786799338](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1606786799338.png)



![1606786819244](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1606786819244.png)



## 指针

###### 指针与数组

1. 数组名是一个地址信息，也是数组的第一个元素的地址
2. %p就是打印地址
3. 当指针指向数组元素时，对指针进行加减运算，相当于距离指针指向元素向前或者向后移动几个元素例如：   *(p+n)
4. 左值：用于识别或定位一个储存位置的标识符，必须是可变的
5. *指针数组：int *p[5]   每一个数组元素存放一个指针变量*
6. 数组指针：int （*p）[5]  指向数组的指针

###### sizeof()

计算字符串或整形所用字节



## 结构体

###### 定义

```c
struct 结构体名
{
    成员说明表列
}；
    struct 结构体名 结构体变量列表
```

###### 结构体数组

```c
struct  结构体名
{
    成员列表
}数字名字[大小]；//结构体数组的引用说明数组里每一个元素都是含有结构体成员列表的

```

###### 结构体指针

struct   结构体名   *结构体指针名；

```c
struct student
{
    long number;
    char name[20];
    char sex;
    float math;
};
strcut student stu;
strcut student *p=stu;

```

###### 结构体作函数参数

```c
strcut date
{
	int nian;				//这里同样可以运行值传递和地址传递
	int yue;
	int ri;
};
int riqi(struct date *p)
{
(*p).nian=2018;
(*p).yue=5;
(*p).ri=22;
	return 0;
}
```

###### 结构体作函数返回值

```c
strcut store
{
    。。。
}；
  strcut store goods；
    
 stuct store information()  
{
    scanf
        return goods;         //将输入的信息返回进结构体里
}

```

运算符：

1. 圆点运算符---一般形式（*结构体指针变量名）.成员名

   例如：（*p）.name

2. 指向运算符---一般形式 ：结构体指针变量名**- >**成员名

   例如 p->name

## 枚举类型

enum 枚举变量名{取值列表}；

enum 枚举名{} 枚举变量；

for instance:   

enum month{jan=1,feb=2,mar,apr,may}   mon;  如果枚举常量没有赋值，系统默认第一个枚举常量的值为0

int k; scanf(k);														后面的枚举值依次加1

mon=(enum month) k;



## 宏定义#define

1.**无参宏**——  #define 标识符  字符串

例如：#define N 3.1415926     之后直接用N代替3.1415926

如果想终止其作用域可使用  #undef命令

**宏定义和typeof的区别：宏定义只做简单的字符串代换，在预处理完成。而typeof是在编译时处理的，不是简单的替换而是对说明符的重新命名，被命名的标识符具有类型定义说明的功能**

2.带参宏

带参宏定义 #define 宏名（形参） 字符串

带参宏调用 宏名（实参）；

```c
#define MAX(a,b) (a>b)?a:b
int main()
{
    max=MAX(a,b);
}
或者
#define SQ(y) (y)*(y)   //注意一个问题 为了避免出错 宏定义中y变量都要带括号。因为系统运算循序的问题，所以要带上括号提高优先级
   
```

