# 				 Python



### 一、基本操作命令

C:\Users\银晗\AppData\Local\JetBrains

C:\Users\银晗\AppData\Roaming\JetBrains\PyCharm2020.3

如果想输出列表和字典不想带有[' ']或者{' ': ' '},可以使用   列表名.repalce（' [ ' , '  '）将空格代替【

还可以：字符串.join(列表)，将字符串添加到列表的每个元素之间，组成新的字符串

**buf= '  ,   ' .  join(self.name)		将name列表改为逗号连接**

1.  **print命令**

   print('' or "") 不用分号结束

   print（1，2，sep='*'）输出： 1 * 2	sep就是在两个输出值中间加东西

   一般利用变量直接输出		message="hello,python"	print(message)

   地板除法：10//3 =3（自动忽略掉小数部分）

   **乘方运算先与乘法**

   保留小数用round（变量名，保留位数）例子：num=3.333333 	round(num,2)=3.33

   格式化输出：print('%d'  %3或者（括号里写多个数据)）

   改大写用upper.()	改小写用lower.()	首字母大写用title.()

   example:		

   ```python
   name=zhang yinhan
   print(upper.())	print(lower.())	print(title.())
   ```

3. 连接字符串直接用+号

   full_name="zhang"+" "+ "yin"+"han"	print(full_name)

   删除空白：开头 .lstrip()	两端.strip() .rstrip( )删除所有空白

   **使用字符串时注意，双引号之间才能有单引号**

3. 产生随机数：import random

   random.randint( , )随机整数	randon.uniform( , )随机浮点数

   random.choice( )可以从列表，字典，元组中任意选取一个元素返回一个列表

   random.shuffle()可以将一个序列中的元素，随机打乱

4. 字符串函数： 

   1. 字符串.replace(原来的字符串，新的字符串，替换的次数（一个长字符串中可能有多个字符串是一样的所																									以是替换几次），默认是全部替换)

      作用：将一个长字符串中的某一个字符串进行替换

   2. 字符串分割：str .split(sub_str,count)  sub_str 就是按照什么内容切割字符串，count就是切割的次数

      作用：可将字符串中每一个小的字符串拆开存到数组里

   3. 字符串连接：str . join(可迭代对象 )  可迭代对象可以是字符串和列表（列表中每一个数据都是字符串）

      作用：将str添加到可迭代对象的两个对象之间  

      ```python
      mystr='_'.join('hello')		#会把_加入到hello的没两个元素之间
      # 输出：h_e_l_l_o
      ```

   4. 字符串.capitalize( )  把字符串的第一个字符大写

   5. . title( )把每一个字符串开头大写

   6. .islower( )判断字符串开头是否是小写         正确返回Ture  错误返回 Flase

   7. .startwith( str)判断字符串是否以str开头    mystr.startwith(hello)判断mystr是否以hello开头

   8. .endwith( str) 同理判断是否以什么结束    

   9. 字符串.ljust( width)  .rjust(width)  .center(填充空格数)返回原字符串左/右/中对齐，并使用空格填充至长度width

   10.  字符串.partition(str)   将str前后切开，分成3个字符串

5.  使用整形赋值时，用变量打印时python会无法识别，这时用str(变量名)

   age=23	message='happy'+str(age)+'birthday'

6. **python把0，空字符串，和None看成Flase其他都看成Ture**

   短路计算：a and b 如果a为Flase则结果为Flase，返回a的值；如果a为Ture，则返回b的值	a or b 如果a为Ture则结果为Ture，返回a的值，如果a为Flase则返回b的值

7. 注释：Ctrl+/（单行）；多行用三个'''    '''  或者三个"""     """

8. 列表也不用定义了，而且用 [] 包括下标

   修改：要想修改数组元素直接给对应数组位置重新赋值	message[0]='hello'

   添加：.append()会将元素添加到数组最后	message.append('hi')	

   插入：.insert(位置，内容)	message.(0,'hi')

   删除：del +数组名[位置]	 del message[0]

   ​			.pop(位置) 删除并且打印删除的信息	位置为空默认最后一条

   ​			.remove(内容)	删除数组中对应的内容

   **count()可以找到数组某个元素出现的次数	print（数组名.count(查找的元素)）**

   **index（）某个元素出现多次，返回第一次出现的位置**

   **find()在列表中查找某个内容（查找的字符串，开始位置，结束位置)   没找到返回-1**

    要想改变列表的输出方式，使用列表推导式

9. 组织列表（排序）

   1. .sort()按字母排序  .sort(reverse=Ture)按照相反的顺序		但是都是永久性排序不可恢复
   2. sorted(数组名)  临时排序
   3. 倒着打印列表 .reverse()   message.reverse()  print(message)    要想恢复列表再使用一次reverse
   4. 获取列表长度 len(数组名)

10. **for** 自定义名字 **in** 数组名  ＋： **缩进代表括号**，几个元素循环多少遍

11. 打印数字列表 range(开始数字，结束数字-1)      for num in range(1,5)

   而且还可以规定公差  **range(开始数字，结束数字-1,公差)**

11. list()可以直接转化为列表   number=list(range(1,6))

12. 计算函数  max()  min()  sum()  计算**列表中的**最大值，最小值，总和

13. 切片 数组名[开始位置，结束位置，步长]   两头位置为空代表复制  yours=mine[:]     开始位置比结束位置大，步长可以为负数

    切片只能用在字符串中；string[ :  : -1]  快速逆置列表

14. 元组：dimension（内容）  **元组内容不可修改！！！**只能重新定义

15. and就是与  ； or就是或  ；检查元素是否在数组中用in ；'hello' in message；不在用not in

16. if-elif-else(可省略)

    **在执行多代码块的时候，for ，if ，else 后面都要加上冒号：表示括号**

17. 字典

    字典名={'key'：'value'     }  **记得用大括号**		print(字典名['key'])

    添加、修改：字典名['key']=value值

    删除元素：del 字典名['key']   如果用pop返回的是对应key的value值

    ​					.clear()	就清空字典

    ​					keys()输出名字，不输出值

    合并字典：dict1.update(要添加的字典名)   将添加的字典里的键值对直接合并到dict1里面

    ​					但是如果被合并的字典中包含已经存在的键值对时会覆盖原有的键值对

    遍历字典：for key in dict:

    ​					print(f’‘ {key} {dict[key]}’)

    根据key找value值：value=dict[key]

    **根据value值找key：要找的key值=[value for k,v in dict if v==已知的value值]**

18. input('')   message=input('请输入:')  **数据类型为字符串**

    要使用整形只能强制类型转换  age=18 age=int（age）

19. while循环正常使用，break同样   可使用一个标志  active=Ture （运行） active=flase（不允许）

    ​			while  active<=5

20. 函数：关键字def 告诉python你要定义一个函数，冒号结尾

    def name(username):     print('hello')	  name('randy')	

    **调用函数：name()或者message.name( )**

    形参和实参的传递和C差不多，可以一一对应传递，形参可以直接赋值不用实参传递

    *string代表空元组，可以将多个实参存到这里面如何输出

21. 类------***-！！！代码规范：类定义的前后需要两个空行***

    定义：class 类名( object)：	创建对象=类名()		调用：对象.类内函数名()

    常用术语：创建出来的**对象**叫做**类的实例**；创建对象的**动作**叫做**实例化**；**对象的**属性**叫做**实例属性；

    ​					对象调用的方法叫做**实例方法**

    类属性就定义在类内、类内方法前的变量

    魔法方法：

    1. _ _ init_  _ 		调用时机：在创建对象后会自动调用	**作用：给对象添加属性**，给对象属性一个初始值（构造函数）

    2. _ _ str_ _  	调用时机：print(对象) 或者 str（对象）会自动调用   方法只能返回一个字符串

        **必须有return**			**作用：里面写要打印的信息，要想改输出就改这里**
        
        ```python
        class Dog(object):
            def __str__():
                if list:
                    return f' 打印内容'
                else：
                	return f' 打印内容'
        ```
        
        **将类的对象添加到列表中（容器），直接打印列表显示的是自定义对象的引用地址**
        
        列表：furniture_list 		**buf=[  x.name(属性名)  for x in furniture_list ]**	
        
        **修改print**：print( ' , ' .join(buf)  )		现在打印的就是列表了

    规定一个模板，然后你传输参数进去，python会照着模板输出

    ```python
    class car():
        def __init__(self,make,model,year)
        	self.make=make
        	self.model=model	#等等等
        def name(self)
        	all=self.make+' '+self.model
            return all.title()
        def update(self,year,model)
        
    mycar=car('bmw','x5','2020')
    print(mycar.name)
    #修改的话  self.make='laosilaisi'  直接重新赋值就好了
    #想传参数
    mycar.update(2021,laosilaisi)
    #继承---就是承接上面的类比如car
    class electricCar(要继承类的名字car):
    	def __init__(self,make,model,year，money) 
        #这里重写了父类方法，同时要有父类和子类的方法(父在前)
    	#如果子类重写了父类的方法，那么系统不再默认调用，所以使用super手动调用
        	super().__init__(make,model,year)  #super是调用父类的函数，里面写父类的参数
            self.money='100w'
         
        def newname():
            print('the car price is'+str(self.money)+)
     #__str__重写后其实不用调用，之间复制之后如何加上你需要输出的
           
    newcar=electricCar('benchi','x7','2021')	#创建对象
    newcar.name()				#调用方法，当然如果有多层类，可以调用父类、爷爷类，方法随便调用
    ```

    如果前面已经有一个类了，现在这个类传上一个类的对象可以获取上一个类的信息：形参.上一个类的属性

    比如：newobject.area  

    

    在子类中调用父类方法：

    1. 父类名.子类名( self,其他参数)	Dog.bark( self)    必须带上self

    2. super（当前类，self）.方法名（参数）  super(XTQ,self).bark( ) 

    3. super().方法名         例子： super( ).bark( )

       调用指定父类中的方法：

       1. super（类A，self）.方法名（参数）类A的父类中的方法
       2. 类名.方法名（self，参数）

    **在类中改变列表的输出方式列表推导式的运用**：buf_list=[x.name for x   in  furniture_list ]

22. 多继承：子类可以同时继承多个类  class  子类名（父类名，父类名，...）因此下面调用的时候可以调用多个类                     的方法

    ​			如果两个父类都存在同名的方法，子类对象调用的是**第一个父类**的方法

23. _ _ mor_  _:会显示继承的顺序       print(继承的类名._  _ _ mro _ _ )

24. 私有权限：意思是：在什么地方可以使用和操作

    定义：在方法和属性前加上两个下划线_ _，就变为私有

    作用：

    1. 不能再类外部通过对象直接访问和使用，只能在类内部访问和所有
    2. 不能被子类继承

    共有权限：不是私有就是共有

    ```python
    class people(object):
        def __init__(self):
    		self.__ICBCmoney=money
            #这样定义后，创建对象后也无法调用
    #那怎么样取查看和添加呢
    创建对象.__dict__  就可以查看对象的属性信息，类型是字典，字典的key是属性名，value是属性值
    对象名.__dict__ICBCmoney=1000   这样就可以重新添加一个共有属性可以print了
    ```

      打印类对象：print(Dog._ _ dict _ _ ) 

    _ _ dict _ _ 也是魔法方法，可以以**字典的形式打印对象的属性**

    ​	在类中且在类中方法外，每一个实例对象都会显示但不一定相同

25. 静态方法：没有实例对象也没有类对象 

    ​					既不需要访问实例属性或实例方法，也不访问类属性或类方法

    如何定义呢？在def 的上面一行 加 标识符  **@staticmethod**  方法里也就不需要self这个默认参数了

    而且不需要创建对象，类名.静态方法名（） 就可以调用了

    如果要输出静态类属性 需要使用 **@classmethod**   def 方法名（默认参数cls）

26. 多态：在多个类中定义一个公共方法，然后在里面调用父类方法，创建多态的对象就好了

    ```python
    class Dog(object):
        def __init__(self,name)
        	self.name=name
        def play(self):
            print(f'{self.name} is playing')
    class XTQ( Dog):
        def play(self):
    		print(f'{self.name} is following crown')
    #公共方法---多态
    def play_with_dog(obj_dog):
        obj_dog.play()
    xtq=XTQ('xiaohei')
    play_with_dog(Dog/XTQ)#调用父类
    ```

    

27. 涉及到多个文件时  可使用	**from 模块名 import 类名（也可有多个类名）**  	直接在另一个文件中打开并使用前面的类，只需要重新赋予类的形参值就可以了

    import  模块名 	可以导入整个模块
    
28. eval函数：将字符串当成有效的表达式，来求值，并返回计算结果

    1. 基本的数学运算：eval(‘ 1+1’)		#2

    2. 打印重复的字符串 eval(‘hello’*10 ’)  #10个hello

    3. 将字符串转换成列表：eval(‘[1,2,3,4,5]’)  #[1,2,3,4,5]

    4. 将字符串转换成字典: eval(‘{键值对}’)  #{键值对}

       简单来说就是把引号给去掉，返回原来的类型

### 二、文件

1. 读取文件

   with 关键字  在不再需要访问文件后自动将其关闭

   open函数  **open(file，mode，encoding)**（打开的文件名，文件打开方式，**文件的编码格式gbk和utf-8）**

   f=open('1.txt','r')	buf=f.read()读文件

   ```python
   with open('xxx.txt') as file_object:  #记得加冒号
   	contents=file_object.read
       print(content.rstrip())
   ```

   文件对象名.write()	f=open('1.txt','w')以w的方式打开如果文件存在会清空原来的文件，如果不存在就创建					f.write('hello,world')		写进文件

   **如果写f.write('你好，中国')会乱码，是因为没有规定文件的编码，window系统默认是gbk格式，而pycharm默认是utf-8格式**

   **解决：f=open（'1.txt','w',encoding='utf-8'）**

   文件对象名.read() 读取这个文件的全部内容 	但是read函数在末尾会返回一个空行我们要删除这个空行

   read（括号里有数字，代表一次读取的字节数)

   read()读到文件末尾会返回空  （读大文件的时候）

   ```python
   f=open('class','r',encoding='utf-8')
   while Ture:
       buf=f.readline() #一次只读一行，保证文件读取有问题
       if buf:
           print(buf,end=' ')  #避免换行
       else:
           break
   ```

2. 以二进制的方式打开文件：能打开文本文件也能打开二进制文件，但二进制文件只能二进制打开

   ​	什么是二进制文件：具有特殊格式的文件，音频，视频，图片等

   ​	打开格式 rb，wb，ab但是open函数不能规定endocing

   ```python
   f=open('class','wb')	#不能用endocing
   f.write('你好'.encode())	#将字符串转换为二进制字符串
   #如果是读中文的字符串
   buf=f.read()
   print(buf.decode()) #这样就可以读中文了,二进制转字符串
   ```

3. 文件备份：filename=input（“   ”）

   ​					f=open(filename,'rb')	buf=f.read()	f.close()	index=filename.rfind('.')

   ​					new_filename=filename[:index]+'[备份]'+filename[index:]

4. 文件路径

   filepath='文件的位置'		with open(filename) as file_object:

5. 逐行读取：

   ```
   file_name='xxx.txt'
   with open(file_name) as file_object：
   	for line in file_object
   	print(line.rstrip)						#记得删除空行
   
   ```

6. 创建一个包含文件的列表

   ```
   file_name='xxx.txt'
   with open(file_name) as file_object：
   	lines=file_object.readlines()	#readlines将读取逐行读取然后存到lines列表中
   for line in lines:
   	print(line.rstrip())	rstrip删除的是读取的字符串的前后的空格，避免读取出来有换行符
   ```

7. 文件和目录的操作：

   ```python
   #文件重命名
   import os
   os.rename('原文件名','新文件名')
   #删除文件
   import os
   os.remove('要删除的文件名')
   #创建文件夹
   import os
   os.mdkir('创建的文件夹的名字')
   #删除文件夹
   import os
   os.rmkdir('要删除的文件夹名字')	#只能删除空目录
   #切换目录
   os.chdir('目录名字')
   os.chdir('../')	#返回上一级目录
   变量=os.listdir()	#获取文件名列表
   ```

8. 读取文件的操作：

   ```python
   f=open('students.txt','r',encoding='utf-8')
   buf_list=f.readlines()	#返回的是列表
   for buf in buf_list:
       buf=buf.strip()
       info_list=buf.split(',')  #返回的还是一个列表
       #两种方法提取读取的文件的信息
       #方法一
       stu=Student.Student(info_list[0],info_list[1],依次类推后面的信息但是要和类的参数对应)
       stu=Student.Student(*info_list)#*info_list就是将列表拆包这个就是等效替代前面的
       #方法二
       a,b,c,d=buf.strip().split(',')#拆包读取文件中需要的某个信息
       这个如果文件数据过多可能会报错，只能有list[]来指定
   ```

9. 获取当前工作目录的路径：

   1. os.gercwd( ) 和os.path.abspath(‘ . ’)  获取当前工作目录的绝对路径
   2.  os.path.join(‘ 当前工作目录的路径’，‘要创建的新目录’)
   3. os.mkdir(路径+新建目录名字)，rmkdir是删除
   4. os.path.split(os.path.realpath(当前文件名 ) )将文件名与扩展名分开
   5. os.path.splittext()直接得到文件的扩展名

   #### 正则表达式：

   字符匹配：

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

   19. 子模式

       | 语法         | 功能描述                             |
   | ------------ | ------------------------------------ |
       | （?P<命名>） | 为子模式命名                         |
   | （？i）      | 后面跟的就是匹配标志，比如（？i）n\b |
       | （？=）      | 用在正则表达式之后，出现则匹配       |
   | （？！x）    | 用在正则表达式之后，不出现x则匹配    |
   
     
   
20. ()括号可以作为一个分组 ，括号括起来的内容可以作为中间变量记录下来，要想记录下来并使用把后面的那一部分也括起来然后写  （\1）数字就是第几个括号
   
    括号一多容易混，所以用信息加以定义 (？p=< 定义名字>)   例子：(？p=< key1>)
   
    **首先导入模块re**
   
    校验数字：
   
    ![1611386525311](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1611386525311.png)
   
    校验字符的表达式：
   
    ![1611386578655](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1611386578655.png)
   
       
   
       ![1611386615056](C:\Users\银晗\AppData\Roaming\Typora\typora-user-images\1611386615056.png)
   
       re模块的方法：
   
       1. 循环所有匹配：re.findall()
      
          res=re.findall(r‘ r[ua]n’, ‘run ran ren’ )   print(res) 输出run 和ran 
      
          注意如果要找单独的单词，请在要匹配的单词首尾各空一个空格
      
       2. re.match(正则表达式，要匹配的字符串)   result=re.match()   从左到右匹配字符串，只返回匹配到的
      
          如果match匹配到数据用result.group() 返回数据
      
       3. 替换匹配内容：re.sub()
      
          re.sub( 待替换的字符串,要替换上的字符串，原来整个字符串)
      
          过滤掉网页中那些不需要的符号：re.sub( r‘<.+?>’,  ,s)或者(r’</? \w+ >‘ )
      
          实例操作：re.sub(r “ http://.+?/)”,lambda x: x.group(1),s)  用匿名函数输出来替换
      
       4. re.search(r’‘  搜索内容’, ‘匹配内容 ’)  :与match不同终端无法继续
      
       5. 分裂内容：re.split()
      
          res=re.split(r ‘ ,;\.\ \（分裂标志）’  ,  ‘a,b;c.d\e’)   就是把，；. \之间的值全部分裂开输出a,b,c,d,e
      
          返回的是列表 
      
       6. 包装正则表达式：re.compile(匹配表达式)   
      
          compile_re=re.complie(r’ r[ua]n ’)    **compile是创建一个正则表达式的对象，后面能多次调用**
      
          res=compile_re.findall(‘ run ran ren’
      
       7. 贪婪模式：只有不设置限制，系统默认一直往后找 
      
          用括号括起要关闭贪婪模式的表达式 加上一个？

### 模块：

1. 已经写好的模块调用就好了，比如random：已经很熟悉了

2. 你想调用自己写的模块：首先你写好一个文件，里面有写好的函数

   import 要调用的模块文件名

   调用：模块名.函数名（）		**#函数是模块里有的**

3. 调用模块有红线提示：  project目录，找到该目录-->右键-->Mark Direction as-->Sources Root

4. _ _ all _ _变量：可以在每一个代码文件中（模块中）定义，类型是**元组或列表**

   作用：影响form 模块名 import * 导入行为，另外两种import 模块名 和 form 模块名 import  工具名也就是模块里的函数名

   如果没有定义all变量，模块中所有变量都可以导入；如果定义all变量只能导入变量中定义的变量

   原文件模块(my_moudule)：

   ​	**_ _ all _ _=[‘ ’,‘ ’]   列表里写导入要文件模块里的所有变量**

   现文件：

   ​		**from  my_moudule import **（单独使用没有all，就是导入所有的工具）

5. 注意自己定义的模块名不要与系统定义的模块名一样

6. 包：包里面包含多个文件，项目-->右键package就好了  包里有多个模块一次性导入便于使用

   先在init文件中 from . import 模块名 然后在其他文件中才能直接导入模块

   

### 计算机基础：

1. 算法：在计算机根据代码进行运算时，运算时间比运算次数为运算效率，运算效率却决于编写的代码，当计算机所需要计算的数据非常庞大时，必然要提高运算效率，因此不同的算法应运而生，算法越好，计算机运算效率越高。比如谷歌公司每天要处理上亿的数据，算法的效率是非常关键，好的算法可以数十倍的提高计算机运行的时间

    

   

2. 数据结构：就是数据有序的储存在结构化的、方便读取的容器里，比如数组，矩阵，结构体，链表，用下标index可以非常方便快捷的查找和读取数据。而链表则是更高级处理数据的方法，链表拥有节点，可以动态的增减数组大小也可以对数组经行排序，有树形的分支链表就进行逻辑的运算因此诞生了很多高效的排序算法和数据处理的方法

    

3. 图灵机：用于解决可判定的问题，要解决一个问题到底有没有算法解决，图灵机可以用来判断能计算这个问题能不能解决，用纸带记录二进制数字，无论多么复杂的计算都可以判断出是否可行，只是效率的高低而已，因而图灵机成为了通用计算机的模型

    

4. 集成电路：计算机内部最开始时电子管，随后人们发现了更好的晶体管，速度更快价格更便宜，晶体管任然是分立结构，最后科学家将多个独立的部件用电线连接起来，拼装出计算机，或者将多个组件包在一起，变成一个新的独立组件，这就是集成电路，当时发现的半导体材料完美的符合集成电路所需的要求。最终将PCB和IC结合就变成了很小一块的集成电路板大幅减少独立组件和电线，更小，更快捷，更可靠。随后，光刻技术发展，可以轻松地将很复杂的电路刻到晶体管上，这样就可以实现非常复杂的电路也能方法非常小的晶体管上

   

5. 数据储存与文件系统：计算机拥有大量的数据，如何区储存呢？一开始用的纸带，纸卡区储存一些固定的数据，但是如果储存一次性的临时数据这会非常浪费，随后运用磁带、光盘等微型数据储存技术，储存器和寄存器诞生了。储存数据，读取数据变成了非常简便的操作。随后文件系统发展起来，能给更好的归类处理数据，运用物理方法，使得可以瞬间记录成千上万的数据，有了文件我们就可以储存各种数据，图像，音频等，文件有目录显示文件的内容、大小、欸之，文件后缀名，加以系统对文件进行管理就形成了文件管理系统，就非常方便查找储存的文件，而且储存的文件可以增删改删。

6. 命令行界面、屏幕、图形界面：就是人通过键盘将数据或者命令输入到计算机，然后计算机将数据返回来，这种人机的交互方式叫做命令行界面，因此人可以对计算机进行各种命令，获得计算机等等返回值，但是命令行界面是不可见的，随后屏幕应运而生，屏幕会将返回的信息输出到屏幕上，人就可以读取屏幕的返回信息，而且有图形界面将信息显示出来，以后人们就是可以玩计算机了，键盘可以输入你命令和输入的信息，计算机内部进行运算，然后计算机运算的结果通过屏幕和图形界面显示出来，逐渐的游戏就开始发展了起来，通过命令行界面对计算机进行操作再通过屏幕就显示出来，这样玩家就可以操纵游戏啦----

###  											Tips

产生随机数

先import 导入随机数模块

random.randint(起点数字，终止数字)		number=random.randint(1,100) 	print(number)注意一次只能输出一个

print格式

1. python3.6以后可用{}去占位，填充的数据直接写在大括号里面  ,但是引号前面要加一个f 

   print(f'我的年龄死{age}，我的名字{name}')		这样简洁明了

   **两个print之间默认换行可以用end=''来去掉		print('hello',end=' ')  print('world')   输出  hello world（不换行而是空格隔开）	end=后面可以是none或者string**

   ##### 三目运算：

   变量=表达式1 if 判断条件 else 表达式2	：判断条件成立执行表达式1；判断条件不成立执行表达式

   for+else结构：for循环结束后运行else

   ```python
   for i in str:
   	if i<10:
   		print(" ")
   else:
   	print(' ')
   ```

2. .get()  查找字典内的元素

   print（字典名.get('元素')）有对应的value值就返回，没有就返回None

3. 遍历字典：  

   ```python
   for i in dict:
       print(i,dict[i])	#这样不仅可以遍历key值还可以遍历value值
   ```

   也可以这样：**字典.keys()**  获取字典的key值	result=dict.keys()  同理：**字典.value()**  作用就是可以读取key和value值将其转换为list类型

   **字典.items()**	获取所有的键值对

4. 函数要想返回多个值，就将值存在容器里返回，然后再对容器进行操作

   ```python
   def add(a,b)
   	c=a+b
   	d=a-b
   	return ['c','d']
   result=add(10,12)	也可以直接return c,d python默认c，d组成元组进行返回
   c=result[0];d=result[1]
   ```

   5.不定长形参：

   ```python
   del func(*args,**kwargs)#形参前面加一个*，该形参变为不定长元组形参，类型是元组
   #形参加两个**，该形参变为不定长字典形参，可以接收所有的关键字实参，类型是字典
   （a,b,c）是位置参数	（a=1,b=2,c=3）是关键字参数
   ```

   **这样可以直接将传上去的形参直接变成元组和字典**

   **先写位置形参，再写不定长元组形参，再写缺省形参，最后写不定长字典形参**

5. 组包---将多个数据值给一个变量并组成元组

   a=1,2,3	print(a)  输出：（1，2，3）

   拆包---将容器的数据分给多个变量   注意：数据的个数和变量的个数要保持一致

   b,c,d=a   print(b,c,d)	输出：1 2 3

   拆列表，拆字典输出的是key值

6. 交换两个数字的方法：

   方法一：t=a;a=b;b=t	方法二：a,b=b,a直接交换（组包和拆包）

7. extend()方法：在一个列表的末尾一次性加上另一个列表的多个值

   city=['a','b']	side=['c','d']  	city.extend(side)	 print(city) 	输出:[a,b,c,d]

8. 匿名函数：定义的函数没有名字

   匿名函数中不能使用流程控制语句，只能写单行的表达式，不需要return

   语法：lambda 参数列表 ：表达式   （可以返回这个表达式的值）

   ​			参数可以是不定长参数*args和**kwargs，可以返回列表和字典

   ```python
   无参无返回值：
       def 函数名（）：
           函数代码
           f1=(可以给匿名函数命名，以免混了)	lambda:函数代码
   无参有返回值：
   	def 函数名():
           return 1+2
       lambda:1+2
   有参无返回值：
   	def	 函数名(a,b):
           print(a,b)
       lambda a,b:print(a,b)
   有参有返回值：
   	def 函数名(a,b):
           return a+b
       lambda a,b:a+b
   ```

   **匿名函数要写在对应函数的下面，函数一多可以命名后再调用**

   def func(name):

   ​		print(name)

   f2=lambda name: print(name)

   调用：f3(hi)

   先将函数作为参数：

   ```python
   def hanshu(a,b,func):
       num=func(a,b)
       print(num)
   
   def add(a,b):	要想使用匿名函数，必须将函数作为形参 再调用函数后可以用lambda简单实现传递参数
       return (a+b)
   hanshu(10,20,add)
   hanshu(10,20,lambda a,b:a-b)	#输出 -10  
   hanshu(10,20,lambda a,b:a*b)	类似
   ```

9. 列表排序：如果你的列表由字典组成，你可以由字典的元素排序

   list1={'name':'han','age':18

   ​			'name':'yin','age':19}	list**.sort(key=lambda** x: x['name'或者'age'])   就会按照name或者age字母顺序排序

   sort(key=一个函数)   根据列表中元素字符串的长度进行排序：list.sort(key=len)或者	list.sort(key=lambda x :len[x])

   **列表名.sort(key=lambda 形参：（排序规则1，排序规则2，。。。）,如果想倒序的话，reverse=True)**

   list.sort(key=lambda x:(x['age'],x['name']),reverse=True)

10. 列表推导式：

    **变量=[数据生成规则 for 临时变量 in 列表名]		每循环一次就会创建一个数据**

    **数据生成规则的意思是  每循环一次生成一个这样的数据**： 

    list=[i for i  in  range(5)]  输出1-4的数字列表

    list=['hello' for i  in  range(5)]	就会生成5个hello的列表

    list=[f'num={i}' for i  in list]	规则想怎么写就怎么写 

    list=[i+i for i  in list]		规则也可以是表达式or算式

    **变量=[数据生成规则 for 临时变量 in 列表名 if  条件]	：每循环一次并且满足if条件，生成一个数据**

    list=[i for i  in  range(5)  if  i % 2 == 0]

    **变量=[数据生成规则 for 临时变量 in 列表名  for j in 列表名]	：for嵌套循环，第二个for每循环一次生成**

    list=[(i,j) for i  in  range(3) for j in range(5)]  输出:[(1,0),(1,1)......]

11. 字典推导式：dict={i: i**2   for i in range(1,5) }  print(dict)  #key是i，value是i的乘方

    ​				和列表推导式类似

    提取字典里的目标数据：count={key: value ,  for key,value in counts.items()  **if value >=200** }

12. 高阶函数：

    1. abs()函数  完成对数字取绝对值	abs(-10) 输出：10

    2. sound()函数：既可以四舍五入也可以保留小数 round（1.2）=1.2   round（a,2）变量a保留两位小数

    3. map()函数：map（func，list）将传入的函数func作用到list变量的每个元素中，并将结果组成新的列表

       ```python
       list1=[1,2,3]
       def func(x):
           return x**2
       result=map(func,list1)
       prinr(list(result) #[1,4,9]
       ```

       

    4. reduce()函数：reduce(func,list)   **其中func必须有两个参数。**每次func的计算结果和序列的下一个元素做累计计算   累计计算取决与函数的计算功能

       ```python
       inport funtools
       list1=[1,2,3,4,5]
       def func(a,b)
       	return a+b
       result=functools.reduce(func,list1)
       #实现累加
       ```

    5. filter()函数：fliter（func，list1）  func函数用来过滤掉不符合条件的元素，返回一个filter对象，如果要转换为列表，可以用list()转换为列表

    ```python
    list1=[1,2,4,5,6]
    def func():
    	return x%2==0
    result=filter(func,list1)	 #过滤掉不是偶数的数
    print(list(result))
    ```

    **通过value找key：**

     student = {'小萌': '1001', '小智': '1002', '小强': '1003', '小明': '1004'} 

    法1：list (student.keys()) [list (student.values()).index ('1004')]

    结果显示： '小明'

     法2：def get_key (dict, value):

    ​        return [k for k, v in dict.items() if v == value]

    \>>> get_key (student, '1002')

    结果显示：'小智'

    

    

    

    

    

    

    