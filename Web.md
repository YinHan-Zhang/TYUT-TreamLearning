# 						Web开发

### 开始操作：

1. 打开终端：输入 python manage.py startapp  名字

2. 去到djangoproject里，进入setting，找到INSTALLED_APP，添加上新建文件的名字

3. 格式不对，红色报错  鼠标放在红线上按 Alt+Enter

4. 想打开网页渲染，就要创建templates文件夹，右键mark  directory  as  --templates folder 

5. 重构式重命名：shift +F6

6. 创建级联关系时：表内定义名=models.ForeignKey(关联表名,on_delete=models.CASCADE)

7. 创建项目时，配置解释器：file-->setting-->Project-->project interpreter

8. 配置数据库：

   ‘ENGINE’: ‘django.db.backends.mysql ’ ,

   ‘NAME’:	‘数据库的名字不是项目的名字’

   ‘USER’:	‘root’

   ‘PASSWORD’:	密码

   ‘HOST’:localhost

   ‘PORT’:3306

9. 在创建子路由后记得在跟路由里注册: 例：**path(‘app/’,include(‘APP(app的名字).urls’)),**

### 基本知识操作：

_ _ init _ _:

admin:

apps:记得要在setting App中注册名字

models:

- 使用了对象关系映射，将业务逻辑进行一个解耦合
- 关系型数据库，DDL，通过models定义实现，数据表的定义

先写class，然后终端输入两个命令 python manage.py makemigrations和python manage.py migrate

database里就与表了

tests:

views（视图函数）:	这里面写def函数  			第一个参数必须是request，必须返回一个response

​			from django.http  import  HttpResponse

​	def hello(reqeust):

​		return HttpResponse(‘ 双击666 ’)

​	def 函数名（request）:
​		return  HttpResponse(‘输出内容’)	或者 return render（渲染）（request，‘内容’，‘html文件’）

settings:
uris（根路由）:

1. 注册一个路由 

     	-url  		

   ​			-参数1  匹配规则   正则表达式-视图函数		

   ​			-对应的是view中的一个函数，没有括号

   ​	-去view实现对应的视图函数

   ​			-第一个参数是request，永远记得返回response

2. 例：  from APP import views

   urlpatterns=[

   url(r‘^admin/ ’ , admin.site.urls),

   url(r ‘^hello/ ’ ,views.hello),

   url(r ‘正则匹配规则  函数名/ ’ ，调用视图函数 ---views.函数名  )]

3. 如果创建多个文件夹，要创建子路由，这时候要在根路由里敲一个path('two/',include('Two.urls'))   在网页上敲  two/index/才能进入

wsgis:

### 文件操作：

1. 创建表格：先在model里面定义表格，然后去终端输入命令 python manage.py migrations，再 python manage.py migrate 就好了
2. HTML操作：ul>li*5 +Tab键   就会自动生成
3. object： 保存：save（）  查询：get(pk=?)     显示所有：all（）     filter（筛选条件）

### Model： 

##### 模型，表，属性，字段

1. 字段类型：**AutoField**--根据id自动增长的IntegerField    **CharField(max length=字符长度**)--字符串

   **TextField**--大文本字段，一般超过4000使用     **DecimalField**(max_digits（小数点位数总数）=None,decimal_places（小数点后的数字位数）=None):十进制浮点数    **IntegerField**--整数

   **FloatField**--浮点数     	**DateField**([auto_now=Flase,auto_now_add=False])   **FileField**=文件字段

   字段名称及意义：

   null：NULL空值     blank：允许为空白   	db_column:如果未指定，则使用属性的名字

   db_index：若值为Ture，则在表中为此字段创建索引		default：默认值

   primary_key：主键字段    unique：表中要有唯一值

   关系：ForeignKey:一对多，将字段定义在多的端中    ManyToManyField：多对多，将字段定义在两端中

   ​			OneToOnefield：一对一，将字段定义在任意一端中

2. 创建模型类：元信息，定义类

   class Person（models.Model）：

   ​	p_name=models.CharField(max_length=16,unique=True)

   ​	p_age=models.IntegerField(default=18,**db_column(改名)**=‘age’)

   ​	p_sex=models.BooleanField(default=False,db_column=‘sex’)

   ​	class Meta:

   ​		db_table=‘People’

3. 模型过滤：

   filter（筛选进来）：	gt：大于   lt：小于

   exclude（除去）： 可以二者一起用

##### views:

1. ```python
   #先创建对象person=Person()
   def  get_person(request):
   	persons=Person.objects.exclude(age__lt=50.filter(age__lt=80))
       context={
           'persons':person
       }
      	return render(request,'person_list.html',context=context)
   def add_person(request):
       person=Person.objects.create(name='sunck',age=15,sex=True)
   ```

   

















 