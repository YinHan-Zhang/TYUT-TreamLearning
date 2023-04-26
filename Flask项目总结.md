# 博客网站功能概览：



![1642377091806](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1642377091806.png)



```html
 <!--三层模拟<table><tr><td>：第一层 必须指定container属性，第二层必须指定row属性， ；-->
    <!--栅格系统，会将网页分为12等份，设置最外层div的宽度为1200px，每一份就是100px-->
    <div class="container">
        <div class="row">
            <div class="border" :solid 5px blue; width :30%></div>  
            <div class="border" :solid 5px blue; width :40%></div>
            <div class="border" :solid 5px blue; width :30%></div>
			这里应该是一行三列，宽度343的比例，但是我写错了，网页显示是3行
			原因是引号的问题！！！,class后面的双引号，应该包括全部的描述
			<div class="border :solid 5px blue; width :30% "></div>
        </div>
    </div>
```





| 列宽度   | 适配屏幕大小                                          |
| -------- | ----------------------------------------------------- |
| col-xl-3 | 超大型屏幕（大于1200像素）的浏览器窗口显示            |
| col-lg-3 | 大型屏幕（大于992像素）的浏览器窗口显示               |
| col-md-6 | 中型屏幕（大于768像素）的浏览器窗口显示               |
| col-sm-6 | 小型屏幕（大于576像素）上显示为6列宽度                |
| col-12   | 超小屏幕（小于576像素）上显示为12列宽度，与手机端适配 |



```html
<div class="row">
{#            这里太繁琐了，要么电脑端要么手机端，设置两种就好了，不用纠结中间的样式#}
{#            这里是设置了超大屏幕，大，中，小，超小屏幕的显示效果#}
            <div class="col-xl-3 col-lg-3 col-md-6 col-sm-6 col-12 d-none d-sm-block" >第					一行第一列</div>
            
            <!-- d-none正常是隐藏的 d-sm-block 在sm上显示 -->
            
            <div class="col-xl-3 col-lg-3 col-md-6 col-sm-6 col-12 d-none d-sm-flex">第一						行第二列</div>
            <div class="col-xl-3 col-lg-3 col-md-6 col-sm-6 col-12"></div>
            <div class="col-xl-3 col-lg-3 col-md-6 col-sm-6 col-12"></div>
```



## 前端页面开发的步骤：

1.UI设计

2.绘制线框图

3.使用HTML实现线框图

4.填充页面内容

5.对内容进行微调



## 数据库设计

* 用户表

  user_id

  username

  password

  nickname

  avatar:头像

  qq

  role : 用户、编辑、管理员

  credit

  createtime

  updatetime

* 文章表

  article_id

  user_id

  type

  headline

  content

  thumbnail:缩略图文件名

  credit

  readcount ：阅读数

  replycount：回复数

  recommended：被推荐数

  hidden：收藏数

  draft：是否是草稿

  checked：审核

  createtime

  updatetime

* 积分表

  credit_id:积分表唯一的编号

  user_id

  type:积分对应的类别，阅读文章扣积分，评论文章加积分，充值多少加10*多少分

  target：积分消耗对应的目标，该ID阅读和评论文章，对应积分增减

  credit

  createtime

  updatetime

* 收藏表

  favorite_id

  article_id：表外键，关联文章表

  user_id：表外键，关联用户表

  cancleed :是否被取消收藏

  createtime

  updatetime

  

* 评论表

  comment_id

  user_id

  article_id

  content

  ip_addr

  reply_id: 评论是否为回复，是原始的回复还是被回复的评论

  agree_count

  oppose_count

  hidden

## Flask路由参数:

REST： 表述性状态转移 ，非常抽象的概念。REST设计的目的，就是在符合架构原理的前提下，理解和评估以网络为基础的应用软件的架构设计，得到一个功能强大的架构。 

传参的两种方式

1.URL地址自带查询参数:

在网页路由中输入 http: // 127.0.0.1: 5000 / test?username = xxx & password =xxxx

然后,使用  ```request.args.get() ```方法获取参数

```
@app.route('/test')
def test():
    username = request.args.get('username')
    password = request.args.get('password')
    
    return f'你的用户名为{username},密码为：{password}'


```

2.Flask自定义的路由规则和参数

在路由地址中使用<数据类型: 参数名>   ---> 网页路由中输入 http: // 127.0.0.1: 5000 / article / 123

使用``` request.form.get()```方法获取参数

```
@app.route('/article/<int:articleid>')
def article(articleid):
    
    return f'你正在访问编号为{articleid}的文章'
    
@app.route('/user/login',methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    if username == 'yinhan' and password == '123':
        return '登录成功'
    else:
        return '登录失败'
```

GET请求：是访问页面

POST请求：是从页面中拿数据

## ORM

Object-Relational Mapping : 对象关系映射。把数据转换成Python对象

表里面的表 ->Python的类

表里面的列->类的属性

表里面的行->类的实例，字典对象表述

字典对象的Key对应列，value对应值



```
user = User()
    print(User.__dict__) #获取类的属性和方法列表
    print(user.__class__) #通过实例对象获取对应的类名
    print(user.__class__.__dict__) #通过实例对象获取对应的类名,再获取类的方法属性列表
    user.__class__ #获取实例的变量
    user.__setattr__('nickname','yinhan')
    # 设置实例属性 == self.nickname = yinhan ,不同的是前者必须是字符串
    
    user.__getattribute__('usename') #可以拿属性，同样可以拿方法
    # user.__getattribute__(方法名)(方法参数列表)
    
```



 链式操作：返回当前类的实例，然后进行连续操作。就是调用完这个方法之后，还可以连续调用其他的方法

例子：user.chain().select().insert()





## **Session ** 和 **Cookie**

### Session

> 要处理Session，则必须为app实例设置SECRETE_KEY配置参数，配置随机数生成器(Session ID)，再使用Session函数进行操作。

 

**session** 获取： 

**@app.route('/sess')**

**def sess():**

  **session['username'] = 'wuhang'**

  **session['nickname'] = 'hangge'**

  **session['role'] = 'student'**

  **return 'OK'**

 

**session** 读取

**@app.route('/sc/read')**

**def scread():**

**return session.get('username')**

 

### Cookies

>  要处理Cookie，需要使用response对象来往HTTP的响应中写入满足HTTP协议的Cookie要求的信息
>
> （key，value，age）

 

**@app.route('/cookie')**

**def cookie():**

  **response = make_response('自定义cookie')**

  **response.set_cookie('username', 'hang', max_age=30)**

  **response.set_cookie('password', '000198', max_age=30)**

  **return response**

 

**@app.route('/sc/readc')**

**def screadc():**

**return request.cookies.get('password')**

 





## 数据库已创建的Model

```
class Users(db.Model):
    __table__ = Table('users',MetaData(bind=db.engine),autoload=True)

    def find_user_by_id(self,userid):
        row = db.session.query(Users).filter_by(userid=userid).first() 
        #可单表，也可多表联合查询
        return row

    def find_user_by_id(self, userid):
        row = Users.query(Users).filter_by(userid=userid).first() #单表查询
        return row
```

```
user = User()
result = user.field('userid,username,nickname').select(userid=2) 链式操作
print(result)

article = Article()
result=article.select(userid=1)
print(result)

if __name__ == '__main__':
    user = User()
    User.__dict__ #获取类的属性和方法列表
    user.__class__ #通过实例对象获取对应的类名
    user.__class__.__dict__ #通过实例对象获取对应的类名,再获取类的方法属性列表
    user.__class__ #获取实例的变量
    user.__setattr__('nickname','yinhan')
    # 设置实例属性 等价于 self.nickname = yinhan ,不同的是前者必须是字符串
    
    user.__getattribute__('usename') #可以拿属性，同样可以拿方法
    user.__getattribute__(方法名)(方法参数列表)

# return self 
	链式操作：返回当前类的实例，然后进行连续操作。就是调用完这个方法之后，还可以连续调用其他的方法
```



## 数据库的基本操作实现

自定义数据的查询、插入功能：

```python
import pymysql
from pymysql.cursors import DictCursor

class Mysql():
    def __init__(self):
        conn= pymysql.connect(host='localhost',port=3306,user='root',password='20020520zyh'
                              ,database='woniunote',autocommit=True)

        self.cursor = conn.cursor(DictCursor)

    def query(self,sql):
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        return result

    def execute(self,sql):
        try:
            self.cursor.execute(sql)
            return 'OK'
        except:
            return 'Fail'

# 写父类
class Model():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)  # 把v赋给k

    def field(self, columns):  # 传指定的列名
        self.columns = columns
        return self  # 返回self参数，就可以继续进行操作，

    def select(self, **where):
        table = self.__class__.__getattribute__(self, 'table_name')  # 获取子类的表名
        if hasattr(self, 'columns'): # 如果子类指定了列名
            sql = "select %s from %s" % (self.columns, table)
        else:
            sql = "select * from %s" % table
        if where is not None:
            sql += " where"
            for k, v in where.items():
                sql += " %s='%s' and " % (k, v)
            sql += " 1=1"  # 解决多了一个and的问题，1=1是一个永真无效语句
        # print(sql)
        result = Mysql().query(sql)
        return result

    def insert(self):
        keys = []
        values = []
        for k, v in self.__dict__.items():
            keys.append(k)
            values.append(str(v))
        # print(keys, values)
        sql = "insert into %s(%s) values ('%s') " % (self.table_name, ','.join(keys), "','".join(values))
        result = Mysql().execute(sql)
        print(result)

# 定义好父类，子类只用传表名和参数就行了，方法直接调用父类的

class User(Model):
    table_name = 'users'
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

class Article(Model):
    table_name = 'article'
    def __init__(self,**kwargs):
        super().__init__(**kwargs)


```



## 数据库连接与配置

```python
import os

HOSTNAME = '127.0.0.1'
PORT     = '3306'
DATABASE = 'woniunote'
USERNAME = 'root'
PASSWORD = '20020520zyh'
DB_URI = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(USERNAME,PASSWORD,HOSTNAME,PORT,DATABASE)
SQLALCHEMY_DATABASE_URI = DB_URI

SQLALCHEMY_TRACK_MODIFICATIONS = False

# 配置session
SECRET_KEY = os.urandom(24)
 # json数据中文
JSON_AS_ASCII = False

SQLALCHEMY_POOL_SIZE = 100

SQLALCHEMY_MAX_OVERFLOW = 100
SQLALCHEMY_TRACK_MODIFICATIONS = True
```



## SqlAchemy操作



```python
from sqlalchemy import create_engine, Table, MetaData, or_, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, session, scoped_session

engine = create_engine('mysql+pymysql://root:20020520zyh@localhost/woniunote',echo=False,pool_size=1000)

#定义集成的父类及数据连接会话
DBsession = sessionmaker(bind=engine)
dbsession = scoped_session(DBsession) #线程安全
Base = declarative_base()#父类
md = MetaData(bind=engine) #加载表的原数据

class Users(Base):
    __table__ = Table('users', md, autoload=True)

class Article(Base):
    __table__ = Table('article',md,autoload=True)

if __name__ == '__main__':
    result = dbsession.query(Users).filter(Users.userid == 1).all()  条件查询，逻辑运算
    result = dbsession.query(Users).filter_by(userid=1).all()  #字典键值查询
    # for row in result:
    #     print(row.userid,row.username)

    #查询时，直接指定列名，返回的是一个元组
    row = dbsession.query(Users.userid,Users.username).filter(Users.userid<5).first()
    # print(row)

    user = Users(username='hahhaha',password='123456',role='user',credit=50)
    # dbsession.add(user)
    # dbsession.commit()

    #修改
    row = dbsession.query(Users.userid, Users.username).filter(Users.userid==2).first()
    row.username = '章章呀！'
    
    #删除
    row = dbsession.query(Users.userid, Users.username).filter(Users.userid == 2).first().delete()
    dbsession.commit()



   

```

### 执行原生sql语句

```
    result = dbsession.execute("select * from users where userid==2").fetchall()
    dbsession.commit()
```

### 查询：

```python
基础查询
    result = dbsession.query(Users.userid,Users.username).all()
    print(result)
    且查询
    result = dbsession.query(Users).filter(Users.userid < 2, Users.username == '小章').all() 
    或查询
    result = dbsession.query(Users).filter(or_(Users.userid<2,Users.username=='小章')).all() 
    # for row in result:
    #     print(row.username,row.nickname)
	
	升序
    result = dbsession.query(Users).order_by(Users.userid).all()
    # for row in result:
    #     print(row.qq)
    降序
    result = dbsession.query(Users).order_by(Users.userid.desc()).all()
    # for row in result:
    #     print(row.qq)
    
	模糊查询
    result = dbsession.query(Users).filter(Users.username.like('%章%')).all() 
    # for row in result:
    #     print(row.username)
	
	分组查询
    result = dbsession.query(Users).group_by(Users.role).all()  
    result1 = dbsession.query(Users).group_by(Users.role).having(Users.userid>2).first()
    # for row  in result:
    #     print(row.username)

    联合查询 sum、min、max、avg、sum
    result = dbsession.query(func.sum(Users.credit)).first()
    # print(result)

    #多表联合查询,多表查询返回的结果集不是[Model,Model]的数据结构，而是每张表独立的对象来维护的，是一个对象元组
    # result = dbsession.query(Article,Users).join(Users,Article.userid == Users.userid).filter(Article.articleid==1).all()
    # for article,users in result:
    #     print(article.articleid,article.headline,users.username)
    
   #内连接查询一个用户的数据，看不到所有数据的全貌，外连接，可以查询所有用户的所有数据，主外键关系作为联合查询的条件

    result = dbsession.query(Users.userid,Users.nickname,func.sum(Article.readcount))\
    .outerjoin(Article,Users.userid == Article.userid).group_by(Users.userid).all()
    print(result)
```





## 已经创建好的表的操作

对于已经创建好的表，只用一条语句获取数据就ok了

* 核心：定义查询的子函数（成员函数）

```
class Users(db.Model):
    __table__ = Table('users',MetaData(bind=db.engine),autoload=True)

    def find_user_by_id(self,userid):
        row = db.session.query(Users).filter_by(userid=userid).first() 
        #可单表，也可多表联合查询
        return row

    def find_user_by_id(self, userid):
        row = Users.query(Users).filter_by(userid=userid).first() #单表查询
        return row
```



Flask框架本身是最小化Web服务内核，表与表之间的关系，不一定要定义在数据库中，心中有关系就行。（数据库为了维护主外键关系，会增加额外消耗）。

删除数据时，尽量使用软删除(设置标识)，而不是直接硬删除（Delete From）。索引重建。

.first()  直接返回一行数据对象

.all()  直接返回包含多条数据对象的列表

.filter_by(x=y, a=b) 只适用于等值查询，其参数为字典参数的传值方式

.filter() 适用于复杂查询条件的对比，其参数为条件运算

查询过程中，可以使用db.session.query的方式进行查询（支持多表），也可以使用Model.query的方式进行查询（不支持多表），所以优先使用db.session。

 

 

## JSON：

JavaScript Object Notation，是JavaScript的内置数据格式，由JavaScript的数组、对象构成。

[{},{},{}]
 {[], [], []}
 {{},{},{}}

[{[],[],[{},{},{{},{}}]},{},{}]

 [ [ (),(),...] ]

JavaScript的数组 ---> Python 列表是完全一致的定义方式

JavaScript的对象  Python的字典是完全一致的定义 {key:value, key:value}

 

JSON是目前在互联网上交换数据最流行的一种方式，但不是唯一的一种。

文本，CSV，XML，YML，只要JavaScript能够正常处理，即使实现前后端交互。

 

 

{'username': 'reader4@woniuxy.com', 'qq': None, 'updatetime': None, 'password': 'e10adc3949ba59abbe56e057f20f883e', 'role': 'user', 'createtime': None, 'userid': 8, 'nickname': None, 'credit': 5, 'avatar': None}

 



## truncate过滤器

1. Jinja2：truncate过滤器处理中英文混排的情况。

2. 利用JavaScript处理JSON的方式进行原生代码的前端渲染。

   a)    通过后台：原生Python输出HTML，使用模板引擎。浏览器直接绘制HTML.

   b)    通过前端：使用JavaScript动态填充DOM元素（JSON）。对搜索引擎不友好。前后端分离（Web App主流开发模式）à 核心思想：字符串拼接。

   c)    前后端分离可以有效减少服务器渲染HTML的资源消耗，把渲染的过程交给前端浏览器处理。

利用Vue进行前端渲染



## 项目结构概述:

1. 重新创建一个项目，命名为WoniuNote，并完全从零开始。

2. 确保数据库有少量测试数据，并且模拟真实的数据。

3. MVC：

   a)    Model层：根据业务需求，来封装数据库操作，用于操作相应数据。

   b)    Controller层：接收请求，处理业务，返回响应（HTML，JSON，其他）。

   c)    View层：由控制层发起模板页面的填充调用。

4. 所有的输入校验，一定要前后端同步进行。

 

5. 利用前端和AJAX来通过JS动态渲染侧边栏，而不使用Jinja2在后台渲染。本周末练习。（AJAX，JS，前端渲染，JSON）

## Ajax请求：

```python
user.route('/loginfo')
def loginfo():
    # 如果没有登陆，返回None
    if session.get('islogin') is None:
        return jsonify(None)
    else:
        dict = {}
        dict['islogin'] = session.get('islogin')
        dict['userid'] = session.get('userid')
        dict['username'] = session.get('username')
        dict['nickname'] = session.get('nickname')
        dict['role'] = session.get('role')
        return jsonify(dict)
```

从页面session获取数据，返回json格式数据

在前端页面，用js获取页面的响应，即获取到返回的json格式的数据

json格式就是字典与列表的嵌套，所以在js中，遍历获取json中的字符串数据



## 用户登录注册：

1.验证码：邮箱验证码，图片验证码，短信验证码

2.实现原理：

​		前端输入

​		--后台接收地址进而生成随机验证码（将验证码保存：Session临时保存，文件或者数据库永久保存，redis缓存服务器保存）

​		--返回给用户

​		--用户填写并提交到后台

​		--后台进行对比校验



### 验证码的作用：

1.防止自动化代码（爬虫、恶意脚本）来直接发送请求。

2.确认用户信息的真实性，邮箱地址、手机号码



图形验证码的功能实现：
	1.绘制基础的图片：使用pillow将随机验证码文本绘制成一张图片。字体、变形、干扰线。

​			* 生成随机字符串

​			* 对文字的基本处理				

​	2.将验证码图片响应给前端，供前端用户识别并提交给服务器

​	3.校验。成功，则执行后续代码，否则用户重新提交



导包:

```
import random
import string
import time
from datetime import datetime
from io import BytesIO

from PIL import Image, ImageFont, ImageDraw

from smtplib import SMTP_SSL
from email.mime.text import MIMEText
from email.header import Header
```



### 邮箱验证码:

```python
# 邮箱验证码
def send_email(receiver, ecode):
    url = request.path
    # 邮箱地址
    sender = 'Han-WoniuNote <1513032551@qq.com>'
    # 定义邮件内容， 以html格式定义，可以设置颜色，间距
    if url == '/user/reg':
        content = f"<br/>欢迎注册蜗牛笔记博客系统账号，您的邮箱验证码为:<span style='color: red;font-size: 20px'>{ecode}</span>," \
                  f"请复制到注册窗口完成注册，感谢您的支持！</br>"
    else:
        content = f"<br/>重置您的密码，邮箱验证码为:<span style='color: red;font-size: 20px'>{ecode}</span>," \
                  f"请复制到重置密码窗口完成注册，感谢您的支持！</br>"
    # 实例化对象
    message = MIMEText(content, 'html', 'utf-8')
    # 标题
    message['Subject'] = Header('蜗牛笔记验证码', 'utf-8')
    # 发件人
    message['From'] = sender
    # 收件人
    message['To'] = receiver

    # 与qq邮件服务器连接
    smtpObj = SMTP_SSL('smtp.qq.com')
    # 授权登录
    smtpObj.login('1513032551@qq.com', password='zcqjqmgpnczkhcfi')
    # 指定发件人，收件人， 内容
    smtpObj.sendmail(sender, receiver, str(message))
    smtpObj.quit()
```



### 绘制图片验证码

```python
class ImageCode:

    # 生成4位随机字符串
    def gen_text(self):
        code_list = random.sample(string.ascii_letters+string.digits, 4)
        code = ''.join(code_list)
        return code

    # 生成用于绘制字符串的随机颜色
    def rand_color(self):
        red = random.randint(12, 100)
        green = random.randint(88, 210)
        blue = random.randint(0, 200)
        return red, green, blue

    # 绘制干扰线
    def draw_lines(self, draw, num, width, height):
        for num in range(num):
            x1 = random.randint(0, width/2)
            y1 = random.randint(0, height/2)
            x2 = random.randint(0, width)
            y2 = random.randint(height/2, height)
            draw.line(((x1, y1), (x2, y2)), fill='black', width=2)

    # 绘制验证码图片
    def draw_verify_code(self):
        code = self.gen_text()
        width, height = 120, 60
        # 创建图片对象， 设定背景颜色
        im = Image.new('RGB', (width, height), 'white')
        # 选择字体以及字体大小
        size = random.randint(30, 50)
        font = ImageFont.truetype(font='arial.ttf', size=size)
        draw = ImageDraw.Draw(im)
        # 绘制字符串
        for i in range(4):
            draw.text((5 + random.randint(-3, 3) + 20*i, 5 + random.randint(-3, 3)),
                      text=code[i], fill=self.rand_color(), font=font)
        # 绘制随机1-5条干扰线干扰线
        self.draw_lines(draw=draw, num=random.randint(1, 5), width=width, height=height)
        return im, code

    # 生成图片验证码并返回给控制器
    def get_code(self):
        image, code = self.draw_verify_code()
        buf = BytesIO() #实例化字节码对象，将图片以字节码的形式保存到内存里面
        image.save(buf, 'jpeg')
        bstring = buf.getvalue() # 获取buf里面存储的字节码
        return code, bstring

```







## 自动登录：

 1.  利用Cookie的持久化存储来保存用户登录信息。

     ​	Session ID

     ​	利用加密机制，储存一个自定义规则的GUID

     ​	直接保存username和password(md5) 直接发送给服务器

     ​	1.在登录成功后，必须要将Cookie写入浏览器

     ​	2.在接口中，从Cookie中获取用户名和密码

自动登录的代码：

```python
#自动登录
    if session.get('islogin') is None:
        username = request.cookies.get('username')
        password = request.cookies.get('password')
        if username != None and password != None:
            user = Users()
            result = user.find_by_username(username)
            if len(result)>0 and result[0].password == password:
                session['islogin'] = 'true'
                session['userid'] = result[0].userid
                session['nickname'] = result[0].username
                session['role'] = result[0].role
```



## 拦截器

 1.  利用全局拦截器实现自动登录的过程

     用户在除了访问首页，在访问其他页面的时候不会自动登录，除非你在每个视图函数里面都写一段自动登录的代码--->全局拦截器  



```python
# 定义全局拦截器实现自动登录
@app.before_request
def before():
# 设置白名单，某些接口地址无需拦截
    url = request.path  #获取接口路由
    pass_list = ['/user/reg', '/user/login', '/logout']  
    #列表中储存的路由都是不经过拦截器的，注册、登录、注销的接口
    if url in pass_list or url.endswith('.js') or url.endswith('.css') or url.endswith('.png') or url.endswith('.jpg'):
        pass    #还有一些静态的资源，是不用自动登录的，js、css、png、
    # 其他的接口自动登录
    elif session.get('login') != 'true':
        username = request.cookies.get('username')
        password = request.cookies.get('password')
        if username is not None and password is not None:
            user = Users()
            result = user.find_by_username(username)
            if len(result) == 1 and result[0].password == password:
                session['islogin'] = 'true'
                session['userid'] = result[0].userid
                session['username'] = username
                session['nickname'] = result[0].nickname
                session['role'] = result[0].role
        else:
            #拦截需要权限的接口，判断是否具有权限
            deny_list = ['/readall', '/prepost', 'article', '/comment', '/reply', 'favorite']
            import re
            if url in deny_list or re.match(r'/comment/\d+$', url):
                return render_template('no-perm.html')
    else:#如果用户已经登录，则根据角色判断哪些接口可以使用
        role = session.get('role')
        # /article为新增文章或用户投稿接口，  /prepost为作者编辑发布文章接口
        if role != 'eidtor' and (url == '/article' or url == '/prepost'):
            return render_template('no-perm.html')
```



2.蓝图拦截器，在单独一个模块里面，发现用户没有登录

```python
# 局部拦截器，直接从session里面获取islogin判断
@comment.before_request
def before_comment():
    if session.get('islogin') is None or session.get('islogin') != 'true':
        return '你还没有登陆，不能发表评论！'  # 返回的这个提示字符串，在前端html中可以拿来做判断，页面从而给出响应

```





## onclick事件强制刷新：

```
<img src="/vcode" id="loginvcode" class="col-3" style="cursor:pointer;"
                                     onclick="this.src='/vcode?'+Math.random()">
                                     
在url后面加一个随机数，加这个参数并不是传参，目的是骗浏览器，如果不加数字，点击后缓存里有，但是无法更新，但是换一个地址后就可以从缓存里面拿新的
```







## User表和Comment关联查询的结果进行转换：

> 1.查询原始评论与对应用户评论对应的用户信息，带分页参数  find_comment_with_user()
>
> 2.查询回复评论，回复评论不需要分页 	find_reply_with_user()
>
> 3.根据原始评论和回复评论，生成一个关联列表  --  get_comment_user_list

根据一条评论的commentid和replyid两个数据去查，假设这条评论的commentid为44，那么去找回复着条评论的回复评论就是，在commenr表里面去找所有replyid等于44的回复评论

要转换一下格式：数据结构大概是这样的

[  {

原始评论一：对应用户一，

ReplyList[{回复评论一：回复用户一}，{回复评论二：回复用户二}，，{回复评论三：回复用户三}] 

}]



```python
 # 根据用户去查他发表的评论
    def find_comment_with_user(self, articleid, start, count):
        result = dbsession.query(Comment, Users).join(Users, Users.userid == Comment.userid) \
            .filter(Comment.articleid == articleid, Comment.hidden == 0, Comment.replyid == 0) \
            .order_by(Comment.commentid.desc()).limit(count).offset(start).all()
        return result
```



```
    def find_reply_with_user(self, replyid):
        result = dbsession.query(Comment, Users).join(Users, Users.userid == Comment.userid) \
            .filter(Comment.hidden == 0, Comment.replyid == replyid).all()
        return result
```



分页参数：start、count

```
    def get_comment_user_list(self, articleid, start, count):
        result = self.find_comment_with_user(articleid=articleid, start=start, count=count) #查到之后得到一个原始评论结果集
        comment_list = model_join_list(result) #原始评论连接的结果
        
        for comment in comment_list:
        	#获得查询结果集，
            result = self.find_reply_with_user(comment['commentid'])
            comment['reply_list'] = model_join_list(result)
        return comment_list
```





## JSON格式转换

```python
# 将两张表的结果集转化为[{},{}]
#查询到的结果集是这样的[(Comment,users),(Comment,users)] --进行转换
def model_join_list(result):
    list = []
    for obj1, obj2 in result:
        dict = {}
        for k1, v1 in obj1.__dict__.items():
            if not k1.startswith('_sa_instance_state'): #过滤掉内置字段
                if not k1 in dict:
                    if isinstance(v1, datetime):
                        v1 = v1.strftime('%Y-%m-%d %H:%M:%S')
                    dict[k1] = v1
        for k2, v2 in obj2.__dict__.items():
            if not k2.startswith('_sa_instance_state'):
                if not k2 in dict:
                    if isinstance(v2, datetime):
                        v2 = v2.strftime('%Y-%m-%d %H:%M:%S')
                    dict[k2] = v2
        list.append(dict)
    return list
```



## 点赞记录表

蜗牛笔记中，只允许用户对某条评论发表一次意见，并且只允许在赞成和反对之间二选一，后期不允许修改。由于匿名用户也可以点赞，但是匿名用户也可以点赞，但是匿名用户并没有userid，所以点赞表中可以记录userid为0来进行区分。

但是新的问题又产生了，一个登录用户可以通过记录userid和commentid来限制只能对一条评论评论点赞一次，但是匿名用户userid为0，显然无法进行有效限制。

* 所以，记录匿名用户的IP地址便产生了作用，通过IP地址和commentid字段也可以唯一的确定这个匿名用户是否点赞多次。

IP地址的获取：

```
 ipaddr = request.remote_addr
```

判断是否为匿名用户：

```
 if session.get('userid') is None: #判断是否为匿名用户
        userid = 0
```

如果是匿名用户，通过commentid和ipaddr去查：

```
# 匿名用户
        if session.get('userid') is None:
            # 如果是匿名用户通过评论的id和ip地址去判断是否已经点赞
            result = dbsession.query(Opinion).filter_by(commentid=commentid, 																		ipaddr=ipaddr).all()
            if len(result) > 0:
                is_checked = True
```



```
# 更新点赞反对
def update_agree_oppose(self, commentid, type):    
	row = dbsession.query(Comment).filter_by(commentid=commentid).first()    
	if type == '0':        
		row.opposecount = row.opposecount + 1    
	elif type == '1':        
		row.agreecount = row.agreecount + 1    
	dbsession.commit()
```



隐藏评论：

> 功能并不复杂，就是将原始评论的hidden字段修改为1. 
>
> 但是需要考虑的是，如果该条评论已经有了回复，是否接收隐藏？ 蜗牛笔记按照不接受隐藏的逻辑来实现。
>
> 前端页面实时更新点赞和隐藏评论的逻辑是一样的，如果点完赞或者要看见隐藏完之后的页面，数据库就要实现实时更新的操作，增加了数据库的查询压力。
>
> 所以，用户隐藏完评论后，在前端页面直接将隐藏评论删除，并且不再请求新的评论数据，也不刷新页面，这样来达到隐藏的目的

```
# 隐藏评论
    def hide_comment(self, commentid):
    	#如果评论已经有了回复，且回复未完全隐藏，且不接受隐藏操作
    	#返回"Fail" 表示不满足隐藏条件，隐藏成功返回Done
        result = dbsession.query(Comment).filter_by(replyid=commentid, hidden=0).all()
        if len(result) > 0:
            return 'Fail'
        else:
            row = dbsession.query(Comment).filter_by(commentid=commentid).first()
            row.hidden = 1
```

## 评论

添加评论:

```python
@comment.route('/comment', methods=['POST'])
def add():
    articleid = request.form.get('articleid')
    content = request.form.get('content')
    ipaddr = request.remote_addr

    # 对评论内容进行简单检验
    if len(content) < 5 or len(content) > 1000:
        return 'content-invalid'
    comment = Comment()
    if not comment.check_limit_per_5():
        try:
            comment.insert_comment(articleid=articleid, content=content, ipaddr=ipaddr)
            # 评论成功后，文章评论数，积分变动，用户积分发生变化
            Credit().insert_detail(type='添加评论', target=articleid, credit=2)
            Users().update_credit(2)
            Article().update_replycount(articleid=articleid)
            return 'add-pass'
        except:
            return 'add-fail'
    else:
        return 'add-limit'
```



回复评论:

```python
# 回复评论
@comment.route('/reply', methods=['POST'])
def reply():
    commentid = request.form.get('commentid')
    articleid = request.form.get('articleid')
    content = request.form.get('content')
    ipaddr = request.remote_addr  # 从网页获取用户的IP地址

    # 对评论内容进行简单检验，太少或者太多，判断为无效评论
    if len(content) < 5 or len(content) > 1000:
        return 'content-invalid'
    comment = Comment()
    if not comment.check_limit_per_5(): #单个用户只能评论5次
        try:
            comment.insert_reply(articleid=articleid, content=content, ipaddr=ipaddr, commentid=commentid)
            # 评论成功后，文章评论数，积分变动，用户积分发生变化
            Credit().insert_detail(type='回复评论', target=articleid, credit=2)
            Users().update_credit(2)
            Article().update_replycount(articleid=articleid)
            return 'reply-pass'
        except:
            return 'reply-fail'
    else:
        return 'reply-limit'

```



## 权限管理：

基于角色的权限管理：普通用户，管理员，作者，每个角色对应的权限-->也就是他能干嘛

通过什么实现呢？

两种方案：拦截器/单独设置接口

1.拦截器

​	将所有的蜗牛笔记中的后台接口列出来，然后利用Flask的拦截器进行权限的控制，这样就只需要在拦截器中对所有的接口进行判断。如果权限不正确，可以专门为此设计一个权限不足的页面，明确告诉用户这是非法操作

2.通过拦截器判断之后，在一些关键的接口层，再进行第二次判断，以确保权限的控制不会出错。



局部权限拦截器：

```
# 局部拦截器，直接从session里面获取islogin判断
@comment.before_request
def before_comment():
    if session.get('islogin') is None or session.get('islogin') != 'true':
        return '你还没有登陆，不能发表评论！'  
        # 返回的这个提示字符串，在前端html中可以拿来做判断，页面从而给出响应
```





## 文章编辑功能

UEditor插件： 富文本的Web在线编辑器

发布博客文章的时候，必然涉及图片的处理，在一篇HTML文章里面，图片的来源通常有两种渠道：直接复制一个在线图片的url地址或者由本地直接上传一张图片。

那么对于上传图片来说，如果没有服务器端的支持，是不可能上传成功的。

所以需要配置好服务器端，才能接受前端编辑器中上传的图片。



从参数地址中来：

> #request.args是从url地址中获取参数
> http://127.0.0.1/xxx?action=config , 这样就能获取到action的值

```
param = request.args.get('action')  
```



#### 后端接口的请求参数和返回格式一定要和前端接口相匹配：

例如：

uploadimage

请求参数：

```
GET{"action":"uploadimage"}
POST{"upfile":"File Data"}
```

返回格式：

```
{
	"state":"SUCCESS",
	"url":"upload/demo.jpg",
	"title":"demo.jpg",
	"original":"demo.jpg"
}
```

构造格式：

```
	result = dict()  # 构造响应的数据格式
    result['state'] = 'SUCCESS'
    result['url'] = f"/upload/{newname}"
    result['title'] = filename
    result['original'] = filename

    return jsonify(result)  # 以JSON数据格式返回响应，供前端编辑器引用
```

其他的接口也一样



#### 从前端页面拿的数据可以直接保存到本地文件夹：

```
    if request.method == 'GET' and param == 'config':
        return render_template('config.json')
    # 上传图片的请求
    elif request.method == 'POST' and request.args.get('action') == 'uploadimage':
        f = request.files['upfile']  # 网页中固定写好的字典变量，从这获取前端的图片文件数据
        filename = f.filename  # 获取文件名
        
        # 图片压缩
        suffix = filename.split('.')[-1] #获取图片的文字
        newname = time.strftime('%Y%m%d_%H%M%S.' + suffix) #构造新的文件名字
        f.save('./resource/upload/' + newname)  #保存文件到本地

        source = dest = './resource/upload/' + newname #
        compress_image(source, dest, 400)
```



反过来，从本地文件夹拿图片展示到浏览器：

```
 列出所有的图片给前端
 elif request.method == 'GET' and param == 'listimage':
 	 list = []
     filelist = os.listdir('./resource/upload') #列出图片目录下所有的文件
     for filename in filelist:
     	#判断是否为图片
     	if filename.lower().endswith('.png') or filename.lower().endswith('.jpg'):
    	 	list.append({"url": "/upload/%s" % filename})

```



## 文章发布功能

> 文章编辑完成后当然就得发布文章，某种意义上就是一个POST请求而已。
>
> 但是要解决很多问题
>
> **1.图片压缩问题**
>
> ​	作者只是简单的上传并确保前端能正常显示。但是服务器端必须要处理这个问题，否则会消耗大量的服务器端的带宽和硬盘空间。
>
> 同时，图片过大之后，当用户阅读文章时，文章的加载时间也会变长，影响用户体验。
>
> 生成验证码时用到的pillow库便可以用来调整图片的尺寸以及压缩图片的大小--->在压缩图片的时候进行压缩处理
>
> **2.缩略图的处理**
>
> ​	常规处理是作者主动上传一个文章的封面。但是这样处理就会增加作者写文章的负担，还需要专门为文章找一个封面。
>
> ​	所以蜗牛笔记采用的方案是直接从文章内容中查找图片，获取到图片地址后将其作为文章的封面。所以蜗牛笔记采用的方案是直接从文章内容中查找图片，获取到图片地址后作为封面。
>
> ​	1）如果是作者自己上传的图片（图片的地址的域名是本地上传的图片），则直接对该图片进行压缩处理，		  然后保存到对应的缩略图目录下，并同步将文件名保存到article表的thumbnail字段中。
>
> ​	2）如果文章中的图片是引用别的网站中的图片，则直接将该图片下载到服务器中再进行处理。
>
> ​	3）如果作者的文章中不存在任何一张图片，则直接根据文章的类型为其指定文章的缩略图，当然事先必须		  准备好一批缩略图备用	



## 图片压缩

图片压缩分为两种压缩格式：

> 1.压缩图片的尺寸
>
> 2.压缩图片的大小
>
> 通常建议两种压缩方式一起使用。

---



由于图片编码的问题，对于PNG格式的图片，pillow库的Image模块只能通过调整图片尺寸来达到压缩的目的，而不能通过保存图片是进行二次压缩。当然也有其他的解决方案....例如python的pngquant库就可以实现。



```python
from PIL import Image
import os

#定义原始图片的路径
source = 'image/source.jpg'
#以KB为单位获取图片的大小
size = int(os.path.getsize(source)/1024)

im = Image.open(source)
width,height = im.size
if width > 1000:
	#等比例缩放
	height = int(height*100/width)
	width=1000
#调整当前的图片的尺寸（同时也会压缩大小）
dest = im.resize((width,height) ,Image.ANTIALIAS)

size = int(os.path.getsize("image/new.jpg")/1024)
```



#### 压缩图片函数

```
# 图片的压缩
def compress_image(source, dest, width):
    from PIL import Image
    # 若宽度小于width，则不进行调整
    im = Image.open(source)
    x, y = im.size
    if x > width:
        ys = int(y * width / x) #根据宽度，等比例缩放
        xs = width
        temp = im.resize((xs, ys), Image.ANTIALIAS) #压缩尺寸
        temp.save(dest, quality=80)	#压缩质量
    else:
        im.save(dest, quality=80)
```



#### 从获取文章中的图片

```
# 解析文章内容中的图片地址
def parse_image_url(content):
    import re
    #用正则表达式，把网页的内容全部拿下来，从html文本内容中去匹配标签
    temp_list = re.findall(r'<img src="(.+?)"', content)
    url_list = []
    for url in temp_list:
        if url.lower().endswith('.gif'): #过滤掉gif图片
            continue
        url_list.append(url)
    return url_list
```



过程：先解析文章，把图片的路径保存到列表里面 ---->根据路径去访问图片 ，本地图片（图片路径在解析的列表里）直接压缩  ，不是本地图片，先去下载，再去压缩



#### 读取图片

```
################二、输入读取图片###################
# client:
resp = requests.post("http://localhost:5005/stream_predict",
                     files={"file": open("cat.jpg",'rb')})
print(resp.json())

# server:
@app.route('/stream_predict', methods=['POST'])
def stream_predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        batch_result = streamer.predict(img_bytes)
        print(batch_result)
        return 'done'

```



#### 下载图片

> 通过request.get(路径)就可以去访问图片了
>
> 然后open一个文件，以二进制写入保存就ok了

```
# 下载作者上传的别的网站上图片，通过url去访问下载
def download_image(url, dest):
    import requests
    response = requests.get(url)
    with open(file=dest, mode='wb') as file:
        file.write(response.content)
```



#### 缩略图的处理

```
# 解析图片的url并生成缩略图，返回缩略图名称
def generate_thumb(url_list):
    for url in url_list:
        # 如果有本地图片，直接用本地图片做缩略图
        if url.startswith('/upload/'):
            filename = url.split('/')[-1]
            compress_image('./resource/upload/' + filename, './resource/thumb/' + 																				filename, 400)
            return filename

    # 如果没有本地图片，用第一张
    url = url_list[0]
    filename = url.split('/')[-1] #url中获取文件名
    suffix = filename.split('.')[-1] # 通过文件名获得文件的后缀名
    thumbname = time.strftime('%Y%m%d_%H%M%S.' + suffix) #缩略图名字
    download_image(url, './resource/download/' + thumbname)
    compress_image('./resource/download/' + thumbname, './resource/thumb/' + thumbname, 																					400)
    return thumbname

```

​	

#### 缩略图怎么显示呢？

> 你先把图片压缩了,article表里面存着缩略图的名字(有来名字,就能凑出图片的地址,resource文件下xxx.png),要显示,前端<img @src=‘/thumb/{{article.thumbnail}}’>

---

其实,download_imag()和compress_image()这两个函数以及把图片的地址安排的明明白白的了



#### 保存草稿

> 1.保存草稿和插入文章不一样,不能向article表里面直接插入一条新的记录
>
> 2.如果用户保存了草稿没有正式发布,回到了其他的页面,则用户没有入口再回到页面去编辑直接的草稿
>
> 所以,用户保存了草稿就应该,后续的保存都应该是基于前面保存的基础上进行修改,而不是插入一条新的记录,
>
> 同时,还应该给提供用户一个编辑草稿的页面的入口
>
> ---
>
> 如何保存草稿?
>
> 一篇草稿反复保存都应该对于同一条记录进行操作.
>
> 如何知道这是新的草稿还是以及保存过的?  drafted字段来判断
>
> 用户第一次保存草稿的时候,会插入一条新的记录,并且同时生成一个新的articleid,
>
> **保存草稿之后,不要跳转页面,重新为articleid赋值**
>
> 在前端代码中articleid用一个全局变量保存起来,保存草稿就是循环调用保存草稿的函数,article会迭代更新 ,
>
> 如果将这一编号返回给前端,在后续提交请求的时候,再将该articleid带上,后台可根据这个编号来判断是新的文章还是已经存在的文章.
>
> **草稿的前后端交互纽带--->articleid**



```
# 草稿文章进行更改
    def update_article(self, articleid, type, headline, content, thumbnail, credit, drafted=0, checked=1):
        now = time.strftime('%Y-%m-%d %H:%M:%S')
        row = dbsession.query(Article).filter_by(articleid=articleid).first()
        row.type = type
        row.headline = headline
        row.content = content
        row.thumbnail = thumbnail
        row.credit = credit
        row.drafted = drafted
        row.checked = checked
        row.updatetime = now
        dbsession.commit()
        return articleid 	#返回文章id,前端用来跳转页面
```

一旦将草稿文章提交,将drafted更改为0, 即草稿变文章

## 文件上传

> 1.上传到到页面表单,提交的时候同步上传文件
>
> 2.使用Ajax动态请求





## Redis缓存

> Redis是一个开源的使用C语言编写,遵守BSD协议 ,支持网络,可基于内存亦可以持久化的日志型/Key-Values数据库,并提供多种语言的API. 
>
> 通常称为数据结构服务器, 因为值可以是字符串/哈希/列表/集合/有序集合
>
> 哈希类型,就是python中的两个字典类型复合 ,  {"Key": {"Key1":"value1","Key2":"values2"}
>
> ---
>
> 一个Redis库中可以保存多个Key,每条Key可以对应多条数据.



redis命令查询:

> https://www.runoob.com/redis/redis-lists.html

---

Redis默认将数据保存于内存中,这也是缓存服务器的核心工作机制,但是一旦内存出现故障,那么数据将完全丢失.所以Redis也提供数据持久化的操作,在下一次启动Redis后,仍然会加载上一次的数据

目前有两种方式:

1.RDB持久化方式能够在指定的时间间隔内,对数据进行快照存储,这也是Redis默认的持久化策略.

2.AOF持久化方式记录每次对服务器写的操作,当服务器重启的时候会重新执行这些命令来恢复原始的数据.

​	AOF命令以Redis协议追加保存每次写的操作到文件末尾. 由于用户执行的命令可能,存在重复,所以可以直接修改	AOF文件,删除一些重复的命令

3.同时开启两种方式, 优先载入AOF文件来恢复数据.因为在通常情况下,AOF文件保存的数据集要比RDB文件保存的数据集要完整



windows.conf 里面修改配置文件：

```
AOF：
appendonly yes

RDB：
#save 900 1
#save 300 10
#save 60 10000
save 60 2   没60秒修改两次就持久化保存
```



### Python 操作Redis

>  事实上，Redis本身是一个标准的网络服务器，只要遵循Redis的通信网络规范，任何一门编程语言都可以很容易的连接到Redis服务器上并执行命令。
>
> ---
>
> python通过Socket连接到Redis服务器，并发送一批满足Redis协议规范的字符串，为Redis设置了一个变量phone并通过get命令取得值

socket连接操作原理：

```python
import socket
s = socket.socket()
s.connect(('127.0.0.1'),6379) #连接Redis，只要端口正确，就可以远程连接

s.send(b'*3\r\n')  # *3表示发送的命令包含有3个字符串 ,不用b‘’ 字符串就用 .encode()方法进行编码
s.send(b'$3\r\n') # $3表示接下来的字符串有3个字符
s.send(b'phone\r\n')
s.send(b'$11')
s.send(b'13632649491')

r = s.recv(1024) #一条完整的命令发送后接收Redis服务器响应

print(r.decode()) #print输出 “+ok”就代表成功
```



### Redis连接：

```
import connection_pool as connection_pool
from redis.client import Redis
import redis

#连接池连接
pool = redis.ConnectionPool(host ='127.0.0.1',port=6379,decode_responses=True,db=0)
red = redis.Redis(connection_pool==pool)
return red
#普通连接
import redis
r = redis.Redis(host='192.168.2.105', port=6379)
# r = redis.Redis(host='192.168.2.105', port=6379, password=123)  # 带密码认证的redis操作
```



### 时间类型转换

```
 if not k1.startswith('_sa_instance_state'):
     if not k1 in dict:
     	if isinstance(v1, datetime):
    		 v1 = v1.strftime('%Y-%m-%d %H:%M:%S')
```



### 用户信息存入redis

```
# 将所有的用户信息缓存到redis中
def redis_mysql_hash():
    red = redis_connect()
    result = Users().find_all()
    user_list = model_list(result)  # 把模型类转换成json格式的数据类型，[{},{}...]
    for user in user_list:
        red.hset('users_hash', user['username'], str(user))  
        # 遍历用户列表，用户信息添加进redis数据库
```

> 用户的信息，要一条一条的存，一条一条的取，因为这样可以提高查找的效率，如果直接以json格式存取全部的user那样，要遍历整个表的json字符串，效率太低
>
> 所以，一个用户的用户名作为Key，信息作为Value



使用Redis的哈希结构来存：

> 主Key是user_hash ，哈希里面的Key是用户名  里面的value是 密码和其他信息对应的字典
>
> 也就是说，原本是一张表一个用户，表里面是该用户的信息
>
> 现在呢，用user_hash一张表，存所有的用户，我们原先的目的就是提高快速索引用户的方式，字典可以达到目的，所以选哈希结构

```
 red.hset('users_hash', user['username'], str(user))  # 用户信息添加进redis数据库
```



### redis注册/登录

> 用户注册是往mysql数据库里面写东西，登录校验是从redis里面查数据，数据不同步是不行的！







### 缓存文章列表

用户获取文章，之前是用Jinjia2的过滤器对文章内容进行过滤，然后再对文章进行截取处理，显示出文章摘要，这个过程首先要获取所以的文章内容，然后进行处理。

如何用Redis数据库预先将文章数据存入，这会大大降低数据库的查询工作

> 文章首页有文章摘要，你在读取之后，存到redis之前，直接用字符串切片切出来一部分，就行了，就不用过滤器那些操作了

```python
# 缓存文章信息
def redis_article_zsort():
    result = Article().find_all_with_user()
    list = []
    for article, nickname in result:
        dict = {}
        for k, v in article.__dict__.items():
            if not k.startswith('_sa_instance_state'): #跳过内置字段
                #如果某个字段的值是datetime类型，则将其格式改为字符串类型
                if isinstance(v, datetime):
                    v = v.strftime('%Y-%m-%d %H:%M:%S')
                #将文章内容的HTML和不可见的字符删除掉，再截取前80个字符
                elif k == 'content':
                    pattern = re.compile(r'<[^>]+>')
                    temp = pattern.sub('', v)
                    temp = temp.replace('&nbsp;', '')
                    temp = temp.replace('\r', '')			#过滤掉无用字符
                    temp = temp.replace('\n', '')
                    temp = temp.replace('\t', '')
                    v = temp.strip()[0:80]  				#切出摘要
                dict[k] = v
            dict['nickname'] = nickname
            list.append(dict)					#最终构建一个标准的列表+字典的数据结构
            
    red = redis_connect()
    for row in list:
        # red.zadd(键名, {值:排序依据}) ，这是一个根据文章id排序的有序集合
        red.zadd('article', {str(row): row['articleid']})

```





### Redis构建首页

```python
# 首页
@index.route('/redis')
def home_redis():
    red = redis_connect()
    # 获取有序集合总数量
    count = red.zcard('article')
    total = math.ceil(count / 10)
    # 用zrevrange从有序集合中，倒序取10条，即最新文章
    result = red.zrevrange('article', 0, 9)
    #由于加载进来的每一条数据是一个字符串，需要使用eval函数将其转换为字典
    article_list = []
    for row in result:
        article_list.append(eval(row))	#构建一个标准的列表+字典的数据结构

    return render_template('index-redis.html', article_list=article_list, page=1, total=total)
```



> 使用Redis，前端的数据全部从Redis的article_list里面取  -- 写个for遍历aericle_list ,里面就是article对象



![Screenshot_20220125_081848](D:%5CHuawei%20Share%5CHuawei%20Share%5CScreenshot_20220125_081848.jpg)

### Redis分页

```python
# 分页功能
@index.route('/redis/page/<int:page>')
def paginate_redis(page):
    pagesize = 10
    start = (page-1)*pagesize

    red = redis_connect()
    count = red.zcard('article')
    total = math.ceil(count / 10)
    result = red.zrevrange('article', start, start+pagesize-1)
    article_list = []
    for row in result:
        article_list.append(eval(row))

    return render_template('index-redis.html', article_list=article_list, page=page, total=total)
```



## 缓存更新策略：

> 1.把数据库的表一次性保存到缓存当中，然后从缓存中去取。
>
> 2.按需存取：要读取一条数据，优先从Redis取 ，如果Redis里面没有，则再向Mysql 中取（如果数据是第一次存取，则必然会从数据库中取），取的同时，同步将该条数据缓存到Redis里面。第二次取值时，则会从Redis中，取到，并且Redis也进行了更新
>
> 3.如何设计Redis的Key？
>
> ​	(1)使用 路由地址+参数  作为Key  -- ---  redis_page_15 或者  login-username-qiangge@woniuxy.com
>
> ​	(2) 函数 或 方法名 +参数 作为Key ？paginate_redis_15
>
> ​	(3) 使用任意表格当中的列名	
>
> 4.在同一个系统中，存在Redis数据源 和Mysql 数据源时 ，对于数据的操作，还有一些可能出现的情况。
>
> ​	雪崩:   集群  ，消息队列（页面响应需要排队，不要一下子加载所有的数据）
>
> 5.Redis内存以及存满了，如何处理？
>
> ​	 LRU方法，最低使用频率，把最低使用频率的东西给删掉
>
> 6.后台架构：
>
> ​	1.高可用性（HA）：7*24小时不间隔运行，多层结点，一条路线中间某个结点出现问题，系统直接垮掉了。
>
> ​	2.一层至少有x个结点，一个结点不行了，换同一层的另外一个结点继续运行
>
> ​	3.高并发（HC）： 例如：抢票秒杀
>
> ​	4.系统瓶颈的相对比例，其中一个结点具有瓶颈，换到其他的结点，同样会影响效率
>
> ​	5.高性能(HP):   解决高并发，正常并发，长时间运行 等等问题
>
> ​	6.安全性（Security）：
>
> 



## 页面静态化处理：

> 静态化：HTML （JS+CSS）-- 动态生成内容（处理） -- 模板引擎 -- 消耗服务器资源
>
>  为什么要静态化？
>
> 对于首页来说，由于要访问文章列表，要渲染内容摘要，还需要排序，分页，以及右侧推荐侧边栏，至少要执行三次查询，，访问数据库太频繁了。另外，查询完数据，还得渲染处理，静态化就全部省略掉了。



首页、分页的代码其实都是不变的，所以第一遍访问的时候把动态的页面给存起来，下次再访问的时候就读取静态的页面就好了。

如何对动态内容进行静态化处理：

1.预处理：直接把数据库查询和模板渲染的过程先实现一遍，但是不响应，而是直接将渲染完成后的页面写入html文件中

2.当访问首页时，直接读取首页对应的html文件

3.如果访问第二页时，则同样地读取第二页对应的html文件



>  页面没有更新怎么办？
>
> 数据库里面的数据更新,意味着全部静态页面需要重写
>
> 1.定时触发
>
> 2.新增时触发: 新增文章成功后,将已经存在的文章列表全部删除,以便于生成新的静态文件
>
> 3.用户访问时触发:采用这种,用户访问时,先判断是否存在静态页面,存在就直接响应,否则先查询,渲染后再响应
>
> 4.手动更新  手动去删除静态文件
>
> ---
>
> **静态化页面一定要及时更新 , 有些页面不能静态化,有些页面可以静态化**





定时触发:

```
# 死循环和时间判断执行定时更新静态页面

while True:
    now = time.strftime('%H:%M')
    if now == '03:00':
        # 清空静态页面
        list = glob.glob('../template/index-static/*.html')
        for file in list:
            os.remove(file)

        # 清空完成后，重新调用
        requests.get('http://127.0.0.1:5000/sta')
        print('删除成功并重新生成首页')
    time.sleep(60)
```



用户访问触发:

```
# 判断是否存在首页，存在直接读取静态
    if os.path.exists('./template/index-static/index-1.html'):
        return render_template('/index-static/index-1.html')
```

新增触发:

```
# 文章发布成功后，静态页面删除更新
                    list = os.listdir('./template/index-static/')
                    for file in list:
                        os.remove('./template/index-static/' + file)
```

​	

## 伪静态化

通过路由规则的定义,来模拟一个.html 后续的URL地址,让用户或搜索引擎误以为这是一个HTML静态页面

* 跟静态化处理一点关系都没有,对系统性能的提升一点作用都没有.

比如: http://127.0.0.1:5000/page/3-html  看上去是静态化,其实不是,只是看起来而已



# 丞康项目



## 跨域问题

1. [Flask 开启跨域 - JunCode - 博客园 (cnblogs.com)](https://www.cnblogs.com/Jaryer/p/14713828.html) 
2.  [Flask教程(13)--跨域访问 - 掘金 (juejin.cn)](https://juejin.cn/post/6970686954056712228) 
3. 



```
1. 使用 `CORS函数` 配置全局路由 
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)

2. 使用 `@cross_origin` 来配置单行路由 
from flask import Flask, request
from flask_cors import cross_origin

app = Flask(__name__)


@app.route('/')
@cross_origin(supports_credentials=True)
def hello():
    name = request.args.get("name", "World")
    return f'Hello, {name}!'
```









```python
@ct.route('/uploadImg',methods=['GET','POST'])
def get_picture():
    if request.method == 'GET':
        return render_template('CT.html')
    if request.method == 'POST':
        imgnameList = []
        patientid = request.form.get('patientid')
        session['patient'] = patientid
        # img_path = basepath + patientid + "/"
        img_path = basepath + patientid
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        img = request.files.getlist('imag')
        for singleimg in img:
            ext = singleimg.filename.rsplit('.',1)[1]
            imgName = modify_pictureName(patientid) + '.' + ext
            imgPath = os.path.join(img_path,imgName)        
            singleimg.save(imgPath)
            imgnameList.append(imgName)
        session['imgnameList'] = imgnameList
        for i in range(6):
            i = str(i)
            model_predict(i,patientid)
        return redirect('/predict/2')
    else:
        flash('上传失败！重新上传！')
        time.sleep(1)
        return redirect('/uploadImg')
    
def get_path():
    path_list = []
    type_list = []
    for path in os.listdir(r'./common/saved_model'):
        path_list.append(path)
    path_list.remove('init.py')
    for i in path_list:
        type = i.replace('resnet34_unet', '')
        type = type.replace('.pth', '')
        type_list.append(type)
    return path_list, type_list




def modify_pictureName(patientid):
    dir = './resource/CUTGAP2/' + str(patientid)
    now = time.strftime('%Y.%m.%d')
    num = len([i for i in os.listdir(dir) if not os.path.isdir(dir + '/' + i)])
    num += 1
    img_name =str(patientid) + '-' + str(num)+'-'+str(now)
    return img_name



def model_predict(type, patientid):
    path_list, type_list = get_path()
    dir_list, picture_list = get_picture()
    path0 = path_list[int(type)]
    illness_type = type_list[int(type)]

    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 载入模型
    model = resnet34_unet(1, pretrained=False).to(device)
    imgs_path = r"./resource/CUTGAP2"

    path = r'./common/saved_model/' + path0
    model.load_state_dict(torch.load(path, map_location='cpu'))  # 载入训练好的模型
    model.eval()

    dir = './resource/CUTGAP2/' + patientid
    pos=search_dir(imgs_path,dir)

    dirnames = dir_list[pos]
    jpg_list = picture_list[pos]

    for jpg in jpg_list:
        image = Image.open(imgs_path + '\\' + dirnames + '\\' + jpg)

        image = image.resize((256, 256))
        # 将图片转化为tensor
        image = x_transforms(image)
        # 将图片变成模型需要的维度
        image = Variable(torch.unsqueeze(image, dim=0).float(), requires_grad=False)

        # 将图像输入到网络当中进行预测
        predict = model(image)[0]
        toImg = transforms.ToPILImage()
        predict = toImg(predict)
        dir = os.path.join(imgs_path + '\\' + dirnames + '\\' + dirnames + illness_type)
        if not os.path.exists(dir):
            os.makedirs(dir)

        predict.save(dir + '\\' + jpg)
    print(dirnames + illness_type + ' 已完成预测')
    return 'predict success'


@ct.route('/predict/<type>',methods=['GET','POST'])         
# 轮播图需要的标准路径名
def predict(type):
    imgnameList = session.get('imgnameList')
    yuan_images_and_path = []
    patientid = session.get('patient')
    yuanshi_path = '/CUTGAP2/' + patientid + '/'
    result = find_dir(patientid,type)      # 注意传参的顺序
    yuceimgPath = '/CUTGAP2/' + patientid + '/' + result + '/'
    yuceimgList = []
    for i in imgnameList:
        yuceimgList.append(yuceimgPath + i)
        yuan_images_and_path.append(yuanshi_path + i)
    print('yuceimgList',yuceimgList)
    print('yuan_images_and_path',yuan_images_and_path)
    return render_template('resultShow.html',
                      yuceimgList=yuceimgList,yuan_images_and_path=yuan_images_and_path)



def search_dir(imgs_path,dir):   
    dir_list = os.listdir(imgs_path)
    for i,j in enumerate(dir_list):
        dirname = imgs_path+'/'+j
        if dirname == dir:
            return i
    return None


def find_dir(patientid,type):
    dir = r'./resource/CUTGAP2/' + patientid
    imgs_path = r"./resource/CUTGAP2"
    pos = search_dir(imgs_path, dir)
    dir_list, picture_list = get_picture()
    dirnames = dir_list[pos]
    path_list, type_list = get_path()
    illness_type = type_list[int(type)]
    return dirnames + illness_type


def single_patient_data(path):
    # path = r'./resource/CutGAP2'
    area_list = []
    for dirnames in os.listdir(path):
        for filenames in os.listdir(path + '\\' + dirnames):
            if filenames.lower().endswith(('_shibian')):
                sum_shibian = getArea(path + '\\' + dirnames + '\\' + filenames)
                area_list.append((4,sum_shibian))
            elif filenames.lower().endswith(('_feidapao')):
                sum_feidapao = getArea(path + '\\' + dirnames + '\\' + filenames)
                area_list.append((0,sum_feidapao))
            elif filenames.lower().endswith(('_fengwo')):
                sum_fengwo = getArea(path + '\\' + dirnames + '\\' + filenames)
                area_list.append((1,sum_fengwo))
            elif filenames.lower().endswith(('_moboli')):
                sum_moboli = getArea(path + '\\' + dirnames + '\\' + filenames)
                area_list.append((3,sum_moboli))
            elif filenames.lower().endswith(('_wangge')):
                sum_wangge = getArea(path + '\\' + dirnames + '\\' + filenames)
                area_list.append((5,sum_wangge))
            elif filenames.lower().endswith(('_lung')):
                sum_lung = getArea(path + '\\' + dirnames + '\\' + filenames)
                area_list.append((2,sum_lung))
    return area_list





```



```

```

