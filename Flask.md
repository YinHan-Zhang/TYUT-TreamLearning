---

---

# Flask

## 初始化配置

```python
#从flask包中导入flask对象
from flask import Flask
import config

#使用flask创建一个app对象，并且传递__name__参数
app = Flask(__name__)

'''
app.config设置：

单个设置：
设置把字典未json格式数据
app.config['JSON_AS_ASCII']=True

配置文件设置：
JSON_AS_ASCII=False

导入：
import config
app.config.from_object(config)
'''
app.config.from_object(config)



#路由，设置访问的url--根路径
@app.route('/')
def hello_world():
    #输出hello world
    return {'Hello Flask啊!': 2021}

if __name__ == '__main__':
    app.run()

```

## config设置

```
app.config设置：

1.单个设置：
设置把字典未json格式数据
app.config['JSON_AS_ASCII']=True

2.配置文件设置：
例：JSON_AS_ASCII=False

导入：
import config
app.config.from_object(config)
```

## URL与视图

### 访问详情页

```
@app.route("/book/<int:book_id>")
def book_detail(book_id):
    for book in books:
        if book_id == book['id']:
            return book
        else:
            return "fail finded !"
    return "success!"
```

```python
books = [    {"id":1,"name":'<python>'},    
         {"id":2,"name":'<C++>'},
        ]

快速返回json字符串：jsonfy

获取视图：指定url获取更详细的映射
    查询图书详情，需要id的索引，所以我们传入一个int类型的book_id
    同时定义的函数里也要传入book_id这个参数
    <数据类型，参数名称>
    数据类型：
    string:字符串类型，默认的数据类型，接受没有任何斜杠的字符串
    int,float：整形和浮点型
    path:和string类似但是可以传递斜杠
    uuid：uuid类型的字符串
    any：可以指定多种路径

```

### url_for()

```
构造url：根据函数名找url

导入url_for包

1.将来如果修改了url但是没有修改函数名，就不用导出去替换url了，就是说不用去app.route()里面更改url了
2.url_for()会转义一些特殊字符和Unicode字符串，这些事情url_for会自动帮我们搞定

#添加字段url,url_for(函数名，参数名=url路径)
book['url']=url_for("book_detail",book_id=book['id'])
```

### 指定url请求方法

```python
指定http请求方法：常用get和post
1.如果从只是从服务器上获取数据，一般用get请求
2.如果需要把数据发送给服务器，一般用post请求
3.用法：在app.route(,methods=['GET','POST'])
```

## 页面跳转和重定向：

```python
永久性重定向：http的状态码是301，多用于旧网址要转到新网址确保用户的访问
暂时性重定向：http的状态码是302，表示页面的暂时性跳转

导入redirect,request

参数传递的两种形式：
1.作为url的组成部分：/book/1
2.查询字符串：/book?id=1 ，以字符串的形式输入参数id的值

@app.route("/profile")
def profile():
    #获取id
    user_id = request.args.get("id")
    if user_id:
        return "用户个人中心"
    
    #如果用户没有输入id,使用redirect跳转到你设置的页面,url_for通过函数名反向找到url路径
    else: return redirect(url_for("index"))

@app.route('/index')
def index():
    return {'info':'跳转到这里辣！！！'}
```

## 模板

```python
模板：web开发必备模块，因为我们在开发网页的时候，并不只渲染一个纯文本字符串，
而是带有HTML标签的页面

flask中配套的模板是jinja2

要渲染模板，就去templates里面创建HTML文件
并导入render_template模块
也可以这样导入template，app = Flask(__name__,template_folder='路径')

如果要向HTML文件中传递变量，要写成字典的形式
并使用**字典名传参，在html中使用两个{{values}}传入参数

@app.route("/about")
def about():
    context = {
        'username':'zyh'
    }
    return render_template("about.html",**context)

#也可以直接传一个字典的列表，里面包含多个参数，传入后需要遍历列表
user = {'username': 'zyh'}
posts = [
        {
            'author': {'username': 'John'},
            'body': 'Beautiful day in Portland!'
        },
        {
            'author': {'username': 'Susan'},
            'body': 'The Avengers movie was so cool!'
        }
    ]
#当然也可以直接传值，比如title直接赋值
    return render_template('index.html', title='Home', user=user, posts=posts)


@app.route('/index')
def index():
    user = {'username': 'Miguel'}
    return render_template('index.html', title='Home', user=user)
```

HTML文件中：使用传入的参数

 `render_template()`函数传入的参数中的相应值替换`{{...}}`块 

访问格式为：字典名.键名

```html
<html>
    <head>
        <title>{{ title }} - Microblog</title>
    </head>
    <body>
        <h1>Hello, {{ user.username }}!</h1>
    </body>
</html>

```





## 模板过滤器

模板过滤器：

我们想在HTML文件里面使用python内置的函数

jinja里面也内置了很多类似python内置函数的过滤器就把过滤器名放在管道符后面

例：username|length 		返回的就是参数的长度通过管道符 | 实现

常用过滤器： [第三节：模板过滤器 - Python框架Flask基础教程 - 知了传课 (zlkt.net)](https://www.zlkt.net/book/detail/10/278) 



## 控制语句

 所有的控制语句都是放在`{% ... %}`中，并且有一个语句`{% endxxx %}`来进行结束，`Jinja`中常用的控制语句有`if/for..in..` 

 **传入的HTML的参数是列表，使用for遍历**

 `for...in...`：`for`循环可以遍历任何一个序列包括列表、字典、元组。并且可以进行反向遍历 

* for循环语句

  ```
  {% for user in users %}
  	<li>{{ user.username|e }}</li>
  {% endfor %}
  
  {% for key, value in my_dict.iteritems() %}
  	<dt>{{ key|e }}</dt>
  	<dd>{{ value|e }}</dd>
  {% endfor %}
  
  ```

  ---

  ```
  posts = [
          {
              'author': {'username': 'John'},
              'body': 'Beautiful day in Portland!'
          },
          {
              'author': {'username': 'Susan'},
              'body': 'The Avengers movie was so cool!'
          }
      ]
      
  遍历传入的列表    
  <h1>Hi, {{ user.username }}!</h1>
          {% for post in posts %}
          <div><p>{{ post.author.username }} says: <b>{{ post.body }}			</b></p></div>
          {% endfor %}
  ```

* if语句`if`：

  if语句和`python`中的类似，可以使用`>，<，<=，>=，==，!=`来进行判断，也可以通过`and，or，not，()` 

  ```
  {% if kenny.sick %}
  	Kenny is sick.
  {% elif kenny.dead %}
  	You killed Kenny!  You bastard!!!
  {% else %}
  	Kenny looks okay --- so far
  {% endif %}
  ```

   另外，**不可以**使用`continue`和`break`表达式来控制循环的执行 

## 模板继承

> 基础语法：
>
> 父模版：
>
> 在公共的模块下面写空的block
>
> ```
> {%block title%} {%endblock%}
> 
> {%block head%} {%endblock%}
> 
> {%block body%} {%endblock%}
> ```
>
> 
>
> * 这个block的名字不一定是title、head、body  但是名字要一致
>
>   
>
> 子模板：在block中写东西
>
> ```html
> {%block title%}
> 	<title>内容</title>
> {%endblock%}
> 
> {%block body%}
> 	<body>
>      ...
> </body>
> {%endblock%}
> ```
>
> 快捷键：block+tab 
>
> ---
>
> 继承父模版中的代码：{{ super() }} ，不继承就会重载
>
> 另外，模板中不能出现重名的`block`，如果一个地方需要用到另外一个`block`中的内容，可以使用`self.blockname`的方式进行引用   
>
> ```
> <title>   
>   {% block title %}        这是标题    {% endblock %} 
> </title> 
> 
> <h1>{{ self.title() }}</h1>
> 
> 重用title
> ```
>
> 

先定义写好一个父模版--base.html

```python
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{% block title %}{% endblock %}</title>
</head>
<body>
    <ul>
        <li>
            <a href="/">首页</a>
        </li>
        <li>
            <a href="/control">控制</a>
        </li>
        <li>
            <a href="/about">关于我</a>
        </li>

    </ul>
{% block body %}{% endblock %}
<footer style="background-color: burlywood">我是底部标签</footer>
</body>
</html>
```

这个父模版也就是显示在根路径下的页面，上面是一个导航条，点击跳转页面

```python
{% extends "base.html" %}  

{% block title %}
    flask 首页
{% endblock %}

{% block body %}
    <h1>我是首页</h1>
{% endblock %}
```



## include语句和set语句

 `include`语句可以把一个模板引入到另外一个模板中，类似于把一个模板的代码copy到另外一个模板的指定位置 

```
{% include 'header.html' %}
	主体内容
{% include 'footer.html' %}
```



 有时候我们想在在模板中添加变量，这时候赋值语句（set）就派上用场了 

```
{% set name='zhiliao' %}
{% set navigation = [('index.html', 'Index'), ('about.html', 'About')] %}
```

 赋值语句创建的变量在其之后都是有效的，如果不想让一个变量污染全局环境，可以使用`with`语句来创建一个内部的作用域，将`set`语句放在其中，这样创建的变量只在`with`代码块中才有效 

```html
{% with %}
    {% set foo = 42 %}
    {{ foo }}           foo is 42 here
{% endwith %}

{% with foo = 42 %}
    {{ foo }}
{% endwith %}
```

 这两种方式都是等价的，一旦超出`with`代码块，就不能再使用`foo`这个变量了。 



## 静态文件配置

* 静态文件的配置一般放在head标签里面

* `url_for`函数默认会在项目根目录下的`static`文件夹中寻找`about.css`文件，如果找到了，会生成一个相对于项目根目录下的`/static/about.css`路径。 

* filename就是css文件在static文件下的路径

 通过`url_for`全局函数就可以实现，看以下代码： 

```
{%block head%}
	<link href="{{ url_for('static',filename='about.css') }}">
{%block head%}
```



## 蓝图与子域名

> 蓝图就是把视图函数封装到其他的文件里面，要使用就进行导入
>
> 类似于C里面的头文件
>
> 1.在工程目录下创建python package ，然后在里面新建py文件
>
> 2.
>
> ```
> from flask import Blueprint
> 
> #创建flask路径，参数是根路径的名字，模块名，访问的网址
> bp = Blueprint("book",__name__,url_prefix="/book")
> 
> @bp.route('/list')
> def book_list():
>     return "图书列表"
> ```
>
> 导入Blueprint这个包，创建bp，正常写视图函数



## 连接Mysql数据库

根据SQLalchemy这个库来进行操作

```python
from flask_sqlalchemy import SQLAlchemy

HOSTNAME = '127.0.0.1'
PORT = '3306'
DATABASE = 'zi_flask'
USERNAME = 'root'
PASSWORD = '20020520zyh'
DB_URI = 'mysql+pymysql://{}:{}@{}:{}/{}'.format(USERNAME,PASSWORD,HOSTNAME,PORT,DATABASE)
app.config['SQLALCHEMY_DATABASE_URI']=DB_URI
#设置终端是否跟踪每一次修改
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=True

app.py
#导入数据库，要写在视图函数的外面
db = SQLAlchemy(app)


@app.route('/')
def hello_world():
    #验证是否成功连接
    engine = db.get_engine()
    with engine.connect() as conn:
        result = conn.execute('select 1')
        print(result.fetchone())
    # conn = engine.connect()
    # conn.execute("select 1")
    # conn.close()
    return 'Hello World!'
```

## 对数据库的操作

```
#1.添加数据
    article = Article(title='钢铁是怎样炼成的',content='xxx')
    #写入表中
    db.session.add(article)
    #做一个提交操作
    db.session.commit()

    #2.查询数据
    #filter_by:返回一个类列表的对象
    article = Article.query.filter_by(id=1)[0]
    print(article.title)

    #3.修改数据,先查再改
    article = Article.query.filter_by(id=1)[0]
    article.content = 'yyy'
    db.session.commit()

    #4.删除属性
    article = Article.query.filter_by(id=1).delete()
    db.session.commit()
```



## ORM模型

```python
定义ORM模型
class Article(db.Model):#继承自db.model    __tablename__ = 'article'   #创建表的名字
#定义主键    
id = db.Column(db.Integer,primary_key=True,autoincrement=True)    
title = db.Column(db.String(200),nullable=False)    content = db.Column(db.TEXT,nullable=False)   

author_id = db.Column(db.Integer,db.ForeignKey("user.id"))
---


```

### 数据库的操作：

对数据库:

```python

db.session.add(ORM模型)
db.seesion.commit()

#中止会话并删除存储在其中的所有更改
db.session.rollback()
#暂时没有使用ORM迁移数据库，所以现在先删除所有表，再创建
db.drop_all()
#把映射执行到数据库中
db.create_all()
```

对表：

* filter_by

    `filter_by()`的结果是一个只包含具有匹配用户名的对象的查询结果集 

   因为我知道查询用户的结果只可能是有或者没有，所以我通过调用`first()`来完成查询，如果存在则返回用户对象;如果不存在则返回None 

  调用all（）返回所有 	；调用first（）返回一个结果

* query

  ​	ORM模型.query.all()  ：查询所有

  ​	query.get(id) ：查询一条



### request变量

  Flask提供一个`request`变量，其中包含客户端随请求发送的所有信息 

 特别是`request.args`属性，可用友好的字典格式暴露查询字符串的内容 





##  表关系

 表之间的关系存在三种：一对一、一对多、多对多 

* 外键

>  `user_id`字段被初始化为`user.id`的外键，这意味着它引用了来自用户表的`id`值 
>
> 
>
>  `User`类有一个新的`posts`字段，用`db.relationship`初始化。 
>
>  对于一对多关系，`db.relationship`字段通常在“一”的这边定义，并用作访问“多”的便捷方式。
>
> 因此，如果我有一个用户实例`u`，表达式`u.posts`将运行一个数据库查询，返回该用户发表过的所有动态。
>
> `db.relationship`的第一个参数表示代表关系“多”的类。 `backref`参数定义了代表“多”的类的实例反向调用“一”的时候的属性名称。
>
> 这将会为用户动态添加一个属性`post.author`，调用它将返回给该用户动态的用户实例 
>
>  u.posts 从一调用多 ，post.author 从多调用一
>
> 意思就是：
>
> u.posts 是已知用户，去查询该用户的body（发表的说说）、timestamp（时间）等写在post类里面的信息
>
> post.author是 已知 这条说说，去查发表说说的用户 
>
>  **`db.relationship`为用户添加了`posts`属性，并为用户动态添加了`author`属性 **

1.外键的类型一定要原来引用的类型

2.db.ForeignKey("表名.字段名")

3.外键属于数据库层面的，不推荐直接在ORM中使用

```
author_id = db.Column(db.Integer,db.ForeignKey("user.id"))
```

* relationship

```
绑定的关系
1.第一个参数ORM模型名字，必须保持一致
2.backref :反向引用，对方访问我的时候的名称
author = db.relationship("User",backref="articles")

3.在被绑定的类中，使用绑定关系
user = User(username='zhiliao')#调用user类去创建相应的键
article1.author = user #绑定关系，author和user关系绑定
```

```python
#绑定关系
db.backref
1.在反向引用的时候，如果需要传递一些其他的参数,就用这个函数，否则不需要
2.uselist = False:代表反向引用的时候，表示不是一个列表而是一个对象
user = db.relationship("user",backref=db.backref('extention',uselist=False))
```

* 调用其它类创建字段

```
user = User(username='zhiliao')#调用user类去创建相应的键
```

* 



## ORM模型

>  在实际的开发环境中，经常会发生数据库修改的行为。一般我们修改数据库不会直接手动的去修改，而是去修改`ORM`对应的模型，然后再把模型映射到数据库中 
>
> ---
>
>  `flask-migrate`是基于`Alembic`进行的一个封装，并集成到`Flask`中，而所有的迁移操作其实都是`Alembic`做的，他能跟踪模型的变化，并将变化映射到数据库中。 

* 创建migrate对象，绑定app和数据库

```
migrate = Migrate(app,db)
```

* 初始化迁移文件夹
  *   flask db init 
* 把当前模型添加到迁移文件夹中
  * flask db migrate 
* 把迁移文件中对应的数据库操作，真正的映射到数据库中
  *  flask db upgrade 
* 回滚上一次的迁移，降级数据库
  * flask db downgrade



#### models.py

 db.Model，它是Flask-SQLAlchemy中所有模型的基类 

 这个类将表的字段定义为类属性，字段被创建为`db.Column`类的实例 





## 表单验证

记得在config里面配置一下 SECRET_KEY ： 

Flask及其一些扩展使用密钥的值作为加密密钥，用于生成签名或令牌 

 **要求用户输入username和password，并提供一个“remember me”的复选框和提交按钮：** 

* 复选框：BooleanField
* 提交按钮：SubmitField
*  `DataRequired`验证器仅验证字段输入是否为空。 

```python
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')
```

* Method Not Allowed”错误正是由于视图函数还未配置允许`POST`请求。 

**SringField里面的第一个参数，是确保用户在此字段中输入的内容和我们定义的Email的ORM模型里的内容相匹配**



表单验证里面写子函数，是自定义验证器

表单的数据要验证通过，而且还要避免与已有的用户冲突

```
 def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('Please use a different username.')
```

 这两个方法执行数据库查询，并期望结果集为空 ，说明没有重复的用户名



### 头像

 生成电子邮件为`john@example.com`的用户的Gravatar URL： 

```
from hashlib import md5

'https://www.gravatar.com/avatar/' + md5(b'john@example.com').hexdigest()

'https://www.gravatar.com/avatar/d4c74594d841139328695756648b6bd6'
```

头像的类：

```python
from hashlib import md5



class User(UserMixin, db.Model):
	...
	#增加一个avatar类
    def avatar(self, size):
        digest = md5(self.email.lower().encode('utf-8')).hexdigest()
        return 'https://www.gravatar.com/avatar/{}?d=identicon&s={}'.format(digest, size)
```

 `User`类新增的`avatar()`方法需要传入需求头像的像素大小，并返回用户头像图片的URL。 对于没有注册头像的用户，将生成“identicon”类的随机图片。 为了生成MD5哈希值，我首先将电子邮件转换为小写，因为这是Gravatar服务所要求的。 然后，因为Python中的MD5的参数类型需要是字节而不是字符串，所以在将字符串传递给该函数之前，需要将字符串编码为字节。 

HTML中加载头像：

```html
{% block content %}
    <table>
        <tr valign="top">
            <td><img src="{{ user.avatar(128) }}"></td>
            <td><h1>User: {{ user.username }}</h1></td>
        </tr>
    </table>
    <hr>
    {% for post in posts %}
    <table>
        <tr valign="top">
        	#post.author 反向映射找用户！！
            <td><img src="{{ post.author.avatar(36) }}"></td>
            <td>{{ post.author.username }} says:<br>{{ post.body }}</td>
        </tr>
    </table>
    {% endfor %}
{% endblock %}
```



### flash（）函数：

```
HTML文件中这么用：

{% with messages = get_flashed_messages() %}
        {% if messages %}
        <ul>
            {% for message in messages %}
            <li>{{ message }}</li>
            {% endfor %}
        </ul>
        {% endif %}
        {% endwith %}
```

* 解释

 用了`with`结构在当前模板的上下文中来将`get_flashed_messages()`的结果赋值给变量`messages`。

`get_flashed_messages()`是Flask中的一个函数，它返回用`flash()`注册过的消息列表。接下来的条件结构用来检查变量`messages`是否包含元素，如果有，则在``元素中，为每条消息用``元素来包裹渲染。 

* 闪现消息的一个有趣的属性是，一旦通过`get_flashed_messages`函数请求了一次，它们就会从消息列表中移除，所以在调用`flash()`函数后它们只会出现一次。 

### url_for()函数

  它使用URL到视图函数的内部映射关系来生成URL。 例如，`url_for('login')`返回`/login` 

** url_for使用的是函数的名称而不是URL







## 项目重构

所有的文件写在一个文件中太乱了，所以要分文件写

### 1. constants.py文件：

常量文件，用来存放数据库配置



### 2. ext.py文件：

把`db`变量放到一个单独的文件，而不是放在主`app`文件。这样做的目的是为了在大型项目中如果`db`被多个模型文件引用的话，会造成`from your_app import db`这样的方式，但是往往也在`your_app.py`中也会引入模型文件定义的类，这就造成了循环引用。所以最好的办法是把它放在不依赖其他模块的独立文件中。



### 3. models.py文件：

模型文件，用来存放所有的模型，并且注意，因为这里使用的是`flask-script`的方式进行模型和表的映射，因此不需要使用`db.create_all()`的方式创建数据库。



### 4. app.py文件：

这个是主`app`文件，运行文件。并且因为`db`被放到另外一个文件中，所以使用`db.init_app(app)`的方式来绑定数据库。



项目重构之后，再依次执行三个迁移命令即可



### 报错

```
ERROR [flask_migrate] Error: Can't locate revision identified by
'a102121f815c'

解决：
1.创建一个新的数据库
2.去navicat里面alembic_version里删除之前的记录
网址：https://blog.csdn.net/m0_37605642/article/details/90638195
```



## cookie和session

```
导入3个类
from flask import Flask,Response,request,session
```

### cookie

* Resopnse(“页面显示内容”)  ：创建Response对象
* set_cookie(“cookie的键”，“cookie的值”)：给Resopnse对象赋值

```python
@app.route("/set_cookies")
def set_cookies():
    response = Response("cookies 设置")
    response.set_cookie("user_id",'xxx')
    return response

@app.route("/get_cookies")
def get_cookies():
    user_id = request.cookies.get("user_id")
    print("id:",user_id)
    return "获取cookies"
```



### session

在flask中，session先把session加密，然后用session——id作为key，存放在cookie中

* session 

```
#设置session的密码
app.config['SECRET_KEY'] = "123"

创建session字典对象，给session赋值
session['username'] = "zyh到此一游"
```

* session.get(‘键名’)：获取session值

```python
 username = session.get('username')
    print(username)
```

* session删除

```
 session.pop('username') 或者 session['username'] = False
```

* 清空session

```
session.clear
```

* 设置期限

```
# 设置session
 @app.route('/')
 def set():
     session['username'] = 'liefyuan'
     session.permanent = True # 长期有效，一个月的时间有效
     return 'success'
     
 一种更先进的配置有效期的方法：（比如配置7天有效）

1.引入包：from datetime import timedelta

2.配置有效期限：app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7) # 配置7天有效

3.设置：session.permanent = True


#encoding: utf-8
from flask import Flask,session
from datetime import timedelta
import os 
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7) # 配置7天有效 

# 设置session
@app.route('/')
def set():
    session['username'] = 'liefyuan'
    session.permanent = True
    return 'success'## 目标

```



## 表单验证

> 登录网站的时候，要输入用户名和密码，而网站的第一步操作不是去数据库里找数据，而是验证是不是用户名，是不是密码

 `Flask-WTF`是简化了`WTForms`操作的一个第三方库。`WTForms`表单的两个主要功能是验证用户提交数据的合法性以及渲染模板。当然还包括一些其他的功能：`CSRF保护`，文件上传等。安装`Flask-WTF`默认也会安装`WTForms` 



forms.py文件里面写验证规则：

```python
import wtformsfrom wtforms.validators import length,email

class LoginForm(Form):
    name = StringField(validators=[length(min=4,max=25)])
    email = StringField(validators=[email()])
    password = StringField(validators=[DataRequired(),length(min=6,max=10),EqualTo('confirm')])
    confirm = StringField()
```

```
from forms import LoginForm

@app.route("/login")
def login():
    #要登陆肯定需要一个登录页面
    if request.method == 'GET':
        return render_template('login.html')
    else:
        #传入请求的方法
        form = LoginForm(request.form)
        if form.validate():
            return "登陆成功"
        else:
            return "邮箱或者密码错误！"
```

### Field 常用参数：

**Field是wtforms里面的参数**

- label（第一个参数）：`Field`的label的文本。
- validators：验证器。
- id：`Field`的id属性，默认不写为该属性名。
- default：默认值。
- widget：指定的`html`控件。

### 常用Field

- BooleanField：布尔类型的Field，渲染出去是`checkbox`。

- FileField：文件上传Field。

- FloatField：浮点数类型的Field，但是渲染出去的时候是`text`的input。

- IntegerField：整形的Field。同FloatField。

- RadioField：`radio`类型的`input`。

  ```
  class RegistrationForm(FlaskForm):
      gender = wtforms.RadioField(u'性别：',validators=[DataRequired()])
  ```

  

- SelectField：类似于`RadioField`。

- StringField：渲染到模板中的类型为``，并且是最基本的文本验证。

- PasswordField：渲染出来的是一个`password`的`input`标签。

- TextAreaField：渲染出来的是一个`textarea`

  

### 常用验证器：

- Email：验证上传的数据是否为邮箱。
- EqualTo：验证上传的数据是否和另外一个字段相等，常用的就是密码和确认密码两个字段是否相等。
- InputRequired：原始数据的需要验证。如果不是特殊情况，应该使用`InputRequired`。
- Length：长度限制，有min和max两个值进行限制。
- NumberRange：数字的区间，有min和max两个值限制，如果处在这两个数字之间则满足。
- Regexp：自定义正则表达式。
- URL：必须要是`URL`的形式。
- UUID：验证`UUID`。



## 粉丝

多对多的查询

![1639143300020](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1639143300020.png)

 `followers`表是关系的关联表。 此表中的外键都指向用户表中的数据行，因为它将用户关联到用户。 该表中的每个记录代表关注者和被关注者的一个关系。 

follower表：

```
followers = db.Table('followers',
    db.Column('follower_id', db.Integer, db.ForeignKey('user.id')),
    db.Column('followed_id', db.Integer, db.ForeignKey('user.id'))
)
```

在User模型中声明多对多的关系：

```
class User(UserMixin, db.Model):
    # ...
    followed = db.relationship(
        'User', 
        secondary=followers,
        primaryjoin=(followers.c.follower_id == id),
        secondaryjoin=(followers.c.followed_id == id),
        backref=db.backref('followers', lazy='dynamic'), lazy='dynamic')
```

- `'User'`是关系当中的右侧实体（将左侧实体看成是上级类）。由于这是自引用关系，所以我不得不在两侧都使用同一个实体。

- `secondary` 指定了用于该关系的关联表，就是使用我在上面定义的`followers`。

- `primaryjoin` 指明了通过关系表关联到左侧实体（关注者）的条件 。

  关系中的左侧的join条件是关系表中的`follower_id`字段与这个关注者的用户ID匹配。`followers.c.follower_id`表达式引用了该关系表中的`follower_id`列。

- `secondaryjoin` 指明了通过关系表关联到右侧实体（被关注者）的条件 。

   这个条件与`primaryjoin`类似，唯一的区别在于，现在我使用关系表的字段的是`followed_id`了。

- `backref`定义了右侧实体如何访问该关系。

  在左侧，关系被命名为`followed`，所以在右侧我将使用`followers`来表示所有左侧用户的列表，即粉丝列表。附加的`lazy`参数表示这个查询的执行模式，设置为动态模式的查询不会立即执行，直到被调用，这也是我设置用户动态一对多的关系的方式。

- `lazy`和`backref`中的`lazy`类似，只不过当前的这个是应用于左侧实体，`backref`中的是应用于右侧实体。



#### 关注和取消关注：

```
user1.followed.append(user2)
user1.followed.remove(user2)
```

封装到类里面：方便代码重用

```
class User(UserMixin, db.Model):
    #...

    def follow(self, user):
        if not self.is_following(user):
            self.followed.append(user)

    def unfollow(self, user):
        if self.is_following(user):
            self.followed.remove(user)

    def is_following(self, user):
        return self.followed.filter(
            followers.c.followed_id == user.id).count() > 0
```

 `is_following()`方法发出一个关于`followed`关系的查询来检查两个用户之间的关系是否已经存在 

 查询以`count()`方法结束，返回结果的数量。 这个查询的结果是`0`或`1`，因此检查计数是1还是大于0实际上是相等的 



联合查询：

 应用主页中需要展示已登录用户关注的其他所有用户的动态 

 对每个已关注的用户执行一个查询来返回他们的用户动态。最后将所有用户的动态按照日期时间倒序合并到一个列表中 ？

 如果一个用户关注了一千人，会发生什么？ 我需要执行一千个数据库查询来收集所有的用户动态。 然后我需要合并和排序内存中的一千个列表 

不可能！！！

所以用到联合查询：

```
class User(db.Model):
    #...
    def followed_posts(self):
        return Post.query.join(
            followers, 
            (followers.c.followed_id == Post.user_id))
            .filter(followers.c.follower_id == self.id)
                .order_by(
                    Post.timestamp.desc()
                    )
```

解析：

```
Post.query.join(...).filter(...).order_by(...)
```

关联：

follower_id 是粉丝id ，followed_id是被关注者的id ，user_id 是 发表动态的人的id

```
Post.query.join(followers, (followers.c.followed_id == Post.user_id))
```

 第一个参数是followers关联表，第二个参数是join*条件*。 

我的这个调用表达的含义是我希望数据库创建一个临时表，它将用户动态表和关注者表中的数据结合在一起。 数据将根据参数传递的条件进行合并。 

* 关联的表：followers

* 条件： `followed_id`字段必须等于用户动态表的`user_id`字段 

过滤：

```
filter(followers.c.follower_id == self.id)
```

 `self.id`表达式是指我感兴趣的用户的ID。`filter()`挑选临时表中`follower_id`列等于这个ID的行 

**就是我要查看我指定的id用户关注了谁！！！**

排序：

```
order_by(Post.timestamp.desc())
```





# Flask的中包的用法

 

* 初始化Flask对象：app = Flask(_ _ name _ _)

* 外部设置，导入app文件中

  ```
  app.config.from_object(config)
  ```

* app内初始化db对象

  ```
  db.init_app(app)
  ```

* 创建迁移对象

  ```
  migrate = Migrate(app,db)
  ```

* 视图函数写在别的文件夹里面，先导入，再注册

  ```
  app.register_blueprint(qa_bp)app.register_blueprint(user_bp)
  ```

* 使用HTML文件渲染网页

  ```
  render_template(".html")
  ```

* 重定向

  ```
  redirect(url_for("user.login"))
  ```

* ```
  #这个函数的功能是你在网页上输入收件人的邮箱，这个函数把收件人的邮箱捕获并存入到数据库#用GET/POST请求拿到收件人的邮箱,get用request.args,post用request.form
  
  recipients_email = request.form.get("email"
  )
  ```

发送邮件，设置邮件的内容：

```
message = Message(   
	subject="邮箱测试",  
	recipients= [recipients_email],   
	body= f'别慌，这是拿来测试验证码的邮件,         
	验证码是：{capture}',    
	# sender= 默认发送者)

mail.send(message)#发送邮件
```

## 写入数据库

先再model.py创建相应的模板类，使用这个模板类创建表、写入数据库

```
capture_model = EmailCatchModel.query.filter_by(email=recipients_email,capture=capture)
```



* 向数据库添加内容，提交事务

```
db.session.add(capture_model)

db.session.commit()
```



## 表单验证

request.form会储存从前端页面上传上来的数据,然后拿去进行表单验证

```
form = RegisterForm(request.form)
```

从表单里获取数据：

```
email = form.email.data
在if语句里已经通过了验证已经通过了，capture就不用了
capture = form.capture.data
username = form.username.data
password = form.password.data
```



创建验证类：

```
import wtforms
from wtforms.validators import length,email,EqualTo

class RegisterForm(wtforms.Form):
    username = wtforms.StringField(validators=[length(min=3,max=20)])
    email = wtforms.StringField(validators=[email()])
    capture = wtforms.StringField(validators=[length(min=4,max=4)])
    password = wtforms.StringField(validators=[length(min=6,max=20)])
    #密码确认，EqualTo()就是变量相等
    password_confirm = wtforms.StringField(validators=[EqualTo("password")])
    
    def validate_capture(self,field):
       capture = field.data
       #去数据库里查询验证码，看一不一样
       capture_model =EmailCatchModel.query.fliter_by(email=email).first()
       #数据库里没有验证码，或者数据库里没有输入的验证码都认为验证不通过
       if not capture_model or capture_model.lower() != capture.lower():
           raise wtforms.ValidationError("邮箱验证码错误")

    #去数据库中验证邮箱是否已经存在，因为注册的时候一个邮箱只能注册一次
    def validate_email(self, field):
        email = field.data
        user_model = UserModel.query.filter_by(email=email).first()
        if user_model:
            raise wtforms.ValidationError("邮箱已经存在！")
```

## 数据库

创建数据库类：

```
from flask_sqlalchemy import SQLAlchemyfrom
```

邮件类：

```
mail = Mail()
```

去数据库查询数据：

```
capture_model =EmailCatchModel.query.fliter_by(email=email).first()
```



# 项目中各个文件和文件夹的作用：

## blueprints文件夹 :

 视图函数：写网页的页面

* 在这里面定义跟路由，和子路由
* 对网页的主要操作都写在这里
* 所有的模块在这里导入



实战中的用法：

* _ _init _ _ :

  如果要从blueprints文件夹 里导去 其他文件夹 ，先在这个进行起别名

* forms:表单验证

  wtforms包 ：用于验证数据的包

  wtforms.validators 导入用验证数据的长度，格式等方法

  class类的参数：wtforms.Form

  

qa / user :  定义跟路由的网页

主要流程：

1.获取请求方法，request.method == “ GET/POST”

2.表单验证 ，并提取表单中的数据

3.拿到数据后，像数据库中提交数据

4.结束 跳转网页





## migrations

迁移文件夹：

三个操作  ： init 、 migrate、upgrade

这里面储存的是操作数，记录的本次的操纵记录，可以去数据库里查询到

如果执行命令失败，去数据库里删除这条记录，然后重新执行三条语句



## Static

模板文件：

放置前端的模板 	用来渲染网页	HTML文件、js文件、css文件



## Template

编辑HTML文件，渲染网页

base.html :父模版

index.html ：登录模板

login.html ：注册

register.html ：注册



## 主文件夹

### app.py

​	主文件

### config.py

​	设置文件

### decorators.py

​	装饰器文件：把重复执行的函数功能包装在里面

### exts.py

中间件：存放 数据库对象 、 邮件对象

### models.py

创建数据库模板

定义表、字段、键 、关联



# 项目各文件中函数方法的用法总结

## blueprint



报错问题：!!!  点击发布或者登录过后，页面没有跳转登录，应该是表单验证出现问题了





### qa/user



#### 创建视图函数对象

```python
3个参数（视图文件名字，__name__,初始根路由）
bp = Blueprint("qa",__name__,url_prefix="/")
```



#### 从ORM模型中拿数据，也就是从数据库的表中提取数据

```python
query查询，传入question表中所有的数据,并根据时间进行排序,根据表中的字段要加上db.text()!!!
    questions = QuestionModel.query.order_by(db.text("create_time")).all()

将question中带有表单数据传入HTML文件，传入的questions是一个列表类型
return render_template("index.html",questions=questions)
```



#### 装饰器妙用

验证登录的装饰器

```python
判断是否登录 , 如果没有登录，跳转到登录页面
    if hasattr(g,'user'):
        
        普通去写函数判断就是判断g全局遍历是否带有user属性
       
        但是很多页面都需要登录，不可能大量重复的验证登录，我们要采用一些手段，
        这里可以通过装饰器实现
```

装饰器实现

```python
from flask import g,redirect ,url_for
from functools import wraps

def login_required(func):
    #这个装饰器是保留wrapper这个函数的一些属性
    @wraps(func)
    def warpper(*args , **kwargs):
        if hasattr(g,"user"):
            #验证成功，结束函数
            return func(*args,**kwargs)
        else:
            #验证不成功，跳转到登录页面
            return redirect(url_for("user.login"))
    return warpper

@login_required
def public_question():
    
装饰器就是把public_question这个函数当作形参传入装饰器函数，装饰器函数接收之后，return再去执行warpper函数 ，外层的装饰器函数就是一个空壳子，然后去执行warpper函数，public_question的参数都会被传入到warpper函数里，然后执行warpper函数的函数体，最后再返回public_question函数（带着被形参返回）
```



#### 表单验证并返回数据

```python
前端页面接收用户输入的数据，用request.form方法获取，然后传入表单验证函数
form = QuestionForm(request.form)

        if form.validate():
            #通过验证之后，提取数据
            title = form.title.data
            content = form.content.data
            #然后给模板传入参数提交到数据库的表中
            question = QuestionModel(title=title,content=content,author=g.user)
            db.session.add(question)
            db.session.commit()
```



#### flash错误提示：

```python
from flask import flash 

flash("标题或内容格式错误！") #flash的错误提示显示在前端页面

需要在HTML文件的input标签下面加入这段代码

{% for message in get_flashed_messages() %}
    <div class="form-group">
        <div class="text-danger">{{ message }}</div>
    </div>
{% endfor %}
```



#### 页面传参

```python
#要访问详情页，需要传一个参数：用户的id
传参格式： <int :参数名>
前端的路由里是这样的 name？q=xxxx  具体的name的HTML文件里设置
    
@bp.route("/question/<int:question_id>")

def question_detail(question_id):
    # 利用用户id去数据库里面拿数据
    question = QuestionModel.query.get(question_id)
    #这一步时把ORM模型（数据库表里的数据）传入前端页面的模板里面
    return render_template("detail.html",question=question)
```



#### 页面关键字搜索查询

```python
@bp.route("/search")
def search():
 '''
 	input输入框会传入关键字q
     /search？q=xxx
    request.args.get()就是从前端页面拿数据，get里面写input里面name的名字
    q = request.args.get("q")
    filter_by: 直接使用字段的名称
    filter:使用模型.字段名称
    or_：两个之中查到一个即满足
    根据数据库的字段进行排序用使用db.text()
 '''     
    questions = QuestionModel.query.filter_by(or_(QuestionModel.title.contains(q),
    QuestionModel.content.contains(q))).order_by(db.text("-create_time"))
    
    return render_template("index.html",questions=questions)
```



#### 密码加密解密

```python
#密码加密
hash_password = generate_password_hash(password)

 #如果用户存在切加密后的密码和原密码都对，用户的id放入session中保持登录状态并跳转页面
 if user and check_password_hash(user.password,password):
    session['user_id'] = user.id
    
   	#退出登录要清除session
    session.clear()
```





## models文件

### 外键和关联绑定的问题

```python
    #定义外键
    author_id = db.Column(db.Integer,db.ForeignKey("user.id"))
    #关联
    author = db.relationship("UserModel",backref ='questions')
```



### 直接操作表中的数据进行排序

```python
#这个relationship就是从别的ORM模型拿数据，然后反向索引backref给到answer, 然后再进行排序 
#和db.text一样对数据库的中的表进行操作，语法要注意 db.backref(反向索引，排序语句)

    question = db.relationship("QuestionModel",backref=db.backref('answers',
                             order_by= create_time.desc()))
    author = db.relationship("UserModel", backref='answers')
    
```



## app文件

### 钩子函数

```python
'''
钩子函数，在发送网络请求去访问各个视图函数之前，先拿到session中的user_id，这样就不用访问一个视图拿一次session，相当于一个中间层,
提前获取session中的user_id然后分发到各个需要的视图函数
'''


@app.before_request
def before_request():
    user_id = session.get("user_id")
    if user_id:
        try:
            user = UserModel.query.get(user_id)
            #g:全局变量，给g绑定一个叫user的变量，他的值是user这个变量
            # setattr(g,"user",user)
            g.user = user
        except:
            g.user = None
```





### 上下文管理

渲染的所有代码都会执行这块代码

具体流程：

发送请求 -》before_request ->视图函数 -》视图函数返回模板 -》 context_processor

```
@app.context_processor
def context_process():
    if hasattr(g,"user"): #如果g有user这个属性
        return {"user":g.user}
    else:
        return {}
```

## 装饰器

判断是否登录 , 如果没有登录，跳转到登录页面

一般这样：

if hasattr(g,'user'):

但是很多页面都需要登录，不可能大量重复的验证登录，我们要采用一些手段，    



这里可以通过装饰器实现

```
from flask import g,redirect ,url_for
from functools import wraps

def login_required(func):
    #这个装饰器是保留wrapper这个函数的一些属性
    @wraps(func)
    def warpper(*args , **kwargs):
        if hasattr(g,"user"):
            #验证成功，结束函数
            return func(*args,**kwargs)
        else:
            #验证不成功，跳转到登录页面
            return redirect(url_for("user.login"))
    return warpper
```

