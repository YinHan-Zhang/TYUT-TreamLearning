# HTML基础

## HTML网页的大致结构

```html
<html>
<head>
    <meta charset="UTF-8">
    <title>
        <div> 
        </div>
    </title>
</head>    
<body>
    <h1>...</h1>
    <ol>
        <li>...</li>
    </ol>
    <h2>...</h2>
    <ol>
        <li>...</li>
    </ol>
</body>
<html>
```

| 位置标签         | 说明                                                         |
| ---------------- | ------------------------------------------------------------ |
| < div >          | 把一些独立的逻辑部分划分出来，比如网页中独立的栏目板块，（用框框框起来的） |
| header           | 定义头部标签                                                 |
| < h >            | 阶梯层级标签                                                 |
| < span >         | 特殊强调标签                                                 |
| < p >            | 文本内容标签                                                 |
| < !--注释文字--> | 写注释                                                       |
| < footer >       | 底部标签                                                     |
| < section  >     | 定义一个区域，但是作用等同div                                |
| < aside >        | 我是侧边栏                                                   |

#### 效果标签：

让显示更美观

**在html代码中输入空格、回车都是没用的**

| 效果标签                                     | 说明                                         |
| -------------------------------------------- | -------------------------------------------- |
| < br >                                       | 相当于回车（无结束标签）                     |
| &nbsp ;                                      | 相当于空格                                   |
| < hr >                                       | 水平分割线（无结束标签）                     |
| <ul> <li>                                    | 无序列表标签                                 |
| <ol> <li>                                    | 有序列表标签                                 |
| < img src=“ ”alt=“ ” title=“ ”>              | 图片地址、下载失败时的替换文本、鼠标提示文本 |
| < a href="目标网址" title="提示文本"> < /a > | 实现超链接                                   |
| < a href= “” target=“ _self/ _blank”         | 当前页面打开超链接还是新窗口                 |

#### table标签

| < table border=“1” >          | 表格开始,加表格线        |
| ----------------------------- | ------------------------ |
| < tr >                        | 表格的行                 |
| < th >                        | 表格头部单元格           |
| < td >                        | 表格的列                 |
| < thead > < tbody > < tfoot > | 有框框包裹的表头表身表尾 |

#### 表单标签

表单就是把浏览者输入的数据传送到服务器端，这样服务器端就可以处理表单传过来的数据

登录页面例子：

<form    method="post"   action="save.php">
        <label for="username">用户名:</label>
        <input type="text" name="username" />
        <label for="pass">密码:</label>
        <input type="password" name="pass" />
</form>

```html
<form    method="post"   action="save.php">
        <label for="username">用户名:</label>
        <input type="text" name="username" />
        <label for="pass">密码:</label>
        <input type="password" name="pass" />
</form>
```

语法：

```html
<from> method="传送方式" action="服务器文件"</from>

action: 浏览者输入的数据被传送到的地方，比如一个PHP页面

method:数据传送的方式(get/post)

```

input标签：

```html
用户名、密码 -- 文本输入框、密码输入框：
<input type="text/password" name="名称" value="文本" />

1.type:
	type="text": 文本输入框
	type="password":密码输入框
2.name:为文本框命名
3.value:为文本输入框设置默认值(提示作用)

<input -placeholder :输入框提示符>
例：<input type="text" placeholder="请输入提示关键字">

数字输入框：
<input type="number">

网址输入框：
<input type="url">

邮箱输入框：
<input type="email">

文本输入域：
<textarea rows="行数" cols="列数">文本</textarea>
1.cols:多行输入的列数
2.rows:多行输入的行数
```

#### label标签：

```html
<label for="控件id名称"> 输入你的...</label>

注意：for的id名称要和接下来的input的type符合
```

#### 单选框、复选框：

在设计调查表时，用户选择框

```HTML
<input   type="radio/checkbox"   value="值"    name="名称"   checked="checked"/>
1.type:
	"radio":单选框
	"checkbox":复选框
2.value:提交数据到服务器的值
3.name: 为空间命名
4.checked="checked": 该选项被默认选中


```

#### 下拉菜单：

```html
select标签里放option标签，表示下拉列表的选项
option标签放选项内容，不放置其他标签
    <form>
        <select>
            <option value="看书">看书</option>
            <option value="旅游">旅游</option>
            <option value="运动">运动</option>
            <option value="购物">购物</option>
        </select>
    </form>
value:向服务器提交的值
selected="selected" : 默认选中

```

#### 按钮：

```html
提交按钮：
<input type="submit" value="提交">
values:按钮上显示的文字

<input tyoe="reset" value="重置”>
                           
```

## CSS3：

> 什么是CSS？ 它的主要作用是用于定义HTML内容在浏览器内的显示样式，如文字大小、颜色、字体加粗等。

定义标签的显示风格：

在head标签下，style标签中定义你要自定义的显示风格的标签

```html
<style type="text/css">
    span{
        color:red/blue;
    }
</style>
```

- 代码语法：

![1635812495811](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1635812495811.png)

- CSS注释：/* ... */

***

### CSS样式：

#### 内联式：

> 就是把css代码直接写在现有的HTML标签里面

```html
单条：
<p style="color:red">
    这里文字是红色。
</p>

多条：要写在一个引号里，而且用分号隔开
<p style="color:red ; font-size:12px">
    这里文字是红色
</p>
```

#### 嵌入式：

> 每一个标签都使用内嵌式要累死啊！ 所以要统一写 **注意写在head里面**

```html
<style type="text/css">
    span{
        font-size:12px; 设置文字大小
        color:red; 		文字颜色
        font-weight:bold; 字体加粗
    }
</style>
```

#### 外部样式：

> 单独写一个css的代码文件以`.css`为扩展名，在head标签内，使用<link>标签把css文件链接到HTML文件中
>
> **说白了就是导入头文件的意思**

```html
<link href="style.css" rel="stylesheet" type="text/css"
      这行代码一个属性都不能掉
```

**优先级顺序**： 内联式>嵌入式>外部式 		`就近原则`

---

### CSS选择器

> - 标签选择器：<html>  <body>  <h1> <p> <img>
>
> - 类选择器： `.类选择器名称{css样式代码}`   **在head标签里面用**
>
>   1.英文圆点开头（代表属性选择）
>
>   2.类选择器可以任意起名称，除了中文
>
>   造一个标签：``<span class="stress">胆小如鼠<span>``
>
>   选择：``.stress{color:red}``
>
> - id选择器：``#id名{color:red}``   **使用前加#号**
>
>   1.id=“id名称”
>
>   2.id名称在HTML中只能出现一次，独一无二！！！
>
>   ---
>
> 类选择器可以对一个元素进行多重修饰  
>
> ```html
> .stress{
>     color:red;
> }
> .bigsize{
>     font-size:25px;
> }
> <p>到了<span class="stress bigsize">三年级</span>下学期时，我们班上了一节公开课...</p>
> 
> ```
>
> **id选择器不可以，以下代码式错误的**
>
> ```html
> #stressid{
>     color:red;
> }
> #bigsizeid{
>     font-size:25px;
> }
> <p>到了<span id="stressid bigsizeid">三年级</span>下学期时，我们班上了一节公开课...</p>
> ```
>
> ---
>
> - 子选择器：修饰父标签下的子标签 ``.food>li{...}``
>
>   其实和标签的层级选择一样  >层级 * 全部 .属性
>
> - 分组选择：就是俩标签都是一样修饰样式，直接组合到一起
>
>   ``h1,span{color:red}``    =  ``h1{color:red} span{color:red} ``
>
>     

![1644398447726](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644398447726.png)



![1644398634063](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644398634063.png)



![1644398660567](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644398660567.png)



![1644399257749](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644399257749.png)





![1644398762163](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644398762163.png)