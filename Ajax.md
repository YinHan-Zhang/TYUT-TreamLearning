# Ajax

>  AJAX 是一种用于创建快速动态网页的技术。
>
> 通过在后台与服务器进行少量数据交换，AJAX 可以使网页实现异步更新。这意味着可以在不重新加载整个网页的情况下，对网页的某部分进行更新。
>
> 最大的优势：无刷新获取数据

* 懒加载
* 搜索提示

XML：被设计用来传输和存储数据

XML和HTML类似，不同的是HTML中都是预定义标签，而XML中都是自定义标签



HTTP协议：

请求 -- >响应

![1644236510487](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644236510487.png)

![1644236731911](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644236731911.png)



发起Ajax请求的三个方法：

* $.get() :向服务器发送数据
* $.post() ：向服务器获取数据
* $.ajax() ：既可以发送也可以获取

![1644374421897](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644374421897.png)



## $.get()

```
$.get(url,[data],[callback])
```

![1644240931201](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644240931201.png)

* 可以发起不带参数的请求

![1644241013106](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644241013106.png)

```html
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="X-UA-Compatible" content="ie=edge">
  <title>Document</title>
  <script src="./lib/jquery.js"></script>
</head>

<body>
  <button id="btnGET">发起不带参数的GET请求</button>

  <script>
    $(function () {
      $('#btnGET').on('click', function () {
        $.get('http://www.liulongbin.top:3006/api/getbooks', function (res) {
          console.log(res)
           //res是响应形参，console.log(res)是打印响应
           //带参数的get请求
        $.get('http://www.liulongbin.top:3006/api/getbooks', { id: 1 }, function 				(res) {console.log(res)
        })
      })
    })
  </script>
</body>

</html>
```



## $.post()

```html
 <script>
    $(function () {
      $('#btnPOST').on('click', function () {
        $.post('http://www.liulongbin.top:3006/api/addbook', { bookname: '水浒传', author: '施耐庵', publisher: '天津图书出版社' }, function (res) {
          console.log(res)
        })
      })
    })
  </script>
</body>
```



## $.ajax()

![1644242818215](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644242818215.png)

![1644242890655](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644242890655.png)



```html
<body>
  <button id="btnGET">发起GET请求</button>

  <script>
    $(function () {
      $('#btnGET').on('click', function () {
        $.ajax({
          type: 'GET',
          url: 'http://www.liulongbin.top:3006/api/getbooks',
          data: {
            id: 1
          },
          success: function (res) {
            console.log(res)
          }
        })
      })
    })
  </script>
</body>
```





## XMLHttpRequest基本使用

 XMLHttpRequest 用于在后台与服务器交换数据。这意味着可以在不重新加载整个网页的情况下，对网页的某部分进行更新。 

基于xhr对象封装出来的

![Screenshot_20220209_101524_tv.danmaku.bilibilihd](D:%5Cpicture%5CScreenshot_20220209_101524_tv.danmaku.bilibilihd.jpg)

### 使用xhr对象发起Get请求：

> 步骤：
> 1.创建xhr对象
>
> 2.调用xhr.open()函数
>
> 3.调用xhr.send()函数
>
> 4.监听xhr.onreadystatechange事件



```C++
1.创建 XMLHttpRequest 对象的语法：

	variable=new XMLHttpRequest();

2.调用xhr.open()函数
	xhr.open('GET','http://liulongbin.top:3006/api/getbooks')

3.调用xhr.send()函数
	xhr.send()
4.监听xhr.onreadystatechange事件
	xhr.onreadystatechange = function()
	{
	
        //监听xhr对象的请求状态 readyState ; 与服务器响应的状态 status
        if(xhr.readyState == 4 && xhr.status ==200 )
        {
        	//打印服务器响应回来的数据
			console.log(xhr.responseText)

        }
	}
```

### xhr对象的readyState属性：

![Screenshot_20220209_103214_tv.danmaku.bilibilihd](D:%5Cpicture%5CScreenshot_20220209_103214_tv.danmaku.bilibilihd.jpg)



使用xhr对象发起带参数的GET请求，只需要在调用xhr.open()期间，为URL指定地址就好了

也就查询字符串

```
xhr.open('GET','http://www.liulongbin.top:3006/api/getbooks?id=1')
```

### 查询字符串

> 定义：查询字符串（URL 参数）是指在 URL 的末尾加上用于向服务器发送信息的字符串（变量）。
>
> 格式：将英文的 **?** 放在URL 的末尾，然后再加上 **参数＝值** ，想加上多个参数的话，使用 **&** 符号进行分隔。以这个形式，可以将想要发送给服务器的数据添加到 URL 中。



![1644374567885](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644374567885.png)

本质：

![1644374714888](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644374714888.png)



### URL编码

> URL 地址中，只允许出现英文相关的字母、标点符号、数字，因此，在 URL 地址中不允许出现中文字符。
>
> 如果 URL 中需要包含中文这样的字符，则必须对中文字符进行**编码**（转义）。
>
> **URL**编码的原则：
>
> ​	使用安全的字符（没有特殊用途或者特殊意义的可打印字符）去表示那些不安全的字符。
>
> URL编码原则的通俗理解：
>
> ​	使用英文字符去表示非英文字符。

![1644374836823](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644374836823.png)

如何进行编码？

浏览器提供了 URL 编码与解码的 API，分别是：

* encodeURI() 编码的函数

* decodeURI() 解码的函数

![1644374914561](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644374914561.png)

**由于浏览器会自动对 URL 地址进行编码操作，因此，大多数情况下，程序员不需要关心 URL 地址的编码与解码操作。**



### xhr创建post请求

①创建 xhr 对象

②调用 xhr.open() 函数

③**设置** **Content-Type** **属性**（固定写法）

④调用 xhr.send() 函数，**同时指定要发送的数据**

⑤监听 xhr.onreadystatechange 事件



```javascript
// 1. 创建 xhr 对象
var xhr = new XMLHttpRequest()

// 2. 调用 open()
xhr.open('POST', 'http://www.liulongbin.top:3006/api/addbook')

// 3. 设置 Content-Type 属性（固定写法）
xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded')

// 4. 调用 send()，同时将数据以查询字符串的形式，提交给服务器
xhr.send('bookname=水浒传&author=施耐庵&publisher=天津图书出版社')

// 5. 监听 onreadystatechange 事件
xhr.onreadystatechange = function() {
    if (xhr.readyState === 4 && xhr.status === 200) {
        console.log(xhr.responseText)
    }
}
```



### JSON

> 概念：JSON 的英文全称是 JavaScript Object Notation，即“JavaScript 对象表示法”。简单来讲，JSON 就是 Javascript 对象和数组的字符串表示法，它使用文本表示一个 JS 对象或数组的信息，
>
> 因此，**JSON** **的本质是字符串**。
>
> 作用：JSON 是一种轻量级的文本数据交换格式，在作用上类似于 XML，专门用于存储和传输数据，但是 JSON 比 XML 更小、更快、更易解析。

​     JSON 中包含**对象**和**数组**两种结构，通过这两种结构的相互嵌套，可以表示各种复杂的数据结构。  

对象结构：

> ​     数据结构为 { key: value, key: value, … } 的键值对结构。其中，key 必须是使用英文的双引号包裹的字符串，value 的数据类型可以是数字、字符串、布尔值、null、数组、对象6种类型。  

![1644375372746](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644375372746.png)

数组结构：

> ​     数组结构在 JSON 中表示为 [ ] 括起来的内容。数据结构为 [ "java", "javascript", 30, true … ] 。数组中数据的类型可以是数字、字符串、布尔值、null、数组、对象6种类型。  

![1644375403523](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644375403523.png)



#### 注意点：

①属性名必须使用双引号包裹

②字符串类型的值必须使用双引号包裹

③JSON 中不允许使用**单引号**表示字符串

④JSON 中不能写注释

⑤JSON 的最外层必须是对象或数组格式

⑥不能使用 undefined 或函数作为 JSON 的值



JSON 的作用：在计算机与网络之间存储和传输数据。

JSON 的本质：用字符串来表示 Javascript 对象数据或数组数据



与js对象的关系：

![1644375538737](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644375538737.png)

与js对象互转：

![1644375563870](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1644375563870.png)



### 封装Ajax函数



### FormData对象管理数据



```javascript
// 1. 新建 FormData 对象
var fd = new FormData()
// 2. 为 FormData 添加表单项
fd.append('uname', 'zs')
fd.append('upwd', '123456')
// 3. 创建 XHR 对象
var xhr = new XMLHttpRequest()
// 4. 指定请求类型与URL地址
xhr.open('POST', 'http://www.liulongbin.top:3006/api/formdata')
// 5. 直接提交 FormData 对象，这与提交网页表单的效果，完全一样
xhr.send(fd)

// 获取表单元素
 var form = document.querySelector('#form1')
 // 监听表单元素的 submit 事件
 form.addEventListener('submit', function(e) {
    e.preventDefault()
     // 根据 form 表单创建 FormData 对象，会自动将表单数据填充到 FormData 对象中
     var fd = new FormData(form)
     var xhr = new XMLHttpRequest()
     xhr.open('POST', 'http://www.liulongbin.top:3006/api/formdata')
     xhr.send(fd)
     xhr.onreadystatechange = function() {}
})

```



### 上传文件

实现步骤：

①定义 UI 结构

```
<!-- 1. 文件选择框 -->
    <input type="file" id="file1" />
    <!-- 2. 上传按钮 -->
    <button id="btnUpload">上传文件</button>
    <br />
    <!-- 3. 显示上传到服务器上的图片 -->
    <img src="" alt="" id="img" width="800" />

```

②验证是否选择了文件

```
// 1. 获取上传文件的按钮
 var btnUpload = document.querySelector('#btnUpload')
 // 2. 为按钮添加 click 事件监听
 btnUpload.addEventListener('click', function() {
     // 3. 获取到选择的文件列表
     var files = document.querySelector('#file1').files
     if (files.length <= 0) {
         return alert('请选择要上传的文件！')
     }
     // ...后续业务逻辑
 })

```

③向 FormData 中追加文件

​	

```
// 1. 创建 FormData 对象
 var fd = new FormData()
 // 2. 向 FormData 中追加文件
 fd.append('avatar', files[0])

```

④使用 xhr 发起上传文件的请求

​	

```
// 1. 创建 xhr 对象
 var xhr = new XMLHttpRequest()
 // 2. 调用 open 函数，指定请求类型与URL地址。其中，请求类型必须为 POST
 xhr.open('POST', 'http://www.liulongbin.top:3006/api/upload/avatar')
 // 3. 发起请求
 xhr.send(fd)

```

⑤监听 onreadystatechange 事件

```
xhr.onreadystatechange = function() {
  if (xhr.readyState === 4 && xhr.status === 200) {
    var data = JSON.parse(xhr.responseText)
    if (data.status === 200) { // 上传文件成功
      // 将服务器返回的图片地址，设置为 <img> 标签的 src 属性
      document.querySelector('#img').src = 'http://www.liulongbin.top:3006' + data.url
    } else { // 上传文件失败
      console.log(data.message)
    }
  }
}

```



完整代码

```html
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="X-UA-Compatible" content="ie=edge">
  <title>Document</title>
</head>

<body>
  <!-- 1. 文件选择框 -->
  <input type="file" id="file1" />
  <!-- 2. 上传文件的按钮 -->
  <button id="btnUpload">上传文件</button>
  <br />
  <!-- 3. img 标签，来显示上传成功以后的图片 -->
  <img src="" alt="" id="img" width="800" />

  <script>
    // 1. 获取到文件上传按钮
    var btnUpload = document.querySelector('#btnUpload')
    // 2. 为按钮绑定单击事件处理函数
    btnUpload.addEventListener('click', function () {
      // 3. 获取到用户选择的文件列表
      var files = document.querySelector('#file1').files
      if (files.length <= 0) {
        return alert('请选择要上传的文件！')
      }
      var fd = new FormData()
      // 将用户选择的文件，添加到 FormData 中
      fd.append('avatar', files[0])

      var xhr = new XMLHttpRequest()
      xhr.open('POST', 'http://www.liulongbin.top:3006/api/upload/avatar')
      xhr.send(fd)

      xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
          var data = JSON.parse(xhr.responseText)
          if (data.status === 200) {
            // 上传成功
            document.querySelector('#img').src = 'http://www.liulongbin.top:3006' + 					data.url
          } else {
            // 上传失败
            console.log('图片上传失败！' + data.message)
          }
        }
      }
    })
  </script>
</body>

</html>
```



### 进度条

```javascript
//导入的库
<link rel="stylesheet" href="./lib/bootstrap.css" />
    <script src="./lib/jquery.js"></script> 

//BootStrap进度条

    <div class="progress" style="width: 500px; margin: 10px 0;">
      <div class="progress-bar progress-bar-info progress-bar-striped active" id="percent" style="width: 0%">
        0%
      </div>
    </div>



// 创建 XHR 对象
var xhr = new XMLHttpRequest()

// 监听 xhr.upload 的 onprogress 事件
xhr.upload.onprogress = function(e) {
    // e.lengthComputable 是一个布尔值，表示当前上传的资源是否具有可计算的长度
    if (e.lengthComputable) {
        // e.loaded 已传输的字节
        // e.total  需传输的总字节
        var percentComplete = Math.ceil((e.loaded / e.total) * 100)  //计算上传进度
    }
 }

//监听上传的文件
xhr.upload.onload = function() {
     $('#percent')
         // 移除上传中的类样式
         .removeClass()
         // 添加上传完成的类样式
         .addClass('progress-bar progress-bar-success')
 }



```

完整代码：

```html
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="X-UA-Compatible" content="ie=edge">
  <title>Document</title>
  <link rel="stylesheet" href="./lib/bootstrap.css" />
  <script src="./lib/jquery.js"></script>
</head>

<body>
  <!-- 1. 文件选择框 -->
  <input type="file" id="file1" />
  <!-- 2. 上传文件的按钮 -->
  <button id="btnUpload">上传文件</button>

  <!-- bootstrap 中的进度条 -->
  <div class="progress" style="width: 500px; margin: 15px 10px;">
    <div class="progress-bar progress-bar-striped active" style="width: 0%" id="percent">
      0%
    </div>
  </div>

  <br />
  <!-- 3. img 标签，来显示上传成功以后的图片 -->
  <img src="" alt="" id="img" width="800" />

  <script>
    // 1. 获取到文件上传按钮
    var btnUpload = document.querySelector('#btnUpload')
    // 2. 为按钮绑定单击事件处理函数
    btnUpload.addEventListener('click', function () {
      // 3. 获取到用户选择的文件列表
      var files = document.querySelector('#file1').files
      if (files.length <= 0) {
        return alert('请选择要上传的文件！')
      }
      var fd = new FormData()
      // 将用户选择的文件，添加到 FormData 中
      fd.append('avatar', files[0])

      var xhr = new XMLHttpRequest()

      // 监听文件上传的进度
      xhr.upload.onprogress = function (e) {
        if (e.lengthComputable) {
          // 计算出上传的进度
          var procentComplete = Math.ceil((e.loaded / e.total) * 100)
          console.log(procentComplete)
          // 动态设置进度条
          $('#percent').attr('style', 'width: ' + procentComplete + '%;').html(procentComplete + '%')
        }
      }

      xhr.upload.onload = function () {
        $('#percent').removeClass().addClass('progress-bar progress-bar-success')
      }

      xhr.open('POST', 'http://www.liulongbin.top:3006/api/upload/avatar')
      xhr.send(fd)

      xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
          var data = JSON.parse(xhr.responseText)
          if (data.status === 200) {
            // 上传成功
            document.querySelector('#img').src = 'http://www.liulongbin.top:3006' + data.url
          } else {
            // 上传失败
            console.log('图片上传失败！' + data.message)
          }
        }
      }
    })
  </script>
</body>

</html>
```



### 使用ajax上传文件

```javascript
<!-- 导入 jQuery -->
    <script src="./lib/jquery.js"></script>

    <!-- 文件选择框 -->
    <input type="file" id="file1" />
    <!-- 上传文件按钮 -->
    <button id="btnUpload">上传</button>

$('#btnUpload').on('click', function() {
     // 1. 将 jQuery 对象转化为 DOM 对象，并获取选中的文件列表
     var files = $('#file1')[0].files
     // 2. 判断是否选择了文件
     if (files.length <= 0) {
         return alert('请选择图片后再上传！‘)
     }
 })

// 向 FormData 中追加文件
 var fd = new FormData()
 fd.append('avatar', files[0])

$.ajax({
     method: 'POST',
     url: 'http://www.liulongbin.top:3006/api/upload/avatar',
     data: fd,
     // 不修改 Content-Type 属性，使用 FormData 默认的 Content-Type 值
     contentType: false,
     // 不对 FormData 中的数据进行 url 编码，而是将 FormData 数据原样发送到服务器
     processData: false,
     success: function(res) {
         console.log(res)
     }
 })
    
// 自 jQuery 版本 1.8 起，该方法只能被附加到文档
 $(document).ajaxStart(function() {
     $('#loading').show()
 })

    注意： $(document).ajaxStart() 函数会监听当前文档内所有的 Ajax 请求。
	Ajax 请求结束时，执行 ajaxStop 函数。可以在 ajaxStop 的 callback 中隐藏 loading 效果

 // 自 jQuery 版本 1.8 起，该方法只能被附加到文档
 $(document).ajaxStop(function() {
     $('#loading').hide()
 })

```

完整代码：

```html
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="X-UA-Compatible" content="ie=edge">
  <title>Document</title>
  <script src="./lib/jquery.js"></script>
</head>

<body>

  <input type="file" id="file1" />
  <button id="btnUpload">上传文件</button>

  <br />
  <img src="./images/loading.gif" alt="" style="display: none;" id="loading" />

  <script>
    $(function () {
      // 监听到Ajax请求被发起了
      $(document).ajaxStart(function () {
        $('#loading').show()
      })

      // 监听到 Ajax 完成的事件
      $(document).ajaxStop(function () {
        $('#loading').hide()
      })

      $('#btnUpload').on('click', function () {
        var files = $('#file1')[0].files
        if (files.length <= 0) {
          return alert('请选择文件后再上传！')
        }

        var fd = new FormData()
        fd.append('avatar', files[0])

        // 发起 jQuery 的 Ajax 请求，上传文件
        $.ajax({
          method: 'POST',
          url: 'http://www.liulongbin.top:3006/api/upload/avatar',
          data: fd,
          processData: false,
          contentType: false,
          success: function (res) {
            console.log(res)
          }
        })
      })
    })
  </script>

</body>

</html>
```

