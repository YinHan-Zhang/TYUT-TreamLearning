

# Selenium简单使用 



> 自动化办公、龟速收集数据、日常签到，打卡等(不便做具体展示)



## 1. 安装



**chromedriver 下载链接**

|   浏览器    |                        webdriver链接                         |
| :---------: | :----------------------------------------------------------: |
| **Chrome**  |         http://npm.taobao.org/mirrors/chromedriver/          |
| **Firefox** |       https://github.com/mozilla/geckodriver/releases        |
|  **Edge**   | https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/ |



两种方法：

1. 将webdriver放到**python**运行环境下的**scripts**文件夹
2. 执行selenium时候指定webdriver的路径（后面再展示



---

python环境，自行解决

> pip install selenium -i https://pypi.douban.com/simple



## 2. 基础操作

> 最基础的操作是 定位！
>
> 输入信息、点击、等操作都需要先进行定位。
>
> 而定位、输入、点击，这三个操作加起来，可以覆盖大多数的操作了。



> [**文章：Selenium 常用方法(定位、点击。。。)**](https://blog.csdn.net/weixin_45081575/article/details/112306723)



### 初始化代码

```python
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class Demo:
    url = "https://www.baidu.com"

    def __init__(self):
        # 打开浏览器
        self.browser = webdriver.Chrome()
        # 显式等待
        self.wait = WebDriverWait(self.browser, timeout=20)

    def __del__(self):
        self.browser.quit()

    def visit_website(self):
        """
        访问网站
        :return:
        """
        self.browser.get(self.url)

    def main(self):
        self.visit_website()
        time.sleep(10)


if __name__ == '__main__':
    demo = Demo()
    demo.main()

```



### 八大定位方法

> https://www.baidu.com

1. By.ld: 标签中具有id这个属性
2. By.xpath():根据xpath定位
3. By.Name(): 标签中具有name这个属性
4. By.tagName():标签本身，比如<p>，<input>
5. By.className():标签中具有class属性
6. By.CssSelector():通过css选择器
7. By.linktext():  通过标签中的文字定位
8. By.partialLinkText():根据部分文字定位

**xpath定位**

```python
# method one
self.browser.find_element_by_name("tj_briicon")
# method two
self.wait.until(EC.presence_of_element_located((By.NAME, "tj_briicon")))
```



**文字定位**

```python
# method one
self.browser.find_element_by_xpath('//a[contains(text(), "更多")]')
# method two
self.wait.until(EC.presence_of_element_located((By.XPATH, '//a[contains(text(), "更多")]')))
```



- 辅助定位（用于判断界面节点存在与否

```python
# text_to_be_present_in_element
self.wait.until(EC.text_to_be_present_in_element((By.NAME, "tj_briicon"), "更多"))
```



**文字+xpath**

```python
# method one
self.browser.find_element_by_xpath('//a[contains(text(), "更多") and @name="tj_briicon"]')
# method two
self.wait.until(EC.presence_of_element_located((By.XPATH, '//a[contains(text(), "更多") and @name="tj_briicon"]')))
```



---



### 输入

先清空，再输入

**.clear()**

**.send_keys()**



修改源码，貌似不得行



---



### 点击

[**看点击**](#鼠标操作)  			click()

大部分的点击，在 **可点击节点** 后面加一个 **.click()** 即可进行点击。

剩下的极少数部分节点，可以是用以下的几种方法进行点击。



**方法一：**

先获取节点，例如 **node= wait.until(EC.element_to_be_clickable)** 参照 定位节点
然后调用 Selenium 执行 Js 的函数进行 点击。

```python
node = self.wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "title-text")))
self.browser.execute_script("arguments[0].click();", node)　# node.click()
# arguments[1]第2个参数的意思, click() 是浏览器点击操作
```

**方法二：**

先获取节点，例如 **node= wait.until(EC.element_to_be_clickable)** 参照 定位节点
然后调用鼠标事件，进行点击。

```python
node = self.wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "title-text")))

ActionChains(self.browser).click(node).perform()
# 或者
ActionChains(self.browser).move_to_element(node).click().perform()
```

**方法三：**

先获取需要点击的节点的 坐标，即相对于 Selenium 的x轴 和 y轴。
然后调用鼠标事件，进行点击。这个操作有点秀逗，常常失败。

- **xoffset** 和 **yoffset** 分别为节点坐标的 x 和 y

```python
# 定位节点，从而得知其在浏览器中的坐标
ActionChains(self.browser).move_by_offset(xoffset=x, yoffset=y).click().perform()	# 490 44
ActionChains(self.browser).release().perform() # 可选
```



---



### 下拉框



>  **from selenium.webdriver.support.select import Select**

。。。

```python
self.browser.get(r"./demo.html")
node = self.wait.until(EC.presence_of_element_located((By.NAME, "Mobiles")))
Select(node).select_by_visible_text("iPhone")
```



### 鼠标操作

> **from selenium.webdriver import ActionChains**
>
> https://www.baidu.com



**左键单击** 	等同于 **node.click()**

看  [**切换窗口**](#切换window)

```python
# 百度热点 节点
node = self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "title-text")))
ActionChains(self.browser).click(node).perform()
```



**右键单击**

```python
# 百度热点 节点
node = self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "title-text")))
ActionChains(self.browser).context_click(node).perform()
```

**PyAutoGUI**    ---   用程序自动控制鼠标和键盘操作

> **import pyautogui**

```python
pyautogui.typewrite(['down'])
time.sleep(2)
pyautogui.typewrite(['enter'])

node = self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "title-text")))
ActionChains(self.browser).context_click(node).perform()
pyautogui.typewrite(["down"])
time.sleep(2)
pyautogui.typewrite(["enter"])
time.sleep(10)
```



**鼠标悬停**

```python
# 设置节点
setting_node = self.wait.until(EC.presence_of_element_located((By.ID, "s-usersetting-top")))
# 移动到某个元素(悬停
ActionChains(self.browser).move_to_element(setting_node).perform()
# 移动到某个坐标
ActionChains(self.browser).move_by_offset(0, 200).perform()


# 点击 搜索设置
# self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "setpref"))).click()
```



**鼠标拖动**

```python
# https://passport.5173.com/
# 注意看 data-mex
# click_and_hold(element) 之后记得要 释放鼠标  release()

node = self.wait.until(EC.presence_of_element_located((By.ID, "pub_slider_btn")))
ActionChains(self.browser).click_and_hold(node).move_by_offset(290, 0).release().perform()
```

滑块过不了的解决方法  ：[**控制当前已经打开的浏览器**](#5. 控制当前已经打开的浏览器)



**左键双击**

```python
# http://sahitest.com/demo/clicks.htm

click_button = self.wait.until(EC.presence_of_element_located((By.XPATH, "//input[@value='click me']")))
ActionChains(self.browser).double_click(click_button).perform()
```



---



### 键盘操作

> **from selenium.webdriver.common.keys import Keys**



```python
# http://sahitest.com/demo/clicks.htm

# 全选
self.wait.until(EC.presence_of_element_located((By.NAME, "t2"))).send_keys(Keys.CONTROL, "a")
# 剪切
self.wait.until(EC.presence_of_element_located((By.NAME, "t2"))).send_keys(Keys.CONTROL, "x")
# 黏贴
self.wait.until(EC.presence_of_element_located((By.NAME, "t2"))).send_keys(Keys.CONTROL, "v")

# 回车键 "https://www.baidu.com"
Element.send_keys(Keys.ENTER)
```



### 窗口操作



#### 滑动页面

> [**Window scrollTo()：https://www.runoob.com/jsref/met-win-scrollto.html** ](https://www.runoob.com/jsref/met-win-scrollto.html)

 

```python
# method one
browser.execute_script('window.scrollTo(0, document.body.scrollHeight)')	
#document.body.scrollHeight 浏览器所有内容的高度
#让页面滚动到下面,window.scrollBy(0, scrollStep)  window.scrollBy(0,3000)



# method two
browser.execute_script("arguments[0].scrollIntoView();", node)	# 滑动到节点位置
```



#### 打开新窗口

```python
open_window= 'window.open("https://www.bilibili.com")'
self.browser.execute_script(open_window)
```



#### 切换窗口

```python
# 获取当前窗口句柄
self.browser.current_window_handle
# 获取所有窗口句柄
self.browser.window_handles
# 切换窗口
self.browser.switch_to.window(window_handle)
```



#### 切换frame

Selenium 打开页面后，默认是在父级 frame 里面操作，**如果页面中还有子 frame，Selenium 是不能操作子 frame节点的。**

这时就需要使用 switch_to.frame 方法来切换 frame。



```python
def frame(self, frame_reference):
    """
    Switches focus to the specified frame, by index, name, or webelement.

    :Args:
     - frame_reference: The name of the window to switch to, an integer representing the index,
                        or a webelement that is an (i)frame to switch to.

    :Usage:
        driver.switch_to.frame('frame_name')
        driver.switch_to.frame(1)
        driver.switch_to.frame(driver.find_elements_by_tag_name("iframe")[0])
    """
    if isinstance(frame_reference, basestring) and self._driver.w3c:
        try:
            frame_reference = self._driver.find_element(By.ID, frame_reference)
        except NoSuchElementException:
            try:
                frame_reference = self._driver.find_element(By.NAME, frame_reference)
            except NoSuchElementException:
                raise NoSuchFrameException(frame_reference)

    self._driver.execute(Command.SWITCH_TO_FRAME, {'id': frame_reference})

```



```python
# https://music.163.com/

# 切换子框架 可以是索引，name，id等
self.browser.switch_to.frame()
# 切换到父框架
self.browser.switch_to.parent_frame()
# 切换到 默认框架
self.browser.switch_to.default_content()
```



#### 处理弹窗

**alert**

| 方法              | 释义                                           |
| ----------------- | ---------------------------------------------- |
| switch_to.alert() | 切换到 alert                                   |
| text              | 获取 alert 文本内容                            |
| accept()          | 确认                                           |
| dismiss()         | 取消                                           |
| send_keys()       | 有输入框才能使用，否则报错(经测试，无用！！！) |



```python
# 执行弹窗语句
js_code = 'var name = confirm("Please enter your name")'

# 执行Js 代码
self.browser.execute_script(js_code)
# 获取alert 文本
self.browser.switch_to.alert.text
# 点击确定
self.browser.switch_to.alert.accept()
# 点击取消
self.browser.switch_to.alert.dismiss()
```



#### 关闭窗口

```python
# 关闭窗口
self.browser.close()
# 退出浏览器
self.browser.quit()
```



## 3. 常用操作



获取网页截图

```python
browser.save_screenshot('xxx.jpg/png')
```

获取cookie

```python
browser.get_cookies()
```

获取源码

```python
browser.page_source
```



| 浏览器方法                                 |       作用       |
| :----------------------------------------- | :--------------: |
| **browser.get(url)**                       |   **请求url**    |
| **browser.quit()**                         |  **退出浏览器**  |
| **browser.close()**                        | **关闭当前页面** |
| **browser.page_source**                    |   **获取源码**   |
| **browser.page_source.find('keyword')**    | **在源码中查找** |
| **browser.maximize_window() **             |  **窗口最大化**  |
| **browser.set_window_size(width, height)** | **设置窗口大小** |
| **browser.get_cookies()**                  |  **获取cookie**  |
| **browser.current_url**                    |   **获取url**    |
| **browser.execute_script(js_code)**        |  **运行js代码**  |



**节点操作：**

| **节点操作**                           | **作用**         |
| -------------------------------------- | ---------------- |
| node.send_keys('something')            | 在文本框填写内容 |
| button.click()                         | 点击             |
| node.get_attribute('href / id / name') | 获取节点属性     |
| node.text                              | 获取节点文本     |



**无头模式**

```python
from selenium.webdriver.chrome.options import Options
options = Options()
# 为Chrome配置无头模式
options.add_argument("--headless")
browser = webdriver.Chrome(options=options)


browser.get("https://www.bilibili.com")
time.sleep(3)
browser.save_screenshot("demo.png")
```



**隐式等待 和 显式等待**

- 在使用Selenium访问网页的过程中，有时候需要等待网页的加载，所以有时候就需要延时等待一定的时间，确保网页里你需要的内容都加载出来。
- 指定要查找的节点，再指定一个最长等待时间。如果在指定时间内加载出来了这个节点，就返回查找的节点；如果到了指定时间没有加载出该节点，则抛出超时异常。



## 4. 小案例



### QQmail

> 定位、输入、点击
>
> 显式等待、切换frame、



```python
# -*- coding: utf-8 -*-
# @Time     : 2021-09-29  15:45 
# @File     : qq_email_login.py
# @software : PyCharm

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class LoginEmail:
    url_ = "https://mail.qq.com/"

    def __init__(self):
        # 打开浏览器
        self.browser = webdriver.Chrome()
        # 显示等待
        self.wait = WebDriverWait(self.browser, timeout=20)

    def __del__(self):
        # 退出
        self.browser.close()

    def visit_website(self):
        """
        访问网页
        @return:
        """
        self.browser.get(self.url_)

    def login_operation(self):
        """
        登录操作，定位、输入信息、点击
        @return:
        """
        self.switch_frame(frame_name="login_frame")
        uname_edittext = self.wait.until(EC.presence_of_element_located((By.XPATH, "//input[@type='text']")))
        passwd_edittext = self.wait.until(EC.presence_of_element_located((By.XPATH, "//input[@type='password']")))
        login_button = self.wait.until(EC.presence_of_element_located((By.XPATH, "//input[@type='submit']")))
        uname_edittext.send_keys("QQ账号")
        passwd_edittext.send_keys("QQ密码")
        login_button.click()

    def write_mail(self):
        """
        写信
        @return:
        """
        xpath_map = {
            "write_mail": "//a[@id='composebtn']",
            "addressee": "//*[@id='toAreaCtrl']//input",
            "body": "//*[@class='qmEditorIfrmEditArea']",
            "subject": "//*[@id='subject']",
            "file": "//input[@type='file']",
            "filepath": r"E:\test\selenium_case\qq_email_login.py",
            "send": "//a[@name='sendbtn']"
        }
        # 点击 写信
        write_mail_node = self.wait.until(EC.presence_of_element_located((By.XPATH, xpath_map["write_mail"])))
        write_mail_node.click()
        self.switch_frame(frame_name="mainFrame")  # 切换frame
        # 收件人
        addressee_node = self.wait.until(EC.presence_of_element_located((By.XPATH, xpath_map["addressee"])))
        addressee_node.send_keys("123456789@qq.com")
        # 主题
        subject_node = self.wait.until(EC.presence_of_element_located((By.XPATH, xpath_map["subject"])))
        subject_node.send_keys("This is demo.")
        # 添加附件
        file_node = self.wait.until(EC.presence_of_element_located((By.XPATH, xpath_map["file"])))
        file_node.send_keys(xpath_map["filepath"])
        # 切换frame
        body_frame = self.wait.until(EC.presence_of_element_located((By.XPATH, xpath_map["body"])))
        self.switch_frame(frame_name=body_frame)
        # 正文
        body_edittext = self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        body_edittext.send_keys("Hi, This is demo.")
        self.browser.switch_to.parent_frame()  # 切换frame
        # 发送
        send_button = self.wait.until(EC.presence_of_element_located((By.XPATH, xpath_map["send"])))
        send_button.click()

    def switch_window(self):
        """
        切换window窗口
        @return:
        """
        pass

    def switch_frame(self, frame_name):
        """
        切换frame
        @return:
        """
        self.browser.switch_to.frame(frame_name)

    def main(self):
        self.visit_website()
        self.login_operation()
        self.write_mail()
        time.sleep(10)


if __name__ == '__main__':
    le = LoginEmail()
    le.main()

```



### JD

> 解放双手，龟速采集数据下来慢慢看



```python
# -*- coding:utf-8 -*-
# datetime  :   2021-10-10 11:21
# filename  :   JdGoods
# software  :   PyCharm

import os
import csv
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class GetJdData:
    url = "https://www.jd.com"

    def __init__(self):
        # 打开浏览器
        self.browser = webdriver.Chrome()
        # 显示等待
        self.wait = WebDriverWait(self.browser, timeout=20)

    def __del__(self):
        self.browser.close()

    def visit_website(self):
        """
        访问网站
        :return:
        """
        self.browser.get(self.url)

    def search_goods(self):
        """
        搜索商品
        :return:
        """
        # 输入商品
        self.wait.until(EC.presence_of_element_located((By.XPATH, "//input[@type='text']"))).send_keys("华为")
        # 回车
        self.wait.until(EC.presence_of_element_located((By.XPATH, "//input[@type='text']"))).send_keys(Keys.ENTER)
        time.sleep(5)
        while True:
            # 滑动页面
            self.browser.execute_script('window.scrollTo(0, document.body.scrollHeight)')
            # 等待页面加载更多数据
            time.sleep(5)
            # 解析数据
            self.parse_html()
            # 下一页 节点
            next_page_node = self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "pn-next")))
            if "disable" in next_page_node.get_attribute('class'):
                break
            next_page_node.click()
            time.sleep(5)

    def parse_html(self):
        data = list()
        nodes = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, "//li[@class='gl-item']")))
        for node in nodes:
            link = node.find_element_by_xpath(".//div[@class='p-img']/a").get_attribute('href')
            price = node.find_element_by_xpath(".//div[@class='p-price']//i").text
            name = node.find_element_by_xpath(".//div[@class='p-name p-name-type-2']//em").text.replace("\n", ',')
            comment = node.find_element_by_xpath(".//div[@class='p-commit']//a").text
            shop = node.find_element_by_xpath(".//div[@class='p-shop']//a").text
            tag = node.find_element_by_xpath(".//div[@class='p-icons']").text.replace("\n", ',')
            data.append((link, price, name, comment, shop, tag))
        save_to_csv(data, "goodsInfo")

    def main(self):
        self.visit_website()
        self.search_goods()


def save_to_csv(data: list, filename: str):
    if os.path.exists(f"{filename}.csv"):
        with open(f"{filename}.csv", mode='a', newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(data)
    else:
        with open(f"{filename}.csv", mode='w', newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["链接", "价格", "商品名称", "评论人数", "商店", "标签"])
            writer.writerows(data)


if __name__ == '__main__':
    gjd = GetJdData()
    gjd.main()

```



### HR打卡

```
无
```



## 5. 控制当前已经打开的浏览器

> [**参考文章：【Selenium】控制当前已经打开的 chrome浏览器窗口**](https://blog.csdn.net/weixin_45081575/article/details/112621581)
>
> 登录的步骤太难时候，可以选择这种方法。



**步骤如下：**

1. 来到浏览器安装的目录下，打开cmd窗口，并输入以下代码

   ```bash
   chrome.exe --remote-debugging-port=9527 --user-data-dir=“F:\folderpath”
   ```

   - **user-data-dirr=“F:\folderpath”** 是在单独的配置文件中启动 **chrome浏览器**，可以理解为 新的浏览器，记得创建对应文件夹哦；
   - 其中 **9527** 为端口号，可自行指定。

2. 接下的步骤与正常的操作一样了，只需要添加这两句代码

   ```python
   options = Options()
   options.add_experimental_option("debuggerAddress", "127.0.0.1:9527")
   ```

   

**代码部分：**

```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_experimental_option("debuggerAddress", "127.0.0.1:9527")
browser = webdriver.Chrome(options=options)

print(browser.title)
```



**登录B站**

```
~~~
```



完结，撒花~



## 补充

**隐藏Selenium特征文章**

https://mp.weixin.qq.com/s/XOXb_XvsHqgv0MUICahjJw

https://mp.weixin.qq.com/s/U45x8HCPpNe-22LRtXYaNQ

