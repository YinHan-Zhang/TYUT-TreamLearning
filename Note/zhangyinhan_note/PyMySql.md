# PyMysql学习

## 操作流程：

1. 连接数据库
2. 建立cursor游标
3. 输入sql语句
4. execut语句执行
5. fetchone/fetchall 提取数据库内容  
6. commit 更新 / rollback 回滚
7. close关闭数据库

```python
import pymysql 
db = pymysql.connect(host='localhost',user ='root',password='20020520zyh',database='db1')

#获取游标
cursor = db.cursor()
sql = """select * from hero"""
try :
    cursor.execute(sql)
    #提交到数据库
    db.commit()
except:
    #如果发生错误就回滚
    db.rollback()

#关闭数据库
db.close()
```

## 属性详解

| 游标类型     | 描述                                             |
| ------------ | :----------------------------------------------- |
| Cursor       | 普通的游标对象，默认创建的游标对象，元组返回     |
| SSCursor     | 不缓存游标，主要用于当操作需要大量数据返回的时候 |
| DictCursor   | 以字典的形式返回操作结果                         |
| SSDictCursor | 不缓存游标，将结果以字典的形式返回               |



## 数据库查询操作

* fetchone(): 获取单条数据
* fetchone(): 获取多条数据
* rowcount:这是一个只读属性，返回执行execute（）方法后影响的行数

```python
db = pymysql.connect(host='localhost',user ='root',password='20020520zyh',database='db1')
cursor = db.cursor()
sql ="select * from hero"
try :
    cursor.execute(sql)
    results = cursor.fetchall()
    for row in  results :
        id = row[0]
        name = row[1]
        age = row[2]
        skill = row[3]
        grade =row[4]
        print(f"id={id},name={name},age={age},skill={skill},grade={grade}")
except:
    print('Error:unable to fetch data!')
db.close()
```

插入更新删除操作，输入sql语句，**多了一步db.commit()**

### 方法实际运用：

写成函数，写执行语句较简单

```python
db = pymysql.connect(host='localhost',user ='root',password='20020520zyh',database='db1')
cursor = db.cursor()

def doSql(sql):
    cursor.execute(sql)
    cursor.commit()
    
doSql('create database DOUBAN if not exists onelinelearning;')

sql='''
create table if not exists questions(表的名字)(
id INT auto_increment PRIMARY KEY,
wenti CHAR(200) NOT NULL UNIQUE,
daan CHAR(50) NOT NULL
) ENGINE= innodb DEFAULT CHARSET=UTF8MB4;
'''
```



#### 1.执行多个查询命令 cursor.executemany(query ,args)

query : 要在服务器上执行的查询语句

args : 序列，用它做参数

作用：提高插入和更新的性能 ，等同于使用execute（）循环遍历args

```python
str_insert = "INSERT INTO Cu_table (id) VALUES (%s)"
cursor.executemany(str_insert,['A','B','C','D']) #一次插入A B C D 四个值
```

#### 2.获取多行结果（非全部）cursor.fetchmany(size=1)

```python
str_insert = "SELECT * FROM Cu_table"
cursor.execute(str_insert)
result = cursor.fetchmany(3) # 获取前三行
```

* 迭代获取结果：可操作性强  cursor ._ _ iter _ _  创建一个可迭代对象

  ```
  cursor.execute("SELECT * FROM new_futures.m1809")
  it=cursor.__iter__()
  it.__next__()#获取结果集的下一行返回元祖
  
  pd.DataFrame(list(it))
  ```

#### 3.mogrify( query, args=None ): 转换sql语句

作用：

1. ​      转换，把SQL语句和参数相结合，对其中特殊字符进行加\转义，
2. ​      然后返回一个execute()方法发送到数据库确切执行的字符串,避免注入语句生成。

​	

```python
实例1：
 
str_query= "DELETE FROM Cu_table WHERE id=%s AND name=%s"
stri = cursor.mogrify(str_query,["u1' or '1'-- ","Tom"])
print(stri)
# DELETE FROM Cu_table WHERE id='u1\' or \'1\'-- ' AND name='Tom'
  
实例2：
 # INSERT INTO m1809(No,open, close) VALUES (5,2881,2882), (6,2883,2884);数据库插入
args = [(5,2881,2882), (6,2883,2884)]
args_str = ','.join(cursor.mogrify("%s", (x, )) for x in args)#'(5,2881,2882),(6,2883,2884)'
cursor.execute("INSERT INTO m1809(No,open, close) VALUES "+args_str)

```

#### 4.语句传参方法（易错）：主要是引号的问题

1.  参数是元组 ， **必须加括号,只能是(%s)格式;   参数必须是（str,） **

   ```python
   # 实例1.1.1：一个参数
   str_insert= "INSERT INTO m1805 (id) VALUES (%s)"# 只能是(%s)格式
   cursor.execute(str_insert,('7'))                                        #必须加括号必须是字符串
   con.commit()
       
   # 实例1.1.2：多个参数
   str_insert= "INSERT INTO m1805 (id,name) VALUES (%s,%s)"
   # 只能是(%s)格式
   cursor=con.cursor()
   cursor.execute(str_insert,('123','Jim'))#必须加括号必须是字符串
   con.commit()
   
   ```

2.  参数是字典   **必须加括号包括,只能是%(字典键名)s格式;   参数必须是字典 **

   ```
   #实例1.2.1：
   arg = {'x':'33'}
   str_insert= "INSERT INTO m1805 (id) VALUES (%(x)s)"
   cursor.execute(str_insert,arg)
   con.commit()
       
   #实例1.2.2：
   arg = {'x':'44','y':'Bob'}
   str_insert= "INSERT INTO m1805 (id,name) VALUES (%(x)s,%(y)s)"
   cursor.execute(str_insert,ar
   
   ```

3. 传入变量的格式

   ```python
   # 实例2.1：传入变量到语句%s+传入数值到(%s) cursor.execute(后面必须加括号)
       
   table_name='m1805';id1='id';name='name';id_v='55';name_v='Smith'
   str_insert= "INSERT INTO %s (%s,%s) "%(table_name,id1,name)
   #传入变量
   str_insert=str_insert+"VALUES (%s,%s)"
   # 传入变量的数值；必须分开写，佛则会把变量值当成列或表名寻找，最后报错
       
   cursor.execute(str_insert,(id_v,name_v))#必须加括号必须是字符串
   con.commit()
       
   # 实例2.2：传入变量到语句%s+传入数值到语句'%d','%s'分开书写
       
   table_name='m1805';id1='id';name='name';id_v=66;name_v='Smith'
   str_insert= "INSERT INTO %s (%s,%s) "%(table_name,id1,name)
   str_insert=str_insert+"VALUES ('%d','%s')"%(id_v,name_v)
   #注意 ('%d','%s')不同于上面，这样书写会将变量解析为指定的类型
       
   cursor.execute(str_insert)
   con.commit()
       
   # 实例2.3.1：传入变量到语句%s +传入数值到语句'%d','%s'合并书写
   table_name='m1805';id1='id';name='name';id_v=77;name_v='Smith'
   str_insert= "INSERT INTO %s (%s,%s)  VALUES ('%d','%s')"%(table_name,id1,name,id_v,name_v)
       
   cursor.execute(str_insert)
   con.commit()
       
   # 实例2.3.2：综合运用
   #日期以字符串形式传入
   table_name='m1805';id1='id';name='name';id_v=88;name_v='Smith';
   date='2019-1-27';time='12:05:05'
   str_insert= "INSERT INTO %s (%s,%s,%s,%s)  VALUES ('%d','%s','%s','%s')"%(
                     table_name,id1,name,'date','time',id_v,name_v,date,time)
       
   cursor.execute(str_insert)
   con.commit()
   ```

   

## 执行事务

事务机制可以确保数据一致性。

事务应该具有4个属性：原子性、一致性、隔离性、持久性。这四个属性通常称为ACID特性。

* 原子性（atomicity）。一个事务是一个不可分割的工作单位，事务中包括的诸操作要么都做，要么都不做。
* 一致性（consistency）。事务必须是使数据库从一个一致性状态变到另一个一致性状态。一致性与原子性是密切相关的。
* 隔离性（isolation）。一个事务的执行不能被其他事务干扰。即一个事务内部的操作及使用的数据对并发的其他事务是隔离的，并发执行的各个事务之间不能互相干扰。
* 持久性（durability）。持续性也称永久性（permanence），指一个事务一旦提交，它对数据库中数据的改变就应该是永久性的。接下来的其他操作或故障不应该对其有任何影响

#### 两个方法：
commit()：更新游标的所有操作

rollback（）：回滚游标的所有操作



## 实例：封装成类
将mysql的操作设置成类，简化每一次对数据的操作

```python
import pymysql

class DB():
    def __init__(self, host='localhost', port=3306, db='', user='root', passwd='root', charset='utf8'):
        # 建立连接 
        self.conn = pymysql.connect(host=host, port=port, db=db, user=user, passwd=passwd, charset=charset)
        # 创建游标，操作设置为字典类型        
        self.cur = self.conn.cursor(cursor = pymysql.cursors.DictCursor)

    def __enter__(self):
        # 返回游标        
        return self.cur

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 提交数据库并执行        
        self.conn.commit()
        # 关闭游标        
        self.cur.close()
        # 关闭数据库连接        
        self.conn.close()


if __name__ == '__main__':
    with DB(host='localhost',user='root',passwd='20020520zyh',db='db1') as db:
        db.execute('select * from db1')
        print(db)
        for i in db:
            print(i)
# 当 with 后面的代码块全部被执行完之后，将调用前面返回对象的 __exit__() 方法。

# db为游标，使不使用 fetchall() 方法查询结果都一样。
```



## 常见的错误和异常：

*  Tips :在这里我推荐大家如果在SQL执行不成功的时候可以用**pymysql的mogrify函数来查看构造的SQL语句**，以便快速查看SQL出错的地方，加快调试速度。 

![1635121835815](C:%5CUsers%5C%E9%93%B6%E6%99%97%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1635121835815.png)

### 错误代码1064处理：

```python
import pymysql
if __name__ == '__main__':
	 connect = pymysql.connect(host='***.***.***.***',user='****',passwd='***',port=3306,database='test',charset='utf8')
 	cursor = connect.cursor()
 	table_name = 'person'
	values = 'id int primary key auto_increment, name varchar(20)，age int'
 	sql_1 = cursor.mogrify('create table %s (%s)', (table_name,values))
 	sql_2 = cursor.mogrify('create table %s (%s)'% (table_name,values))
 	print(sql_1)
 	print(sql_2)
 	cursor.close()
	connect.close()
```

结果：

```python
#第一种
create table 'person' ('id int primary key auto_increment, name varchar(20), age int')
#第二种
create table person (id int primary key auto_increment, name varchar(20), age int)
```

 **唯一的区别就在于：两者构造的SQL的引号的差别！！**，通过第一种方式构造出来的SQL语句是带有引号的，第二种是没有引号的。很明显，可以在MySQL终端下运行的是第二种



### 错误代码1054处理

插入语句中，字段是中文的时候，报错

```python
import pymysql
if __name__ == '__main__':
	connect = pymysql.connect(host='***.***.***.***',user='****',passwd='***',port=3306,database='test',charset='utf8')
	cursor = connect.cursor()
	table_name = 'person'
	values =  (23,'李明')
#错误
sen1 = cursor.mogrify('insert into %s(age, name) values(%s, %s)' % (table_name, values[0],values[1]))
                                                            |这里|
sen2 = cursor.mogrify('insert into %s(age, name) values(%s, "%s")' % (table_name, values[0],values[1])) 
# 相比上面的错误代码，这里给姓名对应的占位符加上了引号
print(sen)

```



## 适用pymysql读取数据

法一：

```python

import pandas as pd
import pymysql # Linux Windows 都可以用, 可以导入，不能导出， 导出还得要 mysqldb
 
con = pymysql.connect(host='192.168.0.10', user='lwn', passwd='123456', db='PATENT', port = 3306) # 连接
cur = con.cursor()
 
def read_table(cur, sql_order): # sql_order is a string
    try:
        cur.execute(sql_order) # 多少条记录
        data  = cur.fetchall(  )
        frame = pd.DataFrame(list(data))
    except: #, e:
        frame = pd.DataFrame()
        # print e
        # continue 
    return frame
 
con.commit()
cur.close()
con.close()

```



法二：

```python
engine = create_engine('mysql://root:123456@192.168.0.10/SMLRT_822')
table=sql.read_sql_table(I[i], engine)
```



### 详解博客

https://xuyu.in/721.html