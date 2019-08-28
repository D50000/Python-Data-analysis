# -*- coding: utf-8 -*-
"""
Created on Sat May  4 14:16:46 2019

@author: tom05
"""

### python 連線 Oracle 亂碼問題（cx_Oracle）
import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'



import pandas as pd
import cx_Oracle


### 連接資料庫方法一
host = "127.0.0.1"  #資料庫ip
port = "1521"   #埠
sid = "xe"  #資料庫名稱
dsn = cx_Oracle.makedsn(host, port, sid)
#HR是資料使用者名稱，hr是登入密碼（預設使用者名稱和密碼）
conn = cx_Oracle.connect("HR", "hr", dsn)

### 連接資料庫方法二
conn = cx_Oracle.connect('HR', 'hr', 'localhost:1521/xe') 

dir(conn)
conn.dsn
conn.username
print(conn.version) # 11.2.0.2.0 11g

### 資料表查詢範例
# select column1, column2, ... from table
# 查詢員工資料表的所有欄位(查詢所有欄位使用*)
sql = 'select * from employees'
results = pd.read_sql(sql,conn)

# select table1.column1, table1.column2, table2.column2 from table1 join table2 on table1.column1 = table2.column1
# 查詢員工資料表(別名定為e)的員工ID、員工姓名及部門ID欄位，利用部門ID連結部門資料表(別名定為d)查詢部門名稱欄位
sql2 = 'select e.employee_id, e.first_name, e.last_name, e.department_id, d.department_name from employees e join departments d on e.department_id = d.department_id'
results2 = pd.read_sql(sql2,conn)

# 查詢員工資料表(別名定為e)的員工ID、員工姓名及工作ID欄位，利用工作ID連結工作資料表(別名定為j)查詢工作職稱欄位
sql3 = 'select e.employee_id, e.first_name, e.last_name, e.job_id, j.job_title from employees e join jobs j on e.job_id = j.job_id'
results3 = pd.read_sql(sql3,conn)


### 另一種獲得db資料方法con.cursor()(except pd.read_sql())
cursorObj = conn.cursor()
type(cursorObj) # cx_Oracle.Cursor
dir(cursorObj)

#當我們發出查詢命令之後，要將資料取進python做使用可以使用fetch類方法，最簡單的就是fetchall()，他會以list的方式回傳所有資料或者是空list(無資料)。
#另一種方式是一次取出一筆：fetchone()，若沒有資料便會回傳None，事實上根據官方文件，fetchall()也是用fetchone()來實作的
cursorObj.execute('select * from employees')
row = cursorObj.fetchone()
while row is not None:
  print(row)
  row = cursorObj.fetchone()
  
# same as above
cursorObj.execute('select * from employees')
for row in cursorObj:
  print(row)

### con.cursor() with sql scripts
 # 查詢員工資料表(別名定為e)的員工ID、員工姓名及工作ID欄位，利用工作ID連結工作資料表(別名定為j)查詢工作職稱欄位，指定工作ID至jobId變數(只查詢工作ID符合的資料)
sql4 = 'select e.employee_id, e.first_name, e.last_name, e.job_id, j.job_title from employees e join jobs j on e.job_id = j.job_id where e.job_id = :jobId'

cursorObj = conn.cursor()

cursorObj.prepare(sql4)
### filtering columns
cursorObj.execute(None, {'jobId' : 'AD_VP'}) #指定jobId為AD_VP，表示只查詢工作ID為AD_VP的資料
for row in cursorObj:
  print(row)

### 關閉資料庫連接
conn.close()
cursorObj.close()

### 參考文獻
#http://pclevin.blogspot.com/2014/04/oracledb-oracle-database-oracle.html

#http://bhan0507.logdown.com/posts/1855591-oracle-x-python-how-to-use-python-oracle-database

#https://www.jb51.net/article/128548.htm

#https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/356776/
