#!/usr/bin/env python
########################################################## Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and DSARC(資訊與決策科學研究所暨資料科學應用研究中心), NTUB(國立臺北商業大學); the CARS(中華R軟體學會); and the DSBA(臺灣資料科學與商業應用協會)
#######################################################
### Notes: This code is provided without warranty.
### Data sets: oreilly.csv, letterdata.csv
##### 如何載入逗號分隔檔案 #####
#### 法一：以open + read載入逗號分隔檔案 ####
import os
data_dir = "/Users/Vince/cstsouMac/Python/Examples/NHRI/data"
fname = os.path.join(data_dir, "letterdata.csv")

f = open(fname)
# <_io.TextIOWrapper name='/Users/vince/cstsouMac/Python/Examples/Basics/data/letterdata.csv' mode='r' encoding='UTF-8'>
dir(f)[49:54]
help(f.readline)
data = f.read()
f.close()

print(len(data))
print(len('Hello'))
lines = data.split("\n")
print(lines[0:5]) # 一橫列一元素，元素內逗號分隔開
print(lines[0][:35])

header = lines[0].split(',')
print(header[:6])

print(type(lines)) # <class 'list'>
print(len(lines)) # 20002

print(lines[20000:])
lines = lines[1:20001] # 注意！最末20001為空字串，要排除此列(Value Error Cannot Copy Sequence of XX to array axis with dim YY)

print(lines[:1])
print(len(lines))

import numpy as np
float_data = np.zeros((len(lines), len(header) - 1))
print(float_data.shape)

for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

# 細節瞭解
len(lines)

list(enumerate(lines))[:5]

len(list(enumerate(lines)))

lines[1].split(',') # 而非lines[:1].split(',')

values = [float(x) for x in lines[1].split(',')[1:]]
print(type(values))
print(values)
len(values)
####################################
# 下面同上
data_dir = "/Users/Vince/cstsouMac/Python/Examples/NHRI/data"
fname = ''.join([data_dir, "/letterdata.csv"])

f = open(fname)
print(dir(f))
data = f.read()
f.close()
print(type(data))

lines = data.split("\n")
# 一橫列一元素，元素內逗號分隔開
print(lines[:5])
# <class 'list'>
print(type(lines))
# 20002筆觀測值，含欄位名稱與最後的空字串
print(len(lines))
# 依逗號切出首列中的各欄名稱
header = lines[0].split(',')
print(header)
# 注意！編號20001最後一個元素為空字串，要排除此列
print(lines[20000:])
# Python串列取值冒號運算子，前包後不包
lines = lines[1:20001]
# 第一筆觀測值
print(lines[:1])
# 共兩萬筆觀測值
print(len(lines))

import numpy as np
# 宣告numpy二維字符陣列(20000, 17)
data = np.chararray((len(lines), len(header)))
print(data.shape)
# 以enumerate()同時抓取觀測值編號與觀測值
for i, line in enumerate(lines):
    # 串列推導 list comprehension
    values = [x for x in line.split(',')]
    data[i, :] = values

print(header)
print(data)
print(data[1])
len(data[1])

#### 法二：以csv.reader{csv}載入逗號分隔檔案 ####
# 檢查當前路徑法一
pwd()
# 檢查當前路徑法二
import os
os.getcwd()
# 確定當前路徑下有欲讀入之檔案
# 如果沒有，請改變工作路徑，或將欲讀入之檔案複製到當前工作路徑下
os.chdir('/Users/vince/cstsouMac/Python/Examples/NHRI/data')

import csv
help(csv)
import codecs # for Python3
help(codecs)

filename = 'oreilly.csv'
data = [] # An empty list

try:
    #with open(filename) as f:
    with codecs.open(filename, "r", encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f) # f must be an iterator
        # help(reader)
        c = 0
        for row in reader:
            if c == 0:
                header = row
            else:
                data.append(row)
            c += 1
except csv.Error as e:
    print ("Error reading CSV file at line %s: %s" % (reader.line_num, e))
    sys.exit(-1)

type(data) # list
len(data) # 100
help(data)
dir(data)
help(data.pop) # check help for specific method

# 顯示結果
if header:
    print (header)
    print ('==================') # 多一列header分隔線

for datarow in data:
    print (datarow)
    
data[:3]

#### 法二：以模組csv載入逗號分隔檔案(較簡潔的寫法) ####
data = []

try:
    #with open(filename) as f:
    with codecs.open(filename, "r", encoding='utf-8', errors='ignore') as f: # file handler "f"
        reader = csv.reader(f)
        #header = reader.next() # '_csv.reader' object has no attribute 'next'
        header = next(reader)
        data = [row for row in reader] # list or dict comprehension
except csv.Error as e:
    print ("Error reading CSV file at line %s: %s" % (reader.line_num, e))
    sys.exit(-1)

help(csv.reader)

#### 法三：以loadtxt{numpy}載入逗號分隔檔案 ####
import numpy
data = numpy.loadtxt('letterdata.csv', dtype='str', delimiter=',') # "oreilly.csv" 比較特別！無法用此法讀入
type(data) # numpy.ndarray
data.ndim
data.shape
len(data)
data[:3]

#### 法四：以genfromtxt{numpy}載入逗號分隔檔案 ####
data = numpy.genfromtxt('letterdata.csv', dtype='str', delimiter=',') # 結果好像與上面不同！
type(data) # numpy.ndarray

#### 法五：以read_csv{pandas}載入逗號分隔檔案 ####
import pandas as pd
data = pd.read_csv('letterdata.csv')
type(data) # pandas.core.frame.DataFrame
data.shape

oreilly = pd.read_csv('oreilly.csv', encoding='ISO-8859-1') # attention to the encoding (http://stackoverflow.com/questions/18171739/unicodedecodeerror-when-reading-csv-file-in-pandas-with-python)
type(oreilly) # pandas.core.frame.DataFrame
oreilly.shape

help(pd.read_csv)

# dat = pd.from_csv('data.tab', sep='\t')

##### 如何載入Microsoft Excel檔案 #####
#### 法一：以open_workbook, sheet_by_name{xlrd}載入Microsoft Excel檔案 ####

import xlrd
from xlrd.xldate import XLDateAmbiguous

file = 'facebook_checkins_2013-08-24.xls'

wb = xlrd.open_workbook(filename=file)

ws = wb.sheet_by_name('總累積')
help(ws)
dir(ws) # 有cell_type

dataset = []

for r in range(ws.nrows):
    col = []
    for c in range(ws.ncols):
        col.append(ws.cell(r, c).value)
        if ws.cell_type(r, c) == xlrd.XL_CELL_DATE:
            try:
                print (ws.cell_type(r, c))
                from datetime import datetime
                date_value = xlrd.xldate_as_tuple(ws.cell(r, c).value, wb.datemode)
                print (datetime(*date_value))
            except XLDateAmbiguous as e:
                print (e)
    dataset.append(col)

### 追根究底
type(dataset) # list
ws.nrows
ws.ncols
help(ws.cell)
help(ws.cell(2, 1).value)

from pprint import pprint
help(pprint)
pprint(dataset)

#### 法二：以read_excel{pandas}載入Microsoft Excel檔案 ####
import pandas as pd
dataset = pd.read_excel('./data/facebook_checkins_2013-08-24.xls', sheet_name='總累積', header=1)
type(dataset) # pandas.core.frame.DataFrame
#help(dataset)
dir(dataset)[-175:-170]
[name for name in dir(dataset) if name in ["head", "columns", "index"]]

dataset.head()
dataset.columns
dataset.index
dataset.iloc[3:5] # 注意與Python原生及numpy的不同

#### 法三：以ExcelFile{pandas}先建一樣例，再用parse剖析工作表後讀入成為DataFrame ####
xls_file = pd.ExcelFile('facebook_checkins_2013-08-24.xls')
table = xls_file.parse('總累積', header=1)
help(xls_file.parse)
type(table) # pandas.core.frame.DataFrame
table.columns
table.index
table.ix[0:3] # 第0~3筆
table.ix[0] # 第0筆

### Understanding the underscore( _ ) of Python
### https://hackernoon.com/understanding-the-underscore-of-python-309d1a029edc
# For storing the value of last expression in interpreter.
10
_
_*3
_*20

# For ignoring the specific values. (so-called “I don’t care”)
x, _, y = (1, 2, 3) # x = 1, y = 3 
x
y

x, *_, y = (1, 2, 3, 4, 5) # x = 1, y = 5
x
y



