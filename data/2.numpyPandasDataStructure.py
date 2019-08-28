########################################################## Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CADS(資訊與決策科學研究所暨資料科學應用研究中心), NTUB(國立臺北商業大學); the CARS(中華R軟體學會); and the DSBA(臺灣資料科學與商業應用協會)
#######################################################
### Notes: This code is provided without warranty.

### numpy data structures
v = [0.5, 0.75, 1.0, 1.5, 2.0]
m = [v, v, v] # matrix of numbers
m



m[1]



m[1][0] # nested indexing



type(m)



v1 = [0.5, 1.5]
v2=[1,2]
m=[v1,v2]
print(m)
c = [m, m] # cube of numbers
c



c[1][1][0] # 那一個 1 ? 前 or 後 ?



type(c)



v = [0.5, 0.75, 1.0, 1.5, 2.0]
m=[v,v,v]
m



v[0] = 'Python'
m # all changed



from copy import deepcopy
v = [0.5, 0.75, 1.0, 1.5, 2.0]
m = 3 * [deepcopy(v), ]
m



v[0] = 'Python'
m # unchanged



import numpy as np
a = np.array([0, 0.5, 1.0, 1.5, 2.0])
type(a)



a[:2]



a.sum() # sum of all elements



a.std() # standard deviation



a.cumsum() # running cumulative sum



a**2 # vectorization



np.sqrt(a)



b = np.array([a, a * 2]) # a matric of 2 rows and 5 columns
b



b[0] # first row



b[0, 2] # third element of first row



b.sum() # 15.0



b.sum(axis=0) # sum along axis 0, i.e. column-wise sum



b.sum(axis=1) # sum along axis 1, i.e. row-wise sum



c = np.zeros((2, 3, 4), dtype='i', order='C') # also: np.ones()
c



d = np.ones_like(c, dtype='f16', order='C') # also: np.zeros_like()
d



import random
I=5000
%time mat = [[random.gauss(0, 1) for j in range(I)] for i in range(I)] # a nested list comprehension



%time mat = np.random.standard_normal((I, I)) # 很快！



%time mat.sum() # 秒殺！



dt = np.dtype([('Name', 'S10'), ('Age', 'i4'), ('Height', 'f'), ('Children/Pets', 'i4', 2)])
s = np.array([('Smith', 45, 1.83, (0, 1)), ('Jones', 53, 1.72, (2, 2))], dtype=dt)
s



s['Name']



s['Height'].mean()



s[1]['Age']


### Vectorization
r = np.zeros((4, 3))
s = np.ones((4, 3))
r+s



2*r+3



s = np.random.standard_normal(3)
r+s



s = np.random.standard_normal(4) # Why not work?
# r+s # ValueError: operands could not be broadcast together with shapes (4,3) (4,)



r.transpose() + s



np.shape(r.T)



np.shape(r)



def f(x):
    return 3*x+5



f(0.5) # float object



r = np.random.standard_normal((4, 3))
f(r) # NumPy array



r


### pandas data structures
from pandas import Series, DataFrame
import pandas as pd
import numpy as np



obj = Series([4, 7, -5, 3])
obj



obj.values



obj.index # Int64Index([0, 1, 2, 3 ])



obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2



obj2.index



obj2['a']



obj2[['c', 'a', 'd']]


obj2[[True, True, False, True]]
print(obj2[obj2 > 0])
print(obj2 * 2)
np.exp(obj2)



'b' in obj2



sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)
obj3



type(obj3) # pandas.core.series.Series



states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index=states)
obj4



pd.isnull(obj4)



pd.notnull(obj4)



obj4.isnull()



print(obj3)
print(obj4)
print(obj3 + obj4)



obj4.name = 'population'
obj4.index.name = 'state'
obj4



print(obj)
obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
print(obj)



data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'], 'year': [2000, 2001, 2002, 2001, 2002], 'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)
print(data)
print(frame) # index都是0, 1, 2, 3, 4



DataFrame(data, columns=['year', 'state', 'pop']) # 可以改變欄位順序



frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'], index=['one', 'two', 'three', 'four', 'five'])
frame2 # 不包含在data中的欄位'debt'



frame2.columns



print(frame2['state'])
print(type(frame2['state']))
print(frame2.year)
print(type(frame2.year))



frame2.ix['three']



frame2['debt'] = np.arange(5.)
frame2



val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
frame2



frame2['eastern'] = frame2.state == 'Ohio'
frame2



del frame2['eastern']
frame2.columns



pop = {'Nevada': {2001: 2.4, 2002: 2.9}, 'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
pop



frame3 = DataFrame(pop)
frame3



frame3.T



DataFrame(pop, index=[2001, 2002, 2003])



pdata = {'Ohio': frame3['Ohio'][:-1], 'Nevada': frame3['Nevada'][:2]}
DataFrame(pdata)



frame3.index.name = 'year'; frame3.columns.name = 'state'
frame3



frame3.values



frame2.values



obj = Series(range(3), index=['a', 'b', 'c'])
obj



index = obj.index
index



index[1:]



# index[1] = 'd'



index = pd.Index(np.arange(3))
print(index)
obj2 = Series([1.5, -2.5, 0], index=index)



obj2.index is index



type(obj2.index)



frame3



'Ohio' in frame3.columns



2003 in frame3.index



obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj



obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj2



obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)



obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3



obj3.reindex(range(6), method='ffill') # 'ffill' means foward fills



frame = DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'], columns=['Ohio', 'Texas', 'California'])
frame



frame2 = frame.reindex(['a', 'b', 'c', 'd'])
frame2



states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)



frame.reindex(index=['a', 'b', 'c', 'd'], method='ffill', columns=states)



frame.ix[['a','b','c','d'], states]



obj = Series(np.arange(5.), index=['a','b','c','d','e'])
obj



new_obj = obj.drop('c')
new_obj



obj.drop(['d','c']) # 注意中括弧！因為多個索引值需以list串起



data = DataFrame(np.arange(16).reshape((4,4)), index=['Ohio', 'Colorado', 'Utah', 'New York'], columns=['one', 'two', 'three', 'four'])
data



data.drop(['Colorado','Ohio'])



data.drop('two', axis=1)



obj = Series(np.arange(4.), index=['a','b','c','d'])
obj



obj[['b','a','d']]



obj[2:4]



obj['b':'c']



data = DataFrame(np.arange(16).reshape((4, 4)), index=['Ohio', 'Colorado', 'Utah', 'New York'],columns=['one', 'two', 'three', 'four'])
data



data[['three', 'one']]



data[:2]



data[data['three'] > 5]



data < 5



data[data < 5] = 0
data



data.ix['Colorado', ['two', 'three']]



data.ix[['Colorado', 'Utah'], [3, 0, 1]]



data.ix[2]



data.ix[:'Utah', 'two']



data.ix[data.three > 5, :3]



df1 = DataFrame(np.arange(12.).reshape((3,4)), columns=list('abcd'))
df1



df2 = DataFrame(np.arange(20.).reshape((4,5)), columns=list('abcde'))
df2



df1 + df2



df1.add(df2)



df1.add(df2, fill_value=0) # Broadcasting next time

### pandas Series apply()
a = [1, 2, 3] # a list (built-in data structure)
type(a) # list

import numpy as np
a = np.array(a)
type(a) # numpy.ndarray

a.apply(lambda x: x * 2)
# AttributeError: 'numpy.ndarray' object has no attribute 'apply'

import pandas as pd
a = pd.Series(a)
a.apply(lambda x: x * 2)



# > Pandas DataFrame 資料取值的多種方法

import pandas as pd
fb = pd.read_excel("/Users/Vince/cstsouMac/Python/Examples/Basics/data/facebook_checkins_2013-08-24.xls", skiprows=1)

fb3 = pd.read_excel("/Users/Vince/cstsouMac/Python/Examples/Basics/data/facebook_checkins_2013-08-24.xls", skiprows=[0]) # write an email to pandas ?

fb.dtypes

fb.head()


# - 以屬性名稱取出整個縱行。

fb['地標名稱']


# - 運用DataFrame的屬性取出整個縱行。

fb.類別


# - 透過loc方法取值，可以對行列進行限制，不過必須使用行名(label-based indexing)。

fb.loc[:10, ['地區','累積打卡數']] # 注意！0 ~ 10，包括10!


# - 透過iloc方法取值，可以對行列進行限制，不過必須使用行索引(positional indexing)。

fb.iloc[:10, [6, 2]] # 0 ~ 9


# - 透過ix方法取值，可以使用行名與行索引，由於結合了loc與iloc兩者的用法，因此是最實用的資料選取方式。

fb.ix[:10, ['latitude', 'longtitude']] # 注意！0 ~ 10，包括10!

fb.ix[:10, [3, 4]] # 注意！0 ~ 10，包括10!


# - 整行選取可以使用第一、第二種方法，後三(兩, ix deprecated)種方法適合做多條件的彈性選取。

