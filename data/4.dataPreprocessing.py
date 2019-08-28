########################################################## Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CADS(資訊與決策科學研究所暨資料科學應用研究中心), NTUB(國立臺北商業大學); the CARS(中華R軟體學會); and the DSBA(臺灣資料科學與商業應用協會)
#######################################################
### Notes: This code is provided without warranty.
import pandas as pd
print(pd.__version__) # 0.20.3

### Data reshaping
# We can get USArrests ready by the following two ways.
### Way 1: Import USArrests from USArrests.csv
USArrests = pd.read_csv("./data/USArrests.csv")
USArrests.columns = ['state', 'Murder', 'Assault', 'UrbanPop', 'Rape'] # newly added
USArrests = USArrests.set_index('state') # newly added
print(USArrests.head())
print(USArrests.shape)

### Way 2: Import USArrests from R package {datasets}
# pip3 install statsmodels
import statsmodels
print(statsmodels.__version__) # 0.8.0

#%pylab inline # https://matthiasbussonnier.com/posts/10-No-PyLab-Thanks.html
%matplotlib inline
import statsmodels.api as sm
# with or without intercept https://stackoverflow.com/questions/30650257/ols-using-statsmodel-formula-api-versus-statsmodel-api

dir(sm.datasets) # attention to 'get_rdataset'

help(sm.datasets.get_rdataset)

# It needs Internet connection and will take some time.
USArrests = sm.datasets.get_rdataset("USArrests")

type(USArrests) # statsmodels.datasets.utils.Dataset or pandas.core.frame.DataFrame

# help(sm.datasets.utils.Dataset) # too long to show in handouts !

USArrests.keys()

USArrests['title']

USArrests.package

USArrests.data
#type(USArrests.data) # <class 'pandas.core.frame.DataFrame'>

USArrests.items() # 與上面顯示格式不同
# type(USArrests.items()) # <class 'dict_items'>

USArrests = USArrests.data
print(USArrests.head())
print(USArrests.shape)

### Data reshaping by stack() and unstack()
USArrests_l = USArrests.stack()
USArrests_l

USArrests_l.unstack() # axis = 1, 50*4
USArrests_l.unstack(level=0) # transpose of above, 4*50

### Supplement: Data reshaping by Pandas DataFrame
USArrests.index
USArrests['state'] = USArrests.index # newly added
#state = pd.DataFrame.from_dict({'state':list(pd.Series(USArrests.data.index))})
#pd.DataFrame({'state':list(pd.Series(USArrests.data.index))})
#pd.DataFrame.from_dict({'state':pd.Series(USArrests.data.index)})
#pd.DataFrame({'state':pd.Series(USArrests.data.index)})

#state = pd.DataFrame.from_dict({'state1':list(USArrests.data.index), 'state':list(USArrests.data.index)})
#state = state.set_index('state1')

#help(pd.DataFrame.join)
# Append column to pandas dataframe (https://stackoverflow.com/questions/20602947/append-column-to-pandas-dataframe)
#USArrests.data = USArrests.data.join(state) # 如果join索引不同的state，會出現錯誤

#print(state)
#USArrests.data

# Wide to long data transform in pandas (https://stackoverflow.com/questions/37418295/wide-to-long-data-transform-in-pandas)
#help(pd.melt)
USArrests_dfl = pd.melt(USArrests, id_vars=['state'], var_name='fact', value_name='figure')
# USArrests_dfl.to_csv('df.csv')
# Pandas long to wide reshape (https://stackoverflow.com/questions/22798934/pandas-long-to-wide-reshape)
USArrests_dfl.pivot(index='state', columns='fact', values='figure')

### Data sorting
import pandas as pd
#USArrests.data

#USArrests.keys()

# USArrests = pd.DataFrame(USArrests['data'])
type(USArrests)
#type(USArrests['data']) # <class 'pandas.core.frame.DataFrame'>
#USArrests = USArrests.data

USArrests.sort_index() # sort by index on axis 0

USArrests.sort_index(axis=1).head() # sort by index on axis 1

USArrests.sort_index(axis=1, ascending=False).head() # sort by index on axis 1 in descending order

# Python的文化：謹慎地使用.與_
USArrests.sort_values(by="Rape", ascending=False).head() # sort by values on "Rape" (composite functions)

USArrests.sort_values(by=["Rape","UrbanPop"], ascending=False).head() # sort by values on "Rape" and "UrbanPop"

USArrests.rank(axis=1, ascending=False)

USArrests.rank(axis=0, ascending=False).head()

USArrests.rank(axis=0, ascending=False, method="max") # Tie-breaking method from default average to max

### Grouping and summarization
# Summarising, Aggregating, and Grouping data in Python Pandas
# http://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/

import pandas as pd
import numpy as np
import dateutil
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
# Load data from csv file
data = pd.read_csv('./data/phone_data.csv')
data.shape
data.dtypes
data.head()

# Convert date from string to date times
data['date'] = data['date'].apply(dateutil.parser.parse, dayfirst=True) # implicit looping 隱式迴圈
print(data.dtypes)

# Or you can do as following
data['date'] = pd.to_datetime(data['date'])

### What is that?
type(data['date'])
help(data['date'].apply) # Invoke function on values of Series.

series = pd.Series([20, 21, 12], index=['London',
'New York','Helsinki'])
series

# pandas series apply() 的多種用法
# 內建函數
series.apply(np.log)

# 匿名函數
series.apply(lambda x: x**2)

# 自定義函數一
def square(x):
    return x**2
series.apply(square)

# 自定義函數一(注意參數如何傳入！)
def subtract_custom_value(x, custom_value):
    return x-custom_value
series.apply(subtract_custom_value, args=(5,))


### Summarising the DataFrame
data.keys()
data.columns
print(data.columns[:5])
print(data.columns[5:])

# How many rows the dataset
# data['item'].count()

# How many types of item and their frequency?
data['item'].value_counts() # value_counts方法
help(pd.Series.value_counts) # dropna=False

# How many types of network_type and their frequency?
data['network_type'].value_counts() # value_counts方法

# What was the longest phone call / data entry?
data['duration'].max()

# How many seconds of phone calls are recorded in total?
# logical indexing 邏輯值索引或布林值索引
data['item'] == 'call'
data['duration'][data['item'] == 'call'].sum() # 92321.0
 
# How many entries are there for each month?
data['month'].value_counts() # value_counts方法
 
# Number of non-null unique network entries
data['network'].nunique() # 9
data['network'].value_counts()
help(pd.Series.nunique) # Excludes NA values by default
data['network'].nunique(dropna=False) # 9

# Missing values identification
# data['network'].isnull().sum() # 0
data.isnull().sum() # None

#count	Number of non-null observations
#sum	Sum of values
#mean	Mean of values
#mad	Mean absolute deviation
#median	Arithmetic median of values
#min	Minimum
#max	Maximum
#mode	Mode
#abs	Absolute Value
#prod	Product of values
#std	Unbiased standard deviation
#var	Unbiased variance
#sem	Unbiased standard error of the mean
#skew	Unbiased skewness (3rd moment)
#kurt	Unbiased kurtosis (4th moment)
#quantile	Sample quantile (value at %)
#cumsum	Cumulative sum
#cumprod	Cumulative product
#cummax	Cumulative maximum
#cummin	Cumulative minimum

### Summarising Groups in the DataFrame

type(list(data.groupby(['month']))[-1])
type(list(data.groupby(['month']))[-1][1])

list(data.groupby(['month']))[-1][0]
list(data.groupby(['month']))[-1][1].iloc[:,:4]
list(data.groupby(['month']))[-1][1].iloc[:,4:]

# 分組數據是pandas資料框的groupby類型物件
print(type(data.groupby(['month'])))

# groupby類型物件的groups屬性是字典結構
print(type(data.groupby(['month']).groups))

# dict_keys
data.groupby(['month']).groups.keys() # groups -> keys: values

keys = ['2014-11', '2014-12', '2015-01', '2015-02']
[data.groupby(['month']).groups.get(key) for key in keys]

#dir(data.groupby(['month']))
len(data.groupby(['month']).groups['2014-11']) # 230筆

# Get the first entry for each month 各月第一筆資料
data.groupby('month').first()
type(data.groupby('month').first())
print(data.groupby('month').first().iloc[:,:4])
print(data.groupby('month').first().iloc[:,4:])
# Get the sum of the durations per month
#list(data.groupby('month')) # list內嵌tuples
#list(data.groupby('month')['duration'])
data.groupby('month')['duration'].sum()
 
# Get the number of dates / entries in each month
# data.groupby('month')['date'].count()
 
# What is the sum of durations, for calls only, to each network?
data[data['item'] == 'call'].groupby('network')['duration'].sum() # 布林值索引、群組、挑欄位、做計算

###You can also group by more than one variable, allowing more complex queries. (Grouping by multiple variables)
# How many calls, sms, and data ('item') entries are in each month ('month')?
data.groupby(['month', 'item'])['date'].count() # Beautiful!

# How many calls, texts(sms), and data (all kinds of item) are sent per month, split by network_type?
# data.groupby(['month', 'network_type'])['date'].count()

### Groupby output format – Series or DataFrame?
data.groupby('month')['duration'].sum() # produces Pandas Series
type(data.groupby('month')['duration'].sum())
data.groupby('month')[['duration']].sum() # Produces Pandas DataFrame
type(data.groupby('month')[['duration']].sum())

# The groupby output will have an index or multi-index on rows corresponding to your chosen grouping variables. To avoid setting this index, pass “as_index=False” to the groupby operation.
data.groupby('month').agg({"duration": "sum"})
type(data.groupby('month').agg({"duration": "sum"}))
data.groupby('month').agg({"duration": "sum"}).index
data.groupby('month').agg({"duration": "sum"}).columns

data.groupby('month', as_index=False).agg({"duration": "sum"})
type(data.groupby('month', as_index=False).agg({"duration": "sum"}))
data.groupby('month', as_index=False).agg({"duration": "sum"}).index # 索引為流水號
data.groupby('month', as_index=False).agg({"duration": "sum"}).columns # 分組變數亦為欄位名稱

### Multiple Statistics per Group
# Applying a single function to columns in groups
# Instructions for aggregation are provided in the form of a python dictionary. Use the dictionary keys to specify the columns upon which you’d like to operate, and the values to specify the function to run.

# Group the data frame by month and item and extract a number of stats from each group
# data.groupby(['month', 'item']).agg({'duration':sum, # find the sum of the durations for each group
#                                     'network_type': "count", # find the number of network type entries
#                                     'date': 'first'})    # get the first date per group

# Define the aggregation procedure outside of the groupby operation (也可在外面定義aggregation程序)
# aggregations = {
#    'duration':'sum',
#    'date': lambda x: max(x)
#}
# data.groupby('month').agg(aggregations)


### Applying multiple functions to columns in groups
# Group the data frame by month and item and extract a number of stats from each group
data.groupby(['month', 'item']).agg({'duration': [min, max, sum],      # find the min, max, and sum of the duration column
                                     'network_type': "count", # find the number of network type entries
                                     'date': [min, 'first', 'nunique']})    # get the min, first, and number of unique dates per group

    
    
    
    
    
### Supplements: Renaming grouped statistics from groupby operations
# When multiple statistics are calculated on columns, the resulting dataframe will have a multi-index set on the column axis. This can be difficult to work with, and I typically have to rename columns after a groupby operation.
grouped = data.groupby('month').agg({"duration": [min, max, 'mean']}) # {} is needed and NameError: name 'mean' is not defined if you did not enclose it in single quotation mark.
### take a look at multi-index
grouped.columns # MultiIndex(levels=[['duration'], ['min', 'max', 'mean']], labels=[[0, 0, 0], [0, 1, 2]])
grouped.columns = grouped.columns.droplevel(level=0) # Index(['min', 'max', 'mean'], dtype='object') 因為第一階的欄位名稱都是'duration'
grouped = grouped.rename(columns={"min": "min_duration", "max": "max_duration", "mean": "mean_duration"}) # 再更新單階的欄位名稱
grouped.head()

grouped = data.groupby('month').agg({"duration": [min, max, 'mean']}) 
grouped.head()

# Using ravel, and a string join, we can create better names for the columns:
grouped.columns = ["_".join(x) for x in grouped.columns.ravel()] # A cool list comprehension !
grouped.columns
grouped.head()

### (Future Deprecation): The final piece of the puzzle is the ability to rename the newly calculated columns and to calculate multiple statistics from a single column in the original data frame. Such calculations are possible through nested dictionaries (注意巢狀字典結構), or by passing a list of functions for a column.
# Define the aggregation calculations
aggregations = {
    'duration': { # work on the "duration" column
        'total_duration': 'sum',  # get the sum, and call this result 'total_duration'
        'average_duration': 'mean', # get mean, call result 'average_duration'
        'num_calls': 'count'
    },
    'date': {     # Now work on the "date" column
        'max_date': 'max',   # Find the max, call the result "max_date"
        'min_date': 'min',
        'num_days': lambda x: max(x) - min(x)  # Calculate the date range per group
    },
    'network': ["count", "max"]  # Calculate two results for the 'network' column with a list
}
 
# Perform groupby aggregation by "month", but only on the rows that are of type "call"
data[data['item'] == 'call'].groupby('month').agg(aggregations)
# Note that the results have multi-indexed column headers. (date -> max_date, num_days, min_date, network -> count, max, ...)
# The groupby functionality in Pandas is well documented in the official docs and performs at speeds on a par (unless you have massive data and are picky with your milliseconds) with R’s data.table and dplyr libraries.
### Analysis of Weather data using Pandas, Python, and Seaborn (https://www.shanelynn.ie/analysis-of-weather-data-using-pandas-python-and-seaborn/)


### Data cleaning on missing values
import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
0.0,11.0,12.0,'''

csv_data
type(csv_data)

StringIO(csv_data)
type(StringIO(csv_data))

df = pd.read_csv(StringIO(csv_data))
df

help(df.isnull)

df.isnull()
df.isnull().sum() # from variable perspective
df.values # np.array behind

df.dropna()
df.dropna(axis=1)
df.dropna(how='all') # default 'any'

df.dropna(thresh=4) # drop rows that have not at least 4 non-NaN values

df.dropna(subset=['C']) # drop rows that column 'C' has NaN

### Imputation by scikit-learn
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values) # input numpy.ndarray
imputed_data

imputed_data = imr.transform(df) # you can also input pandas.DataFrame
imputed_data

### Case study: algae data set
import os
os.getcwd()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
algae = pd.read_csv("./data/algae.csv") # make it from R by write.csv(algae, file="algae.csv", row.names = FALSE)
print(algae)

type(algae) # 資料物件類型

algae.ndim

algae.shape

list(algae) # 屬性名稱，或algae.columns.values

algae.columns.values

algae.index.values

algae.dtypes # 屬性型別

algae.ftypes # 屬性的稀疏sparse/    dense

algae.describe(include='all')
print(algae['mxPH'].isnull().head())
print("mxPH遺缺 {} 筆觀測值".format(algae['mxPH'].isnull().sum())) # 單變量遺缺值檢查(sum(isnull(algae['mxPH']))) , % format your output

### Supplement
algae_summary = algae.describe(include='all')
algae_new = pd.concat([algae, algae_summary])
############

mxPH_naomit = algae['mxPH'].dropna() # 移除單變量遺缺值
len(mxPH_naomit)

algae.isnull() # 檢視整個資料表的遺缺狀況

algae.dropna(axis=0) # 移除不完整的觀測值
# algae_naomit = algae.dropna(axis=0)
# print(algae_naomit.shape)

algae.dropna(thresh=17) # 以thresh設定最低變數個數門檻 keeping only rows containing a certain number of observations
# algae_over17 = algae.dropna(thresh=17)
# print(algae_over17.shape)

algae_nac = algae.isnull().sum(axis=0) # 各變數遺缺狀況：Chla遺缺數量最多, Cl次之...
algae_nac

algae_nar = algae.isnull().sum(axis=1) # 各觀測值遺缺狀況：遺缺變數個數
algae_nar
# print(algae_nar[60:65])

algae[algae_nar > 0] # 檢視不完整的觀測值(logical or boolean indexing)

print(algae[algae_nar > 0].index) # 遺缺變數大於0的觀測值位置
len(algae[algae_nar > 0].index) # 不完整觀測值個數

algae[algae_nar > algae.shape[1]*.2] # 檢視遺缺變數超過20%的觀測值

algae[algae_nar > algae.shape[1]*.2].index # 遺缺變數超過20%的觀測值位置(18 * .2 = 3.6)

algae = algae.drop(algae[algae_nar > algae.shape[1]*.2].index) # 以drop方法，給IndexRange，移除遺缺嚴重的觀測值
algae

mxPH = algae['mxPH'].dropna() # 須先移除NaNs後再繪圖
#fig, ax = plt.subplots()
#ax.hist(mxPH, alpha=0.9, color='blue')
#plt.show() # 近乎對稱鐘型分佈

fig = plt.figure() # 繪圖的多種方法
ax = fig.add_subplot(111)
ax.hist(mxPH) # high-level plotting
ax.set_xlabel('Values') # low-level plotting (customization)
ax.set_ylabel('Frequency')
ax.set_title('Histogram of mxPH') # low-level plotting
plt.show()

#ax = plt.gca() # 繪圖的多種方法
# the histogram of the data
#ax.hist(mxPH, bins=35, color='r')
#ax.set_xlabel('Values')
#ax.set_ylabel('Frequency')
#ax.set_title('Histogram of mxPH')
#plt.show()

print(algae['mxPH'].describe())
mean = algae['mxPH'].mean()
algae['mxPH'].fillna(mean, inplace=True) # 以算術平均數填補唯一的遺缺值
print(algae['mxPH'].describe()) # 確認是否填補完成

Chla = algae['Chla'].dropna() # 須先移除NaNs後再繪圖
fig, ax = plt.subplots()
ax.hist(Chla, alpha=0.9, color='blue')
ax.set_xlabel('Values')
ax.set_ylabel('Frequency')
ax.set_title('Histogram of Chla')
plt.show() # 右偏不對稱分佈

print(algae['Chla'].describe())
median = algae['Chla'].median()
algae['Chla'].fillna(median, inplace=True) # 以中位數(50%分位數)填補遺缺值
print(algae['Chla'].describe()) # 確認是否填補完成

median

#algae

algae.median(axis=0) # colMeans() in R

var = algae.isnull().sum(axis=0)
var[var > 0]

alCorr = algae.corr() # 自動挑數值變數計算相關係數 -> correlation coefficient matrix，PO4與oPO4高相關

alCorr

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")

# Generate a mask for the upper triangle
mask = np.zeros_like(alCorr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize = (9, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(alCorr, mask=mask, annot=False, cmap=cmap, ax=ax)
f.tight_layout()

#plt.matshow(alCorr)
# https://github.com/statsmodels/statsmodels/issues/5343
# !pip install --upgrade patsy
import statsmodels.formula.api as sm
result = sm.ols(formula="PO4 ~ oPO4", data=algae).fit() # ols: ordinary least square

#dir(result)
#[(name, type(getattr(result, name))) for name in dir(result)]

[name for name in dir(result) if not callable(getattr(result, name))]


print (result.params)

print (result.summary())

type(result.params)

type(algae)

algae.index # pandas.DataFrame取出橫列要用ix方法，參見下面

algae.columns

algae.ix[[27,29],['oPO4', 'PO4']] # algae.loc[[27,29],['oPO4', 'PO4']]

print (algae.ix[27]['PO4']) # 或是chained indexing
algae.set_value(27, 'PO4', result.params[0] + result.params[1]*algae.ix[27]['oPO4']) # pandas.DataFrame改值要用set_value
algae.ix[27]['PO4']

result.params[0] + result.params[1]*algae.ix[27]['oPO4']

### (Categorical) Variable Encoding ####
import pandas as pd

df = pd.DataFrame([['green', 'M', 10.1, 'class1'], ['red', 'L', 13.5, 'class2'], ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']

#labels = ['color', 'size', 'price', 'classlabel']
#df = pd.DataFrame.from_records([('green', 'M', 10.1, 'class1'), ('red', 'L', 13.5, 'class2'), ('blue', 'XL', 15.3, 'class1')], columns = labels)

df

type(df)

print(df.size) # 此size非彼size也！

print(df['size'])

#dir(df['size'])


#df['size'].to_pickle('/Users/Vince/cstsouMac/Python/Examples/Basics/to_pickle.txt') # 要先有檔

#import pickle
#f = open('/Users/Vince/cstsouMac/Python/Examples/Basics/to_pickle.txt', 'rb')
#pickle.load(f)


#help(df['size'].map)

# 標籤編碼(label encoding)
size_mapping = {'XL': 3, 'L': 2, 'M': 1}
df['size'] = df['size'].map(size_mapping)
df

import numpy as np
class_mapping = {label:idx for idx, label in enumerate(np.unique(df['classlabel']))}

class_mapping

df['classlabel'] = df['classlabel'].map(class_mapping)

df

from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'])
y
#y.size

#[(nam, 'func' if callable(getattr(class_le, nam)) else 'attr') for nam in dir(class_le)]

#help(class_le.inverse_transform)
# y 要改成 y.reshape(-1,1)
class_le.inverse_transform(y.reshape(-1,1)) # /Users/Vince/anaconda/lib/python3.5/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
#  if diff:

### 全部都要label encoding, 不能有object(字串)!!!
X = df[['color', 'size', 'price']].values
#X = df[['color', 'size', 'price']]
X
color_le = LabelEncoder()
#X[:, 0]
X[:, 0] = color_le.fit_transform(X[:, 0])
#X.iloc[:, [0]] = color_le.fit_transform(X.iloc[:, [0]]) # 有warning
#X['color'] = color_le.fit_transform(X['color']) # 有warning
X

from sklearn.preprocessing import OneHotEncoder
#help(OneHotEncoder)
ohe = OneHotEncoder(categorical_features=[0]) # color is the first column in the feature matrix X
#ohe.fit_transform(X)
ohe.fit_transform(X).toarray()

#ohe = OneHotEncoder(categorical_features=[0], sparse = False)
#ohe.fit_transform(X)

df[['color', 'size', 'price']]
pd.get_dummies(df[['color', 'size', 'price']])

### Cell Segmentation Case ####
import pandas as pd
import numpy as np
cell = pd.read_csv('/Users/Vince/cstsouMac/Python/Examples/NHRI/data/segmentationOriginal.csv')

### 1. Data Understanding and Missing Values Identifying
cell.head(2)

cell.info() # RangeIndex, Columns, dtypes, memory type

cell.shape

cell.columns.values # 119 variable names

cell.dtypes

cell.describe(include = "all")

cell.isnull().any() # check NA by column

cell.isnull().values.any() # False, means no missing value ! Check the difference between above two !!!!

#cell.isnull()
#type(cell.isnull()) # pandas.core.frame.DataFrame, so .index, .column, and .values three important attributes

#cell.isnull().values
#type(cell.isnull().values) # numpy.ndarray

cell.isnull().sum() # 

### 2. Select the training set
#cell['Case'].nunique()
cell['Case'].unique()
#select the training set
cell_train = cell.loc[cell['Case']=='Train'] # same as cell[cell['Case']=='Train']
cell_train.head()

# 注意cell['Case']與cell[['Case']]的區別！R語言亦有類似的情況！

cell['Case'] # 沒有變數名稱
type(cell['Case']) # <class 'pandas.core.series.Series'>
cell[['Case']] # 有變數名稱
type(cell[['Case']]) # <class 'pandas.core.frame.DataFrame'>


### 3. Create feature matrix
cell_data = cell_train.drop(['Cell','Class','Case'], axis=1)
cell_data.head()

# alternative way to do the same thing
cell_data = cell_train.drop(cell_train.columns[0:3], 1)
cell_data.head()

### 4. Differentiate categorical features from numeric features
print(cell_data.columns)
type(cell_data.columns) # pandas.core.indexes.base.Index

# 法ㄧ：
dir(pd.Series.str)
pd.Series(cell_data.columns).str.contains("Status") # logical indices after making cell_data.columns as pandas.Series
#type(pd.Series(cell_data.columns).str.contains("Status")) # pandas.core.series.Series

cell_data.columns[pd.Series(cell_data.columns).str.contains("Status")] # again pandas.core.indexes.base.Index
#type(cell_data.columns[pd.Series(cell_data.columns).str.contains("Status")]) # pandas.core.indexes.base.Index

len(cell_data.columns[pd.Series(cell_data.columns).str.contains("Status")]) # 58 features with "Status"

cell_num = cell_data.drop(cell_data.columns[pd.Series(cell_data.columns).str.contains("Status")],axis=1)
cell_num.head()

# 法二：
status = []
for h in range(len(list(cell_data.columns))):
    if "Status"in list(cell_data.columns)[h]:
        status.append(list(cell_data.columns)[h])

cell_num = cell_data.drop(status, axis=1)
cell_num.head()

# 法三： The most succinct way I think
cell_cat = cell_data.filter(regex='Status') # Note that this routine does not filter a dataframe on its contents. The filter is applied to the labels of the index.
cell_cat.head()

#pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 500)
cell_cat.apply(lambda x: x.value_counts(), axis=0)


# Separate positive redictors
pos_indx = np.where(cell_data.apply(lambda x: np.all(x > 0)))[0]
cell_data_pos = cell_data.iloc[:, pos_indx]
cell_data_pos.head()
#help(np.all)

### 5. Create class label vector(label encoding and one-hot encoding)
# R語言需要嗎？何時需要？何時不需要？
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder # Encode labels with value between 0 and n_classes-1.

# label encoding
le_class = LabelEncoder().fit(cell['Class'])
Class_label = le_class.transform(cell['Class']) # 0: PS, 1: WS
Class_label.shape # (2019,)

# one-hot encoding
ohe_class=OneHotEncoder(sparse=False).fit(Class_label.reshape(-1,1)) # sparse : boolean, default=True Will return sparse matrix if set True else will return an array.
#help(OneHotEncoder)
ohe_class.get_params()
#{'categorical_features': 'all',
# 'dtype': float,
# 'handle_unknown': 'error',
# 'n_values': 'auto',
# 'sparse': False}
#ohe_class.categorical_features

Class_ohe=ohe_class.transform(Class_label.reshape(-1,1)) # (2019, 2)

Class_label.reshape(-1,1).shape # (2019, 1) different to 1darray (2019,)

Class_ohe.shape # (2019, 2) 2darray
Class_ohe


# 再練習一下最快的方法Fast way to do one-hot encoding or dummy encoding
Class_dum = pd.get_dummies(cell['Class'])
print (Class_dum.head())


### 6. Pick out low variance feature(s) 

from sklearn.feature_selection import VarianceThreshold 
X=cell_num
sel=VarianceThreshold(threshold=0.16) # 0.16
print(sel.fit_transform(X))
#help(sel)

# What's the output?
Y = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
selector = VarianceThreshold()
selector.fit_transform(Y) # remove the first and last attributes because of zero variance

# fit and transform on same object
sel.fit_transform(X).shape # (1009, 49), nine low variance features already removed
dir(sel)

sel.get_support()

cell_num.columns[~sel.get_support()] # ~ like ! in R

import numpy as np
unique, counts = np.unique(sel.get_support(), return_counts=True)

dict(zip(unique, counts)) # {False: 9, True: 49}

# Find the standard deviation and filter
help(cell_num.std)
cell_num.std()
cell_num.std() > .3

threshold = 0.3
print(cell_num.std()[cell_num.std() < threshold].index.values)
# cell_num.drop(cell_num.std()[cell_num.std() < threshold].index.values, axis=1) # 移除變異數過低的屬性, too large

### 7. Transform skewed feature(s) by Box-Cox Transformation
cell_num.skew(0).sort_values()
cell_num.skew(0)[cell_num.skew(0) > 1].index.values # 偏態係數高於1的屬性


import numpy as np
from scipy import stats as sts
#算skewness
#skewValues = sts.skew(cell_num)
print(sts.skew(cell_num)) # numpy.ndarray
type(sts.skew(cell_num))

skewValues = cell_num.apply(sts.skew, axis=0) # pandas.Series
print(skewValues)

### Box-Cox Transformation
# 先試AreaCh1前六筆(只接受一維陣列，自動估計lambda)
from scipy import stats
print(cell['AreaCh1'].head(6))
stats.boxcox(cell['AreaCh1'].head(6))

# stats.boxcox()輸出為兩元素，BC轉換後的AreaCh1與lambda估計值，行成的值組
type(stats.boxcox(cell['AreaCh1'].head(6))) # tuple

# 分別取出BC轉換後的AreaCh1與lambda估計值
stats.boxcox(cell_num['AreaCh1'])[0]
stats.boxcox(cell_num['AreaCh1'])[1]
help(stats.boxcox)

# 補充：另一種Box-Cox公式(可傳入二維陣列，但是要給lambda)
from scipy.special import boxcox1p
lam = 0.16
cell_num_bc = boxcox1p(cell_num, lam)
cell_num_bc

### 以下為講義p.257的練習
# BC轉換傳入的變數值必須為正數
# try except捕捉異常狀況(常用的程式撰寫技巧)
# https://stackoverflow.com/questions/8069057/try-except-inside-a-loop
bc = {}
for col in cell_num.columns:
  try:
    bc[col] = stats.boxcox(cell_num[col])[0]
  except ValueError:
    print('Non-positive columns:{}'.format(col))
  else:
    continue

bc = pd.DataFrame(bc)
print(bc)

#try:
#    for col in cell_data_pos.columns:
#       stats.boxcox(cell_num[col])
#except:
#    continue

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(311)
plt.hist(cell_num['AreaCh1'])
plt.subplot(312)
plt.hist(bc['AreaCh1'])
plt.subplot(313)
plt.hist(cell_num_bc['AreaCh1'])

#cell_num['AreaCh1'].describe()

### 8. Dimensionality Reduction by PCA
from sklearn.decomposition import PCA
dr = PCA()

# 分數矩陣
cell_pca = dr.fit_transform(cell_num)
cell_pca

# 負荷矩陣
# 前十個主成份與58個原始變數的關係
dr.components_[:10] # [:10] can be removed.
type(dr.components_) # numpy.ndarray
dr.components_.shape # (58, 58)

# 陡坡圖(scree plot)決定取幾個主成份
dr.explained_variance_ratio_
import matplotlib.pyplot as plt
plt.plot(range(1, 59), dr.explained_variance_ratio_, '-o')
plt.xlabel('# of components')
plt.ylabel('ratio of variance explained')

# list(range(1,59))
# range(1,59).tolist() # AttributeError: 'range' object has no attribute 'tolist'

# 可能可以降到五維空間中進行後續分析
cell_dr = cell_pca[:,:5]
cell_dr

### 9. Feature Selection by Correlation Filtering

def find_correlation(df, thresh=0.9):
    """
    Given a numeric pd.DataFrame, this will find highly correlated features,
    and return a list of features to remove
    params:
    - df : pd.DataFrame
    - thresh : correlation threshold, will remove one of pairs of features with a correlation greater than this value
    """
    
    corrMatrix = df.corr()
    corrMatrix.loc[:,:] =  np.tril(corrMatrix, k=-1) # 取下三角矩陣

    already_in = set() # 集合結構避免重複計入相同元素
    result = []

    for col in corrMatrix:
        perfect_corr = corrMatrix[col][corrMatrix[col] > thresh].index.tolist() # Index物件轉為list
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr)) # 更新集合
            perfect_corr.append(col)
            result.append(perfect_corr)

    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    return select_flat


select_flat = find_correlation(cell_num, 0.75) # 58 - 32 = 26
select_flat
len(select_flat) # 32


np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
np.tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
np.tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], 1)
np.tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], 0)







