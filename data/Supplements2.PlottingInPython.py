########################################################## Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CADS(資訊與決策科學研究所暨資料科學應用研究中心), NTUB(國立臺北商業大學); the CARS(中華R軟體學會); and the DSBA(臺灣資料科學與商業應用協會)
#######################################################
### Datasets: 唐詩三百首.csv,
### Notes: This code is provided without warranty.

### pyplot style
# Import the required packages,
# with their conventional names
import matplotlib.pyplot as plt
import numpy as np

# Generate the data
x = np.arange(0, 10, 0.2)
y = np.sin(x)

# Generate the plot
plt.plot(x, y)

# Display it on the screen
plt.show()


### Pythonic (more object-oriented, more verbose) style
# Import the required packages,
# with their conventional names
import matplotlib.pyplot as plt
import numpy as np

# Generate the data
# https://stackoverflow.com/questions/45853595/spyder-clear-variable-explorer-along-with-variables-from-memory
# %reset # Go to the IPython console in the Spyder IDE and type %reset. It will prompt you to enter (y/n) as the variables once deleted cannot be retrieved. Type 'y' and hit enter. That's it.
x = np.arange(0, 10, 0.2)
y = np.sin(x)

# Generate the plot
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x, y)

# Display it on the screen
plt.show()

### A Matlab-like coding style
# Import the required packages,
# with their conventional names
from pylab import * # namespace conflicting

# Generate the data
x = np.arange(0, 10, 0.2)
y = np.sin(x)

# Generate the plot
plot(x, y)

# Display it on the screen
show()

### A figure with two plots above each other
# Import the required packages
import matplotlib.pyplot as plt
import numpy as np

# Generate the data
x = np.arange(0, 10, 0.2)
y = np.sin(x)
z = np.cos(x)

# Generate the figure and the axes
fig, axs = plt.subplots(nrows=2, ncols=1)

# On the first axis, plot the sine and label the ordinate
axs[0].plot(x, y) # high-level plotting
axs[0].set_ylabel('Sine') # low-level plotting

# On the second axis, plot the cosine
axs[1].plot(x, z)
axs[1].set_xlabel('Indep. Var.') # low-level plotting
axs[1].set_ylabel('Cosine')

# Display the resulting plot
plt.show()

dir(plt)
help(plt.scatter)

### Reference: Haslwanter, Thomas (2016), An Introductionto Statistics with Python with Applications in the Life Sciences, Springer.

### Python - dir() - how can I differentiate between functions/method and simple attributes?
# (https://stackoverflow.com/questions/26818007/python-dir-how-can-i-differentiate-between-functions-method-and-simple-att)
import math
[name for name in dir(np) if not callable(getattr(np, name))]
[name for name in dir(np) if callable(getattr(np, name))]

[(name,type(getattr(math,name))) for name in dir(math)]
help(getattr)

# filtering dir() results 
[k for k in dir(tf.train) if 'summary' in k]
list(filter(lambda x: x.startswith('(s|S)ummary'), dir(tf.train)))

[(name, 'func' if callable(getattr(math, name)) else 'attr') for name in dir(math)]

# python location on mac osx
# https://stackoverflow.com/questions/6819661/python-location-on-mac-osx
# type -a python


# How to find a Python package's dependencies
# https://stackoverflow.com/questions/29751572/how-to-find-a-python-packages-dependencies
# pip show tornado

# $ pip install pipdeptree
# $ pipdeptree

# The 4 Major Ways to Do String Formatting in Python
# https://dbader.org/blog/python-string-formatting

### Boxplot of Multiple Columns of a Pandas Dataframe on the Same Figure (https://stackoverflow.com/questions/49554139/boxplot-of-multiple-columns-of-a-pandas-dataframe-on-the-same-figure-seaborn)
import pandas as pd
cell = pd.read_csv("/Users/Vince/cstsouMac/Python/Examples/Basics/data/segmentationOriginal.csv")

print(len(cell.columns))
cell.columns
partialCell = cell[['AngleCh1', 'AreaCh1', 'AvgIntenCh1', 'AvgIntenCh2', 'AvgIntenCh3']]

# by pandas.boxplot()
partialCell.boxplot()

ax = partialCell.boxplot() # partialCell is an instance of DataFrame
fig = ax.get_figure()

# by seaborn
import seaborn as sns
sns.boxplot(x="variable", y="value", data=pd.melt(partialCell))

pd.melt(partialCell).head()

### plot histogram from pandas dataframe using the list values in (column, row) pairs (https://stackoverflow.com/questions/49225383/plot-histogram-from-pandas-dataframe-using-the-list-values-in-column-row-pair)
# Histogram of Multiple Columns of a Pandas Dataframe on the Same Figure
# by pandas.boxplot()
# **pandas**資料框也有繪製直方圖的hist()函數，以跨欄比較各量化變數的數值分佈。
partialCell.hist()
# fig = ax[0].get_figure()
# fig.savefig('./_img/pd_hist.png')

help(partialCell.hist)

# by matplotlib.pyplot (not shown in handsout)
# 以**matplotlib.pyplot**套件繪製並排直方圖稍微麻煩些，需撰寫迴圈將資料表中各個欄名取出，再將各欄數值傳入hist()方法中一一繪製，其呈現的結果亦與**pandas**的分面圖不同，而是以疊加的方式繪製直方圖。
import matplotlib.pyplot as plt
for col in partialCell.columns:
    plt.hist(partialCell[col], density=True, alpha=0.5)

# **seaborn**套件中還有許多方便檢視量化數據分佈的繪圖函數
# Visualization with Seaborn https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html
### Histograms and KDE can be combined using distplot
for col in partialCell.columns:
    sns.distplot(partialCell[col])

### We can see the joint distribution and the marginal distributions together using sns.jointplot.
tips = sns.load_dataset('tips')
tips.head()

### Faceted histograms
tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']

grid = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
grid.map(plt.hist, "tip_pct", bins=np.linspace(0, 40, 15));

### Factor plots
with sns.axes_style(style='ticks'):
    g = sns.factorplot("day", "total_bill", "sex", data=tips, kind="box")
    g.set_axis_labels("Day", "Total Bill");

### Joint distributions
with sns.axes_style('white'):
    sns.jointplot("total_bill", "tip", data=tips, kind='hex')

sns.jointplot("total_bill", "tip", data=tips, kind='reg');

#for x in partialCell.columns:
#    bins, counts = partialCell.select(x).rdd.flatMap(lambda x: x).histogram(20)
#    plt.hist(bins[:-1], bins=bins, weights=counts)
#
#
#for x in range(0, len(partialCell.columns)):
#    bins, counts = partialCell.loc[partialCell.index.map(x)].rdd.flatMap(lambda x: x).histogram(20)
#    plt.hist(bins[:-1], bins=bins, weights=counts)

### ggplot2 in Python
#Installation
#$ pip install -U ggplot
## or
#$ conda install -c conda-forge ggplot
## or
#$ pip install git+https://github.com/yhat/ggplot.git

# ImportError: cannot import name 'Timestamp'
# https://stackoverflow.com/questions/50591982/importerror-cannot-import-name-timestamp

%matplotlib inline
from ggplot import *
#import ggplot as gp
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# /Users/Vince/anaconda/lib/python3.6/site-packages/ggplot/utils.py:81: FutureWarning: pandas.tslib is deprecated and will be removed in a future version.
#You can access Timestamp as pandas.Timestamp
#  pd.tslib.Timestamp,


### ggplot port for python http://yhat.github.io/ggpy/
diamonds.head()
diamonds.dtypes
diamonds.describe(include="all")

ggplot(aes(x='price', color='clarity'), data=diamonds) + geom_density() + scale_color_brewer(type='div', palette=7) + facet_wrap('cut')

### Plotnine: another ggplot implmentation in Python
# conda install -c conda-forge plotnine in Windows
from plotnine import *
from plotnine.data import *
ggplot(aes(x='price', color='clarity'), data=diamonds) + geom_density() + scale_color_brewer(type='div', palette=7) + facet_wrap('~cut') + scale_y_continuous(limits= [0, 0.0004])

ggplot(aes(x='price', color='clarity'), data=diamonds) + geom_density() + scale_color_brewer(type='div', palette=7) + facet_wrap('~cut', ncol = 2) + ylim(0, 0.0005)
diamonds.describe(include='all')

# What's the difference between globals(), locals(), and vars()?
# https://stackoverflow.com/questions/7969949/whats-the-difference-between-globals-locals-and-vars

import matplotlib.pyplot as plt
plt.plot([1,2,3,2,3,2,2,1])
