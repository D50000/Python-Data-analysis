# Python Data analysis
Introduce and practice Python in "Data analysis".


## 1.Environment and CLI setup
native: python + library package use `pip` command.  
IDE: Anaconda use `conda` command. (It bundle all the useful tool for data analysis)


## 2.Python native data structures
Every value in Python has a datatype. Since everything is an `object` in Python programming, data types are actually classes and variables are instance (object) of these classes.

- Number（数字）
- String（字符串）: write-protect
- List（列表）: List: similar with simple List in Java.
- Tuple（元组）: write-protect version of List.
- Set（集合）: Set is an unordered collection of unique items. 
- Dictionary（字典）: Dict: similar with Object


## 3.Loading files and importing library
Import library:
import pandas as pd

Import specific methods:
from datetime import datetime, date, time

Read file:
dat = pd.read_csv('./home/user/file.tab', sep = '\t')

ps: 
pandas library
DataFrame: similar with Table

numpy: Array
labels = ['account', 'Jan', 'Feb', 'Mar']

Tuple: similar with Array
nested_tup = (4, 5, 6), (7, 8)



Numpy
Scipy
matplotlib
stasmodels
pandas
scikit-learn
TensorFlow

// drawing graphic 
matplotlib.pyplot
pandas DataFrame
seaborn newer package
ggplot more powerful
bokeh, plotly for interactive graphic


## 4.Function define
def add_and_maybe_multiply(a, b, c=None):
    result = a + b
    if c is not None:
        result = result * c
    return result

str[1:5]



### Reference
https://www.programiz.com/
