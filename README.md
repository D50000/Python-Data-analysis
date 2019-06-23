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

ps:  
Extended Data Type  
pandas library  
DataFrame: similar with Table  

numpy:  
Array  
labels = ['account', 'Jan', 'Feb', 'Mar']  
  
Tuple:  
similar with Array  
nested_tup = (4, 5, 6), (7, 8)  
  

## 3.Loading files and importing library
Import library:
import pandas as pd

Import specific methods:
from datetime import datetime, date, time

Read file:
dat = pd.read_csv('./home/user/file.tab', sep = '\t')


Libray Examples:
- Numpy:
Besides its obvious `scientific uses`, NumPy can also be used as an efficient multi-dimensional container of generic data.

- Scipy:
A Python-based ecosystem of open-source software for mathematics, science, and engineering.

- matplotlib:
Matplotlib is a Python `2D plotting library` which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms.  

- statsmodels:
A Python module that provides classes and functions for the estimation of many different statistical models.
  
- pandas:
An open source, BSD-licensed library providing high-performance, easy-to-use `data structures and data analysis` tools.

- scikit-learn:
Simple and efficient tools for `data mining` and data analysis.

- TensorFlow: (Google)
An open source `machine learning library` for research and production.

- seaborn:
Seaborn is a Python data `visualization library based on matplotlib`. It provides a high-level interface for drawing attractive and informative statistical graphics.

- ggplot:
A plotting system for Python based on R's ggplot2 and the Grammar of `Graphics`.

- bokeh:
An `interactive visualization` library that targets modern web browsers for presentation.

- plotly:
Plotly's Python graphing library makes `interactive, publication-quality graphs` online.


## 4.Function define
def add_and_maybe_multiply(a, b, c=None):
    result = a + b
    if c is not None:
        result = result * c
    return result

str[1:5]



### Reference
https://www.programiz.com/
