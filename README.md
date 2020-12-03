# Python Data analysis
Introduce and practice Python in "Data analysis".


## 1. Environment and CLI setup
native: python + library package use `pip` command.  
IDE: Anaconda use `conda` command. (It bundle all the useful tool for data analysis)


## 2. Python native data structures
Every value in Python has a datatype. Since everything is an `object` in Python programming, data types are actually classes and variables are instance (object) of these classes.

<details>
<summary> Number（數字）</summary>

```python
x = 20
print(x) #print: 20
print(type(x)) #print: <class 'int'>
```
</details>

<details>
<summary> String（字符串）: write-protect </summary>

```python
x = "Hello"
print([0:2]) #print: He
print(type(x)) #print: <class 'str'>
```
</details>

<details>
<summary> List（列表）: List: similar with simple List in Java. </summary>

```python
x = ["apple", "banana", "cherry"]
print(x[1]) #print: banana
print(type(x)) #print: <class 'list'>
```
</details>

<details>
<summary> Tuple（元组）: write-protect version of List. </summary>

```python
x = (5,'program', 1+3j)
print(x[1:3]) #print: ('program', (1+3j))
print(type(x)) #print: <class 'tuple'>
```
</details>

<details>
<summary> Set（集合）: Set is an unordered collection of unique items. </summary>

```python
x = {"apple", "banana", "cherry"}
print(x[1:3]) #print: {"apple", "banana", "cherry"}
print(type(x)) #print: <class 'set'>
```
</details>

<details>
<summary> Dictionary（字典）: Dict: similar with Object </summary>

```python
x = {'dict1': {'innerkey': 'value'}}
print(x['dict1']['innerkey']) #print: value
print(type(x)) #print: <class 'dict'>
```
</details>
  
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
  

## 3. Loading files and importing library
Package, Module, function import examples  
https://pyliaorachel.github.io/blog/tech/python/2017/09/15/pythons-import-trap.html

Import library:  
```import pandas as pd```

Import specific methods:  
```from datetime import datetime, date, time```

Read file:  
```dat = pd.read_csv('./home/user/file.tab', sep = '\t')```
  
<details>
<summary>Library Examples:</summary>

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
</details>

## 4. if...else Statement
 - if...elif...else
 - if x else y (**Ternary Conditional Operator**)
 

## 5. For Loop
 - For Loop
 - While Loop
 - **break** and **continue** statement


## 6. Function Define
 - Declare a python function
 - Lambda Function in python (**Anonymous**)
 - *args vs **kwargs
```
def foo(*args, **kwargs):
    print 'args = ', args
    print 'kwargs = ', kwargs
    print '---------------------------------------'

if __name__ == '__main__':
    foo(1,2,3,4)
    foo(a=1,b=2,c=3)
    foo(1,2,3,4, a=1,b=2,c=3)
    foo('a', 1, None, a=1, b='2', c=3)
    
# Output:
args =  (1, 2, 3, 4) 
kwargs =  {} 
--------------------------------------- 
args =  () 
kwargs =  {'a': 1, 'c': 3, 'b': 2} 
--------------------------------------- 
args =  (1, 2, 3, 4) 
kwargs =  {'a': 1, 'c': 3, 'b': 2} 
--------------------------------------- 
args =  ('a', 1, None) 
kwargs =  {'a': 1, 'c': 3, 'b': '2'} 
---------------------------------------

# Using **kwargs for creating a dictionary.
def kw_dict(**kwargs):
   return kwargs
print kw_dict(a=1,b=2,c=3) == {'a':1, 'b':2, 'c':3}
```
Use ```*args``` and ```**kwargs``` at the same function. Must put *args at first place. Or will show “SyntaxError: non-keyword arg after keyword arg”.


## 7~15. Python Data Analysis Practice
**Practice sheet:** Python-Data-analysis/data/On Practices Python Data Analysis Basics.docx




### Reference
https://www.programiz.com/
