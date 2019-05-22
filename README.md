Python Data analysis

ps: 
dir(); //return all methods.

# 1.Environment and CLI
IDE: Anaconda > conda
???
native: python + library package > pip

# 2.Python native data structures

List: similar with simple List
labels = ['account', 'Jan', 'Feb', 'Mar']

Tuple: similar with Array
nested_tup = (4, 5, 6), (7, 8)

Dict: similar with Object
dictionary = {'account' : 'Jones LLC', 'Jan' : 150, 'Feb' : 200, 'Mar' : 140}

set:

ps: 
pandas library
DataFrame: similar with Table

numpy: Array

# 3.Loading files and importing library
// import library
import pandas as pd

// import specific methods
from datetime import datetime, date, time

// read file
dat = pd.read_csv('./home/user/file.tab', sep = '\t')

Numpy
Scipy
matplotlib
stasmodels
pandas
scikit-learn
TensorFlow

# 4.Function define
def add_and_maybe_multiply(a, b, c=None):
    result = a + b
    if c is not None:
        result = result * c
    return result

# 5.Sting Methods
str[1:5]



