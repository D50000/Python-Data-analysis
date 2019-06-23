import numpy as np
a = np.array([0, 0.5, 1.0, 1.5, 2.0])
type(a)
# normal way to import library

# Import specific methods
### pandas data structures
from pandas import Series, DataFrame
import pandas as pd
import numpy as np

obj = Series([4, 7, -5, 3])
obj

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'], 'year': [2000, 2001, 2002, 2001, 2002], 'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)
frame