# 5.載入numpy模組並簡記為np，將上題的兩個串列轉成numpy的陣列物件，再用二元的加號運算子進行運算，並將結果與上題比較。


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 19:37:04 2019

@author: eddiehsu
"""

### Practice 5

#import library
import numpy as np

# the constructor function for numnpy array
a = [1, 2, 3]
b = [4, 5, 6]

a = np.array(a)

b = np.array(b)

c = a + b