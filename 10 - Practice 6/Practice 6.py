# 6.延續上題#5，運用dot方法將兩numpy陣列物件進行點積運算。



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 19:56:01 2019

@author: eddiehsu
"""

### Practice 6

#import library
import numpy as np

# the constructor function for numnpy array
a = [1, 2, 3]
b = [4, 5, 6]

a = np.array(a)
b = np.array(b)

type(a)
dir(a)
help(a.dot)

a.dot(b)

c = np.eye(2)
d = np.ones((2, 2)) * 2
c.dot(d)

np.ones((3,4))
np.zeros((6,4))

np.ones((3,4,2))
    