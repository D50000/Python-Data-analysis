# 12.載入R語言datasets套件中的紐約空氣品質資料集airquality，並檢視其整體與行列的遺缺狀況。


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 19:33:44 2019

@author: eddiehsu
"""

import pandas as pd

import statsmodels

import statsmodels.api as sm

res = sm.datasets.get_rdataset("airquality")
type(res)

air = res.data

air.isnull()

import seaborn as sns
sns.heatmap(air.isnull())

## 
air.isnull().sum()
## 
air.isnull().sum(axis=1)

# import pyhon visuallization package
import missingno as msno

msno.matrix(air)