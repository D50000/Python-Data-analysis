# 7.載入TRD_Index.txt，對資料集中上海綜合證券指數收益率與深圳綜合證券指數收益率進行相關性分析。


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 20:09:36 2019

@author: eddiehsu
"""


### Practice 7

#import library
import pandas as pd
stock = pd.read_csv('/Users/eddiehsu/Developer/GitHub/Python-Data-analysis/11 - Practice 7/TRD_Index.txt', sep ='\t')
stock.index
stock.columns

# count the stock transaction times
stock.Indexcd
stock.Indexcd.value_counts()

SH = stock[stock.Indexcd == 1]
SZ = stock[stock.Indexcd == 399903]

from matplotlib import pyplot as plt

SH.columns
plt.scatter(SH.Retindex, SZ.Retindex)
plt.xlabel('SH Return Index')
plt.ylabel('SH Return Index')