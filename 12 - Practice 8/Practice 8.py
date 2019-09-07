# 8.參照chart.png，練習以橫列或縱行為導向、以及運用字典或是串列的原生資料結構，建立中間的pandas DataFrame表，請思考有無領悟到其間的差異。



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 18:09:38 2019

@author: eddiehsu
"""

### Practice 8

#import library
import pandas as pd

# From row-oriented dictionary
sales = [{'account' : 'Jones LLC', 'Jan' : 150, 'Feb' : 200, 'Mar' : 140},
         {'account' : 'Alpha Co', 'Jan' : 200, 'Feb' : 210, 'Mar' : 215},
         {'account' : 'Blue Inc', 'Jan' : 50, 'Feb' : 90, 'Mar' : 95}]

df = pd.DataFrame(sales)

########################

# From row-oriented list
sales1 = [('Jones LLC', 150, 200, 140),
         ('Alpha Co', 200, 210, 215),
         ('Blue Inc', 50, 90, 95)]

labels = ['account', 'Jan', 'Feb', 'Mar']
df1 = pd.DataFrame(sales1, columns = labels)

########################

# From column-oriented dictionary
sales2 = {'account' : ['Jones LLC', 'Alpha Co', 'Blue Inc'],
         'Jan' : [150, 200, 50],
         'Feb' : [200, 210, 90],
         'Mar' : [140, 215, 95]}

df2 = pd.DataFrame(sales2)
df2 = df2[['account', 'Jan', 'Feb', 'Mar']]

########################

# From colum-oriented list
sales3 = [('account', ['Jones LLC', 'Alpha Co', 'Blue Inc']),
         ('Jan', [150, 200, 50]),
         ('Feb', [200, 210, 90]),
         ('Mar', [50, 215, 95])]

df3 = pd.DataFrame.from_dict(dict(sales3))
