# 9.載入pandas模組並簡記為pd，創建下列男女性觀看電視時數的DataFrame，練習連續位置取值與間斷位置取值的pandas資料框語法，並進行男女群組與摘要的計算分析。

""" example:
Gender	TV
0	f	3.4
1	f	3.5
2	m	2.6
3	f	4.7
4	m	4.1
5	m	4.1
6	f	5.1
7	m	3.9
8	f	3.7
9	m	2.1
10	m	4.3
"""




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 18:09:38 2019

@author: eddiehsu
"""

### Practice 9

#import library
import pandas as pd

tv = pd.DataFrame({'Gender' : ['f','f','m','f','m','m','f','m','f','m'], 
                   'TV' : [3,4,5,6,7,8,9,0,1,2]})

list(tv.groupby('Gender'))