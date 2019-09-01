# 3.練習載入定位符分隔的檔案data1.tab，載入過程與逗號分隔檔案.csv完全相同。




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 19:35:32 2019

@author: eddiehsu
"""


### Practice 3

#import library
import pandas as pd
dat = pd.read_csv('/Users/eddiehsu/Developer/GitHub/temp/data.tab', sep = '\ ')