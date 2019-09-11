# 19.請將電信業常用的客戶流失建模資料集churn.csv載入Python環境中:
#   A. 資料理解與遺缺值辨識
#   B. 挑出訓練集(training set)；
#   C. 建立屬性矩陣(feature matrix)；




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 20:10:42 2019

@author: eddiehsu
"""

## Practice 19

## import library
import pandas as pd
## read file
churn = pd.read_csv('/Users/eddiehsu/Developer/GitHub/Python-Data-analysis/15 - Practice 19/churn.csv')

##A. 資料理解與遺缺值辨識
churn.isnull().sum()

##B. 挑出訓練集(training set)；
churn['case'].value_counts()

train = churn.loc[churn['case'] == 'train',: ]

##C. 建立屬性矩陣(feature matrix)；
train.columns
trainX = train.drop(['case', 'churn'], axis=1)

##D. 區分類別與數值屬性；
trainX.columns
trainXcat = ['state',
             'area_code',
             'international_plan',
             'voice_mail_plan']
trainXnum = trainX.drop(trainXcat, axis=1)

## visuallization graphic
import seaborn as sns
sns.pairplot(trainXnum)

##E. 建立類別標籤向量(class label)；
target = train.churn


##F. 低變異過濾(low variance filter)；
##G. 偏斜(skewed)分佈屬性Box-Cox轉換；
##H. 主成份分析維度縮減(dimensionality reduction)；
##I. 高相關過濾。
##J. K-Means集群
from sklearn.cluster import KMeans
cl = KMeans(n_clusters=5)
cl.fit_transform(trainXnum)

dir(cl)
cl.labels_




