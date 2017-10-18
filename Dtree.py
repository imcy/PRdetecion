# -*- coding: utf-8 -*-
"""
Created on Wed May  3 10:36:01 2017

@author: Administrator
读取数据选取特征
"""
import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier #导入决策树
from sklearn import feature_selection
from sklearn.cross_validation import cross_val_score #交叉验证
import matplotlib.pyplot as plt
inputfile= 'F:/研究生/PR_dataset/train/train.csv'

pwd = os.getcwd()
os.chdir(os.path.dirname(inputfile))
train= pd.read_csv(os.path.basename(inputfile))
os.chdir(pwd)

y_train=train['label'] #提取标签
x_train=train.iloc[:,1:2331] #提取所有属性值

dtc=DecisionTreeClassifier(criterion='entropy')
percentile=range(1,100,2)

result=[]
for i in percentile:
    fs=feature_selection.SelectPercentile(feature_selection.chi2,percentile=i)
    X_train_fs=fs.fit_transform(x_train,y_train)
    scores=cross_val_score(dtc, X_train_fs, y_train, cv=5)
    result=np.append(result,scores.mean())
print (result)
opt=np.where(result==result.max())[0]
print ('Optimal number of features %d' %percentile[int(opt)])

plt.plot(percentile,result)
plt.xlabel('percentiles of features')
plt.ylabel('accuracy')
plt.show()