# -*- coding: utf-8 -*-
"""
Created on Wed May  3 10:36:01 2017

@author: Administrator
读取数据选取特征
"""
import pandas as pd
import os
from sklearn.svm import SVC
import numpy as np


inputfile= 'F:/研究生/PR_dataset/train/train.csv'
inputfile2= 'F:/研究生/PR_dataset/test/test.csv'

pwd = os.getcwd()
os.chdir(os.path.dirname(inputfile))
train= pd.read_csv(os.path.basename(inputfile))#reading train data from inputfile
os.chdir(pwd)

pwd = os.getcwd()
os.chdir(os.path.dirname(inputfile2))
test=pd.read_csv(os.path.basename(inputfile2))#reading test data from inputfile
os.chdir(pwd)

y_train=train['label'] #Extract the features
x_train=train.iloc[:,1:2331]  #Extract the label
y_test=test['label'] #Extract the features
x_test=test.iloc[:,1:2331] #Extract the features

svc =SVC(kernel='linear')
c_param=np.logspace(-2,2,5)
param_grid=dict(C=c_param)
from sklearn.grid_search import GridSearchCV
gs=GridSearchCV(svc,param_grid,verbose=2,refit=True,cv=3)
time_=gs.fit(x_train,y_train)
print(gs.best_params_,gs.best_score_)
print(gs.score(x_test,y_test))
