# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 09:32:14 2017

@author: Administrator
"""
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

inputfile= 'F:/研究生/PR_dataset/train/train.csv'
inputfile2= 'F:/研究生/PR_dataset/test/test.csv'

pwd = os.getcwd()
os.chdir(os.path.dirname(inputfile)) 
train= pd.read_csv(os.path.basename(inputfile)) #reading train data from inputfile
os.chdir(pwd)

pwd = os.getcwd()
os.chdir(os.path.dirname(inputfile2))
test=pd.read_csv(os.path.basename(inputfile2))#reading test data from inputfile
os.chdir(pwd)

y_train=train['label'] #Extract the label
x_train=train.iloc[:,1:2331] #Extract the features
y_test=test['label'] 
x_test=test.iloc[:,1:2331] 

model=Sequential()#model initial

model.add(Dense(300, input_dim=2330, init='uniform'))#2330 input，Hidden layer has 200 unit
model.add(Activation('tanh')) #Hidden layer activate function is tanh
model.add(Dropout(0.5))
model.add(Dense(1, init='uniform')) #1 output
model.add(Activation('sigmoid'))#output layer activate function is sigmoid

sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True) #Using gradient descent algorithm 
model.compile(loss='mean_squared_error',optimizer=sgd,metrics=["accuracy"]) #Compile model 
model.fit(np.array(x_train),np.array(y_train),nb_epoch=35,batch_size=50)
loss, accuracy=model.evaluate(np.array(x_test),np.array(y_test),verbose=1)
print("\n")
print("Accuracy = {:.2f}".format(accuracy)) 
print("loss=",loss)

predit_y=model.predict(np.array(x_test))
fpr, tpr, thresholds = roc_curve(y_test, predit_y, pos_label=1) #calculate the roc curve
from sklearn.metrics import auc
print("Auc=",auc(fpr, tpr)) #calculate the auc
plt.plot(fpr,tpr) #draw the roc curve

from sklearn.metrics import confusion_matrix #import confusion matrix
predit_y=model.predict_classes(np.array(x_test)).reshape(len(y_test))

cm = confusion_matrix(y_test,predit_y) #create confusion matrix
plt.matshow(cm, cmap=plt.cm.Greens) #draw confusion matrix
plt.colorbar() #color label
  
for x in range(len(cm)): 
    for y in range(len(cm)):
        plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
  
plt.xlabel('Predicted label') 
plt.ylabel('True label') 
plt.show()
print(classification_report(y_test,predit_y)) #print the report of classification

