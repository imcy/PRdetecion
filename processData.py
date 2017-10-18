# -*- coding: utf-8 -*-
"""
Created on Tue May  2 19:17:21 2017

@author: imcy
Reading all txt file and save in csv 
"""

import pandas as pd

temp1="F:/研究生/PR_dataset/train/pos/"
temp2="F:/研究生/PR_dataset/train/neg/"
outputfile= 'F:/研究生/PR_dataset/train/train_g.csv'
temp3=".txt"
temp4=range(0,200)
new=[]
new2=[]
#Read positive samples
for i in temp4:
    new.append([])
    filename=temp1+str(i)+temp3
    f = open(filename)             # open the txt file
    lines = f.readlines() 
    for line in lines:
        line=line.strip('\n') #Remove each line of line breaks
        new[i].append(float(line))
  
df=pd.DataFrame(new)
df['label']=1 #Add positive sample label
temp5=range(0,500)
#Read negative samples
for i in temp5:
    new2.append([])
    filename=temp2+str(i)+temp3
    f = open(filename)             
    lines = f.readlines() 
    for line in lines:
        line=line.strip('\n') 
        new2[i].append(float(line)) #Change the string type to the float type and add it to the list
  
df2=pd.DataFrame(new2) #Turn the data into dataframe
df2['label']=0#Add negative sample label

frames=[df,df2]
data_all=pd.concat(frames)
data_all=(data_all - data_all.min())/(data_all.max() - data_all.min())#minimum - maximum normalization
data_all.to_csv(outputfile)
