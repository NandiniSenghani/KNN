# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 18:50:58 2020

@author: Nandini senghani
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNC 
from sklearn.model_selection import train_test_split
import seaborn as sns

glass =pd.read_csv("glass.csv")
glass.describe()
glass.head()
glass.isnull().sum()
glass.isna().sum()
glass.columns
sns.pairplot(data=glass)
plt.boxplot(glass.RI,1,"ro",1)
glass.RI.quantile(0.50) # 1.51768
glass.RI.quantile(0.25) # 1.51652
glass.RI.quantile(0.95) # 1.52366
glass.RI=np.where(glass.RI < 1.51652 , 1.51768,glass.RI)
glass.RI=np.where(glass.RI > 1.52366 , 1.51768,glass.RI)
plt.boxplot(glass.Na,1,"ro",1)
glass.Na.quantile(0.50) # 13.3
glass.Na.quantile(0.25) # 12.9075
glass.Na.quantile(0.95) # 14.8535
glass.Na=np.where(glass.Na < 12.9075 , 13.3,glass.Na)
glass.Na=np.where(glass.Na > 14.8535 , 13.3,glass.Na)
plt.boxplot(glass.Mg,1,"ro",1)
plt.boxplot(glass.Al,1,"ro",1)
glass.Al.quantile(0.50) # 1.36
glass.Al.quantile(0.25) # 1.19
glass.Al.quantile(0.95) # 2.394
glass.Al=np.where(glass.Al < 1.19 , 1.36,glass.Al)
glass.Al=np.where(glass.Al > 2.394 , 1.36,glass.Al)
plt.boxplot(glass.Si,1,"ro",1)
glass.Si.quantile(0.50) # 72.79
glass.Si.quantile(0.25) # 72.28
glass.Si.quantile(0.95) # 73.5175
glass.Si=np.where(glass.Si < 72.28 , 72.79,glass.Si)
glass.Si=np.where(glass.Si > 73.5175 , 72.79,glass.Si)
plt.boxplot(glass.K,1,"ro",1)
glass.K.quantile(0.50) #0.555
glass.K.quantile(0.95) #0.76
glass.K=np.where(glass.K > 0.76 , 0.555,glass.K)
plt.boxplot(glass.Ca,1,"ro",1)
glass.Ca.quantile(0.50) # 8.6
glass.Ca.quantile(0.25) # 8.24
glass.Ca.quantile(0.95) # 11.5615
glass.Ca=np.where(glass.Ca < 8.24 , 8.6,glass.Ca)
glass.Ca=np.where(glass.Ca > 11.5615 , 8.6,glass.Ca)
plt.boxplot(glass.Ba,1,"ro",1)
glass.Ba.quantile(0.50) #0.0
glass.Ba.quantile(0.95) #1.57
glass.Ba=np.where(glass.Ba > 1.57 , 0.0,glass.Ba)
plt.boxplot(glass.Fe,1,"ro",1)
glass.Fe.quantile(0.50) #0.0
glass.Fe.quantile(0.95) #0.267
glass.Fe=np.where(glass.Fe > 0.267 , 0.0,glass.Fe)
plt.boxplot(glass.Type,1,"ro",1)
glass.Type.value_counts()
glass.Type.quantile(0.50) #2
glass.Type.quantile(0.95) #7
glass.Type=np.where(glass.Type > 7 , 2,glass.Type)
# While modeling  with transformations the accuracy of the model turns out to be low , So we need to contact the client for further modelling 
# Implementing the model without the above transformations for better accuracy
train,test = train_test_split(glass, test_size=0.3)
train_x=train.iloc[:,:9]
train_y=train.iloc[:,9]
test_x=test.iloc[:,:9]
test_y=test.iloc[:,9]

acc = []

# running KNN algorithm for 3 to 70 nearest neighbours(odd numbers) and 
# storing the accuracy values 
 
for i in range(3,70,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train_x,train_y)
    train_acc = np.mean(neigh.predict(train_x)==train_y)
    test_acc = np.mean(neigh.predict(test_x)==test_y)
    acc.append([train_acc,test_acc])
    
#plotting the train and test 
plt.plot(np.arange(3,70,2),[i[0] for i in acc],"bo-");plt.plot(np.arange(3,70,2),[i[1] for i in acc],"ro-");plt.legend(["train","test"])

##########################################ZOO dataset#############################
Zoo=pd.read_csv("Zoo.csv")
zoo=Zoo.drop("animal name",axis=1)
zoo.describe()
zoo.head()
zoo.isnull().sum()
zoo.isna().sum()
zoo.columns
sns.pairplot(data=zoo)
zoo.hair.value_counts().plot(kind="bar")
zoo.feathers.value_counts().plot(kind="bar")
zoo.legs.value_counts().plot(kind="bar")
zoo.type.value_counts().plot(kind="bar")
zoo.eggs.value_counts().plot(kind="bar")
zoo.milk.value_counts().plot(kind="bar")
zoo.airborne.value_counts().plot(kind="bar")
zoo.aquatic.value_counts().plot(kind="bar")
zoo.predator.value_counts().plot(kind="bar")
zoo.toothed.value_counts().plot(kind="bar")
zoo.backbone.value_counts().plot(kind="bar")
zoo.breathes.value_counts().plot(kind="bar")
zoo.venomous.value_counts().plot(kind="bar")
zoo.fins.value_counts().plot(kind="bar")
zoo.domestic.value_counts().plot(kind="bar")
zoo.catsize.value_counts().plot(kind="bar")
train,test = train_test_split(zoo, test_size=0.3)
train_x=train.iloc[:,:16]
train_y=train.iloc[:,16]
test_x=test.iloc[:,:16]
test_y=test.iloc[:,16]

acc2 = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values 
 
for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train_x,train_y)
    train_acc2 = np.mean(neigh.predict(train_x)==train_y)
    test_acc2 = np.mean(neigh.predict(test_x)==test_y)
    acc2.append([train_acc2,test_acc2])
  #plotting the train and test   
plt.plot(np.arange(3,50,2),[i[0] for i in acc2],"bo-");plt.plot(np.arange(3,50,2),[i[1] for i in acc2],"ro-");plt.legend(["train","test"])

