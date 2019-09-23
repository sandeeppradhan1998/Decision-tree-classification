# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 04:24:25 2019

@author: Dilip
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#import data set
iris_dataset= pd.read_csv('iris.csv')
x=iris_dataset.iloc[:,:-1].values
y=iris_dataset.iloc[:,4].values

#spit dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# Fitting Random Forest Classification to the dataset
from sklearn.tree import DecisionTreeClassifier
regressor = DecisionTreeClassifier()
regressor.fit(x_train, y_train)

#predict frome the trained model
x_pred=regressor.predict(x_test)


#import confusion matrics
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,x_pred))
print(('\n'))
print(classification_report(y_test,x_pred))
