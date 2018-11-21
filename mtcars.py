# -*- coding: utf-8 -*-
"""
Created on Thu May 31 20:22:01 2018

@author: sunka
"""

import pandas as pd

data = pd.read_csv('mtcars.csv')

print(data)

data.shape

data.columns

model = data['model']

print(model)

from sklearn import preprocessing

encode = preprocessing.LabelEncoder()

data['model'] = encode.fit_transform(data['model'])

print(data)

x = data.iloc[:,1:]
y = data.iloc[:,:1]

from sklearn import model_selection

train_data, test_data, train_target, test_target = model_selection.train_test_split(x,y)

from sklearn import neighbors

train_data = preprocessing.normalize(train_data)
test_data = preprocessing.normalize(test_data)
train_target = preprocessing.normalize(train_target)
test_target = preprocessing.normalize(test_target)

data.shape

supervised = neighbors.KNeighborsClassifier(n_neighbors = 5)
fitting = supervised.fit(train_data,train_target)
result = supervised.predict(test_data)

from matplotlib import pyplot

pyplot.hist(result)
pyplot.hist(test_target, color = 'orange')

pyplot.scatter(result,test_target)

from sklearn import metrics

varience = metrics.r2_score(result,test_target)
print("Varience is " + str(varience))

accu = metrics.accuracy_score(result,test_target)
print("Accuracy is "+str(round(accu*100))+" %")

