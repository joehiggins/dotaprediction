#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:10:48 2017

@author: josephhiggins
"""

import os
import pandas as pd
import numpy as np
import math
import scipy as sp
from sklearn import svm
from matplotlib import pyplot as plt

#Read in data
file_path = '/Users/josephhiggins/Documents/CS 229/Project/Input Data/'
file_name = 'dota2_pro_match_input_data_train1.pkl'
df_train1 = pd.read_pickle(file_path + file_name)
file_name = 'dota2_pro_match_input_data_train2.pkl'
df_train2 = pd.read_pickle(file_path + file_name)
df_train = df_train1.append(df_train2)

file_name = 'dota2_pro_match_input_data_dev.pkl'
df_dev = pd.read_pickle(file_path + file_name)

#Split data into features and labels
X_train = df_train.drop({'radiant_win', 'match_id', 'start_date'}, axis = 1)
y_train = df_train['radiant_win']

X_dev = df_dev.drop({'radiant_win', 'match_id', 'start_date'}, axis = 1)
y_dev = df_dev['radiant_win']

#Run model
#clf = svm.SVC()
clf = svm.LinearSVC(C = 2)

clf.fit(X_train, y_train)
train_predictions = clf.predict(X_train)
dev_predictions = clf.predict(X_dev)

train_output = pd.DataFrame({
         'prediction': list(train_predictions)
        ,'actual': list(y_train)
})

dev_output = pd.DataFrame({
         'prediction': list(dev_predictions)
        ,'actual': list(y_dev)
})

train_output['correct'] = train_output['prediction'] == train_output['actual']
dev_output['correct'] = dev_output['prediction'] == dev_output['actual']
train_correct_predictions = np.sum(train_output['correct'] == True)
dev_correct_predictions = np.sum(dev_output['correct'] == True)
train_size = np.shape(train_output)[0]
dev_size = np.shape(dev_output)[0]

train_pct_correct = train_correct_predictions/train_size
dev_pct_correct = dev_correct_predictions/dev_size

#Run Test
file_path = '/Users/josephhiggins/Documents/CS 229/Project/Input Data/'
file_name = 'dota2_pro_match_input_data_test.pkl'
df_test = pd.read_pickle(file_path + file_name)

X_test = df_test.drop({'radiant_win', 'match_id', 'start_date'}, axis = 1)
y_test = df_test['radiant_win']

test_predictions = clf.predict(X_test)

test_output = pd.DataFrame({
         'prediction': list(test_predictions)
        ,'actual': list(y_test)
})

test_output['correct'] = test_output['prediction'] == test_output['actual']
test_correct_predictions = np.sum(test_output['correct'] == True)
test_size = np.shape(test_output)[0]
test_pct_correct = test_correct_predictions/test_size

print("Train acc: " +str(train_pct_correct))
print("Dev acc: " +str(dev_pct_correct))
print("Test acc: " +str(test_pct_correct))

#histogram by date
test_output['start_date'] = list(df_test['start_date'])
May = test_output[(test_output['start_date'] >= '2017-05-01') & (test_output['start_date'] < '2017-06-01')]
Jun = test_output[(test_output['start_date'] >= '2017-06-01') & (test_output['start_date'] < '2017-07-01')]
Jul = test_output[(test_output['start_date'] >= '2017-07-01') & (test_output['start_date'] < '2017-08-01')]
Aug = test_output[(test_output['start_date'] >= '2017-08-01') & (test_output['start_date'] < '2017-09-01')]
Sep = test_output[(test_output['start_date'] >= '2017-09-01') & (test_output['start_date'] < '2017-10-01')]
Oct = test_output[(test_output['start_date'] >= '2017-10-01') & (test_output['start_date'] < '2017-11-01')]

May_correct = May['prediction'] == May['actual']
Jun_correct = Jun['prediction'] == Jun['actual']
Jul_correct = Jul['prediction'] == Jul['actual']
Aug_correct = Aug['prediction'] == Aug['actual']
Sep_correct = Sep['prediction'] == Sep['actual']
Oct_correct = Oct['prediction'] == Oct['actual']

output = {}
May_pct_correct = np.sum(May_correct == True)/May.shape[0]
Jun_pct_correct = np.sum(Jun_correct == True)/Jun.shape[0]
Jul_pct_correct = np.sum(Jul_correct == True)/Jul.shape[0]
Aug_pct_correct = np.sum(Aug_correct == True)/Aug.shape[0]
Sep_pct_correct = np.sum(Sep_correct == True)/Sep.shape[0]
Oct_pct_correct = np.sum(Oct_correct == True)/Oct.shape[0]

output['May'] = np.sum(May_correct == True)/May.shape[0]
output['Jun'] = np.sum(Jun_correct == True)/Jun.shape[0]
output['Jul'] = np.sum(Jul_correct == True)/Jul.shape[0]
output['Aug'] = np.sum(Aug_correct == True)/Aug.shape[0]
output['Sep'] = np.sum(Sep_correct == True)/Sep.shape[0]
output['Oct'] = np.sum(Oct_correct == True)/Oct.shape[0]


'''
#Sparse version
X_train_sparse = sp.sparse.csr_matrix(X_train)
X_dev_sparse = sp.sparse.csr_matrix(X_dev)

#clf = svm.LinearSVC()
clf = svm.SVC(kernel='linear',C=0.5)

clf.fit(X_train_sparse, y_train)
predictions = clf.predict(X_dev_sparse)

output = pd.DataFrame({
         'prediction': list(predictions)
        ,'actual': list(y_dev)
})

output['correct'] = output['prediction'] == output['actual']
correct_predictions = np.sum(output['correct'] == True)
dev_size = np.shape(output)[0]

pct_correct = correct_predictions/dev_size
pct_correct
'''