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
from sklearn.preprocessing import Imputer

#Read in data
file_path = '/Users/josephhiggins/Documents/CS 229/Project/Input Data/'
file_name = 'dota2_pro_match_input_data_train.pkl'
df_train = pd.read_pickle(file_path + file_name)

file_name = 'dota2_pro_match_input_data_dev.pkl'
df_dev = pd.read_pickle(file_path + file_name)

#Split data into features and labels
X_train = df_train.drop({'radiant_win', 'match_id'}, axis = 1)
y_train = df_train['radiant_win']

X_dev = df_dev.drop({'radiant_win', 'match_id'}, axis = 1)
y_dev = df_dev['radiant_win']


#Run model
#clf = svm.SVC()
clf = svm.LinearSVC()

'''
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp.fit(X_train)
train_imp = imp.transform(X_train)
dev_imp = imp.transform(X_dev)

clf.fit(train_imp, y_train)
predictions = clf.predict(dev_imp)
'''
clf.fit(X_train, y_train)
predictions = clf.predict(X_dev)

output = pd.DataFrame({
         'prediction': list(predictions)
        ,'actual': list(y_dev)
})

output['correct'] = output['prediction'] == output['actual']
correct_predictions = np.sum(output['correct'] == True)
dev_size = np.shape(output)[0]

pct_correct = correct_predictions/dev_size
pct_correct


#Sparse version
X_train_sparse = sp.sparse.csr_matrix(X_train)
X_dev_sparse = sp.sparse.csr_matrix(X_dev)

#clf = svm.LinearSVC()
clf = svm.SVC(kernel='linear',C=2)

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
