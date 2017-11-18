# remember to reactivate tensor flow when you open a new terminal window

import os
import pandas as pd
import numpy as np
import math
import keras

#Read in data
file_path = '/Users/josephhiggins/Documents/CS 229/Project/Input Data/'
file_name = 'dota2_pro_match_input_data_train.pkl'
df_train = pd.read_pickle(file_path + file_name)

file_name = 'dota2_pro_match_input_data_dev.pkl'
df_dev = pd.read_pickle(file_path + file_name)

#create train/dev set
X_train = df_train.drop({'radiant_win', 'match_id'}, axis = 1)
y_train = df_train['radiant_win']

X_dev = df_dev.drop({'radiant_win', 'match_id'}, axis = 1)
y_dev = df_dev['radiant_win']


########################
#train a neural net
from keras.models import Sequential
from keras.layers import Dense, Activation

# questions: what are units
NUM_OUTPUT_UNITS = 1
NUM_VARIABLES = np.shape(X_train)[1]

model = Sequential()
model.add(Dense(units=NUM_OUTPUT_UNITS, input_dim=NUM_VARIABLES, activation='relu')) 
model.add(Dense(math.floor(np.shape(X_train)[1]/2), input_dim=NUM_VARIABLES, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train.values, y_train, epochs=50, batch_size=10)

# Test
train_error = model.evaluate(X_train.values, y_train)
dev_error = model.evaluate(X_dev.values, y_dev)
print(dev_error)

model.metrics_names

##########################