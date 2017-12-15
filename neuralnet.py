# remember to reactivate tensor flow when you open a new terminal window

import os
import pandas as pd
import numpy as np
import math
import keras

CROSS_TRAIN = False
NUM_CHUNKS = 10
#Read in data 
file_path = '/Users/petragrutzik/CSClasses/CS229/dota/dotaprediction/Input Data/' #diff btw input data and pro match details

# #create train/dev set without cross train
# file_name = 'dota2_pro_match_input_data_train.pkl'
# df_train = pd.read_pickle(file_path + file_name)

# file_name = 'dota2_pro_match_input_data_dev.pkl'
# df_dev = pd.read_pickle(file_path + file_name)

# X_train = df_train.drop({'radiant_win', 'match_id'}, axis = 1)
# y_train = df_train['radiant_win']

# X_dev = df_dev.drop({'radiant_win', 'match_id'}, axis = 1)
# y_dev = df_dev['radiant_win']

#create train dev set for cross train
file_name = 'dota2_pro_match_input_data_all.pkl'
df_all = pd.read_pickle(file_path + file_name)
length_chunk = math.floor(len(df_all)/NUM_CHUNKS)

########################
#train a neural net
from keras.models import Sequential
from keras.layers import Dense, Activation

NUM_OUTPUT_UNITS = 1
NUM_VARIABLES = np.shape(df_all)[1] - 2 # because to train, we will later drop 2 columns:'radiant_win', 'match_id'

model = Sequential()
model.add(Dense(units=NUM_VARIABLES, input_dim=NUM_VARIABLES, activation='relu')) 
model.add(Dense(math.floor(NUM_VARIABLES/2), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

if CROSS_TRAIN == True:
    num_rounds_cross_train = NUM_CHUNKS
else:
    num_rounds_cross_train = 1
for i in range(num_rounds_cross_train):
    df_dev = df_all[i:i + length_chunk] 
    df_train = df_all[i + length_chunk+1:] 
    X_train = df_train.drop({'radiant_win', 'match_id'}, axis = 1)
    y_train = df_train['radiant_win']

    X_dev = df_dev.drop({'radiant_win', 'match_id'}, axis = 1)
    print(X_train)
    y_dev = df_dev['radiant_win']

    # Fit the model
    model.fit(X_train.values, y_train, epochs=50, batch_size=10)

    # Test
    train_error = model.evaluate(X_train.values, y_train)
    dev_error = model.evaluate(X_dev.values, y_dev)
    print(dev_error)

    dev_predictions = model.predict(X_dev.values)
    dev_predictions = [round(x[0]) for x in dev_predictions]
    print(dev_predictions)
    model.metrics_names


# Pick the model Mi with the lowest estimated generalization error, and
# retrain that model on the entire training set S. The resulting hypothesis
# is then output as our final answer.

##########################