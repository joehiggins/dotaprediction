# remember to reactivate tensor flow when you open a new terminal window

import os
import pandas as pd
import numpy as np
import math
import keras

#Read in data
#file_path = '/Users/josephhiggins/Documents/CS 229/Project/Input Data/'
file_path = '/Users/longt/Study/Stanford/2017-2018_Classes/CS229_Machine_Learning/Project/Input_Data/'
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
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras import regularizers
from keras.constraints import max_norm


NUM_OUTPUT_UNITS = 1
NUM_VARIABLES = np.shape(X_train)[1]

np.random.seed(100)


model = Sequential()
#model.add(Embedding(input_dim=2, output_dim=10,input_length=NUM_VARIABLES))
#model.add(Flatten())
model.add(Dense(units=NUM_OUTPUT_UNITS, input_dim=NUM_VARIABLES, activation='relu')) 
#model.add(Dense(units=NUM_OUTPUT_UNITS, activation='relu'))
#model.add(Dense(math.floor(NUM_VARIABLES/2), activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(300, activation='relu',
          kernel_regularizer=regularizers.l2(0.001)))
#model.add(Dropout(0.25))
#model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Experiment 1:  No regularizers
# model.add(Dense(units=NUM_OUTPUT_UNITS, input_dim=NUM_VARIABLES, activation='relu')) 
# model.add(Dense(300, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train.values, y_train, epochs=40, batch_size=10)

# Experiment 2: L2 Regularizers with factor 0.001
# model.add(Dense(units=NUM_OUTPUT_UNITS, input_dim=NUM_VARIABLES, activation='relu')) 
# model.add(Dense(300, activation='relu',
#          kernel_regularizer=regularizers.l2(0.001)))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train.values, y_train, epochs=40, batch_size=10)

# Experiment 3: L2 Regularizers with factor 0.002
# model.add(Dense(units=NUM_OUTPUT_UNITS, input_dim=NUM_VARIABLES, activation='relu')) 
# model.add(Dense(300, activation='relu',
#          kernel_regularizer=regularizers.l2(0.002)))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train.values, y_train, epochs=40, batch_size=10)

# Experiment 3: L2 Regularizers with factor 0.0005
# model.add(Dense(units=NUM_OUTPUT_UNITS, input_dim=NUM_VARIABLES, activation='relu')) 
# model.add(Dense(300, activation='relu',
#          kernel_regularizer=regularizers.l2(0.0005)))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train.values, y_train, epochs=40, batch_size=10)

# Experiment 4: Early stopping:
# model.add(Dense(units=NUM_OUTPUT_UNITS, input_dim=NUM_VARIABLES, activation='relu')) 
# model.add(Dense(300, activation='relu',
#          kernel_regularizer=regularizers.l2(0.000)))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train.values, y_train, epochs=30, batch_size=10)

# Experiment 4b: Early stopping:
# model.add(Dense(units=NUM_OUTPUT_UNITS, input_dim=NUM_VARIABLES, activation='relu')) 
# model.add(Dense(300, activation='relu',
#          kernel_regularizer=regularizers.l2(0.000)))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train.values, y_train, epochs=35, batch_size=10)

# Experiment 4b: Early stopping:
# model.add(Dense(units=NUM_OUTPUT_UNITS, input_dim=NUM_VARIABLES, activation='relu')) 
# model.add(Dense(300, activation='relu',
#          kernel_regularizer=regularizers.l2(0.000)))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train.values, y_train, epochs=25, batch_size=10)

# Experiment 5: Early stopping + regularization:
# model.add(Dense(units=NUM_OUTPUT_UNITS, input_dim=NUM_VARIABLES, activation='relu')) 
# model.add(Dense(300, activation='relu',
#          kernel_regularizer=regularizers.l2(0.001)))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train.values, y_train, epochs=30, batch_size=10)

# Experiment 6: Early stopping + regularization + dropout:
# model.add(Dense(units=NUM_OUTPUT_UNITS, input_dim=NUM_VARIABLES, activation='relu')) 
# model.add(Dropout(0.20))
# model.add(Dense(300, activation='relu',
#          kernel_regularizer=regularizers.l2(0.001)))
# model.add(Dropout(0.20))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train.values, y_train, epochs=30, batch_size=10)

# Experiment 6b: Early stopping + regularization + dropout:
# model.add(Dense(units=NUM_OUTPUT_UNITS, input_dim=NUM_VARIABLES, activation='relu')) 
# model.add(Dense(300, activation='relu',
#          kernel_regularizer=regularizers.l2(0.001)))
# model.add(Dropout(0.20))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train.values, y_train, epochs=30, batch_size=10)

# Experiment 6c: Early stopping + regularization + dropout:
# model.add(Dense(units=NUM_OUTPUT_UNITS, input_dim=NUM_VARIABLES, activation='relu')) 
# model.add(Dense(300, activation='relu',
#          kernel_regularizer=regularizers.l2(0.001)))
# model.add(Dropout(0.25))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train.values, y_train, epochs=30, batch_size=10)

# Experiment 6d: Early stopping + regularization + dropout:
# model.add(Dense(units=NUM_OUTPUT_UNITS, input_dim=NUM_VARIABLES, activation='relu')) 
# model.add(Dense(300, activation='relu',
#          kernel_regularizer=regularizers.l2(0.001)))
# model.add(Dropout(0.30))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train.values, y_train, epochs=30, batch_size=10)

# Experiment 6e: Early stopping + regularization + dropout:
# model.add(Dense(units=NUM_OUTPUT_UNITS, input_dim=NUM_VARIABLES, activation='relu')) 
# model.add(Dense(300, activation='relu',
#          kernel_regularizer=regularizers.l2(0.001)))
# model.add(Dropout(0.40))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train.values, y_train, epochs=30, batch_size=10)

# Experiment 6f: Early stopping + regularization + dropout:
# model.add(Dense(units=NUM_OUTPUT_UNITS, input_dim=NUM_VARIABLES, activation='relu')) 
# model.add(Dense(300, activation='relu',
#          kernel_regularizer=regularizers.l2(0.001)))
# model.add(Dropout(0.50))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train.values, y_train, epochs=30, batch_size=10)

# Fit the model, run through loop to produce plots
train_error_arr = np.zeros((40,1))
dev_error_arr = np.zeros((40,1))
epoch_arr = np.arange(30)

#for epoch in range(30):
#    model.fit(X_train.values, y_train, epochs=epoch+1, batch_size=10)
#    train_error = model.evaluate(X_train.values, y_train)
#    dev_error = model.evaluate(X_dev.values, y_dev)
#    train_error_arr[epoch] = train_error[1]
#    dev_error_arr[epoch] = dev_error[1]
#    print("Number of epochs used: ", epoch)
#    print("Train error: ", train_error)
#    print("Dev error: ", dev_error)
#    
# np.savez('rates.npz', train_error=train_error_arr, dev_error=dev_error_arr)
# Test

model.fit(X_train.values, y_train, epochs=30, batch_size=10)
train_error = model.evaluate(X_train.values, y_train)
dev_error = model.evaluate(X_dev.values, y_dev)
print("Train error: ", train_error)
print("Dev error: ", dev_error)

model.save('my_model.h5')

# model = load_model('my_model.h5')











"""
END

"""








"""
# run test:
file_name = 'dota2_pro_match_input_data_test.pkl'
df_test = pd.read_pickle(file_path + file_name)

X_test = df_test.drop({'radiant_win', 'match_id', 'start_date'}, axis = 1)

y_test = df_test['radiant_win']

May = df_test[(df_test['start_date'] >= '2017-05-01') & (df_test['start_date'] < '2017-06-01')]
Jun = df_test[(df_test['start_date'] >= '2017-06-01') & (df_test['start_date'] < '2017-07-01')]
Jul = df_test[(df_test['start_date'] >= '2017-07-01') & (df_test['start_date'] < '2017-08-01')]
Aug = df_test[(df_test['start_date'] >= '2017-08-01') & (df_test['start_date'] < '2017-09-01')]
Sep = df_test[(df_test['start_date'] >= '2017-09-01') & (df_test['start_date'] < '2017-10-01')]
Oct = df_test[(df_test['start_date'] >= '2017-10-01') & (df_test['start_date'] < '2017-11-01')]

df_test.shape

May_X_test = May.drop({'radiant_win', 'match_id', 'start_date'}, axis = 1)
May_y_test = May['radiant_win']

may_error = model.evaluate(May_X_test.values, May_y_test)
print("Train error: ", may_error)


#test_predictions = model.predict(X_test.values)


test_output = pd.DataFrame({
         'prediction': list(test_predictions)
        ,'actual': list(y_test)
})

test_output['correct'] = test_output['prediction'] == test_output['actual']
test_correct_predictions = np.sum(test_output['correct'] == True)
test_size = np.shape(test_output)[0]
test_pct_correct = test_correct_predictions/test_size

print("Test acc: " +str(test_pct_correct))

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

for key, value in output:
    print(key, value)


dev_predictions = model.predict(X_dev.values)
dev_predictions = [round(x[0]) for x in dev_predictions]
model.metrics_names
"""
##########################