# remember to reactivate tensor flow when you open a new terminal window

import os
import pandas as pd
import numpy as np
import math
from sklearn import svm

file_path = '/Users/petragrutzik/CSClasses/CS229/dota/dotaprediction/Tabular Data/'
file_name = 'dota2_pro_match_tabular_data.pkl'
df = pd.read_pickle(file_path + file_name)

#filter out nulls
df = df[~pd.isnull(df['radiant_picks'])]

#filter to before cutoff date
cutoff_date = '2017-05-01'
df = df[df['start_date'] <= cutoff_date]

#get hero pick indicators for training data
radiant_picks = df['radiant_picks']
radiant_picks = list(map(lambda x: 
        list(map(lambda y: str(y), x))
    , radiant_picks))
radiant_picks = list(map(tuple, radiant_picks))
radiant_picks = pd.Series(radiant_picks)
radiant_picks = pd.DataFrame(radiant_picks)
radiant_picks = radiant_picks[0].str.join(sep='*').str.get_dummies(sep='*')

dire_picks = df['dire_picks']
dire_picks = list(map(lambda x: 
        list(map(lambda y: str(y), x))
    , dire_picks))
dire_picks = list(map(tuple, dire_picks))
dire_picks = pd.Series(dire_picks)
dire_picks = pd.DataFrame(dire_picks)
dire_picks = dire_picks[0].str.join(sep='*').str.get_dummies(sep='*')

radiant_picks = radiant_picks.rename(columns = lambda x: x+'_radiant')
dire_picks = dire_picks.rename(columns = lambda x: x+'_dire')

pick_indicators = pd.concat([radiant_picks, dire_picks], axis = 1)

pick_indicators['match_id'] = list(df['match_id'])
pick_indicators['radiant_win'] = list(df['radiant_win'])

#create train/dev set
train_pct = 0.90
msk = np.random.rand(len(df)) < train_pct
df_train = pick_indicators[msk]
df_dev = pick_indicators[~msk]

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
NUM_LAYERS = 3
NUM_VARIABLES = 226

model = Sequential()
model.add(Dense(units=NUM_OUTPUT_UNITS, input_dim=NUM_VARIABLES, activation='relu')) 
model.add(Dense(12, input_dim=226, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
            optimizer='adam')
   
# Fit the model
print(X_train.values.shape)
model.fit(X_train.values, y_train, epochs=50, batch_size=10)

# Test
predictions = model.evaluate(X_dev.values, y_dev)
print(predictions)

##########################





