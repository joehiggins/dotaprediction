#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 20:43:47 2017

@author: josephhiggins
"""

import os
import pandas as pd
import numpy as np
import math
from sklearn import svm

file_path = '/Users/josephhiggins/Documents/CS 229/Project/dotaprediction/Tabular Data/'
file_name = 'dota2_pro_match_tabular_data.pkl'
df = pd.read_pickle(file_path + file_name)

#filter out nulls
df = df[~pd.isnull(df['radiant_picks'])]

#filter to before cutoff date
cutoff_date = '2017-05-01'
df = df[df['start_date'] <= cutoff_date]

#create train/dev set
train_pct = 0.90
msk = np.random.rand(len(df)) < train_pct
df_train = df[msk]
df_dev = df[~msk]

#get hero pick indicators
radiant_picks = df_train['radiant_picks']
radiant_picks = list(map(lambda x: 
        list(map(lambda y: str(y), x))
    , radiant_picks))
radiant_picks = list(map(tuple, radiant_picks))
radiant_picks = pd.Series(radiant_picks)
radiant_picks = pd.DataFrame(radiant_picks)
radiant_picks = radiant_picks[0].str.join(sep='*').str.get_dummies(sep='*')

dire_picks = df_train['dire_picks']
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

#append victory indicators
pick_indicators['match_id'] = list(df_train['match_id'])
pick_indicators['radiant_win'] = list(df_train['radiant_win'])



'''
pick_indicators['25_radiant'][pick_indicators['match_id'] == 3539847416]
df.ix[3539847416]
'''





