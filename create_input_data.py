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
import scipy as sp
from sklearn.preprocessing import Imputer

CROSS_TRAIN = False

file_path = '/Users/josephhiggins/Documents/CS 229/Project/Tabular Data/'
file_name = 'dota2_pro_merged_data.pkl'
df = pd.read_pickle(file_path + file_name)

#filter out nulls
df = df[~pd.isnull(df['radiant_picks'])]

#FEATURE SET 1: get hero pick indicators for training data
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

#FEATURE SET 2: get hero pick indicators for training data
player_fields_to_grab = [
        'xp_per_min_10trail', 
        'gold_per_min_10trail',
        'kills_per_min_10trail',
        'lane_efficiency_10trail',
        'solo_competitive_rank'
]

player_fields_prefixed = []
for player_number in np.arange(0,10):
    for field in player_fields_to_grab:
        player_number_string= 'player_' + str(player_number)
        player_fields_prefixed.append(player_number_string + '_' + field)
        
player_metrics = df[player_fields_prefixed]
player_metrics = player_metrics.reset_index()

#FEATURE SET 3:get team indicators for training data
radiant_team = df['radiant_name']
radiant_team = pd.get_dummies(radiant_team)

dire_team = df['dire_name']
dire_team = pd.get_dummies(dire_team)

radiant_team = radiant_team.rename(columns = lambda x: x+'_radiant')
dire_team = dire_team.rename(columns = lambda x: x+'_dire')

team_indicators = pd.concat([radiant_team, dire_team], axis = 1)
team_indicators = team_indicators.set_index(np.arange(0,np.shape(team_indicators)[0]))

#create train/dev/test set

all_features = pd.concat([pick_indicators, team_indicators, player_metrics], axis = 1)
#all_features = pd.concat([pick_indicators, player_metrics], axis = 1)
#all_features = pd.concat([pick_indicators, team_indicators], axis = 1)
#all_features = pd.concat([pick_indicators], axis = 1)
#all_features = pd.concat([team_indicators], axis = 1)

#Fill in Nans
player_metrics_include = True
if player_metrics_include == True:
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(all_features)
    imputed = imp.transform(all_features)

all_features_imputed = pd.DataFrame(imputed, columns = all_features.columns)

#convert data to zscores
all_features_zscore = all_features_imputed.apply(sp.stats.zscore, axis = 0)
'''
pd.DataFrame({'pre impute': all_features['player_9_solo_competitive_rank'],
              'post impute': all_features_imputed['player_9_solo_competitive_rank'],
              'zscore': all_features_zscore['player_9_solo_competitive_rank']})
'''
#append tracking columns
all_features_zscore['match_id'] = list(df['match_id'])
all_features_zscore['radiant_win'] = list(df['radiant_win'])
all_features_zscore['start_date'] = list(df['start_date'])

#filter to before cutoff date
cutoff_date = '2017-05-01'
all_features_non_test = all_features_zscore[all_features_zscore['start_date'] <= cutoff_date]
all_features_test = all_features_zscore[all_features_zscore['start_date'] > cutoff_date]

#Save output
file_path = '/Users/josephhiggins/Documents/CS 229/Project/Input Data/'

if CROSS_TRAIN == False:
    train_pct = 0.90
    msk = np.random.rand(len(all_features_non_test)) < train_pct
    df_train = all_features_non_test[msk]
    df_dev = all_features_non_test[~msk]
    df_test = all_features_test
    
    df_train_part1 = df_train.iloc[:round(len(df_train)/2), :]
    df_train_part2 = df_train.iloc[round(len(df_train)/2+1):, :]
    
    file_name = 'dota2_pro_match_input_data_train1.pkl'
    df_train_part1.to_pickle(file_path + file_name)

    file_name = 'dota2_pro_match_input_data_train2.pkl'
    df_train_part2.to_pickle(file_path + file_name)

    file_name = 'dota2_pro_match_input_data_dev.pkl'
    df_dev.to_pickle(file_path + file_name)
    
    file_name = 'dota2_pro_match_input_data_test.pkl'
    df_test.to_pickle(file_path + file_name)
else:
    #for cross train
    df.to_pickle(file_path + file_name)
    file_name = 'dota2_pro_match_input_data_all.pkl'


    
'''
pick_indicators['25_radiant'][pick_indicators['match_id'] == 3539847416]
df.ix[3135929855]
'''
