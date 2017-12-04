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

CROSS_TRAIN = True

file_path = '/Users/josephhiggins/Documents/CS 229/Project/dotaprediction/Tabular Data/'
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

'''
#get player indicators for training data
radiant_players = list(zip(
        df['player_0_account_id'],
        df['player_1_account_id'],
        df['player_2_account_id'],
        df['player_3_account_id'],
        df['player_4_account_id']
))
radiant_players= list(map(lambda x: 
        list(map(lambda y: str(y), x))
    , radiant_players))
radiant_players = list(map(tuple, radiant_players))
radiant_players = pd.Series(radiant_players)
radiant_players = pd.DataFrame(radiant_players)
radiant_players = radiant_players[0].str.join(sep='*').str.get_dummies(sep='*')

dire_players = list(zip(
        df['player_5_account_id'],
        df['player_6_account_id'],
        df['player_7_account_id'],
        df['player_8_account_id'],
        df['player_9_account_id']
))
dire_players= list(map(lambda x: 
        list(map(lambda y: str(y), x))
    , dire_players))
dire_players = list(map(tuple, dire_players))
dire_players = pd.Series(dire_players)
dire_players = pd.DataFrame(dire_players)
dire_players = dire_players[0].str.join(sep='*').str.get_dummies(sep='*')
'''
#get team indicators for training data
radiant_team = df['radiant_name']
radiant_team = pd.get_dummies(radiant_team)

dire_team = df['dire_name']
dire_team = pd.get_dummies(dire_team)

radiant_team = radiant_team.rename(columns = lambda x: x+'_radiant')
dire_team = dire_team.rename(columns = lambda x: x+'_dire')

team_indicators = pd.concat([radiant_team, dire_team], axis = 1)
team_indicators = team_indicators.set_index(np.arange(0,np.shape(team_indicators)[0]))

#create train/dev set

all_features = pd.concat([pick_indicators, team_indicators], axis = 1)
#all_features = pd.concat([team_indicators], axis = 1)
all_features['match_id'] = list(df['match_id'])
all_features['radiant_win'] = list(df['radiant_win'])

#Save output
file_path = '/Users/josephhiggins/Documents/CS 229/Project/Input Data/'

if CROSS_TRAIN == False:
    train_pct = 0.90
    msk = np.random.rand(len(df)) < train_pct
    df_train = all_features[msk]
    df_dev = all_features[~msk]

    file_name = 'dota2_pro_match_input_data_train.pkl'
    df_train.to_pickle(file_path + file_name)

    file_name = 'dota2_pro_match_input_data_dev.pkl'
    df_dev.to_pickle(file_path + file_name)
else:
    #for cross train
    df.to_pickle(file_path + file_name)
    file_name = 'dota2_pro_match_input_data_all.pkl


    
'''
pick_indicators['25_radiant'][pick_indicators['match_id'] == 3539847416]
df.ix[3135929855]
'''
