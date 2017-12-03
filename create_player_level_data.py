#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 16:50:22 2017

@author: josephhiggins
"""

'''
data structure
        match_id, account_id, lane_pos, gpm, other_stats
        1,        123,          1,      100
        1,        234,          2,
        1,        345,          3,
        2,        123,          1,
        2,        234,          2,
        2,        345,          3,
                                    
'''

import json
import os
import pandas as pd
import numpy as np
import datetime
import time
import collections
import matplotlib.pyplot as plt

output = []

#Get all the match detail files we need to iterate through
os.chdir('/Users/josephhiggins/Documents/CS 229/Project/Pro Match Details/')
file_path = '/Users/josephhiggins/Documents/CS 229/Project/Pro Match Details/'
file_names = os.listdir(file_path)
file_names.remove('.DS_Store')

def get_field_from_player_stats(field, players):
    field_each_player_list = list(map(lambda x: x.get(field), players))
    return field_each_player_list

def get_player_stats_from_within_match(players):
    od = collections.OrderedDict({
        "account_id": get_field_from_player_stats('account_id', players),
        "lane_efficiency": get_field_from_player_stats('lane_efficiency', players),
        "lane_role": get_field_from_player_stats('lane_role', players),
        "is_roaming": get_field_from_player_stats('is_roaming', players),
        "solo_competitive_rank": get_field_from_player_stats('solo_competitive_rank', players),
        "actions_per_min": get_field_from_player_stats('actions_per_min', players),
        "xp_per_min": get_field_from_player_stats('xp_per_min', players),
        "gold_per_min": get_field_from_player_stats('gold_per_min', players),
        "kills_per_min": get_field_from_player_stats('kills_per_min', players),
        "hero_kills": get_field_from_player_stats('hero_kills', players),
        "assists": get_field_from_player_stats('assists', players)
    })
    player_information = pd.DataFrame(od).as_matrix()
    return player_information

def create_match_player_level_data()
    iteration = 0
    total = len(file_names)
    start_time = time.time()
    for i, file_name in enumerate(file_names):
    
        begin = time.clock()
        
        with open(file_path + file_name) as json_data:
            md = json.load(json_data)
         
        match_id = md.get('match_id')
        players = md.get('players')
        player_information = get_player_stats_from_within_match(players)
        
        for player in player_information:
            row = np.insert(player, 0, match_id)
            output.append(row)
    
        end = time.clock()
        elapsed = end - begin
            
        if i%1000 == 0:
             print("Percent complete: " + str('{:.2%}'.format(i/len(file_names))))
             print("Time elapsed for 1000 records: " + str(('{:.2}'.format(elapsed))))
    
    output = pd.DataFrame(output)
    output = output.rename(columns={
        0: "match_id",
        1: "account_id",
        2: "lane_efficiency",
        3: "lane_role",
        4: "is_roaming",
        5: "solo_competitive_rank",
        6: "actions_per_min",
        7: "xp_per_min",
        8: "gold_per_min",
        9: "kills_per_min",
        10: "hero_kills",
        11: "assists"
    })
    
    file_path = '/Users/josephhiggins/Documents/CS 229/Project/Tabular Data/'
    file_name = 'dota2_pro_match_player_level_data.pkl'
    output.to_pickle(file_path + file_name)

def create_rolling_mean_of_field(data, field, window):
    #Creates rolling average for a given field for a given window, but if fewer windows are observed it uses what it as
    #Also shifts up by 1 so that we dont use data from the current match to predict
    data['rolling'] = data.groupby(['account_id'])[field].apply(pd.rolling_mean, window=window, min_periods= 1).shift(1)
    
    #Sets the first value for each player to NaN since shift moved the final value from the previous player up
    #Makes sense because we have no observations for the GPM of a player in the first game they play
    rolling = data.groupby(['account_id'])['rolling'].apply(set_first_value_to_nan)
    
    return rolling

def set_first_value_to_nan(group):
    group.iloc[0] = np.nan
    return group

def create_player_match_running_averages():
    #Read in data
    file_path = '/Users/josephhiggins/Documents/CS 229/Project/Tabular Data/'
    file_name = 'dota2_pro_match_player_level_data.pkl'
    df = pd.read_pickle(file_path + file_name)
    
    df = df.sort_values(['account_id', 'match_id'])
    print('preparing GPM...')
    df['gold_per_min_10trail'] = create_rolling_mean_of_field(df, 'gold_per_min', 10)
    print('preparing XPM...')
    df['xp_per_min_10trail'] = create_rolling_mean_of_field(df, 'xp_per_min', 10)
    print('preparing KPM...')
    df['kills_per_min_10trail'] = create_rolling_mean_of_field(df, 'kills_per_min', 10)
    print('preparing LE...')
    df['lane_efficiency_10trail'] = create_rolling_mean_of_field(df, 'lane_efficiency', 10)
    
    '''
    df.groupby(['account_id']).apply(len)
    df[['account_id','xp_per_min','xp_per_min_10trail']][df['account_id'].isin([88470])]
    df[['account_id','kills_per_min','kills_per_min_10trail']][df['account_id'].isin([88470])]
    df[['account_id','lane_efficiency','lane_efficiency_10trail']][df['account_id'].isin([88470])]
    plt.scatter(df['gold_per_min_10trail'], df['gold_per_min'], s = 2)
    plt.scatter(df['xp_per_min_10trail'], df['xp_per_min'], s = 2)
    plt.scatter(df['kills_per_min_10trail'], df['kills_per_min'], s = 2)
    plt.scatter(df['lane_efficiency_10trail'], df['lane_efficiency'], s = 2)
    '''