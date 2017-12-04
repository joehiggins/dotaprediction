#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 13:40:49 2017

@author: josephhiggins
"""

import os
import pandas as pd
import numpy as np

os.chdir('/Users/josephhiggins/Documents/CS 229/Project/Tabular Data/')
file_path = '/Users/josephhiggins/Documents/CS 229/Project/Tabular Data/'
game_data_file = 'dota2_pro_match_tabular_data.pkl'
player_data_file = 'dota2_pro_match_player_level_rolling_data.pkl'

game_data = pd.read_pickle(file_path + game_data_file)
player_data = pd.read_pickle(file_path + player_data_file)

def merge_a_player(player_number, game_data, player_data):
 
    player_number_string = 'player_' + str(player_number)

    player_join_fields= [
        'match_id',
        'account_id'
    ]
    player_fields_to_grab = [
        'xp_per_min_10trail', 
        'gold_per_min_10trail',
        'kills_per_min_10trail',
        'lane_efficiency_10trail',
        'solo_competitive_rank',
        'lane_role'
    ]

                   
    merged = game_data.merge(player_data[player_join_fields + player_fields_to_grab], 
                      how='left', 
                      left_on=['match_id', player_number_string + '_account_id'], 
                      right_on=['match_id', 'account_id']
    )
    
    merged = merged.drop(['account_id'], axis = 1)

    field_rename_map = {}
    for field in player_fields_to_grab:
        field_rename_map[field] = player_number_string + '_' + field
    merged = merged.rename(columns=field_rename_map)
    
    return merged

merged_data = game_data
for player_number in np.arange(0,10):
    merged_data = merge_a_player(player_number, merged_data, player_data)

file_name = 'dota2_pro_merged_data.pkl'
merged_data.to_pickle(file_path + file_name)

'''
s_game_data = game_data[game_data['match_id'] == 3540226851]
s_player_data = player_data[player_data['match_id'] == 3540226851]
merged_data[merged_data['match_id'] == 3540226851]
'''