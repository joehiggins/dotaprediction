# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 10:24:39 2017

@author: Joe
"""

import json
import os
import pandas as pd
import numpy as np
import datetime
import time
start_time = time.time()

output = {}

#Get all the match detail files we need to iterate through
os.chdir('/Users/josephhiggins/Documents/CS 229/Project/Pro Match Details/')
file_names = os.listdir(file_path)

def get_team_selections(picks_bans, selection_is_pick, selection_team):
    #param: picks_bans: 'picks_bans' object from match_data dict
    #param: selection_type: 'picks' or 'bans'
    #param: selection_team: 'radiant' or 'dire'

    is_pick = True if selection_is_pick == 'picks' else False
    team = 0 if selection_team == 'radiant' else 1
        
    picks_bans_df = pd.DataFrame(picks_bans)
    heros = picks_bans_df['hero_id'][(picks_bans_df['is_pick'] == is_pick) & (picks_bans_df['team'] == team)]    

    heros_list = list(heros)
    #heros_string = "_".join(map(str, heros_list))+"_"
    
    return heros_list

iteration = 0
total = len(file_names)
for file_name in file_names:

    with open(file_path + file_name) as json_data:
        md = json.load(json_data) #md = match_data
    
    current_match = {}
    
    current_match['match_id'] = md.get('match_id')
    current_match['start_date'] = datetime.datetime.utcfromtimestamp(md.get('start_time')).date().isoformat()
    current_match['patch'] = md.get('patch')
    current_match['league_name'] = md.get('league', {}).get('name')
    current_match['radiant_name'] = md.get('radiant_team', {}).get('name')
    current_match['dire_name'] = md.get('dire_team', {}).get('name')
    current_match['radiant_win'] = md.get('radiant_win')

    if md.get('players'):
        current_match['player_0_account_id'] = md.get('players')[0].get('account_id')
        current_match['player_1_account_id'] = md.get('players')[1].get('account_id')
        current_match['player_2_account_id'] = md.get('players')[2].get('account_id')
        current_match['player_3_account_id'] = md.get('players')[3].get('account_id')
        current_match['player_4_account_id'] = md.get('players')[4].get('account_id')
        current_match['player_5_account_id'] = md.get('players')[5].get('account_id')
        current_match['player_6_account_id'] = md.get('players')[6].get('account_id')
        current_match['player_7_account_id'] = md.get('players')[7].get('account_id')
        current_match['player_8_account_id'] = md.get('players')[8].get('account_id')
        current_match['player_9_account_id'] = md.get('players')[9].get('account_id')
        current_match['player_0_name'] = md.get('players')[0].get('name')
        current_match['player_1_name'] = md.get('players')[1].get('name')
        current_match['player_2_name'] = md.get('players')[2].get('name')
        current_match['player_3_name'] = md.get('players')[3].get('name')
        current_match['player_4_name'] = md.get('players')[4].get('name')
        current_match['player_5_name'] = md.get('players')[5].get('name')
        current_match['player_6_name'] = md.get('players')[6].get('name')
        current_match['player_7_name'] = md.get('players')[7].get('name')
        current_match['player_8_name'] = md.get('players')[8].get('name')
        current_match['player_9_name'] = md.get('players')[9].get('name')
        
    if md.get('picks_bans'):
        current_match['picks_bans'] = md.get('picks_bans')
        current_match['radiant_picks'] = get_team_selections(current_match.get('picks_bans'), 'picks', 'radiant')
        current_match['radiant_bans'] = get_team_selections(current_match.get('picks_bans'), 'bans', 'radiant')
        current_match['dire_picks'] = get_team_selections(current_match.get('picks_bans'), 'picks', 'dire')
        current_match['dire_bans'] = get_team_selections(current_match.get('picks_bans'), 'bans', 'dire')
        current_match['first_pick'] = 'radiant' if current_match.get('picks_bans')[0]['team'] == 0 else 'dire'
    
    output[current_match.get('match_id')] = current_match
    
    iteration += 1
    if iteration%1000 == 0:
        print("Percent complete: " + str('{:.2%}'.format(iteration/total)))

output = pd.DataFrame(output).transpose()

file_path = '/Users/josephhiggins/Documents/CS 229/Project/Tabular Data/'
file_name = 'dota2_pro_match_tabular_data.pkl'
output.to_pickle(file_path + file_name)
