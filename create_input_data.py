#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 20:43:47 2017

@author: josephhiggins
"""

import os
import pandas as pd
import numpy as np

file_path = '/Users/josephhiggins/Documents/CS 229/Project/Tabular Data/'
file_name = 'dota2_pro_match_tabular_data.pkl'
df = pd.read_pickle(file_path + file_name)


account_ids = [
        list(np.unique(df['player_0_account_id'])),
        list(np.unique(df['player_1_account_id'])),
        list(np.unique(df['player_2_account_id'])),
        list(np.unique(df['player_3_account_id'])),
        list(np.unique(df['player_4_account_id'])),
        list(np.unique(df['player_5_account_id'])),
        list(np.unique(df['player_6_account_id'])),
        list(np.unique(df['player_7_account_id'])),
        list(np.unique(df['player_8_account_id'])),
        list(np.unique(df['player_9_account_id'])),
]
account_ids = [item for sublist in account_ids for item in sublist]
account_ids = np.unique(account_ids)


#get all unique player account_ids
a = pd.Series(list(
     zip(list(df['player_0_account_id']),
         list(df['player_1_account_id']),
         list(df['player_2_account_id']),
         list(df['player_3_account_id']),
         list(df['player_4_account_id']),
    )
))
a = list(map(list, a))

pd.get_dummies(a)

players = pd.Series(list('abca'))
pd.get_dummies(s)