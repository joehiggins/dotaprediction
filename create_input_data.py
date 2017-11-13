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


#get hero pick indicators
radiant_picks = df['radiant_picks'][~pd.isnull(df['radiant_picks'])]
radiant_picks = list(map(lambda x: 
        list(map(lambda y: str(y), x))
    , radiant_picks))
radiant_picks = list(map(tuple, radiant_picks))
radiant_picks = pd.Series(radiant_picks)
radiant_picks = pd.DataFrame(radiant_picks)
radiant_picks = radiant_picks[0].str.join(sep='*').str.get_dummies(sep='*')

dire_picks = df['dire_picks'][~pd.isnull(df['radiant_picks'])]
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







'''
#get all unique player account_ids
a = pd.Series(list(
     zip(list(map(str, df['player_0_account_id'])),
         list(map(str, df['player_1_account_id'])),
         list(map(str, df['player_2_account_id'])),
         list(map(str, df['player_3_account_id'])),
         list(map(str, df['player_4_account_id'])),
    )
))
a = pd.DataFrame(a)
b = a.head(5)
b = b[0].str.join(sep='*').str.get_dummies(sep='*')
'''