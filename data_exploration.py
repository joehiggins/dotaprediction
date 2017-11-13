import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

os.chdir('/Users/josephhiggins/Documents/CS 229/Project/Tabular Data/')
#file_path = '../Tabular Data/'
file_name = 'dota2_pro_match_tabular_data.csv'

data = pd.read_csv(file_path + file_name)

startdate_as_datetime = list(map(lambda x: datetime.strptime(x, '%Y-%m-%d'), data['start_date']))
plt.hist(startdate_as_datetime, bins='auto')

cutoff_date = '2017-05-01'

number_of_games = np.shape(data)[0]
games_before_cutoff = np.shape(data[data['start_date'] <= cutoff_date])[0]
games_after_cutoff = np.shape(data[data['start_date'] > cutoff_date])[0]

games_before_cutoff/number_of_games
games_after_cutoff/number_of_games