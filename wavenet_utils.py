
import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import seaborn as sns                            # more plots

from keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd
import numpy as np

from keras.callbacks import TensorBoard
import os
from random import seed
from time import time

seed(42)
np.random.seed(42)

df = pd.read_csv("Traffic Management/training.csv")
df.head()


df[['h','m']] = df['timestamp'].str.split(':',expand=True)
df['h'] = df['h'].astype('int64')
df['m'] = df['m'].astype('int64')
df['dow'] = df['day'] % 7
df['timestamp'] = ((df['day']-1) * 24 * 60) + (df['h'] * 60) + df['m']

print(df.head())

f = df.sort_values(by=['timestamp', 'geohash6'], ascending=True)

print(f.head())
print(f.tail())

timestamp_df = df.groupby('timestamp', as_index=False)\
.agg({'day':'count'})\
.rename(columns={'day':'count'})\
.sort_values(by='count', ascending=False)
print("Total encoded hash",len(timestamp_df))

geohashes_df = df.groupby('geohash6', as_index=False)\
.agg({'day':'count'})\
.rename(columns={'day':'count'})\
.sort_values(by='count', ascending=False)
print("Total encoded hash",len(geohashes_df))


print(geohashes_df.head())
np.set_printoptions(precision=7, suppress=True)
#matrix = np.zeros((5847, 1329))
#matrix = np.random.normal(0, 0.0000001, (5000, 1000))
#matrix = np.full((5856, 1329), 0.0000001)
geomaps = {}

for index, row in f.iterrows():
    if row['geohash6'] not in geomaps:
       geomaps[row['geohash6']] = np.full(5856, 0.0000001)

    geomaps[row['geohash6']][int(row['timestamp'] // 15)] = row['demand']

import pandas as pd
d = pd.DataFrame.from_dict(geomaps, orient='index')
d.columns = [(i * 15) for i in range(d.shape[1])]
print(d.head())
d.to_hdf('data.h5', key='df', mode='w')