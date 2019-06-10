import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import seaborn as sns                            # more plots
import geohash
from math import cos, sin
from keras.callbacks import TensorBoard
import os
from random import seed
from time import time
from math import sqrt
from sklearn.metrics import mean_squared_error
from keras import Sequential
from keras.layers import Input, LSTM, Embedding, Dense, Lambda, Dropout

from utils import timeseries, timeseries_generator, generate_synthetic_data, predict


def get_data(file, split_ratio=0.8):
    df = pd.read_csv(file)
    df.head()

    df[['h','m']] = df['timestamp'].str.split(':',expand=True)
    df['h'] = df['h'].astype('int64')
    df['m'] = df['m'].astype('int64')
    df['mins'] = (df['h'] * 60) + df['m']
    df['mins_norm'] = df['mins'] / 1440
    df['dow'] = df['day'] % 7

    # Resolve GeoCodes
    geohashes_df = df.groupby('geohash6', as_index=False).agg({'day':'count'}).rename(columns={'day':'count'}).sort_values(by='count', ascending=False)
    geohashes_df['lat'] = None
    geohashes_df['lat_err'] = None
    geohashes_df['long'] = None
    geohashes_df['long_err'] = None
    geohashes_df['x'] = None
    geohashes_df['y'] = None
    geohashes_df['z'] = None

    for i in range(len(geohashes_df)):
        geo_decoded = geohash.decode_exactly(geohashes_df.loc[i,'geohash6'])
        geohashes_df.loc[i,'lat'] = geo_decoded[0]
        geohashes_df.loc[i,'long'] = geo_decoded[1]
        geohashes_df.loc[i,'lat_err'] = geo_decoded[2]
        geohashes_df.loc[i,'long_err'] = geo_decoded[3]
        
        # https://datascience.stackexchange.com/a/13575
        geohashes_df.loc[i,'x'] = cos(geo_decoded[0]) * cos(geo_decoded[1]) # cos(lat) * cos(lon)
        geohashes_df.loc[i,'y'] = cos(geo_decoded[0]) * sin(geo_decoded[1]) # cos(lat) * sin(lon)
        geohashes_df.loc[i,'z'] = sin(geo_decoded[0]) # sin(lat)

    df = df.merge(geohashes_df.drop(columns=['count']), on='geohash6', how='inner')

    dfx = df.sort_values(by=['day', 'mins', 'geohash6'], ascending=True)
    X = dfx[['dow', 'mins_norm', 'x', 'y', 'z', 'demand']]
    y = dfx[['demand']]

    split_n = int(split_ratio*len(y))
    train_X = X.iloc[:split_n,:].values
    train_y = y.iloc[:split_n,:].values

    test_X = X.iloc[split_n:,:].values
    test_y = y.iloc[split_n:,:].values

    print("Train shape", train_X.shape, train_y.shape)
    print("Test shape", test_X.shape, test_y.shape)

    return train_X, train_y, test_X, test_y



seed(42)
np.random.seed(42)

past_steps = 10
future_steps = 5

BATCH_SIZE = 50000

train_X, train_y, test_X, test_y = get_data("training.csv")
FEATURE = train_X.shape[1]


model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(past_steps, FEATURE)),
    Dropout(0.2),
    LSTM(100),
    Dropout(0.2),
    Dense(future_steps)
])
model.compile(loss='mape', optimizer='adam', metrics=['mape', 'mse', 'mae', 'cosine'])
model.summary()


tensorboard = TensorBoard(log_dir=os.path.join("/tmp/logs", str(time())))

g = timeseries_generator(train_X, train_y, BATCH_SIZE, past_steps, future_steps)
past = model.fit_generator(g, steps_per_epoch=len(train_X) // BATCH_SIZE, epochs=1, verbose=1, callbacks=[tensorboard])

test_Xs, test_ys = timeseries(test_X, test_y, past_steps, future_steps)
print(test_Xs.shape, test_ys.shape)
print(test_X.shape, test_y.shape)
yhat = model.predict(test_Xs, batch_size=BATCH_SIZE, verbose=0)
mse = sqrt(mean_squared_error(test_ys, yhat))
print('RMSE: %f' % mse)

evaluate = model.evaluate(test_Xs, test_ys, batch_size=BATCH_SIZE)
print("Evaluation", evaluate)
plt.plot(past.history['loss'])
#plt.plot(past.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Time')
plt.legend(['Train','Valid'])
plt.show()