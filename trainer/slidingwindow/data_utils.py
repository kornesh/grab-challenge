import pandas as pd
import numpy as np
from random import seed
from tensorflow.python.lib.io import file_io
import geohash
from math import cos, sin

seed(42)
np.random.seed(42)


def generate_synthetic_data(start=1, end=100):
    # define input sequence
    in_seq1 = np.array([[i] for i in range(start, end, 10)], dtype=np.float32)
    in_seq2 = np.array([[(i+5)]
                        for i in range(start, end, 10)], dtype=np.float32)
    out_seq = np.array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

    # convert to [rows, columns] structure
    in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))

    # horizontally stack columns
    dataset = np.hstack((in_seq1, in_seq2, out_seq))

    df = pd.DataFrame(dataset, columns=['x1', 'x2', 'y'])
    print(df.head())
    print(df.tail())
    X = df[['x1', 'x2', 'y']].values
    y = df[['y']].values
    return X, y

# Based on https://medium.com/@fromtheast/implement-fit-generator-in-keras-61aa2786ce98
# and https://github.com/keras-team/keras/issues/1627
#

def keras_timeseries_generator(X, y, batch_size, past_steps, future_steps):
    num_batches = len(X) // batch_size
    #print("generator", num_batches)
    while True:
      for batch_index in range(num_batches):
        Xs = []
        ys = []
        start = batch_index * batch_size
        end = (batch_index + 1) * batch_size
        #print(start, end)
        for i in range(start, end):
            s = i + past_steps
            e = s + future_steps
            #print("i", i, "s", s, "e", e)
            if e > len(X):
                #print("continue", e)
                continue
            _x = X[i:s]
            _y = y[s:e].reshape(future_steps)
            #print(_y.reshape(1, future_steps))
            Xs.append(_x)
            ys.append(_y)
        #print("x", _x.shape, _x, "y", _y.shape, _y)
        Xs = np.array(Xs)
        ys = np.array(ys)
        #print("Xs", Xs.shape, "ys", ys.shape)
        yield Xs, ys

def timeseries_generator(X, y, batch_size, past_steps, future_steps):
    num_batches = len(X) // batch_size
    for batch_index in range(num_batches):
        Xs = []
        ys = []
        start = batch_index * batch_size
        end = (batch_index + 1) * batch_size
        for i in range(start, end):
            s = i + past_steps
            e = s + future_steps
            if e > len(X):
                continue
            _x = X[i:s]
            _y = y[s:e].reshape(future_steps)
            Xs.append(_x)
            ys.append(_y)
        yield np.array(Xs), np.array(ys)


# This is the root cause of OOMs during prediction lol. Use the generator instead
def timeseries(X, y, past_steps, future_steps):
    Xs = []
    ys = []
    for i in range(len(X)):
        s = i + past_steps
        e = s + future_steps
        #print("i", i, "s", s, "e", e)
        if e > len(X):
            #print("continue", e)
            continue
        _x = X[i:s]
        _y = y[s:e].reshape(future_steps)
        #print(_y.reshape(1, future_steps))
        Xs.append(_x)
        ys.append(_y)
        #print("x", _x.shape, _x, "y", _y.shape, _y)
    Xs = np.array(Xs)
    ys = np.array(ys)

    return (Xs, ys)

def get_data(file_path):
    print("Loading data..", file_path)
    with file_io.FileIO(file_path, mode='rb') as input_f:
        df = pd.read_csv(input_f)
    df[['h', 'm']] = df['timestamp'].str.split(':', expand=True)
    df['h'] = df['h'].astype('int64')
    df['m'] = df['m'].astype('int64')
    df['mins'] = (df['h'] * 60) + df['m']
    df['mins_norm'] = df['mins'] / 1440
    df['dow'] = df['day'] % 7

    # Resolve GeoCodes
    geohashes_df = df.groupby('geohash6', as_index=False).agg({'day': 'count'}).rename(
        columns={'day': 'count'}).sort_values(by='count', ascending=False)
    geohashes_df['lat'] = None
    geohashes_df['lat_err'] = None
    geohashes_df['long'] = None
    geohashes_df['long_err'] = None
    geohashes_df['x'] = None
    geohashes_df['y'] = None
    geohashes_df['z'] = None

    for i in range(len(geohashes_df)):
        geo_decoded = geohash.decode_exactly(geohashes_df.loc[i, 'geohash6'])
        geohashes_df.loc[i, 'lat'] = geo_decoded[0]
        geohashes_df.loc[i, 'long'] = geo_decoded[1]
        geohashes_df.loc[i, 'lat_err'] = geo_decoded[2]
        geohashes_df.loc[i, 'long_err'] = geo_decoded[3]

        # https://datascience.stackexchange.com/a/13575
        geohashes_df.loc[i, 'x'] = cos(
            geo_decoded[0]) * cos(geo_decoded[1])  # cos(lat) * cos(lon)
        geohashes_df.loc[i, 'y'] = cos(
            geo_decoded[0]) * sin(geo_decoded[1])  # cos(lat) * sin(lon)
        geohashes_df.loc[i, 'z'] = sin(geo_decoded[0])  # sin(lat)

    df = df.merge(geohashes_df.drop(
        columns=['count']), on='geohash6', how='inner')

    return df

def train_test_split(df, test_days=14):
    train = df.loc[df['day'] < (df['day'].max() - test_days)]
    test = df.loc[df['day'] >= (df['day'].max() - test_days)]

    train_X, train_y = postprocess_data(train)
    test_X, test_y = postprocess_data(test)

    return train_X, train_y, test_X, test_y

def postprocess_data(df):
    df = df.sort_values(by=['geohash6', 'day', 'mins'], ascending=True)

    print(df.shape)
    print(df.head())
    print(df.tail())

    X = df[['dow', 'mins_norm', 'x', 'y', 'z', 'demand']]
    y = df[['demand']]

    return X.values, y.values

if __name__ == "__main__":
    # @TODO(kornesh): Write proper tests!
    X, y = generate_synthetic_data(0, 1000)
    pd.DataFrame(X).to_csv('synthetic.csv')
    X, y = generate_synthetic_data(0, 100)
    print(X)
    print(y)
    past_steps = 3
    future_steps = 3
    batch_size = 4

    generator = timeseries_generator(
        X, y, batch_size, past_steps, future_steps)

    for x, y in generator:
        print('%s => %s' % (x, y))

    X, y = generate_synthetic_data(0, 100)

    Xs, ys = timeseries(X, y, past_steps, future_steps)
    print(Xs, ys)
