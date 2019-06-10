import pandas as pd
import numpy as np
import os
from random import seed
from time import time
import argparse

if __name__ == '__main__':
    # Parse the input arguments for common Cloud ML Engine options
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        help='Input data')
    parser.add_argument(
        '--output',
        type=str,
        default="data.h5",
        help='Transformed output data')
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    df[['h','m']] = df['timestamp'].str.split(':', expand=True)
    df['h'] = df['h'].astype('int32')
    df['m'] = df['m'].astype('int32')
    df['dow'] = df['day'] % 7
    df['timestamp'] = ((df['day']-1) * 24 * 60) + (df['h'] * 60) + df['m']

    f = df.sort_values(by=['timestamp', 'geohash6'], ascending=True)

    print(f.head())
    print(f.tail())

    timestamp_min = df['timestamp'].min()
    timestamp_max = df['timestamp'].max()
    matrix_width = ((timestamp_max - timestamp_min) // 15) + 1

    print("timestamp_min", timestamp_min, "timestamp_max", timestamp_max, "matrix_width", matrix_width)

    pd.options.display.max_columns = 9
    pd.options.display.float_format = '{:.7f}'.format

    #np.set_printoptions(precision=7, suppress=True)
    #matrix = np.zeros((5847, 1329))
    #matrix = np.random.normal(0, 0.0000001, (5000, 1000))
    #matrix = np.full((5856, 1329), 0.0000001)
    
    geomaps = {}

    for index, row in f.iterrows():
        if row['geohash6'] not in geomaps:
            geomaps[row['geohash6']] = np.full(matrix_width, 0.0000001)
        geomaps[row['geohash6']][int((row['timestamp'] - timestamp_min) // 15)] = row['demand']

    d = pd.DataFrame.from_dict(geomaps, orient='index')
    d.columns = [i for i in range(timestamp_min, timestamp_max + 1, 15)]
    print(d.head())
    print(d.tail())
    d.to_hdf(args.output, key='df', mode='w')