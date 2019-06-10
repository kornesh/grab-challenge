import tensorflow as tf
import os
import numpy as np
import sys
import argparse
from trainer.slidingwindow.data_utils import timeseries, get_data, train_test_split
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    type=str,
    help='Full model .h5 file')
parser.add_argument(
    '--test-days',
    type=int,
    default=14,
    help='Look ahead / predict steps')
parser.add_argument(
    '--output-prefix',
    type=str,
    default="data",
    help='Full model .h5 file')
args = parser.parse_args()

'''
train_X, train_y, test_X, test_y = train_test_split(get_data(args.input), args.test_days)

def write(filename, value):
    df = pd.DataFrame(value)
    df.columns = ['dow', 'mins_norm', 'x', 'y', 'z', 'demand']
    df.to_csv(filename, index=False)
    print("Saved %s" % filename)
'''

df = pd.read_csv(args.input)
train = df.loc[df['day'] < (df['day'].max() - args.test_days)]
test = df.loc[df['day'] >= (df['day'].max() - args.test_days)]

train.to_csv("%s_train.csv" % args.output_prefix, index=False)
test.to_csv("%s_test.csv" % args.output_prefix, index=False)