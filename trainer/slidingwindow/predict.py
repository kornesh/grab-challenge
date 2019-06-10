import tensorflow as tf
import os
import numpy as np
import sys
import argparse
from keras.models import load_model
from trainer.slidingwindow.model import sMAPE, root_mean_squared_error
from trainer.slidingwindow.data_utils import timeseries, get_data
import keras.losses

keras.losses.custom_loss = sMAPE

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    type=str,
    help='Full model .h5 file')
parser.add_argument(
    '--data',
    type=str,
    help='Data to evaluate')
parser.add_argument(
    '--past-steps',
    type=int,
    default=10,
    help='Look back / prior steps to feed the model')
parser.add_argument(
    '--future-steps',
    type=int,
    default=5,
    help='Look ahead / predict steps')
parser.add_argument(
    '--batch-size',
    type=int,
    default=50000,
    help='Batch size for steps')

args = parser.parse_args()

X, y = get_data(args.data)
Xs, ys = timeseries(X.values, y.values, args.past_steps, args.future_steps)

model = load_model(args.model, custom_objects={'sMAPE': sMAPE, 'root_mean_squared_error': root_mean_squared_error})

predicts = model.predict(Xs, steps=1)
#print("predicts", predicts.shape, predicts)
#for i in range(10, 100, 10):
#    prediction_graph(Xs, ys, predicts, i)

evaluation = model.evaluate(Xs, ys, batch_size=args.batch_size)

metrics = ['sMAPE', 'mse', 'mape', 'mae', 'cosine', 'rmse', 'sMAPE']
for i, v in zip(metrics, evaluation):
    print(i, v)

