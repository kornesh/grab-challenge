import tensorflow as tf
import os
import numpy as np
import sys
import argparse
from keras.models import load_model
from trainer.slidingwindow.model import sMAPE, root_mean_squared_error
from trainer.slidingwindow.data_utils import keras_timeseries_generator, get_data, postprocess_data
from trainer.graph_utils import prediction_graph
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
    default=100,
    help='Batch size for steps')

args = parser.parse_args()

X, y = postprocess_data(get_data(args.data))
generator = keras_timeseries_generator(X, y, args.batch_size, args.past_steps, args.future_steps)

model = load_model(args.model, custom_objects={'sMAPE': sMAPE, 'root_mean_squared_error': root_mean_squared_error})

#predicts = model.predict_generator(generator, steps=1)
#print("predicts", predicts.shape, predicts)
#for i in range(10, 100, 10):
#   prediction_graph(Xs, ys, predicts, i, args.past_steps, args.future_steps)

evaluation = model.evaluate_generator(generator, steps=1)

metrics = ['sMAPE', 'mse', 'mape', 'mae', 'cosine', 'rmse', 'sMAPE']
for i, v in zip(metrics, evaluation):
    print(i, v)

