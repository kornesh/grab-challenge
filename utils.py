from keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd
import numpy as np

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras.callbacks import TensorBoard
import os
from random import seed
from time import time

seed(42)
np.random.seed(42)

def generate_synthetic_data(start=1, end=100):
	# define input sequence
	in_seq1 = np.array([[i] for i in range(start, end, 10)], dtype=np.float32)
	in_seq2 = np.array([[(i+5)] for i in range(start, end, 10)], dtype=np.float32)
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

def timeseries_generator(X, y, batch_size, past_steps, future_steps):
    num_batches = len(X) // batch_size
    #print("generator", num_batches)
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

def predict(test_y, yhat, i):
  truth = test_y[i:i+15].reshape(15)
  predicted = yhat[i].reshape(5)
  #predicted = test_y[-5:]
  #predicted = test_ys[-1:].reshape(5)


  print(truth[0:10])
  print(truth[10:])
  print(predicted)

  plt.figure(figsize=(10,6))   
  plt.plot(range(0, 11), truth[0:11])
  plt.plot(range(10,15), truth[10:], color='orange')
  plt.plot(range(10,15), predicted, color='teal',linestyle='--')

  plt.legend(['Truth','Target','Predictions'])
  plt.show

if __name__ == "__main__":
	# @TODO(kornesh): Write proper tests!
	X, y = generate_synthetic_data(0, 100)
	print(X)
	print(y)

	past_steps = 3
	future_steps = 3
	batch_size  = 4

	generator = timeseries_generator(X, y, batch_size, past_steps, future_steps)

	for x, y in generator:
		print('%s => %s' % (x, y))

	X, y = generate_synthetic_data(0, 100)

	Xs, ys = timeseries(X, y, past_steps, future_steps)
	print(Xs, ys)