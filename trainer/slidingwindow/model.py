import tensorflow as tf
import numpy as np
import pandas as pd
import geohash
from math import cos, sin, sqrt
from keras.callbacks import TensorBoard
import os
from random import seed
from time import time
from sklearn.metrics import mean_squared_error
from keras import Sequential
from keras.layers import LSTM, Dense, Lambda, Dropout
from trainer.utils import copy_file_to_gcs, to_savedmodel
from trainer.slidingwindow.data_utils import timeseries, keras_timeseries_generator
import argparse
from tensorflow.python.lib.io import file_io
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping

seed(42)
np.random.seed(42)

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)


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

    dfx = df.sort_values(by=['day', 'mins', 'geohash6'], ascending=True)
    X = dfx[['dow', 'mins_norm', 'x', 'y', 'z', 'demand']]
    y = dfx[['demand']]

    return X, y

#https://stackoverflow.com/questions/43855162/rmse-rmsle-loss-function-in-keras
#https://github.com/keras-team/keras/issues/10706
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# https://www.kaggle.com/asuilin/smape-variants
# https://www.kaggle.com/c/allstate-claims-severity/discussion/26461
def sMAPE(y_true, y_pred):
    #Symmetric mean absolute percentage error
    return 100 * K.mean(K.abs(y_pred - y_true) / (K.abs(y_pred) + K.abs(y_true)), axis=-1)

def model_fn(past_steps, future_steps, features):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(past_steps, features)),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(future_steps)
    ])
    model.compile(loss=sMAPE, optimizer='adam',
        metrics=['mse', 'mape', 'mae', 'cosine', root_mean_squared_error, sMAPE])
    model.summary()
    return model


if __name__ == '__main__':
    # Parse the input arguments for common Cloud ML Engine options
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-file',
        help='Cloud Storage bucket or local path to training data')
    parser.add_argument(
        '--job-dir',
        type=str,
        default="/tmp/timeseries",
        help='Cloud storage bucket to export the model and store temp files')
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
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10,
        help='Maximum number of epochs on which to train')
    args = parser.parse_args()
    arguments = args.__dict__

    try:
        os.makedirs(args.job_dir)
    except:
        pass

    X, y = get_data(args.train_file)

    split_n = int(0.8*len(y))
    train_X = X.iloc[:split_n, :].values
    train_y = y.iloc[:split_n, :].values

    test_X = X.iloc[split_n:, :].values
    test_y = y.iloc[split_n:, :].values

    print("Train shape", train_X.shape, train_y.shape)
    print("Test shape", test_X.shape, test_y.shape)

    # Personal note: `steps_per_epoch` is used to generate the *entire dataset* once by calling the generator `steps_per_epoch` times
    # where as `epochs` gives the number of times the model is trained over the *entire dataset*.
    # (steps_per_epoch=10, epochs=1) and (steps_per_epoch=1, epochs=10) would yeild the same results.
    # the difference is the interval at when are fired, and usually used for callbacks.
    # Reference: https://github.com/keras-team/keras/issues/7506
    # Also incorrect batch sizes can result in ValueError: Error when checking input: expected lstm_1_input to have 3 dimensions, but got array with shape (0, 1)
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    past_steps = args.past_steps
    future_steps = args.future_steps
    features = train_X.shape[1]

    MODEL_FILENAME = 'model.h5'

    print("Args", args)

    model = model_fn(past_steps, future_steps, features)

    generator = keras_timeseries_generator(
        train_X, train_y, BATCH_SIZE, past_steps, future_steps)

    val_generator = keras_timeseries_generator(
        test_X, test_y, BATCH_SIZE, past_steps, future_steps)

    checkpoint_path = "model-epoch{epoch:02d}-val_loss{val_loss:.5f}.h5"
    if not args.job_dir.startswith('gs://'):
        checkpoint_path = os.path.join(args.job_dir, checkpoint_path)

    class GCSCheckpoint(ModelCheckpoint):
        def on_epoch_end(self, epoch, logs=None):
            super().on_epoch_end(epoch, logs)
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            #print("ModelCheckpoint args", args.job_dir)
            if args.job_dir.startswith('gs://') and os.path.isfile(filepath):
                copy_file_to_gcs(args.job_dir, filepath)

    checkpoint = GCSCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    earlystop = EarlyStopping(monitor='val_loss', patience=50, verbose=1)

    tensorboard = TensorBoard(log_dir=os.path.join(
        args.job_dir, "logs/{}".format(time())))

    history = model.fit_generator(generator,
        steps_per_epoch=len(train_X) // BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=1,
        callbacks=[tensorboard, checkpoint, earlystop],
        validation_data=val_generator,
        validation_steps=len(test_X)//BATCH_SIZE)

    # model.save_weights('./weights.h5')

    test_Xs, test_ys = timeseries(test_X, test_y, past_steps, future_steps)
    print("test_Xs", test_Xs.shape, "test_ys", test_ys.shape)
    print("test_X", test_X.shape, "test_y", test_y.shape)
    yhat = model.predict(test_Xs, batch_size=BATCH_SIZE, verbose=0)
    mse = sqrt(mean_squared_error(test_ys, yhat))
    print('RMSE: %f' % mse)

    evaluate = model.evaluate(test_Xs, test_ys, batch_size=BATCH_SIZE)
    print("Evaluation", evaluate)

    # Unhappy hack to workaround h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    if args.job_dir.startswith('gs://'):
        model.save(MODEL_FILENAME)
        copy_file_to_gcs(args.job_dir, MODEL_FILENAME)
    else:
        model.save(os.path.join(args.job_dir, MODEL_FILENAME))

    # Convert the Keras model to TensorFlow SavedModel.
    to_savedmodel(model, os.path.join(args.job_dir, 'export'))