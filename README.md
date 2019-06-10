# Grab AI Challenge: Traffic Management

Each of the `geohash6` can be considered a separate discontinous time series. Traditional time series method like ARIMA can make good prediction for series specific problems but it doesn't scale well to predict thousands of series as it fails to capture the general patterns among those series.

I'll be taking minimalistic approach to feature engineering since I'll be only considering neural networks which are capable of learning features automatically.

# Model 1: Sliding Window with LSTM (Final solution)
```
       x1     x2      y
0     0.0    5.0    5.0
1    10.0   15.0   25.0
2    20.0   25.0   45.0
3    30.0   35.0   65.0
4    40.0   45.0   85.0
5    50.0   55.0  105.0
6    60.0   65.0  125.0
7    70.0   75.0  145.0
8    80.0   85.0  165.0
9    90.0   95.0  185.0
10  100.0  105.0  205.0
```
The basic idea behind this model is to convert the matrix above to the one below and feed it to through a vanilla LSTM model.
```
        X           =>          y
[[  0.   5.   5.]
[ 10.  15.  25.]
[ 20.  25.  45.]]   =>   [ 65.  85. 105.]

[[ 10.  15.  25.]
[ 20.  25.  45.]
[ 30.  35.  65.]]   =>   [ 85. 105. 125.]

[[ 20.  25.  45.]
[ 30.  35.  65.]
[ 40.  45.  85.]]   =>   [105. 125. 145.]
```
## Architecture
I haven't really spend much time on experiementing with different architectures. The current one looks like this
```python
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(past_steps, num_features)),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(future_steps)
    ])
```

## Hyperparameter tunning
I have not performed manual hyperparameter tuning as I will be doing it using [ML Engine](https://cloud.google.com/ml-engine/docs/tensorflow/using-hyperparameter-tuning) once I'm satisfied with the current model.

## Training on ML Engine using TPU
My attempts to train this model with `past-steps` more than 100 have always resulted in OOM error, even on `BASIC_TPU`. So the current model will be only looking at past 100 records to predict next 5 `future-steps`.
```bash
export JOB_NAME=timeseries_15
export GCS_TRAIN_FILE=gs://sr-semantic-search/grab-challenge/training.csv
export JOB_DIR=gs://sr-semantic-search/jobs/$JOB_NAME
gcloud ml-engine jobs submit training $JOB_NAME \
--scale-tier BASIC_TPU \
--python-version=3.5 \
--stream-logs \
--runtime-version 1.12 \
--job-dir $JOB_DIR \
--package-path trainer \
--module-name trainer.slidingwindow.model \
--region us-central1 \
-- \
--train-file $GCS_TRAIN_FILE \
--num-epochs 50 \
--batch-size 1000 \
--past-steps 100
```

## Training locally
```bash
gcloud ml-engine local train \
--package-path trainer \
--module-name trainer.slidingwindow.model \
-- \
--train-file train_dev.csv \
--job-dir dialog --num-epochs 10 --batch-size 10
```


## Prediction / evaluation
```bash
# head -n 100000 training.csv > evaluation.csv
wget -O model.h5 https://storage.googleapis.com/sr-semantic-search/jobs/timeseries_14/model-epoch01-val_loss43.72372.h5
python -m trainer.slidingwindow.predict --past-steps 100 --batch-size 10000 --model model.h5 --data evaluation.csv
```

# Model 2: Encoder Decoder with teacher forcing (WIP)
Attempting to try the encoder decoder model with teacher forcing described in [Keras blog](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html) with this problem.

```
        geohash6  day  timestamp    demand  h  m  dow
      1   qp03q8    1          0  0.021055  0  0    1
      2   qp03wn    1          0  0.032721  0  0    1
      3   qp03x5    1          0  0.157956  0  0    1
      4   qp03xt    1          0  0.120760  0  0    1
      5   qp03yb    1          0  0.244790  0  0    1

                        ...

4206317   qp09e8   61      87825  0.000591  23  45    5
4206318   qp09eb   61      87825  0.014968  23  45    5
4206319   qp09g1   61      87825  0.053489  23  45    5
4206320   qp09g7   61      87825  0.028502  23  45    5
4206321   qp09u8   61      87825  0.027069  23  45    5
```

In this model we compute `timestamp = ((day - 1) * 24 * 60) + (h * 60) + m` and transform the data above into `timestamp` vs `geohash6` table as shown below. Note that we're considering each of `geohash6` as separate series without resolving/calculating it's latitude and longitude. Also note that we're using the value `0.0000001` when there is no data available, to prevent vanishing/exploding gradients.
```
           0         15        30       ...        87795     87810     87825
qp03q8 0.0210552 0.0000001 0.0000001    ...    0.0000001 0.0000001 0.0000001
qp03wn 0.0327211 0.0000001 0.0000001    ...    0.0000001 0.0000001 0.0000001
qp03x5 0.1579564 0.0000001 0.0000001    ...    0.0000001 0.0000001 0.0000001
qp03xt 0.1207595 0.0000001 0.0000001    ...    0.0000001 0.0000001 0.0000001
qp03yb 0.2447899 0.0000001 0.0000001    ...    0.0000001 0.0000001 0.0000001

                                        ...

qp0djv 0.0000001 0.0000001 0.0000001    ...    0.0000001 0.0000001 0.0000001
qp08g7 0.0000001 0.0000001 0.0000001    ...    0.0000001 0.0000001 0.0000001
qp0d5q 0.0000001 0.0000001 0.0000001    ...    0.0000001 0.0000001 0.0000001
qp09cz 0.0000001 0.0000001 0.0000001    ...    0.0000001 0.0000001 0.0000001
qp08uj 0.0000001 0.0000001 0.0000001    ...    0.0000001 0.0000001 0.0000001
```
## Data transformation
This transformation is pretty slow as it is looping through entire dataset one by one. I'm sure there are better ways to achieve this but since we only have to do it once, I'm not worried about the performance. We transform the data and save it as a `.hdf5` file which can be loaded fast into memory.
```bash
python -m trainer.encoderdecoder.data_utils --input training_dev.csv --output training_dev.h5
```

# Model 3: WaveNet (To-do)

DeepMind's [Wavenet](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) architecure uses CNNs with *dilated causal convolution layer*  to learn temporal patterns as well as long-term dependencies across a large timesteps, better than recurrent neural networks like LSTM. We can capture years worth of context with less than 10 dilated convolutions! It's a CNN, so it's faster than RNNs too.

I'm planning to make an attempt at implmenting this.

![](https://storage.googleapis.com/deepmind-live-cms-alt/documents/BlogPost-Fig2-Anim-160908-r01.gif)

# Model 4: Neural ODE (To-do)

NIPS 2018's one of the best paper, titled [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) takes Microsoft's residual layers to whole new level by reframing it to use ODE solvers (like Euler's method).

ODE is a function that describes change of a system over continous period of time. This paper  proposes neural networks as the continously evolving system. So instead of specifying and training a network with discrete layers and updating hidden units layer by layer, we specify their derivative with respect to depth instead.

![](https://i.imgur.com/IHR3iC2.gif)