# Grab AI Challenge: Traffic Management

Each of the `geohash6` can be considered as a separate discontinous time series. Traditional time series methods like ARIMA can make good predictions on series specific problems but it doesn't scale well to predict thousands of series as it fails to capture the general patterns among those series.

I'll be taking minimalistic approach to feature engineering since I'll be only considering neural networks which are capable of learning features automatically.

Models performance is only one of four criterias that is being considered for [this challenge](https://www.aiforsea.com/traffic-management). Instead of solely focusing on achieving best model performance possible, I am more interested in exploring newer and much more efficient architectures like [WaveNet](https://arxiv.org/abs/1711.10433) and [Neural ODE](https://arxiv.org/abs/1806.07366).

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
I added a few additonal features like `mins = (h * 60) + m` and it's normalized version `mins_norm = mins / (60 * 24)`. After extracting latitude and longitude from `geohash6`, we can convert them into [Cartesian coordinates](https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates). It turns out these 2 features actually represent a three dimensional space. Unlike in latitude and longitude, close points in these 3 dimensions are also close in reality. So this standardizes the features nicely.
```
        geohash6  day timestamp    demand   h   m  mins  mins_norm  dow      lat     lat_err     long    long_err         x         y         z
      0   qp02zd    1       0:0  0.022396   0   0     0        0.0    1 -5.47943  0.00274658  90.6866  0.00549316 -0.633821  0.282699  0.719967
      1   qp02zr    1       0:0  0.042046   0   0     0        0.0    1 -5.45197  0.00274658  90.6757  0.00549316 -0.612472  0.281284  0.738754
      2   qp02zt    1       0:0  0.001112   0   0     0        0.0    1 -5.46295  0.00274658  90.6866  0.00549316   -0.6229  0.277828  0.731305
      3   qp02zu    1       0:0  0.001831   0   0     0        0.0    1 -5.46844  0.00274658  90.6976  0.00549316 -0.629592  0.272559  0.727548
      4   qp02zv    1       0:0  0.006886   0   0     0        0.0    1 -5.46295  0.00274658  90.6976  0.00549316 -0.625915  0.270968  0.731305

                                                                ...

4206317   qp0dj1   61     23:45  0.040727  23  45  1425   0.989583   5  -5.2652  0.00274658  90.9283  0.00549316 -0.516798  0.0928899  0.851053
4206318   qp0dj4   61     23:45  0.003259  23  45  1425   0.989583   5  -5.2597  0.00274658  90.9283  0.00549316 -0.512189  0.0920615  0.853925
4206319   qp0dj5   61     23:45  0.058156  23  45  1425   0.989583   5 -5.25421  0.00274658  90.9283  0.00549316 -0.507565  0.0912302   0.85677
4206320   qp0djh   61     23:45  0.011935  23  45  1425   0.989583   5 -5.24872  0.00274658  90.9283  0.00549316 -0.502925  0.0903963   0.85959
4206321   qp0djk   61     23:45  0.003414  23  45  1425   0.989583   5 -5.24872  0.00274658  90.9393  0.00549316 -0.503887  0.0848656   0.85959
```

As for their error (`lat_err` and `long_err`), I have no idea what to do with them. We could just push all of these features to a neural network and let it figure out which features are important and which are not. However, I had hard time training a model with just 6 features, even on a TPU. It is possible that there's a memory leak with Keras' [fit_generator](https://www.google.com/search?q=keras+fit_generator+memory+leak) method. 

This challenge allows us to look-back up to 14 days or at least `(14 * 24 * 60) / 15 = 1344` intervals. This is the lower limit, `past-steps >= 1344`, since each interval can have multiple series from different `geohash6` locations, as long as it's within the 14 days window. However, for this model, even with just 6 features below, I was only able to train `past-steps = 100` anything above that resulted in out-of-memory error, even on `BASIC_TPU`. This could be due to the exponential nature of the sliding window method I used to generate inputs to the model.

The below is the final features that I used to generate inputs for this model with sliding window method described above.

```
         dow  mins_norm         x         y         z    demand
      0    1        0.0 -0.633821  0.282699  0.719967  0.022396
      1    1        0.0 -0.612472  0.281284  0.738754  0.042046
      2    1        0.0   -0.6229  0.277828  0.731305  0.001112
      3    1        0.0 -0.629592  0.272559  0.727548  0.001831
      4    1        0.0 -0.625915  0.270968  0.731305  0.006886

                                ...

4206317    5   0.989583 -0.516798  0.0928899  0.851053  0.040727
4206318    5   0.989583 -0.512189  0.0920615  0.853925  0.003259
4206319    5   0.989583 -0.507565  0.0912302   0.85677  0.058156
4206320    5   0.989583 -0.502925  0.0903963   0.85959  0.011935
4206321    5   0.989583 -0.503887  0.0848656   0.85959  0.003414
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
Attempting to implement the encoder decoder model with teacher forcing described in [Keras blog](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html) by transforming the dataset into a continous series as described below.

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

I'm planning to make an attempt at implementing this.

![](https://storage.googleapis.com/deepmind-live-cms-alt/documents/BlogPost-Fig2-Anim-160908-r01.gif)

# Model 4: Neural ODE (To-do)

NIPS 2018's one of the best paper, titled [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) takes Microsoft's residual layers to whole new level by reframing it to use ODE solvers (like Euler's method).

ODE is a function that describes change of a system over continous period of time. This paper  proposes neural networks as the continously evolving system. So instead of specifying and training a network with discrete layers and updating hidden units layer by layer, we specify their derivative with respect to depth instead.

![](https://i.imgur.com/IHR3iC2.gif)