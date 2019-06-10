# Sliding Window
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
The idea is to convert the matrix above to the one below
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

# Prediction
```bash
# head -n 100000 training.csv > evaluation.csv
wget -O model.h5 https://storage.googleapis.com/sr-semantic-search/jobs/timeseries_14/model-epoch01-val_loss43.72372.h5
python -m trainer.slidingwindow.predict --past-steps 100 --batch-size 10000 --model model.h5 --data evaluation.csv
```

# Training on ML Engine
```bash
export JOB_NAME=timeseries_7
export GCS_TRAIN_FILE=gs://sr-semantic-search/grab-challenge/training.csv
export JOB_DIR=gs://sr-semantic-search/jobs/$JOB_NAME
gcloud ml-engine jobs submit training $JOB_NAME \
--scale-tier basic-gpu \
--python-version=3.5 \
--stream-logs \
--runtime-version 1.12 \
--job-dir $JOB_DIR \
--package-path trainer \
--module-name trainer.slidingwindow.model \
--region us-west1 \
-- \
--train-file $GCS_TRAIN_FILE \
--num-epochs 50 \
--batch-size 50000 \
--past-steps 100
```

# Training locally
```bash
gcloud ml-engine local train \
--package-path trainer \
--module-name trainer.slidingwindow.model \
-- \
--train-file train_dev.csv \
--job-dir dialog --num-epochs 10 --batch-size 10
```