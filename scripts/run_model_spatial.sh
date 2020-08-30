#!/bin/sh

EPOCHS=10
BATCH=8

SAVE_ID="model0"

python3 train_model.py bs --trim --epochs $EPOCHS --batch_size $BATCH --save_id $SAVE_ID
python3 eval_model.py bs --trim --save_id $SAVE_ID

OUTPUT_NAME="csv_output/output_"$SAVE_ID".csv"

python3 analysis/action_metrics.py $OUTPUT_NAME
