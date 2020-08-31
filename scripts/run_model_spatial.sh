#!/bin/sh

SEGMENTS=16
EPOCHS=10
BATCH=3

SAVE_ID="model0"

echo "---"
python3 train_model.py bs --trim --epochs $EPOCHS --batch_size $BATCH --save_id $SAVE_ID --num_segments $SEGMENTS

echo "---"
echo "python3 eval_model.py bs --trim --save_id "$SAVE_ID" --num_segments "$SEGMENTS
python3 eval_model.py bs --trim --save_id $SAVE_ID --num_segments $SEGMENTS

OUTPUT_NAME="csv_output/output_"$SAVE_ID".csv"

python3 analysis/action_metrics.py $OUTPUT_NAME
