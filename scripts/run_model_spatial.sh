#!/bin/sh

SEGMENTS=16
EPOCHS=200
ALPHA=0.0005
BATCH=3

SAVE_ID="model0"
OUTPUT_NAME="csv_output/output_"$SAVE_ID".csv"

TRAIN_CMD="python3 train_spatial.py bs --trim --epochs "$EPOCHS" --batch_size "$BATCH" --save_id "$SAVE_ID" --num_segments "$SEGMENTS" --lr "$ALPHA
EVAL_CMD="python3 eval_spatial.py bs --trim --save_id "$SAVE_ID
ANALYZE_CMD="python3 analysis/action_metrics.py "$OUTPUT_NAME

echo $TRAIN_CMD
echo ""
echo $EVAL_CMD
echo ""
echo $ANALYZE_CMD
echo ""

echo "====="
echo "TRAIN"
echo "====="

echo $TRAIN_CMD
#eval $TRAIN_CMD

echo "====="
echo "EVAL"
echo "====="

echo $EVAL_CMD
eval $EVAL_CMD

echo "====="
echo "ANALYZE"
echo "====="

echo $ANALYZE_CMD
eval $ANALYZE_CMD