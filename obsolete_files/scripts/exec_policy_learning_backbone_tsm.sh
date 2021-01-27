#!/bin/sh

SEGMENTS=16 #16
EPOCHS=50
ALPHA=0.0001
BATCH=4
BOTTLENECK=64

SAVE_ID="tsm_bottleneck_"$BOTTLENECK"_4"

RUN_CMD="python3 exec_policy_learning_backbone_tsm.py bs --epochs "$EPOCHS" --batch_size "$BATCH" --save_id "$SAVE_ID" --lr "$ALPHA" --bottleneck_size "$BOTTLENECK

echo $RUN_CMD

echo "====="
echo "Run"
echo "====="

echo $RUN_CMD
eval $RUN_CMD
