#!/bin/sh

SEGMENTS=16
EPOCHS=200
ALPHA=0.0005
BATCH=3

SAVE_ID="model0"
SAVE_ID_DITRL=$SAVE_ID"_ditrl"

BACKBONE_MODEL="saved_models/saved_model_"$SAVE_ID".backbone.pt"
OUTPUT_NAME_DITRL="csv_output/output_"$SAVE_ID_DITRL".csv"

GENERATE_ITR_CMD="python3 train_temporal_pipeline.py bs --trim --batch_size "$BATCH" --save_id "$SAVE_ID_DITRL" --ditrl --backbone_modelname "$BACKBONE_MODEL" --num_segments "$SEGMENTS
TRAIN_CMD="python3 train_temporal_ext.py bs --trim --epochs "$EPOCHS" --batch_size "$BATCH" --save_id "$SAVE_ID_DITRL" --ditrl --lr "$ALPHA
EVAL_CMD="python3 eval_temporal.py bs --trim --save_id "$SAVE_ID_DITRL" --ditrl"
ANALYZE_CMD="python3 analysis/action_metrics.py "$OUTPUT_NAME_DITRL

echo $GENERATE_ITR_CMD
echo ""
echo $TRAIN_CMD
echo ""
echo $EVAL_CMD
echo ""
echo $ANALYZE_CMD
echo ""

echo "====="
echo "GENERATE ITRs"
echo "====="

echo $GENERATE_ITR_CMD
eval $GENERATE_ITR_CMD

echo "====="
echo "TRAIN"
echo "====="

echo $TRAIN_CMD
eval $TRAIN_CMD

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