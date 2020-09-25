#!/bin/sh

SEGMENTS=16
EPOCHS=50
ALPHA=0.0001
BATCH=4
BOTTLENECK=32

SAVE_ID="regular_tsm_full_bs_bn32"
SAVE_ID_DITRL=$SAVE_ID"_ditrl3"

BACKBONE_MODEL="saved_models/saved_model_"$SAVE_ID".backbone.pt"
OUTPUT_NAME_DITRL="csv_output/output_"$SAVE_ID_DITRL".csv"

GENERATE_ITR_CMD="python3 train_temporal_pipeline.py bs --trim --batch_size "$BATCH" --save_id "$SAVE_ID_DITRL" --ditrl --backbone_modelname "$BACKBONE_MODEL" --num_segments "$SEGMENTS" --bottleneck "$BOTTLENECK
TRAIN_CMD="python3 train_temporal_ext.py bs --trim --epochs "$EPOCHS" --batch_size "$BATCH" --save_id "$SAVE_ID_DITRL" --ditrl --lr "$ALPHA" --bottleneck "$BOTTLENECK
EVAL_CMD="python3 eval_temporal.py bs --trim --save_id "$SAVE_ID_DITRL" --ditrl --bottleneck "$BOTTLENECK
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