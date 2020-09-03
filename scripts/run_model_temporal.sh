#!/bin/sh

SEGMENTS=8
EPOCHS=200
ALPHA=0.0005
BATCH=3

SAVE_ID="model0"
SAVE_ID_DITRL=$SAVE_ID"_ditrl"

BACKBONE_MODEL="saved_models/saved_model_"$SAVE_ID".backbone.pt"

echo "====="
echo "TRAIN"
echo "====="

echo "python3 train_model.py bs --trim --epochs "$EPOCHS" --batch_size "$BATCH" --save_id "$SAVE_ID_DITRL" --ditrl --backbone_modelname "$BACKBONE_MODEL" --num_segments "$SEGMENTS" --lr "$ALPHA
python3 train_model.py bs --trim --epochs $EPOCHS --batch_size $BATCH --save_id $SAVE_ID_DITRL --ditrl --backbone_modelname $BACKBONE_MODEL --num_segments $SEGMENTS --lr $ALPHA

echo "====="
echo "EVAL"
echo "====="

echo "python3 eval_model.py bs --trim --save_id "$SAVE_ID_DITRL" --ditrl"
python3 eval_model.py bs --trim --save_id $SAVE_ID_DITRL --ditrl 

echo "====="
echo "ANALYZE"
echo "====="

OUTPUT_NAME_DITRL="csv_output/output_"$SAVE_ID_DITRL".csv"
python3 analysis/action_metrics.py $OUTPUT_NAME_DITRL