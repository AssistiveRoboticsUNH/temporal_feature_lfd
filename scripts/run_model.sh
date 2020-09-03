#!/bin/sh

SEGMENTS=8
EPOCHS=200
ALPHA=0.0005
BATCH=3

SAVE_ID="model0"
SAVE_ID_DITRL=$SAVE_ID"_ditrl"

BACKBONE_MODEL="saved_models/saved_model_"$SAVE_ID".backbone.pt"

echo $SAVE_ID
echo $SAVE_ID_DITRL
echo $BACKBONE_MODEL

echo "====="
echo "TRAIN"
echo "====="
echo "python3 train_model.py bs --trim --epochs "$EPOCHS" --batch_size "$BATCH" --save_id "$SAVE_ID" --num_segments "$SEGMENTS" --lr "$ALPHA
python3 train_model.py bs --trim --epochs $EPOCHS --batch_size $BATCH --save_id $SAVE_ID --num_segments $SEGMENTS --lr $ALPHA
echo "-----"
BATCH=1
echo "python3 train_model.py bs --trim --epochs "$EPOCHS" --batch_size "$BATCH" --save_id "$SAVE_ID_DITRL" --ditrl --backbone_modelname "$BACKBONE_MODEL" --num_segments "$SEGMENTS" --lr "$ALPHA
python3 train_model.py bs --trim --epochs $EPOCHS --batch_size $BATCH --save_id $SAVE_ID_DITRL --ditrl --backbone_modelname $BACKBONE_MODEL --num_segments $SEGMENTS --lr $ALPHA

echo "====="
echo "EVAL"
echo "====="
echo "python3 eval_model.py bs --trim --save_id "$SAVE_ID
python3 eval_model.py bs --trim --save_id $SAVE_ID
echo "-----"
echo "python3 eval_model.py bs --trim --save_id "$SAVE_ID_DITRL
python3 eval_model.py bs --trim --save_id $SAVE_ID_DITRL



echo "====="
echo "ANALYZE"
echo "====="

OUTPUT_NAME="csv_output/output_"$SAVE_ID".csv"
OUTPUT_NAME_DITRL="csv_output/output_"$SAVE_ID_DITRL".csv"

python3 analysis/action_metrics.py $OUTPUT_NAME
echo "-----"
python3 analysis/action_metrics.py $OUTPUT_NAME_DITRL