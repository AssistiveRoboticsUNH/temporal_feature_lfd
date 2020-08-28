#!/bin/sh

EPOCHS=10

SAVE_ID="model0"
SAVE_ID_DITRL=$SAVE_ID"_ditrl"

BACKBONE_MODEL="saved_models/saved_model_"$SAVE_ID".backbone.pt"

echo $SAVE_ID
echo $SAVE_ID_DITRL
echo $BACKBONE_MODEL

python3 train_model.py bs --trim --epochs $EPOCHS --save_id $SAVE_ID
python3 train_model.py bs --trim --epochs $EPOCHS --save_id $SAVE_ID_DITRL --ditrl --backbone_modelname $BACKBONE_MODEL

python3 eval_model.py bs --trim --save_id $SAVE_ID
python3 eval_model.py bs --trim --save_id $SAVE_ID_DITRL

OUTPUT_NAME="csv_output/output_"$SAVE_ID".csv"
OUTPUT_NAME_DITRL="csv_output/output_"$SAVE_ID_DITRL".csv"

python3 analysis/action_metrics.py $OUTPUT_NAME
python3 analysis/action_metrics.py $OUTPUT_NAME_DITRL