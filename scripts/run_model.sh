#!/bin/sh

SAVE_ID="model0"
SAVE_ID_DITRL="model0_ditrl"

BACKBONE_MODEL="saved_models/saved_model_"+$SAVE_ID+".backbone.pt"

echo $SAVE_ID
echo $SAVE_ID_DITRL
echo $BACKBONE_MODEL

#python3 train_model.py bs --trim --epochs 10 
python3 train_model.py bs --trim --epochs 10 --ditrl --backbone_modelname $BACKBONE_MODEL --save_id $SAVE_ID_DITRL

#python3 eval_model.py bs --trim --epochs 10 --save_id $(SAVE_ID)
python3 eval_model.py bs --trim --epochs 10 --save_id $SAVE_ID_DITRL

#python3 analysis/action_metrics.py csv_output/output_$(SAVE_ID).csv
python3 analysis/action_metrics.py "csv_output/output_"+$SAVE_ID_DITRL+".csv"