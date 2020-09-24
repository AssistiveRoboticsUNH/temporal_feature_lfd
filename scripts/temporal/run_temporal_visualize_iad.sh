#!/bin/sh

SEGMENTS=16
BATCH=4
BOTTLENECK=32

SAVE_ID="regular_tsm_full_bs_bn32"
SAVE_ID_DITRL=$SAVE_ID #"_ditrl"

BACKBONE_MODEL="saved_models/saved_model_"$SAVE_ID".backbone.pt"
VIZUALIZE_IAD="python3 iad_visualization.py bs --trim --batch_size "$BATCH" --save_id "$SAVE_ID_DITRL" --ditrl --backbone_modelname "$BACKBONE_MODEL" --num_segments "$SEGMENTS" --bottleneck "$BOTTLENECK

echo $VIZUALIZE_IAD

echo "====="
echo "VIZUALIZE IADs"
echo "====="

echo $VIZUALIZE_IAD
eval $VIZUALIZE_IAD
