#!/bin/sh

BATCH_SIZE=4
EPOCHS=50 # (on shield_ip)

GRID_CMD="python3 grid_spatial.py bs --trim --epochs "$EPOCHS" --batch_size "$BATCH_SIZE

echo "====="
echo "EXECUTE GRID SEARCH"
echo "====="

echo $GRID_CMD
eval $GRID_CMD
