#!/bin/sh

EPOCHS=3

GRID_CMD="python3 grid_spatial.py bs --trim --epochs"$EPOCHS

echo "====="
echo "EXECUTE GRID SEARCH"
echo "====="

echo $GRID_CMD
eval $GRID_CMD
