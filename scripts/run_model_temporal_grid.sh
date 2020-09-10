#!/bin/sh

EPOCHS=200

GRID_CMD="python3 grid_temporal.py bs --trim --epochs "$EPOCHS

echo "====="
echo "EXECUTE GRID SEARCH"
echo "====="

echo $GRID_CMD
eval $GRID_CMD
