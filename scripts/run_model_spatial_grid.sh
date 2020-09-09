#!/bin/sh

GRID_CMD="python3 grid_spatial.py bs --trim"

echo "====="
echo "EXECUTE GRID SEARCH"
echo "====="

echo $GRID_CMD
eval $GRID_CMD
