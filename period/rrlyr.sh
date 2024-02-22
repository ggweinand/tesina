#!/bin/bash

FILTERED="../filter/filtered_$1_features.csv"
RRLYR="rrlyr_$1_features.csv"

head -n1 "$FILTERED" > "$RRLYR"
grep 'RRL' "$FILTERED" >> "$RRLYR"

python rrlyr_lc.py "$1"
