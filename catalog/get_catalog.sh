#!/bin/bash


for tile in "b206" "b214" "b216" "b220" "b228" "b234" "b247" "b248" "b261" "b262" "b263" "b264" "b277" "b278" "b356" "b360" "b396"
do
    carpyncho download-catalog ${tile} features --out ${tile}_features.csv
    carpyncho download-catalog ${tile} lc --out ${tile}_lc.csv
done
