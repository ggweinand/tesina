#!/bin/bash

FILENAME=min_obs_all_snr20.csv

touch ${FILENAME}

echo "bm_src_id,vs_type,obs_threshold,PeriodLS" > ${FILENAME}

for tile in "b206" "b214" "b216" "b220" "b228" "b234" "b247" "b248" "b261" "b262" "b263" "b264" "b277" "b278" "b356" "b360" "b396"
do
    tail -n +2 min_obs_${tile}_snr20.csv >> ${FILENAME}
done
