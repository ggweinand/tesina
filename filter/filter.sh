#!/bin/bash

python filter_mag.py "$1"
python filter_snr.py "$1"
python filter_features.py "$1"
python filter_lc.py "$1"
