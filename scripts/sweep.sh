#!/bin/bash

time=$(date +%s)
log_file="logs/sweep_${time}.log"
mkdir -p "$(dirname "$log_file")"
echo $log_file

python3 -m scripts.sweep &>> "$log_file"
