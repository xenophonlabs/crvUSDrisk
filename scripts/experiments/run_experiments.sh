#!/bin/bash

time=$(date +%s)
log_file="logs/experiments_${time}.log"
mkdir -p "$(dirname "$log_file")"
echo $log_file

python3 -m scripts.experiments.generic &>> "$log_file"
echo "Generic experiments done" &>> "$log_file"

python3 -m scripts.experiments.sweep_chainlink_limits &>> "$log_file"
echo "Chainlink sweep experiments done" &>> "$log_file"

python3 -m scripts.experiments.sweep_debt_ceilings &>> "$log_file"
echo "Debt ceiling sweep experiments done" &>> "$log_file"
