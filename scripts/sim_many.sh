#!/bin/bash

scenarios=(
    "baseline" 
    "adverse vol" 
    "severe vol" 
    "adverse drift"
    "severe drift"
    "adverse growth"
    "severe growth"
    "adverse crvusd liquidity"
    "severe crvusd liquidity"
    "severe vol and adverse drift"
    "severe vol and severe drift"
    "severe vol and adverse growth"
    "severe vol and severe growth"
    "severe vol and adverse crvusd liquidity"
    "severe vol and severe crvusd liquidity"
)
num_iters=100
num_runs=5

time=$(date +%s)
log_file="logs/sim_${time}.log"
mkdir -p "$(dirname "$log_file")"
echo $log_file

for (( i=1; i<=num_runs; i++ )); do
    for scenario in "${scenarios[@]}"; do
        echo "Running Scenario: ${scenario}, run: ${i}"
        python3 -m scripts.sim "$scenario" "$num_iters" -mp &>> "$log_file"
    done
done
