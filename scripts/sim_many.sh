#!/bin/bash

# Define the scenarios, markets, number of iterations, and number of runs
scenarios=("baseline_micro" "baseline_macro" "high_volatility")
markets="sfrxeth,wbtc,weth,wsteth"
markets_str="sfrxeth_wbtc_weth_wsteth"
num_iters=1000
num_runs=5

for scenario in "${scenarios[@]}"; do
    # Execute the script num_runs times for each scenario
    for (( i=1; i<=num_runs; i++ )); do
        log_file="logs/${scenario}/${markets_str}/log_${num_iters}_iters_${i}.log"
        echo $log_file
        mkdir -p "$(dirname "$log_file")"
        python3 -m scripts.sim "$scenario" "$markets" "$num_iters" -mp &> "$log_file"
    done
done
