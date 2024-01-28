"""
Script to sweep LLAMMA fees.
"""
from src.configs.parameters import FEE_SWEEP
from ..sweep import sweep

# pylint: disable=duplicate-code
scenarios = [
    "baseline",
    "adverse vol",
    "severe vol",
]

if __name__ == "__main__":
    sweep(
        "fees",
        scenarios,
        num_iter=1000,
        to_sweep=FEE_SWEEP,
    )
