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
    "adverse flash crash",
]

if __name__ == "__main__":
    sweep(
        "fees",
        scenarios,
        num_iter=250,
        to_sweep=FEE_SWEEP,
    )
