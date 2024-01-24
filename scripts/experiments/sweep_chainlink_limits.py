"""
Script to sweep the chainlink oracle limits.
"""
from src.configs.parameters import CHAINLINK_LIMIT_SWEEP
from ..sweep import sweep

# pylint: disable=duplicate-code
scenarios = [
    "adverse flash crash",
    "severe flash crash",
    "adverse depeg",
]

if __name__ == "__main__":
    sweep(
        "chainlink_limits",
        scenarios,
        num_iter=250,
        to_sweep=CHAINLINK_LIMIT_SWEEP,
    )
