"""
Script to sweep the chainlink oracle limits.
"""
from src.configs.parameters import CHAINLINK_LIMIT_SWEEP
from ..sweep import sweep

# pylint: disable=duplicate-code
scenarios = [
    "adverse depeg",
]

if __name__ == "__main__":
    sweep(
        "chainlink_limits",
        scenarios,
        num_iter=1000,
        to_sweep=CHAINLINK_LIMIT_SWEEP,
    )
