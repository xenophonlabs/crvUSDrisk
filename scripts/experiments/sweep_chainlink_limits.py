"""
Script to sweep the chainlink oracle limits.
"""
from ..sweep import sweep
from src.configs.parameters import CHAINLINK_LIMIT_SWEEP

scenarios = [
    "baseline",
    "adverse vol",
    "severe vol",
    "severe vol and adverse crvusd liquidity",
    "severe vol and severe crvusd liquidity",
    "severe vol and very severe crvUSD liquidity",
]

if __name__ == "__main__":
    sweep(
        "chainlink_limits",
        scenarios,
        num_iter=250,
        to_sweep=CHAINLINK_LIMIT_SWEEP,
    )
