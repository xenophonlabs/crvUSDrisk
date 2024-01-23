"""
Script to sweep the debt ceilings for all markets.

Note
----
We sweep debt ceilings by multiplying existing debt 
ceilings by a scalar for all markets simultaneously.
"""
from src.configs.parameters import DEBT_CEILING_SWEEP
from ..sweep import sweep

# pylint: disable=duplicate-code
scenarios = [
    "baseline",
    "adverse vol",
    "severe vol",
    "adverse flash crash",
    "severe flash crash",
    "severe vol and adverse growth",
    "severe vol and severe growth",
]

if __name__ == "__main__":
    sweep(
        "debt_ceilings",
        scenarios,
        num_iter=250,
        to_sweep=DEBT_CEILING_SWEEP,
    )
