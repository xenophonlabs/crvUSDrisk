"""
Script to sweep the debt ceilings for all markets.

Note
----
We sweep debt ceilings by multiplying existing debt 
ceilings by a scalar for all markets simultaneously.
"""
from ..sweep import sweep
from ...src.configs.parameters import DEBT_CEILING_SWEEP

scenarios = [
    "baseline",
    "adverse vol",
    "severe vol",
    "severe vol and adverse growth",
    "severe vol and severe growth",
]

sweep(
    "debt_ceilings",
    scenarios,
    num_iter=250,
    to_sweep=DEBT_CEILING_SWEEP,
)
