"""
Script to run all scenarios with no parameter sweeps.

Note
----
The flash crash scenarios are DEPRECATED. Our simulation
horizon is too coarse to achieve reasonable results with them,
and on a shorter timescale jumps approximate volatility. As we
run our "severe" scenario with upwards of 250% annualized volatility
(which is a lot), jumps become unnecessary.
"""
from ..sweep import sweep

scenarios = [
    "baseline",
    "adverse vol",
    "severe vol",
    "adverse drift",
    "severe drift",
    "adverse growth",
    "severe growth",
    "adverse crvusd liquidity",
    "severe crvusd liquidity",
    "very severe crvusd liquidity",
    # "adverse flash crash",
    # "severe flash crash",
    "adverse depeg",
    "severe depeg",
    "severe vol and adverse drift",
    "severe vol and severe drift",
    "severe vol and adverse growth",
    "severe vol and severe growth",
    "severe vol and adverse crvusd liquidity",
    "severe vol and severe crvusd liquidity",
    "severe vol and very severe crvUSD liquidity",
    # "adverse flash crash and adverse growth",
    # "adverse flash crash and severe growth",
    # "adverse flash crash and adverse crvusd liquidity",
    # "adverse flash crash and severe crvusd liquidity",
]

if __name__ == "__main__":
    sweep(
        "generic",
        scenarios,
        num_iter=250,
        to_sweep=[{}],
    )
