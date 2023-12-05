"""
Provides the `MetricsResult` and `MetricsProcessor` classes,
as well as some helper functions.
"""
from typing import Any
import pandas as pd
from ..utils import get_crvusd_index
from ..sim.scenario import Scenario


def price_str(pool, crvusd_is_quote=True):
    idx = get_crvusd_index(pool)
    if not crvusd_is_quote:
        idx = idx ^ 1
    coins = pool.metadata["coins"]["names"]
    return f"{coins[idx^1]}_{coins[idx]}"


def entity_str(entity: Any, type: str):
    """
    Get a simplified name for the pool
    to use in metrics column names.
    """
    if type == "aggregator":
        return "aggregator"
    elif type == "llamma":
        name = entity.name.replace("Curve.fi Stablecoin ", "")
        return "_".join("llamma", name)
    elif type == "controller":
        name = entity.pool.AMM.name.replace("Curve.fi Stablecoin ", "")
        return "_".join("controller", name)
    elif type == "stableswap":
        name = entity.name.replace("Curve.fi Factory Plain Pool: ", "")
        name = name.replace("/", "_")
        return "_".join("stableswap", name)
    elif type == "pk":
        name = entity.POOL.name.replace("Curve.fi Stablecoin ", "")
        return "_".join("pk", name)


def price(pool, crvusd_is_quote=True):
    idx = get_crvusd_index(pool)
    if not crvusd_is_quote:
        idx = idx ^ 1
    return pool.price(idx ^ 1, idx)


def bal_str(pool, i):
    _pool_str = pool_str(pool)
    coin = pool.metadata["coins"]["names"][i]
    return _pool_str + "_" + coin + "_bal"


def lp_supply_str(pool):
    return pool_str(pool) + "_lp_supply"


class MetricsResult:
    """
    Stores metrics data for a single simulation
    """

    def __init__(self):
        pass


class MetricsProcessor:
    """
    Stores and massages metrics data
    for a single simulation

    Metrics
    ------------
    Agents: Arbitrage Profit
    Agents: Arbitrage Count
    Agents: Update Profit
    Agents: Update Count
    Agents: Liquidation Profit <- when are liquidations "bad"?
    Agents: Liquidation Count

    Pools: Balances
    Pools: Price
    StableSwap: LP Token Supply
    StableSwap: Virtual Price
    PK: Minted
    PK: Burned
    Aggregator: Price <- distance from peg?
    LLAMMA: Arbitrage volume
    Controller: System Health
    Controller: Bad Debt
    Controller: Liquidation volume
    ERC20: Total Supply
    ERC20: Volume

    General: Net unbacked crvusd
    """

    columns_base = [
        "timestamp",
    ]

    columns_agents = [
        "arbitrageur_profit",
        "arbitrageur_count",
        "keeper_profit",
        "keeper_count",
        "liquidator_profit",
        "liquidator_count",
    ]

    def __init__(self, scenario: Scenario):
        # TODO should handle multiple LLAMMAs
        # and be LLAMMA specific. Maybe there's
        # a better way to ID them than with
        # hashed addresses?
        columns_sim = []

        self.columns = self.columns_base + self.columns_agents + columns_sim

        self.df = pd.DataFrame(columns=self.columns)

        # TODO maybe should track volume in
        # all pools/markets.

    def update(self, scenario: Scenario):
        """
        Process metrics for the current timestep
        of the input `scenario`.
        """
        # Agents

    def process(self) -> MetricsResult:
        # TODO join with price paths df
        return MetricsResult()
