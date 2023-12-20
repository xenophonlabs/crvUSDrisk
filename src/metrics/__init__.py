"""
Provides Metrics classes for each simulated entity.

Metrics
-------
Arbitrageur: Profit
Arbitrageur: Count
Arbitrageur: Volume TODO
Liquidator:  Profit
Liquidator:  Count
Liquidator: Volume TODO
Keeper: Profit
Keeper: Count
Keeper: Volume TODO
LLAMMA: Price
LLAMMA: Oracle price
LLAMMA: Fees x
LLAMMA: Fees y
LLAMMA: Balances FIXME currently just sum over bands
Controller: System Health
Controller: Bad Debt
Controller: Number of loans
Controller: Debt
Controller: Users to liquidate
Controller: When are liquidations "bad"? TODO
StableSwap: Price
StableSwap: MA Price
StableSwap: LP Token Supply
StableSwap: Virtual Price
StableSwap: Balances
PK: Debt
PK: Profit
Aggregator: Price
ERC20: Total Supply
ERC20: Net unbacked crvusd TODO
"""

from typing import List, Type
from .agent import AgentMetrics
from .aggregator import AggregatorMetrics
from .base import Metric
from .controller import ControllerMetrics
from .llamma import LLAMMAMetrics
from .pegkeeper import PegKeeperMetrics
from .stablecoin import StablecoinMetrics
from .stableswap import StableSwapMetrics

DEFAULT_METRICS = [
    AgentMetrics,
    AggregatorMetrics,
    ControllerMetrics,
    LLAMMAMetrics,
    PegKeeperMetrics,
    StablecoinMetrics,
    StableSwapMetrics,
]

__all__ = [
    "DEFAULT_METRICS",
    "Metric",
]


def init_metrics(metrics: List[Type[Metric]], **kwargs) -> List[Metric]:
    """
    Initialize metrics.

    Parameters
    ----------
    metrics : List[Type[Metric]]
        List of metric classes.

    Returns
    -------
    List[Metric]
        List of metric instances.
    """
    return [metric(**kwargs) for metric in metrics]
