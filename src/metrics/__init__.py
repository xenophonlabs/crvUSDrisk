"""
Provides Metrics classes for each simulated entity.
"""
from __future__ import annotations
from typing import List, Type, TYPE_CHECKING
from .base import Metric
from .metrics import (
    BadDebtMetric,
    SystemHealthMetric,
    BorrowerLossMetric,
    ValueLeakageMetric,
    PegStrengthMetric,
    LiquidationsMetric,
    PegKeeperMetric,
    LiquidityMetric,
    MiscMetric,
)

if TYPE_CHECKING:
    from ..sim.scenario import Scenario

DEFAULT_METRICS = [
    BadDebtMetric,
    SystemHealthMetric,
    BorrowerLossMetric,
    ValueLeakageMetric,
    PegStrengthMetric,
    LiquidationsMetric,
    PegKeeperMetric,
    LiquidityMetric,
    MiscMetric,
]

__all__ = [
    "DEFAULT_METRICS",
    "Metric",
]


def init_metrics(metrics: List[Type[Metric]], scenario: Scenario) -> List[Metric]:
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
    return [metric(scenario) for metric in metrics]
