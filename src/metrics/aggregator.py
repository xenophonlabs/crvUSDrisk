"""
Provides metrics on the Aggregator.
"""

from typing import TYPE_CHECKING, Union, Dict, List
from functools import cached_property
from .base import Metric
from .utils import entity_str

if TYPE_CHECKING:
    from crvusdsim.pool.sim_interface.sim_aggregator import SimAggregateStablePrice


class AggregatorMetrics(Metric):
    """
    Metrics computed on the Aggregator.
    """

    def __init__(self, **kwargs) -> None:
        self.aggregator: SimAggregateStablePrice = kwargs["aggregator"]

    @cached_property
    def config(self) -> dict:
        summary: Dict[str, List[str]] = {}
        plot: Dict[str, dict] = {}
        aggregator_str = entity_str(self.aggregator, "aggregator")
        summary[aggregator_str + "_price"] = ["mean", "min", "max"]
        plot[aggregator_str + "_price"] = {
            "title": "Aggregator Price",
            "kind": "line",
        }
        return {"functions": {"summary": summary}, "plot": plot}

    def compute(self) -> Dict[str, Union[int, float]]:
        res = []
        res.append(self.aggregator.price() / 1e18)
        return dict(zip(self.cols, res))

    def prune(self) -> None:
        del self.aggregator
