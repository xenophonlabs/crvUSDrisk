"""
Provides metrics on the Aggregator.
"""

from typing import TYPE_CHECKING, Union, Dict
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
        summary: Dict[str, str] = {}
        aggregator_str = entity_str(self.aggregator, "aggregator")
        summary[aggregator_str + "_price"] = "mean"
        # TODO plot config
        return {"functions": {"summary": summary}}

    def compute(self) -> Dict[str, Union[int, float]]:
        res = []
        res.append(self.aggregator.price())
        return dict(zip(self.cols, res))
