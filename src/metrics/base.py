"""
Provides a base class for metrics. 

Each metric computes a lower dimension (e.g. scalar)
value of the current state of the simulation to avoid
large memory consumption from logging all entity hundreds
of thousands of times (multiple GB).

We group metrics by entity in case they have some shared
computation that we don't want to duplicate.
"""
from __future__ import annotations
from typing import List, Dict, Union, TYPE_CHECKING
from abc import ABC, abstractmethod
from functools import cached_property

if TYPE_CHECKING:
    from ..sim.scenario import Scenario


class Metric(ABC):
    """
    Base abstract class for Metrics results.
    """

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.config = self._config()

    @cached_property
    def key_metric(self) -> str:
        """Key metric name."""
        raise NotImplementedError

    def _config(self) -> Dict[str, List[str]]:
        """Aggregation config."""
        raise NotImplementedError

    @cached_property
    def cols(self) -> List[str]:
        """Column names."""
        return list(self.config.keys())

    @abstractmethod
    def compute(self, **kwargs) -> Dict[str, Union[int, float]]:
        """
        Method for computing the metrics, implemented
        by the child class.
        """
        raise NotImplementedError

    def prune(self) -> None:
        """
        Method for pruning the metric, implemented
        by the child class.
        """
        del self.scenario
