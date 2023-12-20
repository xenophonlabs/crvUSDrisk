"""
Provides a base class for metrics. 

Each metric computes a lower dimension (e.g. scalar)
value of the current state of the simulation to avoid
large memory consumption from logging all entity hundreds
of thousands of times (multiple GB).

We group metrics by entity in case they have some shared
computation that we don't want to duplicate.
"""

from typing import List, Dict, Union
from abc import ABC, abstractmethod
from functools import cached_property


class Metric(ABC):
    """
    Base abstract class for Metrics results.
    """

    @cached_property
    @abstractmethod
    def config(self) -> dict:
        """
        Config defines the functions, names,
        aggregation, and plotting of the metric.
        """
        raise NotImplementedError

    @cached_property
    def cols(self) -> List[str]:
        """
        Names for the metrics provided by the Metric class.
        """
        return list(self.config["functions"]["summary"].keys())

    @abstractmethod
    def compute(self) -> Dict[str, Union[int, float]]:
        """
        Method for computing the metrics, implemented
        by the child class.
        """
        raise NotImplementedError
