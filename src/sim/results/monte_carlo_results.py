"""
Provides the `MonteCarloResults` dataclass.
"""
from __future__ import annotations
from typing import List, TYPE_CHECKING
from dataclasses import dataclass
import pandas as pd

if TYPE_CHECKING:
    from .single_sim_results import SingleSimResults


@dataclass
class MonteCarloResults:  # pylint: disable=too-few-public-methods
    """
    Stores metrics data aggregated over many simulations.
    """

    data: List[SingleSimResults]

    def plot(self):
        """Plot metrics."""

    def summarize(self):
        """Summarize metrics."""
        summary = pd.concat([x.summarize() for x in self.data])
        summary.index = range(len(summary))
        return summary
