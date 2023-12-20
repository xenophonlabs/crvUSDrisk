"""
Provides the `MonteCarloResults` dataclass.
"""
from typing import Any
from dataclasses import dataclass


@dataclass
class MonteCarloResults:  # pylint: disable=too-few-public-methods
    """
    Stores metrics data aggregated over many simulations.
    """

    results: Any

    def plot(self):
        """Plot metrics."""

    def summarize(self):
        """Summarize metrics."""
