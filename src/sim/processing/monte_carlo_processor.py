"""
Provides the `MonteCarloProcessor` class,
which aggregates metrics results from multiple 
simulation runs.
"""
from typing import List
from ..results import MonteCarloResults, SingleSimResults


class MonteCarloProcessor:
    """
    Stores and processes metrics data
    for many simulations.
    """

    def __init__(self, metadata: dict | None = None) -> None:
        self.metadata = metadata
        self.results: List[SingleSimResults] = []

    def collect(self, single_sim_results: SingleSimResults) -> None:
        """
        Store results from single simulation.
        """
        self.results.append(single_sim_results)

    def process(self, prune: bool = True) -> MonteCarloResults:
        """
        Process single sim results into aggregate metrics.
        """
        if prune:
            self.prune()
        return MonteCarloResults(self.results, self.metadata)

    def prune(self) -> None:
        """Prune size."""
        for run in self.results:
            for metric in run.metrics:
                metric.prune()
