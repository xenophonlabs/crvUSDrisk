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

    def __init__(self) -> None:
        self.results: List[SingleSimResults] = []

    def collect(self, single_sim_results: SingleSimResults) -> None:
        """
        Store results from single simulation.
        """
        self.results.append(single_sim_results)

    def process(self) -> MonteCarloResults:
        """
        Process single sim results into aggregate metrics.
        """
        return MonteCarloResults(self.results)
