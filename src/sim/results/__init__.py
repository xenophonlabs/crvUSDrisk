"""
Provides the results dataclasses to store
simulation metrics.
"""

from .single_sim_results import SingleSimResults
from .monte_carlo_results import MonteCarloResults

__all__ = ["SingleSimResults", "MonteCarloResults"]
