"""
Provides the simulation strategies for 
each scenario.
"""

from .baseline import BaselineStrategy
from .strategy import Strategy

STRATEGIES = {
    "baseline_micro": BaselineStrategy,
    "baseline_macro": BaselineStrategy,
}
