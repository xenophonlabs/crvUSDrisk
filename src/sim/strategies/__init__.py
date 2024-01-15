"""
Provides the simulation strategies for 
each scenario.
"""
from .strategy import Strategy
from .baseline import BaselineStrategy
from .volatility import HighVolatilityStrategy

STRATEGIES = {
    "baseline_micro": BaselineStrategy,
    "baseline_macro": BaselineStrategy,
    "high_volatility": HighVolatilityStrategy,
}
