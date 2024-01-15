"""
Provides the simulation strategies for 
each scenario.
"""
from typing import Dict, Type
from .strategy import Strategy
from .baseline import BaselineStrategy
from .volatility import HighVolatilityStrategy
from .internal_crunch import InternalLiquidityCrunchStrategy

STRATEGIES: Dict[str, Type[Strategy]] = {
    "baseline_micro": BaselineStrategy,
    "baseline_macro": BaselineStrategy,
    "high_volatility": HighVolatilityStrategy,
    "internal_crunch": InternalLiquidityCrunchStrategy,
}
