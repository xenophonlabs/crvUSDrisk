"""
Provides the simulation strategies for 
each scenario.
"""

from .baseline import simulate as simulate_baseline

SCENARIO_MAP = {"baseline": simulate_baseline}
