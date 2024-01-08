"""
Provides the simulation strategies for 
each scenario.
"""

from .baseline import simulate as simulate_baseline

SCENARIO_MAP = {
    "baseline_micro": simulate_baseline,
    "baseline_macro": simulate_baseline,
}
