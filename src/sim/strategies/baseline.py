"""
Provides the `BaselineStrategy` class.
"""
from .strategy import Strategy
from ..scenario import Scenario
from ...configs import STABLE_CG_IDS


# pylint: disable=too-few-public-methods
class BaselineStrategy(Strategy):
    """
    Implements the scenario shocks for the baseline
    scenarios.
    """

    def apply_shocks(self, scenario_template: Scenario) -> None:
        """
        Applies the shocks to the baseline scenario.
        """
        # Remove drift from collateral assets to enforce random walk
        for k, v in scenario_template.price_config["params"].items():
            if k not in STABLE_CG_IDS:
                v["mu"] = 0
