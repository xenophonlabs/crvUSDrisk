"""
Provides the `InternalLiquidityCrunch` class.
"""
from .strategy import Strategy
from ..scenario import Scenario


# pylint: disable=too-few-public-methods
class InternalLiquidityCrunchStrategy(Strategy):
    """
    Implements the scenario shocks for the crvUSD Liquidity
    Crunch scenarios.
    """

    def apply_shocks(self, scenario_template: Scenario) -> None:
        """
        Applies the shocks to the high volatility scenario.
        """
        scenario_template.target_liquidity_ratio = scenario_template.liquidity_config[
            "stressed_ratio"
        ]
