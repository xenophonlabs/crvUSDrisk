"""
Provides the `HighVolatilityStrategy` class.
"""
from .strategy import Strategy
from ..scenario import Scenario
from ...configs import STABLE_CG_IDS, TOKEN_DTOs
from ...configs.tokens import COINGECKO_IDS_INV


# pylint: disable=too-few-public-methods
class HighVolatilityStrategy(Strategy):
    """
    Implements the scenario shocks for the high volatility
    scenarios.
    """

    def apply_shocks(self, scenario_template: Scenario) -> None:
        """
        Applies the shocks to the high volatility scenario.
        """
        for k, v in scenario_template.price_config["params"].items():
            if k not in STABLE_CG_IDS:
                token = TOKEN_DTOs[COINGECKO_IDS_INV[k]]
                v["mu"] = 0
                v["sigma"] = scenario_template.config["volatility"][token.symbol]
