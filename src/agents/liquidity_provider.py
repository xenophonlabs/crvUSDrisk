"""Provides the `LiquidityProvider` class."""
from .agent import Agent


# pylint: disable=too-few-public-methods
class LiquidityProvider(Agent):
    """
    The LiquidityProvider either adds or removes
    liquidity from Curve pools.
    """
