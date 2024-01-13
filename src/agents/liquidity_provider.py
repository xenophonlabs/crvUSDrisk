"""Provides the `LiquidityProvider` class."""
import secrets
import numpy as np
from crvusdsim.pool.sim_interface import SimCurveStableSwapPool
from .agent import Agent


# pylint: disable=too-few-public-methods
class LiquidityProvider(Agent):
    """
    The LiquidityProvider either adds or removes
    liquidity from Curve pools.
    """

    def __init__(self) -> None:
        super().__init__()
        self.address = "0x" + secrets.token_hex(20)

    def add_liquidity(
        self,
        pool: SimCurveStableSwapPool,
        mean: np.ndarray,
        cov: np.ndarray,
        scale_factor: int,
    ) -> None:
        """
        Resample liquidity amounts from a multivariate normal distribution,
        and add it to the given pool.

        TODO what sanity checks here?
        """
        _amounts = np.random.multivariate_normal(mean * scale_factor, cov, 1)
        amounts = [int(b * 1e36 / r) for b, r in zip(_amounts, pool.rates)]
        pool.add_liquidity(amounts, _receiver=self.address)
