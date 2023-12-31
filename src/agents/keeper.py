"""Provides the `Keeper` class."""

from typing import List, Tuple
from crvusdsim.pool import PegKeeper
from .agent import Agent
from ..logging import get_logger

PRECISION = 1e18

logger = get_logger(__name__)


def get_pk_symbols(pk: PegKeeper) -> str:
    """Get symbols for a PegKeeper."""
    return pk.POOL.name.replace("Curve.fi Factory Plain Pool: ", "")


# pylint: disable=too-few-public-methods
class Keeper(Agent):
    """
    Keeper calls the `update` method on `PegKeeper`s.
    """

    address: str = "DEFAULT_KEEPER"

    def __init__(self, tolerance: float = 0):
        """
        Note
        ----
        The `tolerance` must be in LP share units
        (decimals = 18).
        """
        assert tolerance >= 0

        self.tolerance = tolerance
        self._profit = 0.0
        self._count = 0

    def update(self, pks: List[PegKeeper]) -> Tuple[float, int]:
        """
        Checks if any PegKeeper is profitable to update. Updates
        if profitable.

        Parameters
        ----------
        pks : List[PegKeeper]
            List of PegKeeper objects to check.

        Returns
        -------
        profit : float
            Profit from updating.
        count : int
            Number of Peg Keepers updated.
        """
        profit = 0.0
        count = 0
        for pk in pks:
            estimate = pk.estimate_caller_profit()
            if estimate > self.tolerance:
                _profit = pk.update(self.address) / 1e18
                assert _profit > 0, "Update not profitable."
                profit += _profit * pk.POOL.get_virtual_price() / 1e18
                count += 1
                logger.debug(
                    "Updating %s Peg Keeper with profit %d.",
                    get_pk_symbols(pk),
                    round(profit / PRECISION),
                )
            else:
                logger.debug(
                    "Not updating %s Peg Keeper.",
                    get_pk_symbols(pk),
                )
        self._profit += profit
        self._count += count
        return profit, count
