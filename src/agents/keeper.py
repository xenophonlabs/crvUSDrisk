from ..modules.pegkeeper import PegKeeper
from typing import List
import logging
from .agent import Agent

PRECISION = 1e18


class Keeper(Agent):
    """
    Keeper calls the `update` method on Peg Keepers.
    """

    def __init__(self, tolerance: float = 0):
        assert tolerance >= 0

        self.tolerance = tolerance
        self._profit = 0
        self._count = 0

    def update(self, pks: List[PegKeeper], ts: int) -> tuple(float, int):
        """
        Checks if any PegKeeper is profitable to update. Updates
        if profitable.

        Parameters
        ----------
        pks : List[PegKeeper]
            List of PegKeeper objects to check.
        ts : int
            Timestamp of update.

        Returns
        -------
        profit : float
            Profit from updating.
        count : int
            Number of Peg Keepers updated.
        """
        profit = 0
        count = 0
        for pk in pks:
            if pk.estimate_caller_profit(ts) > self.tolerance:
                _profit = pk.update(ts)
                assert _profit > 0, "Update not profitable."
                profit += _profit
                count += 1
                logging.info(f"Updating {pk.name} Peg Keeper with pnl {round(profit)}.")
        self._profit += profit
        self._count += count
        return profit, count
