from ..modules.pegkeeper import PegKeeper
from ..modules.llamma import LLAMMA
from ..utils import external_swap

from curvesim.pool import CurvePool
from typing import List


class Arbitrageur:
    """
    Arbitrageur either calls update() on the Peg Keeper
    or arbitrages (swaps) StableSwap pools.
    TODO ensure arbitrage pnl is in USD (NOT crvUSD) units!
    """

    __slots__ = (
        "tolerance",  # min profit required to act
        "update_pnl",
        "update_count",
        "arbitrage_pnl",
        "arbitrage_count",
        "verbose",  # print statements
    )

    def __init__(
        self,
        tolerance: float,
        verbose: bool = False,
    ) -> None:
        assert tolerance >= 0

        self.tolerance = tolerance
        self.update_pnl = 0
        self.update_count = 0
        self.arbitrage_pnl = 0
        self.arbitrage_count = 0
        self.verbose = verbose

    def update(self, pks: List[PegKeeper], ts: int):
        """
        Checks if any PegKeeper is profitable to update.

        Parameters
        ----------
        pks : List[PegKeeper]
            List of PegKeeper objects to check.
        ts : int
            Timestamp of update.
        
        Returns
        -------

        """
        for pk in pks:
            if pk.estimate_caller_profit(ts) > self.tolerance:
                pnl = pk.update(ts)
                self.print(f"Updating {pk.name} Peg Keeper with pnl {round(pnl)}.")
                self.update_pnl += pnl
                self.update_count += 1
    
    def arbitrage(self, pools: List[CurvePool], ts: int):
        """
        Checks if any StableSwap pool is profitable to arbitrage.

        Note
        ----
        It is not trivial to derive optimal arb analytically since
        solving the StableSwap invariant requires numerical approaches
        like Newton's method. Therefore, we perform a simple linear search.
        TODO Ultimately would be cool to implement this using the 
        existing Arbitrageur pipeline from curvesim, perhaps we could 
        implement a `LiquidityLimitedArbitrage` class!
        """
        # I can check trades here with 
        # pool.use_snapshot_context()!

    def print(self, txt):
        if self.verbose:
            print(txt)
