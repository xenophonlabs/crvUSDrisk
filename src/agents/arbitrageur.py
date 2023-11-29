import logging
from .agent import Agent

PRECISION = 1e18


class Arbitrageur(Agent):
    """
    Arbitrageur performs cyclic arbitrages between the following
    Curve pools:
        - StableSwap pools
        - TriCrypto-ng pools TODO
        - LLAMMAs.
    TODO need to investigate which pools to include in arbitrage
    search (e.g. TriCRV, other crvUSD pools, etc..). Otherwise, we
    are artificially constraining the available crvUSD liquidity.
    """

    def __init__(self, tolerance: float = 0):
        assert tolerance >= 0

        self.tolerance = tolerance
        self._profit = 0
        self._count = 0

    def arbitrage(self, cycles):
        """
        Identify optimal arbitrages involving crvUSD of the form:

            crvUSD pool -> crvUSD pool -> External Market

        LLAMMA Example: ETH/crvUSD -> USDC/crvUSD -> USDC/ETH
        StableSwap Example: USDC/crvUSD -> USDT/crvUSD -> USDT/USDC

        Parameters
        ----------
        cycles : List[List[Pool]
            List of cycles, where each cycle is a list of pools. Pools
            can be ExternalMarket, SimCurveStableSwapPool, or SimLLAMMAPool.

        Returns
        -------
        profit : float
            Total profit from arbitrage.
        count : int
            Number of arbitrages executed.

        Note
        ----
        TODO need to handle LLAMMAs
        TODO need to handle longer cycles
        TODO need to handle different cycle formats?
        TODO need to track pool-specific metrics
        """
        profit = 0
        count = 0

        while True:
            # TODO this returns the profit in the units of
            # the basis token (first token in, last token out).
            # We should (1) require that the basis token be USDC, USDT, WETH,
            # and (2) mark this profit to current USD?
            best = self.find_best_arbitrage(cycles)

            if best and best.expected_profit > self.tolerance:
                _profit = best.execute()
                assert profit > self.tolerance, RuntimeError("Trade unprofitable.")
                profit += _profit
                count += 1
            else:
                logging.info("No more profitable arbitrages.")
                break

        self._profit += profit
        self._count += count

        return profit, count

    def find_best_arbitrage(self, cycles):
        """Find the optimal liquidity-constrained cyclic arbitrages."""
        best_trade = None
        best_profit = 0

        for cycle in cycles:
            # Populate cycle with the optimal amt_in, and calc expected profit
            cycle.optimize()
            if cycle.expected_profit > best_profit:
                best_profit = cycle.expected_profit
                best_trade = cycle.trade

        return best_trade
