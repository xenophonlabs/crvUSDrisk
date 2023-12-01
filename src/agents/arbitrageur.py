"""Provides the `Arbitrageur` class."""
import logging
from typing import List, Tuple, Optional
from .agent import Agent
from ..trades import Cycle
from ..prices import PriceSample

PRECISION = 1e18


class Arbitrageur(Agent):
    """
    Arbitrageur performs cyclic arbitrages between the following
    Curve pools:
        - StableSwap pools
        - TriCrypto-ng pools TODO
        - LLAMMAs.
    TODO need to investigate which pools to include in arbitrage
    search (e.g. TriCRV, other crvusd pools, etc..). Otherwise, we
    are artificially constraining the available crvusd liquidity.
    """

    def __init__(self, tolerance: float = 1):
        # tolerance in units of USD
        assert tolerance > 0  # default is one dollah

        self.tolerance: float = tolerance
        self._profit: float = 0
        self._count: int = 0

    def arbitrage(self, cycles: List[Cycle], prices: PriceSample) -> Tuple[float, int]:
        """
        Identify optimal arbitrages involving crvusd of the form:

            crvusd pool -> crvusd pool -> External Market

        LLAMMA Example: WETH/crvusd -> USDC/crvusd -> USDC/WETH
        StableSwap Example: USDC/crvusd -> USDT/crvusd -> USDT/USDC

        Parameters
        ----------
        cycles : List[Cycle]
            List of cycles. Cycles are an ordered list of `Trade`s.
        prices : PriceSample
            Current USD market prices for each coin.

        Returns
        -------
        profit : float
            Total profit from arbitrage.
        count : int
            Number of arbitrages executed.

        Note
        ----
        TODO need to handle longer cycles
        TODO need to handle different cycle formats?
        TODO need to track pool-specific metrics
        """
        profit = 0.0
        count = 0

        while True:
            # TODO should we require that the basis token be USDC, USDT, WETH?
            best_cycle, best_profit = self.find_best_arbitrage(cycles, prices)

            if best_cycle and best_profit > self.tolerance:
                # Dollarize profit
                _profit = (
                    best_cycle.execute() * prices.prices_usd[best_cycle.basis_address]
                )
                assert _profit == best_profit, RuntimeError(
                    "Expected profit != actual profit."
                )
                profit += best_profit
                count += 1
            else:
                logging.info("No more profitable arbitrages.")
                break

        self._profit += profit
        self._count += count

        return profit, count

    def find_best_arbitrage(
        self, cycles: List[Cycle], prices: PriceSample
    ) -> Tuple[Optional[Cycle], float]:
        """
        Find the optimal liquidity-constrained cyclic arbitrages.
        Dollarize the profit by marking it to current USD market price.
        TODO does this dollarization make sense?

        Parameters
        ----------
        cycles : List[Cycle]
            List of cycles. Cycles are an ordered list of `Trade`s.
        prices : PriceSample
            Current USD market prices for each coin.

        Returns
        -------
        best_cycle : Cycle
            The optimal cycle.
        best_profit : float
            The dollarized profit of the optimal cycle.
        """
        best_cycle = None
        best_profit = 0.0

        for cycle in cycles:
            cycle.optimize()
            assert cycle.expected_profit
            # Dollarize the expected profit
            expected_profit = (
                cycle.expected_profit * prices.prices_usd[cycle.basis_address]
            )
            if expected_profit > best_profit:
                best_profit = expected_profit
                best_cycle = cycle

        return best_cycle, best_profit
