"""Provides the `Arbitrageur` class."""
from typing import List, Tuple, cast
import math
import multiprocessing as mp
from multiprocessing.queues import Queue
from .agent import Agent
from ..trades import Cycle, Swap
from ..trades.cycle import _optimize_mem  # TODO remove, using for testing
from ..prices import PriceSample
from ..configs import DEFAULT_PROFIT_TOLERANCE
from ..logging import (
    get_logger,
    multiprocessing_logging_queue,
    configure_multiprocess_logging,
)

PRECISION = 1e18

logger = get_logger(__name__)

mp.set_start_method("fork", force=True)  # Avoid lock overhead on shared pool objs


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

    def __init__(self, tolerance: float = DEFAULT_PROFIT_TOLERANCE):
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
                logger.info("Executing arbitrage: %s.", best_cycle)
                # Dollarize profit
                _profit = (
                    best_cycle.execute() * prices.prices_usd[best_cycle.basis_address]
                )
                assert abs(_profit - best_profit) < 1, RuntimeError(
                    "Expected profit != actual profit."
                )
                profit += best_profit
                count += 1
            else:
                logger.info("No more profitable arbitrages.")
                break

        # logger.info("Cache info: %s", _optimize_mem.cache_info())

        self._profit += profit
        self._count += count

        return profit, count

    # pylint: disable=too-many-locals
    def find_best_arbitrage(
        self, cycles: List[Cycle], prices: PriceSample, multiprocess: bool = True
    ) -> Tuple[Cycle | None, float]:
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
        multiprocess : bool, default=True
            Whether to use multiprocessing to find the optimal cycle.
            Default is True.

        Returns
        -------
        best_cycle : Cycle
            The optimal cycle.
        best_profit : float
            The dollarized profit of the optimal cycle.
        """
        # TODO test that the expected optimal cycle is
        # the same as the actual optimal cycle following
        # starmap. Test with and without multiprocessing
        # for results.
        # TODO what if I make cycle.optimize() not mutate
        # the cycle at all? Everything is done in the snapshot
        # context
        cpu_count = mp.cpu_count()
        if multiprocess and cpu_count > 1:
            with multiprocessing_logging_queue() as logging_queue:
                args = [(cycle, prices, logging_queue) for cycle in cycles]
                with mp.Pool(cpu_count) as pool:
                    results = pool.starmap(optimize_cycle, args)
                best = max(results, key=lambda x: x[1])
                best_index = results.index(best)
                best_cycle = cycles[best_index]
                best_amt, best_profit = best
        else:
            best_profit = -math.inf
            best_amt = 0
            best_cycle = None
            for cycle in cycles:
                # Don't need to re-optimize
                amt, profit = optimize_cycle(cycle, prices)
                if profit > best_profit:
                    best_cycle = cycle
                    best_profit = profit
                    best_amt = amt

        if best_cycle:
            best_cycle.populate(best_amt, use_snapshot_context=False)

        return best_cycle, best_profit


# TODO cache should be here?
def optimize_cycle(
    cycle: Cycle, prices: PriceSample, logging_queue: Queue | None = None
) -> Tuple[int, float]:
    """Wrapper to optimize cycle."""
    if logging_queue:
        configure_multiprocess_logging(logging_queue)

    # Get 1 dollar's worth (marked to market)
    trade = cast(Swap, cycle.trades[0])
    decimals = trade.get_decimals(trade.i)
    xatol = int(10**decimals / prices.prices_usd[cycle.basis_address])

    amt, profit = cycle.optimize(xatol=xatol)

    # Dollarize the expected profit
    profit *= prices.prices_usd[cycle.basis_address]

    return amt, profit
