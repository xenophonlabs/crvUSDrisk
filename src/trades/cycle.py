"""
Provides the `Cycle` class for optimizing and
exeucuting a sequence of trades.
"""
from typing import Sequence, cast
from functools import lru_cache
from scipy.optimize import minimize_scalar
from crvusdsim.pool import SimLLAMMAPool, SimCurveStableSwapPool
from curvesim.pool import SimCurvePool
from .trade import Swap, Liquidation
from ..logging import get_logger
from ..modules import ExternalMarket

TOLERANCE = 1e-6

logger = get_logger(__name__)


class Cycle:
    """
    The `Cycle` class represents a sequence of trades that
    satisfy the following conditions:
    1. The output token of each trade is the input token of
    the next trade, including the first/last trades.
    2. The output amount from each trade is equal to the input
    amount of the next trade, except for the first/last trades.

    A `Cycle` can include both `Swap` and `Liquidation` trades.
    """

    def __init__(
        self,
        trades: Sequence[Swap | Liquidation],
        expected_profit: float | None = None,
    ):
        self.trades = trades
        self.n: int = len(trades)
        self.expected_profit = expected_profit
        self.basis_address: str = trades[0].get_address(trades[0].i)
        self.state_key: StateKey = StateKey(self)  # for memoization

        # check that this is a cycle
        for i, trade in enumerate(trades):
            if i != self.n - 1:
                next_trade = trades[i + 1]
            else:
                next_trade = trades[0]
            token_out = trade.get_address(trade.j)
            token_in = next_trade.get_address(next_trade.i)
            assert token_in == token_out, "Trades do not form a cycle."

    def execute(self) -> float:
        """Execute trades."""
        logger.info("Executing cycle %s.", self)

        trade = self.trades[0]
        amt_in = trade.amt

        for i, trade in enumerate(self.trades):
            logger.info("Executing trade %s.", trade)
            if i != self.n - 1:
                amt_out, decimals = trade.do()
                assert (
                    abs(amt_out - self.trades[i + 1].amt) / 10**decimals < TOLERANCE
                ), f"Trade {i+1} output {amt_out} != Trade {i+2} input {self.trades[i+1].amt}."
            else:
                amt_out, decimals = trade.do()

        profit = (amt_out - amt_in) / 10**decimals
        if abs(profit - self.expected_profit) > TOLERANCE:
            logger.warning(
                "Expected profit %f != actual profit %f.", self.expected_profit, profit
            )

        return profit

    def optimize(self):
        """
        Optimize the `amt_in` for the first trade in the cycle.
        """
        _optimize_mem(self.state_key)

    def populate(self, amt_in: float | int) -> float:
        """
        Populate the amt_in to all trades in cycle, and the expected_profit.

        Parameters
        ----------
        amt_in : float | int
            The amount of the first trade in the cycle.

        Returns
        -------
        expected_profit : float
            The expected profit of the cycle.
        """
        amt_in = int(amt_in)  # cast float to int
        # TODO add a unit test to ensure snapshot context is used correctly

        trade = self.trades[0]
        trade.amt = amt_in

        for i, trade in enumerate(self.trades):
            amt, decimals = trade.do(use_snapshot_context=True)
            if i != self.n - 1:
                self.trades[i + 1].amt = amt

        expected_profit = float((amt - amt_in) / 10**decimals)
        self.expected_profit = expected_profit
        return expected_profit

    def __repr__(self) -> str:
        return f"Cycle(Trades: {self.trades}, Expected Profit: {self.expected_profit})"


# pylint: disable=too-few-public-methods
class StateKey:
    """
    Get the hash key for LRU cache.

    Note
    ----
    We don't want to get `amt` for trade since this
    is the quantity that gets populated by optimize.
    Similarly, we don't want to use the cycle's
    `expected_profit`.

    It is crucial that whatever goes into the `state_key`
    provides a complete picture of the pool we are trading on.
    If any component of the pool which affects trade execution
    has changed, this MUST invalidate the cache!

    TODO we could implement __hash__ on the `Snapshot` class,
    in which case we could just hash the snapshot.
    """

    def __init__(self, cycle: Cycle):
        self.cycle = cycle

    def __hash__(self):
        trades = self.cycle.trades
        hashes = []
        for trade in trades:
            if isinstance(trade, Swap):
                if isinstance(trade.pool, ExternalMarket):
                    # we implement a custom hash for this
                    # the only thing that matters is price.
                    pool_hash = hash(trade.pool)
                elif isinstance(trade.pool, SimLLAMMAPool):
                    pool_hash = hash(
                        (
                            trade.pool.address,
                            trade.pool.active_band,
                            trade.pool.old_p_o,  # TODO is this updated?
                            tuple(trade.pool.bands_x.items()),
                            tuple(trade.pool.bands_y.items()),
                        )
                    )
                elif isinstance(trade.pool, (SimCurveStableSwapPool, SimCurvePool)):
                    pool_hash = hash(
                        (
                            trade.pool.address,
                            tuple(trade.pool.balances),
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"Pool type {type(trade.pool)} not supported."
                    )
                hashes.append(pool_hash)
            elif isinstance(trade, Liquidation):
                # controller_hash = hash((
                #     trade.controller.AMM.address,
                #     trade,
                # ))
                # hashes.append(controller_hash)
                raise NotImplementedError(
                    "Liquidation not supported in optimization."
                )  # TODO
        return hash(tuple(hashes))


@lru_cache(maxsize=1000)  # TODO might need to be larger when considering many pools
def _optimize_mem(state_key: StateKey) -> None:
    """
    Memoized optimization for the `amt_in` for the
    first trade in the cycle.

    NOTE all trades in `Cycle` MUST BE of type `Swap`.
    """
    cycle = state_key.cycle
    trades = cast(Sequence[Swap], cycle.trades)  # for mypy
    trade = trades[0]
    high = float(trade.pool.get_max_trade_size(trade.i, trade.j))

    res = minimize_scalar(
        lambda x: -cycle.populate(x),
        args=(),
        bounds=(0, high),
        method="bounded",
    )

    if res.success:
        cycle.populate(res.x)
    else:
        raise RuntimeError(res.message)
