"""
Provides the `Cycle` class for optimizing and
exeucuting a sequence of trades.
"""
import math
from typing import Sequence, cast, Dict, Any, Tuple
from functools import lru_cache
from scipy.optimize import minimize_scalar
from crvusdsim.pool import SimLLAMMAPool, SimCurveStableSwapPool
from curvesim.pool import SimCurvePool
from .trade import Swap, Liquidation
from ..logging import get_logger
from ..modules import ExternalMarket

TOLERANCE = 1e-6  # TODO should make this smaller

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
        amt = amt_in

        for i, trade in enumerate(self.trades):
            logger.info("Executing trade %s.", trade)
            amt_out, decimals = trade.execute(amt, use_snapshot_context=False)
            if i != self.n - 1:
                amt = amt_out
                # TODO move this to test file
                next_trade = self.trades[i + 1]
                next_trade_amt = next_trade.amt
                # Check that cycle was populated correctly
                if amt_out == 0 or isinstance(next_trade, Swap):
                    cond = amt_out == next_trade_amt
                else:  # liquidation
                    cond = (
                        abs(amt_out - next_trade_amt) / next_trade_amt < TOLERANCE
                    )  # pool.get_dx() is not exact
                assert (
                    cond
                ), f"Trade {i + 1} output {amt_out} != Trade {i + 2,} input {next_trade_amt}."

        profit = (amt_out - amt_in) / 10**decimals
        if abs(profit - self.expected_profit) > TOLERANCE:
            logger.warning(
                "Expected profit %f != actual profit %f.", self.expected_profit, profit
            )

        return profit

    def optimize(self, xatol: int | None = None) -> Tuple[int, float]:
        """
        Optimize the `amt_in` for the first trade in the cycle.
        """
        return _optimize_mem(self.state_key, xatol=xatol)

    def populate(self, amt_in: float | int, use_snapshot_context: bool = True) -> float:
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
        if amt_in < 0:
            return -math.inf

        amt_in = int(amt_in)  # cast float to int
        amt = amt_in
        trade = self.trades[0]

        if not use_snapshot_context:
            trade.amt = amt_in

        for i, trade in enumerate(self.trades):
            amt_out, decimals = trade.execute(amt, use_snapshot_context=True)
            if i != self.n - 1:
                amt = amt_out
                if not use_snapshot_context:
                    self.trades[i + 1].amt = amt

        expected_profit = float((amt_out - amt_in) / 10**decimals)

        if not use_snapshot_context:
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
                pool = trade.pool
                if isinstance(pool, ExternalMarket):
                    coins_hash = hash(tuple(hash(coin) for coin in pool.coins))
                    if not pool.prices:
                        prices_hash = hash(pool.prices)
                    else:
                        # convert nested dict into hashable tuple
                        prices_hash = hash(
                            tuple(
                                sorted(
                                    {
                                        key: tuple(sorted(pj.items()))
                                        for key, pj in pool.prices.items()
                                    }.items()
                                )
                            )
                        )
                    pool_hash = hash((coins_hash, pool.n, prices_hash))
                elif isinstance(pool, SimLLAMMAPool):
                    pool_hash = hash(
                        (
                            pool.address,
                            pool.active_band,
                            pool.old_p_o,  # TODO is this updated?
                            tuple(pool.bands_x.items()),
                            tuple(pool.bands_y.items()),
                        )
                    )
                elif isinstance(pool, (SimCurveStableSwapPool, SimCurvePool)):
                    pool_hash = hash(
                        (
                            pool.address,
                            tuple(pool.balances),
                        )
                    )
                else:
                    raise NotImplementedError(f"Pool type {type(pool)} not supported.")
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


@lru_cache(maxsize=100)  # TODO might need to increase for more cycles
def _optimize_mem(state_key: StateKey, xatol: int | None = None) -> Tuple[int, float]:
    """
    Memoized optimization for the `amt_in` for the
    first trade in the cycle.

    We set the `xatol` parameter for the minimization. This
    provides a massive speed up. We set the absolute tolerance
    to $1 in the token being traded based on current market price.
    We use the `bounded` method because it allows us to specify
    absolute rather than relative tolerances. Furthermore, we
    have found `bounded` to perform much faster than `brent`
    for our use case.

    Note
    ----
    All trades in `Cycle` MUST BE of type `Swap`.
    """
    cycle = state_key.cycle
    trades = cast(Sequence[Swap], cycle.trades)
    trade = trades[0]
    low = 0
    high = float(trade.pool.get_max_trade_size(trade.i, trade.j))

    if high == 0:
        logger.info("No liquidity for %s.", str(cycle))
        return 0, 0.0

    kwargs: Dict[str, Any] = {
        "args": (),
        "bounds": (low, high),
        "method": "bounded",
    }

    if xatol:
        kwargs["options"] = {"xatol": xatol}

    res = minimize_scalar(lambda x: -cycle.populate(x), **kwargs)

    if res.success:
        return int(res.x), -res.fun
    raise RuntimeError(res.message)
