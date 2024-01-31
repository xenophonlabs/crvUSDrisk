"""
Provides testing utilities.
"""
from typing import List
from crvusdsim.pool.crvusd.utils import BlocktimestampMixins
from src.sim import Scenario
from src.trades import Cycle, Swap


def increment_timestamps(objs: List[BlocktimestampMixins], td: int = 60 * 60) -> None:
    """
    Increment the timestep for all the input objects.
    """
    ts = objs[0]._block_timestamp + td  # pylint: disable=protected-access
    for obj in objs:
        obj._block_timestamp = ts  # pylint: disable=protected-access


def approx(x1: int | float, x2: int | float, tol: float = 1e-3) -> bool:
    """
    Check that abs(x1 - x2)/x1 <= tol.
    """
    if x1 == 0:
        return abs(x1 - x2) <= tol
    return abs(x1 - x2) / x1 <= tol


def scale_prices(_scenario: Scenario, address: str, multiplier: float) -> None:
    """
    Modify the current price of the input token.
    """
    sample = _scenario.curr_price
    sample.prices_usd[address] *= multiplier
    sample.update(sample.prices_usd)
    _scenario.update_market_prices(sample)


def do_trade(cycle: Cycle) -> None:
    """
    Dummy trade to alter pool state.
    """
    trade = cycle.trades[0]
    assert isinstance(trade, Swap)
    pool = trade.pool  # shouldn't be external market
    high = pool.get_max_trade_size(trade.i, trade.j)
    pool.trade(trade.i, trade.j, high)
