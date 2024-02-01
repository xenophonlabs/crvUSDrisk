"""
Test the `Cycle` class, which is responsible for
all the trading logic in the model.
"""
from copy import deepcopy
import random
import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as st
from crvusdsim.pool.sim_interface import SimLLAMMAPool
from src.sim import Scenario
from src.trades.cycle import _optimize_mem
from src.trades import Swap
from ...utils import approx, scale_prices, do_trade


TOLERANCE = 0.1
MAX_CYCLES = 56


def test_cycle_init(scenario: Scenario) -> None:
    """
    Test that the Cycle is, in fact, a cycle.
    """
    for cycle in scenario.cycles:
        trades = cycle.trades
        assert len(trades) == scenario.arb_cycle_length
        for i, trade in enumerate(trades):
            if i != cycle.n - 1:
                next_trade = trades[i + 1]
            else:
                next_trade = trades[0]
            token_out = trade.get_address(trade.j)
            token_in = next_trade.get_address(next_trade.i)
            assert token_in == token_out
            if isinstance(trade, Swap) and isinstance(trade.pool, SimLLAMMAPool):
                assert i in cycle.llamma_trades


def test_optimize(scenario: Scenario) -> None:
    """
    Test that the `Cycle.optimize` method returns approximately
    the same result as a brute-force search.

    This test is probabilistic in nature, since `optimize` is a
    scipy.optimize method and is not guaranteed to be exactly optimal.
    We aim for a success rate of 90%. Improving the cycle optimization
    should allow us to test for higher success rates. Running thousands
    of these tests, the success rate converges to about 96%, but we set
    it lower to prevent false negatives.
    """
    _scenario = deepcopy(scenario)

    count = 0
    failures = 0
    for _ in range(10):
        # Mix the prices up a bit
        for coin in scenario.coins:
            scale_prices(_scenario, coin.address, random.uniform(0.99, 1.01))

        cycles = _scenario.cycles
        for cycle in cycles:
            cycle.freeze_oracles()

            trade = cycle.trades[0]
            assert isinstance(trade, Swap)
            high = trade.pool.get_max_trade_size(trade.i, trade.j)
            amts = np.linspace(0, high, 100)
            amts = [int(amt) for amt in amts]

            profits = [cycle.populate(amt) for amt in amts]
            best_profit_linspace = int(max(profits))
            best_profit_optimize = cycle.optimize()[1]

            if best_profit_linspace > 0:
                count += 1
                try:
                    assert approx(best_profit_linspace, best_profit_optimize, 0.01)
                except AssertionError:
                    try:
                        assert best_profit_linspace < best_profit_optimize
                    except AssertionError:
                        failures += (
                            1  # if the opt profit is < lin profit, it's a failure
                        )

            cycle.unfreeze_oracles()

    # if this exceeds the tolerance by a little, it's probably a statistical fluke.
    # increase the number of iterations to get a more accurate result. it will just be
    # a little slower.
    assert failures / count < TOLERANCE


def test_state_key(scenario: Scenario) -> None:
    """
    Test that changing the underlying pools in a
    cycle invalidates the state key.
    """
    _scenario = deepcopy(scenario)
    cycle = _scenario.cycles[0]

    key = hash(cycle.state_key)

    # do some trading
    do_trade(cycle)

    # check that state key is invalidated
    assert key != hash(cycle.state_key)


def test_cycle_cache(scenario: Scenario) -> None:
    """
    Test that the LRU cache is being hit for the
    optimize method.
    """
    _optimize_mem.cache_clear()
    cache = _optimize_mem.cache_info()
    hits = cache.hits
    misses = cache.misses

    _scenario = deepcopy(scenario)
    cycle = _scenario.cycles[0]

    # Check single cycle caching
    cycle.optimize()
    cycle.optimize()
    cache = _optimize_mem.cache_info()
    assert cache.misses == misses + 1
    assert cache.hits == hits + 1

    hits = cache.hits
    misses = cache.misses

    # Check that performing a trade invalidates the cache
    do_trade(cycle)
    cycle.optimize()
    cache = _optimize_mem.cache_info()
    assert cache.misses == misses + 1
    assert cache.hits == hits


@given(idx=st.integers(min_value=0, max_value=MAX_CYCLES - 1))
@settings(deadline=None)
def test_populate_execute(scenario: Scenario, idx: int) -> None:
    """
    Test that the Cycle populates and executes as expected, with the
    amount out of each trade being the amount in of the
    next trade.
    """
    _scenario = deepcopy(scenario)
    cycle = _scenario.cycles[idx]

    # Mix the prices up a bit
    for coin in scenario.coins:
        scale_prices(_scenario, coin.address, random.uniform(0.99, 1.01))

    amt_in, exp_profit = cycle.optimize()
    cycle.populate(amt_in, use_snapshot_context=False)
    cycle.execute()

    trades = cycle.trades
    for i, trade in enumerate(trades):
        amt_out = trade.out
        if i != cycle.n - 1:
            next_trade = trades[i + 1]
            next_trade_amt = next_trade.amt
            if amt_out == 0 or isinstance(next_trade, Swap):
                assert amt_out == next_trade_amt
            else:  # liquidation
                # pool.get_dx() is not exact
                assert abs(amt_out - next_trade_amt) / next_trade_amt > 1e-4

    decimals = trades[0].get_decimals(trades[0].i)
    act_profit = (amt_out - amt_in) / 10**decimals
    assert approx(act_profit, exp_profit, 0.01)
