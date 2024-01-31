"""
Provides testing suite for Swap trades.
"""
from copy import deepcopy
from hypothesis import given
import hypothesis.strategies as st
from src.sim.scenario import Scenario
from src.trades import Swap
from ...utils import approx

MAX_SPOOLS = 4
MAX_LLAMMAS = 4
MAX_MARKETS = 28


@given(
    pool_idx=st.integers(min_value=0, max_value=MAX_SPOOLS - 1),
    i=st.integers(min_value=0, max_value=1),
)
def test_spools(scenario: Scenario, pool_idx: int, i: int) -> None:
    """
    Test that trades on stableswap pools are stateless and
    will not alter the pool. This is key to optimizing cycles.

    Also test that the expected amount of tokens went into and
    came out of the pool.
    """
    _scenario = deepcopy(scenario)

    spool = _scenario.stableswap_pools[pool_idx]
    old = spool.get_snapshot()

    # Execute a large trade
    j = i ^ 1
    amt_in = spool.get_max_trade_size(i, j)
    swap = Swap(spool, i, j, amt_in)
    amt_out_stateless, _ = swap.execute(amt_in, use_snapshot_context=True)

    # Assert that the pool has not changed
    new = spool.get_snapshot()
    assert old.balances == new.balances
    assert (
        old._block_timestamp  # pylint: disable=protected-access
        == new._block_timestamp  # pylint: disable=protected-access
    )
    assert old.last_price == new.last_price
    assert old.ma_price == new.ma_price
    assert old.ma_last_time == new.ma_last_time

    # Now check that the function works the same stateful
    amt_out_statefull, _ = swap.execute(amt_in, use_snapshot_context=False)
    assert amt_out_stateless == amt_out_statefull

    # Assert pool changed as expected
    new = spool.get_snapshot()
    assert approx(amt_in, new.balances[i] - old.balances[i])
    assert approx(amt_out_statefull, old.balances[j] - new.balances[j])


@given(
    pool_idx=st.integers(min_value=0, max_value=MAX_LLAMMAS - 1),
    i=st.integers(min_value=0, max_value=1),
)
def test_llammas(scenario: Scenario, pool_idx: int, i: int) -> None:
    """
    Test that trades on LLAMMAs are stateless and
    will not alter the pool. This is key to optimizing cycles.

    Also test that the expected amount of tokens went into and
    came out of the pool.
    """
    _scenario = deepcopy(scenario)

    llamma = _scenario.llammas[pool_idx]
    old = llamma.get_snapshot()

    # Execute a large trade
    j = i ^ 1
    amt_in = llamma.get_max_trade_size(i, j)
    swap = Swap(llamma, i, j, amt_in)
    amt_out_stateless, _ = swap.execute(amt_in, use_snapshot_context=True)

    # Assert that the pool has not changed
    new = llamma.get_snapshot()
    assert old.bands_x == new.bands_x
    assert old.bands_y == new.bands_y
    assert (
        old._block_timestamp  # pylint: disable=protected-access
        == new._block_timestamp  # pylint: disable=protected-access
    )
    assert old.user_shares == new.user_shares
    assert old.total_shares == new.total_shares
    assert old.bands_fees_x == new.bands_fees_x
    assert old.bands_fees_y == new.bands_fees_y
    assert old.active_band == new.active_band

    # Now check that the function works the same stateful
    amt_out_statefull, _ = swap.execute(amt_in, use_snapshot_context=False)
    assert amt_out_stateless == amt_out_statefull

    # Assert pool changed as expected
    new = llamma.get_snapshot()
    old_bals = [
        sum(old.bands_x.values()),
        sum(old.bands_y.values()) / llamma.COLLATERAL_PRECISION,
    ]
    new_bals = [
        sum(new.bands_x.values()),
        sum(new.bands_y.values()) / llamma.COLLATERAL_PRECISION,
    ]
    assert approx(amt_in, new_bals[i] - old_bals[i])
    assert approx(amt_out_statefull, old_bals[j] - new_bals[j])


@given(
    pool_idx=st.integers(min_value=0, max_value=MAX_MARKETS - 1),
    i=st.integers(min_value=0, max_value=1),
)
def test_markets(scenario: Scenario, pool_idx: int, i: int) -> None:
    """
    Test that trades on External Markets are stateless and
    will not alter the pool. This is key to optimizing cycles.
    """
    print(len(scenario.markets))
    _scenario = deepcopy(scenario)

    market = [*_scenario.markets.values()][pool_idx]
    old = market.__dict__.copy()

    # Execute a large trade
    j = i ^ 1
    amt_in = market.get_max_trade_size(i, j)
    swap = Swap(market, i, j, amt_in)
    amt_out_stateless, _ = swap.execute(amt_in, use_snapshot_context=True)

    # Assert that the pool has not changed
    new = market.__dict__.copy()
    assert old == new

    # Now check that the function works the same stateful
    amt_out_statefull, _ = swap.execute(amt_in, use_snapshot_context=False)
    assert amt_out_stateless == amt_out_statefull
