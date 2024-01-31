"""
Provides a test suite for the Arbitrageur agent.
A lot of the testing logic for arbitrages is in
test/unit/trades/
"""
from copy import deepcopy
from src.sim import Scenario
from ...utils import approx

simple_types = (int, float, str, bool, list, dict, tuple, set)


def test_statelessness(scenario: Scenario) -> None:
    """
    Make sure that finding the best arbitrage
    does not affect underlying pools.
    """
    _scenario = deepcopy(scenario)

    spools = _scenario.stableswap_pools
    spool_snapshots = {p.address: p.get_snapshot() for p in spools}

    llammas = _scenario.llammas
    llamma_snapshots = {l.address: l.get_snapshot() for l in llammas}

    _scenario.arbitrageur.find_best_arbitrage(_scenario.cycles, _scenario.curr_price)

    for spool in spools:
        old = spool_snapshots[spool.address]
        new = spool.get_snapshot()
        assert old.balances == new.balances
        assert old._block_timestamp == new._block_timestamp
        assert old.last_price == new.last_price
        assert old.ma_price == new.ma_price
        assert old.ma_last_time == new.ma_last_time

    for llamma in llammas:
        old = llamma_snapshots[llamma.address]
        new = llamma.get_snapshot()
        assert old.bands_x == new.bands_x
        assert old.bands_y == new.bands_y
        assert old._block_timestamp == new._block_timestamp
        assert old.user_shares == new.user_shares
        assert old.total_shares == new.total_shares
        assert old.bands_fees_x == new.bands_fees_x
        assert old.bands_fees_y == new.bands_fees_y
        assert old.active_band == new.active_band


def test_find_best_arbitrage(scenario: Scenario) -> None:
    """
    Test that the find_best_arbitrage function returns
    a cycle, and that when we execute it we get the
    expected profit.

    Note
    ----
    We indirectly test that the find_best_arbitrage returns the
    "optimal" arbitrage by testing the cycle.optimize function
    in test/unit/trades/test_cycle.py.
    """
    _scenario = deepcopy(scenario)
    cycle, expected_profit = _scenario.arbitrageur.find_best_arbitrage(
        _scenario.cycles, _scenario.curr_price
    )
    assert cycle is not None
    assert expected_profit != 0

    # Execute
    actual_profit = (
        cycle.execute() * _scenario.curr_price.prices_usd[cycle.basis_address]
    )
    assert approx(actual_profit, expected_profit)


def test_arbitrage(scenario: Scenario) -> None:
    """
    Test that raising the price of a token
    creates an arbitrage that the arbitrageur
    takes advantage of, pushing prices in the right direction.
    """
    _scenario = deepcopy(scenario)
    controller = _scenario.controllers[0]
    llamma = controller.AMM
    arbitrageur = _scenario.arbitrageur

    prev_collat_price = llamma.get_p()

    # Raise the collateral price
    collateral = controller.COLLATERAL_TOKEN.address
    sample = _scenario.curr_price
    sample.prices_usd[collateral] *= 2
    sample.update(sample.prices_usd)
    _scenario.update_market_prices(sample)

    # Check that the arbitrages were executed
    arbitrageur.arbitrage(_scenario.cycles, _scenario.curr_price)
    assert arbitrageur.profit() > 0
    assert arbitrageur.count() > 0

    # Check that the price of the collateral token has increased
    new_collat_price = llamma.get_p()
    assert new_collat_price > prev_collat_price
