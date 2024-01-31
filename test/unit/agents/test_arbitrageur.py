"""
Provides a test suite for the Arbitrageur agent.
A lot of the testing logic for arbitrages is in
test/unit/trades/
"""
from copy import deepcopy
from src.sim import Scenario
from ...utils import approx, scale_prices

simple_types = (int, float, str, bool, list, dict, tuple, set)


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

    # Double the collateral price
    scale_prices(_scenario, controller.COLLATERAL_TOKEN.address, 2.0)

    # Check that the arbitrages were executed
    arbitrageur.arbitrage(_scenario.cycles, _scenario.curr_price)
    assert arbitrageur.profit() > 0
    assert arbitrageur.count() > 0

    # Check that the price of the collateral token has increased
    new_collat_price = llamma.get_p()
    assert new_collat_price > prev_collat_price
