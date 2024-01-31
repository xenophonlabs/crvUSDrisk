"""
Provides a test suite for the Liquidator agent.
These also test the debt resampling in many ways.
"""
from copy import deepcopy
import math
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
from crvusdsim.pool.sim_interface.sim_controller import DEFAULT_LIQUIDATOR
from src.configs import MODELLED_MARKETS
from src.sim import Scenario
from src.utils import get_crvusd_index
from src.agents import Liquidator
from ...utils import approx


@pytest.fixture(scope="module")
def liquidation_tests_passed() -> dict:
    """Track successes over hypothesis tests."""
    return {"passed": False}


@given(i=st.integers(min_value=0, max_value=len(MODELLED_MARKETS) - 1))
@settings(deadline=None)
def test_one_liquidation(  # pylint: disable=too-many-locals, redefined-outer-name
    scenario: Scenario, liquidation_tests_passed: dict, i: int
) -> None:
    """
    For a position that should be liquidated, test that the
    liquidation results in the expected asset transfers.
    """
    _scenario = deepcopy(scenario)
    _scenario.resample_debt()  # this should usually result in some underwater positions

    controller = _scenario.controllers[i]
    to_liquidate = controller.users_to_liquidate()

    if len(to_liquidate) == 0:
        # sometimes this happens and its ok
        return

    position = to_liquidate[0]
    liquidator = _scenario.liquidator
    liquidator.tolerance = -math.inf  # force liquidation
    prices = _scenario.curr_price

    to_repay = int(controller.tokens_to_liquidate(position.user))
    y = int(controller.AMM.get_sum_xy(position.user)[1])

    paths = liquidator.paths[controller.address]
    crvusd_bals = [
        path.crvusd_pool.balances[get_crvusd_index(path.crvusd_pool)] for path in paths
    ]
    collat_bal = controller.COLLATERAL_TOKEN.balanceOf[DEFAULT_LIQUIDATOR]

    assert liquidator.maybe_liquidate(position, controller, prices)

    new_crvusd_bals = [
        path.crvusd_pool.balances[get_crvusd_index(path.crvusd_pool)] for path in paths
    ]
    new_collat_bal = controller.COLLATERAL_TOKEN.balanceOf[DEFAULT_LIQUIDATOR]

    diffs = []
    for old, new in zip(crvusd_bals, new_crvusd_bals):
        diffs.append(old - new)

    assert (
        sum(1 for item in diffs if item != 0) == 1
    )  # should only be one nonzero difference
    assert approx(to_repay, sum(diffs))
    assert approx(new_collat_bal - collat_bal, y)

    liquidation_tests_passed["passed"] = True


def test_at_least_one_liquidation_test_passed(
    liquidation_tests_passed: dict,  # pylint: disable=redefined-outer-name
) -> None:
    """
    Ensure that the liquidation test passed on at
    least one controller. Sometimes, they don't all
    pass because the KDE resampling didn't result
    in any users being liquidatable.
    """
    assert liquidation_tests_passed["passed"]


def test_perform_liquidations(scenario: Scenario) -> None:
    """
    Test that performing liquidations will result in all
    controllers having no users to liquidate.
    """
    _scenario = deepcopy(scenario)
    _scenario.resample_debt()  # this should usually result in some underwater positions

    to_liquidate = {c.address: c.users_to_liquidate() for c in _scenario.controllers}
    n = sum(len(v) for v in to_liquidate.values())
    assert n > 0

    liquidator = _scenario.liquidator
    liquidator.tolerance = -math.inf  # force liquidation

    prev_count = liquidator.count()
    prev_profit = liquidator.profit()

    liquidator.perform_liquidations(_scenario.controllers, _scenario.curr_price)

    assert liquidator.count() == prev_count + n
    assert liquidator.profit() != prev_profit

    to_liquidate = {c.address: c.users_to_liquidate() for c in _scenario.controllers}
    n = sum(len(v) for v in to_liquidate.values())
    assert n == 0


def test_paths(scenario: Scenario) -> None:
    """
    Test that the liquidator paths are set up correctly.
    """
    controllers = {c.address: c for c in scenario.controllers}
    liquidator = Liquidator()
    liquidator.set_paths(
        scenario.controllers, scenario.stableswap_pools, scenario.markets
    )
    crvusd = scenario.stablecoin.address
    assert len(liquidator.paths) == len(controllers)
    for controller_address, paths in liquidator.paths.items():
        assert len(paths) == len(liquidator.basis_tokens)
        controller = controllers[controller_address]
        collat: str = controller.COLLATERAL_TOKEN.address
        for path in paths:
            assert path.crvusd_pool in scenario.stableswap_pools
            assert path.collat_pool in scenario.markets.values()

            assert collat in path.collat_pool.coin_addresses
            assert crvusd in path.crvusd_pool.coin_addresses

            assert path.basis_token.address in path.collat_pool.coin_addresses
            assert path.basis_token.address in path.crvusd_pool.coin_addresses
