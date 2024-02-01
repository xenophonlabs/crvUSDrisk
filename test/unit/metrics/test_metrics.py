"""
Provides test for the metrics and the SingleSimProcessor.
We test the most important/intricate metrics like Bad Debt
and Borrower Losses. We don't test metrics that track simple
things like prices, although such tests could be added.
"""
from copy import deepcopy
import random
from hypothesis import given, settings
import hypothesis.strategies as st
from src.metrics import (
    DEFAULT_METRICS,
    init_metrics,
    BadDebtMetric,
    BorrowerLossMetric,
    LiquidationsMetric,
    ProfitsMetric,
)
from src.sim import Scenario
from src.sim.processing import SingleSimProcessor
from ...utils import scale_prices, approx


def get_bad_debt(_scenario: Scenario) -> int:
    """
    Get the bad debt in the controller.
    """
    to_liquidate = {c.address: c.users_to_liquidate() for c in _scenario.controllers}

    bad_debt = 0
    for positions in to_liquidate.values():
        for position in positions:
            bad_debt += position.debt

    return bad_debt


def test_metrics(scenario: Scenario) -> None:
    """
    Test basic metric attributes.
    """
    metrics = init_metrics(DEFAULT_METRICS, deepcopy(scenario))
    for metric in metrics:
        # Child class attributes
        assert hasattr(metric, "key_metric")
        assert hasattr(metric, "config")
        assert hasattr(metric, "compute")

        cfg = metric.config
        assert metric.key_metric in cfg
        for agg_funcs in cfg.values():
            assert isinstance(agg_funcs, list)
            assert agg_funcs


def test_bad_debt_metric(scenario: Scenario) -> None:
    """
    Test that the bad debt metric is working.
    """
    _scenario = deepcopy(scenario)

    total_debt = _scenario.total_debt

    # Check there is no bad debt
    to_liquidate = {c: c.users_to_liquidate() for c in _scenario.controllers}
    n = sum(len(v) for v in to_liquidate.values())
    assert n == 0

    processor = SingleSimProcessor(_scenario, [BadDebtMetric])
    processor.update(_scenario.curr_price.timestamp, inplace=True)

    assert processor.results.iloc[-1][BadDebtMetric.key_metric] == 0

    # Force some bad debt
    _scenario.resample_debt()  # this should usually result in some underwater positions
    bad_debt = get_bad_debt(_scenario)
    assert bad_debt > 0

    processor.update(_scenario.curr_price.timestamp, inplace=True)

    actual = processor.results.iloc[-1][BadDebtMetric.key_metric]
    expected = bad_debt / total_debt * 100

    assert approx(actual, expected)


@given(i=st.integers(min_value=0, max_value=3))
@settings(deadline=None)
def test_borrower_loss_metric(scenario: Scenario, i: int) -> None:
    """
    Test that the borrower loss metrics are working.
    """
    _scenario = deepcopy(scenario)
    _scenario.prepare_for_run(resample=False)

    # Check there is no bad debt
    to_liquidate = {c: c.users_to_liquidate() for c in _scenario.controllers}
    n = sum(len(v) for v in to_liquidate.values())
    assert n == 0

    processor = SingleSimProcessor(_scenario, [BorrowerLossMetric])
    processor.update(_scenario.curr_price.timestamp, inplace=True)

    metric = processor.metrics[0]
    for m in metric.config:
        # Ensure that metrics are zero at init
        assert processor.results.iloc[-1][m] == 0

    # Force LLAMMA arbs
    collat = _scenario.controllers[i].COLLATERAL_TOKEN.address
    scale_prices(_scenario, collat, 0.5)

    # Perform arbitrages
    _scenario.arbitrageur.arbitrage(_scenario.cycles, _scenario.curr_price)

    # Users should have losses from LVR
    processor.update(_scenario.curr_price.timestamp, inplace=True)
    assert processor.results.iloc[-1][BorrowerLossMetric.key_metric] > 0


def test_liquidations_metric(scenario: Scenario) -> None:
    """
    Test that the bad debt metric is working.
    """
    _scenario = deepcopy(scenario)

    total_debt = _scenario.total_debt

    # Check there is no bad debt
    to_liquidate = {c: c.users_to_liquidate() for c in _scenario.controllers}
    n = sum(len(v) for v in to_liquidate.values())
    assert n == 0

    processor = SingleSimProcessor(_scenario, [LiquidationsMetric])
    processor.update(_scenario.curr_price.timestamp, inplace=True)

    assert processor.results.iloc[-1][LiquidationsMetric.key_metric] == 0

    # Force some bad debt
    _scenario.resample_debt()  # this should usually result in some underwater positions

    bad_debt = get_bad_debt(_scenario)
    assert bad_debt > 0

    _scenario.liquidator.perform_liquidations(
        _scenario.controllers, _scenario.curr_price
    )

    processor.update(_scenario.curr_price.timestamp, inplace=True)

    new_bad_debt = get_bad_debt(_scenario)

    actual = processor.results.iloc[-1][LiquidationsMetric.key_metric]
    expected = (bad_debt - new_bad_debt) / total_debt * 100

    assert approx(actual, expected)


# pylint: disable=too-many-locals
def test_profits_metric(scenario: Scenario) -> None:
    """
    Test that the profits metric is working.
    """
    _scenario = deepcopy(scenario)

    processor = SingleSimProcessor(_scenario, [ProfitsMetric])
    processor.update(_scenario.curr_price.timestamp, inplace=True)

    assert processor.results.iloc[-1][ProfitsMetric.key_metric] == 0

    # Mix up the prices
    for coin in _scenario.coins:
        scale_prices(_scenario, coin.address, random.uniform(0.9, 1.1))

    _scenario.arbitrageur.arbitrage(_scenario.cycles, _scenario.curr_price)

    processor.update(_scenario.curr_price.timestamp, inplace=True)

    profits = 0
    prices = _scenario.curr_price.prices_usd
    crvusd_price = prices[_scenario.stablecoin.address]
    for llamma in _scenario.llammas:
        llamma_bands_fees_x = sum(llamma.bands_fees_x.values())
        llamma_bands_fees_y = sum(llamma.bands_fees_y.values())

        bands_fees_x = llamma_bands_fees_x
        bands_fees_y = llamma_bands_fees_y

        profit_x = bands_fees_x * crvusd_price / 1e18
        profit_y = bands_fees_y * prices[llamma.COLLATERAL_TOKEN.address] / 1e18
        profits += profit_x + profit_y  # cum

    actual = processor.results.iloc[-1][ProfitsMetric.key_metric]
    expected = profits / (_scenario.total_debt / 1e18) * 100

    assert approx(actual, expected)
