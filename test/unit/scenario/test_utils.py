"""
Test that the utility functions for resampling
the Controllers and LLAMMAS work as expected.
"""
from copy import deepcopy
from hypothesis import given
import hypothesis.strategies as st
from src.configs import MODELLED_MARKETS
from src.sim import Scenario
from src.sim.utils import raise_controller_price, clear_controller, find_active_band


@given(i=st.integers(min_value=0, max_value=len(MODELLED_MARKETS) - 1))
def test_raise_controller_price(scenario: Scenario, i: int) -> None:
    """
    Test that raising the controller price works as expected.
    """
    controller = deepcopy(scenario.controllers[i])
    llamma = controller.AMM
    oracle = llamma.price_oracle_contract

    prev_min_band = llamma.min_band
    prev_active_band = llamma.active_band
    prev_p = llamma.get_p()
    prev_o_p = oracle.price()

    oracle.freeze()
    raise_controller_price(controller)

    assert llamma.min_band < prev_min_band
    assert llamma.active_band < prev_active_band
    assert llamma.get_p() > prev_p
    assert oracle.price() > prev_o_p


@given(i=st.integers(min_value=0, max_value=len(MODELLED_MARKETS) - 1))
def test_clear_controller(scenario: Scenario, i: int) -> None:
    """
    Test that clearing the controller works as expected.
    """
    controller = deepcopy(scenario.controllers[i])
    clear_controller(controller)
    assert controller.n_loans == 0
    assert len(controller.loan) == 0
    assert controller.total_debt() == 0


@given(i=st.integers(min_value=0, max_value=len(MODELLED_MARKETS) - 1))
def test_find_active_band(scenario: Scenario, i: int) -> None:
    """
    Test that finding the active band works as expected.
    """
    llamma = deepcopy(scenario.llammas[i])
    prev_active_band = llamma.active_band
    assert llamma.min_band < llamma.active_band
    assert sum(llamma.bands_y.values()) > 0
    find_active_band(llamma)
    assert llamma.active_band == prev_active_band
    assert llamma.min_band < llamma.active_band
    assert llamma.max_band > llamma.active_band
    for band in range(llamma.min_band, llamma.active_band):
        assert llamma.bands_y[band] == 0
    for band in range(llamma.active_band + 1, llamma.max_band):
        assert llamma.bands_x[band] == 0
    assert llamma.bands_y[llamma.active_band] > 0
