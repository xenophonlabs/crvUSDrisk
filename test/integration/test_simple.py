"""
Provides a very simple integration test that
just runs the simulation and checks that no errors
are raised. Better integration tests would be great.
"""
from unittest.mock import patch
from src.sim import simulate
from src.configs import MODELLED_MARKETS
from ..conftest import (
    mocked_get_current_prices,
    mocked_get_sim_market,
    mocked_get_quotes,
)


def test_simple() -> None:
    """
    Test that the simulation runs without error.
    """
    with (
        patch("src.configs.get_current_prices", side_effect=mocked_get_current_prices),
        patch("src.sim.scenario.get", side_effect=mocked_get_sim_market),
        patch("src.sim.scenario.get_quotes", side_effect=mocked_get_quotes),
    ):
        simulate("test", MODELLED_MARKETS, num_iter=1, ncpu=1)
