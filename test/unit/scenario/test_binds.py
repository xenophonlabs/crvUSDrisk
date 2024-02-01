"""
Test the scenario's binds for Curve assets.
"""
from typing import List
from unittest.mock import patch
from crvusdsim.pool import SimMarketInstance
from src.sim import Scenario
from src.sim.utils import rebind_markets
from ...conftest import mocked_get_sim_market


def validate_binds(sim_markets: List[SimMarketInstance]) -> None:
    """
    Ensure that all simulated contracts are sharing the same
    underlying instances for the SHARED resources.
    """
    SHARED = ["stablecoin", "aggregator", "stableswap_pools", "peg_keepers", "factory"]
    for k, master in sim_markets[0].__dict__.items():
        for sim_market in sim_markets[1:]:
            if k in SHARED:
                assert sim_market.__dict__[k] is master
            else:
                assert sim_market.__dict__[k] is not master


def test_binds(scenario: Scenario) -> None:
    """
    Test the binds.
    """
    with patch("src.sim.scenario.get", side_effect=mocked_get_sim_market):
        sim_markets = scenario.fetch_markets()
    rebind_markets(sim_markets)
    validate_binds(sim_markets)
