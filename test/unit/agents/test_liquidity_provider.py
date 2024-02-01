"""
Provides a test suite for the Liquidity Provider agent.
"""
from typing import List
from copy import deepcopy
from hypothesis import given, settings
import hypothesis.strategies as st
import numpy as np
from src.sim import Scenario
from src.agents import LiquidityProvider
from ...utils import approx


@given(
    i=st.integers(min_value=0, max_value=3),
    _amounts=st.lists(
        st.integers(min_value=1, max_value=1_000_000), min_size=2, max_size=2
    ),
)
@settings(max_examples=10, deadline=None)
def test_add_liquidity(scenario: Scenario, i: int, _amounts: List[int]) -> None:
    """
    The LP is very simple and just deposits into
    stableswap pools.
    """
    _scenario = deepcopy(scenario)
    lp = LiquidityProvider()
    spool = _scenario.stableswap_pools[i]
    old_balances = deepcopy(spool.balances)
    amounts = np.array(
        [amt * 10**decimals for amt, decimals in zip(_amounts, spool.coin_decimals)]
    )
    lp.add_liquidity(spool, amounts)
    new_balances = spool.balances

    for idx, (old, new) in enumerate(zip(old_balances, new_balances)):
        assert approx(old + amounts[idx], new)
