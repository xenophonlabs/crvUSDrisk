"""
Test the Keeper agent.

Note
----
The logic for this test is in `demote_pegkeep.ipynb`.
"""
from src.sim import Scenario
from ...utils import increment_timestamps


# pylint: disable=too-many-locals
def test_update(scenario: Scenario) -> None:
    """
    Test that raising PK pool prices allows the
    Keeper to make a profit from updates.
    """
    keeper = scenario.keeper  # fresh keeper
    init_count = keeper.count()
    init_profit = keeper.profit()
    assert init_count == 0
    assert init_profit == 0

    aggregator = scenario.aggregator
    p_agg_prev = aggregator.price()

    spools = scenario.stableswap_pools
    pks = scenario.peg_keepers

    keeper.update(pks)
    prev_count = keeper.count()
    prev_profit = keeper.profit()

    for pk in pks:
        spool = pk.POOL
        normalized_balances = [
            b * r / 1e18 for b, r in zip(spool.balances, spool.rates)
        ]
        diff = normalized_balances[pk.I] - normalized_balances[pk.I ^ 1]
        if diff < 1:
            continue
        diff = int(diff * 1e18 / spool.rates[pk.I ^ 1])  # convert to peg coin units
        amt_in, _, _ = spool.trade(pk.I ^ 1, pk.I, diff)
        assert amt_in == diff
        normalized_balances = [
            b * r / 1e18 for b, r in zip(spool.balances, spool.rates)
        ]
    increment_timestamps([aggregator, *pks, *spools])

    keeper.update(pks)
    assert keeper.count() > prev_count
    assert keeper.profit() > prev_profit

    p_agg = aggregator.price()
    assert p_agg > p_agg_prev
