"""
Provides utility functions for the simulation.
"""
from typing import List
from crvusdsim.pool import SimMarketInstance


def rebind_markets(sim_markets: List[SimMarketInstance]) -> None:
    """
    Given a collection of sim_market objects,
    ensure all shared objects point to a single instance.

    For example, each input market will have its own set of stableswap
    pools. This function ensures only a single set of stableswap pools
    is used across all markets, discarding the others.
    """
    master = sim_markets[0]
    master_stablecoin = master.stablecoin
    master_aggregator = master.aggregator
    master_factory = master.factory
    master_spools = master.stableswap_pools
    master_pks = master.peg_keepers

    tricrypto_seen = {}
    for tpool in master.tricrypto:
        tricrypto_seen[tpool.address] = tpool

    stableswap_seen = {}
    for spool in master_spools:
        stableswap_seen[spool.address] = spool

    for sim_market in sim_markets[1:]:
        debt_ceiling = sim_market.stablecoin.balanceOf[sim_market.controller.address]

        # Top-level objects
        sim_market.stablecoin = master_stablecoin
        sim_market.aggregator = master_aggregator
        sim_market.factory = master_factory
        sim_market.stableswap_pools = master_spools
        sim_market.peg_keepers = master_pks

        # LLAMMA
        sim_market.pool.BORROWED_TOKEN = master_stablecoin
        master_stablecoin.mint(
            sim_market.pool.address, sum(sim_market.pool.bands_x.values())
        )

        # Controller
        sim_market.controller.STABLECOIN = master_stablecoin
        sim_market.controller.FACTORY = master_factory

        # MPolicy
        sim_market.policy.CONTROLLER_FACTORY = master_factory
        sim_market.policy.peg_keepers = master_pks

        # Price Oracle
        for i, tpool in enumerate(sim_market.price_oracle.tricrypto):
            # Rebind tricrypto pools
            if tpool.address not in tricrypto_seen:
                tricrypto_seen[tpool.address] = tpool
            sim_market.price_oracle.tricrypto[i] = tricrypto_seen[tpool.address]

        if hasattr(sim_market.price_oracle, "stableswap"):
            for i, spool in enumerate(sim_market.price_oracle.stableswap):
                # Rebind stableswap pools
                sim_market.price_oracle.stableswap[i] = stableswap_seen[spool.address]

        sim_market.price_oracle.stable_aggregator = master_aggregator
        sim_market.price_oracle.factory = master_factory

        # Tricrypto
        sim_market.tricrypto = sim_market.price_oracle.tricrypto

        # Add market to master factory
        master_factory._add_market_without_creating(  # pylint: disable=protected-access
            sim_market.pool,
            sim_market.controller,
            sim_market.policy,
            sim_market.collateral_token,
            debt_ceiling,
        )

    validate_binds(sim_markets)


def validate_binds(sim_markets: List[SimMarketInstance]) -> None:
    """
    Ensure binds are correct.
    TODO move to test file
    TODO dig deeper into shared objects (e.g. pool.BORROWED_TOKEN)
    """
    SHARED = ["stablecoin", "aggregator", "stableswap_pools", "peg_keepers", "factory"]
    for k, master in sim_markets[0].__dict__.items():
        for sim_market in sim_markets[1:]:
            if k in SHARED:
                assert sim_market.__dict__[k] is master
            else:
                assert sim_market.__dict__[k] is not master
