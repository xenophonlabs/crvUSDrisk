"""
Provides utility functions for the simulation.
"""
from typing import List
from crvusdsim.pool import SimMarketInstance, SimController, SimLLAMMAPool
from ..configs import ALIASES_LLAMMA, MODELLED_MARKETS


def parse_markets(market_names: List[str]) -> List[str]:
    """
    Parse list of market names.
    """
    aliases = []
    for alias in market_names:
        alias = alias.lower()
        if alias[:2] == "0x":
            alias = ALIASES_LLAMMA[alias]
        assert alias in MODELLED_MARKETS, ValueError(
            f"Only {MODELLED_MARKETS} markets are supported, not {alias}."
        )
        aliases.append(alias)
    return aliases


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
        debt_ceiling = sim_market.factory.debt_ceiling[sim_market.controller.address]

        # Top-level objects
        sim_market.stablecoin = master_stablecoin
        sim_market.aggregator = master_aggregator
        sim_market.factory = master_factory
        sim_market.stableswap_pools = master_spools
        sim_market.peg_keepers = master_pks

        # LLAMMA
        sim_market.pool.BORROWED_TOKEN = master_stablecoin
        crvusd_balance = sum(sim_market.pool.bands_x.values())
        if crvusd_balance > 0:
            master_stablecoin.mint(sim_market.pool.address, crvusd_balance)

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
        master_stablecoin.burnFrom(
            sim_market.controller.address, sim_market.controller.total_debt()
        )


def find_active_band(llamma: SimLLAMMAPool) -> None:
    """Find the active band for a LLAMMA."""
    min_band = llamma.min_band
    for n in range(llamma.min_band, llamma.max_band):
        if llamma.bands_x[n] == 0 and llamma.bands_y[n] == 0:
            min_band += 1
        if llamma.bands_y[n] > 0:
            llamma.active_band = n
            break
    llamma.min_band = min_band


def clear_controller(controller: SimController) -> None:
    """Clear all controller states."""
    users = list(controller.loan.keys())
    for user in users:
        debt = controller._debt(user)[0]  # pylint: disable=protected-access
        if debt > 0:
            controller.STABLECOIN._mint(user, debt)  # pylint: disable=protected-access
            controller.repay(debt, user)
        del controller.loan[user]

    for band in range(controller.AMM.min_band, controller.AMM.max_band):
        # Might be some dust left due to discrepancy between user
        # snapshot and band snapshot
        controller.AMM.bands_x[band] = 0
        controller.AMM.bands_y[band] = 0


def raise_controller_price(controller: SimController) -> None:
    """
    Raise controller price by a little bit.
    """
    llamma = controller.AMM

    min_band = llamma.min_band - 3  # Offset by a little bit
    active_band = min_band + 1

    llamma.active_band = active_band
    llamma.min_band = min_band

    p = llamma.p_oracle_up(active_band)
    llamma.price_oracle_contract.last_price = p

    ts = controller._block_timestamp  # pylint: disable=protected-access
    controller.prepare_for_trades(ts + 60 * 60)
    llamma.prepare_for_trades(ts + 60 * 60)
