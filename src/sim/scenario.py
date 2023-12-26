"""
Provides the `Scenario` class for running simulations.
"""

from typing import List, Tuple, Set
from itertools import combinations
from datetime import datetime, timedelta
import pandas as pd
from crvusdsim.pool import get  # type: ignore
from ..prices import PriceSample, PricePaths
from ..configs import TOKEN_DTOs, get_scenario_config, get_price_config, CRVUSD_DTO
from ..modules import ExternalMarket
from ..agents import Arbitrageur, Liquidator, Keeper, Borrower, LiquidityProvider
from ..data_transfer_objects import TokenDTO
from ..types import MarketsType
from ..utils.poolgraph import PoolGraph
from ..logging import get_logger
from ..utils import get_quotes

logger = get_logger(__name__)


class Scenario:
    """
    The `Scenario` object holds ALL of the objects required to run
    a simulation. This includes the pricepaths, the external markets,
    all crvusdsim modules (LLAMMAs, Controllers, etc.), and all agents.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, scenario: str, market_name: str):
        self.market_name = market_name
        self.config = config = get_scenario_config(scenario)
        self.name: str = config["name"]
        self.description: str = config["description"]
        self.num_steps: int = config["N"]
        self.freq: str = config["freq"]

        self.price_config = get_price_config(self.freq)

        self.generate_sim_market()  # must be first
        self.generate_pricepaths()
        self.generate_markets()
        self.generate_agents()

    def generate_markets(self) -> pd.DataFrame:
        """Generate the external markets for the scenario."""
        offset = timedelta(seconds=self.config["quotes"]["offset"])
        period = timedelta(seconds=self.config["quotes"]["period"])

        end = datetime.now() - offset
        start = end - period

        quotes = get_quotes(
            int(start.timestamp()),
            int(end.timestamp()),
            self.coins,
        )
        logger.debug("Using %d 1Inch quotes from %s to %s", quotes.shape[0], start, end)

        self.markets: MarketsType = {}
        for pair in self.pairs:
            market = ExternalMarket(pair)
            market.fit(quotes)
            self.markets[pair] = market

        return quotes

    def generate_pricepaths(self) -> None:
        """
        Generate the pricepaths for the scenario.
        """
        self.pricepaths: PricePaths = PricePaths(self.num_steps, self.price_config)

    def generate_agents(self) -> None:
        """Generate the agents for the scenario."""
        self.arbitrageur: Arbitrageur = Arbitrageur()
        self.liquidator: Liquidator = Liquidator()
        self.keeper: Keeper = Keeper()
        self.borrower: Borrower = Borrower()
        self.liquidity_provider: LiquidityProvider = LiquidityProvider()

        # Set liquidator paths
        self.liquidator.set_paths(self.controller, self.stableswap_pools, self.markets)
        # Set arbitrage cycles
        self.graph = PoolGraph(
            self.stableswap_pools + [self.llamma] + list(self.markets.values())
        )
        self.arb_cycle_length = self.config["arb_cycle_length"]
        self.cycles = self.graph.find_cycles(n=self.arb_cycle_length)

        # For convenience
        self.agents = [
            self.arbitrageur,
            self.liquidator,
            self.keeper,
            self.borrower,
            self.liquidity_provider,
        ]

    def generate_sim_market(self) -> None:
        """
        Generate the crvusd modules to simulate, including
        LLAMMAs, Controllers, StableSwap pools, etc.
        """
        # TODO handle generation of multiple markets
        logger.info("Fetching sim_market from subgraph.")
        sim_market = get(self.market_name, bands_data="controller")

        metadata = sim_market.pool.metadata

        logger.info(
            "Market snapshot as %s", datetime.fromtimestamp(int(metadata["timestamp"]))
        )
        logger.info(
            "Bands snapshot as %s",
            datetime.fromtimestamp(int(metadata["bands"][0]["timestamp"])),
        )
        logger.info(
            "Users snapshot as %s",
            datetime.fromtimestamp(int(metadata["userStates"][0]["timestamp"])),
        )

        # NOTE huge memory consumption
        del sim_market.pool.metadata["bands"]
        del sim_market.pool.metadata["userStates"]  # TODO compare these before deleting

        self.llamma = sim_market.pool
        self.controller = sim_market.controller
        self.stableswap_pools = sim_market.stableswap_pools
        self.aggregator = sim_market.aggregator
        self.price_oracle = sim_market.price_oracle
        self.peg_keepers = sim_market.peg_keepers
        self.policy = sim_market.policy
        self.factory = sim_market.factory
        self.stablecoin = sim_market.stablecoin

        # Convenient reference to all pools
        # TODO add other LLAMMAs or TriCrypto pools
        self.pools = [self.llamma] + self.stableswap_pools

        self.coins: Set[TokenDTO] = set()
        for pool in self.pools:
            for a in pool.coin_addresses:
                if a.lower() != CRVUSD_DTO.address.lower():
                    self.coins.add(TOKEN_DTOs[a.lower()])

        self.pairs: List[Tuple[TokenDTO, TokenDTO]] = [
            (sorted_pair[0], sorted_pair[1])
            for pair in combinations(self.coins, 2)
            for sorted_pair in [sorted(pair)]
        ]

    def update_market_prices(self, sample: PriceSample) -> None:
        """Update market prices with a new sample."""
        for pair in self.pairs:
            self.markets[pair].update_price(sample.prices)

    def prepare_for_run(self) -> None:
        """
        Prepare all modules for a simulation run.

        Perform initial setup in this order:
        1. Run a single step of price arbitrages.
        2. Liquidate users that were loaded underwater.
        """
        sample = self.pricepaths[0]
        ts = int(sample.timestamp.timestamp())

        # Setup timestamp related attrs for all modules
        self._increment_timestamp(ts)

        # Set external market prices
        self.prepare_for_trades(sample)

        # Equilibrate pool prices
        arbitrageur = Arbitrageur()  # diff arbitrageur
        profit, count = arbitrageur.arbitrage(self.cycles, sample)
        logger.debug(
            "Equilibrated prices with %d arbitrages with total profit %d", count, profit
        )

        # Liquidate users that were loaded underwater
        controller = self.controller
        to_liquidate = controller.users_to_liquidate()
        n = len(to_liquidate)
        if n > 0:
            logger.debug("%d users were loaded underwater.", n)

        # Check that only a small portion of debt is liquidatable at start
        damage = 0
        for pos in to_liquidate:
            damage += pos.debt
            logger.debug("Liquidating %s: with debt %d.", pos.user, pos.debt)
        pct = round(damage / controller.total_debt() * 100, 2)
        args = (
            "%.2f%% of debt was incorrectly loaded with sub-zero health (%d crvUSD)",
            pct,
            damage / 1e18,
        )
        if pct > 1:
            logger.warning(*args)
        else:
            logger.debug(*args)

        controller.after_trades(do_liquidate=True)  # liquidations

    def _increment_timestamp(self, ts: int) -> None:
        """Increment the timestamp for all modules."""
        self.aggregator._increment_timestamp(ts)  # pylint: disable=protected-access
        self.aggregator.last_timestamp = ts

        self.llamma._increment_timestamp(ts)  #  pylint: disable=protected-access
        self.llamma.prev_p_o_time = ts
        self.llamma.rate_time = ts
        self.price_oracle._increment_timestamp(ts)  #  pylint: disable=protected-access
        self.price_oracle.last_prices_timestamp = ts

        self.controller._increment_timestamp(ts)  #  pylint: disable=protected-access

        for spool in self.stableswap_pools:
            spool._increment_timestamp(ts)  #  pylint: disable=protected-access
            spool.ma_last_time = ts

        for pk in self.peg_keepers:
            pk._increment_timestamp(ts)  #  pylint: disable=protected-access

    def prepare_for_trades(self, sample: PriceSample) -> None:
        """Prepare all modules for a new time step."""
        ts = sample.timestamp
        self.update_market_prices(sample)

        # FIXME Manually update LLAMMA oracle price
        _p = int(
            sample.prices_usd[self.llamma.COLLATERAL_TOKEN.address]
            * 10**self.llamma.COLLATERAL_TOKEN.decimals
        )
        self.price_oracle.set_price(_p)

        self.llamma.prepare_for_trades(ts)
        self.controller.prepare_for_trades(ts)
        self.aggregator.prepare_for_trades(ts)
        for spool in self.stableswap_pools:
            spool.prepare_for_trades(ts)
        for peg_keeper in self.peg_keepers:
            peg_keeper.prepare_for_trades(ts)

    def after_trades(self) -> None:
        """Perform post processing for all modules at the end of a time step."""

    def perform_actions(self, prices: PriceSample) -> None:
        """Perform all agent actions for a time step."""
        _, _ = self.arbitrageur.arbitrage(self.cycles, prices)
        _, _ = self.liquidator.perform_liquidations(self.controller)
        _, _ = self.keeper.update(self.peg_keepers)
        # TODO what is the right order for the actions?
        # TODO need to incorporate non-PK pools for arbs (e.g. TriCRV)
        # TODO need to incorporate LPs add/remove liquidity
        # ^ currently USDT pool has more liquidity and this doesn't change since we
        # only model swaps.
        # TODO borrower
