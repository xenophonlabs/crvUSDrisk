"""
Provides the `Scenario` class for running simulations.
"""

from typing import List, Tuple, Set, cast
from itertools import combinations
from datetime import datetime
import pandas as pd
from crvusdsim.pool import get, SimMarketInstance  # type: ignore
from crvusdsim.pool.crvusd.price_oracle.crypto_with_stable_price import Oracle
from curvesim.pool.sim_interface import SimCurveCryptoPool
from .utils import rebind_markets
from ..prices import PriceSample, PricePaths
from ..configs import TOKEN_DTOs, get_scenario_config, get_price_config, CRVUSD_DTO
from ..configs.tokens import WETH, WSTETH, SFRXETH
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
    def __init__(self, scenario: str, market_names: List[str]):
        self.market_names = market_names
        self.config = config = get_scenario_config(scenario)
        self.name: str = config["name"]
        self.description: str = config["description"]
        self.num_steps: int = config["N"]
        self.freq: str = config["freq"]

        self.price_config = get_price_config(
            self.freq, self.config["prices"]["start"], self.config["prices"]["end"]
        )

        self.generate_sim_market()  # must be first
        self.generate_pricepaths()
        self.generate_markets()
        self.generate_agents()

    def generate_markets(self) -> pd.DataFrame:
        """Generate the external markets for the scenario."""
        start = self.config["quotes"]["start"]
        end = self.config["quotes"]["end"]

        quotes = get_quotes(
            start,
            end,
            self.coins,
        )
        logger.info(
            "Using %d 1Inch quotes from %s to %s",
            quotes.shape[0],
            datetime.fromtimestamp(start),
            datetime.fromtimestamp(end),
        )

        self.quotes_start = start
        self.quotes_end = end

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
        self.liquidator.set_paths(self.controllers, self.stableswap_pools, self.markets)

        # Set arbitrage cycles
        # TODO include TBTC TriCrypto pool
        self.graph = PoolGraph(
            self.stableswap_pools + self.llammas + list(self.markets.values())
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
        sim_markets: List[SimMarketInstance] = []
        for market_names in self.market_names:
            logger.info("Fetching %s market from subgraph", market_names)

            sim_market = get(
                market_names, bands_data="controller", use_simple_oracle=False
            )

            metadata = sim_market.pool.metadata

            logger.info(
                "Market snapshot as %s",
                datetime.fromtimestamp(int(metadata["timestamp"])),
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
            del sim_market.pool.metadata[
                "userStates"
            ]  # TODO compare these before deleting

            sim_markets.append(sim_market)

        # Rebind all markets to use the same shared resources
        rebind_markets(sim_markets)

        # Set scenario attributes
        self.llammas = [sim_market.pool for sim_market in sim_markets]
        self.controllers = [sim_market.controller for sim_market in sim_markets]
        self.oracles = cast(
            List[Oracle], [sim_market.price_oracle for sim_market in sim_markets]
        )
        self.policies = [sim_market.policy for sim_market in sim_markets]
        self.stableswap_pools = sim_markets[0].stableswap_pools
        self.aggregator = sim_markets[0].aggregator
        self.peg_keepers = sim_markets[0].peg_keepers
        self.factory = sim_markets[0].factory
        self.stablecoin = sim_markets[0].stablecoin

        self.tricryptos: Set[SimCurveCryptoPool] = set()
        for sim_market in sim_markets:
            self.tricryptos.update(sim_market.tricrypto)

        self.coins: Set[TokenDTO] = set()
        for pool in self.llammas + self.stableswap_pools:
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
        arbitrageur.arbitrage(self.cycles, sample)
        logger.info(
            "Equilibrated prices with %d arbitrages with total profit %d",
            arbitrageur.count,
            arbitrageur.profit,
        )

        # Liquidate users that were loaded underwater
        for controller in self.controllers:
            name = controller.AMM.name.replace("Curve.fi Stablecoin ", "")
            logger.info("Validating loaded positions in %s Controller.", name)

            to_liquidate = controller.users_to_liquidate()
            n = len(to_liquidate)

            # Check that only a small portion of debt is liquidatable at start
            damage = 0
            for pos in to_liquidate:
                damage += pos.debt
                logger.info("Liquidating %s: with debt %d.", pos.user, pos.debt)
            pct = round(damage / controller.total_debt() * 100, 2)
            args = (
                "%.2f%% of debt (%d positions) was incorrectly \
                    loaded with sub-zero health (%d crvUSD)",
                pct,
                n,
                damage / 1e18,
            )
            if pct > 1:
                logger.warning(*args)
            else:
                logger.info(*args)

            controller.after_trades(do_liquidate=True)  # liquidations

    def _increment_timestamp(self, ts: int) -> None:
        """Increment the timestamp for all modules."""
        self.aggregator._increment_timestamp(ts)  # pylint: disable=protected-access

        for llamma in self.llammas:
            llamma._increment_timestamp(ts)  #  pylint: disable=protected-access
            llamma.prev_p_o_time = ts
            llamma.rate_time = ts

        for oracle in self.oracles:
            oracle._increment_timestamp(ts)  #  pylint: disable=protected-access

        for controller in self.controllers:
            controller._increment_timestamp(ts)  #  pylint: disable=protected-access

        for spool in self.stableswap_pools:
            spool._increment_timestamp(ts)  #  pylint: disable=protected-access
            spool.ma_last_time = ts

        for tpool in self.tricryptos:
            tpool._increment_timestamp(ts)  #  pylint: disable=protected-access
            tpool.last_prices_timestamp = ts

        for pk in self.peg_keepers:
            pk._increment_timestamp(ts)  #  pylint: disable=protected-access

    def update_tricrypto_prices(self, sample: PriceSample) -> None:
        """Update TriCrypto prices with a new sample."""
        for tpool in self.tricryptos:
            tpool.prepare_for_run(
                pd.DataFrame(
                    [
                        [
                            # Get pairwise prices for TriCrypto pool assets
                            sample.prices[pair[0].lower()][pair[1].lower()]
                            for pair in list(combinations(tpool.assets.addresses, 2))
                        ]
                    ]
                )
            )

    def update_staked_prices(self, sample: PriceSample) -> None:
        """Update staked prices with a new sample."""
        # Need to know which assets (staked and base)
        for llamma in self.llammas:
            oracle = cast(Oracle, llamma.price_oracle_contract)
            if hasattr(oracle, "staked_oracle"):
                derivative = llamma.COLLATERAL_TOKEN.address
                base = WETH
                assert derivative in [WSTETH, SFRXETH]
                p_staked = int(sample.prices[derivative][base] * 10**18)
                oracle.staked_oracle.update(p_staked)

    def prepare_for_trades(self, sample: PriceSample) -> None:
        """Prepare all modules for a new time step."""
        ts = sample.timestamp
        self.update_market_prices(sample)
        self.update_tricrypto_prices(sample)
        self.update_staked_prices(sample)

        for llamma in self.llammas:
            llamma.prepare_for_trades(ts)  # this increments oracle timestamp

        for controller in self.controllers:
            controller.prepare_for_trades(ts)

        self.aggregator.prepare_for_trades(ts)

        for spool in self.stableswap_pools:
            spool.prepare_for_trades(ts)

        for tpool in self.tricryptos:
            tpool.prepare_for_trades(ts)

        for peg_keeper in self.peg_keepers:
            peg_keeper.prepare_for_trades(ts)

        sample.prices_usd[CRVUSD_DTO.address] = self.aggregator.price() / 1e18

    def after_trades(self) -> None:
        """Perform post processing for all modules at the end of a time step."""

    def perform_actions(self, prices: PriceSample) -> None:
        """Perform all agent actions for a time step."""
        self.arbitrageur.arbitrage(self.cycles, prices)
        self.liquidator.perform_liquidations(self.controllers, prices)
        self.keeper.update(self.peg_keepers)
        # TODO what is the right order for the actions?
        # TODO need to incorporate LPs add/remove liquidity
        # ^ currently USDT pool has more liquidity and this doesn't change since we
        # only model swaps.
        # TODO borrower
