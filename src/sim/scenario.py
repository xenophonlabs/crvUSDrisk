"""Provides the `Scenario` class for running simulations."""
from typing import List, Tuple, Set
from itertools import combinations
from datetime import datetime, timedelta
from crvusdsim.pool import get  # type: ignore
from ..prices import PricePaths, PriceSample
from ..configs import TOKEN_DTOs, get_scenario_config, CRVUSD_DTO
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

        self.generate_sim_market()  # must be first
        self.generate_pricepaths()
        self.generate_markets()
        self.generate_agents()

        sample = self.pricepaths[0]
        self.timestamp = sample.timestamp
        self.curr_price = sample

    def generate_markets(self) -> None:
        """Generate the external markets for the scenario."""
        offset = timedelta(seconds=self.config["quotes"]["offset"])
        period = timedelta(seconds=self.config["quotes"]["period"])

        end = datetime.now() - offset
        start = end - period

        self.quotes = quotes = get_quotes(
            int(start.timestamp()),
            int(end.timestamp()),
            self.coins,
        )
        logger.info("Using %d 1Inch quotes from %s to %s", quotes.shape[0], start, end)

        self.markets: MarketsType = {}
        for pair in self.pairs:
            market = ExternalMarket(pair)
            market.fit(quotes)
            self.markets[pair] = market

    def generate_pricepaths(self) -> None:
        """
        Generate the pricepaths for the scenario.
        """
        self.pricepaths: PricePaths = PricePaths(self.freq, self.num_steps)

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

    def generate_sim_market(self) -> None:
        """
        Generate the crvusd modules to simulate, including
        LLAMMAs, Controllers, StableSwap pools, etc.
        """
        # TODO handle generation of multiple markets
        logger.info("Fetching sim_market from subgraph.")
        sim_market = get(self.market_name, bands_data="controller")

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
        """TODO Prepare all modules for a simulation run."""
        # We need oracles/aggregators/etc to agree on price at the
        # start of the simulation. EXCEPT: we aren't currently simulating
        # crvUSD price explicitly! This means that the simulated pools
        # are TELLING us what the price is, not the other way around.
        prices = self.pricepaths.prices
        # All `prepare_for_run` methods mostly set the timestamp and initial price
        # of pools/oracles to the first price/timestamp of the price sampler.
        self.llamma.prepare_for_run(prices, keep_price=True)
        self.controller.prepare_for_run(prices)
        self.aggregator.prepare_for_run(self.pricepaths, keep_price=True)
        # self.policy.prepare_for_run(prices)
        # self.factory.prepare_for_run(prices)
        # self.price_oracle.prepare_for_run(prices)
        for spool in self.stableswap_pools:
            spool.prepare_for_run(prices, keep_price=True)
        for peg_keeper in self.peg_keepers:
            peg_keeper.prepare_for_run(prices)

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
        # self.policy.prepare_for_trades(ts)
        # self.factory.prepare_for_trades(ts)
        # self.price_oracle.prepare_for_trades(ts)
        for spool in self.stableswap_pools:
            spool.prepare_for_trades(ts)
        for peg_keeper in self.peg_keepers:
            peg_keeper.prepare_for_trades(ts)

    def after_trades(self) -> None:
        """Perform post processing for all modules at the end of a time step."""
        # TODO any post processing required?
        # NOTE controller.after_trades() forces liquidations,
        # we do NOT want to do this.

    def perform_actions(self, prices: PriceSample) -> None:
        """Perform all agent actions for a time step."""
        _, _ = self.liquidator.perform_liquidations(self.controller)
        _, _ = self.arbitrageur.arbitrage(self.cycles, prices)
        _, _ = self.keeper.update(self.peg_keepers)
        # TODO what is the right order for the actions?
        # TODO need to incorporate non-PK pools for arbs (e.g. TriCRV)
        # TODO need to incorporate LPs add/remove liquidity
        # ^ currently USDT pool has more liquidity and this doesn't change since we
        # only model swaps.
        # TODO borrower
