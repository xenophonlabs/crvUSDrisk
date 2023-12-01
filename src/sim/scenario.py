"""Provides the `Scenario` class for running simulations."""
import json
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations
from crvusdsim.pool import get  # type: ignore
from ..prices import PricePaths, PriceSample
from ..configs import TOKEN_DTOs, DEFAULT_PROFIT_TOLERANCE
from ..modules import ExternalMarket
from ..db.datahandler import DataHandler
from ..agents import Arbitrageur
from ..agents import Liquidator
from ..data_transfer_objects import TokenDTO
from ..metrics.metricsprocessor import MetricsProcessor, MetricsResult
from ..types import MarketsType


@dataclass
class Scenario:
    """
    The `Scenario` object holds ALL of the objects required to run
    a simulation. This includes the pricepaths, the external markets,
    all crvusdsim modules (LLAMMAs, Controllers, etc.), and all agents.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, fn: str):
        "Generate the scenario from the stress test scenario config file."

        with open(fn, "r", encoding="utf-8") as f:
            logging.info("Reading price config from %s.", fn)
            self.config = config = json.load(f)

        self.name: str = config["name"]
        self.description: str = config["description"]
        self.num_steps: int = config["N"]
        self.price_config: str = config["price_config"]
        self.coins: List[TokenDTO] = [TOKEN_DTOs[a] for a in config["coins"]]
        self.pairs: List[Tuple[TokenDTO, TokenDTO]] = [
            (sorted_pair[0], sorted_pair[1])
            for pair in combinations(self.coins, 2)
            for sorted_pair in [sorted(pair)]
        ]

        self.generate_pricepaths()
        self.generate_markets()
        self.generate_sim_market()
        self.generate_agents()
        self.generate_metrics_processor()

    def generate_markets(self) -> None:
        """Generate the external markets for the scenario."""
        with DataHandler() as datahandler:
            quotes = datahandler.get_quotes(process=True)
            logging.info("Using %d 1Inch quotes.", quotes.shape[0])
        logging.info("Fitting external markets against 1inch quotes.")
        self.markets: MarketsType = {}
        for pair in self.pairs:
            market = ExternalMarket(pair)
            market.fit(quotes)
            self.markets[pair] = market

    def generate_pricepaths(self, fn: Optional[str] = None) -> None:
        """
        Generate the pricepaths for the scenario.
        """
        fn = fn if fn else self.price_config  # override
        self.pricepaths: PricePaths = PricePaths(fn, self.num_steps)

    def generate_agents(self) -> None:
        """Generate the agents for the scenario."""
        self.arbitrageur: Arbitrageur = Arbitrageur(DEFAULT_PROFIT_TOLERANCE)
        self.liquidator: Liquidator = Liquidator(DEFAULT_PROFIT_TOLERANCE)
        # TODO set liquidator paths

    def generate_sim_market(self) -> None:
        """
        Generate the crvusd modules to simulate, including
        LLAMMAs, Controllers, StableSwap pools, etc.
        """
        # TODO handle generation of multiple markets
        # TODO unpack sim_market objects
        # TODO assert that shared modules `is` the same object
        # assert pool == controller.AMM, "`controller.AMM` is not `pool`"
        # assert pool.BORROWED_TOKEN == controller.STABLECOIN
        # assert pool.COLLATERAL_TOKEN == controller.COLLATERAL_TOKEN
        sim_market = get("weth", bands_data="controller")
        self.llamma = sim_market.pool
        self.controller = sim_market.controller

    def generate_metrics_processor(self) -> None:
        """Generate the metrics processor for the scenario."""
        self.metricsprocessor: MetricsProcessor = MetricsProcessor()

    def update_market_prices(self, sample: PriceSample) -> None:
        """Update market prices with a new sample."""
        for pair in self.pairs:
            self.markets[pair].update_price(sample.prices)

    def prepare_for_run(self, prices: PriceSample) -> None:
        """Prepare all modules for a simulation run."""

    def prepare_for_trades(self, ts: int) -> None:
        """Prepare all modules for a new time step."""
        # TODO prepare aggregator for trades? <- update timestamps
        # TODO prepare pegkeeper for trades? <- update timestamps
        # TODO prepare stableswap pools for trades? <- update timestamps
        # TODO prepare tricryptoo pools for trades? <- update timestamps

    def after_trades(self) -> None:
        """Perform post processing for all modules at the end of a time step."""
        # TODO any post processing required?
        # NOTE controller.after_trades() forces liquidations,
        # we do NOT want to do this.

    def perform_actions(self, prices: PriceSample) -> None:
        """Perform all agent actions for a time step."""
        # TODO arbitrageur
        # TODO liquidator
        # TODO keeper
        # TODO borrower
        # TODO liquidity_provider

    def update_metrics(self) -> None:
        """Update the metrics processor with data from the current time step."""
        self.metricsprocessor.update()

    def process_metrics(self) -> MetricsResult:
        """Process the metrics for this simulation run."""
        return self.metricsprocessor.process()
