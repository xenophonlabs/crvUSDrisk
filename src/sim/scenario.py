import json
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
from itertools import combinations
from crvusdsim.pool import get
from ..prices import PricePaths, PriceSample
from ..configs import TOKEN_DTOs, DEFAULT_PROFIT_TOLERANCE
from ..modules import ExternalMarket
from ..db.datahandler import DataHandler
from ..agents.arbitrageur import Arbitrageur
from ..agents.liquidator import Liquidator
from ..data_transfer_objects import TokenDTO
from ..metrics.metricsprocessor import MetricsProcessor


@dataclass
class Scenario:
    """
    The `Scenario` object holds ALL of the objects required to run
    a simulation. This includes the pricepaths, the external markets,
    all crvusdsim modules (LLAMMAs, Controllers, etc.), and all agents.
    """

    def __init__(self, fn: str):
        "Generate the scenario from the stress test scenario config file."

        with open(fn, "r") as f:
            logging.info(f"Reading price config from {fn}.")
            self.config = config = json.load(f)

        self.name: str = config["name"]
        self.description: str = config["description"]
        self.N: int = config["N"]
        self.price_config: str = config["price_config"]
        self.coins: List[TokenDTO] = [TOKEN_DTOs[a] for a in config["coins"]]
        self.pairs: List[Tuple[TokenDTO]] = [
            tuple(sorted(pair)) for pair in combinations(self.coins, 2)
        ]
        self.generate()

    def generate(self):
        self.generate_pricepaths()
        self.generate_markets()
        self.generate_sim_market()
        self.generate_agents()
        self.generate_metrics_processor()

    def generate_markets(self):
        """
        Generate the external markets for the scenario.

        TODO put the parameters for the quotes
        queries and market k_scale, etc. in config.
        """
        with DataHandler() as datahandler:
            logging.info("Fetching 1inch quotes.")
            quotes = datahandler.get_quotes(process=True)
            logging.info(f"We have {quotes.shape[0]} quotes.")

        logging.info("Fitting external markets against 1inch quotes.")
        self.markets: Dict[Tuple[TokenDTO], ExternalMarket] = dict()
        for pair in self.pairs:
            market = ExternalMarket(
                pair,
                1.25,
            )
            market.fit(quotes)
            self.markets[pair] = market

    def generate_pricepaths(self, fn: str = None):
        """
        Generate the pricepaths for the scenario.
        """
        fn = fn if fn else self.price_config  # override
        self.pricepaths: PricePaths = PricePaths(fn, self.N)

    def generate_agents(self):
        """
        Generate the agents for the scenario.
        """
        self.arbitrageur: Arbitrageur = Arbitrageur(DEFAULT_PROFIT_TOLERANCE)
        self.liquidator: Liquidator = Liquidator(DEFAULT_PROFIT_TOLERANCE)
        # TODO set liquidator paths

    def generate_sim_market(self):
        # TODO handle generation of multiple markets
        # TODO unpack sim_market objects
        # TODO assert that shared modules `is` the same object
        # assert pool == controller.AMM, "`controller.AMM` is not `pool`"
        # assert pool.BORROWED_TOKEN == controller.STABLECOIN
        # assert pool.COLLATERAL_TOKEN == controller.COLLATERAL_TOKEN
        sim_market = get("weth", bands_data="controller")
        self.llamma = sim_market.pool
        self.controller = sim_market.controller

    def generate_metrics_processor(self):
        # TODO implement metrics processor
        self.metrics: MetricsProcessor = MetricsProcessor()

    def update_market_prices(self, sample: PriceSample):
        """
        Update the market prices for the scenario.
        """
        for pair in self.pairs:
            self.markets[pair].update_price(sample.prices)

    def prepare_for_run(self, prices: PriceSample):
        pass

    def prepare_for_trades(self, ts: int):
        # TODO prepare aggregator for trades? <- update timestamps
        # TODO prepare pegkeeper for trades? <- update timestamps
        # TODO prepare stableswap pools for trades? <- update timestamps
        # TODO prepare tricryptoo pools for trades? <- update timestamps
        pass

    def after_trades(self):
        # TODO any post processing required?
        # NOTE controller.after_trades() forces liquidations,
        # we do NOT want to do this.
        pass

    def perform_actions(self, prices: PriceSample):
        # TODO arbitrageur
        # TODO liquidator
        # TODO keeper
        # TODO borrower
        # TODO liquidity_provider
        pass

    def update_metrics(self):
        pass

    def process_metrics(self):
        pass
