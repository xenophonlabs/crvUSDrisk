import json
import logging
from typing import Dict
from dataclasses import dataclass
from itertools import combinations
from ..prices import PricePaths, PriceSample
from ..configs import TOKEN_DTOs
from ..modules import ExternalMarket
from ..db.datahandler import DataHandler
from ..agents.arbitrageur import Arbitrageur
from ..agents.liquidator import Liquidator


@dataclass
class Scenario:
    def __init__(self, fn: str):
        """
        Generate the scenario from the stress
        test scenario config file.
        """

        with open(fn, "r") as f:
            logging.info(f"Reading price config from {fn}.")
            self.config = config = json.load(f)

        self.name = config["name"]
        self.description = config["description"]
        self.N = config["N"]
        self.price_config = config["price_config"]
        self.coins = [TOKEN_DTOs[a] for a in config["coins"]]
        self.pairs = [tuple(sorted(pair)) for pair in combinations(self.coins, 2)]

    def generate_markets(self) -> Dict[tuple, ExternalMarket]:
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
        markets = dict()
        for pair in self.pairs:
            market = ExternalMarket(
                pair,
                1.25,
            )
            market.fit(quotes)
            markets[pair] = market
        self.markets: Dict[tuple, ExternalMarket] = markets
        return markets

    def generate_pricepaths(self, fn: str = None) -> PricePaths:
        """
        Generate the pricepaths for the scenario.
        """
        fn = fn if fn else self.price_config  # override
        self.pricepaths: PricePaths = PricePaths(fn, self.N)
        return self.pricepaths

    def generate_agents(self):
        """
        Generate the agents for the scenario.
        """
        self.arbitrageur: Arbitrageur = Arbitrageur(0)
        self.liquidator: Liquidator = Liquidator(0)
        # TODO set liquidator paths

    def update_market_prices(self, sample: PriceSample):
        """
        Update the market prices for the scenario.
        """
        for pair in self.pairs:
            self.markets[pair].update_price(sample.prices)
