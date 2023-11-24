import json
import logging
from dataclasses import dataclass
from itertools import permutations
from collections import defaultdict
from .prices import PricePaths
from ..modules.market import ExternalMarket
from ..db.datahandler import DataHandler
from ..utils import get_decimals_from_config
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
        self.coins = config["coins"]
        self.pairs = list(permutations(self.coins, 2))

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
        markets = defaultdict(dict)
        for pair in self.pairs:
            in_token, out_token = pair
            quotes_ = quotes.loc[pair]
            market = ExternalMarket(
                in_token,
                out_token,
                get_decimals_from_config(in_token),
                get_decimals_from_config(out_token),
                1.25,
            )
            market.fit(quotes_)
            markets[in_token][out_token] = market

        return markets

    def generate_pricepaths(self, fn=None):
        """
        Generate the pricepaths for the scenario.
        """
        fn = fn if fn else self.price_config  # override
        return PricePaths(fn, self.N)

    def generate_agents(self):
        """
        Generate the agents for the scenario.
        """
        arbitrageur = Arbitrageur(0)
        liquidator = Liquidator(0)
        # TODO set liquidator paths

    @staticmethod
    def update_market_prices(markets, sample):
        """
        Update the market prices for the scenario.
        """
        for in_token in markets:
            for out_token in markets[in_token]:
                markets[in_token][out_token].update_price(sample[in_token][out_token])
