import json
import logging
from dataclasses import dataclass
from prices import PricePaths
from itertools import permutations
from collections import defaultdict
from ..modules.market import ExternalMarket
from ..db.datahandler import DataHandler


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
        self.pairs = list(permutations(self.coins))

    def generate_markets(self):
        """
        Generate the markets for the scenario.

        TODO put the parameters for the quotes
        queries and market k_scale, etc. in config.
        """
        with DataHandler as datahandler:
            logging.info("Fetching 1inch quotes.")
            quotes = datahandler.get_quotes(process=True)
            logging.info(f"We have {quotes.shape[0]} quotes.")

        logging.info("Fitting external markets against 1inch quotes.")
        markets = defaultdict(dict)
        for pair in self.pairs:
            in_token, out_token = pair
            quotes_ = quotes.loc[pair]
            market = ExternalMarket(in_token, out_token, 1.25)
            market.fit(quotes_)
            markets[in_token][out_token] = market

        return markets

    def generate_pricepaths(self):
        """
        Generate the pricepaths for the scenario.
        """
        return PricePaths(self.price_config, self.N)
