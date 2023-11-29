import json
import logging
import pandas as pd
from typing import Dict
from dataclasses import dataclass
from itertools import permutations
from collections import defaultdict
from .utils import gen_cor_prices, gran, factor
from ..network.coingecko import address_from_coin_id, get_current_prices

def get_pairwise_prices(_prices):
    pairs = list(permutations(_prices.keys(), 2))
    prices = defaultdict(dict)
    for pair in pairs:
        in_token, out_token = pair
        prices[in_token][out_token] = _prices[in_token] / _prices[out_token]
    return prices

@dataclass
class PriceSample:
    def __init__(self, ts: int, _prices: Dict[str, float]):
        self.ts = ts
        self._prices = _prices
        self.prices = get_pairwise_prices(_prices)

    def update(self, _prices: Dict[str, float]):
        """
        Update USD prices for any subset of tokens. 
        Recalculate pairwise prices.
        """
        self._prices.update(_prices)
        self.prices = get_pairwise_prices(self._prices)

    def __getitem__(self, address: str) -> Dict[str, float]:
        """
        Return prices for selling specified token.
        Example: self[<address1>] = {<address2>: 0.9, <address3>: 1.1}
        """
        return self.prices[address]

    def __repr__(self):
        return f"PriceSample({self.ts}, {self._prices})"


class PricePaths:
    """
    Convenient class for storing params required
    to generate prices.
    """

    def __init__(self, fn: str, N: int):
        """
        Generate price paths from config file.

        Parameters
        ----------
        fn : str
            Path to price config file.
        N : int
            Number of timesteps.

        TODO integrate with curvesim PriceSampler?
        """

        with open(fn, "r") as f:
            logging.info(f"Reading price config from {fn}.")
            config = json.load(f)

        self.N = N
        self.params = config["params"]
        self.cov = pd.DataFrame.from_dict(config["cov"])
        self.freq = config["freq"]
        self.gran = gran(self.freq)  # in seconds
        self.coin_ids = list(self.params.keys())
        self.coins = [address_from_coin_id(coin_id) for coin_id in self.coin_ids]
        self.S0s = get_current_prices(self.coin_ids)
        self.annual_factor = factor(self.freq)
        self.dt = 1 / self.annual_factor
        self.T = self.N * self.dt
        self.S = gen_cor_prices(
            self.coin_ids,
            self.T,
            self.dt,
            self.S0s,
            self.cov,
            self.params,
            timestamps=True,
            gran=self.gran,
        )

    def __iter__(self):
        """
        Yields
        ------
        :class: `PriceSample`
        """
        for ts, prices in self.S.iterrows():
            yield PriceSample(ts, prices.to_dict())

    def __getitem__(self, i: int) -> PriceSample:
        ts = self.S.index[i]
        prices = self.S.iloc[i]
        return PriceSample(ts, prices.to_dict())
