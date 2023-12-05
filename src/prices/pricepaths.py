"""
Provides the `PricePaths` for generating and iterating
through `PriceSample`s. A `PriceSample` stores USD token prices
at a given timestep and converts them to pairwise prices.
"""
from __future__ import annotations
from typing import Dict, TYPE_CHECKING
from dataclasses import dataclass
from itertools import permutations
from collections import defaultdict
import pandas as pd
from .utils import gen_cor_prices, get_gran, get_factor
from ..configs import get_config
from ..network.coingecko import address_from_coin_id, get_current_prices

if TYPE_CHECKING:
    from ..types import PairwisePricesType


def get_pairwise_prices(prices_usd: Dict[str, float]) -> "PairwisePricesType":
    """Convert USD prices into pairwise prices."""
    pairs = list(permutations(prices_usd.keys(), 2))
    prices: "PairwisePricesType" = defaultdict(dict)
    for pair in pairs:
        in_token, out_token = pair
        prices[in_token][out_token] = prices_usd[in_token] / prices_usd[out_token]
    return prices


@dataclass
class PriceSample:
    """
    Stores USD and pairwise prices at a given timestep.
    """

    def __init__(self, ts: int, prices_usd: Dict[str, float]):
        self.timestamp = ts
        self.prices_usd = prices_usd  # USD
        self.prices = get_pairwise_prices(prices_usd)

    def update(self, prices_usd: Dict[str, float]):
        """
        Update USD prices for any subset of tokens.
        Recalculate pairwise prices.
        """
        self.prices_usd.update(prices_usd)
        self.prices = get_pairwise_prices(self.prices_usd)

    def __getitem__(self, address: str) -> Dict[str, float]:
        """
        Return prices for selling specified token.
        Example: self[<address1>] = {<address2>: 0.9, <address3>: 1.1}
        """
        return self.prices[address]

    def __repr__(self):
        return f"PriceSample({self.timestamp}, {self.prices_usd})"


class PricePaths:
    """
    `PricePaths` generates prices from the price config file
    and implements an iterator over `PriceSample`s.
    """

    def __init__(self, fn: str, num_steps: int):
        """
        Generate price paths from config file.
        TODO integrate with curvesim PriceSampler?
        """
        self.config = config = get_config(fn, "prices")

        freq = config["freq"]
        coin_ids = list(config["params"].keys())
        annual_factor = get_factor(freq)
        dt = 1 / annual_factor

        self.prices = gen_cor_prices(
            coin_ids,  # List of coin IDs (coingecko)
            num_steps * dt,  # Time horizon in years
            dt,  # Time step in years
            get_current_prices(coin_ids),
            pd.DataFrame.from_dict(config["cov"]),
            config["params"],
            timestamps=True,
            gran=get_gran(freq),
        )

        self.coins = [address_from_coin_id(coin_id) for coin_id in coin_ids]

    def __iter__(self):
        """
        Yields
        ------
        :class: `PriceSample`
        """
        for ts, sample in self.prices.iterrows():
            yield PriceSample(ts, sample.to_dict())

    def __getitem__(self, i: int) -> PriceSample:
        ts = self.prices.index[i]
        sample = self.prices.iloc[i]
        return PriceSample(ts, sample.to_dict())
