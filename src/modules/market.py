import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import permutations
from typing import Any, List, Dict
from sklearn.neighbors import KNeighborsRegressor
from ..data_transfer_objects import TokenDTO


class ExternalMarket:
    """
    A representation of external liquidity venues
    for relevant tokens. These markets are directional
    to account for asymmetric price impact.

    Note
    ----
    We always assume that the External Market is *at*
    the market price.
    """

    def __init__(
        self,
        coins: List[TokenDTO],
        k_scale=1.25,
    ):
        # TODO markets should be compatible in decimals
        # with the actual pools. This means we should
        # correct our training data to use the actual decimals.
        n = len(coins)
        assert n == 2

        self.coins = coins
        self.pair_indices = list(permutations(range(n), 2))
        self.n = n
        self.k_scale = k_scale
        self.prices = None
        self.ks = defaultdict(dict)
        self.models = defaultdict(dict)
        # self.models[i][j] is the model for swapping i->j

    @property
    def name(self):
        return f"External Market ({self.coins[0].symbol}, {self.coins[1].symbol})"

    @property
    def coin_names(self):
        return [c.name for c in self.coins]

    @property
    def coin_symbols(self):
        return [c.symbol for c in self.coins]

    @property
    def coin_addresses(self):
        return [c.address for c in self.coins]

    @property
    def coin_decimals(self):
        return [c.decimals for c in self.coins]

    def price(self, i: int, j: int) -> float:
        return self.prices[i][j]

    def update_price(self, prices: Dict[str, Dict[str, float]]):
        """
        Update the markets prices.

        Parameters
        ----------
        prices : PriceSample or or Dict[str, Dict[str, float]]
            prices[token1][token2] is the price of token1 in terms of token2.
            Notice that token1, token2 are addresses.
        """
        self.prices = {
            self.coin_addresses.index(token_in): {
                self.coin_addresses.index(token_out): price
                for token_out, price in token_out_prices.items()
                if token_out in self.coin_addresses
            }
            for token_in, token_out_prices in prices.items()
            if token_in in self.coin_addresses
        }

    def fit(self, quotes: pd.DataFrame):
        """
        Fit a KNN regression to the price impact data for each
        pair of tokens.

        Parameters
        ----------
        quotes : pd.DataFrame
            DataFrame of 1inch quotes.
        """
        for i, j in self.pair_indices:
            quotes_ = quotes.loc[(self.coin_addresses[i], self.coin_addresses[j])]

            self.ks[i][j] = k = int(len(quotes_["hour"].unique()) * self.k_scale)

            X = quotes_["in_amount"].values.reshape(-1, 1)
            y = quotes_["price_impact"].values

            model = KNeighborsRegressor(n_neighbors=k, weights="distance")
            model.fit(X, y)
            self.models[i][j] = model

    def trade(self, i: int, j: int, size: Any) -> Any:
        """
        Execute a trade on the external market using
        the current price.

        Parameters
        ----------
        i : int
            The index of the token_in.
        j : int
            The index of the token_out.
        size : Any
            The amount of token_in to sell for each trade.
            Can be an int/float (1 trade), or an array of ints/floats.

        Returns
        -------
        int
            The amount of token j the user gets out for each trade.
            Can be an int/float (1 trade), or an array of ints/floats.

        Note
        ----
        The market's fee should already be incorporated into the
        price impact estimation.
        """
        assert self.prices, "Prices not set for External Market."
        out = size * self.prices[i][j] * (1 - self.price_impact(i, j, size))
        # Correct for decimals
        out = out / 10 ** self.coin_decimals[i] * 10 ** self.coin_decimals[j]
        return int(out)

    def price_impact(self, i: int, j: int, size: Any) -> Any:
        """
        We model price impact using a KNN regression.

        Parameters
        ----------
        i : int
            The index of the token_in.
        j : int
            The index of the token_out.
        size : Any
            The amount of token_in to sell. Could be

        Returns
        -------
        float
            The price_impact for given trade.

        TODO this should account for trades that happened "recently".
        Maybe there's a "half-life" at which liquidity replenishes. Mean
        reverting process: learn the speed of mean-reversion (which is
        basically a half-life measure). Implement this on the "price" attr.
        """
        if isinstance(size, (float, int)):
            size = np.array(size).reshape(-1, 1)
        elif isinstance(size, list):
            size = np.array(size).reshape(-1, 1)
        elif size.ndim == 1:
            size = size.reshape(-1, 1)

        model = self.models[i][j]
        impact = model.predict(size)

        if np.any(impact < 0):
            logging.error(
                f"Price impact for {self.coin_symbols[i]} -> {self.coin_symbols[i]} is negative: {impact}!"
            )
        elif np.any(impact > 1):
            logging.error(
                f"Price impact for {self.coin_symbols[i]} -> {self.coin_symbols[i]} is over 100%: {impact}!"
            )
        return np.clip(impact, 0, 1)

    def __repr__(self):
        return self.name
