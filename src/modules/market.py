"""
Provides the `ExternalMarket` class for modeling
swaps in external liquidity venues.
"""
from __future__ import annotations
from collections import defaultdict
from itertools import permutations
from typing import Any, Tuple, Dict, TYPE_CHECKING
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from ..data_transfer_objects import TokenDTO
from ..logging import get_logger

if TYPE_CHECKING:
    from ..types import PairwisePricesType


logger = get_logger(__name__)


class ExternalMarket:
    """
    A representation of external liquidity venues
    for relevant tokens. These markets are statistical
    models trained on 1inch quotes. We are currently
    using an IsotonicRegression to model fees and price impact.

    Note
    ----
    We always assume that the External Market is *at*
    the market price.
    """

    def __init__(
        self,
        coins: Tuple[TokenDTO, TokenDTO],
    ):
        n = len(coins)
        assert n == 2

        self.coins = coins
        self.pair_indices = list(permutations(range(n), 2))
        self.n = n
        self.prices: Dict[int, Dict[int, float]] | None = None
        self.models: Dict[int, Dict[int, IsotonicRegression]] = defaultdict(dict)

    @property
    def name(self):
        """Market name."""
        return f"External Market ({self.coins[0].symbol}, {self.coins[1].symbol})"

    @property
    def coin_names(self):
        """List of coin names in the market."""
        return [c.name for c in self.coins]

    @property
    def coin_symbols(self):
        """List of coin symbols in the market."""
        return [c.symbol for c in self.coins]

    @property
    def coin_addresses(self):
        """List of coin addresses in the market."""
        return [c.address for c in self.coins]

    @property
    def coin_decimals(self):
        """List of coin decimals in the market."""
        return [c.decimals for c in self.coins]

    def price(self, i: int, j: int) -> float:
        """Get the price of token i in terms of token j."""
        if not self.prices:
            raise ValueError("Prices not set for External Market.")
        return self.prices[i][j]

    def update_price(self, prices: "PairwisePricesType"):
        """
        Update the markets prices.

        Parameters
        ----------
        prices : PairwisePricesType
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
        Fit an IsotonicRegression to the price impact data for each
        pair of tokens.

        Parameters
        ----------
        quotes : pd.DataFrame
            DataFrame of 1inch quotes.
        """
        for i, j in self.pair_indices:
            quotes_ = quotes.loc[(self.coin_addresses[i], self.coin_addresses[j])]

            x = quotes_["in_amount"].values.reshape(-1, 1)
            y = quotes_["price_impact"].values

            model = IsotonicRegression(
                y_min=0, y_max=1, increasing=True, out_of_bounds="clip"
            )
            model.fit(x, y)
            self.models[i][j] = model

    def trade(self, i: int, j: int, size: int) -> Any:
        """
        Execute a trade on the external market using
        the current price.

        Parameters
        ----------
        i : int
            The index of the token_in.
        j : int
            The index of the token_out.
        size : int
            The amount of token_in to sell for each trade.

        Returns
        -------
        int
            The amount of token j the user gets out.

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
            The amount of token_in to sell. Could be an int
            or a list of ints.

        Returns
        -------
        int or List[int]
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
            logger.error(
                "Price impact for %s -> %s is negative: %f!",
                self.coin_symbols[i],
                self.coin_symbols[i],
                impact,
            )
        elif np.any(impact > 1):
            logger.error(
                "Price impact for %s -> %s is over 100%%: %f!",
                self.coin_symbols[i],
                self.coin_symbols[i],
                impact,
            )
        return np.clip(impact, 0, 1)

    def __repr__(self):
        return self.name
