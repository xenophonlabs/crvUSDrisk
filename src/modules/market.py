import logging
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from typing import Union, List


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
        token_in: str,
        token_out: str,
        decimals_in: int=18,
        decimals_out: int=18,
        k_scale=1.25,
    ):
        # TODO token_in/out should be Token objs
        # so we don't have to keep passing decimals/names/addresses
        # as args into funcs.
        self.token_in = token_in
        self.token_out = token_out
        self.k_scale = k_scale
        self.coin_addresses = [token_in, token_out]
        self.coin_decimals = [decimals_in, decimals_out]
        self.price = None

    def update_price(self, price: float):
        self.price = price

    def fit(self, df):
        """
        Fit a KNN regression to the price impact data.
        """
        self.k = k = int(len(df["hour"].unique()) * self.k_scale)

        X = df["in_amount"].values.reshape(-1, 1)
        y = df["price_impact"].values

        model = KNeighborsRegressor(n_neighbors=k, weights="distance")
        model.fit(X, y)

        self.model = model

    def trade(self, i: int, j: int, amt_in: float) -> float:
        """
        Execute a trade on the external market using
        the current price.

        Parameters
        ----------
        amt_in : float
            The amount of token_in to sell.

        Returns
        -------
        int
            The amount of token j the user gets out

        Note
        ----
        The market's fee should already be incorporated into the
        price impact estimation.
        """
        assert self.price, "Price not set for External Market."
        return amt_in * self.price * (1 - self.price_impact(amt_in))

    def price_impact(self, amt_in: Union[List[List[float]], List[float], float]):
        """
        We model price impact using a KNN regression.

        Parameters
        ----------
        amt_in : float
            The amount of token_in to sell.

        Returns
        -------
        float
            The price_impact (decimals) for given trade.

        TODO this should account for trades that happened "recently".
        Maybe there's a "half-life" at which liquidity replenishes. Mean
        reverting process: learn the speed of mean-reversion (which is
        basically a half-life measure). Implement this on the "price" attr.
        """
        if isinstance(amt_in, float):
            amt_in = np.array(amt_in).reshape(-1, 1)
        elif isinstance(amt_in, list):
            amt_in = np.array(amt_in).reshape(-1, 1)
        elif amt_in.ndim == 1:
            amt_in = amt_in.reshape(-1, 1)

        impact = self.model.predict(amt_in)
        if np.any(impact < 0):
            logging.error(
                f"Price impact for {self.token_in} -> {self.token_out} is negative: {impact}!"
            )
        elif np.any(impact > 1):
            logging.error(
                f"Price impact for {self.token_in} -> {self.token_out} is over 100%: {impact}!"
            )
        return np.clip(impact, 0, 1)
