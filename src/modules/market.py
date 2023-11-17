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

    def __init__(self, token_in: str, token_out: str, k_scale=1.25):
        self.token_in = token_in
        self.token_out = token_out
        self.k_scale = k_scale

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

    def trade(self, amt_in, price):
        """
        Execute a trade on the external market using
        the current price.

        Parameters
        ----------
        amt_in : float
            The amount of token_in to sell.
        price : float
            The external market price for exchanging
            token i for token j.

        Returns
        -------
        int
            The amount of token j the user gets out

        Note
        ----
        The market's fee should already be incorporated into the
        price impact estimation.
        """
        return amt_in * price * (1 - self.price_impact(amt_in))

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

    # ARCHIVED. TODO remove
    # def __init__(self, token_in: str, token_out: str, m: float, b: float):
    #     """
    #     Initialize market with OLS params:

    #     price_impact = m * trade_size + b

    #     Parameters
    #     ----------
    #     token_in = str
    #         The token to sell (address).
    #     token_out = str
    #         The token to buy (address).
    #     m : float
    #         The slope of the OLS regression.
    #     b : float
    #         The intercept of the OLS regression.

    #     Note
    #     ----
    #     Eventually include a multi-variate
    #     regression on volatility as well.
    #     """
    #     self.token_in = token_in
    #     self.token_out = token_out
    #     self.m = m
    #     self.b = b

    # def price_impact(self, amt_in):
    #     """
    #     We model price impact as a linear regression
    #     with trade size. The coefficients of the linear
    #     regression are passed into the constructor.

    #     Parameters
    #     ----------
    #     amt_in : float
    #         The amount of token_in to sell.

    #     Returns
    #     -------
    #     float
    #         The price_impact (decimals) for given trade.
    #     """
    #     impact = amt_in * self.m + self.b
    #     if impact < 0:
    #         logging.warning(
    #             f"Price impact for {self.token_in} -> {self.token_out} is negative: {impact}!"
    #         )
    #     elif impact > 1:
    #         logging.warning(
    #             f"Price impact for {self.token_in} -> {self.token_out} is over 100%: {impact}!"
    #         )
    #     return min(max(impact, 0), 1)
