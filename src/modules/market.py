import logging


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

    def __init__(self, token_in: str, token_out: str, m: float, b: float):
        """
        Initialize market with OLS params:

        price_impact = m * trade_size + b

        Parameters
        ----------
        token_in = str
            The token to sell (address).
        token_out = str
            The token to buy (address).
        m : float
            The slope of the OLS regression.
        b : float
            The intercept of the OLS regression.

        Note
        ----
        Eventually include a multi-variate
        regression on volatility as well.
        """
        self.token_in = token_in
        self.token_out = token_out
        self.m = m
        self.b = b

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

    def price_impact(self, amt_in):
        """
        We model price impact as a linear regression
        with trade size. The coefficients of the linear
        regression are passed into the constructor.

        Parameters
        ----------
        amt_in : float
            The amount of token_in to sell.

        Returns
        -------
        float
            The price_impact (decimals) for given trade.

        Note
        ----
        FIXME should never be negative.
        FIXME this is completely broken for WBTC
        FIXME the curve doesn't fit certain illiquid pairs very well.
        """
        impact = amt_in * self.m + self.b
        if impact < 0:
            logging.warning(
                f"Price impact for {self.token_in} -> {self.token_out} is negative: {impact}!"
            )
        elif impact > 1:
            logging.warning(
                f"Price impact for {self.token_in} -> {self.token_out} is over 100%: {impact}!"
            )
        return min(max(impact, 0), 1)
