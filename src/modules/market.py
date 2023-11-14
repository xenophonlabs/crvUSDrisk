class ExternalMarket:
    """
    A representation of external liquidity venues
    for relevant tokens. Examples include:
        - WETH/USDC market on Uniswap v3.
        - USDC/USDT market on Binance.
    This object allows us to estimate the price impact
    of trades on external markets.

    Note
    ----
    We always assume that the External Market is *at*
    the market price.

    TODO would this be way faster if it were just implemented
    using Numpy operations?
    """

    def __init__(self, n, coefs, intercepts):
        """
        Initialize market with OLS params:

        price_impact = m * trade_size + b

        Parameters
        ----------
        n : int
            Number of tokens in market.
        coefs : List[List[float]]
            coefs[i][j] is the slope of the OLS regression
            for selling token i for token j.
        b : List[List[float]]
            intercepts[i][j] is the intercept of the OLS regression
            for selling token i for token j.
        """
        self.n = n
        self.coefs = coefs
        self.intercepts = intercepts

    def trade(self, amt_in, price, i, j):
        """
        Execute a trade on the external market using
        the current price.

        Parameters
        ----------
        price : float
            The external market price for exchanging
            token i for token j.
        i : int
            The index of the in token.
        j : int
            The index of the out token.

        Returns
        -------
        int
            The amount of token j the user gets out

        Note
        ----
        The market's fee should already be incorporated into the
        price impact estimation.
        """
        return amt_in * price * (1 + self.price_impact(amt_in, i, j))

    def price_impact(self, amt_in, i, j):
        """
        We model price impact as a linear regression
        with trade size. The coefficients of the linear
        regression are passed into the constructor.

        Parameters
        ----------
        price : float
            The external market price for exchanging
            token i for token j.
        i : int
            The index of the in token.
        j : int
            The index of the out token.

        Returns
        -------
        float
            The price_impact (decimals) for given trade.
        """
        # return -0.003
        return amt_in * self.coefs[i][j] + self.intercepts[i][j]
