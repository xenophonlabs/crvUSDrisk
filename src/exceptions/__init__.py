class crvUSDRiskException(Exception):
    """Base class for exceptions in this module."""


class coingeckoRateLimitException(crvUSDRiskException):
    """Coingecko API Rate Limit Exceeded."""


class ccxtInvalidSymbolException(crvUSDRiskException):
    """Exchange does not support input symbol."""
